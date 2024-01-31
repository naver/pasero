# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
from typing import Optional, Union
import regex
import json
import itertools
import collections
import logging
import numpy as np
import torch
import copy
from typing import Optional, Sequence, Iterable, Iterator
from pasero.tokenizers import sep, bos, eos, unk, pad
from pasero.tokenizers import PaseroTokenizer, SentencePieceTokenizer, HuggingFaceTokenizer, CharacterTokenizer, load_vocab
from pasero.tokenizers.noise import mask, noisify
from pasero import utils
from pasero.config import PreprocessingConfig, NoiseConfig
from stopes.pipelines.monolingual.utils.text_normalizer import remove_non_printing_char, replace_unicode_punct


BPEModel = Union[HuggingFaceTokenizer, PaseroTokenizer, SentencePieceTokenizer, CharacterTokenizer]


logger = logging.getLogger('preprocessing')


copy_tag = '<PHL>'


def mask_padding(ids: Sequence[int], eos_idx: int, padding_idx: int) -> list[int]:
    """ Return a mask with ones at padding token positions """
    mask = []
    for token_id in ids:
        if token_id == padding_idx:
            mask.append(1)
        else:
            mask.append(0)
        if token_id == eos_idx:
            break
    return mask + (len(ids) - len(mask)) * [1]


class Dictionary:
    """
    Implements fairseq-style dictionaries, which map text tokens to integer ids (which can be used for indexing
    embedding matrices).

    Such a dictionary contains one token and its frequency per line. The rank in the file gives the id for this token.

    The ids are shifted by 4 positions and the special tokens are automatically inserted at the beginning (like in 
    fairseq), unless at least </s> is already in the dictionary.

    For example, here is the beginning of a text dictionary:

    ```
    <T> 0             # ID=4
    <U> 0             # ID=5
    <BT> 0            # ID=6
    ▁ 35270518203     # ID=7
    e 22795941272     # ID=8
    ...
    ```
    In this scenario, `<sep>`, `<pad>`, `</s>` and `<unk>` or automatically assigned ids 0, 1, 2 and 3. And `<s>` is
    assigned the same id as `</s>` (i.e., same token is used for EOS and BOS).

    Here is another possible scenario, which lets us change the default values for the special tokens (useful for 
    pre-trained models):

    ```
    <pad> 0           # ID=0
    <s> 0             # ID=1
    </s> 0            # ID=2
    <0x00> 0          # ID=3
    <0x01> 0          # ID=4
    ...
    ```
    `Dictionary` detects that `</s>` is already there, so it does not try to add special tokens and each token 
    in the text file is assigned its line number (starting at zero) as id.

    Note that if they are not specified, `<s>` and `<unk>` are given respectively the same value as `</s>` and `<pad>`.
    And if it is not there, `<pad>` is automatically given the id -1.

    A dictionary can also be created directly from a HuggingFace-style JSON vocabulary (which maps tokens to their id).
    This is done automatically by calling `build(...)` with a file path ending in '.json'
    """

    tokens: list[str]        # maps ids to strs
    indices: dict[str, int]  # maps strs to ids

    @classmethod
    def build(cls, path: str, size: Optional[int] = None):
        """
        Args:
            path: path to the dict file (text file with one token and its frequency per line, delimited by a
                whitespace) or to a json vocabulary
            size: pad the dictionary to this size
        """
        if path.endswith('.json'):
            with open(path) as dict_file:
                indices = json.load(dict_file)  # same format as HuggingFace's tokenizer.vocab
                return cls(indices, size=size)
        else:
            tokens = load_vocab(path)
            return cls(tokens, size=size)

    def __init__(self, vocab: Union[list, dict], size: Optional[int] = None):
        """
        Initialize a Dictionary either from a list of tokens (fairseq-style) or from a dictionary that directly maps
        each token to its id (HuggingFace-style).
        """
        assert not isinstance(vocab, str), 'to initialize a dictionary from a path, call Dictionary.build(...)'

        if isinstance(vocab, dict):
            self.indices = dict(vocab)
            vocab_size = max(self.indices.values()) + 1
            self.tokens = [unk] * vocab_size
            for w, i in self.indices.items():
                self.tokens[i] = w
        else:
            self.tokens = list(vocab)
            token_set = set(self.tokens)
            
            if eos not in token_set:  # the dict does not contain special tokens (fairseq-style)
                for token in sep, bos, pad, eos, unk:
                    assert token not in token_set, (
                        "dictionary has a partial set of special tokens: it should either have none of them "
                        "(fairseq-style), or all of them (custom-style). "
                        "A custom dictionary should contain at least '</s>', and a fairseq dictionary is not allowed to "
                        "contain '</s>', '<s>', '<unk>', '<pad>' or '<sep>'"
                    )
                # automatically add special tokens
                self.tokens = [sep, pad, eos, unk] + self.tokens  # <s> is not used in fairseq checkpoints, so we use
                # this position for something else

            self.indices = {w: i for i, w in enumerate(self.tokens)}

        assert all(token in self.indices for token in self.tokens)
        assert len(self.tokens) == max(self.indices.values()) + 1
        if size is not None:
            self.extend(size)

        assert all(token_id >= 0 for token_id in self.indices.values()), 'negative token ids are not allowed'
        # These special token ids may be ignored, because `Task.setup_for_model()` modifies their values to match those
        # of the model.
        self.eos_idx = self.indices.get(eos)
        self.padding_idx = self.indices.get(pad, self.indices.get(unk))
        self.bos_idx = self.indices.get(bos, self.eos_idx)      # if not defined, bos_idx = eos_idx
        self.unk_idx = self.indices.get(unk, self.padding_idx)  # if not defined, unk_idx = padding_idx
        self.sep_idx = self.indices.get(sep, self.bos_idx)

    def extend(self, size: int) -> None:
        """ Extend this dictionary to given size by padding with dummy tokens """
        i = 0
        while size > len(self.tokens):
            w = f'madeupword{i:04}'
            if w not in self.indices:
                self.indices[w] = len(self.tokens)
                self.tokens.append(w)
            i += 1

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.indices

    def __getitem__(self, idx: int) -> str:
        # self.tokens may not accurately reflect special tokens. For instance, it's important that 
        # eos_idx -> eos, even if eos and bos share the same id.
        if idx == self.eos_idx:
            return eos
        elif idx == self.padding_idx:
            return pad
        elif idx == self.bos_idx:
            return bos
        elif idx == self.sep_idx:
            return sep
        elif idx == self.unk_idx:
            return unk
        else:
            return self.tokens[idx]

    def __setitem__(self, idx: int, token: str):
        self.tokens[idx] = token
        self.indices[token] = idx

    def __eq__(self, other) -> bool:
        return isinstance(other, Dictionary) and other.tokens == self.tokens

    def idx(self, token: str) -> int:
        return self.indices.get(token, self.unk_idx)

    def to_indices(
        self,
        tokens: list[str],
        max_len: Optional[int] = None,
        append_eos: bool = True,
        prepend_bos: bool = False,
        truncate_left: bool = False,
    ) -> np.ndarray:
        ids = [self.idx(token) for token in tokens]
        
        if max_len is not None:
            max_len = max_len - int(append_eos) - int(prepend_bos)
            ids = ids[-max_len:] if truncate_left else ids[:max_len]
        if prepend_bos:
            ids.insert(0, self.bos_idx)
        if append_eos:
            ids.append(self.eos_idx)
        return np.array(ids, dtype=np.int64)

    def to_string(self, ids: Sequence[int]) -> list[str]:
        return [self[token_id] for token_id in ids if token_id != self.padding_idx]

    def remap_embed(self, old_embed: torch.Tensor, old_dict, default: Optional[str] = None) -> torch.Tensor:
        default_idx = old_dict.indices[default] if default else None
        embed = []
        unk_count = 0
        for index, token in enumerate(self.tokens):
            token = self.tokens[index]
            if token in old_dict.indices:
                old_index = old_dict.indices[token]
                v = old_embed[old_index]
            elif not default:
                v = torch.empty_like(old_embed[0])
                utils.embed_init(v)
                unk_count += 1
            else:
                old_index = default_idx
                v = old_embed[old_index]
                unk_count += 1
            embed.append(v)
        logger.info(f"re-mapped embeddings: {unk_count}/{len(embed)} tokens mapped to '{default}'")
        return torch.stack(embed, dim=0)


_LANG_CODE_PREFIX = 'lang:'
_DOMAIN_TAG_PREFIX = 'domain:'

_LANG_CODE_REGEX = regex.compile(f'<{regex.escape(_LANG_CODE_PREFIX)}(.+?)>')
_DOMAIN_TAG_REGEX = regex.compile(f'<{regex.escape(_DOMAIN_TAG_PREFIX)}(.+?)>')


def is_lang_code(token: str) -> bool:
    return bool(_LANG_CODE_REGEX.fullmatch(token))

def is_domain_tag(token: str) -> bool:
    return bool(_DOMAIN_TAG_REGEX.fullmatch(token))

def is_tag(token: str) -> bool:
    return is_lang_code(token) or is_domain_tag(token)

def split_tags(line: str) -> list[str]:
    tokens = line.split()
    tags = list(itertools.takewhile(is_tag, tokens))
    for tag in tags:
        reg = rf'\s*{regex.escape(tag)}\s*'
        # only strip this tag and the whitespaces around it
        # keep the other whitespaces, as this may be used on multi-line strings
        line = regex.sub(reg, '', line, count=1)
    return [*tags, line]

def get_lang_code(lang: str) -> str:
    return f'<{_LANG_CODE_PREFIX}{lang}>' if lang else None

def get_domain_tag(domain: str) -> str:
    return f'<{_DOMAIN_TAG_PREFIX}{domain}>' if domain else None


class TextPreprocessor:
    """
    Handles every step of the pre-processing:
    - optional punctuation normalization
    - BPE/subword tokenization (using tokenizer classes defined in "pasero/tokenizers")
    - tagging (language codes, domain tags, etc.)
    - binarization (i.e., mapping of tokens into dictionary ids), using `Dictionary`
    - debinarization
    - detokenization
    """
    def __init__(
        self,
        cfg: PreprocessingConfig,
        dir: str,
        **kwargs,
    ):
        """
        Args:
            - cfg: configuration of this preprocessor
            - dir: directory where to look for tokenizer files (typically equals to `--data-dir` at training and 
                `--model-dir` at inference)
            - kwargs: override the parameters in `cfg` by these keyword arguments
        """
        self.training = False
        self.dir = dir

        self.cfg = copy.copy(cfg)

        if cfg.keep_whitespaces:
            # This is not compatible with any type of normalization
            assert not cfg.normalize_punctuation
        
        if not self.cfg.tokenizer_path:
            tokenizer_path = self.default_tokenizer_path(self.cfg.tokenizer)
            tokenizer_path = tokenizer_path or dir   # for HF tokenizers
            self.cfg.tokenizer_path = tokenizer_path

        if self.cfg.bpe_dropout or self.cfg.spell_out:
            assert self.cfg.tokenizer == 'pasero', '--spell-out and --bpe-dropout are only compatible with ' \
                '--tokenizer pasero'

        # Parameters in `cfg` can be overriden with keyword arguments
        for k, v in kwargs.items():
            setattr(self.cfg, k, v)

        self.dict_path = self.tokenizer_path = self.vocab_path = None

        self.load_tokenizer()
        assert not cfg.masking or mask in self.dictionary, f'{mask} is OOV'

        # special tokens that will be protected from tokenization. Note that lang and domain tags are not included: 
        # they are handled by split_tags, and only when they are at the beginning of the sequence
        protected_tokens = [sep, bos, eos, unk] + cfg.protect_tokens
        protected_tokens_regex = '|'.join(regex.escape(token) for token in protected_tokens)
        self.protected_tokens_regex = regex.compile(protected_tokens_regex)
        # split both according to special tokens and stop sequences (we don't want the tokenization of stop sequences 
        # to be conflated with the surrounding characters)
        split_tokens = protected_tokens + cfg.stop_sequences
        split_tokens_regex = '|'.join(regex.escape(token) for token in split_tokens)
        split_tokens_regex = f'({split_tokens_regex})'  # add capturing group for regex.split()
        self.split_tokens_regex = regex.compile(split_tokens_regex)
        # note that lang and domain tags
        self.set_stop_sequences(cfg.stop_sequences)

    @property
    def bos_idx(self) -> int:
        return self.dictionary.bos_idx
    
    @bos_idx.setter
    def bos_idx(self, value: int):
        self.dictionary.bos_idx = value
        self.dictionary[value] = bos
    
    @property
    def eos_idx(self) -> int:
        return self.dictionary.eos_idx
    
    @eos_idx.setter
    def eos_idx(self, value: int):
        self.dictionary.eos_idx = value
        self.dictionary[value] = eos
    
    @property
    def padding_idx(self) -> int:
        return self.dictionary.padding_idx
    
    @padding_idx.setter
    def padding_idx(self, value: int):
        self.dictionary.padding_idx = value
        self.dictionary[value] = pad
    
    @property
    def unk_idx(self) -> int:
        return self.dictionary.unk_idx
    
    @unk_idx.setter
    def unk_idx(self, value: int):
        self.dictionary.unk_idx = value
        self.dictionary[value] = unk

    @classmethod
    def default_tokenizer_path(cls, tokenizer: str) -> str:
        if tokenizer == 'sentencepiece':
            return 'spm.model'
        elif tokenizer == 'pasero':
            return 'bpecodes'
        else:
            return None

    def train(self):
        """
        Set this preprocessor into training mode, which will enable some stochastic features like BPE dropout 
        and noise generation. This should be called before preprocessing training examples
        """
        self.training = True
    
    def eval(self):
        """
        Set this preprocessor into evaluation mode, which will disable some stochastic features like BPE dropout 
        and noise generation. This should be called before preprocessing validation or inference examples
        """
        self.training = False

    def infer_langs(self) -> set[str]:
        """ Infer a set of covered languages from the language codes in the dictionary """
        langs = []
        for token in self.dictionary:
            if (match := regex.fullmatch(_LANG_CODE_REGEX, token)):
                lang = match.group(1)
                langs.append(lang)
        return set(langs)

    @property
    def num_symbols(self) -> int:
        return len(self.dictionary)

    @property
    def files(self):
        """
        Set of files that should be copied to the model directory
        """
        return {self.tokenizer_path, self.dict_path} - {None}

    def load_tokenizer(self) -> Optional[BPEModel]:
        requires_dict = self.cfg.tokenizer in ('none', 'char', 'pasero')  # some tokenizers may have a built-in
        # dictionary (SentencePiece and HuggingFace), while others need a separate dict file

        # Attempt to load a dictionary, from given "--dict" option or by looking for "dict.json" or "dict.txt" files
        # If such a dictionary is found, it will override the tokenizer's built-in dictionary (if any).
        # In the case of SentencePiece and Pasero, this dictionary will also be used for BPE filtering (i.e., the models
        # won't be allowed to generate subwords that are out of vocabulary).
        # Note that if a "dict.json" or "dict.txt" file exists, it will be loaded by default (even if "--dict" is not 
        # set). To avoid loading it (e.g., if one such file exists but it does not match our tokenizer),
        # "--dict" can be set to a dummy value (e.g., "--dict none")
        if self.cfg.dict:
            self.dict_path = utils.find_file(self.cfg.dict, dirs=[self.dir, '.'], fail=requires_dict)  # --dict can be 
            # relative to data/model dir or to the working directory
        else:
            self.dict_path = utils.find_file('dict.json', 'dict.txt', dirs=[self.dir], fail=requires_dict)

        if self.dict_path is None:
            self.dictionary = None
        else:
            self.dictionary = Dictionary.build(self.dict_path)
            logger.info(f'{self.dictionary.__class__.__name__}: {len(self.dictionary)} symbols')

        if self.cfg.tokenizer == 'none':
            self.tokenizer_path = None
            self._tokenizer = None
        elif self.cfg.tokenizer == 'char':
            self.tokenizer_path = None
            self._tokenizer = CharacterTokenizer()
        elif self.cfg.tokenizer == 'hf':
            self.tokenizer_path = None
            # `tokenizer_path` should be the name of a HuggingFace tokenizer (not a file path). If it is not specified,
            # assume the data/model dir is a valid HuggingFace space
            self._tokenizer = HuggingFaceTokenizer(self.cfg.tokenizer_path or self.dir)
            if self.dictionary is None:  # build the dictionary from this tokenizer's built-in vocab
                self.dictionary = Dictionary(self._tokenizer.vocab)
        elif self.cfg.tokenizer == 'pasero':
            self.tokenizer_path = utils.find_file(self.cfg.tokenizer_path, dirs=[self.dir, '.'])
            self._tokenizer = PaseroTokenizer(
                self.tokenizer_path, self.dictionary, inline_case=self.cfg.inline_case
            )
            logger.info(f'{self._tokenizer.__class__.__name__}: {len(self._tokenizer)} merge operations')
        elif self.cfg.tokenizer == 'sentencepiece':
            self.tokenizer_path = utils.find_file(self.cfg.tokenizer_path, dirs=[self.dir, '.'])
            self._tokenizer = SentencePieceTokenizer(
                self.tokenizer_path, self.dictionary, inline_case=self.cfg.inline_case
            )
            if self.dictionary is None:  # build the dictionary from this tokenizer's built-in vocab
                self.dictionary = Dictionary(self._tokenizer.vocab)
            logger.info(f'{self._tokenizer.__class__.__name__}: {len(self._tokenizer)} merge operations')
        else:
            raise ValueError(f"Unknown tokenizer type: '{self.cfg.tokenizer}'")

        assert self.dictionary is not None

    @property
    def inference_options(self) -> dict:
        """
        Returns the preprocessing options that are needed to run the model at inference.
        Called by `Task` at training to generate an config file with all the inference options.
        
        This can be different from the options at training because: 1) some options are only used at training,
        2) some paths may have changed (tokenizer files are copied to the model directory)
        """
        noise_options = NoiseConfig().as_dict()  # options to exclude since they are only for training:
        # PreprocessingConfig is a subclass of NoiseConfig, so it contains all its options
        defaults = PreprocessingConfig().as_dict()

        options = {}
        for name, default in defaults.items():
            value = getattr(self.cfg, name)
            if name not in noise_options and value != default:
                options[name] = value

        paths = {'tokenizer_path': self.tokenizer_path, 'dict': self.dict_path}

        for name, value in paths.items():
            options.pop(name, None)  # paths given in training config may have changed (they can be relative to the 
            # data dir or working dir), use the values in `paths` instead
            if value is not None:
                value = os.path.basename(value)  # preprocessor files will be copied at the root of the model directory
                options[name] = value

        if self.cfg.tokenizer == 'hf':  # HuggingFace tokenizer files are not copied to the model directory, 
            # so the inference options should just contain the same path or HF model name as in training
            options['tokenizer_path'] = self.cfg.tokenizer_path

        return options

    def get_oov(self, tokens: list[str]) -> tuple[collections.Counter, set]:
        counts = collections.Counter(token for token in tokens)
        oov = {w for w in counts if w not in self.dictionary}
        return counts, oov

    def binarize(
        self,
        tokens: list[str],
        max_len: Optional[int] = None,
        append_eos: bool = True,
        prepend_bos: bool = False,
        as_tensor: bool = False,
        truncate_left: bool = False,
    ) -> Union[np.ndarray, torch.LongTensor]:
        ids = self.dictionary.to_indices(
            tokens,
            max_len=max_len,
            append_eos=append_eos,
            prepend_bos=prepend_bos,
            truncate_left=truncate_left,
        )
        return torch.as_tensor(ids, dtype=torch.long) if as_tensor else ids

    def escape_emojis(self, line: str) -> tuple[str, list[str]]:
        dictionary = self.dictionary
        if copy_tag in dictionary:
            placeholder = copy_tag
        elif '🙂' in dictionary:
            placeholder = '🙂'
        else:
            return line, []

        import emoji
        emojis = []
        for e in emoji.emoji_list(line):
            e = e['emoji']
            if e not in dictionary:
                emojis.append(e)
                line = line.replace(e, placeholder)
        return line, emojis

    def deescape_emojis(self, line: str, emojis: list[str]) -> str:
        placeholder = copy_tag if copy_tag in line else '🙂'
        for emoji in emojis:
            line = regex.sub(regex.escape(placeholder), emoji, line, count=1)
        line = line.replace(copy_tag, '')
        return ' '.join(line.split(' '))

    def tokenize(self, line: str) -> list[str]:
        if not self.cfg.keep_whitespaces:
            line = remove_non_printing_char(line)
            line = ' '.join(line.split())  # the line above may result in consecutive whitespaces

            if line and self.cfg.normalize_punctuation:
                line = replace_unicode_punct(line)

        if self.training:
            line = noisify(line, **vars(self.cfg))

        if not line or self.cfg.tokenizer == 'none':
            return line.split()
        elif self.cfg.tokenizer != 'none':
            dropout = self.cfg.bpe_dropout if self.training else 0.0
            spell_out = self.cfg.spell_out if self.training else 0.0
            tokens = []
            for split in self.split_tokens_regex.split(line):
                if not split:
                    continue
                elif self.protected_tokens_regex.fullmatch(split):
                    tokens.append(split)
                else:
                    # FIXME: this will add a prefix whitespace after each special token, is this ok?
                    # For example: "Hello </s>World!" -> ['▁Hello', '▁', '</s>', '▁World']
                    tokens += self._tokenizer.tokenize(split, dropout=dropout, spell_out=spell_out)
            return tokens
    
    def debinarize(self, ids: Sequence[int]) -> list[str]:
        """
        Transform a sequence of dictionary ids into a sequence of tokens.

        ```
        debinarize([109, 4, 23911, 4122, 35, 87, 61, 4676, 491, 2839, 2292])
        # ['▁les', '<T>', '▁chauss', 'ettes', '▁de', '▁l', '\'', 'arch', 'id', 'uch', 'esse']"
        ```
        """
        return self.dictionary.to_string(ids)

    def detokenize(self, tokens: list[str]) -> str:
        """
        Transform a sequence of (whitespace-delimited) tokens into a sequence of words.

        ```
        detokenize(['▁les', '<T>', '▁chauss', 'ettes', '▁de', '▁l', '\'', 'arch', 'id', 'uch', 'esse'])
        # "Les chaussettes de l'archiduchesse"
        ```
        """
        tokens = self.remove_special_tokens(tokens)  # remove <unk> and </s>

        if self._tokenizer is not None:
            line = self._tokenizer.detokenize(tokens)
        else:
            line = ' '.join(tokens)
        
        if not self.cfg.keep_whitespaces:
            line = line.rstrip()  # remove trailing line breaks and whitespaces

        return line

    def set_stop_sequences(self, stop_sequences: list[str]) -> None:
        """
        Convert the --stop-sequences option (list of tokenized text sequences) into a sequence of ids that can be 
        understood by the model or the decoding algorithms.
        """
        self.raw_stop_sequences = stop_sequences
        self.tok_stop_sequences = [self.tokenize(stop_seq) for stop_seq in self.raw_stop_sequences]
        self.bin_stop_sequences = [
            self.binarize(tokens, append_eos=False, as_tensor=True)
            for tokens in self.tok_stop_sequences
        ]
    
    @property
    def blacklist(self) -> list[int]:
        """
        Convert the --blacklist option (list of tokens) into a list of dictionary ids that can be understood by the
        model or the decoding algorithms.
        """
        return [self.dictionary.idx(token) for token in self.cfg.blacklist]

    def is_special_token(self, token: str) -> bool:
        return (
            token in (unk, eos) or
            any(len(stop_seq) == 1 and token == stop_seq[0] for stop_seq in self.tok_stop_sequences)  # strip stop
            # sequences that contain a single token
        )

    def remove_special_tokens(self, tokens: list[str]) -> list[str]:
        tokens = [token for token in tokens if not self.is_special_token(token)]
        # decoding stops *after* generating the stop sequences, so we need to strip them from the output
        for stop_seq in self.tok_stop_sequences:
            if stop_seq and len(stop_seq) > 1:  # stop sequences of length 1 were already stripped above
                if tokens[-len(stop_seq):] == stop_seq:
                    tokens = tokens[:-len(stop_seq)]
                    break
        return tokens

    def detokenize_on_the_fly(self, tokens: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
        """ 
        Detokenize given stream of tokens on the fly (i.e., into a stream of words).
        The returned iterator yields tuples `(word, [tokens...])`

        For instance:

        ```
        tokens = ['▁les', '<T>', '▁chauss', 'ettes', '▁de', '▁l', "'", 'arch', 'id', 'uch', 'esse']
        token_stream = iter(tokens)

        for word, tokens in detokenize_on_the_fly(token_stream):
            print(repr(word), tokens)

        ' Les'             ['▁les', '<T>']
        ' chaussettes'     ['▁chauss', 'ettes']
        ' de'              ['▁de']
        " l'archiduchesse" ['▁l', "'", 'arch', 'id', 'uch', 'esse']
        """
        yield from self._tokenizer.detokenize_on_the_fly(
            token for token in tokens if not self.is_special_token(token)
        )

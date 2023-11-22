# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
from typing import Optional, Union
import regex
import itertools
import collections
import logging
import numpy as np
import torch
import copy
from typing import Optional, Sequence, Iterable, Iterator
from pasero.tokenizers import PaseroTokenizer, SentencePieceTokenizer, HuggingFaceTokenizer, CharacterTokenizer, load_vocab
from pasero.tokenizers.noise import mask, noisify
from pasero import utils
from pasero.config import PreprocessingConfig, NoiseConfig
from stopes.pipelines.monolingual.utils.text_normalizer import remove_non_printing_char, replace_unicode_punct


BPEModel = Union[HuggingFaceTokenizer, PaseroTokenizer, SentencePieceTokenizer, CharacterTokenizer]


logger = logging.getLogger('preprocessing')


sep, bos, pad, eos, unk = '<sep>', '<s>', '<pad>', '</s>', '<unk>'
copy_tag = '<PHL>'


def mask_padding(ids: Sequence[int], eos_idx: int, padding_idx: int) -> list[int]:
    """ Return a mask with ones at padding token positions or after the EOS """
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
    """

    def __init__(self, path: str, size: Optional[int] = None):
        """
        Args:
            path: path to the dict file (text file with one token and its frequency per line, delimited by a
                whitespace)
            size: pad the dictionary to this size
        """
        self.tokens = list(load_vocab(path))
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
            self.tokens = [sep, pad, eos, unk] + self.tokens  # <s> is not used in fairseq checkpoints, so we use this
            # position for something else

        self.tokens = self.tokens[:size]
        i = 0
        while size is not None and size > len(self.tokens):
            w = f'madeupword{i:04}'
            if w not in token_set:
                self.tokens.append(w)
            i += 1
        self.indices = {w: i for i, w in enumerate(self.tokens)}

        # at the very minimum eos_idx and padding_idx have to be defined
        self.eos_idx = self.indices[eos]
        self.padding_idx = self.indices.get(pad, -1)
        self.bos_idx = self.indices.get(bos, self.eos_idx)      # if not defined, bos_idx = eos_idx
        self.unk_idx = self.indices.get(unk, self.padding_idx)  # if not defined, unk_idx = padding_idx
        self.sep_idx = self.indices.get(sep, self.bos_idx)
        
        self.special_tokens = utils.SpecialTokens(
            padding_idx=self.padding_idx,
            eos_idx=self.eos_idx,
            bos_idx=self.bos_idx,
            unk_idx=self.unk_idx,
            sep_idx=self.sep_idx,
        )

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.indices

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def __eq__(self, other) -> bool:
        return isinstance(other, Dictionary) and other.tokens == self.tokens

    def idx(self, token: str) -> int:
        return self.indices.get(token, self.unk_idx)

    def to_indices(
        self,
        line: str,
        max_len: Optional[int] = None,
        append_eos: bool = True,
        prepend_bos: bool = False,
        truncate_left: bool = False,
    ) -> np.ndarray:
        tokens = line.split(' ') if line else []
        ids = [self.idx(token) for token in tokens]
        
        if max_len is not None:
            max_len = max_len - int(append_eos) - int(prepend_bos)
            ids = ids[-max_len:] if truncate_left else ids[:max_len]
        if prepend_bos:
            ids.insert(0, self.bos_idx)
        if append_eos:
            ids.append(self.eos_idx)
        return np.array(ids, dtype=np.int64)

    def to_string(self, ids: Sequence[int], keep_padding: bool = False) -> str:
        mask = mask_padding(ids, eos_idx=self.eos_idx, padding_idx=self.padding_idx)
        tokens = [self[token_id] for token_id, is_padding in zip(ids, mask) if keep_padding or not is_padding]
        return ' '.join(tokens)

    def remap_embed(self, old_embed: torch.Tensor, old_dict, default: Optional[str] = None) -> torch.Tensor:
        default_idx = old_dict.indices[default] if default else None
        embed = []
        unk_count = 0
        for index, token in enumerate(self.tokens):
            token = self.tokens[index]
            if token in old_dict.indices:
                old_index = old_dict.indices[token]
                v = old_embed[old_index]
            elif default is None:
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

_LANG_CODE_REGEX = regex.compile(f'<{regex.escape(_LANG_CODE_PREFIX)}([^>]+)>')
_DOMAIN_TAG_REGEX = regex.compile(f'<{regex.escape(_DOMAIN_TAG_PREFIX)}([^>]+)>')

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

        # Parameters in `cfg` can be overloaded with keyword arguments
        for k, v in kwargs.items():
            setattr(self.cfg, k, v)

        self.dict_path = self.tokenizer_path = self.vocab_path = None

        self.load_dict()
        assert not cfg.masking or mask in self.dictionary, f'{mask} is OOV'

        special_tokens = self.dictionary.special_tokens
        self.padding_idx = special_tokens.padding_idx
        self.eos_idx = special_tokens.eos_idx
        self.unk_idx = special_tokens.unk_idx
        self.bos_idx = special_tokens.bos_idx
        self.special_tokens = special_tokens

        self.load_tokenizer()  # TODO: allow per-language BPE vocabularies
    
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

    def _find_file(self, path: str):
        """ Looks for a file whose path is either absolute or relative to `self.dir` """
        return utils.find_file(path, dirs=[self.dir, '.'])

    def load_dict(self):
        self.dict_path = self._find_file(self.cfg.dict)
        self.dictionary = Dictionary(self.dict_path)
        logger.info(f'{self.dictionary.__class__.__name__}: {len(self.dictionary)} symbols')

    @property
    def files(self):
        """
        Set of files that should be copied to the model directory
        """
        return {self.tokenizer_path, self.dict_path} - {None}

    def load_tokenizer(self) -> Optional[BPEModel]:
        if self.cfg.tokenizer == 'none':
            self._tokenizer = None
            return
        elif self.cfg.tokenizer == 'char':
            self._tokenizer = CharacterTokenizer()
            return
        elif self.cfg.tokenizer == 'hf':
            # `tokenizer` should be the name of a HuggingFace tokenizer (not a file path)
            self._tokenizer = HuggingFaceTokenizer(
                self.cfg.tokenizer_path or self.dir,
                add_prefix_space=self.cfg.hf_add_prefix_space,
            )
            # self.tokenizer_path = None for HF tokenizers, since they don't correspond to a single file, but rather
            # a full model directory or HuggingFace repo
            return

        self.tokenizer_path = self._find_file(self.cfg.tokenizer_path)

        if self.cfg.tokenizer_vocab and self.cfg.vocabulary_threshold != -1:
            self.vocab_path = self._find_file(self.cfg.tokenizer_vocab)
            # intersection of provided vocab and Pasero dictionary
            vocab = load_vocab(self.vocab_path, self.cfg.vocabulary_threshold)
            vocab = [k for k in vocab if k in self.dictionary]
        else:  # we don't want the BPE model to generate OOV subwords
            vocab = self.dictionary

        if self.cfg.tokenizer == 'pasero':
            self._tokenizer = PaseroTokenizer.read(self.tokenizer_path, vocab=vocab, inline_case=self.cfg.inline_case)
        else:
            self._tokenizer = SentencePieceTokenizer(self.tokenizer_path, vocab, inline_case=self.cfg.inline_case)
        
        desc = f'{self._tokenizer.__class__.__name__}: {len(self._tokenizer)} merge operations'
        if self._tokenizer.vocab:
            desc += f', vocab size {len(self._tokenizer.vocab)}'
        logger.info(desc)

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

        paths = {
            'tokenizer_path': self.tokenizer_path,
            'dict': self.dict_path,
            'tokenizer_vocab': self.vocab_path,
        }

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

    def get_oov(self, line: str) -> tuple[collections.Counter, set]:
        counts = collections.Counter(token for token in line.split(' '))
        oov = {w for w in counts if w not in self.dictionary}
        return counts, oov

    def binarize(
        self,
        line: str,
        max_len: Optional[int] = None,
        append_eos: bool = True,
        prepend_bos: bool = False,
        as_tensor: bool = False,
        truncate_left: bool = False,
    ) -> Optional[Union[np.ndarray, torch.LongTensor]]:
        ids = self.dictionary.to_indices(
            line,
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

    def tokenize(self, line: str) -> Optional[str]:
        if not self.cfg.keep_whitespaces:
            line = remove_non_printing_char(line)
            line = ' '.join(line.split())  # the line above may result in consecutive whitespaces

            if line and self.cfg.normalize_punctuation:
                line = replace_unicode_punct(line)

        if self.training:
            line = noisify(line, **vars(self.cfg))

        if line and self.cfg.tokenizer != 'none':
            dropout = self.cfg.bpe_dropout if self.training else 0.0
            spell_out = self.cfg.spell_out if self.training else 0.0
            line = self._tokenizer.tokenize(line, dropout=dropout, spell_out=spell_out)

        return line
    
    def debinarize(self, ids: Sequence[int], keep_padding: bool = False) -> str:
        """
        Transform a sequence of dictionary ids into a sequence of tokens.

        ```
        debinarize([109, 4, 23911, 4122, 35, 87, 61, 4676, 491, 2839, 2292])
        # "▁les <T> ▁chauss ettes ▁de ▁l ' arch id uch esse"
        ```
        """
        return self.dictionary.to_string(ids, keep_padding=keep_padding)

    def detokenize(self, line: str) -> str:
        """
        Transform a sequence of (whitespace-delimited) tokens into a sequence of words.

        ```
        detokenize("▁les <T> ▁chauss ettes ▁de ▁l ' arch id uch esse")
        # "Les chaussettes de l'archiduchesse"
        ```
        """
        line = self.remove_special_tokens(line)  # remove <unk> and </s>

        if self._tokenizer is not None:
            line = self._tokenizer.detokenize(line)
        
        if not self.cfg.keep_whitespaces:
            line = line.rstrip()  # remove trailing line breaks and whitespaces

        return line

    @property
    def stop_sequences(self) -> list[torch.LongTensor]:
        """
        Convert the --stop-sequences option (list of tokenized text sequences) into sequences of ids that can be 
        understood by the model or the decoding algorithms.
        """
        stop_sequences = [
            self.binarize(stop_seq, append_eos=False, as_tensor=True)
            for stop_seq in self.cfg.stop_sequences
        ]
        return [stop_seq for stop_seq in stop_sequences if len(stop_seq) > 0]
    
    @property
    def blacklist(self) -> list[int]:
        """
        Convert the --blacklist option (list of tokens) into a list of dictionary ids that can be understood by the
        model or the decoding algorithms.
        """
        return [self.dictionary.idx(token) for token in self.cfg.blacklist]

    def remove_special_tokens(self, line: str) -> str:
        special_tokens = {unk, eos}
        tokens = []
        for token in line.split():
            if token not in special_tokens:
                tokens.append(token)
        line = ' '.join(tokens)
        # decoding stops *after* generating the stop sequences, so we need to strip them from the output
        for stop_seq in self.cfg.stop_sequences:
            line = line.removesuffix(stop_seq).rstrip(' ')
        return line

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
            self.remove_special_tokens(tok)
            for tok in tokens
        )

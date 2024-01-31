# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import regex
import unicodedata
from typing import Optional, Iterable, Iterator
from .noise import mask
from .pasero_tokenizer import inline_case_to_cased, PaseroTokenizer, detokenize
from .pasero_tokenizer import _NO_MIXED_CASE_REGEX, _TITLE_CODE, _UPPER_CODE, _LOWER_CODE, _CASE_SYMBOLS


sep, bos, pad, eos, unk = '<sep>', '<s>', '<pad>', '</s>', '<unk>'


def load_vocab(path: str, threshold: Optional[int] = None) -> list[str]:
    """
    Loads a vocabulary in the Pasero/fairseq format: one token per line, with an optional frequency.
    Returns the ordered list of tokens. This is typically used to initialize `Dictionary`, by using the token's 
    position in the list as token id (after prepending special tokens).
    Note that this is different from `HuggingFaceTokenizer.vocab` which directly returns a dict mapping each 
    token to its id.
    """
    with open(path, newline='\n') as vocab_file:
        vocab = []
        for line in vocab_file:
            word, freq = regex.match(r'(.+?)(\s\d+)?$', line).groups()
            freq = int(freq) if freq else 0
            if threshold is None or freq >= threshold:
                vocab.append(word)
        return vocab


class SentencePieceTokenizer(object):
    def __init__(self, path: str, vocab: Optional[list[str]] = None, inline_case: bool = False):
        self.path = path
        self.inline_case = inline_case
        import sentencepiece as spm
        self._tokenizer = spm.SentencePieceProcessor(model_file=path)
        if vocab:
            self._tokenizer.SetVocabulary(list(vocab))
            self._vocab = vocab
        else:
            self._vocab = [self._tokenizer.IdToPiece(i) for i in range(self._tokenizer.vocab_size())]

    def __getstate__(self):
        return {
            'path': self.path,
            'vocab': self._vocab,
            'inline_case': self.inline_case,
        }

    def __setstate__(self, state):
        # SentencePiece models can have issues with serialization (e.g., Llama's tokenizer doesn't work well with 
        # multiprocessing's spawn method), so when pickling/unpickling the model, we only store its configuration and
        # reload it.
        self.__init__(**state)

    def __len__(self):
        return len(self._tokenizer)

    @staticmethod
    def _clean(line):
        return regex.sub(r'\s+', ' ', line).strip()

    def _get_case(self, s):
        if s.istitle():
            return '<T>'
        if s.isupper():
            return '<U>'
        elif s.islower() or s.lower() == s:
            return _LOWER_CODE
        else:
            return None

    def _tokenize(self, x: str) -> list[str]:
        pieces = []
        for piece in self._tokenizer.EncodeAsPieces(x):
            if self._tokenizer.IsUnknown(self._tokenizer.PieceToId(piece)):
                pieces += list(piece)
            else:
                pieces.append(piece)
        return pieces

    def tokenize(self, x: str, **_) -> list[str]:
        if not self.inline_case:
            return self._tokenize(x)

        orig = self._clean(unicodedata.normalize('NFKC', x))
        orig_lower = ' '.join(y if len(x) == len(y) else x for x, y in ((w, w.lower()) for w in orig.split()))
        # only lowercase words whose length is not modified by lowercasing
        wordpieces = self._tokenize(orig_lower)

        output = []
        j = 0
        for wordpiece in wordpieces:
            if wordpiece == '▁':
                output.append(wordpiece)
                continue

            prefix = ''
            try:
                if wordpiece.startswith('▁'):
                    prefix = '▁'
                    wordpiece = wordpiece[1:]
                i = orig_lower.find(wordpiece, j)
            except:
                output.append(prefix + wordpiece)
                continue

            j = i + len(wordpiece)
            cased = orig[i:j]

            case = self._get_case(cased)
            if len(cased) == len(wordpiece) and case is None:
                cased_split = _NO_MIXED_CASE_REGEX.findall(cased)
                k = 0
                for n, s in enumerate(cased_split):
                    case = self._get_case(s)
                    output += [(prefix if n == 0 else '') + wordpiece[k:k + len(s)], _CASE_SYMBOLS[case]]
                    k += len(s)
            else:
                output += [prefix + wordpiece, _CASE_SYMBOLS[case]]

        return [w for w in output if w is not None]

    def detokenize(self, tokens: list[str]) -> str:
        if '<T>' in tokens or '<U>' in tokens:
            tokens = inline_case_to_cased(tokens)
        line = self._tokenizer.decode(tokens)

        # The Llama tokenizer uses hexadecimal codes to encode some special symbols (e.g., line break, tabulation, etc.)
        # Those should be converted automatically by 'decode', but this doesn't work when setting a vocabulary
        # with sp.SetVocabulary, so we convert them manually. Examples:
        # '<0x0A>' -> b'\x0a' -> '\n'
        # '<0xE2><0x99><0xAA>' -> b'\xe2\x99\xaa' -> '♪'

        # Split the string into non-hex and hex parts
        segments = regex.split(r'((?:<0x..>)+)', line)
        
        # Convert the hexadecimal byte sequences and recombine the segments
        for i, segment in enumerate(segments):
            if segment.startswith('<0x'):
                try:
                    segments[i] = bytes.fromhex(segment.replace('<0x', '').replace('>', '')).decode()
                except UnicodeDecodeError:
                    segments[i] = ''
        return ''.join(segments)

    def detokenize_on_the_fly(self, tokens: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
        def detok(tokens: list[str]) -> str:
            prefix = ' ' if tokens[0][0] == '▁' else ''
            return prefix + self.detokenize(tokens)
        
        prev_tokens = []
        for token in tokens:
            if not token:
                continue
            if prev_tokens and token[0] == '▁':
                yield detok(prev_tokens), prev_tokens
                prev_tokens = []
            prev_tokens.append(token)
        if prev_tokens:
            yield detok(prev_tokens), prev_tokens

    @property
    def vocab(self) -> list[int]:
        return self._vocab


class HuggingFaceTokenizer(object):
    def __init__(self, path: str):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        # TODO: should we set legacy=False? https://github.com/huggingface/transformers/pull/24565
        vocab = dict(self._tokenizer.vocab)
        # Remap special tokens that may have a different name according to this tokenizer
        eos_token = self._tokenizer.eos_token
        if eos_token is not None:
            vocab[eos] = vocab[eos_token]
        bos_token = self._tokenizer.bos_token
        if bos_token is not None:
            vocab[bos] = vocab[bos_token]
        pad_token = self._tokenizer.pad_token
        if pad_token is not None and pad_token != eos_token:
            vocab[pad] = vocab[pad_token]
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._tokenizer)

    def tokenize(self, x: str, **_) -> list[str]:
        tokens = self._tokenizer.tokenize(x)
        return tokens

    def detokenize(self, tokens: list[str]) -> str:
        return self._tokenizer.convert_tokens_to_string(tokens)

    def detokenize_on_the_fly(self, tokens: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
        all_tokens = ['.']  # start with non empty prefix to disable SentencePiece's annoying behavior with prefix 
        # whitespace:
        # ["▁Hello", "▁world"] -> "Hello world" (no whitespace before "Hello", because it is the first token)
        # [".", "▁Hello", "▁world"] -> ". Hello world" (whitespace before "Hello", because it isn't the first token)
        prev_detok = self._tokenizer.convert_tokens_to_string(all_tokens)  # this dummy prefix will be stripped
        
        for token in tokens:
            all_tokens.append(token)
            detok = self._tokenizer.convert_tokens_to_string(all_tokens).rstrip('�')
            word = detok[len(prev_detok):]
            yield word, [token]
            prev_detok = detok

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab


class CharacterTokenizer(object):
    def __init__(self):
        self._vocab = self._tokenizer = None

    def __len__(self) -> int:
        return 0

    def tokenize(self, x: str, **_) -> list[str]:
        x = ' '.join(x.split()).replace(' ', '▁')
        return list(x)

    def detokenize(self, tokens: list[str]) -> str:
        x = ''.join(tokens).replace('▁', ' ')
        return ' '.join(x.split())

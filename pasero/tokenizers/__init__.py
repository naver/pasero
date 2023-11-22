# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import sys
import regex
import unicodedata
from collections import defaultdict
from typing import Optional, Iterable, Iterator
from .noise import mask
from .pasero_tokenizer import inline_case_to_cased, PaseroTokenizer, detokenize
from .pasero_tokenizer import _NO_MIXED_CASE_REGEX, _TITLE_CODE, _UPPER_CODE, _LOWER_CODE, _CASE_SYMBOLS



def load_vocab(path: str, threshold: Optional[int] = None) -> list[str]:
    with open(path, newline='\n') as vocab_file:
        vocab = []
        for line in vocab_file:
            word, freq = regex.match(r'(.+?)(\s\d+)?$', line).groups()
            freq = int(freq) if freq else 0
            if threshold is None or freq >= threshold:
                vocab.append(word)
        return vocab


def build_dict(vocab, dict_path=None, dict_custom_symbols=[], dict_placeholders=0, dict_padding_offset=4,
               dict_padding_factor=8, dict_min_freq=10, dict_max_size=None, **_):
    dictionary = dict.fromkeys(['<T>', '<U>', '<BT>', '<PHL>', mask], 0)
    
    if not isinstance(vocab, dict):  # vocab can be a list or set
        vocab = dict.fromkeys(vocab, 0)

    vocab = dict(vocab)    # convert Counter to dict (because Counter's update has a different behavior)
    # count all characters and add missing characters to the dictionary
    chars = defaultdict(int)
    for word, count in vocab.items():
        if word not in dictionary:   # do not spell out special tokens
            for char in word:
                chars[char] += count
    vocab.update(chars)   # unexpected behavior with Counter
    vocab = {w: c for w, c in vocab.items() if not c or c >= dict_min_freq}
    vocab = sorted(vocab.items(), key=lambda p: (-p[1], p[0]))  # sort by count, then alphabetically
    dictionary.update(dict(vocab))
    
    special_symbols = []
    for token in sorted(dict_custom_symbols):
        if token not in dictionary:
            special_symbols.append((token, 0))

    i = 0
    for _ in range(dict_placeholders):
        special_symbols.append((f'madeupword{i:04}', 0))
        i += 1

    dictionary = list(dictionary.items())

    if dict_max_size is not None:
        assert len(special_symbols) < dict_max_size
        dictionary = dictionary[:dict_max_size - len(special_symbols)]
    
    dictionary += special_symbols

    while (len(dictionary) + dict_padding_offset) % dict_padding_factor != 0:
        dictionary.append((f'madeupword{i:04}', 0))
        i += 1

    if dict_path is not None:
        if dict_path == '-':
            dict_file = sys.stdout
        else:
            dirname = os.path.dirname(dict_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            dict_file = open(dict_path, 'w')
        dict_file.writelines(f'{token} {count}\n' for token, count in dictionary)
    return dictionary


class SentencePieceTokenizer(object):
    def __init__(self, sentencepiece_model, vocab=None, inline_case=False):
        self.sentencepiece_model = sentencepiece_model
        self.vocab = vocab
        self.inline_case = inline_case
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=sentencepiece_model)
        
        if vocab:
            self.sp.SetVocabulary(list(vocab))

    def __getstate__(self):
        return {
            'sentencepiece_model': self.sentencepiece_model,
            'vocab': self.vocab,
            'inline_case': self.inline_case,
        }

    def __setstate__(self, state):
        # SentencePiece models can have issues with serialization (e.g., Llama's tokenizer doesn't work well with 
        # multiprocessing's spawn method), so when pickling/unpickling the model, we only store its configuration and
        # reload it.
        self.__init__(**state)

    def __len__(self):
        return len(self.sp)

    @staticmethod
    def _clean(line):
        return regex.sub(r'\s+', ' ', line).strip()

    def _get_case(self, s):
        if s.istitle():
            return _TITLE_CODE
        if s.isupper():
            return _UPPER_CODE
        elif s.islower() or s.lower() == s:
            return _LOWER_CODE
        else:
            return None

    def _tokenize(self, x):
        pieces = []
        for piece in self.sp.EncodeAsPieces(x):
            if self.sp.IsUnknown(self.sp.PieceToId(piece)):
                pieces += list(piece)
            else:
                pieces.append(piece)
        return ' '.join(pieces)

    def tokenize(self, x, **_):
        if not self.inline_case:
            return self._tokenize(x)

        orig = self._clean(unicodedata.normalize('NFKC', x))
        orig_lower = ' '.join(y if len(x) == len(y) else x for x, y in ((w, w.lower()) for w in orig.split()))
        # only lowercase words whose length is not modified by lowercasing
        line = self._clean(self._tokenize(orig_lower))

        output = []
        j = 0
        for wordpiece in line.split():
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

        return ' '.join(w for w in output if w is not None)

    def detokenize(self, line):
        tokens = line.split(' ')
        if '<T>' in tokens or '<U>' in tokens:
            tokens = inline_case_to_cased(tokens)
        line = self.sp.decode(tokens)

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
            return prefix + self.detokenize(' '.join(tokens))
        
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


class HuggingFaceTokenizer(object):
    def __init__(self, huggingface_model: str, add_prefix_space: bool = False):
        # FIXME: should add_prefix_space=True be the default behavior? It seems the BLOOM models where not trained
        # this way (https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/tokenization_bloom_fast.py#L68)
        from transformers import AutoTokenizer
        self.add_prefix_space = add_prefix_space
        self.model = AutoTokenizer.from_pretrained(huggingface_model)
        self.vocab = None

    def __len__(self) -> int:
        return len(self.model)

    def tokenize(self, x: str, **_) -> str:
        if self.add_prefix_space:
            x = f' {x}'
        tokens = self.model.tokenize(x)
        tokens = [token.replace(' ', 'Ġ') for token in tokens]  # the whitespace ' ' has a special meaning in 
        # Pasero, it is used as a delimiter betwen tokens. Make sure that the subwords do not contain whitespaces.
        # FIXME: this trick works with the BLOOM and MPT tokenizers but may fail with other tokenizers
        return ' '.join(tokens)

    def detokenize(self, x: str) -> str:
        x = self.model.convert_tokens_to_string(x.split(' '))
        if self.add_prefix_space:
            x = x.removeprefix(' ')
        return x

    def detokenize_on_the_fly(self, tokens: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
        all_tokens = ['.']  # start with non empty prefix to disable SentencePiece's annoying behavior with prefix 
        # whitespace:
        # ["▁Hello", "▁world"] -> "Hello world" (no whitespace before "Hello", because it is the first token)
        # [".", "▁Hello", "▁world"] -> ". Hello world" (whitespace before "Hello", because it isn't the first token)
        prev_detok = self.model.convert_tokens_to_string(all_tokens)  # this dummy prefix will be stripped
        
        for token in tokens:
            all_tokens.append(token)
            detok = self.model.convert_tokens_to_string(all_tokens).rstrip('�')
            word = detok[len(prev_detok):]
            yield word, [token]
            prev_detok = detok


class CharacterTokenizer(object):
    def __init__(self):
        self.vocab = self.model = None

    def __len__(self) -> int:
        return 0

    def tokenize(self, x: str, **_) -> str:
        x = ' '.join(x.split()).replace(' ', '▁')
        return ' '.join(x)

    def detokenize(self, x: str) -> str:
        x = x.replace('▁', ' ')
        return ' '.join(x.split())

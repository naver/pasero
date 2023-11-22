#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import regex
import io
import argparse
import logging
from collections import Counter, defaultdict

from pasero.tokenizers import PaseroTokenizer, build_dict, detokenize, load_vocab, noise
from pasero.tokenizers.noise import noisify
from pasero.preprocessing import _LANG_CODE_PREFIX, split_tags

def init_logging(stream=sys.stderr):
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level='INFO',
        stream=stream,
    )

description="""
Pasero Tokenizer

Custom features:
- SentencePiece-like tokenization: whitespaces are kept and replaced with '▁'
- splitting rules based on the script (--split-by-script) and character type (whitespace/digit/punctuation/letter)
- NFKC normalization
- NAVER LABS Europe's inline casing
- fairseq/Pasero dictionary generation
- temperature-based sampling from multiple files

Available commands
------------------
pasero-tokenize: tokenize text into subwords using a Pasero BPE model
pasero-detokenize: merge BPE units back into words
pasero-build-tokenizer: create a new Pasero BPE model and dictionary
pasero-build-dict: create a Pasero dictionary from tokenized text
pasero-noisify: insert random noise
"""

tokenize_parser = argparse.ArgumentParser()
tokenize_parser.add_argument('bpe_codes', help='path to the BPE model (text file containing the merge operations)')
tokenize_parser.add_argument('--input', '-i', help="input file (default: standard input)")
tokenize_parser.add_argument('--output', '-o', help="output file (default: standard output)")
tokenize_parser.add_argument('--vocabulary', help='path to a vocabulary containing pairs of subwords and corresponding '
                             'their frequency')
tokenize_parser.add_argument('-t', '--threshold', help='only generate subwords whose frequency in the vocabulary file '
                             'is at least this value')
tokenize_parser.add_argument('--unk', help="replace OOV tokens by this symbol")
tokenize_parser.add_argument('--spell-out', type=float, default=0.0, help='spell out each subword with this '
                             'probability')
tokenize_parser.add_argument('--dropout', type=float, default=0.0, help='BPE dropout rate')


detokenize_parser = argparse.ArgumentParser()
detokenize_parser.add_argument('--input', '-i', help="input file (default: standard input)")
detokenize_parser.add_argument('--output', '-o', help="output file (default: standard output)")


noise_parser = argparse.ArgumentParser()
noise_parser.add_argument('--input', '-i', help="input file (default: standard input)")
noise_parser.add_argument('--output', '-o', help="output file (default: standard output)")
noise_parser.add_argument('--seed', type=int, default=1234, help="random seed for reproducible noise")
noise.add_args(noise_parser)


def add_dict_args(parser):
    parser.add_argument('--dict-placeholders', type=int, default=0, help='pad the dictionary with this many dummy '
                        'symbols (default: %(default)s)')
    parser.add_argument('--dict-padding-factor', type=int, default=8, help='the dictionary size (including special '
                        'symbols) must be a multiple of this value (default: %(default)s)')
    parser.add_argument('--dict-padding-offset', type=int, default=4, help='number of special symbols that are not '
                        'included in the dictionary (default: %(default)s)')
    parser.add_argument('--dict-min-freq', type=int, default=10, help='minimum frequency of a character to be '
                        'included in the dictionary (default: %(default)s)')
    parser.add_argument('--dict-char-coverage', type=float, default=1, help='only the most frequent characters will '
                        'be kept, whose total coverage exceeds this ratio')
    parser.add_argument('--dict-custom-symbols', nargs='*', default=[], help='add these symbols to the dictionary')
    parser.add_argument('--dict-max-size', type=int, help='maximum size of the dictionary')


dict_parser = argparse.ArgumentParser()
dict_parser.add_argument('--input', '-i', help="input file (default: standard input)")
dict_parser.add_argument('-o', '-d', '--dict-path', help='output path of the generated dictionary', default='-')
dict_parser.add_argument('--max-lines', type=int, help='maximum number of lines to read from the input. The real line '
                         'counts and word counts will be estimated from file sizes')
add_dict_args(dict_parser)


train_parser = argparse.ArgumentParser()
train_parser.add_argument('--inputs', '-i', metavar='PATH', nargs='+', help='input text (default: standard '
                          'input)')
train_parser.add_argument('--output', '-o', metavar='PATH', help='output file for BPE codes (default: standard output)')
train_parser.add_argument('--existing-bpe-path', help='load this BPE model and generate vocabularies with it')
train_parser.add_argument('--symbols', '-s', dest='num_symbols', type=int, default=8000, help='number of merge '
                          'operations (default: %(default)s)')
train_parser.add_argument('--verbose', '-v', action=argparse.BooleanOptionalAction, default=False, help='verbose mode')
train_parser.add_argument('--nfkc', action=argparse.BooleanOptionalAction, default=False, help='perform Unicode NFKC '
                          'normalization')
train_parser.add_argument('--split-by-script', action=argparse.BooleanOptionalAction, default=True, help='split by '
                          'Unicode script')
train_parser.add_argument('--delimiter', help='also split using this delimiter')
train_parser.add_argument('--inline-case', action=argparse.BooleanOptionalAction, default=True, help='apply inline '
                          'casing')
train_parser.add_argument('-d', '--dict-path', help='generate a comprehensive dictionary compatible with fairseq and '
                          'Pasero, containing all BPE units and characters')
train_parser.add_argument('--vocab-path', help='generate one dictionary per language containing the frequency of each '
                          'token for that language. The --dict-* options also apply. Unless it includes a {lang}, '
                          'the path is suffixed with the language as a file extension: PATH.LANG')
add_dict_args(train_parser)
train_parser.add_argument('--lang-codes', nargs='*', help='automatically add these language codes to the '
                          'dictionary (if empty, language codes are inferred from the input file names)')
train_parser.add_argument('--temperature', type=float, default=1.0, help='oversample lower-resource languages using '
                          'this temperature parameter (>1: closer to uniform sampling; default: %(default)s)')
train_parser.add_argument('--tokenization', type=int, default=2, help="value between 0 and 4 defining the tokenization "
                          "aggressivity level (default: %(default)s)."
                          "Tokenization is also impacted by --split-by-script\n"
                          "Example with input '@Test1000 :-)':\n"
                          "    0: ['@Test1000', ':-)']\n"
                          "    1: ['@', 'Test1000', ':-)']\n"
                          "    2: ['@', 'Test', '1000', ':-)']\n"
                          "    3: ['@', 'Test', '1000', ':', '-', ')']\n"
                          "    4: ['@', 'Test', '1', '0', '0', '0', ':', '-', ')']\n")
train_parser.add_argument('--protect-regex', help='anything matching this regular expression will be ignored')
train_parser.add_argument('--threads', type=int, help='spawn that many Python processes (only the vocabulary creation '
                          'is parallelizable)')
train_parser.add_argument('--buffer-size', type=int, default=10000, help='process this many lines at once (necessary '
                          'for multi-threading, default: %(default)s)')
train_parser.add_argument('--max-lines', type=int, default=10000000, help='maximum number of lines read per input file '
                          '(default: %(default)s). The real line counts and word counts will be estimated from file '
                          'sizes')


def main_tokenize():
    args = tokenize_parser.parse_args()
    vocab = load_vocab(args.vocabulary, args.threshold) if args.vocabulary else None
    bpe_model = PaseroTokenizer.read(args.bpe_codes, vocab=vocab)
    infile = open(args.input) if args.input else sys.stdin
    outfile = open(args.output, 'w') if args.output else sys.stdout
    try:
        outfile.writelines(
            bpe_model.tokenize(line, unk=args.unk, spell_out=args.spell_out, dropout=args.dropout) + '\n'
            for line in infile
        )
    except (KeyboardInterrupt, BrokenPipeError):
        sys.stdout = None   # ugly trick to avoid broken pipe errors


def main_build_dict():
    args = dict_parser.parse_args()
    infile = open(args.input) if args.input else sys.stdin
    vocab = defaultdict(int)
    if args.max_lines:
        assert args.input
    line_count = 0
    while not args.max_lines or line_count < args.max_lines:
        line = infile.readline()
        if not line:
            break
        if not line.strip():
            continue
        line_count += 1
        for token in line.split():
            vocab[token] += 1
    if args.max_lines:
        read_bytes = infile.tell()
        infile.seek(0, io.SEEK_END)
        total_bytes = infile.tell()
        r = total_bytes / read_bytes
        for k in vocab:
            vocab[k] = int(vocab[k] * r)
    vocab = Counter(vocab)
    build_dict(vocab, **vars(args))


def main_detokenize():
    args = detokenize_parser.parse_args()
    infile = open(args.input) if args.input else sys.stdin
    outfile = open(args.output, 'w') if args.output else sys.stdout
    try:
        outfile.writelines(detokenize(split_tags(line)[-1]) + '\n' for line in infile)
    except (KeyboardInterrupt, BrokenPipeError):
        sys.stdout = None


def main_noisify():
    args = noise_parser.parse_args()
    infile = open(args.input) if args.input else sys.stdin
    outfile = open(args.output, 'w') if args.output else sys.stdout
    noise.seed(args.seed)
    try:
        for line in infile:
            *tags, line = split_tags(line)
            line = noisify(line, **vars(args))
            print(*tags, line, file=outfile)
    except (KeyboardInterrupt, BrokenPipeError):
        sys.stdout = None


def main_train():
    args = train_parser.parse_args()
    args.inputs = args.inputs or [None]
    bpe_model, vocabs = PaseroTokenizer.train(**vars(args))

    args.inputs.sort()
    
    if args.lang_codes is not None:
        if args.lang_codes:
            lang_codes = args.lang_codes
        else:
            assert all(args.inputs)
            lang_codes = [regex.search(r'\.([a-z_-]{2,})$', filename) for filename in args.inputs if filename]
            lang_codes = [m.group(1) for m in lang_codes if m]
        lang_codes = sorted(set(f'<{_LANG_CODE_PREFIX}{lang}>' for lang in lang_codes))
        args.dict_custom_symbols += lang_codes
    
    if args.dict_path is not None:
        vocab = sum(vocabs.values(), Counter())
        build_dict(vocab, **vars(args))

    if args.vocab_path is not None:
        for lang, vocab in vocabs.items():
            if '{lang}' in args.vocab_path:
                vocab_path = args.vocab_path.replace('{lang}', lang)
            else:
                vocab_path = f'{args.vocab_path}.{lang}'
            kwargs = {**vars(args), 'dict_path': vocab_path}
            build_dict(vocabs[lang], **kwargs)

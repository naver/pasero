#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('sentencepiece_model')
parser.add_argument('--input', '-i', help='input file (default: standard input)')
parser.add_argument('--output', '-o', help='output file (default: standard output)')

if __name__ == '__main__':
    args = parser.parse_args()
    infile = open(args.input) if args.input and args.input != '-' else sys.stdin
    outfile = open(args.output, 'w') if args.output else sys.stdout
    model = spm.SentencePieceProcessor(model_file=args.sentencepiece_model)
    try:
        for line in infile:
            pieces = model.EncodeAsPieces(line.strip())
            print(*pieces, file=outfile)
    except (KeyboardInterrupt, BrokenPipeError):
        sys.stdout = None

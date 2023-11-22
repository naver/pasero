#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description="Read tokenized text from standard input and print all the unique tokens, "
    "sorted from most frequent to least frequent")
parser.add_argument('--input', '-i', help="input file (default: standard input)")
parser.add_argument('--output', '-o', help="output file (default: standard output)")

if __name__ == '__main__':
    args = parser.parse_args()
    infile = open(args.input) if args.input and args.input != '-' else sys.stdin
    outfile = open(args.output, 'w') if args.output else sys.stdout
    vocab = Counter(word for line in infile for word in line.split())
    try:
        outfile.writelines(f'{w} {c}\n' for w, c in vocab.most_common())
    except (KeyboardInterrupt, BrokenPipeError):
        sys.stdout = None

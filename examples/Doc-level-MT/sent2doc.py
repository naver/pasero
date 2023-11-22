#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import sys
import random

description="""
This script reads pairs of sentences from standard input (one pair of sentences per line, where source and target
document are separated by tab) and converts them into documents. Document boundaries can be decided in two different
ways:
- an empty line on both sides = end of the document
- a given number of consecutive lines per document
"""

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--tag', default='<sep>', help='sentence delimiter (default: <sep>)')
parser.add_argument('-o', '--output', nargs='+', help='instead of writing to standard output, write to this file')
parser.add_argument('--doc-size', '--max-doc-size', type=int, help='create documents by grouping this many consecutive '
                    'sentences')
parser.add_argument('--min-doc-size', type=int, help='if set, document sizes are uniformly sampled between '
                    'MIN_DOC_SIZE and DOC_SIZE')
parser.add_argument('--seed', type=int, default=42, help='random seed for sampling document sizes')

args = parser.parse_args()

if args.min_doc_size:
    assert args.doc_size
if args.doc_size:
    args.min_doc_size = args.min_doc_size or args.doc_size
else:
    args.min_doc_size = 1

random.seed(args.seed)
tag = ' {} '.format(args.tag.strip())

def write_docs(src_doc, tgt_doc, outfile):
    print(tag.join(src_doc), tag.join(tgt_doc), sep='\t', file=outfile)

def random_doc_dize():
    if args.doc_size and args.min_doc_size < args.doc_size:
        return random.randint(args.min_doc_size, args.doc_size)
    else:
        return args.doc_size or float('inf')

if args.output is None or args.output == '-':
    outfile = sys.stdout
else:
    outfile = open(args.output, 'w')

try:
    src_doc, tgt_doc = [], []
    doc_size = random_doc_dize()
    
    for line in sys.stdin:
        src_sent, tgt_sent = line.split('\t')

        src_sent = src_sent.strip()
        tgt_sent = tgt_sent.strip()

        if not src_sent and not tgt_sent:  # document boundaries in some corpora are set by empty lines on both sides
            if len(src_doc) >= args.min_doc_size:
                write_docs(src_doc, tgt_doc, outfile)
            src_doc, tgt_doc = [], []
            doc_size = random_doc_dize()
            continue

        src_doc.append(src_sent)
        tgt_doc.append(tgt_sent)

        if len(src_doc) == doc_size:
            write_docs(src_doc, tgt_doc, outfile)
            src_doc, tgt_doc = [], []
            doc_size = random_doc_dize()

    if len(src_doc) >= args.min_doc_size:
        write_docs(src_doc, tgt_doc, outfile)
finally:
    if outfile is not sys.stdout:
        outfile.close()

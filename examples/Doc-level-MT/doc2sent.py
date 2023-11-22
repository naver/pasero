#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import sys

description="""
This script reads documents from standard input (one document per line, where sentences are delimited with <sep>), and outputs
one sentence per line.
The '--context' option outputs sentences preceded with their context (a few previous sentences).
"""

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--tag', default='<sep>', help='sentence delimiter (default: <sep>)')
parser.add_argument('--context', type=int, nargs='?', const=0, help='output one sentence per line along with its context. '
                    'Values greater than zero mean that at most that many preceding sentences will be used as context.')
parser.add_argument('--only-context', action='store_true', help='only output the context and not the sentences themselves')

parser.add_argument('-o', '--output', nargs='+', help='instead of writing to standard output, write to this/these file(s). '
                    'If two files are given, sentences and their context are saved in two separate files.')

args = parser.parse_args()

tag = ' {} '.format(args.tag)

assert args.output is None or len(args.output) == 1 or args.context is not None and len(args.output) == 2

if args.output is None:
    outputs = [sys.stdout]
else:
    outputs = [open(filename, 'w') for filename in args.output]

try:
    for line in sys.stdin:
        doc = line.strip('\n').split(args.tag)
        doc = [sent.strip() for sent in doc]

        if args.context is None and not args.only_context:
            for sent in doc:
                print(sent, file=outputs[0])
        else:
            for i, sent in enumerate(doc):
                *context, sent = doc[:i + 1]
                if args.context is not None and args.context > 0:
                    context = context[-args.context:]

                context = tag.join(context)

                if args.only_context:
                    print(context, file=outputs[0])
                elif len(outputs) == 2:
                    print(context, file=outputs[0])
                    print(sent, file=outputs[1])
                else:
                    if context:
                        context += tag

                    print(context + sent)
except (KeyboardInterrupt, BrokenPipeError):
    pass
finally:
    for output in outputs:
        if output is not sys.stdout:
            output.close()

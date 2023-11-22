#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import sys
import json

description="""
This script reads pairs of documents from standard input (one pair of document per line, where source and target
document are separated by tab and sentences within a document by "<sep>") and converts them in a JSON dialogue format 
where the source sentences are attributed to the user and target sentences to the assistant.
"""

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--tag', default='<sep>', help='sentence delimiter (default: <sep>)')
parser.add_argument('-o', '--output', nargs='+', help='instead of writing to standard output, write to this file')
parser.add_argument('--system-prompt', help='start each dialogue with this system prompt')

args = parser.parse_args()

tag = ' {} '.format(args.tag.strip())

if args.output is None or args.output == '-':
    outfile = sys.stdout
else:
    outfile = open(args.output, 'w')

try:
    for line in sys.stdin:
        src_doc, tgt_doc = line.split('\t')
        src_doc = src_doc.strip().split(args.tag)
        tgt_doc = tgt_doc.strip().split(args.tag)

        conversation = []
        if args.system_prompt:
            conversation.append({'role': 'system', 'content': args.system_prompt})
        for src_sent, tgt_sent in zip(src_doc, tgt_doc):
            src_sent = src_sent.strip()
            tgt_sent = tgt_sent.strip()
            conversation.append({'role': 'user', 'content': src_sent})
            conversation.append({'role': 'assistant', 'content': tgt_sent})
        
        print(json.dumps(conversation), file=outfile)
    pass
finally:
    if outfile is not sys.stdout:
        outfile.close()

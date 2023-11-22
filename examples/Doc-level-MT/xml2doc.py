#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import sys
import regex

description = """
This script converts contextual MT files from XML format to an easier-to-process
format with one document per line. It reads from standard input and writes to standard output.

Input format: one sentence per line, documents are delimited by XML tags (e.g., <doc>)
Output format: one document per line, sentences are delimited with a special tag (e.g., <sep>)
"""

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--doc-tag', default='doc', help='XML tag marking the start or end of a document (default: doc)')
parser.add_argument('--tag', default='<sep>', help='output sentence delimiter (default: <sep>)')
parser.add_argument('--skip-xml', action='store_true', help='skip all lines that start and end with XML tags')
parser.add_argument('--seg-tag', action='store_true', help='input text lines are enclosed between <seg> tags')
parser.add_argument('--origlang', nargs='+', help='only keep documents with this \'origlang\' attribute value')
parser.add_argument('--not-origlang', nargs='+', help='exclude documents with this \'origlang\' attribute value')

args = parser.parse_args()

tag = ' {} '.format(args.tag)

document = []
origlang = None

try:
    for line in sys.stdin:
        line = line.replace(args.tag, '')
        line = regex.sub('\s+', ' ', line.strip())
        seg_regex = r'<seg( [^>]*)?>\s*(.*?)\s*(</seg>)$'

        if regex.match(r'</?{}'.format(args.doc_tag), line): # new document
            if document and (not args.origlang or origlang in args.origlang) and (not args.not_origlang or origlang not in args.not_origlang):
                print(tag.join(document))
            m = regex.search(r'origlang="(.*?)"', line)
            if m:
                origlang = m.group(1)
            document = []
            continue

        if args.seg_tag:
            m = regex.match(r'<seg( [^>]*)?>\s*(.*?)\s*(</seg>)$', line)
            if m:
                document.append(m.group(2))
                continue

        if not args.skip_xml or not regex.match(r'<.*>', line):
            document.append(line)
    if document and (not args.origlang or origlang in args.origlang) and (not args.not_origlang or origlang not in args.not_origlang):
        print(tag.join(document))
except (KeyboardInterrupt, BrokenPipeError):
    pass

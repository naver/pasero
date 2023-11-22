#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import os
import logging
from pasero.evaluation import LineReader, FilterByLang, Dedup, FilterByLen, Shuffle

parser = argparse.ArgumentParser(
    description="Filter given corpus by removing lines that are in the wrong language, have mismatched length or "
    "are duplicates; can also shuffle it"
)
parser.add_argument('files', nargs='+', help='parallel input text files (the languages for langid filtering are '
    'inferred from the file extensions')
parser.add_argument('-o', '--output', nargs='+', help='save the filtered corpus into these files')
parser.add_argument('--input-indices', help='use the line ids in this file to pre-filter the input lines')
parser.add_argument('--indices', help='save the line ids of the filtered corpus in this file')
parser.add_argument('--actions', nargs='+', required=True, choices=['clean', 'length', 'langid', 'dedup', 'shuffle'],
    help='perform these actions in this order (warning: shuffle and dedup are memory-hungry)')
parser.add_argument('-v', '--verbose', action='store_true', help='show progress')
parser.add_argument('--continue', dest='continue_', action='store_true', help='continue filtering the corpus if the '
    'output line id file already exists (not compatible with dedup and shuffle actions and with "-o")')
parser.add_argument('--langs', nargs='+', help='NLLB-200 language codes of the input files (e.g., English = eng_Latn)')

args = parser.parse_args()

assert args.output or args.indices
assert not args.output or len(args.output) == len(args.files)
assert not args.continue_ or ('shuffle' not in args.actions and 'dedup' not in args.actions)
assert not args.continue_ or not args.output
assert args.output != args.files
assert args.langs or 'langid' not in args.actions

correct_init = total_init = 0
indices_init = []
if args.continue_ and os.path.isfile(args.indices):
    with open(args.indices) as index_file:
        try:
            indices_init = list(map(int, index_file))[:-1]
            total_init = max(indices_init, default=-1) + 1
            correct_init = len(indices_init)
        except:
            pass

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level='DEBUG' if args.verbose else 'INFO',
)
logger = logging.getLogger('filter-corpus')

filenames = [os.path.basename(path) for path in args.files]
langs = args.langs or [path.split('.')[-1] for path in args.files]
logger.info(args)

lines = LineReader(*args.files, start=total_init)

if args.input_indices:
    input_indices = set(map(int, open(args.input_indices)))
    filtered = ((line_index, line_tuple) for line_index, line_tuple in lines if line_index in input_indices)
else:
    filtered = lines

for action in args.actions:
    if action == 'clean':
        # normalize whitespaces and removes empty lines
        filtered = FilterByLen(filtered, min_len=1)
    elif action == 'length':
        # remove any line pair whose length ratio is too high, or whose length is too short
        filtered = FilterByLen(filtered, langs=langs, char_level=True, length_correction=True, max_ratio=9, min_len=15)
    elif action == 'langid':
        # remove any line pair that is not in the right language
        filtered = FilterByLang(filtered, langs=langs)
    elif action == 'dedup':
        # perform deduplication: remove any line pair where either side was already seen
        filtered = Dedup(filtered, lc=True, monolingual=True)
    elif action == 'shuffle':
        filtered = Shuffle(filtered)
    else:
        raise NotImplementedError

for path in (args.output or []) + [args.indices]:
    if path is not None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

output_files = [open(path, 'w') for path in args.output] if args.output else []

if args.indices:
    index_file = open(args.indices, 'w')
    for line_index in indices_init:
        index_file.write(f'{line_index}\n')
else:
    index_file = None

for i, (line_index, line_tuple) in enumerate(filtered, 1):
    if i % 100000 == 0:
        total = total_init + lines.total
        correct = correct_init + filtered.correct
        logger.debug(f'{total=} {correct=} ({correct/total:.2%})')
        for outfile in output_files:
            outfile.flush()
        if index_file is not None:
            index_file.flush()
    
    if not all(line_tuple):  # filter out empty lines, even without length filtering
        continue

    for line, outfile in zip(line_tuple, output_files):
        outfile.write(line + '\n')
    
    if index_file is not None:
        index_file.write(f'{line_index}\n')

total = total_init + lines.total
correct = correct_init + filtered.correct
logger.info(f'finished: {total=} {correct=} ({correct/total:.2%})')

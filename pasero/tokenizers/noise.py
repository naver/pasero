# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import regex
import string
import numpy as np


mask = '<mask>'
chars = list(string.ascii_letters)


def add_args(parser):
    parser.add_argument('--space-noise', type=float, default=0.0, help='drop or insert whitespaces with this probability')
    parser.add_argument('--punct-noise', type=float, default=0.0, help='drop punctuation symbols with this probability')
    parser.add_argument('--char-noise', type=float, default=0.0, help='apply character-level operations with this probability')
    parser.add_argument('--noise-ops', nargs='+', default=['ins', 'del', 'sub', 'swap'],
                        choices=['ins', 'del', 'sub', 'swap'], help='list of allowed character noise operations '
                        '(insertions, deletions, substitutions or swaps), operations are sampled uniformly from this '
                        'list')
    parser.add_argument('--word-noise', type=float, default=0.0, help='drop entire words with this probability')
    parser.add_argument('--masking', type=float, default=0.0, help='mask entire words with this probability')


def word_split(line):
    tokens = list(filter(None, regex.split("(\W)", line)))
    is_word = [not regex.match('\W', token) for token in tokens]
    return tokens, is_word


def coin_toss(prob=0.5):
    return np.random.random() < prob


def random_char():
    return np.random.choice(chars)


def seed(seed):
    np.random.seed(seed)


def noisify(line, noise_ops=['ins', 'del', 'sub', 'swap'], char_noise=0.1, word_noise=0, space_noise=0, punct_noise=0,
            masking=0, **_):
    
    if word_noise or space_noise or punct_noise:
        tokens, is_word = word_split(line)

        for i in range(len(tokens)):
            if coin_toss(space_noise):
                if tokens[i] == ' ':
                    tokens[i] = ''
                else:
                    tokens[i] = ' ' + tokens[i]
            if not is_word[i] and tokens[i] != ' ' and coin_toss(punct_noise):
                tokens[i] = ''
            if is_word[i] and coin_toss(word_noise):
                tokens[i] = ''

        line = ''.join(tokens)
        line = ' '.join(line.split())   # remove extra whitespaces
    
    if char_noise:
        chars = list(line)
        for i, c in enumerate(chars):
            if c != ' ' and coin_toss(char_noise):
                op = np.random.choice(noise_ops)
                if op == 'ins':
                    chars[i] = random_char() + c
                elif op == 'sub':
                    chars[i] = random_char()
                elif op == 'del':
                    chars[i] = ''
                elif op == 'swap' and i > 0 and chars[i - 1] != ' ':
                    chars[i - 1], chars[i] = chars[i], chars[i - 1]
    
        line = ''.join(chars)
        line = ' '.join(line.split())

    if masking:
        tokens, is_word = word_split(line)
        pos = 0
        while pos < len(tokens):
            if is_word[pos] and coin_toss(masking):
                tokens[pos] = mask
                is_word[pos] = False
            pos += 1
        line = ''.join(tokens)
        mask_ = regex.escape(mask)
        line = regex.sub(f'{mask_}( ?{mask_})*', mask, line)  # replace consecutive masks with a single mask
        line = ' '.join(line.split())
    
    return line

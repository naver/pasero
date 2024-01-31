#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import os
import json
import argparse
from transformers import AutoTokenizer

description = """
Helper script to extract a Pasero dictionary from an existing HuggingFace tokenizer.
Tested with Llama and T5. The dictionary is written to the standard output.

While Pasero does not need this dictionary with `--tokenizer hf` is used, this facilitates vocabulary editing.
Some HuggingFace models may also have a tokenizer whose vocab size that does not match the model's vocab size.
For instance T5's tokenizer has a vocab of size 32000 while T5 has 32128 embeddings. Dummy tokens can be appended to
the Pasero dictionary to fix this mismatch.

Example of usage:
```
scripts/hf-tokenizer-to-dict.py models/flan-t5-large models/flan-t5-large/dict.json
```
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)
parser.add_argument('hf_model_path', help='path to a local copy of a HuggingFace model')
parser.add_argument('dict_path', help='output path for the json dict')
parser.add_argument('--padding-idx', type=int, help='set a custom index for the padding token')
parser.add_argument('--eos-idx', type=int, help='set a custom index for the eos token')
parser.add_argument('--bos-idx', type=int, help='set a custom index for the bos token')

if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isdir(args.hf_model_path)
    config_path = os.path.join(args.hf_model_path, 'config.json')
    assert os.path.isfile(config_path)
    config = json.load(open(config_path))
    vocab_size = config['vocab_size']

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    
    vocab = tokenizer.vocab

    # Sometimes the tokenizer does not have the correct special token ids, try to retrieve them from the model's
    # config file. This happens with T5 for instance, where <pad> and <s> share the same ID. In this case, <s> should
    # be written in the dictionary, otherwise Pasero will use </s> as BOS.
    pad_idx = args.padding_idx
    if pad_idx is None:
        pad_idx = config.get('pad_token_id')

    bos_idx = args.bos_idx
    if bos_idx is None:
        bos_idx = config.get('bos_token_id')
    if bos_idx is None:
        bos_idx = config.get('decoder_start_token_id')  # other possible name for BOS
    
    eos_idx = args.eos_idx
    if eos_idx is None:
        eos_idx = config.get('eos_token_id')
    
    if pad_idx is not None:
        vocab['<pad>'] = pad_idx
    if bos_idx is not None:
        vocab['<s>'] = bos_idx
    if eos_idx is not None:
        vocab['</s>'] = eos_idx
    
    vocab = {w: i for w, i in vocab.items() if i < vocab_size}  # truncate vocab if needed
    size = max(vocab.values())
    i = 0
    while size < vocab_size:  # extend vocab if needed
        dummy = f'madeupword{i:05}'
        if dummy not in vocab:
            vocab[dummy] = size
            size += 1
        i += 1

    with open(args.dict_path, 'w') as dict_file:
        json.dump(vocab, dict_file)

#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import torch
from pasero.models.transformer import Transformer
from pasero.utils import find_checkpoint_shards, load_checkpoint

description = """
Helper script to merge model checkpoints trained with Tensor Parallelism into a single checkpoint that can be used
for inference with `pasero-decode`. Note that this is not needed to finetune a model without Tensor Parallelism or
a different TP size, as `Trainer` can automatically reshard the checkpoints.

Example of usage:
```
scripts/merge-tp-ckpt.py models/ParaCrawl/fr-en.tp-4/model_best.bin -o models/ParaCrawl/fr-en.tp-4/model_merged.bin
pasero-decode models/ParaCrawl/fr-en.tp-4/model_merged.bin [DECODING_OPTIONS]
```
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)
parser.add_argument('input_ckpt', help='path to the first shard of the checkpoint to merge')
parser.add_argument('-o', '--output-ckpt', required=True, help='output path for the merged checkpoint')

if __name__ == '__main__':
    args = parser.parse_args()
    ckpt_paths = find_checkpoint_shards(args.input_ckpt)
    checkpoints = [load_checkpoint(path) for path in ckpt_paths]
    model_states = [ckpt['model'] for ckpt in checkpoints]
    model_state = Transformer.unshard_state_dict(*model_states, total_shard_count=len(checkpoints))
    ckpt = checkpoints[0]
    ckpt['model'] = model_state
    ckpt['args']['tp_size'] = 1
    torch.save(ckpt, args.output_ckpt)

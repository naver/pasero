#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import torch
from pasero.models.mixture_of_experts import MOETransformer
from pasero.utils import find_checkpoint_shards, load_checkpoint

description = """
Helper script to merge Mixture-of-Experts checkpoints trained with Tutel into a single MoE checkpoint that can be used
for inference with `pasero-decode`. Note that this is not needed to finetune a Tutel model with a different number or 
GPUs or a different MoE impl (e.g., "basic" or "fused"), as `Trainer` can automatically convert and reshard the 
checkpoints. Decoding can also be done directly from Tutel MoE models if the DP size at inference is the same as at
training.

Example of usage:
```
scripts/merge-tutel-ckpt.py models/ParaCrawl/fr-en.moe-tutel/model_best.bin -o models/ParaCrawl/fr-en.moe-tutel/model_merged.bin
pasero-decode models/ParaCrawl/fr-en.moe-tutel/model_merged.bin [DECODING_OPTIONS]
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
    model_state = MOETransformer.unshard_state_dict(*model_states, total_shard_count=len(checkpoints))
    ckpt = checkpoints[0]
    ckpt['model'] = model_state
    ckpt['args']['moe_impl'] = 'basic'  # simpler MoE format that runs on a single GPU
    torch.save(ckpt, args.output_ckpt)

#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import argparse
import torch
import os
import sys
from pasero.models.transformer import Transformer
from pasero.utils import find_checkpoint_shards, load_checkpoint

description = r"""
Helper script to merge model checkpoints trained with Tensor Parallelism into a single checkpoint that can be used
for inference with `pasero-decode` or `pasero-serve`. Note that this is not needed to finetune a model without Tensor 
Parallelism or a different TP size, as `Trainer` can automatically reshard the checkpoints.

Example of usage:
```
# Merge checkpoints:
scripts/merge-tp-ckpt.py models/ParaCrawl/fr-en.tp-4/model_best.bin -o models/ParaCrawl/fr-en.tp-4/model_merged.bin
# Decode with the merged checkpoint:
pasero-decode models/ParaCrawl/fr-en.tp-4/model_merged.bin [DECODING_OPTIONS...]
```

If the input checkpoint is symbolic link, the "-o" option can be omitted and the symbolic link will be replaced 
by the merged checkpoint. For example:
```
# Merge checkpoints:
scripts/merge-tp-ckpt.py models/ParaCrawl/fr-en.tp-4/model_best.bin
# or
scripts/merge-tp-ckpt.py models/ParaCrawl/fr-en.tp-4  # looks for 'model_best.bin' or 'model_last.bin' symlinks
# Decode with the merged checkpoint (loads 'model_best.bin' in priority):
pasero-decode models/ParaCrawl/fr-en.tp-4 [DECODING_OPTIONS...]
```
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)
parser.add_argument('dir_or_ckpt', help='path to the first shard of the checkpoint to merge')
parser.add_argument('-o', '--output-ckpt', help='output path for the merged checkpoint')

if __name__ == '__main__':
    args = parser.parse_args()
    
    if os.path.isdir(args.dir_or_ckpt):
        input_ckpt = None
        for filename in 'model_best.bin', 'model_last.bin':
            path = os.path.join(args.dir_or_ckpt, filename)
            if os.path.exists(path):
                input_ckpt = path
                break
        assert input_ckpt is not None, f"did not find a valid checkpoint in '{args.dir_or_ckpt}'"
    elif os.path.isfile(args.dir_or_ckpt):
        input_ckpt = args.dir_or_ckpt
    else:
        raise ValueError(f"'{args.input_ckpt}' does not exist")

    if args.output_ckpt is None and os.path.islink(input_ckpt):  # if no -o is given and the input is a symlink 
        # (e.g., "model_last.bin"), overwrite the symlink
        args.output_ckpt = input_ckpt
    
    assert args.output_ckpt, 'missing -o/--output-ckpt option'

    ckpt_paths = find_checkpoint_shards(input_ckpt)
    filenames = [os.path.basename(path) for path in ckpt_paths]
    print(filenames, '-->', repr(os.path.basename(args.output_ckpt)), file=sys.stderr)
    checkpoints = [load_checkpoint(path) for path in ckpt_paths]
    model_states = [ckpt['model'] for ckpt in checkpoints]
    model_state = Transformer.unshard_state_dict(*model_states, total_shard_count=len(checkpoints))
    ckpt = checkpoints[0]
    ckpt['model'] = model_state
    ckpt['args']['tp_size'] = 1

    if os.path.islink(args.output_ckpt):
        os.unlink(args.output_ckpt)  # delete symlinks because torch would save a checkpoint at the destination 
        # of the symlink instead of overwriting the symlink instead
    torch.save(ckpt, args.output_ckpt)

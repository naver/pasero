#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import torch
import argparse

parser = argparse.ArgumentParser(description="Take several model checkpoints, average their weights and save the "
    "resulting checkpoint at the specified location")
parser.add_argument('checkpoints', nargs='+', help='paths to the checkpoints to average')
parser.add_argument('-o', '--output', required=True, help='output path for the averaged checkpoint')

args = parser.parse_args()

ckpt = torch.load(args.checkpoints[0], map_location='cpu')

model = {k: [v] for k, v in ckpt['model'].items()}
for ckpt_file in args.checkpoints[1:]:
    model_ = torch.load(ckpt_file, map_location='cpu')['model']
    for k, v in model_.items():
        model[k].append(v)

ckpt['model'] = {k: sum(v) / len(v) for k, v in model.items()}
torch.save(ckpt, args.output)

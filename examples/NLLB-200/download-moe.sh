#!/usr/bin/env bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

# Download and conversion script for the NLLB Mixture-of-Experts model

mkdir -p models/NLLB-200

pushd models/NLLB-200

# download and extract the 54B MoE model (can take some time)
wget --trust-server-names https://tinyurl.com/nllb200moe54bmodel
tar xzf model.tar.gz
mv model_moe_54b/checkpoint_2_300000-shared.pt 54B_moe.bin  # "dense" part of the model

# drastically reduce the size of the checkpoint by remove unnecessary data from it (e.g., optimizer states)
python3 -c "
import torch
ckpt = torch.load('54B_moe.bin')
ckpt.pop('last_optimizer_state')  # remove optimizer states which are very large
# replace duplicated shared parameters by references
ckpt['model']['decoder.embed_tokens.weight'] = ckpt['model']['encoder.embed_tokens.weight']
ckpt['model']['decoder.output_projection.weight'] = ckpt['model']['encoder.embed_tokens.weight']
torch.save(ckpt, '54B_moe.bin')
"

mkdir experts
# create one checkpoint per expert: makes it easier to prune the NLLB-model
for expert_rank in {0..127}; do
path=model_moe_54b/checkpoint_2_300000-rank-${expert_rank}.pt
python3 -c "
import torch
params = torch.load('$path')['model']
for module in 'encoder', 'decoder':
    for layer_id in 3, 7, 11, 15, 19, 23:
        path = f'experts/{module}-{layer_id}-expert-$expert_rank.bin'
        prefix = f'{module}.layers.{layer_id}.'
        expert = {k: v for k, v in params.items() if k.startswith(prefix)}
        torch.save(expert, path)
"
rm $path
done
rmdir model_moe_54b

# retrieve the expert ids for lang-specific pruning
wget https://raw.githubusercontent.com/naver/nllb-pruning/main/experts.json
cp examples/NLLB-200/inference.yaml .

popd

# download the NLLB tokenizer files
examples/NLLB-200/download-dict.sh

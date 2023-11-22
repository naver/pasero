#!/usr/bin/env bash
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

# Download and conversion script for the NLLB Dense models

mkdir -p models/NLLB-200

pushd models/NLLB-200

# download the dense NLLB models
wget --trust-server-names https://tinyurl.com/nllb200dense3bcheckpoint -O 3.3B_dense.bin
wget --trust-server-names https://tinyurl.com/nllb200densedst1bcheckpoint -O 1.3B_distilled.bin
wget --trust-server-names https://tinyurl.com/nllb200densedst600mcheckpoint -O 600M_distilled.bin

# drastically reduce the size of the checkpoints by remove unnecessary data from them (e.g., optimizer states)
for path in 3.3B_dense.bin 1.3B_distilled.bin 600M_distilled.bin; do
python3 -c "
import torch
ckpt = torch.load('$path')
ckpt.pop('last_optimizer_state')  # remove optimizer states which are very large
# replace duplicated shared parameters by references
ckpt['model']['decoder.embed_tokens.weight'] = ckpt['model']['encoder.embed_tokens.weight']
ckpt['model']['decoder.output_projection.weight'] = ckpt['model']['encoder.embed_tokens.weight']
torch.save(ckpt, '$path')
"
done

popd

# download the NLLB tokenizer files
examples/NLLB-200/download-dict.sh

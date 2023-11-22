#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import torch
import argparse
import regex
from pasero.models.transformer import Transformer

bloom_mapping = {
    'h.0.input_layernorm.weight': ['decoder.layers.0.self_attn_layer_norm.weight'],
    'h.0.input_layernorm.bias': ['decoder.layers.0.self_attn_layer_norm.bias'],
    'h.0.post_attention_layernorm.weight': ['decoder.layers.0.final_layer_norm.weight'],
    'h.0.post_attention_layernorm.bias': ['decoder.layers.0.final_layer_norm.bias'],
    'ln_f.weight': ['decoder.layer_norm.weight'],
    'ln_f.bias': ['decoder.layer_norm.bias'],
    'word_embeddings.weight': ['decoder.embed_tokens.weight'],
    'word_embeddings_layernorm.weight': ['decoder.layernorm_embedding.weight'],
    'word_embeddings_layernorm.bias': ['decoder.layernorm_embedding.bias'],
    'h.0.self_attention.query_key_value.weight': [
        'decoder.layers.0.self_attn.q_proj.weight',
        'decoder.layers.0.self_attn.k_proj.weight',
        'decoder.layers.0.self_attn.v_proj.weight',
    ],
    'h.0.self_attention.query_key_value.bias': [
        'decoder.layers.0.self_attn.q_proj.bias',
        'decoder.layers.0.self_attn.k_proj.bias',
        'decoder.layers.0.self_attn.v_proj.bias',
    ],
    'h.0.self_attention.dense.weight': ['decoder.layers.0.self_attn.out_proj.weight'],
    'h.0.self_attention.dense.bias': ['decoder.layers.0.self_attn.out_proj.bias'],
    'h.0.mlp.dense_h_to_4h.weight': ['decoder.layers.0.fc1.weight'],
    'h.0.mlp.dense_h_to_4h.bias': ['decoder.layers.0.fc1.bias'],
    'h.0.mlp.dense_4h_to_h.weight': ['decoder.layers.0.fc2.weight'],
    'h.0.mlp.dense_4h_to_h.bias': ['decoder.layers.0.fc2.bias'],
}

llama_official_mapping = {
    'norm.weight': ['decoder.layer_norm.weight'],
    'tok_embeddings.weight': ['decoder.embed_tokens.weight'],
    'output.weight': ['decoder.output_projection.weight'],
    'layers.0.attention_norm.weight': ['decoder.layers.0.self_attn_layer_norm.weight'],
    'layers.0.ffn_norm.weight': ['decoder.layers.0.final_layer_norm.weight'],
    'layers.0.attention.wq.weight': ['decoder.layers.0.self_attn.q_proj.weight'],
    'layers.0.attention.wk.weight': ['decoder.layers.0.self_attn.k_proj.weight'],
    'layers.0.attention.wv.weight': ['decoder.layers.0.self_attn.v_proj.weight'],
    'layers.0.attention.wo.weight': ['decoder.layers.0.self_attn.out_proj.weight'],
    'layers.0.attention.inner_attention.rope.freqs': [],
    'layers.0.feed_forward.w1.weight': ['decoder.layers.0.fc1.weight'],
    'layers.0.feed_forward.w2.weight': ['decoder.layers.0.fc2.weight'],
    'layers.0.feed_forward.w3.weight': ['decoder.layers.0.fc3.weight'],
    'rope.freqs': [],
}

llama_mapping = {
    'model.embed_tokens.weight': ['decoder.embed_tokens.weight'],
    'model.norm.weight': ['decoder.layer_norm.weight'],
    'lm_head.weight': ['decoder.output_projection.weight'],
    'model.layers.0.input_layernorm.weight': ['decoder.layers.0.self_attn_layer_norm.weight'],
    'model.layers.0.post_attention_layernorm.weight': ['decoder.layers.0.final_layer_norm.weight'],
    'model.layers.0.self_attn.q_proj.weight': ['decoder.layers.0.self_attn.q_proj.weight'],
    'model.layers.0.self_attn.k_proj.weight': ['decoder.layers.0.self_attn.k_proj.weight'],
    'model.layers.0.self_attn.v_proj.weight': ['decoder.layers.0.self_attn.v_proj.weight'],
    'model.layers.0.self_attn.o_proj.weight': ['decoder.layers.0.self_attn.out_proj.weight'],
    'model.layers.0.self_attn.rotary_emb.inv_freq': [],
    'model.layers.0.mlp.gate_proj.weight': ['decoder.layers.0.fc1.weight'],
    'model.layers.0.mlp.down_proj.weight': ['decoder.layers.0.fc2.weight'],
    'model.layers.0.mlp.up_proj.weight': ['decoder.layers.0.fc3.weight'],
}

mpt_mapping = {
    'transformer.wte.weight': ['decoder.embed_tokens.weight'],
    'transformer.blocks.0.ffn.up_proj.weight': ['decoder.layers.0.fc1.weight'],
    'transformer.blocks.0.ffn.down_proj.weight': ['decoder.layers.0.fc2.weight'],
    'transformer.blocks.0.attn.Wqkv.weight': [
        'decoder.layers.0.self_attn.q_proj.weight',
        'decoder.layers.0.self_attn.k_proj.weight',
        'decoder.layers.0.self_attn.v_proj.weight',
    ],
    'transformer.blocks.0.attn.out_proj.weight': ['decoder.layers.0.self_attn.out_proj.weight'],
    'transformer.blocks.0.norm_1.weight': ['decoder.layers.0.self_attn_layer_norm.weight'],
    'transformer.blocks.0.norm_2.weight': ['decoder.layers.0.final_layer_norm.weight'],
    'transformer.norm_f.weight': ['decoder.layer_norm.weight'],
}

falcon_7b_mapping = {
    'transformer.word_embeddings.weight': ['decoder.embed_tokens.weight'],
    'lm_head.weight': [],
    'transformer.h.0.input_layernorm.weight': ['decoder.layers.0.self_attn_layer_norm.weight'],
    'transformer.h.0.input_layernorm.bias': ['decoder.layers.0.self_attn_layer_norm.bias'],
    'transformer.ln_f.weight': ['decoder.layer_norm.weight'],
    'transformer.ln_f.bias': ['decoder.layer_norm.bias'],
    'transformer.h.0.mlp.dense_h_to_4h.weight': ['decoder.layers.0.fc1.weight'],
    'transformer.h.0.mlp.dense_4h_to_h.weight': ['decoder.layers.0.fc2.weight'],
    'transformer.h.0.self_attention.query_key_value.weight': [
        'decoder.layers.0.self_attn.q_proj.weight',
        'decoder.layers.0.self_attn.k_proj.weight',
        'decoder.layers.0.self_attn.v_proj.weight',
    ],
    'transformer.h.0.self_attention.dense.weight': ['decoder.layers.0.self_attn.out_proj.weight'],
}

falcon_40b_mapping = {
    'transformer.word_embeddings.weight': ['decoder.embed_tokens.weight'],
    'lm_head.weight': [],
    'transformer.h.0.ln_attn.weight': ['decoder.layers.0.self_attn_layer_norm.weight'],
    'transformer.h.0.ln_attn.bias': ['decoder.layers.0.self_attn_layer_norm.bias'],
    'transformer.h.0.ln_mlp.weight': ['decoder.layers.0.final_layer_norm.weight'],
    'transformer.h.0.ln_mlp.bias': ['decoder.layers.0.final_layer_norm.bias'],
    'transformer.ln_f.weight': ['decoder.layer_norm.weight'],
    'transformer.ln_f.bias': ['decoder.layer_norm.bias'],
    'transformer.h.0.mlp.dense_h_to_4h.weight': ['decoder.layers.0.fc1.weight'],
    'transformer.h.0.mlp.dense_4h_to_h.weight': ['decoder.layers.0.fc2.weight'],
    'transformer.h.0.self_attention.query_key_value.weight': [
        'decoder.layers.0.self_attn.q_proj.weight',
        'decoder.layers.0.self_attn.k_proj.weight',
        'decoder.layers.0.self_attn.v_proj.weight',
    ],
    'transformer.h.0.self_attention.dense.weight': ['decoder.layers.0.self_attn.out_proj.weight'],
}

t5_mapping = {
    'encoder.embed_tokens.weight': ['encoder.embed_tokens.weight'],
    'decoder.embed_tokens.weight': [],
    'shared.weight': [],
    'lm_head.weight': ['decoder.output_projection.weight'],
    'encoder.final_layer_norm.weight': ['encoder.layer_norm.weight'],
    'decoder.final_layer_norm.weight': ['decoder.layer_norm.weight'],
    'encoder.block.0.layer.0.SelfAttention.q.weight': ['encoder.layers.0.self_attn.q_proj.weight'],
    'encoder.block.0.layer.0.SelfAttention.k.weight': ['encoder.layers.0.self_attn.k_proj.weight'],
    'encoder.block.0.layer.0.SelfAttention.v.weight': ['encoder.layers.0.self_attn.v_proj.weight'],
    'encoder.block.0.layer.0.SelfAttention.o.weight': ['encoder.layers.0.self_attn.out_proj.weight'],
    'encoder.block.0.layer.1.DenseReluDense.wi_0.weight': ['encoder.layers.0.fc1.weight'],
    'encoder.block.0.layer.1.DenseReluDense.wi_1.weight': ['encoder.layers.0.fc3.weight'],
    'encoder.block.0.layer.1.DenseReluDense.wo.weight': ['encoder.layers.0.fc2.weight'],
    'encoder.block.0.layer.0.layer_norm.weight': ['encoder.layers.0.self_attn_layer_norm.weight'],
    'encoder.block.0.layer.1.layer_norm.weight': ['encoder.layers.0.final_layer_norm.weight'],
    'decoder.block.0.layer.0.SelfAttention.q.weight': ['decoder.layers.0.self_attn.q_proj.weight'],
    'decoder.block.0.layer.0.SelfAttention.k.weight': ['decoder.layers.0.self_attn.k_proj.weight'],
    'decoder.block.0.layer.0.SelfAttention.v.weight': ['decoder.layers.0.self_attn.v_proj.weight'],
    'decoder.block.0.layer.0.SelfAttention.o.weight': ['decoder.layers.0.self_attn.out_proj.weight'],
    'decoder.block.0.layer.1.EncDecAttention.q.weight': ['decoder.layers.0.encoder_attn.q_proj.weight'],
    'decoder.block.0.layer.1.EncDecAttention.k.weight': ['decoder.layers.0.encoder_attn.k_proj.weight'],
    'decoder.block.0.layer.1.EncDecAttention.v.weight': ['decoder.layers.0.encoder_attn.v_proj.weight'],
    'decoder.block.0.layer.1.EncDecAttention.o.weight': ['decoder.layers.0.encoder_attn.out_proj.weight'],
    'decoder.block.0.layer.2.DenseReluDense.wi_0.weight': ['decoder.layers.0.fc1.weight'],
    'decoder.block.0.layer.2.DenseReluDense.wi_1.weight': ['decoder.layers.0.fc3.weight'],
    'decoder.block.0.layer.2.DenseReluDense.wo.weight': ['decoder.layers.0.fc2.weight'],
    'decoder.block.0.layer.0.layer_norm.weight': ['decoder.layers.0.self_attn_layer_norm.weight'],
    'decoder.block.0.layer.1.layer_norm.weight': ['decoder.layers.0.encoder_attn_layer_norm.weight'],
    'decoder.block.0.layer.2.layer_norm.weight': ['decoder.layers.0.final_layer_norm.weight'],
    'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight': ['encoder.layers.0.self_attn.t5_embed.relative_attention_bias.weight'],
    'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight': ['decoder.layers.0.self_attn.t5_embed.relative_attention_bias.weight'],
}


mappings = {
    'bloom': bloom_mapping,
    'llama_official': llama_official_mapping,
    'llama': llama_mapping,
    'mistral': llama_mapping,
    'mpt': mpt_mapping,
    'falcon_7b': falcon_7b_mapping,
    'falcon_40b': falcon_40b_mapping,
    't5': t5_mapping,
}


description = """
Helper script to convert HuggingFace checkpoints into the Pasero format. A Pasero-style dictionary will also have to 
be extracted using `scripts/hf-tokenizer-to-dict.py`. The original llama checkpoints (different from the HuggingFace 
ones) are also supported with `--arch llama_official`.

The right hyper-parameters will have to be provided by command-line when decoding with `pasero-decode` (--tokenizer, 
--tokenizer-path, --task, --arch), or existing 'inference.yaml' (available in the respective "examples/" folders) can be 
copied to the model directory.

Example of usage:
```
scripts/convert-hf-ckpt.py models/llama-2-7b/pytorch_model*.bin -o models/llama-2-7b/model_best.bin --arch llama --dtype float16
```
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=description)
parser.add_argument('input_ckpt', nargs='+', help='paths to the HuggingFace checkpoints to convert '
                    '(can be several shards)')
parser.add_argument('-o', '--output-ckpt', required=True, help='output path for the new Pasero-style checkpoint')
parser.add_argument('--heads', type=int, help="number of attention heads in this model "
                    "(required by the 'bloom' and 'llama_official' architectures)")
parser.add_argument('--arch', choices=list(mappings), required=True, help='which architecture this model belongs to')
parser.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16', 'float32'],
                    help='the data type of the output checkpoint: float16 and bfloat16 are twice as compact as float32')
args = parser.parse_args()

if args.arch in ('llama_official', 'bloom'):
    assert args.heads

dtype = getattr(torch, args.dtype)

models = []

encoder_layers = 0
decoder_layers = 0

for path in args.input_ckpt:
    print(f"loading '{path}")
    ckpt = torch.load(path, map_location='cpu')

    if args.arch == 'whisper':
        # special case for whisper, whose parameter names almost exactly match ours
        model = {
            (
                k.removeprefix('model.')
                .replace('encoder.conv1.', 'encoder.subsample.conv_layers.0.')
                .replace('encoder.conv2.', 'encoder.subsample.conv_layers.1.')
            ): v
            for k, v in ckpt.items()
        }
        models.append(model)
        continue
    
    patterns = [
        (regex.escape(k).replace(r'\.0\.', r'\.(?P<layer_id>\d+)\.', 1), v)
        for k, v in mappings[args.arch].items()
    ]

    model = {}

    for name, value in ckpt.items():
        shape = value.shape
        found = False
        for pattern, new_names in patterns:
            if (m := regex.fullmatch(pattern, name)):
                layer_id = m.groupdict().get('layer_id')
                layer_id = -1 if layer_id is None else int(layer_id)

                if len(new_names) == 0:
                    print(f'{name} ->')
                elif new_names[0].startswith('encoder.'):
                    encoder_layers = max(encoder_layers, layer_id + 1)
                elif new_names[0].startswith('decoder.'):
                    decoder_layers = max(decoder_layers, layer_id + 1)

                for i, new_name in enumerate(new_names):
                    
                    new_name = new_name.replace('.0.', f'.{layer_id}.', 1)
                    new_dim = shape[0] // len(new_names)

                    if len(new_names) == 1:
                        value_ = value
                        if args.arch == 'llama_official' and (name.endswith('.wq.weight') or name.endswith('.wk.weight')):
                            value_ = value_.reshape(args.heads//len(args.input_ckpt), -1, 2, shape[-1])
                            value_ = value_.transpose(1, 2).reshape(-1, shape[-1])
                    # fused qkv attention layers
                    elif args.arch == 'bloom':  # reshape the attention matrix to avoid having to implement a specific
                        # attention computation in Pasero
                        assert len(new_names) == 3
                        dim = shape[0]
                        value_ = value.reshape(args.heads, 3, -1)[:,i].reshape(dim, -1).squeeze(1)
                    elif args.arch == 'falcon_40b':
                        assert len(new_names) == 3
                        value_ = value.view(8, -1, 64, 8192)  # num_kv x (num_heads/num_kv * 2) x head_dim x embed_dim
                        # multi-query attention has smaller weights for k and v
                        if i == 0:
                            value_ = value_[:,:-2]  # num_kv x num_heads/num_kv x head_dim x embed_dim
                        elif i == 1:
                            value_ = value_[:,-2]   # num_kv x head_dim x embed_dim
                        elif i == 2:
                            value_ = value_[:,-1]   # num_kv x head_dim x embed_dim
                        value_ = value_.reshape(-1, 8192)
                    elif args.arch == 'falcon_7b' or args.arch == 'mpt':
                        assert len(new_names) == 3
                        q_dim = shape[1]
                        k_dim = v_dim = (shape[0] - q_dim) // 2
                        dims = [q_dim, k_dim, v_dim]
                        dim = dims[i]
                        value_ = value[:dim]
                        value = value[dim:]
                    else:
                        raise Exception
                    
                    shape_ = 'x'.join(map(str, value_.shape))
                    if len(new_names) == 1:
                        print(f'{name} -> {new_name} ({shape_})')
                    else:
                        print(f'{name}[{i}] -> {new_name} ({shape_})')

                    model[new_name] = value_.to(dtype)
                found = True
                break

        if not found:
            raise Exception(f"'{name}' not found in mapping")
    
    models.append(model)


# In our implementation of T5 positional encoding, each attention layers has an attention bias parameter (whose 
# values are tied across layers)
if args.arch == 't5':
    encoder_t5_embed_key = 'encoder.layers.0.self_attn.t5_embed.relative_attention_bias.weight'
    if encoder_t5_embed_key in model:
        for layer_id in range(1, encoder_layers):
            new_name = encoder_t5_embed_key.replace('encoder.layers.0.', f'encoder.layers.{layer_id}.')
            model[new_name] = model[encoder_t5_embed_key]
    
    decoder_t5_embed_key = 'decoder.layers.0.self_attn.t5_embed.relative_attention_bias.weight'
    if decoder_t5_embed_key in model:
        for layer_id in range(1, encoder_layers):
            new_name = decoder_t5_embed_key.replace('decoder.layers.0.', f'decoder.layers.{layer_id}.')
            model[new_name] = model[decoder_t5_embed_key]


if args.arch == 'llama_official':
    model = Transformer._unshard_state_dict(*models)
    model['decoder.embed_tokens.weight'] = torch.cat([model_['decoder.embed_tokens.weight'] for model_ in models], dim=1)
    model['decoder.output_projection.weight'] = torch.cat([model_['decoder.output_projection.weight'] for model_ in models], dim=0)
else:
    # HuggingFace model shards are not for tensor parallelism but for pipeline parallelism: each checkpoint contains 
    # a different set of entire weights
    model = {k: v for model_ in models for k, v in model_.items()}

del models

for k, v in model.items():
    model[k] = v.clone()
print(f"writing to '{args.output_ckpt}'")
torch.save({'model': model}, args.output_ckpt)

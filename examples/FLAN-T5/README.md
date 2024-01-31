# FLAN-T5 & co

## Download and convert the model

```bash
huggingface-cli download google/flan-t5-large --local-dir models/flan-t5-large --local-dir-use-symlinks False \
--exclude "pytorch_model.bin" "flax_model.msgpack" "tf_model.h5"

# Convert the HuggingFace checkpoint to the Pasero format:
scripts/convert-hf-ckpt.py models/flan-t5-large/model.safetensors -o models/flan-t5-large/model_bfloat16.bin --arch t5 --dtype bfloat16

# Create a Pasero dictionary of the right size by padding the HuggingFace tokenizer's vocab with dummy tokens:
scripts/hf-tokenizer-to-dict.py models/flan-t5-large models/flan-t5-large/dict.json
```

This example is for FLAN-T5, but can be applied to mT5 as well, which shares the same architecture.

## Finetune it for French-English machine translation

```bash
pasero-train -c examples/ParaCrawl/training.yaml -s fr -t en \
--ckpt models/flan-t5-large/model_bfloat16.bin --arch t5_large --bos-idx 1 --dtype bfloat16 \
--tokenizer hf --tokenizer-path models/flan-t5-large --dict models/flan-t5-large/dict.json \
--batch-size 4096 --virtual-dp-size 64 --label-smoothing 0 \
-o models/ParaCrawl/fr-en/flan-t5-large
```

# FLAN-T5 & co

## Download and convert the model

```bash
pushd models
git lfs install
git clone https://huggingface.co/google/flan-t5-large
popd

# Convert the HuggingFace checkpoints to Pasero format
scripts/convert-hf-ckpt.py models/flan-t5-large/pytorch_model.bin -o models/flan-t5-large/model_bfloat16.bin --arch t5 --dtype bfloat16

# Convert the HuggingFace tokenizer to Pasero dict
scripts/hf-tokenizer-to-dict.py models/flan-t5-large > models/flan-t5-large/dict.txt
```

This example is for FLAN-T5, but can be applied to mT5 as well, which shares the same architecture.

## Finetune it for French-English machine translation

```bash
pasero-train -c examples/ParaCrawl/training.yaml -s fr -t en \
--ckpt models/flan-t5-large/model_bfloat16.bin --arch t5_large --dtype bfloat16 \
--dict models/flan-t5-large/dict.txt --tokenizer sentencepiece --tokenizer-path models/flan-t5-large/spiece.model \
--batch-size 4096 --virtual-dp-size 64 --label-smoothing 0 \
-o models/ParaCrawl/fr-en/flan-t5-large
```

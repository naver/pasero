## Zero-shot cross-lingual transfer in instruction tuning of LLMs

Paper: https://arxiv.org/abs/2402.14778

We show that instruction tuning on English data can successfully transfer to other languages, but only if multilinguality is taken into account at hyperparameter tuning. We also analyze strong and weak sides of predictions made in zero-shot transfer regime, and advocate for more fine-grained evaluation in multilingual generation.  

### Training

#### Step 1: Download and convert Llama 2 7B /  Llama 2 13B / Tower-7B (`llama-2-7b` as an examples)

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models/llama-2-7b --local-dir-use-symlinks False --exclude "pytorch_model*" --token ACCESS_TOKEN  # https://huggingface.co/settings/tokens

# Convert the HuggingFace checkpoint to the Pasero format
scripts/convert-hf-ckpt.py models/llama-2-7b/model-*.safetensors -o models/llama-2-7b/model_best.bin --arch llama --dtype float16

# Create an "inference.yaml" file for pasero-decode and pasero-serve
echo "tokenizer: hf
sampling: False
repeat_penalty: 1.2
arch: llama_7b
task: dialogue" > models/llama-2-7b/inference.yaml
```

#### Step 2: Tune model

Config files are in `config`. Example with config  `config_llama7b2_dolly_en_ft.yaml`:

```
pasero-train -c config_llama7b2_dolly_en_ft.yaml -o llama7b2_dolly_en_ft
```

General instructions on setting uo and training with `pasero` are available in main Pasero readme: https://github.com/naver/pasero/blob/main/README.md

#### Step 3: Merge checkpoints

Tensor parallelism creates sharded checkpoints, which can be merged like follows:
```bash
scripts/merge-tp-ckpt.py models/llama7b2_dolly_en_ft
```

### Evaluation
Evaluation code will be added soon. Evaluation data is available in `data` folder, and evaluation prompt and instructions are available in `evaluation` folder.

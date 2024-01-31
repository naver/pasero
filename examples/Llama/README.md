# Llama 2

This README gives steps to use Llama 2 models with Pasero and finetune them for machine translation or dialogue.
Note that the same examples can be used with Mistral 7B.

## Download and convert Llama 2 13B

```bash
huggingface-cli download meta-llama/Llama-2-13b-hf --local-dir models/llama-2-13b --local-dir-use-symlinks False --exclude "pytorch_model*" --token ACCESS_TOKEN  # https://huggingface.co/settings/tokens

# Convert the HuggingFace checkpoint to the Pasero format
scripts/convert-hf-ckpt.py models/llama-2-13b/model-*.safetensors -o models/llama-2-13b/model_best.bin --arch llama --dtype float16

# Create an "inference.yaml" file for pasero-decode and pasero-serve
echo "tokenizer: hf
sampling: True
sampling_topp: 0.92
arch: llama_13b
task: language_modeling" > models/llama-2-13b/inference.yaml
```

## Download and convert Mistral 7B

```bash
mkdir -p models
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir models/mistral-7b --local-dir-use-symlinks False --exclude "pytorch_model*"

# Convert the HuggingFace checkpoint to the Pasero format
scripts/convert-hf-ckpt.py models/mistral-7b/model-*.safetensors -o models/mistral-7b/model_bfloat16.bin --arch mistral --dtype bfloat16

# Create an "inference.yaml" file for pasero-decode and pasero-serve
echo "tokenizer: hf
ckpt: model_bfloat16.bin
dtype: bfloat16
sampling: True
sampling_topp: 0.92
arch: mistral_7b
task: language_modeling
batch_size: 32768" > models/mistral-7b/inference.yaml
```

## Finetune Llama on French-English machine translation

This example finetunes LLama 2 13B on French-English machine translation using the `translation` task: the sources and targets are concatenated into a single decoder input and the model is trained (with full finetuning or LoRA) on next word prediction.

Other examples in [`examples/Doc-level-MT`](/examples/Doc-level-MT/README.md) show how Llama-style models can be finetuned on *document-level*
machine translation, using either the `doc_level_translation` task or `dialogue` task.

```bash
examples/ParaCrawl/download.sh fr  # download the ParaCrawl Fr-En data
examples/download-flores.sh  # download the FLORES valid and test sets

MODEL_DIR=models/llama-2-13b

OPTS=(
    -s fr -t en --data-dir data/ParaCrawl  # train on ParaCrawl Fr-En
    --train-corpora "ParaCrawl.{pair}" --valid-corpora "data/FLORES/FLORES-valid"
    --ckpt ${MODEL_DIR}/model_best.bin --arch llama_13b  # initialize with Llama-2-13b
    --dtype bfloat16  # for A100s and earlier, bfloat16 is more stable and memory-efficient than float16
    --tokenizer hf --tokenizer-path ${MODEL_DIR}  # Llama tokenizer
    --prompt-loss 0.3  # the loss of the source tokens will be scaled by 0.3
    --max-target-len 512  # max length concatenated source and targets: 512 tokens (256 source tokens,
    # 256 target tokens)
    --beam-size 1  # greedy decoding at evaluation
    --batch-size 4096  # max number of tokens per batch
    --max-steps 10000 --valid-interval 1000  # train for 10k updates and save/evaluate every 1k updates
    --label-smoothing 0 --dropout 0  # disable regularization
    -o ${MODEL_DIR}-mt-fr-en  # save the finetuned model to a different directory
)

FT_OPTS=(
    # Similar options to the LIMA paper
    --weight-decay 0.1 --adam-betas 0.9 0.95 --lr 1e-5 --min-lr 1e-6 --warmup 0 --virtual-dp-size 10
    # Full finetuning takes a lot of GPU memory: do tensor parallelism and activation checkpointing
    --tp-size 4  # Tensor Parallelism (the model is sharded across 4 GPUs)
    --checkpoint-activations  # considerably reduces memory usage by recomputing the activations
    # during the backward pass instead of storing them
)
LORA_OPTS=(
    # Parameter-efficient finetuning with Low-Rank Adapters
    --lr 2.5e-4 --dp-size 4 --virtual-dp-size 4 --lora-rank 16
    --save-trainable-only  # save disk space by only saving the LoRA parameters
)

# Full finetuning with 4-way tensor parallelism (train on 80G A100s to avoid OOMs)
pasero-train ${OPTS[@]} ${FT_OPTS[@]}

# Parameter-efficient finetuning with 4-way data parallelism
pasero-train ${OPTS[@]} ${LORA_OPTS[@]}
```

Tensor parallelism creates sharded checkpoints, which can be merged like follows:
```bash
scripts/merge-tp-ckpt.py models/llama-2-13b-mt-fr-en
```

Decoding can then be done on a single GPU using this merged checkpoint:

```bash
# Evaluation on FLORES-valid Fr-En
pasero-decode models/llama-2-13b-mt-fr-en -e data/FLORES/FLORES-valid -s fr -t en

# Interactive decoding
pasero-decode models/llama-2-13b-mt-fr-en -s fr -t en -n 1 -v

# Web playground
pasero-serve models/llama-2-13b-mt-fr-en  # then go to http://HOST:8000 where HOST 
# is "localhost" if running locally, or the name or IP of your server if running remotely
```

## Finetune LLMs on dialogue

Download a dialogue dataset, UltraChat-200k in this example (the dialogue dataset used to finetune [Zephyr-7B-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)):

```bash
git clone https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k data/ultrachat_200k
pushd data/ultrachat_200k/data
python3 -c "import pandas, json, os
for filename in os.listdir('.'):
    if filename.endswith('.parquet'):
        df = pandas.read_parquet(filename)
        json_filename = filename.replace('.parquet', '.jsonl')
        with open(json_filename, 'w') as json_file:
            for conversation in df.to_dict()['messages'].values():
                json_file.write(json.dumps(conversation.tolist()) + '\n')"
head -n1000 test_sft-00000-of-00001-f7dfac4afe5b93f4.jsonl > ../valid.jsonl
cat train_sft-*.jsonl > ../train.jsonl
popd
```

Finetune Mistral 7B on this dataset, using the `dialogue` task:

```bash
pasero-train -c examples/Llama/dialogue.yaml -o models/mistral-7b-ultrachat  # train on A100s with 2-way
# tensor parallelism
```

Try the dialogue model:
```bash
# merge the tensor parallelism shards
scripts/merge-tp-ckpt.py models/mistral-7b-ultrachat

# Web playground
pasero-serve models/mistral-7b-ultrachat  # then go to http://HOST:8000 where HOST is "localhost" if
# running locally, or the name or IP of your server if running remotely
```

The same can be done with Llama 2 13B:

```bash
OPTS=(
    -c examples/Llama/dialogue.yaml  # override the Mistral 7B config to work with Llama 2 13B
    --ckpt models/llama-2-13b/model_best.bin --arch llama_13b  # initialize with Llama 2 13B
    --tokenizer-path models/llama-2-13b  # HuggingFace Llama 2 tokenizer
    --tp-size 4  # Tensor Parallelism (the model is sharded across 4 GPUs)
    --checkpoint-activations  # considerably reduces memory usage by recomputing the activations during the backward
    # pass instead of storing them
    -o models/llama-2-13b-ultrachat  # save the finetuned model to a different directory
)

pasero-train ${OPTS[@]}  # train on 4 A100s with 4-way tensor parallelism
```

## Serve Llama and Mistral chat models

Pasero supports a bunch of existing chat models thanks to `dialogue` task. For instance, [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) is a finetuned version of Mistral 7B for dialogue:

```bash

huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir models/mistral-7b-instruct --local-dir-use-symlinks False --exclude "pytorch_model*"

# Convert the HuggingFace checkpoint to the Pasero format
scripts/convert-hf-ckpt.py models/mistral-7b-instruct/model-*.safetensors -o models/mistral-7b-instruct/model_bfloat16.bin --arch mistral --dtype bfloat16

# Create an "inference.yaml" file for pasero-decode and pasero-serve
echo "tokenizer: hf
tokenizer_path: mistralai/Mistral-7B-Instruct-v0.2
ckpt: model_bfloat16.bin
dtype: bfloat16
sampling: True
sampling_topp: 0.92
arch: mistral_7b
task: dialogue
chat_template: mistral
batch_size: 32768
model_args:
    sliding_window: null
    rope_base: 1000000" > models/mistral-7b-instruct/inference.yaml

# Serve the model
pasero-serve models/mistral-7b-instruct  # access the playground at http://HOST:8000
```
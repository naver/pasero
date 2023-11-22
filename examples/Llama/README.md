# Llama 2

This README gives steps to use Llama 2 models with Pasero and finetune them for machine translation or dialogue.

## Download and convert Llama-2-13b

```bash
pushd models
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-13b-hf llama-2-13b
# enter username and password
popd

# Convert the HuggingFace checkpoints to Pasero format
scripts/convert-hf-ckpt.py models/llama-2-13b/pytorch*.bin -o models/llama-2-13b/model_bfloat16.bin --arch llama --dtype bfloat16

# Convert the HuggingFace tokenizer to Pasero dict
scripts/hf-tokenizer-to-dict.py models/llama-2-13b > models/llama-2-13b/dict.txt

# Create an "inference.yaml" file for pasero-decode and pasero-serve
echo "tokenizer: sentencepiece
tokenizer_path: tokenizer.model
sampling: True
sampling_topp: 0.92
keep_whitespaces: True
arch: llama_13b
dtype: bfloat16
ckpt: model_bfloat16.bin
task: language_modeling" > models/llama-2-13b/inference.yaml
```

## Finetune Llama on French-English machine translation

```bash
examples/ParaCrawl/download.sh fr  # download the ParaCrawl Fr-En data
examples/download-flores.sh  # download the FLORES valid and test sets

MODEL_DIR=models/llama-2-13b

OPTS=(
    -s fr -t en --data-dir data/ParaCrawl  # train on ParaCrawl Fr-En
    --train-corpora "ParaCrawl.{pair}" --valid-corpora "data/FLORES/FLORES-valid"
    --ckpt ${MODEL_DIR}/model_bfloat16.bin --arch llama_13b  # initialize with Llama-2-13b
    --dtype bfloat16  # for A100s and earlier, bfloat16 is more stable and memory-efficient than float16
    --dict ${MODEL_DIR}/dict.txt  # Pasero dict extracted above
    --tokenizer sentencepiece --tokenizer-path ${MODEL_DIR}/tokenizer.model  # Llama tokenizer
    --target-tags "<0x0A>"  # each source and target sentence will be concatenated with a newline between them
    --prompt-loss 0.3  # the loss of the source tokens will be scaled by 0.3
    --max-target-len 512  # max length for this concatenation: 512 tokens (256 source tokens & 256 target tokens)
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
    --checkpoint-activations  # considerably reduces memory usage by recomputing the activations during the backward
    # pass instead of storing them
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
scripts/merge-tp-ckpt.py models/llama-2-13b-mt-fr-en/model_best.bin -o models/llama-2-13b-mt-fr-en/model_merged.bin
```

Decoding can then be done on a single GPU using this merged checkpoint:

```bash
# Evaluation on FLORES-valid Fr-En
pasero-decode models/llama-2-13b-mt-fr-en/model_merged.bin -e data/FLORES/FLORES-valid -s fr -t en

# Interactive decoding
pasero-decode models/llama-2-13b-mt-fr-en/model_merged.bin -s fr -t en -n 1 -v

# Web playground
pasero-serve models/llama-2-13b-mt-fr-en/model_merged.bin  # then go to http://HOST:8000 where HOST is "localhost" if
# running locally, or the name or IP of your server if running remotely
```

### Doc-level machine translation

Llama can also be finetuned on document-level French<->English machine translation as follows. The example uses 
Llama-2-7B and parameter-efficient finetuning with LoRA.

```bash
examples/Doc-level-MT/download.sh  # download doc-level Fr-En data

MODEL_DIR=models/llama-2-7b

OPTS=(
    -c examples/Doc-level-MT/training.yaml --lang-pairs fr-en en-fr  # train a fr<->en model on doc-level data
    --ckpt ${MODEL_DIR}/model_bfloat16.bin --arch llama_7b  # initialize with Llama-2-7b
    --dtype bfloat16  # for A100s and earlier, bfloat16 is more stable and memory-efficient than float16
    --dict ${MODEL_DIR}/dict.txt  # Pasero dict extracted above
    --tokenizer sentencepiece --tokenizer-path ${MODEL_DIR}/tokenizer.model  # Llama tokenizer
    --sent-sep "<0x0A>"  # newline as a separator between sentences in a document
    --target-tags "<0x0A>" "<0x0A>"  # 2 newlines as a separator between the source and target document
    --keep-whitespaces  # to prevent newlines from being removed at inference
    --prompt-loss 0.3  # the loss of the source tokens will be scaled by 0.3
    --max-target-len 1024  # max length for this concatenation: 1024 tokens (512 source tokens & 512 target tokens)
    --beam-size 1  # greedy decoding at evaluation
    --batch-size 4096 --virtual-dp-size 4 --dp-size 4  # effective batch size ~ 4096*4 tokens
    --max-steps 10000 --valid-interval 2000  # train for 10k updates and save/evaluate every 2k updates
    --label-smoothing 0 --dropout 0 --weight-decay 0.1  # Llama-like regularization
    --lr 5e-4  # max learning rate, with inverse sqrt schedule and 4000 steps of warmup
    --lora-rank 16 --save-trainable-only  # parameter-efficient finetuning
    -o ${MODEL_DIR}-doc-level-mt-fr-en  # save the finetuned model to a different directory
)

LORA_OPTS=(
    # Parameter-efficient finetuning with Low-Rank Adapters
    --lr 2.5e-4 --dp-size 4 --virtual-dp-size 4 --lora-rank 16
    --save-trainable-only  # save disk space by only saving the LoRA parameters
)

pasero-train ${OPTS[@]} ${LORA_OPTS[@]}  # train on 4 GPUs (preferably 4 A100s) with 4-way data parallelism
```

```bash
pasero-decode models/llama-2-7b-doc-level-mt-fr-en -s en -t fr -v -n 1
# This is an English sentence.\nHere is another example.
# H-2	▁This ▁is ▁an ▁English ▁sentence . <0x0A> ▁Here ▁is ▁another ▁example . <0x0A> <0x0A> ▁C ' est ▁une ▁phrase ▁en ▁anglais . <0x0A> ▁Vo ici ▁un ▁autre ▁exemple . </s>
# D-2	C'est une phrase en anglais.
# Voici un autre exemple.
```

## Finetune LLama on dialogue

Download a dialogue dataset, UltraChat in this example:

```bash
mkdir -p data/dialogue/ultrachat
pushd data/dialogue/ultrachat
wget https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_0.jsonl  # this example uses just a subset
# of the full training set

# Convert to the Pasero dialogue format
python3 -c "
import sys, json
for line in sys.stdin:
    try:
        js = json.loads(line)
    except:
        continue
    data = js['data']
    conversation = []
    for i, content in enumerate(data):
        role = 'user' if i % 2 == 0 else 'assistant'
        conversation.append({'role': role, 'content': content})
    print(json.dumps(conversation))
" < train_0.jsonl > all.jsonl
head -n-1000 all.jsonl > train.jsonl
tail -n1000 all.jsonl > valid.jsonl  # use the last 1k dialogues for evaluation
popd
```

Finetune Llama-2-13b on this dataset, using the `dialogue` task:

```bash
MODEL_DIR=models/llama-2-13b

OPTS=(
    --task dialogue --chat-template zephyr  # other templates are available (e.g., ChatMT and Llama-2)
    --data-dir data/dialogue/ultrachat  # train on UltraChat
    --train-corpora train.jsonl --valid-corpora valid.jsonl
    --ckpt ${MODEL_DIR}/model_bfloat16.bin --arch llama_13b  # initialize with Llama-2-13b
    --dtype bfloat16  # for A100s and earlier, bfloat16 is more stable and memory-efficient than float16
    --dict ${MODEL_DIR}/dict.txt  # Pasero dict extracted above
    --tokenizer hf --tokenizer-path ${MODEL_DIR}  # Llama tokenizer
    --prompt-loss 0  # disable the loss on the prompt tokens (user messages)
    --decoder-max-len 2048  # max length of a dialogue
    --batch-size 8192 --virtual-dp-size 4  # effective batch size ~ 8192*4 tokens
    --max-steps 10000 --valid-interval 1000  # train for 10k updates (~200k dialogues)
    --label-smoothing 0 --dropout 0 --weight-decay 0.1  # Llama-like regularization
    --lr 1e-5 --adam-betas 0.9 0.95 --warmup 1000  # inverse sqrt schedule with 1000 steps of warmup
    --tp-size 4  # Tensor Parallelism (the model is sharded across 4 GPUs)
    --checkpoint-activations  # considerably reduces memory usage by recomputing the activations during the backward
    # pass instead of storing them
    -o ${MODEL_DIR}-ultrachat  # save the finetuned model to a different directory
)

pasero-train ${OPTS[@]}  # train on 4 GPUs (preferably 4 A100s) with 4-way tensor parallelism
```

Try the dialogue model:
```bash
# merge the tensor parallelism shards
scripts/merge-tp-ckpt.py models/llama-2-13b-ultrachat/model_best.bin -o models/llama-2-13b-ultrachat/model_merged.bin

# Web playground
pasero-serve models/llama-2-13b-ultrachat/model_merged.bin  # then go to http://HOST:8000 where HOST is "localhost" if
# running locally, or the name or IP of your server if running remotely
```

## Serve Llama and Mistral chat models

Pasero supports a bunch of existing chat models thanks to `dialogue` task. For instance, [Zephyr 7B Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) is a finetuned 
version of Mistral 7B for dialogue:

```bash
pushd models
git lfs install
git clone https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
popd

# Convert the HuggingFace checkpoints to Pasero format: Mistral 7B and Zephyr 7B have the same checkpoint format as Llama 7B
scripts/convert-hf-ckpt.py models/zephyr-7b-beta/pytorch*.bin -o models/zephyr-7b-beta/model_bfloat16.bin --arch llama --dtype bfloat16

# Convert the HuggingFace tokenizer to Pasero dict
scripts/hf-tokenizer-to-dict.py models/zephyr-7b-beta > models/zephyr-7b-beta/dict.txt

# Create an "inference.yaml" file for pasero-decode and pasero-serve
echo "tokenizer: hf
ckpt: model_bfloat16.bin
dtype: bfloat16
sampling: True
sampling_topp: 0.92
keep_whitespaces: True
arch: mistral_7b
task: dialogue
chat_template: zephyr
system_prompt: \"You are a helpful, respectful and honest AI assistant.\"" \
> models/zephyr-7b-beta/inference.yaml

# Serve the model
pasero-serve models/zephyr-7b-beta  # access the playground at http://localhost:8000
```
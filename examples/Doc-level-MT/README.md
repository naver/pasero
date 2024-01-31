# Document-level French-English machine translation

Pasero supports document-level translation thanks to `--task doc_level_translation`. It assumes either sentence-level parallel corpora (one sentence per line) whose sentences are ordered: consecutive sentences are likely to belong to the same document and can be randomly merged into the same training example; or doc-level parallel corpora (one doc per line) whose sentences are separated by `<sep>`.
Read `examples/Doc-level-MT/training.yaml` for an explanation of the different training options.

The script `examples/Doc-level-MT/download.sh` can be used to download doc-level (or rather ordered sent-level) French-English training
data and test sets.

The examples below are training recipes for finetuning decoder-only language models on doc-level translation, using either the 
`doc_level_translation` task or the `dialogue` task. But not that the former is also compatible with encoder-decoder Transformers.

## Finetuning Mistral 7B on doc-level MT

This example finetunes Mistral 7B on French<->English document-level machine translation using parameter-efficient finetuning with LoRA.
Read [`examples/Llama`](/examples/Llama/README.md) for explanations on how to download and convert Mistral 7B checkpoints and other advice for finetuning or using Llama-style models.

```bash
examples/Doc-level-MT/download.sh  # download doc-level Fr-En data

MODEL_DIR=models/mistral-7b

OPTS=(
    -c examples/Doc-level-MT/training.yaml --lang-pairs fr-en en-fr  # train a fr<->en model on doc-level data
    --ckpt ${MODEL_DIR}/model_bfloat16.bin --arch mistral_7b  # initialize with Mistral 7B
    --dtype bfloat16  # for A100s and earlier, bfloat16 is more stable and memory-efficient than float16
    --tokenizer hf --tokenizer-path ${MODEL_DIR}  # Llama tokenizer
    --dict none  # use the HuggingFace tokenizer's built-in vocab, not the dict in data/Doc-level
    --sent-sep "<0x0A>"  # newline as a separator between sentences in a document
    --keep-whitespaces  # to prevent newlines from being removed at inference
    --prompt-loss 0.3  # the loss of the source tokens will be scaled by 0.3
    --decoder-max-len 1024   # max length for concatenated source and target documents: 1024 tokens
    # (512 source tokens, 512 target tokens)
    --beam-size 1  # greedy decoding at evaluation
    --batch-size 4096 --virtual-dp-size 4 --dp-size 4  # effective batch size ~ 4096*4 tokens
    --max-steps 10000 --valid-interval 1000  # train for 10k updates and save/evaluate every 1k updates
    --label-smoothing 0 --dropout 0 --weight-decay 0.1  # Llama-like regularization
    --lr 5e-4  # max learning rate, with inverse sqrt schedule and 4000 steps of warmup
    --lora-rank 16 --save-trainable-only  # parameter-efficient finetuning
    -o ${MODEL_DIR}-doc-level-mt-fr-en  # save the finetuned model to a different directory
)

LORA_OPTS=(
    # Parameter-efficient finetuning with Low-Rank Adapters
    --lr 2.5e-4
    --lora-rank 16
    --save-trainable-only  # save disk space by only saving the LoRA parameters
)

pasero-train ${OPTS[@]} ${LORA_OPTS[@]}  # train on 4 GPUs (preferably 4 A100s) with 4-way data parallelism
```

```bash
pasero-decode models/mistral-7b-doc-level-mt-fr-en -s en -t fr -v -n 1
# This is an English sentence.\nHere is another example.
# H-2	▁This ▁is ▁an ▁English ▁sentence . <0x0A> ▁Here ▁is ▁another ▁example . </s> ▁C ' est ▁une ▁phrase ▁en ▁anglais . <0x0A> ▁Vo ici ▁un ▁autre ▁exemple . </s>
# D-2	C'est une phrase en anglais.
# Voici un autre exemple.
```

## Machine translation as dialogue

Document-level machine translation can be framed as a dialogue task, where the user messages are the source-language
sentences and the assistant messages are the target-language sentences.
One advantage of doing this instead of using the doc-level translation task is that it lets you define custom
templates. It also allows simultaneous contextual translation, where not all the source sentences are known in advance.

The example below finetunes TinyLlama 1.1B, but the same can be applied on bigger models like Mistral or Llamas.

```bash
# Download doc-level Fr-En machine translation data
examples/Doc-level-MT/download.sh

# Create a dialogue dataset from OpenSubtitles
DATA_DIR=data/Doc-level

# English -> French
paste ${DATA_DIR}/OpenSubtitles.en-fr.{en,fr} | \
examples/Doc-level-MT/sent2doc.py --min-doc-size 1 --doc-size 10 | \
examples/Doc-level-MT/doc2dialogue.py | head -n 1000000 > ${DATA_DIR}/OpenSubtitles.en-fr.jsonl
paste ${DATA_DIR}/newstest2013.en-fr.{en,fr} | examples/Doc-level-MT/doc2dialogue.py > ${DATA_DIR}/newstest2013.en-fr.jsonl

# French -> English
paste ${DATA_DIR}/OpenSubtitles.en-fr.{fr,en} | \
examples/Doc-level-MT/sent2doc.py --min-doc-size 1 --doc-size 10 | \
examples/Doc-level-MT/doc2dialogue.py | head -n 1000000 > ${DATA_DIR}/OpenSubtitles.fr-en.jsonl
paste ${DATA_DIR}/newstest2013.en-fr.{fr,en} | examples/Doc-level-MT/doc2dialogue.py > ${DATA_DIR}/newstest2013.fr-en.jsonl

# Finetune TinyLlama-1.1B on Fr-En translation
MODEL_DIR=models/tinyllama-1.1b
OPTS=(
    --task dialogue --chat-template zephyr  # other templates are available (e.g., ChatMT and Llama-2)
    --data-dir data/Doc-level --dict none  # set dict to a dummy value to prevent pasero from 
    # loading "data/Doc-level/dict.txt" (which is not for TinyLlama)
    --train-corpora OpenSubtitles.fr-en.jsonl --valid-corpora newstest2013.fr-en.jsonl
    --ckpt ${MODEL_DIR}/model_bfloat16.bin --arch llama_1b  # initialize with TinyLlama-1.1B
    --tokenizer hf --tokenizer-path ${MODEL_DIR}  # TinyLlama tokenizer
    --prompt-loss 0.3  # let the model learn a bit about French
    --decoder-max-len 2048  # max length of a dialogue
    --batch-size 8192 --virtual-dp-size 4  # effective batch size ~8192*4 tokens
    --max-steps 10000 --valid-interval 1000  # train for 10k updates (~200k dialogues)
    --buffer-size 10000  # load that many dialogues before batching (should be smaller than the corpus size)
    --label-smoothing 0 --dropout 0 --weight-decay 0.1  # Llama-like regularization
    --lr 1e-5 --adam-betas 0.9 0.95 --warmup 1000  # inverse sqrt schedule with 1000 steps of warmup
    --metrics spbleu chrf --beam-size 1  # do inference every 1000 steps and evaluate the output with these metrics
    --early-stopping-metric chrf  # pick best checkpoint with chrF instead of valid perplexity
    -o ${MODEL_DIR}-doc-level-mt-fr-en  # save the finetuned model to a different directory
)
pasero-train ${OPTS[@]} --float16 --checkpoint-activations  # ~4 hours on 4 V100s

# Serve the model
pasero-serve ${MODEL_DIR} --port 8000  # go to http://HOST:8000 to use the model
```
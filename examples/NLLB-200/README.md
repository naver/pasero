
# NLLB-200 dense models

The dense [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) models (600M, 1.3B and 3.3B) can be downloaded and converted to the Pasero format by running `examples/NLLB-200/download-dense.sh`

Then decoding from them is easy:
```bash
pasero-decode models/NLLB-200/3.3B_dense.bin -e data/FLORES/FLORES-valid -s eng_Latn -t fra_Latn --source-lang-code --target-lang-code --tokenizer sentencepiece
```

NLLB-200 uses longer language codes than usual (e.g., `eng_Latn` instead of `en`). A mapping between these language codes and the more common 2-letter language codes (such as used in M2M-100 or our ParaCrawl examples) is given in [lang-mapping.txt](examples/NLLB-200/lang-mapping.txt). The [FLORES download script](examples/download-flores.sh) automatically creates symbolic links for all language code variants (e.g., `FLORES-valid.en -> FLORES-valid.eng_Latn`).

The `--source-lang-code` and `--target-lang-code` options prepend the source language code to each input line and force the model to output the target language code at the first decoding step.
Note that we can actually skip the tokenizer and lang code options, which are specified in `inference.yaml`.

For convenience, the download script also creates a dictionary with 2-letter language codes:
```bash
pasero-decode models/NLLB-200/3.3B_dense.bin -s en -t fr -n 1 -v --dict dict-short-codes.txt
```

## Noise adapters for NLLB-200

Train small encoder adapters that can make NLLB-200 more robust to English typos.

```bash
MODEL_DIR=models/NLLB-200

OPTS=(
    -c examples/ParaCrawl/training.yaml -s en -t fr  # re-use and overload the bilingual ParaCrawl configuration
    --max-steps 2000  # we don't need to train for too long
    --max-source-len 256 --max-target-len 256  # avoid CUDA memory spikes by filtering too long examples
    --tokenizer sentencepiece --tokenizer-path ${MODEL_DIR}/spm.model  # NLLB-200 tokenizer
    --source-lang-code --target-lang-code  # NLLB-200 tagging
    --normalize-punctuation  # NLLB-200 punctuation normalization
    --dict ${MODEL_DIR}/dict-short-codes.txt  # NLLB-200 dict with 2-letter language codes
    --ckpt ${MODEL_DIR}/3.3B_dense.bin --arch adapter_nllb_3b3  # NLLB-200 checkpoint and architecture
    --encoder-adapters noise --decoder-adapters  # train encoder adapters
    --space-noise 0.1 --punct-noise 0.1 --char-noise 0.05  # enable source-side noise
    -o ${MODEL_DIR}/3.3B-en-fr-noise  # save to a new model directory
    # Uncomment below to reduce GPU memory usage (to train on 32G V100s):
    --checkpoint-activations --batch-size 4000 --virtual-dp-size 64
)
pasero-train ${OPTS[@]}
```
### Try it!

```bash
pasero-decode models/NLLB-200/3.3B-en-fr-noise -s en -t fr --encoder-adapters noise -v -n 1
# Here is an example -> Voici un exemple.
# Heree uis anexampel -> Voici un exemple.
```
This also works fine for translating in other languages than French.

## Fast language-specific decoder for NLLB-200

### Build a French subset of the dictionary
```bash
scripts/spm-encode.py models/NLLB-200/spm.model < data/ParaCrawl/ParaCrawl.en-fr.fr | head -n1000000 | \
pasero-build-dict -o models/NLLB-200/dict.fr.txt --dict-max-size 16000 --dict-custom-symbols "<lang:fr>"
```

### Train a shallow LSTM decoder, while freezing the encoder

```bash
MODEL_DIR=models/NLLB-200

OPTS=(
    -c examples/ParaCrawl/training.yaml -s en -t fr  # re-use and overload the bilingual ParaCrawl configuration
    --max-steps 10000  # we don't need to train for too long
    --max-source-len 256 --max-target-len 256  # avoid CUDA memory spikes by filtering too long examples
    --tokenizer sentencepiece --tokenizer-path ${MODEL_DIR}/spm.model  # NLLB-200 tokenizer
    --source-lang-code --target-lang-code  # NLLB-200 tagging
    --normalize-punctuation  # NLLB-200 punctuation normalization
    --dict ${MODEL_DIR}/dict-short-codes.txt  # NLLB-200 dict with 2-letter language codes
    --target-dict ${MODEL_DIR}/dict.fr.txt --old-target-dict ${MODEL_DIR}/dict-short-codes.txt  # new French-only target dict
    --no-shared-embeddings  # source and target dict are now different, target embeddings will be re-maped
    --ckpt ${MODEL_DIR}/3.3B_dense.bin --flexible  # NLLB-200 checkpoint partial loading
    --train-params-regex 'encoder\.layers\.23\.|decoder\.'  # which parameters to train (the rest are frozen),
    # look at train.log to find the full list of parameter names. Last encoder layer and entire decoder are trained
    --arch hybrid_transformer_big --embed-dim 2048 --encoder-layers 24 --encoder-ffn-dim 8192 --encoder-prenorm
    # NLLB-200 3.3B architecture with an LSTM decoder (i.e., we keep the encoder but replace the decoder)
    --decoder-hidden-size 2048
    -o ${MODEL_DIR}/3.3B-en-fr-lstm  # save to a new model directory
    # Uncomment below to reduce GPU memory usage (to train on 32G V100s):
    --checkpoint-activations --batch-size 4000 --virtual-dp-size 64
)
pasero-train ${OPTS[@]}
```
### Now compute scores

```bash
pasero-decode models/NLLB-200/3.3B-en-fr-lstm -l en-fr de-fr -e data/FLORES/FLORES-test

pasero-decode models/NLLB-200/3.3B_dense.pt --dict dict-short-codes.txt -l en-fr de-fr -e data/FLORES/FLORES-test

pasero-decode models/NLLB-200/1.3B_distilled.bin --dict dict-short-codes.txt -l en-fr de-fr -e data/FLORES/FLORES-test
```

| Decoder | EN-FR spBLEU | Time (s) | DE-FR spBLEU |
|---------|------------|------------|------------|
| NLLB-200 decoder | 55.81 | 289 | 44.76 |
| LSTM decoder | 54.01 | 27 | 40.93 |
| NLLB-200 1.3B | 54.69 | 132 | 43.96 |

We speed up the model by 10x at the cost of 1.8 BLEU point. German-French translation performance drops by 3 BLEU points but remains reasonable considering we trained with English-French data only.

## Test-time vocabulary filtering

Using the French dictionary created above, one can also speed up decoding with the NLLB-200 models thanks to 
[test-time vocabulary filtering](https://aclanthology.org/2021.emnlp-main.674/):

```bash
pasero-decode models/NLLB-200/3.3B_dense.bin -s en -t fr -v -n 1 --dict dict-short-codes.txt --target-dict dict.fr.txt --old-target-dict dict-short-codes.txt
```

This can result in a small drop in translation quality.

# NLLB-200 Mixture-of-Experts

## Decoding with Tutel

First download the NLLB-200 Mixture-of-Experts model by running `examples/download-moe.sh`

```bash
# Install Microsoft's tutel (fast implementation of MoE layers)
pip install --upgrade git+https://github.com/microsoft/tutel@main

# Decode from French and German into English

pasero-decode models/NLLB-200/54B_moe.bin --task nllb_translation \
--expert-ckpt models/NLLB-200/experts/*.bin \
-e data/FLORES/FLORES-valid \
-l fra_Latn-eng_Latn deu_Latn-eng_Latn \
-o "outputs/FLORES-valid.{pair}.out" \
-n 1012 --batch-size 16000 --lines-per-batch 400 \
--dp-size 8
```

Note that decoding with all 128 experts requires more than 4 V100s, as each GPU needs to hold a full copy of the dense model (5.8 GiB) as well as 128/N experts (0.84 GiB per expert). It is possible to run it on 8 V100s or 4 A100s.
Decoding with 64 experts is possible on 4 V100s, with similar performance:

```bash
pasero-decode models/NLLB-200/54B_moe.bin --task nllb_translation \
--expert-ckpt models/NLLB-200/experts/*-{0-63}.bin \
-e data/FLORES/FLORES-valid \
-l fra_Latn-eng_Latn deu_Latn-eng_Latn \
-o "outputs/FLORES-valid.{pair}.out" \
-n 1012 --batch-size 16000 --lines-per-batch 400 \
--dp-size 4
```

Or use the `--encoder-decoder-swapping` option, which halves the GPU memory usage (at the cost of much higher RAM usage), by moving the encoder to the CPU once it has finished its job and only then move the decoder to the GPU:
```bash
pasero-decode models/NLLB-200/54B_moe.bin --task nllb_translation \
--expert-ckpt models/NLLB-200/experts/*.bin \
-e data/FLORES/FLORES-valid \
-l fra_Latn-eng_Latn deu_Latn-eng_Latn \
-o "outputs/FLORES-valid.{pair}.out" \
-n 1012 --batch-size 16000 --lines-per-batch 400 \
--dp-size 4 --encoder-decoder-swapping
```

The `--batch-size` and `--lines-per-batch` values may be increased (at the risk of running into out of memory errors) to improve decoding speed. Note that those are actually divided by the number of GPUs: each GPU will receive at most 50 lines per batch in the current setting. A different maximum length can also be specified with `--max-output-len` at the cost of higher memory usage and slower decoding (default: 100).

The scores (and potential errors) are written to the SLURM output in `tmp/SLURM_JOB_ID`. The FLORES-test chrF++ scores can be compared against those in [`54B_moe_metrics.csv`](https://tinyurl.com/nllb200moe54bmetrics) (list of scores released by FAIR).

To obtain gate statistics, the `--moe-stats` option can be added, in which case the statistics will be appended at the end of the output files (lines starting with `MOE\t`).

## Pruning experts to fit the model in 1 GPU

Like in [our paper about expert pruning in NLLB-200](https://arxiv.org/abs/2212.09811), experts can be pruned per language.
Have a look at [this other repo](https://github.com/naver/nllb-pruning) for more pruning options using HuggingFace Transformers.

```bash
# Decode from French and German into English with 80% expert pruning (expert ids specified in 'experts.json')
# Language-specific experts (French or German encoder experts and English decoder experts) will be loaded automatically

pasero-decode models/NLLB-200/54B_moe.bin --task nllb_translation \
--expert-dir models/NLLB-200/experts \
--expert-json models/NLLB-200/experts.json \
-e data/FLORES/FLORES-valid \
-l fra_Latn-eng_Latn deu_Latn-eng_Latn \
-o "outputs/FLORES-valid.{pair}.out"
```

## Reference

```bibtex
@inproceedings{koishekenov-etal-2023-memory,
    title = "Memory-efficient {NLLB}-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model",
    author = "Koishekenov, Yeskendir  and
      Berard, Alexandre  and
      Nikoulina, Vassilina",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    url = "https://aclanthology.org/2023.acl-long.198",
}
```
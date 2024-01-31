## 26-language ParaCrawl models

This recipe is for training multilingual models similar to the ones presented in the following papers:

- [Efficient Inference for Multilingual Neural Machine Translation](https://aclanthology.org/2021.emnlp-main.674/), EMNLP 2021
- [Continual Learning in Multilingual NMT via Language-Specific Embeddings](https://aclanthology.org/2021.wmt-1.62/), WMT 2021

The difference is that the papers above were only on 20 languages and used an older version of ParaCrawl (ParaCrawl v7 instead of v9).

### First, download the ParaCrawl corpora

Be advised that this command will take a lot of memory and disk space and a very long time to run:
```bash
examples/ParaCrawl-Euro/download.sh
```

### Optionally create a Pasero BPE model and dictionary

These are already available in [examples/ParaCrawl-Euro](/examples/ParaCrawl-Euro)

```bash
pasero-build-tokenizer -i data/ParaCrawl-Euro/multiparallel/ParaCrawl.{en,fr,de,es,it,pt,nl,nb,cs,pl,sv,da,el,fi,hr,hu,bg,ro,sk,lt,lv,sl,et,ga,is,mt} \
-o data/ParaCrawl-Euro/bpecodes -d data/ParaCrawl-Euro/dict.txt -s 64000 --dict-min-freq 100 --nfkc --temperature 5 --lang-codes \
--vocab-path data/ParaCrawl-Euro/dict.{lang}.txt
```

### Train an English-centric model

```bash
pasero-train -c examples/ParaCrawl-Euro/training-en-centric.yaml --dp-size 4 -o models/ParaCrawl-Euro/wide.12-2.en-centric
```

### Finetune the English-centric model with multi-parallel data

```bash
pasero-train -c examples/ParaCrawl-Euro/training.yaml --dp-size 4 --ckpt models/ParaCrawl-Euro/wide.12-2.en-centric/model_last.bin -o models/ParaCrawl-Euro/wide.12-2.multi-parallel --lr 0.0003
```

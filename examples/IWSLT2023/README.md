# Speech-to-text translation

Pasero can be used to reproduce [our winning system](https://aclanthology.org/2023.iwslt-1.10) to the [IWSLT 2023 Low-Resource Track](https://aclanthology.org/2023.iwslt-1.1/).

This system is based on strong pre-trained models for speech and text:
- The model is initialized with [NLLB-200](https://arxiv.org/abs/2207.04672) and then finetuned on speech-to-text data from mTEDx, with speech features extracted 
with a multilingual Wav2vec model.
- The finetuning is parameter-efficient: only the bottom 3 encoder layers are trained, along with a convolution layer to reduce the length of the speech inputs, and tiny adapters after every Transfomer layer.
- We show in the paper that in addition to improving the state of the art by a large margin, this architecture is data- and compute-efficient and it has excellent zero-shot abilities (e.g., it can translate Tamasheq to English).

```bash
# Download the IWSLT2023 data and NLLB model
examples/IWSLT2023/download.sh

# Download the speech feature model and extract features (run this on a machine with a GPU)
examples/IWSLT2023/prepare.sh w2v2nima

# Train our "Contrastive 1" model: non-ensemble version of our top-ranking Tamasheq-French model at IWSLT 2023
pasero-train -c examples/IWSLT2023/taq-fr-contrastive-1.yaml -o models/iwslt2023/taq-fr-contrastive-1

# Decode the IWSLT 2022 Tamasheq test set with it
pasero-decode models/iwslt2023/taq-fr-contrastive-1 -i data/iwslt2023/w2v2nima-8/tamasheq/test.npy.taq \
-r data/iwslt2023/w2v2nima-8/tamasheq/test.fr --encoder-adapters default --decoder-adapters default
```

In this work, we also showed that the same architecture can achieve excellent results on high-resource languages:

```bash
# Download the speech feature model and extract features (run this on a machine with a GPU)
examples/IWSLT2023/prepare.sh xlsr128

# Train our model for the IWSLT2021 multilingual setting
pasero-train -c examples/IWSLT2023/xlsr+nllb-iwslt2021.yaml -o models/iwslt2023/xlsr+nllb-iwslt2021

# Decode the IWSLT 2021 Fr-En test set with it
pasero-decode models/iwslt2023/xlsr+nllb-iwslt2021 -i data/iwslt2023/xlsr128-18/mtedx/fr-en/iwslt2021.npy.fr \
-r data/iwslt2023/xlsr128-18/mtedx/fr-en/iwslt2021.en --encoder-adapters default --decoder-adapters default
```

## Reference

```bibtex
@inproceedings{gow-smith-etal-2023-naver,
    title = "{NAVER} {LABS} {E}urope{'}s Multilingual Speech Translation Systems for the {IWSLT} 2023 Low-Resource Track",
    author = "Gow-Smith, Edward  and
      Berard, Alexandre  and
      Zanon Boito, Marcely  and
      Calapodescu, Ioan",
    booktitle = "Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023)",
    url = "https://aclanthology.org/2023.iwslt-1.10",
}
```
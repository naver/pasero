# Whisper

## Converting the checkpoints

```bash
MODEL_DIR=models
mkdir -p ${MODEL_DIR}
pushd ${MODEL_DIR}
git clone https://huggingface.co/openai/whisper-base
popd
scripts/convert-hf-ckpt.py ${MODEL_DIR}/whisper-base/pytorch_model.bin --arch whisper -o ${MODEL_DIR}/whisper-base/model_best.bin
cp examples/Whisper/{dict.txt,inference.yaml} ${MODEL_DIR}/whisper-base
```

## Extracting features

```bash
# download the LibriSpeech test set (raw audio)
LIBRISPEECH_DIR=data/LibriSpeech
mkdir -p ${LIBRISPEECH_DIR}
pushd ${LIBRISPEECH_DIR}
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar xzf test-clean.tar.gz
popd

# convert the audio into features
cat ${LIBRISPEECH_DIR}/test-clean/*/*/*.txt | cut -d' ' -f1 | \
examples/Whisper/extract-features.py -o test-clean.npy.en \
--audio-dirs ${LIBRISPEECH_DIR}/test-clean/*/* \
--file-extension flac --dtype float16
```

Note that this script will pad all sequences to 3000, which is the same as what Whisper's official implementation on HuggingFace does.
This rather inefficient padding can be disabled with `--no-padding`, but this will result in hallucinations (Whisper was probably trained with this padding).

## Decoding

Transcribe to English:

```bash
pasero-decode ${MODEL_DIR}/whisper-base -i test-clean.npy.en -t en -v  # -v is for verbose outputs (tokenization, scores, etc.)
```

Some default settings are specified in `${MODEL_DIR}/whisper-base/inference.yaml`, but they can be overriden with command-line options.
For example, to translate French to English line by line with beam search:

```bash
pasero-decode ${MODEL_DIR}/whisper-base -i mtedx-valid.npy.fr -t en --target-tags "<|en|>" "<|translate|>" "<|notimestamps|>" --buffer-size 1 --beam-size 3 -v
```

Use `--arch whisper_large` if you're using Whisper Large.

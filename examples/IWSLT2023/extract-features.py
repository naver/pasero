#!/usr/bin/env python3
# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import sys
import os
import functools
import argparse
import numpy as np
import torch
import torchaudio
import tqdm
from typing import Optional, Iterable, Iterator
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from torch import Tensor
import torch.nn.functional as F

from pasero.files import NumpyFile


description="""
Reads a list of audio filenames (and optional start and end position) from standard input, extracts features
from those and stores the result in a numpy format compatible with Pasero.

Example "segments" file:

706tS6pW0BA 6.92 7.12
706tS6pW0BA 8.70 18.63
706tS6pW0BA 21.03 34.63
706tS6pW0BA 36.07 38.30
...

```
examples/IWSLT2023/w2v2-to-numpy.py models/IWSLT2022-Niger-Mali --layer-id 8 \
--audio-dirs data/tamasheq/test2023 -i segments -o data/tamasheq/test2023.taq-fr.npy.taq --file-extension ".wav"
```
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('huggingface_model', help='path to a huggingface model directory')

parser.add_argument('--audio-dirs', nargs='+', default=['.'], help='directories containing the audio files')
parser.add_argument('--file-extension', help='append this extension to the input filenames')

parser.add_argument('-i', '--input', help='read segment filenames from this input file instead of standard input')
parser.add_argument('--txt-file', help='read lines from this text file in parallel with the audio segments and '
                    'print the lines to standard output, empty audio segments and the corresponding lines are skipped')
parser.add_argument('-o', '--output', required=True, help='binary file that will contain the output features')

parser.add_argument('--dtype', default='float16', choices=['float16', 'float32'], help='convert the features to this '
                    'type, float16 is half as compact as float32 but it might not work as well')
parser.add_argument('--device', default='cuda', help='run the model on this device (e.g., cpu, cuda:1); default: cuda')
parser.add_argument('--sampling-rate', type=int, default=16000, help='resample the audio to use this sample rate')
parser.add_argument('--layer-id', type=int, default=-1, help='extract features at this layer')

parser.add_argument('--batch-size', type=int, default=10, help='number of audio samples to batch together')
parser.add_argument('--max-length', type=int, default=30*16000, help='maximum length of an audio sample after '
                    'resampling (longer inputs will be truncated)')


@functools.lru_cache(3)  # saves a lot of time when multiple consecutive segments are from the same large audio file
def load_audio(path: str, sampling_rate: int):
    waveform, sampling_rate_ = torchaudio.load(path)
    if waveform.size(1) == 0:
        return None
    waveform = torchaudio.functional.resample(waveform, sampling_rate_, sampling_rate)
    waveform = waveform.mean(dim=0)
    return waveform


def load_and_split_audio(path: str, sampling_rate: int, start: Optional[float] = None, end: Optional[float] = None) -> Tensor:
    waveform = load_audio(path, sampling_rate)
    if waveform is None:
        return None
    start = None if start is None else int(start * sampling_rate)
    end = None if end is None else int(end * sampling_rate)
    return waveform[start:end]


def find_file(path: str, dirs: list[str]) -> str:
    for dir in dirs:
        found_path = os.path.join(dir, path)
        if os.path.exists(found_path):
            return found_path

    # don't raise an exception, just return None. This example will be skipped
    print(f"Audio file '{path}' wasn't found anywhere", file=sys.stderr)


def make_batches(samples: Iterable[Tensor], batch_size: int) -> Iterator[list[Tensor]]:
    batch = []
    for sample in samples:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(sample)
    if batch:
        yield batch


def get_features(
    samples: Iterable[Tensor],
    model: Wav2Vec2Model,
    processor: Wav2Vec2FeatureExtractor,
    sampling_rate: int,
    batch_size: int = 10,
    layer_id: int = -1,
) -> Iterator[np.ndarray]:
    
    all_lengths = []

    for batch in make_batches(samples, batch_size=batch_size):

        with torch.inference_mode():
            batch = [F.layer_norm(sample, sample.shape) for sample in batch]
            batch = [sample.numpy() for sample in batch]
            batch = processor(
                batch,
                return_attention_mask=True,
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors='pt',
            )

            feats = model(**batch.to(model.device)).hidden_states[layer_id]

            lengths = batch.attention_mask.sum(dim=-1)
            lengths = model._get_feat_extract_output_lengths(batch.attention_mask.sum(dim=-1))

            for x, length in zip(feats, lengths):
                length = length.item()
                yield x[:length].cpu().numpy()
                all_lengths.append(length)

    print(f'lines={len(all_lengths)} tokens={sum(all_lengths)}', file=sys.stderr)
    all_lengths = np.array(all_lengths, dtype=np.int64)
    print(f'length stats | max={all_lengths.max()} min={all_lengths.min()} avg={all_lengths.mean():.1f} '
        f'95th={np.quantile(all_lengths, 0.95):.1f}',
        f'99th={np.quantile(all_lengths, 0.99):.1f}',
        file=sys.stderr)


if __name__ == '__main__':

    args = parser.parse_args()
    if args.file_extension:
        args.file_extension = '.' + args.file_extension.lstrip('.')  # add "." if missing (e.g., "wav" -> ".wav")

    paths = []
    infile = open(args.input) if args.input else sys.stdin
    for line in infile:
        filename, *times = line.rsplit(maxsplit=3)
        if args.file_extension:
            filename = filename.removesuffix(args.file_extension) + args.file_extension
        path = find_file(filename, args.audio_dirs)
        if times:
            start, end = times
            start, end = float(start), float(end)
        else:
            start, end = 0, None
        paths.append((path, start, end))

    txt_file = open(args.txt_file) if args.txt_file else None

    def get_samples():
        for path, start, end in tqdm.tqdm(paths, total=len(paths)):
            # Read text file in parallel to skip lines that correspond to audio samples that are empty
            line = None if txt_file is None else next(txt_file)
            if path is None:  # if file doesn't exist (skips text line)
                continue
            sample = load_and_split_audio(path, sampling_rate=args.sampling_rate, start=start, end=end)
            if sample is None:  # if audio sample is empty (skips text line)
                continue
            # truncate audio samples that are too long
            sample = sample[:args.max_length]
            yield sample
            if line is not None:
                print(line.strip())

    samples = get_samples()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.huggingface_model)
    model = Wav2Vec2Model.from_pretrained(args.huggingface_model, output_hidden_states=True)
    model = model.to(args.device)

    feats = get_features(
        samples,
        model,
        processor,
        batch_size=args.batch_size,
        layer_id=args.layer_id,
        sampling_rate=args.sampling_rate,
    )

    NumpyFile.build(args.output, feats, dtype=args.dtype, num_feats=len(paths))

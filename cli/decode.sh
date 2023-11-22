#!/usr/bin/env bash

#SBATCH -o tmp/%j
#SBATCH --open-mode=append
#SBATCH --mem-per-gpu=60G
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 pasero-decode $@
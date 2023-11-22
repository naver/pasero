#!/usr/bin/env bash

#SBATCH -o tmp/%j
#SBATCH --open-mode=append
#SBATCH --mem-per-gpu=60G
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=6

OPTS=$@

trap 'kill -s INT $(jobs -p); wait; exit' SIGINT
trap 'kill -s TERM $(jobs -p); wait; exit' SIGTERM
trap 'kill -s USR1 $(jobs -p); wait' SIGUSR1

pasero-train ${OPTS[@]} &
wait
#!/usr/bin/env bash
# Run per-class evaluation or single-WAV inference on the DGX Spark.
# Executes scripts/evalModel.py inside the training image so all
# dependencies (librosa, torch, etc.) are available without a host install.
#
# Usage — val-set evaluation:
#   bash scripts/evalDGX.sh \
#       --checkpoint checkpoints/best.ckpt \
#       --labelEncoder checkpoints/labelEncoder.json \
#       --valCsv dataset/val.csv \
#       --useCategories [--tuneThresholds]
#
# Usage — single-clip inference:
#   bash scripts/evalDGX.sh \
#       --checkpoint checkpoints/best.ckpt \
#       --labelEncoder checkpoints/labelEncoder.json \
#       --wav dataset/clips/<clip>.wav

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
DATA_DIR="${AIRCRAFT_DATA_DIR:-/mnt/aircraft-data/AircraftData}"
IMAGE="aircraft-audio-training:latest"

docker run --rm \
    --gpus all \
    --ipc host \
    -v "${PROJECT_ROOT}:/home/jdn/Code/AircraftAudioId:ro" \
    -v "${DATA_DIR}/dataset:/home/jdn/Code/AircraftAudioId/dataset:ro" \
    -v "${CHECKPOINT_DIR}:/checkpoints:ro" \
    --entrypoint python \
    "${IMAGE}" \
    /home/jdn/Code/AircraftAudioId/scripts/evalModel.py \
    "$@"

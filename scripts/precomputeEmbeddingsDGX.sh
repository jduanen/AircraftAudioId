#!/usr/bin/env bash
# Pre-compute PANNs embeddings on the DGX Spark and write them alongside the WAV files.
# Run once before training with --backbone panns (and again after adding new clips,
# with --skipExisting).
#
# Usage:
#   bash scripts/precomputeEmbeddingsDGX.sh [--skipExisting]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="aircraft-audio-training:latest"

# panns_inference downloads its ~300 MB CNN14 checkpoint to ~/panns_data on
# first use; mount it from the host so it downloads once, not per container.
mkdir -p /home/jdn/panns_data

docker run --rm \
    --gpus all \
    --ipc host \
    -v "${PROJECT_ROOT}:/home/jdn/Code/AircraftAudioId:ro" \
    -v "${PROJECT_ROOT}/dataset:/home/jdn/Code/AircraftAudioId/dataset" \
    -v "/home/jdn/panns_data:/root/panns_data" \
    --entrypoint python \
    "${IMAGE}" \
    /home/jdn/Code/AircraftAudioId/scripts/precomputeEmbeddings.py \
    --trainCsv dataset/train.csv \
    --valCsv dataset/val.csv \
    "$@"

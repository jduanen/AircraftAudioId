#!/usr/bin/env bash
# Pre-compute mel spectrograms on the DGX Spark and write them alongside the WAV files.
# Run once before training to eliminate per-epoch CPU spectrogram computation.
#
# Usage:
#   bash scripts/precomputeDGX.sh [--workers 16]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="aircraft-audio-training:latest"

docker run --rm \
    --ipc host \
    -v "${PROJECT_ROOT}:/home/jdn/Code/AircraftAudioId:ro" \
    -v "${PROJECT_ROOT}/dataset:/home/jdn/Code/AircraftAudioId/dataset" \
    --entrypoint python \
    "${IMAGE}" \
    /home/jdn/Code/AircraftAudioId/scripts/precomputeSpecs.py \
    --trainCsv dataset/train.csv \
    --valCsv dataset/val.csv \
    "$@"

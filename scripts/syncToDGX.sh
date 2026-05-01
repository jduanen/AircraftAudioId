#!/usr/bin/env bash
# Sync the project to the DGX Spark before training.
# Run this on the recording server, not on the DGX Spark.
#
# Usage:
#   bash scripts/syncToDGX.sh <dgx-hostname-or-ip>
#
# What is synced:
#   src/          — classifier package
#   scripts/      — including trainDGX.sh
#   docker/       — Dockerfile and compose file
#   pyproject.toml
#
# What is NOT synced:
#   dataset/      — lives on the NFS-exported drive; mounted by Docker at training time
#   recordings/   — raw audio (large, not needed for training)
#   checkpoints/  — model output lives on the DGX Spark
#   .git/         — not needed on the training machine
#
# The destination path /home/jdn/Code/AircraftAudioId/ is preserved so
# absolute clip paths in the CSV files remain valid inside the container.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/syncToDGX.sh <dgx-hostname-or-ip>"
    exit 1
fi

DGX_HOST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_PATH="jdn@${DGX_HOST}:/home/jdn/Code/AircraftAudioId/"

echo "Syncing to ${DGX_HOST}:${REMOTE_PATH} ..."

rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='recordings/' \
    --exclude='dataset/' \
    --exclude='checkpoints/' \
    --exclude='*.egg-info/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --filter='protect checkpoints/' \
    "${PROJECT_ROOT}/" \
    "${REMOTE_PATH}"

echo "Sync complete."

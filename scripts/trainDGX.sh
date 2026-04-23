#!/usr/bin/env bash
# Launch Phase 1 training on the DGX Spark.
# Run this ON the DGX Spark (via SSH), after syncing with syncToDGX.sh.
#
# Usage:
#   bash scripts/trainDGX.sh [toolchain.py args...]
#
# Examples:
#   bash scripts/trainDGX.sh --useCategories
#
#   bash scripts/trainDGX.sh \
#       --useCategories \
#       --batchSize 64 \
#       --maxEpochs 100 \
#       --lr 2e-4 \
#       --compile
#
#   # Custom checkpoint directory on the DGX host:
#   CHECKPOINT_DIR=/home/jdn/Data/checkpoints bash scripts/trainDGX.sh --useCategories
#
# All arguments are forwarded verbatim to toolchain.py.
# --trainCsv, --valCsv, and --outputDir are set by docker-compose.training.yml;
# pass them explicitly here to override.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.training.yml"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
mkdir -p "${CHECKPOINT_DIR}"

# Pass extra args through an env var to avoid shell quoting issues at the
# Compose command interpolation boundary.
export TRAINING_ARGS="${*}"

echo "=============================================="
echo "  AircraftAudioId — Phase 1 Training"
echo "  Checkpoints : ${CHECKPOINT_DIR}"
echo "  Extra args  : ${TRAINING_ARGS:-<none>}"
echo "=============================================="

docker compose \
    --file "${COMPOSE_FILE}" \
    --project-name aircraft-training \
    build --pull

docker compose \
    --file "${COMPOSE_FILE}" \
    --project-name aircraft-training \
    run --rm \
    aircraft-training \
    ${TRAINING_ARGS:+${TRAINING_ARGS}}

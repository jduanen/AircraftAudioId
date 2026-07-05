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

# Archive any checkpoints left over from a previous run. Lightning's
# save_top_k only manages the top-3 within one run, not across separate
# trainDGX.sh invocations, so old .ckpt files otherwise accumulate
# indefinitely. That's more than just clutter: evalBestDGX.sh picks the
# highest-val_f1 checkpoint that loads successfully, and an old checkpoint
# from an unchanged architecture loads just fine even though it was trained
# on different data — silently returning a stale result with no error at
# all. Archiving (not deleting) keeps old checkpoints available for manual
# inspection while guaranteeing the checkpoint dir only ever contains the
# current run's output.
shopt -s nullglob
existingCkpts=("${CHECKPOINT_DIR}"/*.ckpt)
shopt -u nullglob
if [[ ${#existingCkpts[@]} -gt 0 ]]; then
    archiveDir="${CHECKPOINT_DIR}/archive/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${archiveDir}"
    mv "${existingCkpts[@]}" "${archiveDir}/"
    echo "Archived ${#existingCkpts[@]} checkpoint(s) from a previous run to ${archiveDir}"
fi

# --build flag forces image rebuild; strip it before forwarding to toolchain.py.
BUILD=false
REMAINING_ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--build" ]] && BUILD=true || REMAINING_ARGS+=("$arg")
done
export TRAINING_ARGS="${REMAINING_ARGS[*]:-}"

echo "=============================================="
echo "  AircraftAudioId — Phase 1 Training"
echo "  Checkpoints : ${CHECKPOINT_DIR}"
echo "  Extra args  : ${TRAINING_ARGS:-<none>}"
echo "=============================================="

if $BUILD || ! docker image inspect aircraft-audio-training:latest >/dev/null 2>&1; then
    bash "${SCRIPT_DIR}/buildImageDGX.sh"
fi

docker compose \
    --file "${COMPOSE_FILE}" \
    --project-name aircraft-training \
    run --rm \
    aircraft-training

#!/usr/bin/env bash
# Try checkpoints in descending val_f1 order (parsed from filename) and run
# evalDGX.sh on the first one that loads successfully. Run this ON the DGX
# Spark, after a training run has produced checkpoints named
# epoch=NN-val_f1=D.DDD.ckpt (the format written by toolchain.py's
# ModelCheckpoint callback).
#
# Checkpoints from every past run accumulate in the same directory (Lightning's
# save_top_k only manages the top-3 within one run, not across separate
# trainDGX.sh invocations). A leftover checkpoint from a since-changed model
# architecture (e.g. a different conv1 channel count) can still have the
# highest val_f1 in the directory but fails to load. There's no reliable way
# to tell "which run a file belongs to" from a timestamp when runs happen
# back-to-back (a fixed time window was tried and doesn't hold up), so instead
# this just tries the best-scoring checkpoint, and on load failure falls
# through to the next-best, and so on.
#
# Usage:
#   bash scripts/evalBestDGX.sh [evalDGX.sh args...]
#
# Example:
#   bash scripts/evalBestDGX.sh \
#       --labelEncoder /checkpoints/labelEncoder.json \
#       --valCsv dataset/val.csv --useCategories \
#       --tuneThresholds --saveThresholds /checkpoints/thresholds.json
#
# All arguments are forwarded verbatim to evalDGX.sh; do not pass --checkpoint,
# it is filled in automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"

shopt -s nullglob
candidates=("${CHECKPOINT_DIR}"/epoch=*-val_f1=*.ckpt)
shopt -u nullglob

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No checkpoints matching 'epoch=*-val_f1=*.ckpt' found in ${CHECKPOINT_DIR}" >&2
    exit 1
fi

mapfile -t sorted < <(
    for f in "${candidates[@]}"; do
        score="${f##*val_f1=}"
        score="${score%.ckpt}"
        echo "${score} ${f}"
    done | sort -rn -k1,1 | cut -d' ' -f2-
)

for f in "${sorted[@]}"; do
    bestName="$(basename "$f")"
    echo "Trying checkpoint: ${bestName}"
    if bash "${SCRIPT_DIR}/evalDGX.sh" --checkpoint "/checkpoints/${bestName}" "$@"; then
        exit 0
    fi
    echo "  Failed to load '${bestName}' (likely architecture mismatch with a previous run) — trying next-best." >&2
done

echo "No checkpoint in ${CHECKPOINT_DIR} could be loaded successfully." >&2
exit 1

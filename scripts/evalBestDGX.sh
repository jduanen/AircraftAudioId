#!/usr/bin/env bash
# Find the checkpoint with the highest val_f1 (parsed from its filename) and
# run evalDGX.sh on it. Run this ON the DGX Spark, after a training run has
# produced checkpoints named epoch=NN-val_f1=D.DDD.ckpt (the format written by
# toolchain.py's ModelCheckpoint callback).
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

best=""
bestScore="-1"
for f in "${candidates[@]}"; do
    score="${f##*val_f1=}"
    score="${score%.ckpt}"
    if awk -v a="$score" -v b="$bestScore" 'BEGIN{exit !(a>b)}'; then
        bestScore="$score"
        best="$f"
    fi
done

bestName="$(basename "$best")"
echo "Best checkpoint: ${bestName}  (val_f1=${bestScore})"

bash "${SCRIPT_DIR}/evalDGX.sh" --checkpoint "/checkpoints/${bestName}" "$@"

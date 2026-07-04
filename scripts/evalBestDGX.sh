#!/usr/bin/env bash
# Find the checkpoint with the highest val_f1 (parsed from its filename) from
# the MOST RECENT training run and run evalDGX.sh on it. Run this ON the DGX
# Spark, after a training run has produced checkpoints named
# epoch=NN-val_f1=D.DDD.ckpt (the format written by toolchain.py's
# ModelCheckpoint callback).
#
# Checkpoints from every past run accumulate in the same directory (Lightning's
# save_top_k only manages the top-3 within one run, not across separate
# trainDGX.sh invocations). Picking the all-time-best val_f1 can select a
# leftover checkpoint from a since-changed model architecture (e.g. a
# different conv1 channel count), which fails to load or silently loads wrong
# weights. Restrict candidates to those written within RUN_WINDOW_SECS of the
# most recently modified checkpoint to isolate the latest run.
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
RUN_WINDOW_SECS="${RUN_WINDOW_SECS:-3600}"   # generous: a full trainDGX.sh run finishes in minutes

shopt -s nullglob
candidates=("${CHECKPOINT_DIR}"/epoch=*-val_f1=*.ckpt)
shopt -u nullglob

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No checkpoints matching 'epoch=*-val_f1=*.ckpt' found in ${CHECKPOINT_DIR}" >&2
    exit 1
fi

newestMtime=-1
for f in "${candidates[@]}"; do
    mtime=$(stat -c %Y "$f")
    (( mtime > newestMtime )) && newestMtime=$mtime
done

best=""
bestScore="-1"
nExcluded=0
for f in "${candidates[@]}"; do
    mtime=$(stat -c %Y "$f")
    if (( newestMtime - mtime > RUN_WINDOW_SECS )); then
        nExcluded=$((nExcluded + 1))
        continue   # too old, belongs to a previous (possibly incompatible) run
    fi
    score="${f##*val_f1=}"
    score="${score%.ckpt}"
    if awk -v a="$score" -v b="$bestScore" 'BEGIN{exit !(a>b)}'; then
        bestScore="$score"
        best="$f"
    fi
done

if [[ $nExcluded -gt 0 ]]; then
    echo "Ignored ${nExcluded} checkpoint(s) older than ${RUN_WINDOW_SECS}s (from a previous run)."
fi

bestName="$(basename "$best")"
echo "Best checkpoint: ${bestName}  (val_f1=${bestScore})"

bash "${SCRIPT_DIR}/evalDGX.sh" --checkpoint "/checkpoints/${bestName}" "$@"

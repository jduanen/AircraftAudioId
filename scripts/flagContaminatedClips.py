#!/usr/bin/env python3
"""
Detect background-noise contamination in dataset clips and flag them for
exclusion: wind, yard machinery (lawn mowers, chainsaws), birds, crowd noise
(cheering/clapping/applause), and near-silent clips.

Wind/machinery/birds/crowd detection uses a pretrained PANNs (AudioSet) audio
tagger -- no training data or fine-tuning needed, just inference against
AudioSet's existing label set. Near-silence is checked directly from clipRms
(same threshold convention as evalClipQuality.py), no audio read required.

"Machinery" deliberately checks only Lawn mower/Chainsaw -- AudioSet's generic
Engine/Aircraft engine/Jet engine labels are excluded because real aircraft
clips are expected to trigger those too.

Output is a CSV with filepath/type_categories/reason/score columns: the
filepath+type_categories columns make it directly viewable with vizSpecs.py,
and filepath alone is all excludeFlaggedClips.py needs, so the two downstream
steps work unmodified:

    python scripts/vizSpecs.py --csv flagged.csv --play          # spot-check
    python scripts/excludeFlaggedClips.py --flagged flagged.csv --datasetDir dataset

Usage:
    python scripts/flagContaminatedClips.py \\
        --csv dataset/train.csv --csv dataset/val.csv \\
        --output flagged_contamination.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

PANNS_SAMPLE_RATE = 32000

GROUPS = {
    "wind":      ["Wind", "Wind noise (microphone)", "Rustling leaves"],
    "machinery": ["Lawn mower", "Chainsaw"],
    "birds":     ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"],
    "crowd":     ["Cheering", "Clapping", "Applause", "Crowd"],
}


def _rmsDb(clipRms) -> float:
    return 20.0 * np.log10(max(float(clipRms) if clipRms is not None else 0.0, 1e-9))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Flag wind/machinery/bird/crowd/silence-contaminated clips for exclusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv", action="append", required=True,
                    help="Dataset CSV to scan (filepath + type_categories/vehicle_types columns). "
                         "Repeatable, e.g. --csv dataset/train.csv --csv dataset/val.csv.")
    p.add_argument("--output", required=True, help="Path to write the flagged-clips CSV.")
    p.add_argument("--threshold", type=float, default=0.15,
                    help="AudioSet tag probability above which a clip is flagged for that group "
                         "(default: 0.15).")
    p.add_argument("--quietThresholdDb", type=float, default=-55.0,
                    help="Flag clips with clipRms below this dBFS as near-silent, no audio read "
                         "required (default: -55, matching evalClipQuality.py).")
    p.add_argument("--batchSize", type=int, default=32, help="Inference batch size (default: 32).")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="Falls back to cpu automatically if cuda is unavailable.")
    args = p.parse_args()

    dfs = [pd.read_csv(c) for c in args.csv]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="filepath").reset_index(drop=True)
    labelCol = "type_categories" if "type_categories" in df.columns else "vehicle_types"
    print(f"Loaded {len(df)} unique clip(s) from {len(args.csv)} CSV(s)")

    from panns_inference import AudioTagging, labels as pannsLabels
    groupIndices = {g: [pannsLabels.index(lbl) for lbl in labs] for g, labs in GROUPS.items()}
    at = AudioTagging(checkpoint_path=None, device=args.device)

    flagged: list[dict] = []
    batchAudio: list[np.ndarray] = []
    batchRows: list[pd.Series] = []

    def flushBatch() -> None:
        if not batchAudio:
            return
        maxLen = max(len(a) for a in batchAudio)
        padded = np.stack([np.pad(a, (0, maxLen - len(a))) for a in batchAudio])
        clipwise, _ = at.inference(padded)
        for j, row in enumerate(batchRows):
            reasons, bestScore = [], 0.0
            for g, idxs in groupIndices.items():
                score = float(clipwise[j, idxs].max())
                if score >= args.threshold:
                    reasons.append(g)
                    bestScore = max(bestScore, score)
            if reasons:
                flagged.append({"filepath": row["filepath"], labelCol: row[labelCol],
                                 "reason": ",".join(reasons), "score": round(bestScore, 3)})
        batchAudio.clear()
        batchRows.clear()

    hasRms = "clipRms" in df.columns
    n = len(df)
    for i, row in df.iterrows():
        if hasRms and _rmsDb(row["clipRms"]) < args.quietThresholdDb:
            flagged.append({"filepath": row["filepath"], labelCol: row[labelCol],
                             "reason": "quiet", "score": round(_rmsDb(row["clipRms"]), 1)})
        else:
            try:
                y, _ = librosa.load(row["filepath"], sr=PANNS_SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  [!] failed to load {row['filepath']}: {e}")
                y = None
            if y is not None:
                batchAudio.append(y)
                batchRows.append(row)
                if len(batchAudio) >= args.batchSize:
                    flushBatch()
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{n} clips processed")
    flushBatch()

    print(f"\nFlagged {len(flagged)}/{n} clip(s):")
    from collections import Counter
    reasonCounts = Counter(r for f in flagged for r in f["reason"].split(","))
    for reason, count in reasonCounts.most_common():
        print(f"  {reason:<12} {count}")

    if not flagged:
        print("\nNothing to write.")
        return

    pd.DataFrame(flagged).sort_values("score", ascending=False).to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    print("Review before excluding:")
    print(f"  python scripts/vizSpecs.py --csv {args.output} --play")
    print("Then exclude:")
    print(f"  python scripts/excludeFlaggedClips.py --flagged {args.output} --datasetDir <dir>")


if __name__ == "__main__":
    main()

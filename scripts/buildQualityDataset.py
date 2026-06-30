#!/usr/bin/env python3
"""
Build a quality-filtered training dataset by keeping the best N clips per category.

For each coarse category in the dataset, clips are ranked by audio quality and
the top N are retained. The filtered dataset is re-split into train/val CSVs by
flyover event to prevent data leakage. Null/background samples are always kept.

Ranking modes:
  Fast (default) — ranks by RMS dBFS from the clipRms column. No audio reads.
  Deep (--deepAnalysis) — ranks by composite quality score across 7 audio metrics.
    Requires soundfile + librosa.

Usage:
  # Keep best 500 clips per category, rank by RMS (no audio reads):
  python scripts/buildQualityDataset.py \\
      --datasetCsv dataset/dataset.csv \\
      --outputDir dataset_best/ \\
      --bestN 500

  # Keep best 500 clips per category, rank by composite score:
  python scripts/buildQualityDataset.py \\
      --datasetCsv dataset/dataset.csv \\
      --outputDir dataset_best/ \\
      --bestN 500 --deepAnalysis
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from aircraftAudio.dataset.clipExport import splitByEvent
from evalClipQuality import _rmsDb, _compositeScore, _runDeepAnalysis


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a quality-filtered training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--datasetCsv", type=Path, required=True,
        help="Full dataset CSV produced by buildDataset.py.",
    )
    p.add_argument(
        "--outputDir", type=Path, required=True,
        help="Directory to write train.csv and val.csv.",
    )
    p.add_argument(
        "--bestN", type=int, required=True,
        help="Keep the best N clips per category. Categories with fewer clips keep all.",
    )
    p.add_argument(
        "--deepAnalysis", action="store_true",
        help="Rank by composite quality score (7 audio metrics). "
             "Without this flag, ranks by RMS only (no audio reads).",
    )
    p.add_argument(
        "--trainFrac", type=float, default=0.8,
        help="Fraction of flyover events for the training split (default: 0.8).",
    )
    args = p.parse_args()

    if not args.datasetCsv.exists():
        sys.exit(f"CSV not found: {args.datasetCsv}")

    df = pd.read_csv(args.datasetCsv)
    print(f"\nLoaded {len(df)} clips from {args.datasetCsv}")

    # Null/background samples are always kept — separate them before ranking
    nullMask = df.get(
        "isNullSample", pd.Series(False, index=df.index)
    ).fillna(False).astype(bool)
    nullDf = df[nullMask]
    clipDf = df[~nullMask].reset_index(drop=True)

    # Parse categories once (exclude "unknown" — those clips don't rank for any class)
    parsed = [json.loads(raw) for raw in clipDf["type_categories"]]
    allCategories = sorted({cat for cats in parsed for cat in cats if cat != "unknown"})

    print(f"Categories: {', '.join(allCategories)}")
    print(f"Null samples: {len(nullDf)}  Aircraft clips: {len(clipDf)}")

    # Deep mode: analyse all aircraft clips once, build filepath → composite score map
    scores = None
    if args.deepAnalysis:
        print(f"\nRunning deep analysis on {len(clipDf)} clips ...")
        results, nMissing = _runDeepAnalysis(clipDf)
        if nMissing:
            print(f"  [!] {nMissing} clips missing or unreadable — skipped")
        scores = {r["filepath"]: _compositeScore(r) for r in results}

    # Select best N per category, accumulate union of kept filepaths
    keptFilepaths: set[str] = set()
    rankMode = "composite score" if args.deepAnalysis else "RMS"
    print(f"\n  Best {args.bestN} per category  (ranked by {rankMode}):")
    print(f"  {'Category':<22}  {'Available':>10}  {'Kept':>6}")
    print("  " + "─" * 42)

    for cat in allCategories:
        catMask = pd.Series([cat in cats for cats in parsed], index=clipDf.index)
        catDf = clipDf[catMask]

        if scores is not None:
            ranked = catDf.assign(
                _score=catDf["filepath"].map(scores).fillna(0.0)
            ).sort_values("_score", ascending=False)
        else:
            ranked = catDf.assign(
                _score=catDf["clipRms"].apply(_rmsDb)
            ).sort_values("_score", ascending=False)

        best = ranked["filepath"].tolist()[:args.bestN]
        keptFilepaths.update(best)
        print(f"  {cat:<22}  {len(catDf):>10}  {len(best):>6}")

    # Build filtered dataset: best aircraft clips + all null samples
    filteredDf = pd.concat([
        clipDf[clipDf["filepath"].isin(keptFilepaths)],
        nullDf,
    ], ignore_index=True)

    print(f"\nFiltered: {len(df)} → {len(filteredDf)} clips "
          f"({len(nullDf)} null samples + "
          f"{len(filteredDf) - len(nullDf)} aircraft clips retained)")

    # Re-split by recording event to prevent data leakage
    trainDf, valDf = splitByEvent(filteredDf, trainFrac=args.trainFrac)

    args.outputDir.mkdir(parents=True, exist_ok=True)
    trainDf.to_csv(args.outputDir / "train.csv", index=False)
    valDf.to_csv(args.outputDir / "val.csv", index=False)

    print(f"Split: {len(trainDf)} train ({trainDf['recordingId'].nunique()} events) / "
          f"{len(valDf)} val ({valDf['recordingId'].nunique()} events)")
    print(f"Output: {args.outputDir}/train.csv, {args.outputDir}/val.csv\n")


if __name__ == "__main__":
    main()

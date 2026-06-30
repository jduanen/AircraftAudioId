#!/usr/bin/env python3
"""
Extract clips from raw recordings and keep only the best N per category.

Combines clip extraction (buildDataset.py) with quality filtering
(buildQualityDataset.py) in a single pass.  For each coarse category the clips
are ranked by audio quality and the top N are retained.  The result is written
as train.csv / val.csv split by flyover event to prevent data leakage.
Null/background samples are always kept.

Ranking modes:
  Fast (default) — ranks by RMS dBFS from the clipRms column.  No extra audio
    reads beyond what clip extraction already requires.
  Deep (--deepAnalysis) — ranks by composite quality score across 7 audio
    metrics.  Reads each clip WAV again after extraction; requires librosa.

Usage:
  # Fast: extract all clips, keep best 500 per category by RMS:
  python scripts/buildQualityDatasetFromRecordings.py \\
      --recordingsDir AircraftData/recordings \\
      --outputDir dataset_best/ \\
      --bestN 500 \\
      --faaDatabaseDir AircraftData/ReleasableAircraft

  # Deep: rank by composite quality score:
  python scripts/buildQualityDatasetFromRecordings.py \\
      --recordingsDir AircraftData/recordings \\
      --outputDir dataset_best/ \\
      --bestN 500 \\
      --faaDatabaseDir AircraftData/ReleasableAircraft \\
      --deepAnalysis
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from aircraftAudio.dataset.clipExport import buildClipDataset, splitByEvent
from evalClipQuality import _rmsDb, _compositeScore, _runDeepAnalysis


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract clips from raw recordings and keep the best N per category.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--recordingsDir", type=Path, required=True,
        help="Directory produced by record.py (contains audio/ and metadata/).",
    )
    p.add_argument(
        "--outputDir", type=Path, required=True,
        help="Directory to write clips/, dataset.csv, train.csv, and val.csv.",
    )
    p.add_argument(
        "--bestN", type=int, required=True,
        help="Keep the best N clips per category. Categories with fewer clips keep all.",
    )
    p.add_argument(
        "--faaDatabaseDir", type=Path, default=None,
        help="Path to unzipped FAA ReleasableAircraft directory (strongly recommended). "
             "Without it, type_categories falls back to a keyword heuristic that "
             "misclassifies turboprop Pipers, Malibu variants, and others.",
    )
    p.add_argument(
        "--deepAnalysis", action="store_true",
        help="Rank by composite quality score (7 audio metrics). "
             "Without this flag, ranks by RMS only.",
    )
    p.add_argument(
        "--clipSecs", type=float, default=5.0,
        help="Duration of each extracted clip in seconds (default: 5.0).",
    )
    p.add_argument(
        "--trainFrac", type=float, default=0.8,
        help="Fraction of flyover events for the training split (default: 0.8).",
    )
    p.add_argument(
        "--maxDistanceKm", type=float, default=None,
        help="Skip states where the aircraft is farther than this (km).",
    )
    p.add_argument(
        "--dropUnknown", action="store_true",
        help="Exclude clips whose type_categories are entirely 'unknown'. "
             "Null/background clips are kept regardless.",
    )
    p.add_argument(
        "--autoCorrectClock", action="store_true",
        help="Estimate per-recording clock skew from state timestamps. "
             "Recommended for existing recordings with inconsistent skew.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes for clip extraction (default: 1).",
    )
    args = p.parse_args()

    if not args.recordingsDir.exists():
        sys.exit(f"Recordings directory not found: {args.recordingsDir}")

    if args.faaDatabaseDir is None:
        print(
            "\n[WARNING] --faaDatabaseDir not provided. type_categories will use a "
            "keyword heuristic that misclassifies common aircraft. Pass "
            "--faaDatabaseDir for authoritative categories.\n"
        )

    # ── Step 1: extract all clips from raw recordings ─────────────────────────
    print(f"\nExtracting clips from {args.recordingsDir} ...")
    df = buildClipDataset(
        recordingsDir=args.recordingsDir,
        outputDir=args.outputDir,
        clipSecs=args.clipSecs,
        maxDistanceKm=args.maxDistanceKm,
        faaDatabaseDir=args.faaDatabaseDir,
        autoCorrectClock=args.autoCorrectClock,
        dropUnknown=args.dropUnknown,
        workers=args.workers,
    )

    if df.empty:
        sys.exit("No clips extracted — check that recordings were made with the updated recorder.py.")

    print(f"Extracted {len(df)} clips total.")

    # ── Step 2: quality filtering ─────────────────────────────────────────────
    nullMask = df.get(
        "isNullSample", pd.Series(False, index=df.index)
    ).fillna(False).astype(bool)
    nullDf = df[nullMask]
    clipDf = df[~nullMask].reset_index(drop=True)

    parsed = [json.loads(raw) for raw in clipDf["type_categories"]]
    allCategories = sorted({cat for cats in parsed for cat in cats if cat != "unknown"})

    print(f"Categories: {', '.join(allCategories)}")
    print(f"Null samples: {len(nullDf)}  Aircraft clips: {len(clipDf)}")

    scores = None
    if args.deepAnalysis:
        print(f"\nRunning deep analysis on {len(clipDf)} clips ...")
        results, nMissing = _runDeepAnalysis(clipDf)
        if nMissing:
            print(f"  [!] {nMissing} clips missing or unreadable — skipped")
        scores = {r["filepath"]: _compositeScore(r) for r in results}

    rankMode = "composite score" if args.deepAnalysis else "RMS"
    print(f"\n  Best {args.bestN} per category  (ranked by {rankMode}):")
    print(f"  {'Category':<22}  {'Available':>10}  {'Kept':>6}")
    print("  " + "─" * 42)

    keptFilepaths: set[str] = set()
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

    filteredDf = pd.concat([
        clipDf[clipDf["filepath"].isin(keptFilepaths)],
        nullDf,
    ], ignore_index=True)

    print(f"\nFiltered: {len(df)} → {len(filteredDf)} clips "
          f"({len(nullDf)} null samples + "
          f"{len(filteredDf) - len(nullDf)} aircraft clips retained)")

    # ── Step 3: split and write ───────────────────────────────────────────────
    trainDf, valDf = splitByEvent(filteredDf, trainFrac=args.trainFrac)

    args.outputDir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outputDir / "dataset.csv", index=False)
    trainDf.to_csv(args.outputDir / "train.csv", index=False)
    valDf.to_csv(args.outputDir / "val.csv", index=False)

    print(f"Split: {len(trainDf)} train ({trainDf['recordingId'].nunique()} events) / "
          f"{len(valDf)} val ({valDf['recordingId'].nunique()} events)")
    print(f"Output: {args.outputDir}/\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Incrementally add newly-recorded clips to an existing quality-filtered dataset.

Finds recordings that are not yet in the dataset's raw pool (dataset.csv) and
whose session has been confirmed complete by a session_<timestamp>.json summary
file (written by recorder.py on SIGUSR1 or normal shutdown — see
src/aircraftAudio/record/recorder.py). Recordings more recent than the latest
session summary are still possibly mid-flight and are left for the next run.

Extracts clips from those recordings, scores each one by audio quality, and
appends to train.csv/val.csv only the clips that are at least as good as the
current worst clip already kept for that category — the same quality bar
buildQualityDataset*.py applied when the dataset was originally built. A clip
qualifies if it meets the bar for *any* of its categories; a brand-new
category with no clips yet in the dataset has no bar and any clip qualifies.
Null/background clips are always kept, matching buildDataset.py convention.

Ranking modes:
  Fast (default) — compares RMS dBFS. No extra audio reads: the new clips'
    RMS is computed during extraction, and the currently-kept clips' RMS is
    already stored in train.csv/val.csv.
  Deep (--deepAnalysis) — compares composite quality score across 7 audio
    metrics (see evalClipQuality.py). Re-reads every clip currently kept in
    train.csv/val.csv once (to establish the per-category bar) plus every
    newly extracted clip.

Usage:
  python scripts/addNewRecordings.py \\
      --recordingsDir /mnt/nvme/aircraft_data/recordings \\
      --datasetDir /mnt/nvme/aircraft_data/datasets/dataset_best3000 \\
      --faaDatabaseDir data/ReleasableAircraft

  python scripts/addNewRecordings.py \\
      --recordingsDir /mnt/nvme/aircraft_data/recordings \\
      --datasetDir /mnt/nvme/aircraft_data/datasets/dataset_best3000 \\
      --faaDatabaseDir data/ReleasableAircraft \\
      --deepAnalysis
"""

import sys
import json
import tempfile
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from aircraftAudio.dataset.clipExport import buildClipDataset, splitByEvent
from evalClipQuality import _rmsDb, _compositeScore, _runDeepAnalysis


def _sessionCutoff(recordingsDir: Path) -> datetime:
    """Latest sessionEnd timestamp across all session_<timestamp>.json summaries."""
    sessionFiles = sorted(recordingsDir.glob("session_*.json"))
    if not sessionFiles:
        sys.exit(
            f"No session_*.json summary files found in {recordingsDir}. "
            "Send SIGUSR1 to a running record.py (or let it exit normally) "
            "before running this script — see scripts/record.py."
        )
    ends = []
    for f in sessionFiles:
        with open(f) as fh:
            ends.append(datetime.fromisoformat(json.load(fh)["sessionEnd"]))
    return max(ends)


def _findNewRecordings(recordingsDir: Path, existingIds: set, cutoff: datetime) -> list[Path]:
    """Metadata files not already in the dataset, confirmed by a session summary."""
    metaPaths = sorted((recordingsDir / "metadata").glob("*.json"))
    new = []
    for metaPath in metaPaths:
        recordingId = metaPath.stem
        if recordingId in existingIds:
            continue
        try:
            ts = datetime.strptime(recordingId[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if ts <= cutoff:
            new.append(metaPath)
    return new


def _stageRecordings(metaPaths: list[Path], recordingsDir: Path, stagingDir: Path) -> None:
    """Symlink the given recordings into a scratch dir shaped like recordingsDir."""
    (stagingDir / "metadata").mkdir(parents=True)
    (stagingDir / "audio").mkdir(parents=True)
    for metaPath in metaPaths:
        wavPath = recordingsDir / "audio" / f"{metaPath.stem}.wav"
        if not wavPath.exists():
            continue
        (stagingDir / "metadata" / metaPath.name).symlink_to(metaPath.resolve())
        (stagingDir / "audio" / wavPath.name).symlink_to(wavPath.resolve())


def _categoriesOf(row) -> list[str]:
    return [c for c in json.loads(row["type_categories"]) if c != "unknown"]


def _isNull(row) -> bool:
    return row["flightPhase"] == "null"


def _perCategoryThresholds(keptDf: pd.DataFrame, deep: bool) -> dict[str, float]:
    """
    Per-category minimum quality score among currently-kept clips — the bar a
    new clip must meet or beat. Higher score = better for both RMS dBFS and
    the composite score, so `min()` over each category's kept clips gives the
    "at least as good as the worst clip already in the dataset" bar.
    """
    aircraftDf = keptDf[~keptDf.apply(_isNull, axis=1)].reset_index(drop=True)

    if deep:
        print(f"\nRunning deep analysis on {len(aircraftDf)} currently-kept clips "
              f"(one-time threshold pass) ...")
        results, nMissing = _runDeepAnalysis(aircraftDf)
        if nMissing:
            print(f"  [!] {nMissing} currently-kept clips missing or unreadable — excluded from threshold")
        scoreByFilepath = {r["filepath"]: _compositeScore(r) for r in results}
        scoreOf = lambda row: scoreByFilepath.get(str(Path(row["filepath"])))
    else:
        scoreOf = lambda row: _rmsDb(row["clipRms"])

    thresholds: dict[str, float] = {}
    for _, row in aircraftDf.iterrows():
        score = scoreOf(row)
        if score is None:
            continue
        for cat in _categoriesOf(row):
            thresholds[cat] = min(thresholds.get(cat, score), score)
    return thresholds


def main() -> None:
    p = argparse.ArgumentParser(
        description="Add newly-recorded clips to an existing dataset, gated by per-category quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--recordingsDir", type=Path, required=True,
        help="Directory produced by record.py (contains audio/, metadata/, session_*.json).",
    )
    p.add_argument(
        "--datasetDir", type=Path, required=True,
        help="Existing dataset directory containing dataset.csv, train.csv, val.csv.",
    )
    p.add_argument(
        "--faaDatabaseDir", type=Path, default=None,
        help="Path to unzipped FAA ReleasableAircraft directory (strongly recommended).",
    )
    p.add_argument(
        "--deepAnalysis", action="store_true",
        help="Rank by composite quality score (7 audio metrics) instead of RMS only. "
             "Re-reads every currently-kept clip once to establish the threshold.",
    )
    p.add_argument("--clipSecs", type=float, default=5.0,
                    help="Duration of each extracted clip in seconds (default: 5.0).")
    p.add_argument("--trainFrac", type=float, default=0.8,
                    help="Fraction of newly-added flyover events assigned to train (default: 0.8).")
    p.add_argument("--maxDistanceKm", type=float, default=None,
                    help="Skip states where the aircraft is farther than this (km).")
    p.add_argument("--dropUnknown", action="store_true",
                    help="Exclude clips whose type_categories are entirely 'unknown' from the raw pool.")
    p.add_argument("--autoCorrectClock", action="store_true",
                    help="Estimate per-recording clock skew from state timestamps.")
    p.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes for clip extraction (default: 1).")
    args = p.parse_args()

    if not (args.datasetDir / "train.csv").exists() or not (args.datasetDir / "val.csv").exists():
        sys.exit(f"{args.datasetDir} must already contain train.csv and val.csv — "
                  f"run buildQualityDataset*.py first to establish the initial dataset.")
    if not args.recordingsDir.exists():
        sys.exit(f"Recordings directory not found: {args.recordingsDir}")
    if args.faaDatabaseDir is None:
        print("\n[WARNING] --faaDatabaseDir not provided. type_categories will use a "
              "keyword heuristic that misclassifies common aircraft.\n")

    datasetCsv = args.datasetDir / "dataset.csv"
    existingIds = set()
    if datasetCsv.exists():
        existingIds = set(pd.read_csv(datasetCsv, usecols=["recordingId"])["recordingId"])

    cutoff = _sessionCutoff(args.recordingsDir)
    print(f"Session cutoff (latest confirmed session): {cutoff.isoformat()}")

    newMetaPaths = _findNewRecordings(args.recordingsDir, existingIds, cutoff)
    if not newMetaPaths:
        print("No new, session-confirmed recordings found. Nothing to do.")
        return
    print(f"Found {len(newMetaPaths)} new recording(s) to consider.")

    with tempfile.TemporaryDirectory(prefix="addNewRecordings_") as stagingDirName:
        stagingDir = Path(stagingDirName)
        _stageRecordings(newMetaPaths, args.recordingsDir, stagingDir)
        mergedDf = buildClipDataset(
            recordingsDir=stagingDir,
            outputDir=args.datasetDir,
            clipSecs=args.clipSecs,
            maxDistanceKm=args.maxDistanceKm,
            faaDatabaseDir=args.faaDatabaseDir,
            autoCorrectClock=args.autoCorrectClock,
            dropUnknown=args.dropUnknown,
            workers=args.workers,
            skipExisting=True,
        )

    newIds = {p.stem for p in newMetaPaths}
    candidatesDf = mergedDf[mergedDf["recordingId"].isin(newIds)].reset_index(drop=True)
    if candidatesDf.empty:
        print("No clips were extracted from the new recordings (silent audio or no alignment).")
        return
    print(f"Extracted {len(candidatesDf)} candidate clip(s) from the new recordings.")

    trainDf = pd.read_csv(args.datasetDir / "train.csv")
    valDf = pd.read_csv(args.datasetDir / "val.csv")
    keptDf = pd.concat([trainDf, valDf], ignore_index=True)

    thresholds = _perCategoryThresholds(keptDf, deep=args.deepAnalysis)
    rankMode = "composite score" if args.deepAnalysis else "RMS dBFS"

    nullRows = candidatesDf[candidatesDf["flightPhase"] == "null"]
    aircraftRows = candidatesDf[candidatesDf["flightPhase"] != "null"].reset_index(drop=True)

    if args.deepAnalysis and not aircraftRows.empty:
        print(f"\nRunning deep analysis on {len(aircraftRows)} new candidate clip(s) ...")
        results, nMissing = _runDeepAnalysis(aircraftRows)
        if nMissing:
            print(f"  [!] {nMissing} new clips missing or unreadable — excluded")
        scoreByFilepath = {r["filepath"]: _compositeScore(r) for r in results}
        scoreOf = lambda row: scoreByFilepath.get(str(Path(row["filepath"])))
    else:
        scoreOf = lambda row: _rmsDb(row["clipRms"])

    availCounts: dict[str, int] = {}
    keptCounts: dict[str, int] = {}
    keepMask = []
    for _, row in aircraftRows.iterrows():
        cats = _categoriesOf(row)
        for cat in cats:
            availCounts[cat] = availCounts.get(cat, 0) + 1
        score = scoreOf(row)
        qualifies = score is not None and bool(cats) and any(
            score >= thresholds.get(cat, float("-inf")) for cat in cats
        )
        keepMask.append(qualifies)
        if qualifies:
            for cat in cats:
                keptCounts[cat] = keptCounts.get(cat, 0) + 1

    keptAircraftDf = aircraftRows[pd.Series(keepMask, index=aircraftRows.index)]

    print(f"\n  New clips passing the per-category quality bar (ranked by {rankMode}):")
    print(f"  {'Category':<22}  {'Threshold':>10}  {'Candidates':>10}  {'Kept':>6}")
    print("  " + "─" * 56)
    for cat in sorted(set(availCounts) | set(thresholds)):
        thr = thresholds.get(cat)
        thrStr = f"{thr:.3f}" if thr is not None else "(new)"
        print(f"  {cat:<22}  {thrStr:>10}  {availCounts.get(cat, 0):>10}  {keptCounts.get(cat, 0):>6}")

    addedDf = pd.concat([keptAircraftDf, nullRows], ignore_index=True)
    print(f"\nAdding {len(addedDf)} clip(s) to the dataset "
          f"({len(keptAircraftDf)} aircraft + {len(nullRows)} null).")

    if addedDf.empty:
        print("Nothing met the quality bar — train.csv/val.csv left unchanged.")
        return

    newTrainDf, newValDf = splitByEvent(addedDf, trainFrac=args.trainFrac)
    trainDf = pd.concat([trainDf, newTrainDf], ignore_index=True)
    valDf = pd.concat([valDf, newValDf], ignore_index=True)
    trainDf.to_csv(args.datasetDir / "train.csv", index=False)
    valDf.to_csv(args.datasetDir / "val.csv", index=False)

    print(f"train.csv: +{len(newTrainDf)} → {len(trainDf)}   "
          f"val.csv: +{len(newValDf)} → {len(valDf)}")


if __name__ == "__main__":
    main()

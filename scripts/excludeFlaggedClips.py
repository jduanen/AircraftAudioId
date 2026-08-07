#!/usr/bin/env python3
"""
Remove flagged clips (from vizSpecs.py --flagOutput) from a dataset's CSVs.

Removes any row whose filepath appears in the flagged CSV from dataset.csv,
train.csv, and val.csv (whichever are present in --datasetDir). Each modified
file is backed up first as backup_<timestamp>_<name>.csv.

Excluding from dataset.csv (the raw pool) matters even though training only
reads train.csv/val.csv -- otherwise a future buildQualityDataset*.py or
addNewRecordings.py re-rank could re-select the same flagged clips right
back in.

Note: the flagged CSV's filepaths must match the prefix used by the target
--datasetDir (e.g. flagging against dataset/train.csv in the repo produces
local paths; flagging against the master dataset dir on /mnt/nvme produces
those paths instead). Run vizSpecs.py --flagOutput against whichever copy's
CSVs you intend to filter, or run this script once per copy.

Usage:
    python scripts/excludeFlaggedClips.py \\
        --flagged bad_helicopter.csv \\
        --datasetDir /mnt/nvme/aircraft_data/datasets/dataset_best3000

    # also delete the underlying WAV/.spec.npy files from disk:
    python scripts/excludeFlaggedClips.py \\
        --flagged bad_helicopter.csv \\
        --datasetDir dataset \\
        --deleteFiles
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(
        description="Remove flagged clips from a dataset's CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--flagged", required=True, type=Path,
                    help="CSV from vizSpecs.py --flagOutput (one 'filepath' column).")
    p.add_argument("--datasetDir", required=True, type=Path,
                    help="Directory containing dataset.csv/train.csv/val.csv to filter.")
    p.add_argument("--deleteFiles", action="store_true",
                    help="Also delete the flagged WAV (and .spec.npy sidecar, if present) from "
                         "disk. Off by default -- removing rows from the CSVs is enough to keep "
                         "flagged clips out of training and future rebuilds. Irreversible.")
    args = p.parse_args()

    if not args.flagged.exists():
        raise SystemExit(f"Flagged CSV not found: {args.flagged}")
    if not args.datasetDir.exists():
        raise SystemExit(f"Dataset directory not found: {args.datasetDir}")

    flagged = set(pd.read_csv(args.flagged)["filepath"])
    print(f"Loaded {len(flagged)} flagged clip(s) from {args.flagged}")
    if not flagged:
        print("Nothing to do.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    touchedAny = False
    for name in ("dataset.csv", "train.csv", "val.csv"):
        csvPath = args.datasetDir / name
        if not csvPath.exists():
            continue
        df = pd.read_csv(csvPath)
        mask = df["filepath"].isin(flagged)
        removed = int(mask.sum())
        if removed == 0:
            print(f"{name}: no flagged rows present, left unchanged")
            continue
        backupPath = args.datasetDir / f"backup_{timestamp}_{name}"
        shutil.copy2(csvPath, backupPath)
        df[~mask].to_csv(csvPath, index=False)
        print(f"{name}: removed {removed} row(s) ({len(df)} -> {len(df) - removed}); "
              f"backed up to {backupPath.name}")
        touchedAny = True

    if not touchedAny:
        print("No flagged filepaths matched any CSV in --datasetDir "
              "(check the path prefix -- see script docstring).")

    if args.deleteFiles:
        deleted = 0
        for fp in flagged:
            wavPath = Path(fp)
            specPath = wavPath.parent / (wavPath.stem + ".spec.npy")
            for path in (wavPath, specPath):
                if path.exists():
                    path.unlink()
                    deleted += 1
        print(f"Deleted {deleted} file(s) from disk.")


if __name__ == "__main__":
    main()

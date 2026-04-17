#!/usr/bin/env python3
"""
Build a clip-level training dataset from recorded flyover sessions.

Usage:
    python scripts/buildDataset.py --recordingsDir ./recordings --outputDir ./dataset
                                   [--clipSecs 5.0]
                                   [--maxDistanceKm 10.0]

Writes:
    <outputDir>/clips/*.wav      — fixed-length audio clips
    <outputDir>/dataset.csv      — one row per clip with labels
    <outputDir>/train.csv        — training split (by flyover event)
    <outputDir>/val.csv          — validation split (by flyover event)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aircraftAudio.dataset.clipExport import buildClipDataset, splitByEvent


def main():
    p = argparse.ArgumentParser(description="Build clip-level training dataset.")
    p.add_argument("--recordingsDir", required=True, type=Path)
    p.add_argument("--outputDir",     required=True, type=Path)
    p.add_argument("--clipSecs",      type=float, default=5.0,
                   help="Duration of each extracted clip in seconds (default: 5.0)")
    p.add_argument("--trainFrac",     type=float, default=0.8,
                   help="Fraction of flyover events for training (default: 0.8)")
    p.add_argument("--minDistanceKm", type=float, default=None,
                   help="Skip states where aircraft is closer than this (km)")
    p.add_argument("--maxDistanceKm", type=float, default=None,
                   help="Skip states where aircraft is farther than this (km)")
    args = p.parse_args()

    df = buildClipDataset(
        recordingsDir=args.recordingsDir,
        outputDir=args.outputDir,
        clipSecs=args.clipSecs,
        minDistanceKm=args.minDistanceKm,
        maxDistanceKm=args.maxDistanceKm,
    )

    if df.empty:
        print("No clips generated — check that recordings were made with the updated recorder.py.")
        return

    trainDf, valDf = splitByEvent(df, trainFrac=args.trainFrac)
    trainDf.to_csv(args.outputDir / "train.csv", index=False)
    valDf.to_csv(args.outputDir / "val.csv", index=False)

    print(f"\nSplit: {len(trainDf)} train clips ({trainDf['recordingId'].nunique()} events) / "
          f"{len(valDf)} val clips ({valDf['recordingId'].nunique()} events)")


if __name__ == "__main__":
    main()

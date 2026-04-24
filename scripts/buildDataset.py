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

from aircraftAudio.dataset.clipExport import buildClipDataset, splitByEvent, balanceDataset


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
    p.add_argument("--faaDatabaseDir", type=Path, default=None,
                   help="Path to unzipped FAA ReleasableAircraft directory (STRONGLY RECOMMENDED). "
                        "Without it, type_categories falls back to an imprecise keyword heuristic "
                        "that misclassifies turboprop Pipers, Malibu variants, and others. "
                        "Download: https://www.faa.gov/licenses_certificates/aircraft_certification/"
                        "aircraft_registry/releasable_aircraft_download")
    p.add_argument("--clockCorrection", type=float, default=None,
                   help="Pi−server clock offset in seconds (positive = Pi was ahead). "
                        "Used for existing recordings without a stored clockSkewSecs.")
    p.add_argument("--autoCorrectClock", action="store_true",
                   help="Estimate per-recording clock skew from state timestamps. "
                        "Recommended for existing recordings with inconsistent skew.")
    p.add_argument("--maxCoTrackRatio", type=float, default=None,
                   help="Exclude clips where any co-tracked aircraft is within "
                        "(ratio × primary distance).  E.g. 2.0 keeps only clips "
                        "where the next-closest aircraft is more than 2× farther.")
    p.add_argument("--dropUnknown", action="store_true",
                   help="Exclude clips whose type_categories are entirely 'unknown'. "
                        "Null/background clips are kept regardless.")
    p.add_argument("--balanceClasses", action="store_true",
                   help="Downsample so each label (including null) has equal clip count. "
                        "Uses the rarest label count unless --maxPerClass is also given.")
    p.add_argument("--maxPerClass", type=int, default=None,
                   help="Cap clips per label at this number (implies --balanceClasses). "
                        "Useful when the rarest class still has more clips than desired.")
    p.add_argument("--stratifyPhase", action="store_true",
                   help="When balancing, split each label into (label × flightPhase) "
                        "buckets so approach and departure clips are kept in equal numbers.")
    args = p.parse_args()

    if args.faaDatabaseDir is None:
        print(
            "\n[WARNING] --faaDatabaseDir not provided. type_categories will use a keyword "
            "heuristic that misclassifies common aircraft (e.g. turboprop Pipers show up as "
            "piston_single). Training on bad labels wastes GPU time. Download ReleasableAircraft "
            "from faa.gov and pass --faaDatabaseDir for authoritative categories.\n"
        )

    df = buildClipDataset(
        recordingsDir=args.recordingsDir,
        outputDir=args.outputDir,
        clipSecs=args.clipSecs,
        minDistanceKm=args.minDistanceKm,
        maxDistanceKm=args.maxDistanceKm,
        faaDatabaseDir=args.faaDatabaseDir,
        clockCorrectionSecs=args.clockCorrection,
        autoCorrectClock=args.autoCorrectClock,
        maxCoTrackDistanceRatio=args.maxCoTrackRatio,
        dropUnknown=args.dropUnknown,
    )

    if df.empty:
        print("No clips generated — check that recordings were made with the updated recorder.py.")
        return

    if args.balanceClasses or args.maxPerClass is not None:
        before = len(df)
        df = balanceDataset(df, maxPerClass=args.maxPerClass, stratifyPhase=args.stratifyPhase)
        cap = args.maxPerClass or "auto"
        mode = "label × phase" if args.stratifyPhase else "label"
        print(f"Balanced: {before} → {len(df)} clips (cap: {cap}, buckets: {mode})")

    trainDf, valDf = splitByEvent(df, trainFrac=args.trainFrac)
    trainDf.to_csv(args.outputDir / "train.csv", index=False)
    valDf.to_csv(args.outputDir / "val.csv", index=False)

    print(f"\nSplit: {len(trainDf)} train clips ({trainDf['recordingId'].nunique()} events) / "
          f"{len(valDf)} val clips ({valDf['recordingId'].nunique()} events)")


if __name__ == "__main__":
    main()

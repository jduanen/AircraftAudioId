#!/usr/bin/env python3
"""
Export recorded sessions to a training CSV compatible with toolchain.py.

Expected output columns (matches toolchain.py VehicleAudioDataset):
    filepath        — absolute path to the WAV file
    vehicle_types   — JSON-encoded list of aircraft type strings (multi-label)
    duration        — recording duration in seconds
    altitudeFt      — altitude of closest aircraft approach (feet)
    velocityKts     — speed at closest approach (knots)
    bearingDeg      — bearing from observer to aircraft at closest approach (degrees)
    distanceKm      — distance at closest approach (km)
    callsign        — callsign at closest approach
    recordingId     — recording identifier
"""

import json
import argparse
from pathlib import Path

import pandas as pd


def createTrainingDataset(recordingsDir: str | Path, outputCsv: str | Path | None = None) -> pd.DataFrame:
    """
    Walk a recordings directory produced by AircraftRecordingSystem and build
    a training CSV suitable for toolchain.py.

    Args:
        recordingsDir:  Root directory containing audio/ and metadata/ subdirs.
        outputCsv:      Path to write the CSV.  Defaults to
                        <recordingsDir>/dataset.csv.

    Returns:
        DataFrame of all successfully parsed recordings.
    """
    recordingsDir = Path(recordingsDir)
    metadataDir = recordingsDir / "metadata"
    audioDir = recordingsDir / "audio"

    if outputCsv is None:
        outputCsv = recordingsDir / "dataset.csv"
    outputCsv = Path(outputCsv)

    rows = []
    missing = 0

    for jsonPath in sorted(metadataDir.glob("*.json")):
        with open(jsonPath) as f:
            meta = json.load(f)

        wavPath = audioDir / f"{meta['recordingId']}.wav"
        if not wavPath.exists():
            missing += 1
            continue

        closest = meta.get("closestAircraft") or {}

        aircraftType = meta.get("aircraftType")
        # vehicle_types is a multi-label list; for now each recording has one
        # aircraft, so it's a single-element list (or empty if type unknown).
        vehicleTypes = [aircraftType] if aircraftType else []

        rows.append({
            "filepath":     str(wavPath.resolve()),
            "vehicle_types": json.dumps(vehicleTypes),
            "duration":     meta.get("duration", 0.0),
            "altitudeFt":   closest.get("altitudeFt", 0.0),
            "velocityKts":  closest.get("velocityKts", 0.0),
            "bearingDeg":   closest.get("bearingDeg", 0.0),
            "distanceKm":   meta.get("minDistanceKm", 0.0),
            "callsign":     closest.get("callsign") or "",
            "recordingId":  meta.get("recordingId", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outputCsv, index=False)

    print(f"Dataset: {len(df)} recordings written to {outputCsv}")
    if missing:
        print(f"  ({missing} metadata entries skipped — WAV file not found)")

    return df


def buildArgParser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export recordings to a training CSV.")
    p.add_argument("--recordingsDir", required=True, type=Path,
                   help="Root recordings directory (contains audio/ and metadata/)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path (default: <recordingsDir>/dataset.csv)")
    return p


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    createTrainingDataset(args.recordingsDir, args.output)

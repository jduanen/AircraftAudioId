#!/usr/bin/env python3
"""
Clip-level training dataset builder.

Walks a recordings directory produced by AircraftRecordingSystem, uses
align.py to find each ADS-B state's position in the WAV, extracts a
fixed-length audio clip centred on that position, and writes a dataset.csv
with one row per clip.

Label columns written to the CSV:
    filepath        — absolute path to the clip WAV
    recordingId     — flyover event ID (use this for train/test splitting)
    vehicle_types   — JSON list of aircraft type strings (multi-label)
    directionClass  — 0–7, aircraft heading quantised to 8 cardinal directions
                      (0=N, 1=NE, 2=E, ... 7=NW).  -1 if heading unavailable.
    velocityKts     — aircraft speed at this state (knots)
    altitudeFt      — aircraft altitude at this state (feet)
    distanceKm      — distance from observer at this state (km)
    bearingDeg      — bearing from observer to aircraft (degrees)
    headingDeg      — aircraft heading (degrees, direction of travel)
    clipOffsetSecs  — time offset of clip centre from audio start (seconds)
"""

import json
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .align import alignedWindows


def _headingToDirectionClass(headingDeg: float) -> int:
    """
    Quantise a heading (0–360) into one of 8 cardinal direction classes.
    0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW.
    """
    return int((headingDeg + 22.5) / 45.0) % 8


def _extractClip(
    audio: np.ndarray,
    sampleRate: int,
    centreSample: int,
    clipSecs: float,
) -> np.ndarray:
    """
    Return a float32 mono array of length clipSecs * sampleRate centred on
    centreSample.  Pads with zeros if the window extends outside the audio.
    """
    halfLen = int(clipSecs / 2 * sampleRate)
    totalLen = int(clipSecs * sampleRate)
    start = centreSample - halfLen
    end = start + totalLen

    # Pad if necessary
    padLeft = max(0, -start)
    padRight = max(0, end - len(audio))
    safeStart = max(0, start)
    safeEnd = min(len(audio), end)

    clip = audio[safeStart:safeEnd]
    if padLeft > 0 or padRight > 0:
        clip = np.pad(clip, (padLeft, padRight))

    return clip.astype(np.float32)


def buildClipDataset(
    recordingsDir: str | Path,
    outputDir: str | Path,
    clipSecs: float = 5.0,
    outputCsv: Optional[str | Path] = None,
    minDistanceKm: Optional[float] = None,
    maxDistanceKm: Optional[float] = None,
) -> pd.DataFrame:
    """
    Extract per-state clips from all recordings and write a dataset CSV.

    Args:
        recordingsDir:   Root recordings directory (contains audio/ and metadata/).
        outputDir:       Directory to write clip WAVs into.
        clipSecs:        Duration of each extracted clip in seconds.
        outputCsv:       CSV output path.  Defaults to <outputDir>/dataset.csv.
        minDistanceKm:   Skip states where the aircraft is farther than this.
        maxDistanceKm:   Skip states where the aircraft is closer than this.

    Returns:
        DataFrame of all clips written.
    """
    recordingsDir = Path(recordingsDir)
    outputDir = Path(outputDir)
    clipsDir = outputDir / "clips"
    clipsDir.mkdir(parents=True, exist_ok=True)

    if outputCsv is None:
        outputCsv = outputDir / "dataset.csv"

    rows = []
    skippedNoAlignment = 0
    skippedDistance = 0

    for metaPath in sorted((recordingsDir / "metadata").glob("*.json")):
        with open(metaPath) as f:
            meta = json.load(f)

        recordingId = meta.get("recordingId", metaPath.stem)
        wavPath = recordingsDir / "audio" / f"{recordingId}.wav"

        if not wavPath.exists():
            continue

        if meta.get("audioStartTime") is None:
            skippedNoAlignment += 1
            continue

        # Load audio once per recording
        audio, sampleRate = sf.read(str(wavPath), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        aircraftType = meta.get("aircraftType")
        vehicleTypes = [aircraftType] if aircraftType else []

        try:
            states = alignedWindows(metaPath, windowSecs=clipSecs)
        except ValueError:
            skippedNoAlignment += 1
            continue

        for i, state in enumerate(states):
            distKm = state.get("distanceKm", 0.0)
            if minDistanceKm is not None and distKm < minDistanceKm:
                skippedDistance += 1
                continue
            if maxDistanceKm is not None and distKm > maxDistanceKm:
                skippedDistance += 1
                continue

            centreSample = state["sampleIndex"]
            clip = _extractClip(audio, sampleRate, centreSample, clipSecs)

            clipName = f"{recordingId}_s{i:03d}.wav"
            clipPath = clipsDir / clipName
            sf.write(str(clipPath), clip, sampleRate)

            headingDeg = state.get("headingDeg", -1.0)
            dirClass = _headingToDirectionClass(headingDeg) if headingDeg >= 0 else -1

            rows.append({
                "filepath":       str(clipPath.resolve()),
                "recordingId":    recordingId,
                "vehicle_types":  json.dumps(vehicleTypes),
                "directionClass": dirClass,
                "velocityKts":    state.get("velocityKts", 0.0),
                "altitudeFt":     state.get("altitudeFt", 0.0),
                "distanceKm":     distKm,
                "bearingDeg":     state.get("bearingDeg", 0.0),
                "headingDeg":     headingDeg,
                "clipOffsetSecs": state.get("timeOffsetSecs", 0.0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(outputCsv, index=False)

    print(f"Clips written:        {len(df)}")
    print(f"Output directory:     {clipsDir}")
    print(f"CSV:                  {outputCsv}")
    if skippedNoAlignment:
        print(f"Skipped (no audioStartTime — re-record): {skippedNoAlignment}")
    if skippedDistance:
        print(f"Skipped (distance filter): {skippedDistance}")

    return df


def splitByEvent(
    df: pd.DataFrame,
    trainFrac: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a clip dataset into train/val by recordingId (flyover event), not by
    row.  All clips from one flyover stay in the same split.

    Returns:
        (train_df, val_df)
    """
    rng = np.random.default_rng(seed)
    eventIds = df["recordingId"].unique()
    rng.shuffle(eventIds)
    nTrain = max(1, int(len(eventIds) * trainFrac))
    trainEvents = set(eventIds[:nTrain])
    trainDf = df[df["recordingId"].isin(trainEvents)].reset_index(drop=True)
    valDf = df[~df["recordingId"].isin(trainEvents)].reset_index(drop=True)
    return trainDf, valDf

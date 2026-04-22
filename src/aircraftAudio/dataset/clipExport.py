#!/usr/bin/env python3
"""
Clip-level training dataset builder.

Walks a recordings directory produced by AircraftRecordingSystem, uses
align.py to find each ADS-B state's position in the WAV, extracts a
fixed-length audio clip centred on that position, and writes a dataset.csv
with one row per clip.

Label columns written to the CSV:
    filepath          — absolute path to the clip WAV
    recordingId       — flyover event ID (use this for train/test splitting)
    vehicle_types     — JSON list of raw aircraft type strings (multi-label)
    type_categories   — JSON list of coarse category strings, parallel to vehicle_types
                        (piston_single | piston_twin | turboprop | helicopter |
                         business_jet | regional_jet | narrowbody_jet | widebody_jet | unknown)
    isSingle          — 1 if exactly one aircraft was tracked in this recording, else 0
    flightPhase       — "approach" | "closest" | "departure" | "unknown"
                        derived from distance trend across adjacent ADS-B states
    directionClass    — 0–7, aircraft heading quantised to 8 cardinal directions
                        (0=N, 1=NE, 2=E, ... 7=NW).  -1 if heading unavailable.
    velocityKts       — aircraft speed at this state (knots)
    altitudeFt        — aircraft altitude at this state (feet)
    distanceKm        — distance from observer at this state (km)
    bearingDeg        — bearing from observer to aircraft (degrees)
    headingDeg        — aircraft heading (degrees, direction of travel)
    clipOffsetSecs    — time offset of clip centre from audio start (seconds)
"""

import json
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .align import alignedWindows
from .typeCategories import typeToCategory
from .faaDatabase import FaaDatabase


def _flightPhase(distances: list[float], idx: int) -> str:
    """
    Classify a state as approach/closest/departure based on its neighbours.
    `distances` is the full ordered list of distanceKm for all states in the
    recording; `idx` is the position of this state within that list.
    """
    prev = distances[idx - 1] if idx > 0 else None
    nxt  = distances[idx + 1] if idx < len(distances) - 1 else None
    d    = distances[idx]

    if prev is None and nxt is None:
        return "unknown"
    if prev is not None and nxt is not None:
        if prev > d and d < nxt:
            return "closest"
        if prev > d:
            return "approach"
        if d < nxt:
            return "approach"
        return "departure"
    if prev is None:
        return "approach" if (nxt is not None and d <= nxt) else "departure"
    return "departure" if (prev is not None and d >= prev) else "approach"


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
    faaDatabaseDir: Optional[str | Path] = None,
    clockCorrectionSecs: Optional[float] = None,
    autoCorrectClock: bool = False,
    maxCoTrackDistanceRatio: Optional[float] = None,
    dropUnknown: bool = False,
) -> pd.DataFrame:
    """
    Extract per-state clips from all recordings and write a dataset CSV.

    Args:
        recordingsDir:   Root recordings directory (contains audio/ and metadata/).
        outputDir:       Directory to write clip WAVs into.
        clipSecs:        Duration of each extracted clip in seconds.
        outputCsv:       CSV output path.  Defaults to <outputDir>/dataset.csv.
        minDistanceKm:        Skip states where the aircraft is farther than this.
        maxDistanceKm:        Skip states where the aircraft is closer than this.
        faaDatabaseDir:       Path to unzipped FAA ReleasableAircraft directory.
        clockCorrectionSecs:  Manual Pi−server clock offset (seconds) for
                              recordings that pre-date the clockSkewSecs metadata
                              field.  Ignored for recordings that already have
                              clockSkewSecs stored.
        autoCorrectClock:          If True, estimate per-recording clock skew from the
                                   state timestamps when no stored/manual correction is
                                   available.  Recommended for existing recordings.
        maxCoTrackDistanceRatio:   Exclude clips where any co-tracked aircraft is closer
                                   than (ratio × primary distance).  E.g. 2.0 means skip
                                   clips where another aircraft is within 2× the primary's
                                   distance.  None disables the filter.
                         When provided, ICAO24-based lookup is used for
                         type_categories instead of model-string heuristics.
                         Foreign aircraft fall back to the string heuristic.
        dropUnknown:     If True, exclude clips whose type_categories list consists
                         entirely of "unknown" entries (null/background clips are kept).

    Returns:
        DataFrame of all clips written.
    """
    recordingsDir = Path(recordingsDir)
    outputDir = Path(outputDir)
    clipsDir = outputDir / "clips"
    clipsDir.mkdir(parents=True, exist_ok=True)

    if outputCsv is None:
        outputCsv = outputDir / "dataset.csv"

    faaDb: Optional[FaaDatabase] = None
    if faaDatabaseDir is not None:
        faaDb = FaaDatabase(faaDatabaseDir)
        print(f"FAA database loaded: {len(faaDb)} registrations")

    rows = []
    skippedNoAlignment = 0
    skippedDistance = 0
    skippedCoTrack = 0
    nullClips = 0

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

        # ── Null (background) recordings ─────────────────────────────────────
        if meta.get("isNullSample"):
            clip = _extractClip(audio, sampleRate, len(audio) // 2, clipSecs)
            clipName = f"{recordingId}_null.wav"
            clipPath = clipsDir / clipName
            sf.write(str(clipPath), clip, sampleRate)
            rows.append({
                "filepath":         str(clipPath.resolve()),
                "recordingId":      recordingId,
                "vehicle_types":    json.dumps([]),
                "type_categories":  json.dumps([]),
                "isSingle":         0,
                "flightPhase":      "null",
                "directionClass":   -1,
                "velocityKts":      0.0,
                "altitudeFt":       0.0,
                "distanceKm":       0.0,
                "bearingDeg":       0.0,
                "headingDeg":       0.0,
                "clipOffsetSecs":   meta.get("duration", clipSecs) / 2,
            })
            nullClips += 1
            continue

        aircraftType = meta.get("aircraftType")
        vehicleTypes = [aircraftType] if aircraftType else []

        # Resolve category: FAA ICAO24 lookup first, string heuristic fallback.
        primaryIcao = next(
            (s.get("icao24") for s in meta.get("aircraftStates", []) if s.get("icao24")),
            None,
        )
        if faaDb is not None and primaryIcao:
            typeCategories = [faaDb.categoryForIcao24(primaryIcao, aircraftType)]
        else:
            typeCategories = [typeToCategory(t) for t in vehicleTypes]

        # isSingle: true when only one aircraft was tracked in this recording.
        allIcao = {s.get("icao24") for s in meta.get("aircraftStates", []) if s.get("icao24")}
        isSingle = 1 if len(allIcao) == 1 else 0

        # co-tracked aircraft distances for ratio filter
        coTracked = meta.get("coTrackedAircraft", [])

        try:
            states = alignedWindows(
                metaPath,
                windowSecs=clipSecs,
                clockCorrectionSecs=clockCorrectionSecs,
                autoCorrect=autoCorrectClock,
            )
        except ValueError:
            skippedNoAlignment += 1
            continue

        allDistances = [s["distanceKm"] for s in states]

        for i, state in enumerate(states):
            distKm = state.get("distanceKm", 0.0)
            if minDistanceKm is not None and distKm < minDistanceKm:
                skippedDistance += 1
                continue
            if maxDistanceKm is not None and distKm > maxDistanceKm:
                skippedDistance += 1
                continue

            # Distance ratio filter: skip if any co-tracked aircraft is within
            # (ratio × this state's distance).
            if maxCoTrackDistanceRatio is not None and distKm > 0 and coTracked:
                threshold = maxCoTrackDistanceRatio * distKm
                if any(c.get("distanceKm", 999) <= threshold for c in coTracked):
                    skippedCoTrack += 1
                    continue

            centreSample = state["sampleIndex"]
            clip = _extractClip(audio, sampleRate, centreSample, clipSecs)

            clipName = f"{recordingId}_s{i:03d}.wav"
            clipPath = clipsDir / clipName
            sf.write(str(clipPath), clip, sampleRate)

            headingDeg = state.get("headingDeg", -1.0)
            dirClass = _headingToDirectionClass(headingDeg) if headingDeg >= 0 else -1

            rows.append({
                "filepath":         str(clipPath.resolve()),
                "recordingId":      recordingId,
                "vehicle_types":    json.dumps(vehicleTypes),
                "type_categories":  json.dumps(typeCategories),
                "isSingle":         isSingle,
                "flightPhase":      _flightPhase(allDistances, i),
                "directionClass":   dirClass,
                "velocityKts":      state.get("velocityKts", 0.0),
                "altitudeFt":       state.get("altitudeFt", 0.0),
                "distanceKm":       distKm,
                "bearingDeg":       state.get("bearingDeg", 0.0),
                "headingDeg":       headingDeg,
                "clipOffsetSecs":   state.get("timeOffsetSecs", 0.0),
            })

    df = pd.DataFrame(rows)

    skippedUnknown = 0
    if dropUnknown and not df.empty:
        def _allUnknown(catJson: str) -> bool:
            cats = json.loads(catJson)
            return len(cats) > 0 and all(c == "unknown" for c in cats)
        mask = df["type_categories"].apply(_allUnknown)
        skippedUnknown = mask.sum()
        df = df[~mask].reset_index(drop=True)

    df.to_csv(outputCsv, index=False)

    print(f"Clips written:        {len(df)}")
    if nullClips:
        print(f"  of which null:    {nullClips}")
    print(f"Output directory:     {clipsDir}")
    print(f"CSV:                  {outputCsv}")
    if skippedNoAlignment:
        print(f"Skipped (no audioStartTime — re-record): {skippedNoAlignment}")
    if skippedDistance:
        print(f"Skipped (distance filter):   {skippedDistance}")
    if skippedCoTrack:
        print(f"Skipped (co-track ratio):    {skippedCoTrack}")
    if skippedUnknown:
        print(f"Skipped (unknown type):      {skippedUnknown}")

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
    eventIds = np.array(df["recordingId"].unique(), dtype=str)
    rng.shuffle(eventIds)
    nTrain = max(1, int(len(eventIds) * trainFrac))
    trainEvents = set(eventIds[:nTrain])
    trainDf = df[df["recordingId"].isin(trainEvents)].reset_index(drop=True)
    valDf = df[~df["recordingId"].isin(trainEvents)].reset_index(drop=True)
    return trainDf, valDf


def balanceDataset(
    df: pd.DataFrame,
    maxPerClass: Optional[int] = None,
    stratifyPhase: bool = False,
    labelCol: str = "type_categories",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Downsample df so each label (including null/background) has at most
    maxPerClass clips.  If maxPerClass is None, uses the count of the rarest
    bucket as the cap.

    Multi-label clips count toward every label they carry.  The greedy
    algorithm processes buckets from most-common to least-common, dropping
    randomly from each over-represented bucket in turn.

    Args:
        df:             Dataset DataFrame with a type_categories JSON column.
        maxPerClass:    Cap per bucket.  None = auto (rarest bucket count).
        stratifyPhase:  If True, balance within each (label × flightPhase)
                        bucket rather than per label alone.  Requires a
                        flightPhase column in df.  Ensures approach and
                        departure clips are kept in equal numbers per class.
        labelCol:       Column containing JSON-encoded label lists.
        seed:           RNG seed for reproducible downsampling.

    Returns:
        Downsampled DataFrame, index reset.
    """
    rng = np.random.default_rng(seed)
    parsed = df[labelCol].apply(json.loads).tolist()

    phases: list[str] = (
        df["flightPhase"].tolist()
        if stratifyPhase and "flightPhase" in df.columns
        else ["_"] * len(df)
    )

    # Build bucket → row indices.
    # Bucket key is (label, phase) when stratifying, else (label, "_").
    # Null/background clips use label "null" and their own phase value.
    buckets: dict[tuple[str, str], list[int]] = {}
    for i, (cats, phase) in enumerate(zip(parsed, phases)):
        if not cats:
            key = ("null", phase)
            buckets.setdefault(key, []).append(i)
        else:
            for c in cats:
                key = (c, phase)
                buckets.setdefault(key, []).append(i)

    if maxPerClass is None:
        maxPerClass = min(len(v) for v in buckets.values())

    keepMask = np.ones(len(df), dtype=bool)

    # Process most-common buckets first so rarer buckets lose fewer clips.
    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
        currentIndices = [i for i in buckets[key] if keepMask[i]]
        excess = len(currentIndices) - maxPerClass
        if excess > 0:
            toDrop = rng.choice(currentIndices, excess, replace=False)
            keepMask[toDrop] = False

    return df[keepMask].reset_index(drop=True)

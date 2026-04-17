#!/usr/bin/env python3
"""
Audio-to-ADS-B alignment.

Maps each ADS-B state snapshot in a recording's metadata to its corresponding
position in the WAV file, using the Pi-side audioStartTime and the server-side
capturedAt timestamp on each state.

Assumptions:
  - Pi and server clocks are NTP-synced (piCapture.py warns if offset > 100 ms).
  - capturedAt on each AircraftState is set to ReadsbClient._lastPoll (server clock).
  - audioStartTime in RecordingMetadata is from RemoteAudioStream.getBufferStartTime()
    (Pi clock).
  - Residual clock skew between Pi and server is small relative to the 1-second
    ADS-B poll interval.
"""

import json
from pathlib import Path
from typing import Optional


def alignStates(metadataPath: str | Path) -> list[dict]:
    """
    Load a recording metadata JSON and return the ADS-B states annotated with
    their position in the WAV file.

    Each returned dict is the original state dict plus:
        timeOffsetSecs  — seconds from sample 0 to this state (may be negative
                          if the state was captured before the audio window)
        sampleIndex     — integer sample index in the WAV (clamped to [0, N-1])
        inWindow        — True if the state falls within [0, duration]

    Args:
        metadataPath:  Path to a <recordingId>.json metadata file.

    Returns:
        List of annotated state dicts, in original poll order.
    """
    meta = _loadMetadata(metadataPath)

    audioStartTime: Optional[float] = meta.get("audioStartTime")
    duration: float = meta.get("duration", 0.0)
    sampleRate: int = meta.get("sampleRate", 44100)
    totalSamples = int(duration * sampleRate)

    if audioStartTime is None:
        raise ValueError(
            f"Metadata at {metadataPath} has no audioStartTime. "
            "Re-record with the updated recorder.py."
        )

    aligned = []
    for state in meta.get("aircraftStates", []):
        capturedAt: Optional[float] = state.get("capturedAt")
        if capturedAt is None:
            timeOffset = None
            sampleIndex = None
            inWindow = False
        else:
            timeOffset = capturedAt - audioStartTime
            sampleIndex = max(0, min(int(timeOffset * sampleRate), totalSamples - 1))
            inWindow = 0.0 <= timeOffset <= duration

        aligned.append({
            **state,
            "timeOffsetSecs": timeOffset,
            "sampleIndex": sampleIndex,
            "inWindow": inWindow,
        })

    return aligned


def alignedWindows(
    metadataPath: str | Path,
    windowSecs: float = 1.0,
) -> list[dict]:
    """
    Return only the in-window states, each paired with a [start, end] sample
    range representing a `windowSecs`-wide slice centred on the state.

    Useful for cutting the WAV into per-observation clips.

    Each returned dict includes all alignStates fields plus:
        windowStart  — sample index of the window start (clamped to 0)
        windowEnd    — sample index of the window end (clamped to totalSamples)
    """
    meta = _loadMetadata(metadataPath)
    duration: float = meta.get("duration", 0.0)
    sampleRate: int = meta.get("sampleRate", 44100)
    totalSamples = int(duration * sampleRate)
    halfWindow = int(windowSecs / 2 * sampleRate)

    result = []
    for state in alignStates(metadataPath):
        if not state["inWindow"]:
            continue
        idx = state["sampleIndex"]
        result.append({
            **state,
            "windowStart": max(0, idx - halfWindow),
            "windowEnd":   min(totalSamples, idx + halfWindow),
        })
    return result


def _loadMetadata(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)

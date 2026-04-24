#!/usr/bin/env python3
"""
Dataset inspection: quantity, quality, and distribution of collected samples.

Usage:
    python scripts/inspectDataset.py --recordingsDir ./recordings [--datasetCsv ./dataset/dataset.csv]

Sections printed:
    1. Recordings inventory  — count, durations, aircraft types, distances
    2. Alignment health      — in-window vs out-of-window ADS-B states, clock skew diagnosis
    3. Dataset CSV summary   — clips per type, label distributions, train/val split
    4. Audio quality         — RMS level, silence fraction per clip
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, maxVal: float, width: int = 30) -> str:
    filled = int(round(value / maxVal * width)) if maxVal > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _histo(values: list[float], bins: int = 10, label: str = "") -> None:
    if not values:
        print(f"  {label}: (no data)")
        return
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=bins)
    maxCount = counts.max()
    print(f"\n  {label} (n={len(arr)}, min={arr.min():.1f}, "
          f"median={np.median(arr):.1f}, max={arr.max():.1f})")
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        print(f"  {lo:7.1f}–{hi:7.1f}  {_bar(c, maxCount, 25)}  {c}")


def _countBar(counter: Counter, title: str) -> None:
    total = sum(counter.values())
    print(f"\n  {title} (total={total})")
    for key, count in counter.most_common():
        pct = count / total * 100
        print(f"  {str(key):<25}  {_bar(count, counter.most_common(1)[0][1], 25)}  {count:4d} ({pct:.0f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Recordings inventory
# ──────────────────────────────────────────────────────────────────────────────

def inspectRecordings(recordingsDir: Path) -> dict:
    metaDir = recordingsDir / "metadata"
    audioDir = recordingsDir / "audio"

    if not metaDir.exists():
        print("  [!] No metadata/ directory found.")
        return {}

    metaPaths = sorted(p for p in metaDir.glob("*.json") if not p.name.startswith("session_"))
    print(f"\n{'═'*60}")
    print("  RECORDINGS INVENTORY")
    print(f"{'═'*60}")
    print(f"  Metadata files: {len(metaPaths)}")

    hasBothCount = sum(1 for p in metaPaths if (audioDir / f"{p.stem}.wav").exists())
    print(f"  Have matching WAV: {hasBothCount}")
    print(f"  Missing WAV:       {len(metaPaths) - hasBothCount}")

    durations, distances, types, singleCount, multiCount = [], [], Counter(), 0, 0
    hasStartTime, missingStartTime = 0, 0

    for p in metaPaths:
        meta = json.load(open(p))
        dur = meta.get("duration", 0.0)
        durations.append(dur)

        if meta.get("audioStartTime"):
            hasStartTime += 1
        else:
            missingStartTime += 1

        states = meta.get("aircraftStates", [])
        icaos = {s.get("icao24") for s in states if s.get("icao24")}
        if len(icaos) == 1:
            singleCount += 1
        elif len(icaos) > 1:
            multiCount += 1

        for s in states:
            d = s.get("distanceKm")
            if d is not None:
                distances.append(d)

        atype = meta.get("aircraftType")
        if atype:
            types[atype] += 1
        else:
            types["(unknown)"] += 1

    print(f"\n  Has audioStartTime: {hasStartTime}  /  Missing: {missingStartTime}")
    print(f"  Single-aircraft recordings: {singleCount}")
    print(f"  Multi-aircraft recordings:  {multiCount}")
    _histo(durations, bins=8, label="Recording duration (s)")
    _histo(distances, bins=10, label="ADS-B state distance (km)")
    _countBar(types, "Aircraft type distribution")

    return {"metaPaths": metaPaths}


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Alignment health
# ──────────────────────────────────────────────────────────────────────────────

def inspectAlignment(metaPaths: list[Path]) -> None:
    print(f"\n{'═'*60}")
    print("  ALIGNMENT HEALTH")
    print(f"{'═'*60}")

    totalStates = 0
    inWindowStates = 0
    offsets = []
    noStartTime = 0
    noCapturedAt = 0
    perRecordingYield = []

    for p in metaPaths:
        meta = json.load(open(p))
        audioStartTime = meta.get("audioStartTime")
        duration = meta.get("duration", 0.0)

        if audioStartTime is None:
            noStartTime += 1
            continue

        skew = meta.get("clockSkewSecs") or 0.0
        states = meta.get("aircraftStates", [])
        inW = 0
        for s in states:
            capturedAt = s.get("capturedAt")
            totalStates += 1
            if capturedAt is None:
                noCapturedAt += 1
                continue
            offset = capturedAt - audioStartTime + skew
            offsets.append(offset)
            if 0.0 <= offset <= duration:
                inWindowStates += 1
                inW += 1
        if states:
            perRecordingYield.append(inW / len(states))

    print(f"\n  Total ADS-B states:    {totalStates}")
    print(f"  In audio window:       {inWindowStates}  ({100*inWindowStates/max(totalStates,1):.0f}%)")
    print(f"  Out of window:         {totalStates - inWindowStates - noCapturedAt}")
    if noCapturedAt:
        print(f"  Missing capturedAt:    {noCapturedAt}  (old recordings — re-record to fix)")
    if noStartTime:
        print(f"  Missing audioStartTime:{noStartTime}  (old recordings — re-record to fix)")

    if offsets:
        arr = np.array(offsets)
        print(f"\n  capturedAt − audioStartTime (seconds):")
        print(f"    min={arr.min():.1f}  median={np.median(arr):.1f}  max={arr.max():.1f}")
        negFrac = (arr < 0).mean()
        posFrac = (arr > 0).mean()
        print(f"    {negFrac*100:.0f}% before audio start  /  {posFrac*100:.0f}% after audio start")

        # Diagnose clock skew: if all offsets are systematically negative by a large amount
        if np.median(arr) < -5:
            print(f"\n  [!] CLOCK SKEW DETECTED: ADS-B states are captured on average "
                  f"{-np.median(arr):.0f}s BEFORE the audio window starts.")
            print(f"      This means the Pi clock is ahead of the server clock by ~{-np.median(arr):.0f}s.")
            print(f"      Fix: sync both clocks to NTP, then re-record.")
            print(f"      Workaround: set a clock correction offset in align.py.")
        elif np.median(arr) > 5:
            print(f"\n  [!] CLOCK SKEW DETECTED: ADS-B states land {np.median(arr):.0f}s AFTER audio start.")
            print(f"      The server clock may be ahead of the Pi clock.")

    if perRecordingYield:
        goodYield = sum(1 for y in perRecordingYield if y > 0)
        print(f"\n  Recordings producing ≥1 in-window state: {goodYield} / {len(perRecordingYield)}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Dataset CSV
# ──────────────────────────────────────────────────────────────────────────────

def inspectCsv(csvPath: Path) -> None:
    print(f"\n{'═'*60}")
    print("  DATASET CSV SUMMARY")
    print(f"{'═'*60}")

    if not csvPath.exists():
        print(f"  [!] {csvPath} not found — run scripts/buildDataset.py first.")
        return

    try:
        import pandas as pd
        import json as _json
    except ImportError:
        print("  pandas not available — skipping CSV summary.")
        return

    df = pd.read_csv(csvPath)
    print(f"\n  CSV: {csvPath}")
    print(f"  Rows (clips):    {len(df)}")
    print(f"  Flyover events:  {df['recordingId'].nunique()}")
    print(f"  Clips/event avg: {len(df)/max(df['recordingId'].nunique(),1):.1f}")

    # isSingle
    if "isSingle" in df.columns:
        n = df["isSingle"].sum()
        print(f"\n  Single-aircraft clips: {int(n)} ({100*n/len(df):.0f}%)")
        print(f"  Multi-aircraft clips:  {int(len(df)-n)} ({100*(1-n/len(df)):.0f}%)")
    else:
        print("\n  [!] isSingle column missing — rebuild dataset with updated clipExport.py")

    # Coarse category distribution (preferred when available)
    if "type_categories" in df.columns:
        catCounts = Counter()
        for raw in df["type_categories"]:
            for t in _json.loads(raw):
                catCounts[t] += 1
        if catCounts:
            _countBar(catCounts, "Coarse category distribution (type_categories)")
    else:
        print("\n  [!] type_categories column missing — rebuild dataset with updated clipExport.py")

    # Raw type distribution
    typeCounts = Counter()
    for raw in df["vehicle_types"]:
        for t in _json.loads(raw):
            typeCounts[t] += 1
    if typeCounts:
        _countBar(typeCounts, "Raw vehicle_types distribution")
    else:
        print("\n  No vehicle_types labels found.")

    # Flight phase distribution
    if "flightPhase" in df.columns:
        phaseCounts = Counter(df["flightPhase"].tolist())
        _countBar(phaseCounts, "Flight phase distribution")

    # Direction class distribution
    if "directionClass" in df.columns:
        dirLabels = {
            0: "away",             # 0°   flying directly away
            1: "away-right",       # 45°
            2: "crossing-right",   # 90°  crossing left→right
            3: "approaching-right",# 135°
            4: "toward",           # 180° flying directly toward
            5: "approaching-left", # 225°
            6: "crossing-left",    # 270° crossing right→left
            7: "away-left",        # 315°
            -1: "unknown",
        }
        dirCounts = Counter(df["directionClass"].astype(int).tolist())
        total = sum(dirCounts.values())
        print(f"\n  Direction class distribution (n={total})")
        for cls in [-1] + list(range(8)):
            c = dirCounts.get(cls, 0)
            if c:
                print(f"  {dirLabels[cls]:<7} ({cls:2d})  {_bar(c, max(dirCounts.values()), 25)}  {c:4d} ({100*c/total:.0f}%)")

    # Numeric distributions
    for col, label in [("velocityKts","Velocity (kts)"), ("altitudeFt","Altitude (ft)"),
                       ("distanceKm","Distance (km)")]:
        if col in df.columns:
            _histo(df[col].dropna().tolist(), bins=8, label=label)


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Audio quality
# ──────────────────────────────────────────────────────────────────────────────

def inspectAudioQuality(csvPath: Path, maxClips: int = 200) -> None:
    print(f"\n{'═'*60}")
    print("  AUDIO QUALITY")
    print(f"{'═'*60}")

    try:
        import pandas as pd
        import soundfile as sf
    except ImportError:
        print("  pandas/soundfile not available — skipping audio quality check.")
        return

    if not csvPath.exists():
        return

    df = pd.read_csv(csvPath)
    if df.empty:
        print("  No clips to inspect.")
        return

    sample = df.sample(min(maxClips, len(df)), random_state=42)
    rmsValues, silenceFracs, missingFiles = [], [], 0

    for _, row in sample.iterrows():
        p = Path(row["filepath"])
        if not p.exists():
            missingFiles += 1
            continue
        try:
            audio, _ = sf.read(str(p), dtype="float32", always_2d=False)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            silFrac = float(np.mean(np.abs(audio) < 0.005))
            rmsValues.append(rms)
            silenceFracs.append(silFrac)
        except Exception:
            missingFiles += 1

    if missingFiles:
        print(f"\n  Missing/unreadable files: {missingFiles}")

    if rmsValues:
        rmsDb = [20 * np.log10(max(r, 1e-9)) for r in rmsValues]
        _histo(rmsDb, bins=8, label="Clip RMS level (dBFS)")
        _histo([s * 100 for s in silenceFracs], bins=6, label="Silence fraction (%, near-zero samples)")

        lowEnergy = sum(1 for r in rmsValues if r < 0.001)
        print(f"\n  Very quiet clips (<−60 dBFS): {lowEnergy} / {len(rmsValues)}")
    else:
        print("\n  No clips could be read.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Inspect dataset quantity, quality, and distribution.")
    p.add_argument("--recordingsDir", required=True, type=Path,
                   help="Root recordings directory (contains audio/ and metadata/)")
    p.add_argument("--datasetCsv", type=Path, default=None,
                   help="Path to dataset.csv (default: <recordingsDir>/../dataset/dataset.csv)")
    p.add_argument("--maxQualityClips", type=int, default=200,
                   help="Max clips to read for audio quality check (default: 200)")
    args = p.parse_args()

    csvPath = args.datasetCsv or (args.recordingsDir.parent / "dataset" / "dataset.csv")

    result = inspectRecordings(args.recordingsDir)
    metaPaths = result.get("metaPaths", [])
    if metaPaths:
        inspectAlignment(metaPaths)
    inspectCsv(csvPath)
    inspectAudioQuality(csvPath, maxClips=args.maxQualityClips)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()

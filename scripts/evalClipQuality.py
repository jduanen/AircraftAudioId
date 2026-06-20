#!/usr/bin/env python3
"""
Per-class audio clip quality evaluation.

Two analysis tiers are available:

  Fast (default) — reads only the dataset CSV. No audio I/O.
    Uses the pre-computed clipRms column and ADS-B metadata already in the
    CSV to produce a per-class quality summary table, an RMS histogram for
    the selected class, and a flight-phase breakdown.

  Deep (--deepAnalysis) — reads each WAV file. Requires soundfile + librosa.
    Computes seven additional metrics per clip and prints per-class aggregate
    statistics, histograms for the most diagnostic metrics, a ranked worst-
    clips table, and optionally writes a bad-clip file for downstream filtering.

Metrics
───────
Fast (from CSV, no audio reads):
  rmsDb           RMS amplitude in dBFS. Primary signal-strength indicator.
                  Below −55 dBFS the aircraft is likely too quiet to learn from.
  distanceKm      Aircraft distance at clip time. Farther aircraft → weaker signal.
  altitudeFt      Aircraft altitude. High-altitude overflights are geometrically
                  attenuated and may not produce useful ground-level audio.

Deep (from WAV files, opt-in via --deepAnalysis):
  silenceFrac     Fraction of samples with |x| < 0.005.  High values mean the
                  clip is mostly inaudible regardless of RMS.
  clippingFrac    Fraction of samples with |x| > 0.99.  High values indicate
                  ADC saturation, which destroys spectral detail.
  frameEnergyStd  Std dev of per-frame (0.1 s) RMS across the clip.  Very low
                  values mean a temporally flat signal — likely ambient noise
                  with no distinct aircraft event.
  edgeCenterRatio RMS of the first+last 1 s divided by RMS of the middle 3 s.
                  A proper flyover should be louder in the middle (ratio < 1).
                  Ratio > 1 means an inverted or misaligned energy envelope.
  spectralFlatness Mean spectral flatness across STFT frames (0 = tonal/structured,
                  1 = spectrally flat broadband noise).  High flatness suggests
                  the clip contains noise rather than aircraft sound.
  spectralCentroid Mean spectral centroid in Hz.  Very low values (<300 Hz) indicate
                  the clip is dominated by wind or handling rumble.
  lowFreqRatio    Fraction of total spectral energy below 200 Hz.  Wind noise
                  concentrates in this band; high ratio → wind-dominated clip.

Usage
─────
  # All-class RMS summary (no audio reads):
  python scripts/evalClipQuality.py --datasetCsv dataset/dataset.csv

  # Focus on one class with phase breakdown:
  python scripts/evalClipQuality.py \\
      --datasetCsv dataset/dataset.csv --category piston_twin

  # Full deep analysis, export worst clips:
  python scripts/evalClipQuality.py \\
      --datasetCsv dataset/dataset.csv \\
      --category piston_twin \\
      --deepAnalysis --worstN 20 \\
      --outputBadClips bad_piston_twin.txt \\
      --rmsThresholdDb -55
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers (same pattern as inspectDataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, maxVal: float, width: int = 25) -> str:
    filled = int(round(value / maxVal * width)) if maxVal > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _histo(values: list[float], bins: int = 10, label: str = "") -> None:
    if not values:
        print(f"  {label}: (no data)")
        return
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=bins)
    maxCount = counts.max()
    print(f"\n  {label}  (n={len(arr)}, min={arr.min():.3f}, "
          f"median={np.median(arr):.3f}, max={arr.max():.3f})")
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        print(f"  {lo:9.3f}–{hi:9.3f}  {_bar(c, maxCount)}  {c}")


# ─────────────────────────────────────────────────────────────────────────────
# Fast metrics (CSV only)
# ─────────────────────────────────────────────────────────────────────────────

def _rmsDb(clipRms) -> float:
    """Convert linear RMS to dBFS, guarding against zero."""
    return 20.0 * np.log10(max(float(clipRms) if clipRms is not None else 0.0, 1e-9))


def _allClassSummary(df: pd.DataFrame, threshDb: float) -> list[dict]:
    """
    Build a per-class quality summary using only CSV columns.

    Iterates once through the type_categories column to collect class membership,
    then computes RMS, distance, and altitude statistics per class.
    """
    parsed = [json.loads(raw) for raw in df["type_categories"]]
    allCategories = sorted({cat for cats in parsed for cat in cats})

    rows = []
    for cat in allCategories:
        idx = [i for i, cats in enumerate(parsed) if cat in cats]
        sub = df.iloc[idx]
        rmsVals = [_rmsDb(r) for r in sub["clipRms"]]
        belowPct = 100.0 * sum(1 for r in rmsVals if r < threshDb) / len(rmsVals)
        rows.append({
            "category":    cat,
            "clips":       len(sub),
            "meanRmsDb":   float(np.mean(rmsVals)),
            "medianRmsDb": float(np.median(rmsVals)),
            "p10RmsDb":    float(np.percentile(rmsVals, 10)),
            "belowPct":    belowPct,
            "meanDistKm":  float(sub["distanceKm"].mean()) if "distanceKm" in sub else float("nan"),
            "meanAltFt":   float(sub["altitudeFt"].mean()) if "altitudeFt" in sub else float("nan"),
        })
    return rows


def _printClassSummary(rows: list[dict], threshDb: float) -> None:
    print(f"\n{'═'*95}")
    print("  PER-CLASS QUALITY SUMMARY  (fast path — CSV only)")
    print(f"  Low-quality threshold: {threshDb:.0f} dBFS")
    print(f"{'═'*95}")
    hdr = (f"  {'Class':<18}  {'Clips':>6}  {'Mean dBFS':>9}  {'Median':>7}  "
           f"{'P10':>7}  {'%Low':>5}  {'AvgDist km':>10}  {'AvgAlt ft':>9}")
    print(hdr)
    print("  " + "─" * 89)
    for r in rows:
        print(
            f"  {r['category']:<18}  {r['clips']:>6}  {r['meanRmsDb']:>9.1f}  "
            f"{r['medianRmsDb']:>7.1f}  {r['p10RmsDb']:>7.1f}  "
            f"{r['belowPct']:>5.1f}  {r['meanDistKm']:>10.1f}  {r['meanAltFt']:>9.0f}"
        )
    print(f"{'═'*95}")


def _printPhaseSummary(df: pd.DataFrame, category: str, threshDb: float) -> None:
    """Print RMS and low-quality rates broken down by flight phase."""
    if "flightPhase" not in df.columns:
        return
    print(f"\n{'═'*65}")
    print(f"  PHASE BREAKDOWN  [{category}]")
    print(f"{'═'*65}")
    hdr = f"  {'Phase':<14}  {'Clips':>6}  {'Mean dBFS':>9}  {'Median':>7}  {'%Low':>5}"
    print(hdr)
    print("  " + "─" * 46)
    for phase in sorted(df["flightPhase"].dropna().unique()):
        sub = df[df["flightPhase"] == phase]
        rmsVals = [_rmsDb(r) for r in sub["clipRms"]]
        belowPct = 100.0 * sum(1 for r in rmsVals if r < threshDb) / len(rmsVals)
        print(
            f"  {str(phase):<14}  {len(sub):>6}  {np.mean(rmsVals):>9.1f}  "
            f"{np.median(rmsVals):>7.1f}  {belowPct:>5.1f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Deep metrics (reads WAV files)
# ─────────────────────────────────────────────────────────────────────────────

def _deepMetrics(wavPath: str, sampleRate: int = 44100) -> dict:
    """
    Compute audio quality metrics by reading a WAV file.

    Args:
        wavPath:    Path to the clip WAV file.
        sampleRate: Expected sample rate (used for frequency bin calculation).

    Returns:
        Dict with keys: silenceFrac, clippingFrac, frameEnergyStd,
        edgeCenterRatio, spectralFlatness, spectralCentroid, lowFreqRatio.
    """
    import soundfile as sf
    import librosa

    audio, _ = sf.read(str(wavPath), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    # ── Amplitude-domain metrics ──────────────────────────────────────────────

    # Silence: fraction of samples below the noise floor threshold
    silenceFrac = float(np.mean(np.abs(audio) < 0.005))

    # Clipping: fraction of samples at or near ADC saturation
    clippingFrac = float(np.mean(np.abs(audio) > 0.99))

    # Temporal energy variation: std dev of per-frame RMS.
    # A proper flyover event raises energy for several frames then drops;
    # flat noise has near-zero std dev.
    frameSize = max(1, int(sampleRate * 0.1))   # 0.1-second frames ≈ 4410 samples
    nFrames = len(audio) // frameSize
    if nFrames > 1:
        frameRms = np.array([
            np.sqrt(np.mean(audio[i * frameSize:(i + 1) * frameSize] ** 2))
            for i in range(nFrames)
        ])
        frameEnergyStd = float(np.std(frameRms))
    else:
        frameEnergyStd = 0.0

    # Edge-vs-center energy ratio.
    # For a 5-second clip: edges = first 1 s + last 1 s; center = middle 3 s.
    # Well-aligned flyover: center louder than edges (ratio < 1).
    # Inverted or misaligned clip: ratio > 1.
    edgeSamps = min(sampleRate, len(audio) // 4)
    if edgeSamps * 2 < len(audio):
        centerAudio = audio[edgeSamps:-edgeSamps]
        edgeAudio = np.concatenate([audio[:edgeSamps], audio[-edgeSamps:]])
    else:
        centerAudio = audio
        edgeAudio = audio
    centerRms = float(np.sqrt(np.mean(centerAudio ** 2)))
    edgeRms   = float(np.sqrt(np.mean(edgeAudio   ** 2)))
    edgeCenterRatio = edgeRms / max(centerRms, 1e-9)

    # ── Spectral metrics (via STFT magnitude) ─────────────────────────────────
    stft  = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sampleRate, n_fft=1024)

    # Spectral flatness: geometric/arithmetic mean ratio of spectrum per frame.
    # 0 = perfectly tonal (pure tone); 1 = spectrally flat (white noise).
    spectralFlatness = float(np.mean(librosa.feature.spectral_flatness(S=stft)))

    # Spectral centroid: frequency-weighted mean of the spectrum.
    # Very low centroid (<300 Hz) → energy dominated by wind/rumble.
    spectralCentroid = float(np.mean(
        librosa.feature.spectral_centroid(S=stft, sr=sampleRate)
    ))

    # Low-frequency energy ratio: fraction of power below 200 Hz.
    # Wind noise concentrates in the sub-200 Hz band.
    totalEnergy  = float(np.sum(stft ** 2))
    lowFreqEnergy = float(np.sum(stft[freqs < 200.0, :] ** 2))
    lowFreqRatio  = lowFreqEnergy / max(totalEnergy, 1e-10)

    return {
        "silenceFrac":      silenceFrac,
        "clippingFrac":     clippingFrac,
        "frameEnergyStd":   frameEnergyStd,
        "edgeCenterRatio":  edgeCenterRatio,
        "spectralFlatness": spectralFlatness,
        "spectralCentroid": spectralCentroid,
        "lowFreqRatio":     lowFreqRatio,
    }


def _runDeepAnalysis(df: pd.DataFrame) -> tuple[list[dict], int]:
    """
    Run deep metric computation for every row in df.

    Prints progress every 500 clips.  Missing or unreadable files are skipped
    and counted.

    Returns:
        (results, nMissing) where results is a list of per-clip metric dicts.
    """
    results = []
    nMissing = 0
    nTotal = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if i > 0 and i % 500 == 0:
            print(f"  ... {i}/{nTotal} clips processed")
        wavPath = Path(row["filepath"])
        if not wavPath.exists():
            nMissing += 1
            continue
        try:
            metrics = _deepMetrics(str(wavPath))
        except Exception:
            nMissing += 1
            continue
        results.append({
            "filepath":    str(wavPath),
            "rmsDb":       _rmsDb(row["clipRms"]),
            "distanceKm":  float(row.get("distanceKm", float("nan"))),
            "altitudeFt":  float(row.get("altitudeFt", float("nan"))),
            "flightPhase": str(row.get("flightPhase", "")),
            **metrics,
        })

    return results, nMissing


def _printDeepSummary(results: list[dict]) -> None:
    """Print aggregate deep-metric statistics with histograms for key indicators."""
    if not results:
        return

    # Metric label, dict key, display scale factor, unit suffix
    metrics = [
        ("Silence %",        "silenceFrac",       100.0, "%"),
        ("Clipping %",       "clippingFrac",      100.0, "%"),
        ("Frame energy std", "frameEnergyStd",      1.0, ""),
        ("Edge/center ratio","edgeCenterRatio",     1.0, ""),
        ("Spectral flatness","spectralFlatness",    1.0, ""),
        ("Centroid Hz",      "spectralCentroid",    1.0, " Hz"),
        ("Low-freq % <200Hz","lowFreqRatio",       100.0, "%"),
    ]

    print(f"\n{'═'*70}")
    print("  DEEP ANALYSIS SUMMARY")
    print(f"{'═'*70}")
    print(f"  Clips analysed: {len(results)}")
    print(f"\n  {'Metric':<22}  {'Mean':>10}  {'Median':>10}  {'P90':>10}")
    print("  " + "─" * 55)
    for label, key, scale, unit in metrics:
        vals = [r[key] * scale for r in results]
        print(
            f"  {label:<22}  {np.mean(vals):>9.2f}{unit}  "
            f"{np.median(vals):>9.2f}{unit}  "
            f"{np.percentile(vals, 90):>9.2f}{unit}"
        )

    # Histograms for the most diagnostic indicators
    _histo(
        [r["spectralFlatness"] for r in results], bins=8,
        label="Spectral flatness  (0=tonal, 1=noise)",
    )
    _histo(
        [r["lowFreqRatio"] * 100 for r in results], bins=8,
        label="Low-frequency energy ratio %  (high = wind-dominated)",
    )
    _histo(
        [r["frameEnergyStd"] for r in results], bins=8,
        label="Frame energy std dev  (low = flat noise, no event)",
    )
    _histo(
        [r["edgeCenterRatio"] for r in results], bins=8,
        label="Edge/center energy ratio  (>1 = inverted envelope)",
    )
    _histo(
        [r["spectralCentroid"] for r in results], bins=8,
        label="Spectral centroid Hz  (very low = wind/rumble dominated)",
    )


def _printWorstClips(results: list[dict], n: int) -> None:
    """Print the N lowest-RMS clips with all deep metrics."""
    if not results or n <= 0:
        return
    ranked = sorted(results, key=lambda r: r["rmsDb"])[:n]
    print(f"\n{'═'*120}")
    print(f"  WORST {n} CLIPS  (ranked by RMS dBFS ascending)")
    print(f"{'═'*120}")
    print(
        f"  {'RMS dBFS':>8}  {'Sil%':>5}  {'Clip%':>5}  {'Flat':>5}  "
        f"{'Centroid':>8}  {'LF%':>5}  {'ECR':>5}  {'Phase':<12}  Path"
    )
    print("  " + "─" * 114)
    for r in ranked:
        print(
            f"  {r['rmsDb']:>8.1f}  {r['silenceFrac']*100:>5.1f}  "
            f"{r['clippingFrac']*100:>5.2f}  {r['spectralFlatness']:>5.3f}  "
            f"{r['spectralCentroid']:>7.0f}Hz  {r['lowFreqRatio']*100:>5.1f}  "
            f"{r['edgeCenterRatio']:>5.2f}  {r['flightPhase']:<12}  {r['filepath']}"
        )


def _writeBadClips(results: list[dict], outputPath: Path, threshDb: float) -> None:
    """Write one filepath per line for all clips whose RMS is below threshDb."""
    bad = [r["filepath"] for r in results if r["rmsDb"] < threshDb]
    with open(outputPath, "w") as f:
        for p in bad:
            f.write(p + "\n")
    pct = 100.0 * len(bad) / max(len(results), 1)
    print(f"\n  Bad clips: {len(bad)} / {len(results)} ({pct:.1f}%) written to {outputPath}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate audio clip quality per class.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--datasetCsv", type=Path, required=True,
        help="Dataset CSV produced by buildDataset.py "
             "(train.csv, val.csv, or dataset.csv).",
    )
    p.add_argument(
        "--category", type=str, default=None,
        help="Coarse category to focus on (e.g. piston_twin). "
             "Omit for an all-class comparison table only.",
    )
    p.add_argument(
        "--deepAnalysis", action="store_true",
        help="Read each WAV file and compute silence, clipping, spectral, "
             "and temporal metrics.  Slow for large classes; use --maxClips "
             "to cap the number of clips analysed.",
    )
    p.add_argument(
        "--maxClips", type=int, default=None,
        help="Maximum clips to analyse in deep mode.  Clips are sampled "
             "randomly (seed=42).  Default: all clips for the selected class.",
    )
    p.add_argument(
        "--worstN", type=int, default=20,
        help="Number of lowest-RMS clips to print in deep mode (default: 20).",
    )
    p.add_argument(
        "--outputBadClips", type=Path, default=None,
        help="Write one filepath per line for clips below --rmsThresholdDb.  "
             "Requires --deepAnalysis.",
    )
    p.add_argument(
        "--rmsThresholdDb", type=float, default=-55.0,
        help="dBFS threshold below which a clip is flagged as low quality "
             "(default: −55 dBFS).",
    )
    args = p.parse_args()

    if not args.datasetCsv.exists():
        sys.exit(f"CSV not found: {args.datasetCsv}")

    df = pd.read_csv(args.datasetCsv)
    print(f"\nLoaded {len(df)} clips from {args.datasetCsv}")

    # ── All-class RMS summary (always) ────────────────────────────────────────
    summaryRows = _allClassSummary(df, args.rmsThresholdDb)
    _printClassSummary(summaryRows, args.rmsThresholdDb)

    # ── Per-class detail ──────────────────────────────────────────────────────
    if args.category:
        mask = df["type_categories"].apply(
            lambda raw: args.category in json.loads(raw)
        )
        classDf = df[mask].reset_index(drop=True)
        if classDf.empty:
            sys.exit(f"No clips found for category '{args.category}'.")
        print(f"\n  Filtered to '{args.category}': {len(classDf)} clips")

        rmsVals = [_rmsDb(r) for r in classDf["clipRms"]]
        _histo(rmsVals, bins=10, label=f"RMS dBFS  [{args.category}]")
        _printPhaseSummary(classDf, args.category, args.rmsThresholdDb)

        # ── Deep analysis ─────────────────────────────────────────────────────
        if args.deepAnalysis:
            analysisDf = classDf
            if args.maxClips and len(classDf) > args.maxClips:
                analysisDf = classDf.sample(
                    args.maxClips, random_state=42
                ).reset_index(drop=True)
                print(f"\n  Sampling {args.maxClips} of {len(classDf)} clips for deep analysis.")
            else:
                print(f"\n  Running deep analysis on {len(analysisDf)} clips ...")

            results, nMissing = _runDeepAnalysis(analysisDf)
            if nMissing:
                print(f"  [!] {nMissing} clips missing or unreadable — skipped")

            _printDeepSummary(results)
            _printWorstClips(results, args.worstN)

            if args.outputBadClips:
                _writeBadClips(results, args.outputBadClips, args.rmsThresholdDb)

    elif args.deepAnalysis:
        print("\n  [note] --deepAnalysis without --category runs on the full dataset.")
        print("  This may take a long time. Use --category to focus on one class,")
        print("  or --maxClips to limit the number of files read.")

        analysisDf = df
        if args.maxClips and len(df) > args.maxClips:
            analysisDf = df.sample(
                args.maxClips, random_state=42
            ).reset_index(drop=True)
            print(f"\n  Sampling {args.maxClips} of {len(df)} clips.")
        else:
            print(f"\n  Running deep analysis on {len(analysisDf)} clips ...")

        results, nMissing = _runDeepAnalysis(analysisDf)
        if nMissing:
            print(f"  [!] {nMissing} clips missing or unreadable — skipped")

        _printDeepSummary(results)
        _printWorstClips(results, args.worstN)

        if args.outputBadClips:
            _writeBadClips(results, args.outputBadClips, args.rmsThresholdDb)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()

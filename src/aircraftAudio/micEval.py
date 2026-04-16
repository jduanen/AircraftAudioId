#!/usr/bin/env python3
"""
USB microphone evaluation tool.

Evaluates attached audio input devices and ranks them by noise floor, SNR,
and frequency response quality.  No calibrated test equipment required —
the passive phase runs automatically from silence, and the active phase
prompts you to play a reference tone from a nearby speaker.

Metrics computed:
  Passive (silence):
    - Noise floor (dBFS)   — RMS of the captured silence window
    - Self-noise (dBFS)    — minimum RMS across 1-second windows
    - Clipping headroom    — peak amplitude (lower = more headroom remaining)
    - Max sample rate      — highest rate the device accepts

  Active (reference tone, optional):
    - SNR (dB)             — reference RMS minus noise floor
    - Spectral flatness    — how flat the frequency response is (0–1, 1=perfect)
    - Frequency response   — per-band RMS across 8 octave bands (125–16kHz)
"""

import time
import json
import argparse
import sounddevice as sd
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

try:
    from scipy.signal import welch
    from scipy.stats import gmean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


SAMPLE_RATE = 44100
PASSIVE_DURATION_SECS = 10.0
ACTIVE_DURATION_SECS = 10.0

OCTAVE_BANDS = [
    ("125 Hz",   88,   177),
    ("250 Hz",   177,  354),
    ("500 Hz",   354,  707),
    ("1 kHz",    707,  1414),
    ("2 kHz",    1414, 2828),
    ("4 kHz",    2828, 5657),
    ("8 kHz",    5657, 11314),
    ("16 kHz",   11314, 20000),
]


def _rmsDb(samples: np.ndarray) -> float:
    """RMS level in dBFS (0 dBFS = full scale)."""
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms < 1e-10:
        return -120.0
    return 20 * np.log10(rms / 32768.0)


def _peakDb(samples: np.ndarray) -> float:
    peak = np.max(np.abs(samples.astype(np.float64)))
    if peak < 1:
        return -120.0
    return 20 * np.log10(peak / 32768.0)


def _selfNoiseDb(samples: np.ndarray, sampleRate: int) -> float:
    """Minimum RMS across 1-second windows."""
    windowSize = sampleRate
    minRms = float("inf")
    for start in range(0, len(samples) - windowSize, windowSize):
        rms = np.sqrt(np.mean(samples[start:start + windowSize].astype(np.float64) ** 2))
        minRms = min(minRms, rms)
    if minRms < 1e-10 or minRms == float("inf"):
        return -120.0
    return 20 * np.log10(minRms / 32768.0)


def _spectralFlatness(samples: np.ndarray, sampleRate: int) -> Optional[float]:
    """Wiener entropy / spectral flatness in [0, 1]. Requires scipy."""
    if not SCIPY_AVAILABLE:
        return None
    freqs, psd = welch(samples.astype(np.float64), fs=sampleRate, nperseg=2048)
    psd = psd[psd > 0]
    if len(psd) == 0:
        return None
    return float(gmean(psd) / np.mean(psd))


def _octaveBandLevels(samples: np.ndarray, sampleRate: int) -> dict[str, float]:
    """RMS level (dBFS) in each octave band via FFT."""
    N = len(samples)
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    freqs = np.fft.rfftfreq(N, d=1.0 / sampleRate)
    levels = {}
    for label, fLow, fHigh in OCTAVE_BANDS:
        mask = (freqs >= fLow) & (freqs < fHigh)
        if mask.any():
            bandRms = np.sqrt(np.mean(spectrum[mask] ** 2))
            levels[label] = 20 * np.log10(bandRms / (N / 2)) if bandRms > 0 else -120.0
        else:
            levels[label] = -120.0
    return levels


def _recordDevice(deviceIndex: int, durationSecs: float, sampleRate: int) -> Optional[np.ndarray]:
    """Record from a device; return int16 mono array or None on error."""
    try:
        audio = sd.rec(
            int(durationSecs * sampleRate),
            samplerate=sampleRate,
            channels=1,
            dtype="int16",
            device=deviceIndex,
        )
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"  [error] Could not record from device {deviceIndex}: {e}")
        return None


def _listInputDevices() -> list[dict]:
    """Return all input-capable devices as dicts with index."""
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append({"index": i, **d})
    return devices


def _maxSupportedRate(deviceIndex: int) -> int:
    """Find the highest sample rate the device supports from a standard set."""
    candidates = [192000, 96000, 48000, 44100, 22050, 16000, 8000]
    for rate in candidates:
        try:
            sd.check_input_settings(device=deviceIndex, samplerate=rate, channels=1)
            return rate
        except Exception:
            continue
    return 0


def evaluateDevices(
    deviceIndices: Optional[list[int]] = None,
    passiveDurationSecs: float = PASSIVE_DURATION_SECS,
    activeDurationSecs: float = ACTIVE_DURATION_SECS,
    outputDir: Optional[Path] = None,
) -> list[dict]:
    """
    Evaluate microphone devices.

    Args:
        deviceIndices:        Specific device indices to test; None = all input devices.
        passiveDurationSecs:  How long to record silence for passive metrics.
        activeDurationSecs:   How long to record reference tone for active metrics.
        outputDir:            If set, saves per-device WAV recordings and a JSON report.

    Returns:
        List of result dicts, sorted by noise floor (best first).
    """
    if outputDir:
        outputDir.mkdir(parents=True, exist_ok=True)

    allDevices = _listInputDevices()
    if not allDevices:
        print("No input devices found.")
        return []

    if deviceIndices is not None:
        targets = [d for d in allDevices if d["index"] in deviceIndices]
    else:
        targets = allDevices

    if not targets:
        print("No matching input devices found.")
        return []

    print(f"\nFound {len(targets)} input device(s) to evaluate:\n")
    for d in targets:
        print(f"  [{d['index']}] {d['name']}")

    results = []

    # ----------------------------------------------------------------
    # Phase 1: Passive (silence)
    # ----------------------------------------------------------------
    print(f"\n--- PASSIVE PHASE: recording {passiveDurationSecs:.0f}s of silence ---")
    print("Keep the environment as quiet as possible.\n")

    for d in targets:
        idx = d["index"]
        name = d["name"]
        print(f"Recording device [{idx}]: {name} ...", end=" ", flush=True)

        samples = _recordDevice(idx, passiveDurationSecs, SAMPLE_RATE)
        if samples is None:
            results.append({"index": idx, "name": name, "error": True})
            continue

        print("done")

        maxRate = _maxSupportedRate(idx)
        noiseFloor = _rmsDb(samples)
        selfNoise = _selfNoiseDb(samples, SAMPLE_RATE)
        peakHeadroom = _peakDb(samples)

        result = {
            "index": idx,
            "name": name,
            "maxSampleRate": maxRate,
            "noiseFloorDbfs": round(noiseFloor, 1),
            "selfNoiseDbfs": round(selfNoise, 1),
            "peakHeadroomDbfs": round(peakHeadroom, 1),
            "snrDb": None,
            "spectralFlatness": None,
            "octaveBandLevelsDbfs": None,
            "error": False,
        }
        results.append(result)

        if outputDir:
            wavPath = outputDir / f"device{idx}_passive.wav"
            sf.write(str(wavPath), samples.astype(np.float32) / 32768.0, SAMPLE_RATE)

    # ----------------------------------------------------------------
    # Phase 2: Active (reference tone)
    # ----------------------------------------------------------------
    activeDevices = [r for r in results if not r.get("error")]
    if activeDevices:
        print(f"\n--- ACTIVE PHASE: reference tone (optional) ---")
        print("Play a steady tone (e.g., 1 kHz sine) from a phone speaker ~0.5m from the mic.")
        ans = input("Ready to record? [y/N]: ").strip().lower()

        if ans == "y":
            print(f"Recording {activeDurationSecs:.0f}s reference tone ...\n")
            for r in activeDevices:
                idx = r["index"]
                print(f"  Device [{idx}]: {r['name']} ...", end=" ", flush=True)

                samples = _recordDevice(idx, activeDurationSecs, SAMPLE_RATE)
                if samples is None:
                    print("error")
                    continue

                print("done")

                signalDb = _rmsDb(samples)
                r["snrDb"] = round(signalDb - r["noiseFloorDbfs"], 1)
                r["spectralFlatness"] = (
                    round(_spectralFlatness(samples, SAMPLE_RATE), 4)
                    if SCIPY_AVAILABLE else None
                )
                r["octaveBandLevelsDbfs"] = _octaveBandLevels(samples, SAMPLE_RATE)

                if outputDir:
                    wavPath = outputDir / f"device{idx}_active.wav"
                    sf.write(str(wavPath), samples.astype(np.float32) / 32768.0, SAMPLE_RATE)
        else:
            print("Active phase skipped.")

    # ----------------------------------------------------------------
    # Sort and report
    # ----------------------------------------------------------------
    good = [r for r in results if not r.get("error")]
    good.sort(key=lambda r: r["noiseFloorDbfs"])

    _printReport(good)

    if outputDir:
        reportPath = outputDir / "micEvalReport.json"
        with open(reportPath, "w") as f:
            json.dump(good, f, indent=2)
        print(f"\nReport saved to {reportPath}")

    return good


def _printReport(results: list[dict]) -> None:
    if not results:
        return

    header = (
        f"\n{'Rank':<5} {'Dev':>4}  {'Name':<32}  "
        f"{'Noise floor':>12}  {'Self-noise':>10}  {'Peak hdroom':>11}  "
        f"{'SNR':>8}  {'MaxRate':>8}"
    )
    print("\n" + "="*len(header.expandtabs()))
    print("RESULTS (ranked best → worst noise floor)")
    print("="*len(header.expandtabs()))
    print(header)
    print("-"*len(header.expandtabs()))

    for i, r in enumerate(results, 1):
        snr = f"{r['snrDb']:.1f} dB" if r["snrDb"] is not None else "   n/a"
        print(
            f"{i:<5} [{r['index']:>2}]  {r['name']:<32}  "
            f"{r['noiseFloorDbfs']:>9.1f} dBFS  "
            f"{r['selfNoiseDbfs']:>8.1f} dBFS  "
            f"{r['peakHeadroomDbfs']:>9.1f} dBFS  "
            f"{snr:>8}  "
            f"{r['maxSampleRate']:>7} Hz"
        )

    print()
    best = results[0]
    print(f"Recommended: device [{best['index']}] — {best['name']}")
    print(f"  Use --device {best['index']} with scripts/capture.py")


def buildArgParser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate and compare USB microphones on the Pi.")
    p.add_argument("--duration", type=float, default=PASSIVE_DURATION_SECS,
                   dest="passiveDuration", help="Passive recording duration in seconds")
    p.add_argument("--outputDir", type=Path, default=None,
                   help="Save WAV recordings and JSON report here")
    p.add_argument("--devices", type=str, default=None,
                   help="Comma-separated device indices to test (default: all input devices)")
    return p


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    indices = (
        [int(x) for x in args.devices.split(",")]
        if args.devices else None
    )
    evaluateDevices(
        deviceIndices=indices,
        passiveDurationSecs=args.passiveDuration,
        outputDir=args.outputDir,
    )

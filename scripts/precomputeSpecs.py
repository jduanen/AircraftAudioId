#!/usr/bin/env python3
"""
Pre-compute mel spectrograms for all clips in train/val CSVs.
Saves <clip>.spec.npy alongside each WAV. Run once before training.

Usage:
    python scripts/precomputeSpecs.py \
        --trainCsv dataset/train.csv \
        --valCsv dataset/val.csv \
        [--workers 16]
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aircraftClassifier.training.toolchain import SAMPLE_RATE, CLIP_SECS


def _computeAndSave(wavPath: str) -> None:
    specPath = Path(wavPath).parent / (Path(wavPath).stem + ".spec.npy")
    if specPath.exists():
        return

    targetLen = int(SAMPLE_RATE * CLIP_SECS)
    waveform, _ = librosa.load(wavPath, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECS)
    if len(waveform) < targetLen:
        waveform = np.pad(waveform, (0, targetLen - len(waveform)))

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128,
    )
    np.save(specPath, librosa.power_to_db(mel, top_db=80).astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trainCsv", required=True)
    p.add_argument("--valCsv",   required=True)
    p.add_argument("--workers",  type=int, default=16)
    args = p.parse_args()

    paths = pd.concat([
        pd.read_csv(args.trainCsv),
        pd.read_csv(args.valCsv),
    ])["filepath"].unique().tolist()

    print(f"Pre-computing {len(paths)} spectrograms with {args.workers} workers...")
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_computeAndSave, paths), 1):
            if i % 200 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}")
    print("Done.")


if __name__ == "__main__":
    main()

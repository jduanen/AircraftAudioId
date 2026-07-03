#!/usr/bin/env python3
"""
Pre-compute PANNs CNN14 embeddings for all clips in train/val CSVs.
Saves <clip>.panns.npy (2048-dim float32) alongside each WAV.
Run once before training with --backbone panns.

PANNs is pretrained on AudioSet (527 classes including aircraft, helicopter,
jet engine, and propeller sounds); the frozen embeddings transfer far better
to this task than ImageNet features.

Usage:
    python scripts/precomputeEmbeddings.py \
        --trainCsv dataset/train.csv \
        --valCsv dataset/val.csv \
        [--skipExisting]   # skip clips that already have a .panns.npy; default recomputes all
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
from panns_inference import AudioTagging

# PANNs CNN14 was trained on 32 kHz audio.
PANNS_SAMPLE_RATE = 32000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trainCsv",     required=True)
    p.add_argument("--valCsv",       required=True)
    p.add_argument("--skipExisting", action="store_true",
                   help="Skip clips that already have a .panns.npy file (default: recompute all)")
    args = p.parse_args()

    paths = pd.concat([
        pd.read_csv(args.trainCsv),
        pd.read_csv(args.valCsv),
    ])["filepath"].unique().tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading PANNs CNN14 on {device} (first run downloads a ~300 MB checkpoint to ~/panns_data)...")
    model = AudioTagging(checkpoint_path=None, device=device)

    action = "Skipping existing, computing new" if args.skipExisting else "Re-computing all"
    print(f"{action}: {len(paths)} embeddings...")
    for i, wavPath in enumerate(paths, 1):
        embPath = Path(wavPath).parent / (Path(wavPath).stem + ".panns.npy")
        if not (args.skipExisting and embPath.exists()):
            audio, _ = librosa.load(wavPath, sr=PANNS_SAMPLE_RATE, mono=True)
            _, embedding = model.inference(audio[None, :])
            np.save(embPath, embedding[0].astype(np.float32))
        if i % 200 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}")
    print("Done.")


if __name__ == "__main__":
    main()

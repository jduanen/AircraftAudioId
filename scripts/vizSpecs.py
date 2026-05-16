#!/usr/bin/env python3
"""
Visualize mel spectrograms from the dataset.

Loads from <clip>.spec.npy if pre-computed, otherwise falls back to librosa.

Usage:
    python scripts/vizSpecs.py --csv dataset/train.csv
    python scripts/vizSpecs.py --csv dataset/train.csv --category helicopter --n 12
    python scripts/vizSpecs.py --csv dataset/train.csv --output specs.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aircraftClassifier.training.toolchain import SAMPLE_RATE, CLIP_SECS


def _loadSpec(wavPath: str) -> np.ndarray:
    specPath = Path(wavPath).parent / (Path(wavPath).stem + ".spec.npy")
    if specPath.exists():
        return np.load(specPath)
    import librosa
    targetLen = int(SAMPLE_RATE * CLIP_SECS)
    waveform, _ = librosa.load(wavPath, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECS)
    if len(waveform) < targetLen:
        waveform = np.pad(waveform, (0, targetLen - len(waveform)))
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128,
    )
    return librosa.power_to_db(mel, top_db=80).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      required=True, help="Path to train.csv or val.csv")
    p.add_argument("--n",        type=int, default=9,   help="Number of clips to display (default: 9)")
    p.add_argument("--cols",     type=int, default=3,   help="Grid columns (default: 3)")
    p.add_argument("--category", type=str, default=None, help="Filter to a single coarse category")
    p.add_argument("--output",   type=str, default=None, help="Save to file instead of displaying")
    p.add_argument("--seed",     type=int, default=42,  help="Random seed for sampling (change to see different clips)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    labelCol = "type_categories" if "type_categories" in df.columns else "vehicle_types"

    if args.category:
        df = df[df[labelCol].apply(lambda x: args.category in json.loads(x))]
        if df.empty:
            sys.exit(f"No clips found for category '{args.category}'")

    sample = df.sample(min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    cols = args.cols
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.8))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(sample.iterrows()):
        spec = _loadSpec(row["filepath"])
        axes[i].imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        axes[i].set_title(", ".join(json.loads(row[labelCol])), fontsize=8)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for ax in axes[len(sample):]:
        ax.set_visible(False)

    title = Path(args.csv).name
    if args.category:
        title += f" — {args.category}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
```

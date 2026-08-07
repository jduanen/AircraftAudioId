#!/usr/bin/env python3
"""
Visualize mel spectrograms from the dataset.

Loads from <clip>.spec.npy if pre-computed, otherwise falls back to librosa.

Usage:
    python scripts/vizSpecs.py --csv dataset/train.csv
    python scripts/vizSpecs.py --csv dataset/train.csv --category helicopter --n 12
    python scripts/vizSpecs.py --csv dataset/train.csv --output specs.png
    python scripts/vizSpecs.py --csv dataset/train.csv --category helicopter --flagOutput bad_helicopter.csv
"""

import argparse
import json
import sys
from pathlib import Path

import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aircraftClassifier.training.toolchain import SAMPLE_RATE, CLIP_SECS, _dualBandMelDb


def _loadSpec(wavPath: str, channel: int) -> np.ndarray:
    """Return one band (0=low, 1=high) of the dual-band mel spectrogram, shape (N_MELS, nFrames)."""
    specPath = Path(wavPath).parent / (Path(wavPath).stem + ".spec.npy")
    if specPath.exists():
        return np.load(specPath)[channel]
    import librosa
    targetLen = int(SAMPLE_RATE * CLIP_SECS)
    waveform, _ = librosa.load(wavPath, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECS)
    if len(waveform) < targetLen:
        waveform = np.pad(waveform, (0, targetLen - len(waveform)))
    return _dualBandMelDb(waveform)[channel]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      required=True, help="Path to train.csv or val.csv")
    p.add_argument("--n",        type=int, default=9,   help="Number of clips to display (default: 9)")
    p.add_argument("--cols",     type=int, default=3,   help="Grid columns (default: 3)")
    p.add_argument("--category", type=str, default=None, help="Filter to a single coarse category")
    p.add_argument("--output",   type=str, default=None, help="Save to file instead of displaying")
    p.add_argument("--seed",     type=int, default=42,  help="Random seed for sampling (change to see different clips)")
    p.add_argument("--play",     action="store_true",   help="Click a spectrogram to play its audio")
    p.add_argument("--channel",  type=int, default=0, choices=[0, 1],
                   help="Which dual-band channel to display: 0=low band (<=8kHz), 1=high band (>=8kHz). Default: 0")
    p.add_argument("--flagOutput", type=str, default=None,
                   help="Click a spectrogram to flag it for exclusion (red border); flagged filepaths are "
                        "written to this CSV after every click. Re-running with the same --flagOutput file "
                        "loads and extends the existing flags, so you can page through a category across "
                        "multiple --seed runs and accumulate into one file. If --play is also given, "
                        "left-click plays and right-click flags; otherwise any click flags.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    labelCol = "type_categories" if "type_categories" in df.columns else "vehicle_types"

    if args.category:
        df = df[df[labelCol].apply(lambda x: args.category in json.loads(x))]
        if df.empty:
            sys.exit(f"No clips found for category '{args.category}'")

    sample = df.sample(min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    flaggedPaths = set()
    if args.flagOutput and Path(args.flagOutput).exists():
        flaggedPaths = set(pd.read_csv(args.flagOutput)["filepath"])
        print(f"Loaded {len(flaggedPaths)} previously-flagged clip(s) from {args.flagOutput}")

    cols = args.cols
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.8))
    axes = np.array(axes).flatten()

    axToPath = {}
    for i, (_, row) in enumerate(sample.iterrows()):
        spec = _loadSpec(row["filepath"], args.channel)
        axes[i].imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        axes[i].set_title(", ".join(json.loads(row[labelCol])), fontsize=8)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axToPath[axes[i]] = row["filepath"]
        if row["filepath"] in flaggedPaths:
            for spine in axes[i].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    for ax in axes[len(sample):]:
        ax.set_visible(False)

    title = Path(args.csv).name
    title += f" — {'low' if args.channel == 0 else 'high'} band"
    if args.category:
        title += f" — {args.category}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if args.play or args.flagOutput:
        if args.play:
            import sounddevice as sd
            import soundfile as sf

        def _playAudio(path):
            print(f"Playing: {Path(path).name}")
            def _play():
                audio, sr = sf.read(path, dtype="float32", always_2d=False)
                sd.stop()
                sd.play(audio, sr)
            threading.Thread(target=_play, daemon=True).start()

        def _toggleFlag(ax, path):
            if path in flaggedPaths:
                flaggedPaths.discard(path)
                color, width = "black", 0.8
                print(f"Unflagged ({len(flaggedPaths)} total): {Path(path).name}")
            else:
                flaggedPaths.add(path)
                color, width = "red", 3
                print(f"Flagged ({len(flaggedPaths)} total): {Path(path).name}")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(width)
            fig.canvas.draw_idle()
            pd.DataFrame({"filepath": sorted(flaggedPaths)}).to_csv(args.flagOutput, index=False)

        def _onclick(event):
            if event.inaxes not in axToPath:
                return
            path = axToPath[event.inaxes]
            if args.flagOutput and (not args.play or event.button == 3):
                _toggleFlag(event.inaxes, path)
            elif args.play and event.button == 1:
                _playAudio(path)

        fig.canvas.mpl_connect("button_press_event", _onclick)
        if args.play and args.flagOutput:
            print("Left-click to play audio, right-click to flag for exclusion.")
        elif args.play:
            print("Click a spectrogram to play its audio.")
        else:
            print(f"Click a spectrogram to flag it for exclusion (-> {args.flagOutput}).")

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Per-class evaluation of a trained checkpoint, and single-WAV inference.

Usage — evaluate on a validation CSV:
    python scripts/evalModel.py \
        --checkpoint checkpoints/best.ckpt \
        --labelEncoder checkpoints/labelEncoder.json \
        --valCsv dataset/val.csv \
        [--useCategories] [--threshold 0.5] [--tuneThresholds]

    # If labelEncoder.json is missing (older checkpoint), rebuild from both CSVs:
    python scripts/evalModel.py \
        --checkpoint checkpoints/best.ckpt \
        --trainCsv dataset/train.csv \
        --valCsv dataset/val.csv \
        [--useCategories]

Usage — quick inference on a single WAV:
    python scripts/evalModel.py \
        --checkpoint checkpoints/best.ckpt \
        --labelEncoder checkpoints/labelEncoder.json \
        --wav /path/to/clip.wav
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import librosa
import numpy as np
import torch
import pandas as pd

from aircraftClassifier.training.toolchain import (
    VehicleAudioDataset,
    VehicleSoundClassifier,
    buildLabelEncoder,
    SAMPLE_RATE,
    CLIP_SECS,
    _dualBandMelDb,
)


def _loadLabelEncoder(args) -> dict[str, int]:
    if args.labelEncoder and Path(args.labelEncoder).exists():
        with open(args.labelEncoder) as f:
            return json.load(f)

    if args.trainCsv and args.valCsv:
        trainDf = pd.read_csv(args.trainCsv)
        valDf = pd.read_csv(args.valCsv)
        combined = pd.concat([trainDf, valDf], ignore_index=True)
        return buildLabelEncoder(combined, useCategories=args.useCategories)

    if args.valCsv:
        valDf = pd.read_csv(args.valCsv)
        return buildLabelEncoder(valDf, useCategories=args.useCategories)

    raise ValueError(
        "Provide --labelEncoder (checkpoints/labelEncoder.json) "
        "or --trainCsv + --valCsv to rebuild the label mapping."
    )


def _loadModel(checkpointPath: str, nClasses: int, device: torch.device) -> VehicleSoundClassifier:
    ckpt = torch.load(checkpointPath, map_location="cpu", weights_only=False)
    ckptNClasses = ckpt.get("hyper_parameters", {}).get("nClasses", nClasses)
    if ckptNClasses != nClasses:
        print(
            f"  Warning: checkpoint has {ckptNClasses} output classes "
            f"but label encoder has {nClasses}."
        )
    model = VehicleSoundClassifier.load_from_checkpoint(
        checkpointPath,
        nClasses=ckptNClasses,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def _inferWav(wavPath: str, model: VehicleSoundClassifier, labelEncoder: dict[str, int],
              threshold: float, device: torch.device, backbone: str = "resnet18") -> None:
    if backbone == "panns":
        from panns_inference import AudioTagging
        audio, _ = librosa.load(wavPath, sr=32000, mono=True)
        tagger = AudioTagging(checkpoint_path=None, device=device.type)
        _, embedding = tagger.inference(audio[None, :])
        x = torch.from_numpy(embedding).float().to(device)
    else:
        targetLen = int(SAMPLE_RATE * CLIP_SECS)
        waveform, _ = librosa.load(wavPath, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECS)
        if len(waveform) < targetLen:
            waveform = np.pad(waveform, (0, targetLen - len(waveform)))

        x = torch.from_numpy(_dualBandMelDb(waveform)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    indexToLabel = {v: k for k, v in labelEncoder.items()}
    print(f"\nPredictions for: {wavPath}")
    print(f"{'Class':<25}  {'Prob':>6}  {'Active':>6}")
    print("─" * 45)
    for i, p in enumerate(sorted(range(len(probs)), key=lambda x: -probs[x])):
        label = indexToLabel.get(p, f"class_{p}")
        active = "✓" if probs[p] >= threshold else ""
        print(f"  {label:<23}  {probs[p]:6.3f}  {active:>6}")


def _confusionBreakdown(
    allProbs: np.ndarray,
    allLabels: np.ndarray,
    indexToLabel: dict[int, str],
    targetClass: str,
) -> None:
    """
    For clips whose ground truth includes targetClass, show what the model
    actually predicts: mean probability assigned to every class, and how
    often each class wins the argmax (i.e. what the model "thinks" the
    clip is, treating the top-1 prediction as its best guess).
    """
    labelToIndex = {v: k for k, v in indexToLabel.items()}
    if targetClass not in labelToIndex:
        print(f"\n  [confusion] Unknown class '{targetClass}' — skipping.")
        return

    targetIdx = labelToIndex[targetClass]
    mask = allLabels[:, targetIdx] == 1
    nTarget = int(mask.sum())
    if nTarget == 0:
        print(f"\n  [confusion] No val clips labeled '{targetClass}' — skipping.")
        return

    subProbs = allProbs[mask]
    meanProb = subProbs.mean(axis=0)
    top1 = subProbs.argmax(axis=1)
    nClasses = allProbs.shape[1]
    top1Counts = np.bincount(top1, minlength=nClasses)

    print(f"\n{'═'*60}")
    print(f"  CONFUSION BREAKDOWN — true class: {targetClass}  (n={nTarget})")
    print(f"{'═'*60}")
    print(f"  {'Class':<22}  {'Mean Prob':>9}  {'Top-1 Count':>11}  {'Top-1 %':>8}")
    print("  " + "─" * 56)
    order = sorted(range(nClasses), key=lambda c: -meanProb[c])
    for c in order:
        label = indexToLabel.get(c, f"class_{c}")
        marker = " *" if c == targetIdx else "  "
        print(
            f"  {label:<22}{marker}  {meanProb[c]:9.3f}  {top1Counts[c]:11d}  "
            f"{100*top1Counts[c]/nTarget:7.1f}%"
        )
    print("  " + "─" * 56)
    print(f"  (* = the true class itself)")


def _evalCsv(
    valCsv: str,
    labelEncoder: dict[str, int],
    model: VehicleSoundClassifier,
    useCategories: bool,
    threshold: float,
    tuneThresholds: bool,
    saveThresholds: str | None,
    batchSize: int,
    workers: int,
    device: torch.device,
    backbone: str = "resnet18",
    confusionFor: str | None = None,
) -> None:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        average_precision_score,
        f1_score,
    )

    valDf = pd.read_csv(valCsv)
    ds = VehicleAudioDataset(valDf, labelEncoder, augment=False, useCategories=useCategories,
                             backbone=backbone)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batchSize, shuffle=False,
        num_workers=workers, pin_memory=(device.type == "cuda"),
    )

    allProbs = []
    allLabels = []

    model.eval()
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            logits = model(specs)
            probs = torch.sigmoid(logits).cpu().numpy()
            allProbs.append(probs)
            allLabels.append(labels.numpy())

    allProbs = np.concatenate(allProbs, axis=0)     # (N, C)
    allLabels = np.concatenate(allLabels, axis=0)   # (N, C)

    indexToLabel = {v: k for k, v in labelEncoder.items()}
    nClasses = len(labelEncoder)

    # Per-class AP
    apScores = []
    for c in range(nClasses):
        if allLabels[:, c].sum() > 0:
            ap = average_precision_score(allLabels[:, c], allProbs[:, c])
        else:
            ap = float("nan")
        apScores.append(ap)
    mAP = np.nanmean(apScores)

    # Per-class F1 at fixed threshold
    thresholds = np.full(nClasses, threshold)

    if tuneThresholds:
        print(f"\nTuning per-class thresholds on val set...")
        for c in range(nClasses):
            if allLabels[:, c].sum() == 0:
                continue
            bestF1, bestT = 0.0, threshold
            for t in np.linspace(0.1, 0.9, 33):
                preds = (allProbs[:, c] >= t).astype(int)
                f1 = f1_score(allLabels[:, c], preds, zero_division=0)
                if f1 > bestF1:
                    bestF1, bestT = f1, t
            thresholds[c] = bestT

        if saveThresholds:
            thresholdMap = {indexToLabel[c]: float(thresholds[c]) for c in range(nClasses)}
            with open(saveThresholds, "w") as f:
                json.dump(thresholdMap, f, indent=2)
            print(f"  Thresholds saved to {saveThresholds}")

    binaryPreds = (allProbs >= thresholds).astype(int)
    prec, rec, f1, support = precision_recall_fscore_support(
        allLabels, binaryPreds, average=None, zero_division=0
    )
    macroF1 = f1_score(allLabels, binaryPreds, average="macro", zero_division=0)

    # Print table
    print(f"\n{'═'*72}")
    print("  PER-CLASS EVALUATION")
    print(f"  Checkpoint threshold: {threshold:.2f}  |  Tuned: {'yes' if tuneThresholds else 'no'}")
    print(f"{'═'*72}")
    hdr = f"  {'Class':<22}  {'Thresh':>6}  {'AP':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Support':>7}"
    print(hdr)
    print("  " + "─" * 68)

    sortedClasses = sorted(range(nClasses), key=lambda c: -apScores[c] if not np.isnan(apScores[c]) else -1)
    for c in sortedClasses:
        label = indexToLabel.get(c, f"class_{c}")
        apStr = f"{apScores[c]:.3f}" if not np.isnan(apScores[c]) else "  N/A"
        print(
            f"  {label:<22}  {thresholds[c]:6.2f}  {apStr:>6}  "
            f"{f1[c]:6.3f}  {prec[c]:6.3f}  {rec[c]:6.3f}  {int(support[c]):>7}"
        )

    print("  " + "─" * 68)
    print(f"  {'MACRO':<22}  {'':>6}  {mAP:6.3f}  {macroF1:6.3f}")
    print(f"{'═'*72}")
    print(f"\n  Val clips: {len(valDf)}  |  Classes: {nClasses}  |  mAP: {mAP:.4f}  |  Macro-F1: {macroF1:.4f}")

    # Class imbalance snapshot
    positiveRates = allLabels.mean(axis=0)
    print(f"\n  Most imbalanced classes (positive rate):")
    for c in sorted(range(nClasses), key=lambda c: positiveRates[c]):
        label = indexToLabel.get(c, f"class_{c}")
        print(f"    {label:<25}  {positiveRates[c]*100:5.1f}%")

    if confusionFor:
        _confusionBreakdown(allProbs, allLabels, indexToLabel, confusionFor)


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    p.add_argument("--checkpoint",     required=True, type=str, help="Path to .ckpt file")
    p.add_argument("--labelEncoder",   type=str, default=None,
                   help="Path to labelEncoder.json saved by training (checkpoints/labelEncoder.json)")
    p.add_argument("--valCsv",         type=str, default=None, help="Validation CSV for bulk eval")
    p.add_argument("--trainCsv",       type=str, default=None,
                   help="Training CSV — needed to rebuild label encoder if labelEncoder.json is missing")
    p.add_argument("--wav",            type=str, default=None, help="Single WAV file for quick inference")
    p.add_argument("--useCategories",  action="store_true",
                   help="Use type_categories column (match the --useCategories flag used during training)")
    p.add_argument("--threshold",      type=float, default=0.5,
                   help="Classification threshold (default: 0.5)")
    p.add_argument("--tuneThresholds", action="store_true",
                   help="Find per-class optimal threshold on the val set (reported alongside fixed threshold)")
    p.add_argument("--saveThresholds", type=str, default=None,
                   help="Write tuned per-class thresholds to this JSON file (requires --tuneThresholds)")
    p.add_argument("--confusionFor",   type=str, default=None,
                   help="For clips truly labeled this class, show mean predicted probability "
                        "and top-1 prediction breakdown across all classes (which classes the "
                        "model actually confuses this one with).")
    p.add_argument("--batchSize",      type=int, default=64)
    p.add_argument("--workers",        type=int, default=4)
    args = p.parse_args()

    if args.valCsv is None and args.wav is None:
        p.error("Provide --valCsv for bulk eval or --wav for single-file inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labelEncoder = _loadLabelEncoder(args)
    nClasses = len(labelEncoder)
    print(f"Classes ({nClasses}): {', '.join(sorted(labelEncoder))}")

    model = _loadModel(args.checkpoint, nClasses, device)

    # Older checkpoints predate the backbone hparam; they are all ResNet.
    backbone = model.hparams.get("backbone", "resnet18")
    if backbone != "resnet18":
        print(f"Backbone: {backbone}")

    if model.hparams.nClasses != nClasses:
        sys.exit(
            f"  Checkpoint outputs {model.hparams.nClasses} classes but label encoder has "
            f"{nClasses} — cannot evaluate.\n"
            f"  Use the labelEncoder_<timestamp>.json saved alongside this checkpoint, or\n"
            f"  rebuild: --trainCsv dataset/train.csv --valCsv dataset/val.csv (without --labelEncoder)."
        )

    if args.wav:
        _inferWav(args.wav, model, labelEncoder, args.threshold, device, backbone=backbone)

    if args.valCsv:
        _evalCsv(
            valCsv=args.valCsv,
            labelEncoder=labelEncoder,
            model=model,
            useCategories=args.useCategories,
            threshold=args.threshold,
            tuneThresholds=args.tuneThresholds,
            saveThresholds=args.saveThresholds,
            batchSize=args.batchSize,
            workers=args.workers,
            device=device,
            backbone=backbone,
            confusionFor=args.confusionFor,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training toolchain: dataset, model, and training loop.

Expects CSVs produced by scripts/buildDataset.py (one row per clip).
Required columns: filepath, vehicle_types (JSON list), recordingId.
Optional columns used when present: type_categories, directionClass, velocityKts.

Usage:
    python -m aircraftClassifier.training.toolchain \
        --trainCsv dataset/train.csv --valCsv dataset/val.csv --useCategories
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchvision import models

from ..augmentation.audioAug import buildAugPipeline


# Audio pipeline constants. Clips are resampled to SAMPLE_RATE on load;
# recorder.py saves at 44100 Hz so resampling always occurs.
SAMPLE_RATE = 44100  # 22050
CLIP_SECS   = 5.0

# Mel spectrogram config. n_fft=2048 (up from 1024) halves the underlying FFT
# bin width (43.1 Hz -> 21.5 Hz). See DESIGN_NOTES.md "Experiment Log —
# Backbone & Spectrogram Investigation" for the analysis behind these values.
#
# Dual-band, non-overlapping mel channels instead of one global fmax: a single
# cutoff can't serve every class — fmax=8000 helped helicopter/turboprop
# (whose signal is concentrated low) but hurt widebody_jet/piston_single
# (which have real content in 8-12 kHz); fmax=12000 was the reverse. Each
# channel gets the full N_MELS budget for its own band instead of splitting it.
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
FMAX_LOW   = 8000    # channel 0: 0-8000 Hz, full N_MELS bins
FMIN_HIGH  = 8000     # channel 1: 8000 Hz-Nyquist, full N_MELS bins


def _dualBandMelDb(waveform: np.ndarray) -> np.ndarray:
    """Compute two non-overlapping mel-band spectrograms and stack as (2, N_MELS, nFrames)."""
    low = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX_LOW,
    )
    high = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN_HIGH,
    )
    return np.stack([
        librosa.power_to_db(low, top_db=80),
        librosa.power_to_db(high, top_db=80),
    ]).astype(np.float32)


def _specAugment(spec: torch.Tensor, freqMaskF: int = 20, timeMaskT: int = 40) -> torch.Tensor:
    spec = spec.clone()
    _, nMels, nFrames = spec.shape
    f = int(torch.randint(0, freqMaskF + 1, (1,)))
    f0 = int(torch.randint(0, nMels - f + 1, (1,)))
    spec[:, f0:f0 + f, :] = 0.0
    t = int(torch.randint(0, timeMaskT + 1, (1,)))
    t0 = int(torch.randint(0, nFrames - t + 1, (1,)))
    spec[:, :, t0:t0 + t] = 0.0
    return spec


def buildLabelEncoder(df: pd.DataFrame, useCategories: bool = False) -> dict[str, int]:
    """
    Build a stable label-string → class-index mapping from the full dataset.
    Must be computed before splitting so train and val share the same mapping.

    Args:
        useCategories: If True, encode from the coarse type_categories column
                       instead of the raw vehicle_types strings.  Requires the
                       dataset to have been built with the updated clipExport.py.
    """
    col = "type_categories" if useCategories and "type_categories" in df.columns else "vehicle_types"
    allTypes = sorted(
        {t for types in df[col].apply(json.loads) for t in types}
    )
    return {t: i for i, t in enumerate(allTypes)}


def computePosWeight(df: pd.DataFrame, labelEncoder: dict[str, int], labelCol: str) -> torch.Tensor:
    """
    Compute per-class positive weight for BCEWithLogitsLoss to handle class imbalance.
    pos_weight[i] = (number of negative samples) / (number of positive samples) for class i.
    Clipped to [0.1, 10] to prevent extreme values from rare/common classes.
    Classes with fewer than ~10% of the majority class should be collected more aggressively
    rather than compensated for with extreme pos_weight values.
    """
    nSamples = len(df)
    counts = torch.zeros(len(labelEncoder))
    for labels in df[labelCol].apply(json.loads):
        for t in labels:
            if t in labelEncoder:
                counts[labelEncoder[t]] += 1
    negCounts = nSamples - counts
    posWeight = torch.clamp(negCounts / counts.clamp(min=1), min=0.1, max=10.0)
    return posWeight


class VehicleAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labelEncoder: dict[str, int],
        augment: bool = False,
        bgNoiseDir: str | None = None,
        useCategories: bool = False,
        backbone: str = "resnet18",
    ):
        self.df = df.reset_index(drop=True)
        self.labelEncoder = labelEncoder
        self.nClasses = len(labelEncoder)
        self.backbone = backbone
        labelCol = (
            "type_categories"
            if useCategories and "type_categories" in df.columns
            else "vehicle_types"
        )
        self._parsedLabels: list[list[str]] = [
            json.loads(v) for v in self.df[labelCol]
        ]
        self._filepaths: list[str] = self.df["filepath"].tolist()
        self.targetLen = int(SAMPLE_RATE * CLIP_SECS)
        self.augment = augment
        self.augmentFn = buildAugPipeline(bgNoiseDir=bgNoiseDir) if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        wavPath = self._filepaths[idx]

        if self.backbone == "panns":
            embPath = Path(wavPath).parent / (Path(wavPath).stem + ".panns.npy")
            if not embPath.exists():
                raise FileNotFoundError(
                    f"Missing PANNs embedding {embPath} — "
                    f"run scripts/precomputeEmbeddings.py first."
                )
            emb = torch.from_numpy(np.load(embPath)).float()
            typeLabel = torch.zeros(self.nClasses)
            for t in self._parsedLabels[idx]:
                if t in self.labelEncoder:
                    typeLabel[self.labelEncoder[t]] = 1.0
            return emb, typeLabel

        specPath = Path(wavPath).parent / (Path(wavPath).stem + ".spec.npy")

        if specPath.exists():
            spec = torch.from_numpy(np.load(specPath)).float()
            if self.augment:
                spec = _specAugment(spec)
        else:
            waveform, _ = librosa.load(wavPath, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECS)
            if len(waveform) < self.targetLen:
                waveform = np.pad(waveform, (0, self.targetLen - len(waveform)))
            if self.augmentFn:
                waveform = self.augmentFn(waveform, sample_rate=SAMPLE_RATE)
            spec = torch.from_numpy(_dualBandMelDb(waveform)).float()

        typeLabel = torch.zeros(self.nClasses)
        for t in self._parsedLabels[idx]:
            if t in self.labelEncoder:
                typeLabel[self.labelEncoder[t]] = 1.0

        return spec, typeLabel


class VehicleSoundClassifier(pl.LightningModule):
    def __init__(
        self,
        nClasses: int,
        lr: float = 1e-4,
        maxEpochs: int = 50,
        posWeight: torch.Tensor | None = None,
        weightDecay: float = 0.01,
        freezeBackbone: bool = False,
        unfreezeEpoch: int | None = None,
        backbone: str = "resnet18",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["posWeight"])
        # persistent=False: posWeight is training-only; exclude from checkpoint so
        # load_from_checkpoint doesn't hit a state_dict mismatch at eval time.
        self.register_buffer("posWeight", posWeight, persistent=False)

        if backbone == "panns":
            # Input is a precomputed 2048-dim PANNs embedding; the "backbone"
            # already ran offline in scripts/precomputeEmbeddings.py.
            self.backbone = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, nClasses),
            )
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

            # Dual-stem: each frequency band (see _dualBandMelDb) gets its own
            # stem so band-specific features develop before the bands are
            # merged, instead of a single Conv2d(2,64,...) that blends both
            # bands from the very first layer (tried first; didn't cleanly
            # separate the two classes of results — see DESIGN_NOTES.md).
            # Both stems are fresh-initialized (mirrors the original single-
            # channel conv1 replacement, which also discarded pretrained
            # weights), so there's no pretrained weight to remap here either.
            def _makeStem() -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

            self.stemLow  = _makeStem()   # channel 0: 0-8000 Hz
            self.stemHigh = _makeStem()   # channel 1: 8000 Hz-Nyquist
            # Fuse the two 64-channel stem outputs (concat -> 128ch) back down
            # to 64ch so the pretrained layer1-4 trunk sees the input width
            # it was designed for.
            self.fuse = nn.Conv2d(128, 64, kernel_size=1, bias=False)
            # Shared trunk: pretrained layer1-4 + avgpool, untouched.
            self.trunk = nn.Sequential(*list(resnet.children())[4:-1])

            # Freeze both stems + fuse + layer1-3; keep layer4 + avgpool
            # trainable. Unfreezes fully at unfreezeEpoch if set.
            if freezeBackbone:
                for module in (self.stemLow, self.stemHigh, self.fuse):
                    for param in module.parameters():
                        param.requires_grad = False
                for i, child in enumerate(self.trunk.children()):
                    if i < 3:  # layer1, layer2, layer3; layer4(3)+avgpool(4) stay trainable
                        for param in child.parameters():
                            param.requires_grad = False

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, nClasses),
            )

        self.trainF1 = torchmetrics.F1Score(
            task="multilabel", num_labels=nClasses, average="macro"
        )
        self.valF1 = torchmetrics.F1Score(
            task="multilabel", num_labels=nClasses, average="macro"
        )
        self.valMap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=nClasses, average="macro"
        )

    def forward(self, x):
        if self.hparams.backbone == "panns":
            return self.classifier(self.backbone(x))
        low  = self.stemLow(x[:, 0:1])
        high = self.stemHigh(x[:, 1:2])
        fused = self.fuse(torch.cat([low, high], dim=1))
        return self.classifier(self.trunk(fused))

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.BCEWithLogitsLoss(pos_weight=self.posWeight)(logits, labels)

    def training_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        loss = self._loss(logits, labels)
        self.trainF1(torch.sigmoid(logits), labels.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.trainF1, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        preds = torch.sigmoid(logits)
        loss = self._loss(logits, labels)
        self.valF1(preds, labels.int())
        self.valMap(preds, labels.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1",  self.valF1,  on_epoch=True, prog_bar=True)
        self.log("val_mAP", self.valMap, on_epoch=True)

    def on_train_epoch_start(self):
        if (
            self.hparams.unfreezeEpoch is not None
            and self.current_epoch == self.hparams.unfreezeEpoch
        ):
            print(f"\n[epoch {self.current_epoch}] Unfreezing full backbone.")
            for module in (self.stemLow, self.stemHigh, self.fuse, self.trunk):
                for param in module.parameters():
                    param.requires_grad = True

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weightDecay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.maxEpochs)
        return [opt], [sch]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trainCsv",   required=True, type=str)
    p.add_argument("--valCsv",     required=True, type=str)
    p.add_argument("--batchSize",  type=int,   default=32)
    p.add_argument("--maxEpochs",  type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--bgNoiseDir", type=str,   default=None,
                   help="Folder of background noise WAVs for AddBackgroundNoise augmentation")
    p.add_argument("--useCategories", action="store_true",
                   help="Train on coarse type_categories labels instead of raw vehicle_types strings")
    p.add_argument("--minClipsPerClass", type=int, default=None,
                   help="Drop classes with fewer than N clips in the combined train+val set. "
                        "Prevents rare classes with no val samples from poisoning val_f1.")
    p.add_argument("--noPosWeight", action="store_true",
                   help="Disable automatic pos_weight class balancing in BCEWithLogitsLoss")
    p.add_argument("--backbone",   type=str,   default="resnet18",
                   choices=["resnet18", "panns"],
                   help="Feature extractor. 'panns' trains an MLP head on precomputed "
                        "AudioSet-pretrained PANNs embeddings (run scripts/precomputeEmbeddings.py first).")
    p.add_argument("--weightDecay",     type=float, default=0.01,
                   help="AdamW weight_decay (default: 0.01). Increase to 0.05–0.1 for regularization.")
    p.add_argument("--freezeBackbone",  action="store_true",
                   help="Freeze conv1..layer3; train only layer4 + classifier. "
                        "Strongest anti-overfitting lever for small datasets.")
    p.add_argument("--unfreezeEpoch",   type=int, default=None,
                   help="Epoch at which to unfreeze the full backbone for end-to-end fine-tuning.")
    p.add_argument("--precision",  type=str,  default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"],
                   help="AMP precision mode. bf16-mixed is optimal for Blackwell (DGX Spark GB10). "
                        "Use 16-mixed on Turing/Volta GPUs.")
    p.add_argument("--patience",   type=int,  default=10,
                   help="EarlyStopping patience in epochs (default: 10). Set > unfreezeEpoch when using --freezeBackbone.")
    p.add_argument("--compile",    action="store_true",
                   help="Apply torch.compile(mode='reduce-overhead') for ~15-20%% speedup.")
    p.add_argument("--outputDir",  type=str,  default="./checkpoints",
                   help="Directory for model checkpoints.")
    args = p.parse_args()

    if args.backbone == "panns" and (args.freezeBackbone or args.unfreezeEpoch or args.compile):
        p.error("--freezeBackbone/--unfreezeEpoch/--compile do not apply with --backbone panns "
                "(the PANNs backbone is frozen offline; only the MLP head is trained).")

    torch.backends.cudnn.benchmark = True

    trainDf = pd.read_csv(args.trainCsv)
    valDf   = pd.read_csv(args.valCsv)
    combinedDf = pd.concat([trainDf, valDf], ignore_index=True)

    labelEncoder = buildLabelEncoder(combinedDf, useCategories=args.useCategories)

    if args.minClipsPerClass:
        labelCol = (
            "type_categories"
            if args.useCategories and "type_categories" in combinedDf.columns
            else "vehicle_types"
        )
        valCounts: dict[str, int] = {}
        for labels in valDf[labelCol].apply(json.loads):
            for t in labels:
                valCounts[t] = valCounts.get(t, 0) + 1
        kept = sorted(cls for cls in labelEncoder if valCounts.get(cls, 0) >= args.minClipsPerClass)
        dropped = sorted(set(labelEncoder) - set(kept))
        if dropped:
            print(f"Dropping {len(dropped)} class(es) with < {args.minClipsPerClass} val clips: {', '.join(dropped)}")
        labelEncoder = {cls: i for i, cls in enumerate(kept)}

    # Persist the encoder so eval / inference scripts don't need the train CSV.
    # Also save a timestamped copy so old checkpoints can be matched to their encoder
    # if a later training run overwrites labelEncoder.json with a different class set.
    import shutil
    from datetime import datetime
    Path(args.outputDir).mkdir(parents=True, exist_ok=True)
    encoderPath = Path(args.outputDir) / "labelEncoder.json"
    with open(encoderPath, "w") as _f:
        json.dump(labelEncoder, _f, indent=2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(encoderPath, Path(args.outputDir) / f"labelEncoder_{ts}.json")
    print(f"Label encoder saved: {encoderPath}")

    labelCol = (
        "type_categories"
        if args.useCategories and "type_categories" in trainDf.columns
        else "vehicle_types"
    )

    # Print class distribution so imbalance is visible before training starts.
    print(f"\nClasses ({len(labelEncoder)}):")
    trainCounts = {cls: 0 for cls in labelEncoder}
    for labels in trainDf[labelCol].apply(json.loads):
        for t in labels:
            if t in trainCounts:
                trainCounts[t] += 1
    valCountsDisplay = {cls: 0 for cls in labelEncoder}
    for labels in valDf[labelCol].apply(json.loads):
        for t in labels:
            if t in valCountsDisplay:
                valCountsDisplay[t] += 1
    for cls, idx in labelEncoder.items():
        print(f"  [{idx}] {cls}: {trainCounts[cls]} train  {valCountsDisplay[cls]} val")

    posWeight = None
    if not args.noPosWeight:
        posWeight = computePosWeight(trainDf, labelEncoder, labelCol)
        print(f"\npos_weight: { {cls: f'{posWeight[i]:.1f}' for cls, i in labelEncoder.items()} }")

    print()

    # Augmentation operates on waveforms/spectrograms, so it does not apply to
    # precomputed PANNs embeddings.
    trainDs = VehicleAudioDataset(
        trainDf, labelEncoder, augment=(args.backbone != "panns"),
        bgNoiseDir=args.bgNoiseDir, useCategories=args.useCategories,
        backbone=args.backbone,
    )
    valDs = VehicleAudioDataset(
        valDf, labelEncoder, augment=False,
        useCategories=args.useCategories,
        backbone=args.backbone,
    )

    trainLoader = torch.utils.data.DataLoader(
        trainDs, batch_size=args.batchSize, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    valLoader = torch.utils.data.DataLoader(
        valDs, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = VehicleSoundClassifier(
        nClasses=len(labelEncoder),
        lr=args.lr,
        maxEpochs=args.maxEpochs,
        posWeight=posWeight,
        weightDecay=args.weightDecay,
        freezeBackbone=args.freezeBackbone,
        unfreezeEpoch=args.unfreezeEpoch,
        backbone=args.backbone,
    )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    trainer = pl.Trainer(
        max_epochs=args.maxEpochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        default_root_dir=args.outputDir,
        num_sanity_val_steps=0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.outputDir,
                filename="epoch={epoch:02d}-val_f1={val_f1:.3f}",
                monitor="val_f1",
                mode="max",
                save_top_k=3,
                auto_insert_metric_name=False,
            ),
            pl.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=args.patience),
        ],
    )
    trainer.fit(model, trainLoader, valLoader)


if __name__ == "__main__":
    main()

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
SAMPLE_RATE = 22050
CLIP_SECS   = 5.0


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
    Clipped to [0.1, 100] to prevent extreme values from rare/common classes.
    """
    nSamples = len(df)
    counts = torch.zeros(len(labelEncoder))
    for labels in df[labelCol].apply(json.loads):
        for t in labels:
            if t in labelEncoder:
                counts[labelEncoder[t]] += 1
    negCounts = nSamples - counts
    posWeight = torch.clamp(negCounts / counts.clamp(min=1), min=0.1, max=100.0)
    return posWeight


class VehicleAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labelEncoder: dict[str, int],
        augment: bool = False,
        bgNoiseDir: str | None = None,
        useCategories: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.labelEncoder = labelEncoder
        self.nClasses = len(labelEncoder)
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

        self.augmentFn = buildAugPipeline(bgNoiseDir=bgNoiseDir) if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # librosa handles decoding, mono downmix, and resampling in one call
        waveform, _ = librosa.load(
            self._filepaths[idx], sr=SAMPLE_RATE, mono=True, duration=CLIP_SECS,
        )

        if len(waveform) < self.targetLen:
            waveform = np.pad(waveform, (0, self.targetLen - len(waveform)))

        if self.augmentFn:
            waveform = self.augmentFn(waveform, sample_rate=SAMPLE_RATE)

        mel = librosa.feature.melspectrogram(
            y=waveform, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128,
        )
        spec = torch.from_numpy(
            librosa.power_to_db(mel, top_db=80)
        ).unsqueeze(0).float()

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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["posWeight"])
        # persistent=False: posWeight is training-only; exclude from checkpoint so
        # load_from_checkpoint doesn't hit a state_dict mismatch at eval time.
        self.register_buffer("posWeight", posWeight, persistent=False)

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        return self.classifier(self.backbone(x))

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

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
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
    p.add_argument("--noPosWeight", action="store_true",
                   help="Disable automatic pos_weight class balancing in BCEWithLogitsLoss")
    p.add_argument("--precision",  type=str,  default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"],
                   help="AMP precision mode. bf16-mixed is optimal for Blackwell (DGX Spark GB10). "
                        "Use 16-mixed on Turing/Volta GPUs.")
    p.add_argument("--compile",    action="store_true",
                   help="Apply torch.compile(mode='reduce-overhead') for ~15-20%% speedup.")
    p.add_argument("--outputDir",  type=str,  default="./checkpoints",
                   help="Directory for model checkpoints.")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True

    trainDf = pd.read_csv(args.trainCsv)
    valDf   = pd.read_csv(args.valCsv)
    combinedDf = pd.concat([trainDf, valDf], ignore_index=True)

    labelEncoder = buildLabelEncoder(combinedDf, useCategories=args.useCategories)

    # Persist the encoder so eval / inference scripts don't need the train CSV.
    Path(args.outputDir).mkdir(parents=True, exist_ok=True)
    encoderPath = Path(args.outputDir) / "labelEncoder.json"
    with open(encoderPath, "w") as _f:
        json.dump(labelEncoder, _f, indent=2)
    print(f"Label encoder saved: {encoderPath}")

    labelCol = (
        "type_categories"
        if args.useCategories and "type_categories" in trainDf.columns
        else "vehicle_types"
    )

    # Print class distribution so imbalance is visible before training starts.
    print(f"\nClasses ({len(labelEncoder)}):")
    classCounts = {cls: 0 for cls in labelEncoder}
    for labels in trainDf[labelCol].apply(json.loads):
        for t in labels:
            if t in classCounts:
                classCounts[t] += 1
    for cls, idx in labelEncoder.items():
        print(f"  [{idx}] {cls}: {classCounts[cls]} train clips")

    posWeight = None
    if not args.noPosWeight:
        posWeight = computePosWeight(trainDf, labelEncoder, labelCol)
        print(f"\npos_weight: { {cls: f'{posWeight[i]:.1f}' for cls, i in labelEncoder.items()} }")

    print()

    trainDs = VehicleAudioDataset(
        trainDf, labelEncoder, augment=True,
        bgNoiseDir=args.bgNoiseDir, useCategories=args.useCategories,
    )
    valDs = VehicleAudioDataset(
        valDf, labelEncoder, augment=False,
        useCategories=args.useCategories,
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
    )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    trainer = pl.Trainer(
        max_epochs=args.maxEpochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        default_root_dir=args.outputDir,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.outputDir,
                monitor="val_f1",
                mode="max",
                save_top_k=3,
            ),
            pl.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=10),
        ],
    )
    trainer.fit(model, trainLoader, valLoader)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training toolchain: dataset, model, and training loop.

Expects CSVs produced by scripts/buildDataset.py (one row per clip).
Required columns: filepath, vehicle_types (JSON list), recordingId.
Optional columns used when present: directionClass, velocityKts.

Usage:
    python -m aircraftAudio.toolchain --trainCsv dataset/train.csv \
                                      --valCsv   dataset/val.csv
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import pytorch_lightning as pl
import torchmetrics
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from torchvision import models


# Audio pipeline constants. Clips are resampled to SAMPLE_RATE on load;
# recorder.py saves at 44100 Hz so resampling always occurs.
SAMPLE_RATE = 22050
CLIP_SECS   = 5.0


def buildLabelEncoder(df: pd.DataFrame) -> dict[str, int]:
    """
    Build a stable type-string → class-index mapping from the full dataset.
    Must be computed before splitting so train and val share the same mapping.
    """
    allTypes = sorted(
        {t for types in df["vehicle_types"].apply(json.loads) for t in types}
    )
    return {t: i for i, t in enumerate(allTypes)}


class VehicleAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labelEncoder: dict[str, int],
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.labelEncoder = labelEncoder
        self.nClasses = len(labelEncoder)
        self.targetLen = int(SAMPLE_RATE * CLIP_SECS)

        self.augmentFn = None
        if augment:
            self.augmentFn = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
            ])

        self.melSpec = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128,
        )
        self.toDb = T.AmplitudeToDB(top_db=80)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["filepath"])

        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] < self.targetLen:
            waveform = nn.functional.pad(waveform, (0, self.targetLen - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.targetLen]

        if self.augmentFn:
            arr = waveform.numpy()[0]
            arr = self.augmentFn(arr, sample_rate=SAMPLE_RATE)
            waveform = torch.from_numpy(arr).unsqueeze(0)

        spec = self.toDb(self.melSpec(waveform))

        typeLabel = torch.zeros(self.nClasses)
        for t in json.loads(row["vehicle_types"]):
            if t in self.labelEncoder:
                typeLabel[self.labelEncoder[t]] = 1.0

        return spec, typeLabel


class VehicleSoundClassifier(pl.LightningModule):
    def __init__(self, nClasses: int, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

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

    def training_step(self, batch, batch_idx):
        specs, labels = batch
        loss = nn.BCEWithLogitsLoss()(self(specs), labels)
        self.trainF1(torch.sigmoid(self(specs)), labels.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.trainF1, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        preds = torch.sigmoid(logits)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        self.valF1(preds, labels.int())
        self.valMap(preds, labels.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1",  self.valF1,  on_epoch=True, prog_bar=True)
        self.log("val_mAP", self.valMap, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return [opt], [sch]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trainCsv", required=True, type=str)
    p.add_argument("--valCsv",   required=True, type=str)
    p.add_argument("--batchSize", type=int, default=32)
    p.add_argument("--maxEpochs", type=int, default=50)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--workers",   type=int, default=4)
    args = p.parse_args()

    trainDf = pd.read_csv(args.trainCsv)
    valDf   = pd.read_csv(args.valCsv)

    # Label encoder built from the combined set so both splits share the mapping.
    labelEncoder = buildLabelEncoder(pd.concat([trainDf, valDf], ignore_index=True))
    print(f"Classes ({len(labelEncoder)}): {list(labelEncoder.keys())}")

    trainDs = VehicleAudioDataset(trainDf, labelEncoder, augment=True)
    valDs   = VehicleAudioDataset(valDf,   labelEncoder, augment=False)

    trainLoader = torch.utils.data.DataLoader(
        trainDs, batch_size=args.batchSize, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    valLoader = torch.utils.data.DataLoader(
        valDs, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = VehicleSoundClassifier(nClasses=len(labelEncoder), lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.maxEpochs,
        accelerator="gpu",
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=3),
            pl.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=10),
        ],
    )
    trainer.fit(model, trainLoader, valLoader)


if __name__ == "__main__":
    main()

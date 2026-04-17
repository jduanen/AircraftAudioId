import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import pytorch_lightning as pl
import torchmetrics
from torchvision import models
from datasets import Dataset, Audio
import pandas as pd

# ============ Dataset ============
class VehicleAudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, augment=False, sr=22050, duration=5.0):
        self.df = df
        self.sr = sr
        self.target_length = int(sr * duration)
        self.n_classes = 5

        self.augment_fn = None
        if augment:
            self.augment_fn = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
            ])

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row['filepath'])

        # Resample
        if sr != self.sr:
            waveform = T.Resample(sr, self.sr)(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim
        if waveform.shape[1] < self.target_length:
            waveform = nn.functional.pad(waveform, (0, self.target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.target_length]

        # Augment
        if self.augment_fn:
            waveform_np = waveform.numpy()[0]
            waveform_np = self.augment_fn(waveform_np, sample_rate=self.sr)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)

        # Spectrogram
        spec = self.to_db(self.mel_spec(waveform))

        # Multi-hot label
        label = torch.zeros(self.n_classes)
        for vtype in row['vehicle_types']:
            label[vtype] = 1.0

        return spec, label


# ============ Model ============
class VehicleSoundClassifier(pl.LightningModule):
    def __init__(self, n_classes=5, lr=1e-4):
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
            nn.Linear(256, n_classes),
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=n_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=n_classes, average="macro"
        )
        self.val_map = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=n_classes, average="macro"
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        loss = nn.BCEWithLogitsLoss()(logits, labels)

        self.train_f1(torch.sigmoid(logits), labels.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        preds = torch.sigmoid(logits)

        self.val_f1(preds, labels.int())
        self.val_map(preds, labels.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_mAP", self.val_map, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return [opt], [sch]


# ============ Training ============
if __name__ == "__main__":
    # Load your labels CSV
    df = pd.read_csv("labels.csv")
    df['vehicle_types'] = df['vehicle_types'].apply(eval)  # "[0, 2]" -> [0, 2]

    # Split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # Datasets
    train_ds = VehicleAudioDataset(train_df, augment=True)
    val_ds = VehicleAudioDataset(val_df, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = VehicleSoundClassifier(n_classes=5, lr=1e-4)

    # Train
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=3),
            pl.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=10),
        ],
    )

    trainer.fit(model, train_loader, val_loader)

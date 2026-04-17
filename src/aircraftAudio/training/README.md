# training

Runs on the **DGX Spark**. PyTorch Lightning model and training loop for
aircraft type classification from mel spectrograms.

## Module

**`toolchain.py`**

- `buildLabelEncoder(df)` — derives a stable `{type_string: class_index}`
  mapping from the full dataset. Must be built from the combined train+val data
  before splitting so both sets share the same mapping.
- `VehicleAudioDataset` — loads clip WAVs, resamples to 22050 Hz mono, converts
  to mel spectrogram (n_mels=128, n_fft=1024, hop_length=512, top_db=80), and
  returns a multi-hot label vector. Optionally applies audiomentations
  augmentation (Gaussian noise, time stretch, pitch shift).
- `VehicleSoundClassifier` — ResNet-34 backbone (first conv adapted for
  1-channel input) with a multi-label classification head. Trained with
  `BCEWithLogitsLoss`. Logs macro F1 and mAP on validation.

## Entry point

```bash
python -m aircraftAudio.toolchain \
    --trainCsv dataset/train.csv \
    --valCsv   dataset/val.csv \
    [--clipSecs 5.0] [--batchSize 32] [--maxEpochs 50] [--lr 1e-4]
```

## Current scope

Phase 1 only: vehicle type classification (multi-label). Direction and speed
heads are not yet implemented — `directionClass` and `velocityKts` columns in
the CSV are reserved for phase 2.

## Dependencies

`torch`, `torchaudio`, `torchvision`, `pytorch-lightning`, `torchmetrics`,
`audiomentations`, `pandas`.

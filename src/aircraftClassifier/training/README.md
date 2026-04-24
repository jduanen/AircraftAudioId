# training

Training loop, multi-task train step, and experiment tracking.

## Files

**`toolchain.py`** — `VehicleAudioDataset`, `VehicleSoundClassifier`, `buildLabelEncoder`  
Main training pipeline. `VehicleAudioDataset` loads clip WAVs from the CSV
produced by `scripts/buildDataset.py`, resamples to 22050 Hz, converts to
mel spectrogram, and returns multi-hot label vectors. `VehicleSoundClassifier`
is a ResNet-34 phase-1 model (vehicle type only) wrapped in PyTorch Lightning.
`buildLabelEncoder` derives the type→class-index mapping from the full dataset.

```bash
python -m aircraftClassifier.training.toolchain \
    --trainCsv dataset/train.csv --valCsv dataset/val.csv
```

**`multiTaskTrain.py`** — `multiTaskTrainStep`  
Manual training step for phase 2 (type + direction + speed heads). Accepts a
`singleMask` to apply direction/speed loss only on single-vehicle samples.
Replace the Lightning training step in `toolchain.py` with this when adding
phase-2 heads.

**`experiment.py`** — `buildTrainer`, `logHyperparameters`, `logAudioSample`  
Helpers for W&B logging and Lightning Trainer configuration. `buildTrainer`
wires up W&B logger, ModelCheckpoint, and EarlyStopping. `logAudioSample`
logs audio clips and spectrograms to W&B for qualitative inspection.

## Phase transition checklist

Moving from phase 1 → phase 2:
1. Switch model to `models/PragmaticMultiVehicleModel`
2. Add `directionClass` and `velocityKts` columns to dataset loading
3. Replace Lightning training step with `multiTaskTrainStep`
4. Start with `manualWeightedLoss`, tune weights, then try `UncertaintyWeightedLoss`

# aircraftClassifier

Model, training, and inference code for aircraft audio classification.
Runs on the **DGX Spark** (training) and wherever inference is needed.

## Subdirectories

| Directory | Purpose |
|---|---|
| `models/` | Model architectures (CustomCNN, ResNetCNN, DeepVehicleCNN, PragmaticModel, SetPredict) |
| `losses/` | Multi-task loss functions (uncertainty-weighted, Hungarian matching) |
| `augmentation/` | Audio augmentation pipelines and spectrogram windowing |
| `pretrained/` | Pretrained model integration (PANNs, AST) and spectrogram precomputation |
| `training/` | Main training loop, multi-task train step, W&B/Lightning experiment tracking |
| `eval/` | Multi-label evaluation metrics (F1, mAP, per-class AP, threshold tuning) |
| `datasets/` | Open-source dataset downloader, HuggingFace and WebDataset integrations |

## Development progression

**Phase 1 (current):** Vehicle type classification only.
Use `training/toolchain.py` with `models/resNetCNN.py`. Labels come from
`src/aircraftAudio/dataset/` via `scripts/buildDataset.py`.

**Phase 2:** Add direction (8-class) and speed (regression) heads.
Switch to `models/pragmaticModel.py` + `losses/multiTaskLoss.py` +
`training/multiTaskTrain.py`.

**Phase 3 (large dataset):** Multi-vehicle set prediction.
Switch to `models/setPredict.py` + `losses/hungarianLoss.py`.
Consider `pretrained/precompute.py` and `datasets/webDataset.py` for
pipeline throughput.

## Entry point

```bash
python -m aircraftClassifier.training.toolchain \
    --trainCsv dataset/train.csv \
    --valCsv   dataset/val.csv
```

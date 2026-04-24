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

## Currently unused files

  ┌─────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────┐
  │            File             │                                           Purpose                                           │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ augmentation/waveformAug.py │ SpecAugment + waveform augmentation (Phase 2 / AST)                                         │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ augmentation/gpuAug.py      │ GPU-side augmentation via torch-audiomentations                                             │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ augmentation/windowing.py   │ Sliding window extractor for long recordings                                                │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ models/resNetCNN.py         │ Standalone ResNet-34 model (the same architecture is re-implemented inline in toolchain.py) │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ models/customCNN.py         │ 4-block custom CNN                                                                          │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ models/deepVehicleCNN.py    │ 5-block deep CNN                                                                            │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ models/pragmaticModel.py    │ Multi-vehicle model with masked loss (Phase 2)                                              │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ models/setPredict.py        │ DETR-style set prediction (Phase 2+)                                                        │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ losses/multiTaskLoss.py     │ Uncertainty-weighted multi-task loss (Phase 2)                                              │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ losses/hungarianLoss.py     │ Hungarian matching loss (Phase 2+)                                                          │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ training/multiTaskTrain.py  │ Direction + speed training step (Phase 2)                                                   │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ training/experiment.py      │ W&B / wandb integration                                                                     │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ eval/metrics.py             │ Multilabel F1, mAP, threshold tuning                                                        │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ pretrained/ast.py           │ AST fine-tuning (alternative Phase 1 approach)                                              │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ pretrained/panns.py         │ PANNs embeddings (alternative Phase 1 approach)                                             │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ pretrained/precompute.py    │ Precompute spectrograms to .pt files                                                        │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ datasets/hfDatasets.py      │ HuggingFace Datasets integration                                                            │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ datasets/webDataset.py      │ WebDataset streaming                                                                        │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ datasets/openSource.py      │ ESC-50 / FSD50K downloader                                                                  │
  └─────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────┘

**N.B.**
  * `models/resNetCNN.py` contains a ResNetCNN class that's essentially the same architecture as 'VehicleSoundClassifier' in 'toolchain.py' 
    - it has the same ResNet-34 backbone and classifier head, but it's never imported
    - the toolchain re-implements it inline

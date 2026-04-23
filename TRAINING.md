# Training on the DGX Spark

Phase 1 classifier (aircraft type from audio) is trained inside an NVIDIA NGC PyTorch
container on the DGX Spark GB10 Grace Blackwell workstation.

## Prerequisites

- Dataset built on the recording server: `python scripts/buildDataset.py ...`
- DGX Spark reachable by SSH: confirm with `ssh <dgx-hostname>`

## Step 1 — Sync dataset to DGX Spark (run on recording server)

```bash
bash scripts/syncToDGX.sh <dgx-hostname>
```

Syncs `src/`, `scripts/`, `docker/`, `pyproject.toml`, and `dataset/` (CSVs + clips)
to `/home/jdn/Code/AircraftAudioId/` on the DGX Spark, preserving absolute paths so
clip paths in the CSV remain valid inside the container.

## Step 2 — Train (run on DGX Spark via SSH)

```bash
ssh <dgx-hostname>
cd /home/jdn/Code/AircraftAudioId

# First run builds the Docker image (~5 min)
bash scripts/trainDGX.sh --useCategories

# Full explicit run
bash scripts/trainDGX.sh \
    --useCategories \
    --batchSize 64 \
    --maxEpochs 100 \
    --lr 2e-4 \
    --compile
```

Checkpoints are saved to `checkpoints/` on the DGX Spark.

To override the checkpoint directory on the host:
```bash
CHECKPOINT_DIR=/home/jdn/Data/AircraftAudio/checkpoints bash scripts/trainDGX.sh --useCategories
```

## toolchain.py arguments added for DGX training

| Argument | Default | Notes |
|---|---|---|
| `--precision` | `bf16-mixed` | Optimal for Blackwell GB10. Use `16-mixed` on older GPUs. |
| `--compile` | off | Enables `torch.compile(mode="reduce-overhead")` (~15–20% speedup) |
| `--outputDir` | `./checkpoints` | Checkpoint directory (mapped to `/checkpoints` inside container) |

## Verify training is running correctly

```bash
# In a second SSH session on the DGX Spark
nvidia-smi

# Confirm in training log:
#   precision=bf16-mixed
#   GPU utilization > 0%
#   checkpoints/*.ckpt appears after epoch 1
```

## Container details

- Base image: `nvcr.io/nvidia/pytorch:25.01-py3` (multi-arch, pulls arm64 on DGX Spark)
- Includes: CUDA-optimized PyTorch 2.x, torchaudio, cuDNN — nothing reinstalled via pip
- Added: pytorch-lightning, audiomentations, pandas, scikit-learn
- Project mounted read-only at its exact host path; checkpoints volume is writable
- `ipc: host` enables shared-memory DataLoader workers (required with `pin_memory=True`)

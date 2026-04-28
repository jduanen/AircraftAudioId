# PLAN.md — Aircraft Audio Classifier: Model Training Plan

Model selection and training strategy for each project phase.
See `DESIGN_NOTES.md` for architecture rationale and `TRAINING.md` for DGX Spark setup.

---

## Model Selection by Dataset Size

Dataset size per class determines which backbone to use. Check current counts with
`python scripts/inspectDataset.py --recordingsDir ./recordings`.

| Clips per class | Recommended backbone | File |
|---|---|---|
| < 100 | PANNs CNN14 embeddings + small MLP head | `src/aircraftClassifier/pretrained/panns.py` |
| 100–500 | AST fine-tuning (AudioSet pretrained) | `src/aircraftClassifier/pretrained/ast.py` |
| 500–2000 | ResNet-34 with ImageNet weights | `src/aircraftClassifier/models/resNetCNN.py` |
| > 2000 | ResNet-34 or DeepVehicleCNN from scratch | `src/aircraftClassifier/models/deepVehicleCNN.py` |

**Current implementation** (`training/toolchain.py`) uses ResNet-34 with ImageNet weights.
Switch to PANNs or AST as the first upgrade once Phase 1 is baseline-trained.

---

## Spectrogram Config (Phase 0 — Before Any Phase)

The spectrogram config is the highest-leverage hyperparameter. Run a config sweep
**before committing to Phase 2+** so the information ceiling is set correctly.

```python
configs_to_try = [
    {"n_mels": 64,  "n_fft": 512,  "hop_length": 256},   # high time resolution
    {"n_mels": 128, "n_fft": 1024, "hop_length": 512},    # balanced — start here
    {"n_mels": 256, "n_fft": 2048, "hop_length": 512},    # high freq resolution
]
```

**Window size sweep:** train once each at 1, 2, 4, 8 seconds. Pick the shortest that
preserves val mAP — shorter clips = more clips per flyover = larger effective dataset.
Current default is 5 seconds.

**Before Phase 2:** move spectrogram computation into the Dataset (not the model) so
configs can be changed without reinstantiating the model. Use
`src/aircraftClassifier/pretrained/precompute.py` to precompute `.pt` tensors.

---

## Phase 1 — Aircraft Type Classification

**Goal:** Multi-label classification of propulsion type (jet / turboprop / piston /
helicopter) from single-aircraft clips.

**Status:** Implemented and runnable. `training/toolchain.py` trains ResNet-34 with
`BCEWithLogitsLoss` + automatic `pos_weight` class balancing.

**Labels:** `type_categories` column (coarse FAA-derived labels). Always build the
dataset with `--faaDatabaseDir` for authoritative labels.

**Training command (DGX Spark):**
```bash
bash scripts/trainDGX.sh \
    --useCategories \
    --batchSize 64 \
    --maxEpochs 100 \
    --lr 2e-4 \
    --compile
```

**Evaluation:**
```bash
python scripts/evalModel.py \
    --checkpoint checkpoints/best.ckpt \
    --labelEncoder checkpoints/labelEncoder.json \
    --valCsv dataset/val.csv \
    --useCategories \
    --tuneThresholds
```

**Done when:** val mAP > 0.80 across all type categories with balanced class support.

**Upgrade path:**
1. Baseline: ResNet-34 with ImageNet weights (current)
2. Next: PANNs CNN14 — swap backbone in `pretrained/panns.py`, freeze CNN14, train
   only the classification head for 10 epochs, then unfreeze and fine-tune at lr=1e-5
3. Later (>500 clips/class): AST fine-tuning via `pretrained/ast.py`

---

## Phase 2 — Direction of Travel

**Goal:** 8-class relative direction classifier (0=away, 1=away-right, …, 4=toward,
…, 7=away-left) on single-aircraft clips only.

**Labels:** `directionClass` column (relative bearing: `(headingDeg − bearingDeg + 360) % 360`
binned to 8 × 45° sectors). Values 0–7; −1 = unknown (exclude from direction loss).

**Architecture:** Add a second head to `VehicleSoundClassifier`:
- Shared ResNet-34 backbone (same as Phase 1)
- Direction head: `Linear(512 → 8)` + softmax + CrossEntropyLoss
- **Masked loss:** backpropagate direction loss only on samples where `isSingle=1`
  and `directionClass >= 0`

**Multi-task loss:**
```python
loss = typeWeight * typeLoss + dirWeight * dirLoss
```
Start with `typeWeight=1.0, dirWeight=0.5`. Use `src/aircraftClassifier/losses/multiTaskLoss.py`.

**Prerequisite check before starting Phase 2:**
```bash
python scripts/inspectDataset.py --recordingsDir ./recordings --datasetCsv dataset/dataset.csv
# Need: ≥200 single-aircraft clips per direction class (1600 total isSingle clips minimum)
# Check: "Direction class distribution" section — all 8 bins populated
```

**Done when:** val direction accuracy > 0.65 on isSingle clips (random chance = 12.5%).

---

## Phase 3 — Speed Estimation

**Goal:** Scalar regression of groundspeed in knots on single-aircraft clips.

**Labels:** `velocityKts` column (from ADS-B state at clip center).

**Architecture:** Add a third head to the shared backbone:
- Speed head: `Linear(512 → 1)` (no activation) + MSELoss (or HuberLoss for outlier
  robustness)
- **Masked loss:** backpropagate speed loss only on samples where `isSingle=1` and
  `velocityKts` is not null

**Multi-task loss:**
```python
loss = typeWeight * typeLoss + dirWeight * dirLoss + speedWeight * speedLoss
```
Speed loss scale differs from classification losses — normalize `velocityKts` to
zero-mean unit-variance before training, then invert the prediction for reporting.

**Done when:** val speed MAE < 30 knots on isSingle clips.

---

## Phase 4 — Multi-Aircraft Handling

**Goal:** Correctly classify clips containing 2+ simultaneous aircraft.

**Decision framework** (check `isSingle` split in `inspectDataset.py`):

| Multi-aircraft fraction | Recommended approach |
|---|---|
| < 20% | Pragmatic: train type on all, mask direction/speed to isSingle=1 |
| 20–80% | Pragmatic type + consider set-based direction/speed |
| > 80% | Set prediction (Approach A) or source separation pre-processing |

**Approach A — Set Prediction:**
- Model: `src/aircraftClassifier/models/setPredict.py`
- Loss: Hungarian matching to assign ground-truth vehicles to prediction slots
  (`src/aircraftClassifier/losses/hungarianLoss.py`)
- Requires: enough multi-aircraft examples with fully labeled ground truth

**Approach B — Pragmatic (start here):**
- Model: `src/aircraftClassifier/models/pragmaticModel.py`
- Type head trained on all clips (multi-label is already correct for this)
- Direction/speed heads masked to isSingle=1 only
- Already scaffolded; this is what Phase 2/3 above describe

---

## Phase 5 — Additional Attributes

**Goal:** Estimate distance, altitude, and specific airframe/engine model.

**Labels available now:** `distanceKm`, `altitudeFt`, `bearingDeg` columns already in
the dataset CSV.

**Architecture:** Additional regression heads on the shared backbone, same masking
pattern as Phase 3.

**Airframe model classification:** Very high class count (thousands of model variants).
Options:
1. Hierarchical: predict manufacturer first, then model within manufacturer
2. Embed ICAO24 type string as a string classifier (many-class softmax)
3. Defer until Phase 1/2/3 are solid and dataset is large (>10k clips)

**Distance/altitude:** Correlated with audio SNR and Doppler characteristics.
Include `distanceKm` and `altitudeFt` as regression targets alongside type. Consider
adding them as auxiliary inputs (condition the classifier on known distance/altitude
at inference time if ADS-B is available live).

---

## Data Collection Priorities

1. **Collect during peak traffic hours:** 6–9am and 5–8pm weekdays near an airport
2. **Vary conditions:** multiple days, times, weather (avoid high wind)
3. **Target rare classes first:** check `inspectDataset.py` type distribution; if
   `helicopter` or `piston_multi` has < 50 clips, prioritize collection near a GA airport
4. **Maintain NTP sync:** run `scripts/syncClocks.sh` (server) and
   `scripts/syncClocksPi.sh` (Pi) before each recording session to minimize clock skew

## Key Guardrails

- Always split train/val **by flyover event** (15-char timestamp prefix), never randomly
- Always build dataset with `--faaDatabaseDir` — keyword heuristic produces wrong labels
- Rebuild dataset CSVs before retraining whenever `clipExport.py` label logic changes
- Direction and speed heads require a sufficient isSingle clip count — verify before
  adding Phase 2/3 heads or their gradient contributions will be near zero

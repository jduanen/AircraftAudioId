# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project to classify aircraft from audio recordings. Goals (in progression order):
1. Classify aircraft type (propulsion: jet/turbine/piston; engine count; wing type: rotary/fixed)
2. Classify direction of travel (8 cardinal directions) for single-aircraft clips
3. Estimate speed from audio
4. Handle clips with multiple simultaneous aircraft
5. Estimate additional attributes (distance, altitude, specific engine/airframe model)

Ground truth labels come from synchronized ADS-B telemetry data captured alongside audio recordings.

## Code Organization

The codebase has two layers:

**`src/aircraftAudio/` — structured package (active development)**
- `record/recorder.py` — `AircraftRecordingSystem`: top-level orchestrator; coordinates ADS-B monitoring and audio buffering; triggers and saves recordings when a flyover is detected; skips saves when `isStreamHealthy()` returns false
- `record/adsb/readsb.py` — `ReadsbClient`: polls a readsb/dump1090 JSON endpoint; returns `AircraftState` objects filtered by radius and altitude
- `record/audioStream/remoteStream.py` — `RemoteAudioStream`: TCP server that receives PCM chunks from the Pi and maintains a 60-second circular buffer; `isStreamHealthy(durationSecs)` gates recordings on active chunk delivery
- `record/aircraftType.py` — `AircraftDatabase`: looks up aircraft type strings from ICAO24 hex codes
- `capture/piCapture.py` — `PiCapture`: runs on the Pi Zero W; captures USB mic audio and streams it over TCP
- `capture/micEval.py` — `evaluateDevices`: measures noise floor, SNR, and frequency response of attached mic/ADC devices
- `dataset/clipExport.py` — `buildClipDataset`: aligns ADS-B states to audio, extracts clips, writes train/val CSVs; skips silent source WAVs (Pi not streaming)
**`scripts/` — server-side entry points**
- `record.py` — run on the Ubuntu server; starts `AircraftRecordingSystem`
- `buildDataset.py` — run on the Ubuntu server; extracts clips and writes `train.csv` / `val.csv`
**`audioCapture/scripts/` — Pi-side entry points**
- `capture.py` — run on the Pi Zero W; starts `PiCapture`

**`src/aircraftClassifier/` — model, training, and inference (runs on DGX Spark)**
- `models/` — `ResNetCNN` (recommended), `CustomCNN`, `DeepVehicleCNN`, `PragmaticMultiVehicleModel`, `SetPredictionModel`
- `losses/` — `UncertaintyWeightedLoss`, `manualWeightedLoss`, `hungarianLoss`
- `augmentation/` — audiomentations pipeline, GPU augmentation, SpecAugment, sliding-window extractor
- `pretrained/` — PANNs embeddings, AST fine-tuning, spectrogram precomputation
- `training/` — `VehicleAudioDataset` + `VehicleSoundClassifier` (main loop), multi-task train step, W&B integration
- `eval/` — multilabel F1, mAP, per-class AP, threshold tuning
- `datasets/` — ESC-50/FSD50K downloader, HuggingFace and WebDataset integrations

**`hold/` — original research sketches (superseded, kept for reference)**

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the data-collection and server-side stack (torch, torchaudio, pytorch-lightning, audiomentations, scikit-learn, pandas, numpy, scipy, sounddevice, soundfile, requests, geopy). Additional optional installs:

```bash
pip install librosa datasets webdataset wandb mlflow transformers panns-inference hearbaseline ntplib
```

## Training on DGX Spark

**Infrastructure:** The DGX Spark (spark-8d0d.local, 192.168.166.7) receives the full project including `dataset/` via rsync from the Ubuntu recording server (192.168.166.13). Docker mounts the project directory read-only; `precomputeDGX.sh` mounts `dataset/` writably so it can write `.spec.npy` files.

Training runs inside `aircraft-audio-training:latest` (built from `docker/Dockerfile.training`), which uses the NGC PyTorch arm64 container as base. Python source changes are live after sync without image rebuild; rebuild only when `Dockerfile.training` changes.

**Workflow (run on Ubuntu recording server):**
```bash
# 1. Sync code to DGX (excludes dataset/ and recordings/)
bash scripts/syncToDGX.sh spark-8d0d.local

# 2. Pre-compute spectrograms once (or after adding new clips)
#    Writes <clip>.spec.npy files to the NFS dataset directory
bash scripts/precomputeDGX.sh

# 3. Train
bash scripts/trainDGX.sh --useCategories [--batchSize 64] [--maxEpochs 50]

# 4. Evaluate
bash scripts/evalDGX.sh \
    --checkpoint /checkpoints/best.ckpt \
    --labelEncoder /checkpoints/labelEncoder.json \
    --valCsv dataset/val.csv --useCategories [--tuneThresholds]
```

Checkpoints land in `./checkpoints/` on the DGX host (not synced back automatically).

**TensorBoard** (from Ubuntu, port-forwarded):
```bash
ssh -L 6006:localhost:6006 jdn@spark-8d0d.local \
    "python3 -m tensorboard.main --logdir /home/jdn/Code/AircraftAudioId/checkpoints/lightning_logs/ --port 6006"
```
Then open `http://localhost:6006`.

## Running the System

**Two-machine deployment:**

Pi Zero W (audio capture):
```bash
python scripts/capture.py --host <server-ip> --port 9876 [--device 1] [--sampleRate 44100]
```

Ubuntu server (record + ADS-B sync):
```bash
python scripts/record.py --lat <lat> --lon <lon> --radiusKm 20 --outputDir ./recordings \
    [--listenPort 9876] [--readsbUrl http://adsbrx.lan/data/aircraft.json]
```

Build training dataset from recordings:
```bash
python scripts/buildDataset.py --recordingsDir ./recordings --outputDir ./dataset \
    [--faaDatabaseDir /path/to/ReleasableAircraft] [--maxDistanceKm 15] [--dropUnknown]
```

Evaluate microphone quality (run on Pi):
```bash
python tools/evalMics.py [--duration 10] [--outputDir ./mic_eval]
```

## Data Flow and Wire Protocol

1. Pi captures USB mic at 44100 Hz mono (int16) and sends framed chunks over TCP:
   `[8B float64 Unix timestamp][4B uint32 byte length][N bytes PCM S16LE]`
2. `RemoteAudioStream` on the server receives chunks into a 60-second circular buffer.
3. `ReadsbClient` polls `readsb` (default: `http://adsbrx.lan/data/aircraft.json`) every second.
4. `AircraftRecordingSystem` tracks aircraft within `radiusKm`. Recording saves when:
   - last 3 distance measurements are increasing (aircraft leaving), or
   - aircraft has been tracked for `MAX_RECORDING_SECS` (30s cap)
   - Save is skipped if `isStreamHealthy(durationSecs)` returns false (Pi not streaming)
5. Saved output per event: `recordings/audio/<id>.wav` + `recordings/metadata/<id>.json`
6. `scripts/buildDataset.py` uses `align.py` to map each ADS-B state to its sample position, cuts fixed-length clips, and writes `dataset/train.csv` + `dataset/val.csv` (split by flyover event to prevent leakage). Silent source WAVs are skipped. CSV columns: `filepath`, `recordingId`, `vehicle_types`, `type_categories`, `isSingle`, `flightPhase`, `directionClass`, `velocityKts`, `altitudeFt`, `distanceKm`, `bearingDeg`, `headingDeg`.
7. `aircraftClassifier/training/toolchain.py` (`VehicleAudioDataset`) consumes the CSV and converts WAV clips → mel-spectrogram → multi-hot label tensor for training. Run via Docker on the DGX Spark using `scripts/trainDGX.sh`. Pre-compute spectrograms first with `scripts/precomputeDGX.sh` to avoid per-epoch CPU bottleneck.

**NTP sync matters:** audio timestamps must align with ADS-B data. Run `sudo timedatectl set-ntp true` on the Pi; `piCapture.py` will warn if NTP offset exceeds 100 ms.

## Model Architecture

ResNet-18 backbone (chosen over ResNet-34 to reduce overfitting on the current small dataset) with three task-specific heads (see `hold/Arena1/architecture.txt`, implemented in `src/aircraftClassifier/models/`):
- **Vehicle type** — multi-label sigmoid + BCEWithLogitsLoss (works for both single- and multi-aircraft clips)
- **Direction** — 8-class softmax, masked loss (backprop only on single-vehicle samples)
- **Speed** — scalar regression, masked loss (single-vehicle samples only)

Audio pipeline: raw WAV at 44100 Hz mono (no resampling) → dual-band mel-spectrogram → dB scale → SpecAugment (train only) → ResNet-18 (first conv adapted for 2-channel input). Dropout is 0.5 in the classifier head.

Spectrograms are pre-computed and cached as `<clip>.spec.npy` alongside each WAV (`scripts/precomputeSpecs.py`). `VehicleAudioDataset.__getitem__` loads the `.npy` directly if present; SpecAugment is applied at load time for the training set. Fall back to librosa at runtime if `.npy` is absent. **Note:** torchaudio is not used — it is ABI-incompatible with the NGC arm64 PyTorch container (`aoti_torch_abi_version` undefined symbol). Use librosa for all audio I/O and spectrogram computation.

Active mel spectrogram config (`SAMPLE_RATE=44100`, `CLIP_SECS=5.0`): a **dual-channel, non-overlapping mel spectrogram**, computed once by `_dualBandMelDb()` in `toolchain.py` and imported by every script that touches spectrograms (`precomputeSpecs.py`, `evalModel.py`, `vizSpecs.py`). Each `.spec.npy` is shape `(2, N_MELS, ~431)`:
```python
# channel 0: 0-8000 Hz,        full N_MELS budget
{"n_mels": 128, "n_fft": 2048, "hop_length": 512, "fmax": FMAX_LOW}    # FMAX_LOW=8000
# channel 1: 8000 Hz-Nyquist,  full N_MELS budget
{"n_mels": 128, "n_fft": 2048, "hop_length": 512, "fmin": FMIN_HIGH}   # FMIN_HIGH=8000
```
Changed 2026-07-03, in two steps, from `{"n_mels": 128, "n_fft": 1024, "hop_length": 512}` (single channel, no fmax cap i.e. 22050):
1. **n_fft 1024→2048, single global fmax.** 92–99% of aircraft flyover signal energy sits below 200 Hz, but the old config only resolved that band into 5–6 mel bins; n_fft=2048 halves the underlying FFT bin width (43.1 Hz → 21.5 Hz). Tried `fmax=8000` (mAP 0.405→0.430, but business_jet/widebody_jet regressed) then `fmax=12000` (recovered those two, but helicopter/turboprop/regional_jet collapsed — mAP 0.409, worse than fmax=8000). No single global cutoff serves both class groups — real tug-of-war over the shared 128-mel-bin budget, not diminishing returns.
2. **Dual-channel.** Give each frequency region its own full 128-mel budget instead of splitting one budget with a single cutoff. `conv1` changed `Conv2d(1,64,...)` → `Conv2d(2,64,...)` (conv1 is replaced from scratch regardless of channel count, so no pretrained-weight remapping issue). Rejected full parallel/duplicate backbones (2× 11M params) as too much added capacity for ~800–1000 clips/class given the overfitting already fought with `--freezeBackbone`/patience/weight-decay.

Time-frame count and non-conv1 model dims are unchanged (`hop_length` untouched). See `DESIGN_NOTES.md` "Experiment Log — Backbone & Spectrogram Investigation" for the full analysis. **Changing `N_FFT`/`HOP_LENGTH`/`N_MELS`/`FMAX_LOW`/`FMIN_HIGH` requires regenerating all `.spec.npy` sidecars** — `VehicleAudioDataset` loads them directly with no staleness check.

Other configs to try:
```python
{"n_mels": 64,  "n_fft": 512,  "hop_length": 256},   # higher time resolution
{"n_mels": 256, "n_fft": 4096, "hop_length": 512},   # even finer resolution per band
```

Multi-vehicle decision framework (based on % of dataset with multiple aircraft):
- <20% → Pragmatic approach (`hold/pragmaticMultiVehicle.py`)
- 20–80% → Pragmatic for type; consider set-based for direction/speed
- >80% → Set-based prediction (`hold/setPredict.py` + `hold/hungarianLoss.py`) or source separation pre-processing

## Critical Data Design Notes

**Train/test split by session/event, not randomly.** Random splitting leaks data — clips from the same flyover appear in both splits. Split on recording session ID or ADS-B event ID.

**Multi-label classification:** sigmoid output + BCEWithLogitsLoss (not softmax/CrossEntropy). See `hold/multiLabel.py` and `hold/computeLoss.py`.

**Large-dataset priority order:**
1. Audio preprocessing quality (spectrogram config)
2. Data pipeline throughput (I/O and GPU utilization)
3. Train/test split integrity
4. Model capacity
5. Multi-task loss weight tuning (`hold/Arena1/multiTaskLoss.py`)

## Open-Source Datasets

| Dataset | Size | Notes |
|---|---|---|
| AeroSonicDB (YPAD-0523) | 625 recordings + ADS-B | Best: synchronized ADS-B |
| AudioSet aircraft subset | ~100k clips | Pre-training scale; coarse labels |
| FSD50K | ~500–1000 clips | Good quality |
| ESC-50 | 80 clips | Mostly helicopters |
| UrbanSound8K (subset) | ~100 clips | Mixed with other sounds |

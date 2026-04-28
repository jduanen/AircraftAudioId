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
- `recorder.py` — `AircraftRecordingSystem`: top-level orchestrator; coordinates ADS-B monitoring and audio buffering; triggers and saves recordings when a flyover is detected
- `adsb/readsb.py` — `ReadsbClient`: polls a readsb/dump1090 JSON endpoint; returns `AircraftState` objects filtered by radius and altitude
- `audioStream/remoteStream.py` — `RemoteAudioStream`: TCP server that receives PCM chunks from the Pi and maintains a 60-second circular buffer
- `audioStream/piCapture.py` — `PiCapture`: runs on the Pi Zero W; captures USB mic audio and streams it over TCP
- `aircraftType.py` — `AircraftDatabase`: looks up aircraft type strings from ICAO24 hex codes
- `micEval.py` — `evaluateDevices`: measures noise floor, SNR, and frequency response of attached mic/ADC devices
**`scripts/` — entry points**
- `record.py` — run on the Ubuntu server; starts `AircraftRecordingSystem`
- `capture.py` — run on the Pi Zero W; starts `PiCapture`
- `buildDataset.py` — run on the Ubuntu server; extracts clips and writes `train.csv` / `val.csv`

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

Export recordings to training CSV:
```bash
python scripts/exportDataset.py --recordingsDir ./recordings [--output dataset.csv]
```

Evaluate microphone quality (run on Pi):
```bash
python scripts/evalMics.py [--duration 10] [--outputDir ./mic_eval]
```

## Data Flow and Wire Protocol

1. Pi captures USB mic at 44100 Hz mono (int16) and sends framed chunks over TCP:
   `[8B float64 Unix timestamp][4B uint32 byte length][N bytes PCM S16LE]`
2. `RemoteAudioStream` on the server receives chunks into a 60-second circular buffer.
3. `ReadsbClient` polls `readsb` (default: `http://adsbrx.lan/data/aircraft.json`) every second.
4. `AircraftRecordingSystem` tracks aircraft within `radiusKm`. Recording saves when:
   - last 3 distance measurements are increasing (aircraft leaving), or
   - aircraft has been tracked for `MAX_RECORDING_SECS` (45s cap)
5. Saved output per event: `recordings/audio/<id>.wav` + `recordings/metadata/<id>.json`
6. `scripts/buildDataset.py` uses `align.py` to map each ADS-B state to its sample position, cuts fixed-length clips, and writes `dataset/train.csv` + `dataset/val.csv` (split by flyover event to prevent leakage). CSV columns: `filepath`, `recordingId`, `vehicle_types`, `directionClass`, `velocityKts`, `altitudeFt`, `distanceKm`, `bearingDeg`, `headingDeg`.
7. `aircraftClassifier/training/toolchain.py` (`VehicleAudioDataset`) consumes the CSV and converts WAV clips → mel-spectrogram → multi-hot label tensor for training:

```bash
python -m aircraftClassifier.training.toolchain \
    --trainCsv dataset/train.csv --valCsv dataset/val.csv
```

**NTP sync matters:** audio timestamps must align with ADS-B data. Run `sudo timedatectl set-ntp true` on the Pi; `piCapture.py` will warn if NTP offset exceeds 100 ms.

## Model Architecture

ResNet-34 backbone with three task-specific heads (see `hold/Arena1/architecture.txt`, implemented in `src/aircraftClassifier/models/`):
- **Vehicle type** — multi-label sigmoid + BCEWithLogitsLoss (works for both single- and multi-aircraft clips)
- **Direction** — 8-class softmax, masked loss (backprop only on single-vehicle samples)
- **Speed** — scalar regression, masked loss (single-vehicle samples only)

Audio pipeline: raw WAV → resample to 22050 Hz mono → mel-spectrogram → dB scale → SpecAugment → ResNet-34 (first conv adapted for 1-channel input).

Mel spectrogram configs to try (parameter choice defines the information ceiling):
```python
configs_to_try = [
    {"n_mels": 64,  "n_fft": 512,  "hop_length": 256},   # higher time resolution
    {"n_mels": 128, "n_fft": 1024, "hop_length": 512},    # balanced (start here)
    {"n_mels": 256, "n_fft": 2048, "hop_length": 512},    # higher freq resolution
]
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

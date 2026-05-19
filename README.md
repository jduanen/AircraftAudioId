# Audio Classification of Aircraft

**WIP**

## Hardware

### ADS-B Metadata Capture (Rpi4B)

See my [ADS-B Receiver Monitor](https://github.com/jduanen/ADSBMonitor) repo for the hardware used to generate the ADS-B metadata this project's training data.

### Flyover Audio Capture (RPi0-2W)

* Raspberry Pi Zero 2W with USB-C microphone
* see [Audio Capture Device](./audioCapture) for details

### ADS-B and Audio Processing (x86 Ubuntu Server w/ GPU)

* Ubuntu machine with i7-7820X, 128GB DRAM, and GTX2080
* ?

### Model Training and Inference (DGX Spark)

* DGX Spark: GB10 (128GB unified DRAM, 20x ARM CPU cores, Blackwell GPU)

## Software

The RPi0-2W is running Trixie and uses ntp to keep its clock synchronized with that of the server

The server is running Ubuntu on a deskside Intel CPU with 128GB of DRAM, and a GTX2080 GPU

### ADS-B Metadata Capture (RPi 4B)

See my [ADS-B Receiver Monitor](https://github.com/jduanen/ADSBMonitor) repo for the software used to generate the ADS-B metadata this project's training data

### Flyover Audio Capture (RPi Zero 2W)

* the `scripts/capture.py` script captures audio samples from the microphone, packetizes them, and sends them over a socket to the `scripts/record.py` script running on the server
* see [Audio Capture Device](./audioCapture) for details on the audio capture subsystem

### ADS-B and Audio Processing (x86 Ubuntu Server w/ GPU)

* **`scripts/record.py`**: runs the recording system that synchronizes and combines ADS-B and audio signals
  - uses `src/aircraftAudio/recorder.py` to coordinate the data from `readsb` and the RPi0-2W audio stream
  - e.g.,
```bash
python scripts/record.py \
    --lat <lat> --lon <lon> \
    --radiusKm <radius_km> \
    --maxAltitudeFt <alt_ft> \
    --outputDir <path> \
    --readsbUrl <url> \
    --nullSampleInterval <secs> \
    --nullSampleDuration <secs>
```
  - other options:
    * minAltitudeFt <alt_ft>: minimum aircraft altitude (default: ?)
    * sampleRate <Hz>: audio sample rate (default: ?Hz)
    * listenPort <portNum>: TCP port to receive Pi audio (default: ????)
    * postTriggerSecs <float>: seconds to keep collecting departure states after the save trigger fires (default: 10)
    * faaDatabaseDir <path>: path to unzipped FAA ReleasableAircraft directory — required for class-cap filtering
    * datasetCsv <path>: path to existing `dataset.csv`; loads current per-class clip counts at startup
    * maxSamplesPerClass <int>: skip recording aircraft whose coarse category already has this many clips in `dataset.csv`; requires `--faaDatabaseDir`; aircraft with unknown/foreign registrations are always recorded
  - **class-cap filtering**: when `--maxSamplesPerClass` is set, the recorder looks up each new aircraft's FAA category and skips it if that category is already at the cap; prints a `[cap]` line for each skipped aircraft at startup showing which classes are capped; counts are loaded from `--datasetCsv` at startup and do not update mid-session — restart with a fresh CSV after rebuilding the dataset to refresh counts

* **`scripts/buildDataset.py`**: reads recordings (meta)data and generates training dataset suitable for input to `toolchain.py`
  - e.g.,
```bash
python scripts/buildDataset.py \
    --recordingsDir <path> \
    --outputDir <path> \
    --faaDatabaseDir <path> \
    --autoCorrectClock \
    --maxCoTrackRatio <float> \
    --dropUnknown
```
  - other options:
    * maxPerClass <int>: cap each class at a given number (?)
    * clipSecs <float>: change clip length in secs (default: 5sec)
    * minDistanceKm <float>: filter out aircraft that are too close (can cause audio clipping)
    * maxDistanceKm <float>: filter out aircraft that are too far away to be heard clearly
    * trainFrac <float>: adjust ratio of train/val split (default: 0.8 -- i.e., 80/20)
    * clockCorrection <float>: manual global clock offset
      - only use if --autoCorrectClock option produces uniformly bad alignment
    * stratifyPhase: use the rarest bucket
      - e.g., (narrowbody_jet, approach), (narrowbody_jet, departure), (piston_single, approach), etc.
      - do this so every label ends up with equal approach and departure counts
      - without this, the existing per-label balancing is unchanged
    * balanceClasses: auto-balance to rarest class count; when downsampling, keeps the highest-RMS clips per class (loudest = best signal quality) rather than selecting randomly
    * skipExisting: skip recordings already in dataset.csv and merge new clips into the existing data — use this for incremental updates when new recordings have been added
  - this produces `dataset/train.csv` and `dataset/val.csv` which plug directly into toolchain.py's VehicleAudioDataset and reference the audio samples in `clips/`
  - `toolchain.py` expects 'filepath' (path to a 5-second clip WAV) and 'vehicle_types' (JSON list, e.g., ["B738"])
  - the generated CSV files contain: `directionClass` (0–7, from `headingDeg`), `velocityKts`, `distanceKm`, `clipRms` (RMS amplitude of the clip — used as a quality score when balancing classes)

* **`scripts/vizSpecs.py`**: visualize mel spectrograms from the dataset as a grid
  - loads from pre-computed `.spec.npy` files if available, otherwise falls back to librosa
  - e.g.,
```bash
# show a random 3×3 grid from the training set
python scripts/vizSpecs.py --csv dataset/train.csv

# show 12 helicopter clips in a 4-column grid
python scripts/vizSpecs.py --csv dataset/train.csv --category helicopter --n 12 --cols 4

# save to a PNG instead of displaying
python scripts/vizSpecs.py --csv dataset/train.csv --output specs.png

# interactive mode: click a spectrogram to play its audio (requires sounddevice + soundfile)
python scripts/vizSpecs.py --csv dataset/train.csv --play
```
  - options:
    * `--csv` (required): path to `train.csv` or `val.csv`
    * `--n` (default 9): number of clips to display
    * `--cols` (default 3): grid columns
    * `--category`: filter to a single coarse category (e.g. `helicopter`, `piston_single`)
    * `--seed` (default 42): random seed — change to see a different sample
    * `--output`: save to file instead of displaying
    * `--play`: enable click-to-play audio; clicking a spectrogram plays the corresponding WAV

* **`scripts/inspectDataset.py`**: provides a measure of the quantity, quality, and distribution of collected training/testing samples
  - this takes an inventory of the samples in the dataset and prints information about the data dataset described in `<recordingsDir>/../dataset/dataset.csv`.
  - e.g.,
```bash
python3 scripts/inspectDataset.py --recordingsDir <path>
```
  - other options:
    * datasetCsv <path>: path to dataset.csv (default: '<recordingsDir>/../dataset/dataset.csv')
    * maxQualityClips <int>: max clips to read for audio quality check (default: 200)
  - the information provided by this program includes:
    * number of Metadata files with matching WAV files and the number of missing WAV files
    * the number of single-aircraft and the number of multi-aircraft recordings
    * a histogram of durations of the recordings
    * a histogram of the distribution of distances of the sampled aircraft
    * a histogram with the percentages and absolute number of each class of aircraft
    * a histogram of the distribution of coarse labels
    * an indication of clock skew between the devices

* **`scripts/icaoLookup.py`**: list unique ICAO24 hex codes seen across all recorded metadata, with optional sample counts, track counts, and FAA registration details
  - e.g.,
```bash
# basic list of codes and most-common callsigns
python3 scripts/icaoLookup.py --recordingsDir ./recordings

# with raw sample count and per-track sighting count
python3 scripts/icaoLookup.py --recordingsDir ./recordings --counts --tracks

# with FAA registration info, sorted by most-recorded
python3 scripts/icaoLookup.py --recordingsDir ./recordings --counts --tracks \
    --faa --faaDatabaseDir ./data/ReleasableAircraft --sortBy samples

# with extended FAA fields
python3 scripts/icaoLookup.py --recordingsDir ./recordings --counts --tracks \
    --faa --faaDatabaseDir ./data/ReleasableAircraft \
    --fields nNumber,manufacturer,model,typeAcft,typeEng,noEngines,noSeats
```
  - options:
    * `--counts`: show raw sample count (total number of ADS-B state entries per code)
    * `--tracks`: show per-track sighting count
    * `--trackInterval <hours>` (default 1.0): minimum gap in hours between recordings to count as a new track
    * `--faa`: show FAA registration info (requires `--faaDatabaseDir`)
    * `--faaDatabaseDir <path>`: path to unzipped FAA ReleasableAircraft directory
    * `--fields <list>` (default: `nNumber,manufacturer,model,category`): comma-separated FAA fields; available: `nNumber`, `manufacturer`, `model`, `category`, `typeAcft`, `typeEng`, `noEngines`, `noSeats`
    * `--sortBy` (default `icao24`): sort by `icao24`, `samples`, `tracks`, or `callsign`
  - **CALLSIGN** is the most frequently seen callsign for each code (stable for N-number aircraft; varies for airline flights)
  - foreign registrations not in the FAA database show blank FAA columns

### Model Training (DGX Spark)

Training runs inside a Docker container on the DGX Spark using scripts in `scripts/`. All commands below are run from the Ubuntu recording server unless noted.

* Phase 1: Classify by vehicle type (multi-label)
  - coarse category labels: `piston_single`, `piston_twin`, `turboprop`, `helicopter`, `business_jet`, `regional_jet`, `narrowbody_jet`, `widebody_jet`
  - model: ResNet-18 backbone, 1-channel mel spectrogram input, multi-label sigmoid head, BCEWithLogitsLoss with pos_weight balancing, dropout 0.5, SpecAugment
  - code: `src/aircraftClassifier/training/toolchain.py` (`VehicleAudioDataset` + `VehicleSoundClassifier`)

```bash
# Sync project + dataset to DGX
bash scripts/syncToDGX.sh spark-8d0d.local

# Pre-compute spectrograms once (or after adding new clips)
bash scripts/precomputeDGX.sh

# Train (basic)
bash scripts/trainDGX.sh --useCategories

# Train with backbone freezing to combat overfitting (recommended for small datasets)
bash scripts/trainDGX.sh \
    --useCategories \
    --freezeBackbone \
    --unfreezeEpoch 20 \
    --weightDecay 0.05 \
    --maxEpochs 60

# Evaluate a checkpoint
bash scripts/evalDGX.sh \
    --checkpoint /checkpoints/best.ckpt \
    --labelEncoder /checkpoints/labelEncoder.json \
    --valCsv dataset/val.csv \
    --useCategories --tuneThresholds
```

  **Overfitting controls** (all passed via `trainDGX.sh`):
  - `--freezeBackbone`: freeze conv1 through layer3; only layer4 + classifier are trained. Strongest single lever for small datasets — prevents the backbone from memorizing training examples. Reduces trainable parameters from ~11M to ~2M.
  - `--unfreezeEpoch N`: at epoch N, unfreeze the full backbone for end-to-end fine-tuning. The cosine LR schedule has decayed by then, so fine-tuning is gentle. Typical value: 15–25 (after early epochs establish a good classifier head).
  - `--weightDecay`: AdamW L2 penalty (default: 0.01). Increase to 0.05–0.1 for additional regularization.
  - `--noPosWeight`: disable automatic pos_weight balancing (not recommended unless the dataset is already balanced).

## Workflow Steps

1) Set up ADS-B capture device
  * Hardware
    - Rpi4B with two SDR dongles, RF splitter, and a dual-mode (1090/9??MHz) antenna
    - ?
  * Software
    - run `readsb` ????
    - make sure clock is synchronized with NTP
      * `timedatectl status`  # indicates whether NTP service is active
    - ?

2) Evaluate and select microphone and ADC
  * use scripts/evalMics.py to select microphone and ADC
  * ????

3) Set up audio capture device
  * Hardware
    - RPi0-2W with ???? ADC and ???? microphone
    - ???? rechargeable battery pack and ???? solar panel
    - waterproof enclosure
    - microphone wind screen
    - ?tower?
  * Software
    - make sure clock is synchronized with NTP
      * `timedatectl status`  # indicates whether NTP service is active
    - ????
```bash
python3 ./scripts/capture.py --host <serverIPA>
```

4) Set up server to gather ADS-B metadata and audio data
  * Hardware
    - ?
  * Software
    - receive ADS-B metadata from Rpi4B and audio samples from RPi0-2W and put them into .recordings/????
    - ?
```bash
python3 scripts/record.py \
      --lat <lat> \
      --lon <lon> \
      --radiusKm 8 \
      --outputDir ./recordings \
      --readsbUrl http://adsbrx.lan/tar1090/data/aircraft.json \
      --nullSampleInterval 90 --nullSampleDuration 10  # saves a 10 sec background clip every 1.5 mins when no aircraft is in range
```

5) Build training and validation dataset
  * ?sync metadata to audio, correct for clock skew,  generate splits, and write out dataset
    - use scripts/buildDataset.py to produce 'dataset/train.csv', 'dataset/val.csv', and 'dataset/clips/\*.wav'
```bash
python3 scripts/buildDataset.py \
      --recordingsDir ./recordings \
      --outputDir ./dataset \
      --faaDatabaseDir ./data/ReleasableAircraft \
      --autoCorrectClock \
      --maxCoTrackRatio 2.0 \  #### TODO figure out good value for this
      --maxDistanceKm 4.0 \
      --dropUnknown \
      --stratifyPhase \
      --balanceClasses  # auto-balance to rarest class count, or --maxPerClass 200  # cap each class at a given number
```
    - defaults to 5sec clips, 80% train and 20% validate, ????
  * balance classes, get ~1000 samples per class (including null cases)
    - ?

6) Pre-compute mel spectrograms
  * Saves a `<clip>.spec.npy` file alongside each WAV in `dataset/clips/`. Training loads these directly instead of computing spectrograms on the fly, which avoids a CPU bottleneck in the DataLoader.
  * Run once after building the dataset, and again whenever new clips are added.
  * On the recording server (CPU only, no GPU needed):
```bash
python scripts/precomputeSpecs.py \
    --trainCsv dataset/train.csv \
    --valCsv dataset/val.csv \
    [--workers 16]        # parallelism; default 16
    [--skipExisting]      # skip clips that already have a .spec.npy; default is to recompute all
```
  * On the DGX Spark (inside the training Docker image):
```bash
bash scripts/precomputeDGX.sh
```
  * The `.spec.npy` files are written alongside the WAVs and are included in the rsync to the DGX. Training falls back to computing spectrograms via librosa at runtime if a `.npy` file is missing.

7) Verify dataset quality and quantity
    - run test to check dataset
    - check the quality, class distribution (including null cases), and sampling context distribution of the dataset
      * want to be sure we have sufficient labeled examples of each category, under different capture circumstances (e.g., weather, time-of-day, etc.), and that there are approximately the same number of examples for each category
```bash
python scripts/inspectDataset.py --recordingsDir ./recordings --datasetCsv ./dataset/dataset.csv
```

8) Set up DGX Spark to train the models
  * Training runs inside `aircraft-audio-training:latest` (built from `docker/Dockerfile.training`)
  * The image is built automatically by `trainDGX.sh` on first run; rebuild only needed when `Dockerfile.training` changes
  * Checkpoints land in `./checkpoints/` on the DGX host
  * The NVIDIA persistence daemon must be running or GPU containers will fail to start:
```bash
sudo systemctl enable --now nvidia-persistenced
```

9) Training
  * Phase 1: classify by propulsion type, engine count, and wing type
```bash
bash scripts/syncToDGX.sh spark-8d0d.local
bash scripts/trainDGX.sh \
    --useCategories \
    --freezeBackbone \
    --unfreezeEpoch 20 \
    --weightDecay 0.05 \
    --maxEpochs 60
```
  * Phase 2: direction of travel (8 cardinal directions) — not yet implemented
  * Phase 3: speed estimation — not yet implemented

10) Evaluation
```bash
bash scripts/evalDGX.sh \
    --checkpoint /checkpoints/best.ckpt \
    --labelEncoder /checkpoints/labelEncoder.json \
    --valCsv dataset/val.csv \
    --useCategories --tuneThresholds
```
  - prints per-class AP, F1, precision, recall, and support; macro mAP and F1 summary

11) Inference
  - ?

## Design Notes

See [Link to design notes](DESIGN_NOTES.md)

## TODO

1. Direction and speed heads are not implemented (Objectives 2 & 3)
  * 'VehicleAudioDataset.__getitem__' (toolchain.py:128) returns (spec, typeLabel), but no 'directionClass' or 'velocityKts'
  * 'VehicleSoundClassifier' has only one output head (multi-label type)
    - the CSV carries the labels but training ignores them, so objectives 2 and 3 are currently unaddressed
  * the CLAUDE.md architecture already specifies three heads with masked loss for direction/speed on single-vehicle samples
    - have to implement the other heads for the other phases

2. Data leak: simultaneous aircraft produce overlapping recordings with different recordingIds
  * in recorder.py, when two aircraft both hit the departure trigger in the same poll, '\_saveRecording' is called twice back-to-back (lines 188–222)
    - each writes a WAV reading from the same circular buffer
    - the audio windows overlap heavily, but the 'recordingId' is {timestamp}\_{icao24} so they look like independent events
  * splitByEvent (clipExport.py:327) splits on recordingId, so nearly-identical audio can land on both sides of the train/val split, resulting in leakage
    - fix: when multiple aircraft trigger save in the same window, emit one recording with both aircraft as a multi-label annotation
      * or group by timestamp prefix in splitByEvent

3. 'typeToCategory' heuristic produces wrong labels for common cases
  * 'typeCategories.py':252 falls back on "piper" → piston_single, but a Piper Meridian is a turboprop and a Piper Malibu exists in both piston and turboprop variants
  * the FAA database (faaDatabaseDir) is authoritative and optional
    - it should be the default path, and 'typeToCategory's' keyword heuristic should only be a fallback for foreign/unknown ICAO24s
  * currently users who forget '--faaDatabaseDir' will silently get bad training labels

4. ImageNet ResNet is suboptimal for audio; PANNs/AST code exists but is unused
  * 'toolchain.py' uses ResNet-18 with ImageNet weights, with conv1 replaced by a fresh 1-channel layer
    - this results in losing the pretrained stem entirely
  * 'src/aircraftClassifier/pretrained' has PANNs and AST integration written but unused
    - PANNs (pretrained on AudioSet, which includes aircraft sounds) typically give a 10–20% F1 lift over ImageNet-initialized CNNs on small aviation datasets
  * this is the single highest-leverage change that can be made

5. AircraftDatabase.getAircraftType blocks on HTTP inside the recording save path
  * 'recorder.py':299 → 'typeDb.getAircraftType(icao24)' synchronously hits OpenSky during '\_saveRecording', which runs in the single monitoring thread
    - a slow OpenSky response delays the save and can make you miss fast-moving aircraft on the next poll
  * fix: move the lookup out of the hot path
    - resolve types lazily in buildDataset.py or via a background thread that pre-populates the cache

6. ~~Mel spectrogram is computed on CPU, per-sample, in the DataLoader~~ — **resolved**
  * `scripts/precomputeSpecs.py` pre-computes and caches `<clip>.spec.npy` alongside each WAV
  * `VehicleAudioDataset.__getitem__` loads `.npy` directly; SpecAugment is applied at load time for the training set

7. ~~No inference/evaluation tooling~~ — **resolved**
  * `scripts/evalModel.py` provides per-class AP/F1/precision/recall table, optional per-class threshold tuning, and single-WAV inference
  * `scripts/evalDGX.sh` runs it inside the training Docker image on the DGX Spark

8. Null samples are not being used for background noise augmentation
  * null (aircraft-free) clips are collected but '--bgNoiseDir' is only a manual opt-in
  * the null-clip directory in 'recordings/' is already the ideal 'bgNoiseDir'
  * Fix: auto-wire it or document the flow
    - 'AddBackgroundNoise' is the single highest-impact augmentation for this task (per the doc comment in audioAug.py:7)

9. MAX_SEEN_POS_SECS = 30 is too permissive for direction labels
  * 'readsb.py':19 keeps positions up to 30s stale
    - a jet at 400 kts moves ~6 km in 30s
  * the recorded 'headingDeg/bearingDeg' at save time could be grossly wrong
  * for direction training (Objective 2), tighten this to ~5s and drop clips whose seenSecs > 5 in 'buildDataset.py'
                                                                                                                                                                                                                     
10. 'directionClass' semantics are ambiguous
  * 'clipExport.py':81 quantizes the aircraft's absolute compass heading, but "direction of travel" from the observer's ear depends on bearing × heading together
    - a northbound aircraft passing east-of-you at 1 km sounds very different from a northbound aircraft passing
  west-of-you at 1 km
  * consider a relative-motion label (e.g. heading-minus-bearing quantized to 8 bins, or Doppler sign) rather than absolute heading.

11. vehicle_types is always a single-element list
  * 'clipExport.py':219: `vehicleTypes = [aircraftType] if aircraftType else []`
    - co-tracked aircraft types are discarded
  * for Objective 4 (i.e., multi-aircraft), co-tracked aircraft's types should appear in vehicle_types of each other's clips (or clips should be shared)
    - this undercuts the multi-label framing
12. Inline model re-implemented; library version ignored
  * 'src/aircraftClassifier/models/resNetCNN.py' defines the same architecture that 'toolchain.py' re-implements inline (currently ResNet-18)
  * fix: either delete the unused file or route training through it
    - only 'toolchain.py' + 'augmentation/audioAug.py' are imported from the classifier package

13. '--useCategories' should default to 'True'
  * the whole dataset pipeline is designed around categories (typeCategories.py, FAA lookup, type_categories column)
    - but 'toolchain.py':212 defaults to the raw 'vehicle_types' strings
    - this creates a class per variant (e.g., 737-800, 737-8H4, 737-824 as separate classes)
  * This will never train well
    - fix: make '--useCategories' default on, or rename to '--rawTypes' with default off

14. Differential learning rates
  * 'toolchain.py':197: single AdamW LR for the whole network
  * the replaced conv1 and the fresh classifier head have no pretraining
    - applying the same 1e-4 means they barely move
  * fix: use parameter groups with ~5–10× higher LR on new layers

## Recommendations
* PANNs vs AST vs. ResNet backbone
  - PANNs (CNN14): Drop-in swap
    * 'panns-inference' gives you a pretrained CNN14 trained on 2M AudioSet clips
      - You strip its final layer and attach your classifier head
    * low risk, probably +10–20% F1 over ImageNet ResNet
      - the code in 'pretrained/'' is already written
    * main downside: CNN14 expects 64-mel by default, your pipeline uses 128-mel, so you'd either adapt the input or re-derive embeddings
  - AST (Audio Spectrogram Transformer): State-of-art but transformer training dynamics are different
    * needs warmup, smaller LR, more data to not overfit
    * downsides
      - on a small dataset it can underperform a well-regularized CNN
      - harder to get right
  - recommendation: swap to PANNs CNN14 now
    * it's highest bang-for-buck before you have a large dataset
    * once you have 10k+ clips, re-evaluate AST
    * the question before implementing:
      - do you want to adapt the existing 128-mel pipeline to match CNN14's expected input, or use a different mel config?
      - CNN14 was trained with 64 mels / 1024 FFT / 320 hop at 32kHz, but the pretrained weights are flexible about input shape since
   the classifier head is replaced

* Direction and speed heads questions
  A) Do you have enough labeled single-aircraft clips?
    - direction and speed heads only backpropagate on 'isSingle'=1 samples
    - if most of your recordings so far are jets at altitude with co-tracked traffic, these heads will barely train
      * worth checking via 'inspectDataset.py' first
  B) what should 'directionClass' mean?
    * currently it's the aircraft's absolute compass heading (e.g. 045° → class 1 = NE)
    * but from an audio perspective, a plane heading NE while passing to your north sounds identical to a plane heading NE while passing to your south
      - same Doppler and same engine spectrum
    * the aurally meaningful label is the aircraft's heading relative to you, not its absolute heading
    * the data to compute this already exists
        - 'headingDeg and bearingDeg' are both in the CSV
    --> changed definition in code

# DataCollection.md

Documentation for the dataset capture and construction phase of AircraftAudioId.

---

## Overview

The data collection system uses two machines:

- **Pi Zero W** (`audiocap.lan`) — USB microphone capture, streams audio over TCP
- **Ubuntu server** (`gpuServer1.lan`) — receives audio, polls ADS-B, detects flyovers, saves recordings

After recording, the server exports a clip-level training dataset locally, then syncs to the DGX Spark for training.

---

## Hardware Setup

| Component | Hardware | Role |
|---|---|---|
| Microphone | USB microphone via Pi Zero W | Audio capture |
| ADS-B receiver | `adsbrx.lan` running readsb/dump1090 | Aircraft position data |
| Recording server | Ubuntu workstation | Orchestration, storage |
| Training machine | DGX Spark (`spark-8d0d.lan`) | Dataset export + model training |

---

## Step 1: Audio Capture (Pi Zero W)

**Script:** `scripts/capture.py` (runs on Pi Zero W)

The Pi captures USB microphone audio at 44100 Hz mono (int16) and streams it to the Ubuntu server over a persistent TCP connection.

**Wire protocol — one frame per chunk:**
```
[ 8 bytes: float64 Unix timestamp, big-endian ]  ← Pi-side wall clock
[ 4 bytes: uint32  byte length, big-endian    ]
[ N bytes: raw PCM S16LE mono                 ]
```

**NTP sync is critical.** The Pi has no hardware real-time clock; without NTP sync, its clock drifts on every boot. `piCapture.py` warns if the NTP offset exceeds 100 ms. Both machines must sync to the same NTP source:

```bash
# On Pi Zero W:
sudo timedatectl set-ntp true
timedatectl show-timesync --all

# On Ubuntu server:
timedatectl status
```

---

## Step 2: Audio Buffering (Server)

**Module:** `src/aircraftAudio/record/audioStream/remoteStream.py` — `RemoteAudioStream`

The server maintains a **60-second circular buffer** of incoming PCM audio. As each TCP frame arrives, the server:

1. Records the Pi-side timestamp and buffer write position (used later for alignment)
2. Measures the Pi−server clock offset: `piTimestamp − serverTime`; keeps a rolling median of 200 samples as `clockSkewSecs`
3. Detects gaps > 2 s (Pi reconnect or clock jump) and fills them with silence
4. Overwrites the oldest audio once the buffer fills

The buffer is always running; recordings are made by calling `getBuffer(durationSecs)` to retrieve the last N seconds.

---

## Step 3: ADS-B Monitoring (Server)

**Module:** `src/aircraftAudio/record/adsb/readsb.py` — `ReadsbClient`

Polls the readsb JSON endpoint (default: `http://adsbrx.lan/data/aircraft.json`) every second. For each aircraft in the response:

- Skips entries with stale position data (`seen_pos > 30 s`)
- Filters by altitude (`minAltitudeFt`, `maxAltitudeFt`)
- Computes **slant distance** (acoustically correct 3-D distance):
  ```
  groundKm   = geodesic distance to aircraft lat/lon
  altitudeKm = altitudeFt × 0.0003048
  distanceKm = sqrt(groundKm² + altitudeKm²)
  ```
- Filters by `radiusKm`
- Computes bearing from observer to aircraft
- Returns `AircraftState` objects with `capturedAt` set to server wall clock

---

## Step 4: Flyover Detection and Recording

**Module:** `src/aircraftAudio/record/recorder.py` — `AircraftRecordingSystem`

**Script:** `scripts/record.py` (run on Ubuntu server)

```bash
python scripts/record.py \
    --lat <lat> --lon <lon> \
    --radiusKm 20 \
    --outputDir ./recordings \
    [--listenPort 9876] \
    [--readsbUrl http://adsbrx.lan/data/aircraft.json] \
    [--minAltitudeFt 500] \
    [--maxAltitudeFt 15000] \
    [--nullSampleInterval 180] \
    [--nullSampleDuration 10] \
    [--postTriggerSecs 10]
```

### Detection Logic

The recorder tracks aircraft by ICAO24 hex code. For each aircraft:

1. **First appearance** — creates a tracking entry; logs callsign, distance, altitude, speed
2. **Each poll** — appends a new `AircraftState` to the tracked states list; updates `lastSeenTime`
3. **Departure trigger** — fires when either condition is met:
   - Last 3 distance measurements are strictly increasing (aircraft leaving), OR
   - Aircraft has been tracked for `MAX_RECORDING_SECS` (30 s)
4. **Post-trigger window** — after the trigger fires, the recorder continues collecting ADS-B states for `postTriggerSecs` (default: 10 s) to capture departure geometry before saving. This balances the number of approach vs departure states per recording.
5. **Save** — triggered by window expiry or the aircraft leaving range (no state for 3 × poll interval)

### Audio Duration

The saved audio window spans the tracked state range plus a 2-second tail, capped at 55 seconds:
```
durationSecs = clamp(stateSpan + 2.0, min=10.0, max=55.0)
```

### Null (Background) Sampling

When `--nullSampleInterval` is set, the recorder periodically saves a background clip when no aircraft are in range. Null clips capture ambient noise for the negative class in training. Default clip duration is 10 seconds; interval of 120–300 seconds is recommended.

### Saved Files

Per flyover event, two files are written:

**`recordings/audio/<recordingId>.wav`** — mono float32 WAV at 44100 Hz

**`recordings/metadata/<recordingId>.json`** — alignment and label metadata:
```json
{
  "recordingId":      "20250430_143012_a3b4c5",
  "startTime":        "2025-04-30T14:30:12.345",
  "audioStartTime":   1746020412.1,    // Pi-side Unix timestamp of sample 0
  "clockSkewSecs":    0.012,           // Pi clock − server clock (median over session)
  "duration":         28.5,
  "sampleRate":       44100,
  "observerLat":      37.5,
  "observerLon":      -122.3,
  "aircraftStates":   [...],           // ordered list of AircraftState dicts
  "closestAircraft":  {...},
  "minDistanceKm":    3.2,
  "aircraftType":     "Cessna 172",
  "coTrackedAircraft": [],
  "isNullSample":     false
}
```

Each entry in `aircraftStates` includes: `icao24`, `callsign`, `latitude`, `longitude`, `altitudeFt`, `velocityKts`, `headingDeg`, `distanceKm`, `bearingDeg`, `seenSecs`, `capturedAt` (server clock Unix timestamp).

---

## Step 5: Dataset Inspection

**Script:** `scripts/inspectDataset.py`

Run this before building the training dataset to verify recording quality:

```bash
python scripts/inspectDataset.py \
    --recordingsDir ./recordings \
    [--datasetCsv ./dataset/dataset.csv]
```

**Sections printed:**

1. **Recordings inventory** — file counts, collection timespan, recording hours, null sample count, duration and distance histograms, aircraft type distribution
2. **Alignment health** — fraction of ADS-B states that fall within their recording's audio window; clock skew diagnosis (warns if median offset is strongly negative or if > 25% of states pre-date the audio window)
3. **Dataset CSV summary** — clips per type, flight phase distribution, direction class distribution, velocity/altitude/distance histograms
4. **Audio quality** — RMS level and silence fraction histograms over a random sample of clips

**Interpreting alignment health:** A positive median `capturedAt − audioStartTime` offset is normal — states are expected to land in the middle of the recording window, not at the start. Only a large negative median (Pi clock ahead of server) indicates a real clock problem.

---

## Step 6: Dataset Construction

**Script:** `scripts/buildDataset.py`

Converts raw recordings into a clip-level training dataset.

```bash
python scripts/buildDataset.py \
    --recordingsDir ./recordings \
    --outputDir ./dataset \
    [--clipSecs 5.0] \
    [--faaDatabaseDir /path/to/ReleasableAircraft] \
    [--minDistanceKm 1.0] \
    [--maxDistanceKm 15.0] \
    [--maxCoTrackRatio 2.0] \
    [--dropUnknown] \
    [--balanceClasses] \
    [--maxPerClass 500] \
    [--stratifyPhase] \
    [--trainFrac 0.8]
```

### Per-State Clip Extraction

For each recording, `clipExport.py` calls `align.py` to map every ADS-B state to its sample position in the WAV:

```
sampleIndex = (capturedAt − audioStartTime + clockSkewSecs) × sampleRate
```

Clock correction priority (highest to lowest):
1. `clockSkewSecs` stored in recording metadata (current recorder.py)
2. `--clockCorrection` manual override (for old recordings without stored skew)
3. `--autoCorrectClock`: per-recording estimate (anchors last state to `duration − 1 s`)
4. No correction (skew = 0)

A fixed-length clip (`clipSecs`, default 5 s) is extracted centred on each in-window state. States outside the audio window are skipped.

### Filtering Options

| Flag | Effect |
|---|---|
| `--minDistanceKm` | Drop clips where aircraft is closer than threshold (reduces near-field saturation) |
| `--maxDistanceKm` | Drop clips where aircraft is farther than threshold (reduces weak-signal clips) |
| `--maxCoTrackRatio` | Drop clips where any co-tracked aircraft is within (ratio × primary distance) |
| `--dropUnknown` | Drop clips with no type label (null clips are kept regardless) |

### Labels Written per Clip

| Column | Description |
|---|---|
| `filepath` | Absolute path to the clip WAV |
| `recordingId` | Flyover event ID — used for train/val splitting |
| `vehicle_types` | JSON list of raw aircraft type strings |
| `type_categories` | JSON list of coarse categories: `piston_single`, `piston_twin`, `turboprop`, `helicopter`, `business_jet`, `regional_jet`, `narrowbody_jet`, `widebody_jet`, `unknown` |
| `isSingle` | 1 if exactly one aircraft was tracked in this recording |
| `flightPhase` | `approach` / `closest` / `departure` / `unknown` — derived from distance trend across adjacent states |
| `directionClass` | 0–7 — aircraft heading relative to observer (0=away, 4=toward, 2=crossing-right, etc.) |
| `velocityKts` | Aircraft speed at this state |
| `altitudeFt` | Aircraft altitude at this state |
| `distanceKm` | Slant distance from observer at this state |
| `bearingDeg` | Bearing from observer to aircraft |
| `headingDeg` | Aircraft heading (direction of travel) |
| `clipOffsetSecs` | Time offset of clip centre from audio start |

**Direction class formula:**
```
relativeDeg = (headingDeg − bearingDeg + 360) % 360
directionClass = floor((relativeDeg + 22.5) / 45) % 8
```
This encodes Doppler geometry independent of compass orientation: 0° = flying directly away, 180° = flying directly toward, 90° = crossing left-to-right.

### Aircraft Type Lookup

When `--faaDatabaseDir` is provided, type categories are resolved via ICAO24 lookup against the FAA ReleasableAircraft database. Without it, the system falls back to a keyword heuristic on the type string that misclassifies turboprop Pipers and similar variants. **Providing the FAA database is strongly recommended.**

Download: `https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download`

### Train/Val Split

The split is done **by flyover event** (recording ID), not by clip, to prevent data leakage. Clips from the same flyover cannot appear in both train and val. The default split is 80/20 (`--trainFrac 0.8`).

After splitting, optional class balancing is applied to the **training set only**; the validation set is left intact to reflect the true distribution.

```bash
# Balance training set so each label has equal clip count,
# keeping approach and departure equal within each label:
--balanceClasses --stratifyPhase
```

### Output Files

```
dataset/
├── clips/             ← fixed-length WAV clips
│   ├── <recordingId>_<stateIdx>.wav
│   └── <recordingId>_null.wav
├── dataset.csv        ← all clips (pre-split)
├── train.csv          ← training split (optionally balanced)
└── val.csv            ← validation split (unbalanced, true distribution)
```

---

## End-to-End Workflow

```bash
# 1. Start audio capture on Pi Zero W
#    (run on Pi via SSH: ssh pi@audiocap.lan)
python scripts/capture.py --host gpuServer1.lan --port 9876

# 2. Start recording system on Ubuntu server
python scripts/record.py \
    --lat 37.5 --lon -122.3 \
    --radiusKm 20 \
    --outputDir ./recordings \
    --nullSampleInterval 180 \
    --postTriggerSecs 10

# 3. Inspect recordings health
python scripts/inspectDataset.py --recordingsDir ./recordings

# 4. Build the dataset
python scripts/buildDataset.py \
    --recordingsDir ./recordings \
    --outputDir ./dataset \
    --faaDatabaseDir /path/to/ReleasableAircraft \
    --maxDistanceKm 15 \
    --dropUnknown \
    --balanceClasses --stratifyPhase

# 5. Inspect the dataset
python scripts/inspectDataset.py \
    --recordingsDir ./recordings \
    --datasetCsv ./dataset/dataset.csv

# 6. Sync to DGX Spark for training
bash scripts/syncToDGX.sh spark-8d0d.lan

# 7. Train (run on DGX Spark via SSH)
ssh spark-8d0d.lan
bash /home/jdn/Code/AircraftAudioId/scripts/trainDGX.sh --useCategories
```

---

## Known Limitations and Gotchas

- **Pi clock drift** — The Pi Zero W has no hardware RTC. After reboot, the clock is wrong until NTP syncs (typically within 30–60 s). Start the capture script only after `timedatectl` confirms sync.
- **Approach/departure imbalance** — The departure trigger fires when the aircraft starts leaving, so recordings naturally contain more approach states than departure states. The `--postTriggerSecs` window mitigates this, but recordings collected before this feature existed will have skewed phase distributions.
- **Co-tracked aircraft** — `isSingle=0` recordings contain audio from multiple simultaneous aircraft. Use `--maxCoTrackRatio` to exclude clips where a second aircraft was nearby at save time; note this is a recording-level filter, not a per-state filter.
- **Unknown types** — Aircraft not in the FAA database (military, foreign-registered) get `type_categories=["unknown"]`. Use `--dropUnknown` to exclude these from training or accept them as a separate class.
- **Old recordings** — Recordings made before `audioStartTime` and `clockSkewSecs` were added to the metadata cannot be aligned and are silently skipped by `buildDataset.py`. Use `--autoCorrectClock` as a best-effort fallback for recordings that have `audioStartTime` but no `clockSkewSecs`.

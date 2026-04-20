# Audio Classification of Aircraft

**WIP**

## Hardware

### ADS-B Capture

--> see ADSBMonitor repo

### Audio Capture

* RPi0-2W
  - added heat sink to RPi0-2W

* Microphone and ADC
  - Dayton Audio iMM-6C, USB-C calibrated microphone, 6mm condenser, CM6542
  - omnidirectional
  - Specs
    * 18Hz-20kHz
    * Max SPL: 120 dB (1% THD)
    * SNR: 70 dBA
  - Measurements
    * Name: iMM-6C: USB Audio (hw:0,0)
    * Noise floor: -68.9 dBFS (good)
    * Self-noise: -71.0 dBFS (~26-30dBA equivalent - limits quiet measurements)
    * Peak headroom: -55.7 dBFS (adequate)
    * SNR: 23.0 dB (poor - might be measurement problem)
    * MaxRate: 96000 Hz (good for wide-band analysis)
    * Spectral flatness: 0.0003

* ?

* **TODO**
  - evaluate other mic and ADC eval boards with SPI interface
  - design and build rechargeable (Nx 16750?) PSU with solar panel
  - design and build weatherproof enclosure
  - find proper mounting location

### ADS-B/Audio Processing

* Ubuntu machine with i7-7820X, 128GB DRAM, and GTX2080
* ?

### Model Training and Inference

* DGX Spark: GB10 (128GB unified DRAM, 20x ARM CPU cores, Blackwell GPU)

## Software

The RPi0-2W is running Trixie and uses ntp to keep its clock synchronized with that of the server.

The server is running Ubuntu on a deskside Intel CPU with 128GB of DRAM, and a GTX2080 GPU.


### ADS-B Capture (RPi Zero 2W)

* **`scripts/capture.py`**: script to stream audio from the RPi0-2W to the server
  - specify: port number, sample rate, audio input device, and target machine IP address
  - uses `src/aircraftAudio/audioStream/piCapture.py`
  - writes sample chunks over persistent TCP connection
    * format: <timeStamp> <chunkLength> <rawPCM_S16LE<mono>
  - e.g.,```bash
python3 ./scripts/capture.py --host <serverIPA>
```
* **`scripts/evalMics.py`**: tool to evaluate the quality of various microphone/ADC choices
  - run tests on RPi0-2W with cabling and power representative of what will be done in production
  - uses `src/aircraftAudio/micEval.py`, which uses the scipy.signal.welch library
  - computes a set of (both active and passive) metrics to judge quality of input signal path

### ADS-B and Audio Processing (Ubuntu Server)

* **`scripts/exportDataset.py`**: exports recorded sessions to a training CSV for use with toolchain.py
  - uses `src/aircraftAudio/export.py` to export the data in the 'VehicleAudioDataset' format
    * uses `pandas`
* **`scripts/record.py`**: runs the recording system that synchronizes and combines ADS-B and audio signals
  - uses `src/aircraftAudio/recorder.py` to coordinate the data from `readsb` and the RPi0-2W audio stream
  - e.g.,```bash
python scripts/record.py --lat <lat> --lon <lon> --radiusKm 8 --outputDir ./recordings --readsbUrl http://adsbrx.lan/tar1090/data/aircraft.json
```
* **`scripts/buildDataset.py`**: reads recordings (meta)data and generates training dataset suitable for input to `toolchain.py`
  - e.g.,
```bash
  python scripts/buildDataset.py --recordingsDir ./recordings --outputDir ./dataset \
      --faaDatabaseDir ./data/ReleasableAircraft

?    --clipSecs 5.0 \
?    --maxDistanceKm 8.0
```
  - this produces `dataset/train.csv` and `dataset/val.csv` which plug directly into toolchain.py's VehicleAudioDataset
  - `toolchain.py` expects 'filepath' (path to a 5-second clip WAV) and 'vehicle_types' (JSON list, e.g., ["B738"])
  - the generated CSV files contain 'directionClass' (i.e., values 0–7, from 'headingDeg') and 'velocityKts' (for when the direction and speed heads are added to the model)
* **`scripts/inspectDataset.py`**: provides a measure of the quantity, quality, and distribution of collected training/testing samples.

### Model Training (DGX Sparc)

* Phase 1: Classify by vehicle type (multi-label, single-aircraft clips)
  - the coarse categories in typeCategories.py are the current working labels:
    * piston_single, piston_twin, turboprop, helicopter, business_jet, regional_jet, narrowbody_jet, widebody_jet
  - ????

## Workflow

1) Set up ADS-B capture device
  * Hardware
    - Rpi4B with two SDR dongles, RF splitter, and a dual-mode (1090/9??MHz) antenna
    - ?
  * Software
    - run `readsb` ????
    - ?

2) Set up audio capture device
  * Hardware
    - RPi0-2W with ???? ADC and ???? microphone
    - ???? rechargeable battery pack and ???? solar panel
    - waterproof enclosure
    - microphone wind screen
    - ?tower?
  * Software
    - ?npt/crony?
    - run `python3 ./scripts/capture.py --host <serverIPA>`

3) Get Training/Validation Dataset
  * Gather and prepare dataset and generate splits
    - **`buildDataset.py`**: produces 'dataset/train.csv', 'dataset/val.csv', and 'dataset/clips/*.wav'
    - ?
```bash
python scripts/buildDataset.py \
      --recordingsDir ./recordings \
      --outputDir ./dataset \
      --faaDatabaseDir ./data/ReleasableAircraft \
      --autoCorrectClock \
      --maxCoTrackRatio 2.0

    - options:
      * clipSecs <float>: change clip length in secs (default: 5sec)
      * minDistanceKm <float>: filter out aircraft that are too close (can cause audio clipping)
      * maxDistanceKm <float>: filter out aircraft that are too far away to be heard clearly
      * trainFrac <float>: adjust ratio of train/val split (defaults to 80/20 -- i.e., 0.8)
      * clockCorrection <float>: manual global clock offset
        - only use if --autoCorrectClock option produces uniformly bad alignment
```

  * Verify dataset
    - before training, run test to check dataset
    - check the quality, class distribution (including null cases), and sampling context distribution of the dataset
      * want to be sure we have sufficient labeled examples of each category, under different capture circumstances (e.g., weather, time-of-day, etc.), and that there are approximately the same number of examples for each category
```bash
python scripts/inspectDataset.py --recordingsDir ./recordings --datasetDir ./dataset
```

4) Training
  * Phase 1: ?single aircraft, classify by type?
    - ?
    - `python -m aircraftClassifier.training.toolchain --trainCsv dataset/train.csv --valCsv dataset/val.csv --useCategories`
```bash
python -m aircraftClassifier.training.toolchain \
      --trainCsv dataset/train.csv \
      --valCsv dataset/val.csv \
      --useCategories \
      --bgNoiseDir dataset/clips  # use null clips as background noise source
```

  * Phase 2: ????
    - ?
  * Phase 3: ????
    - ?

## Design Notes

See DESIGN_NOTES.md <make link>

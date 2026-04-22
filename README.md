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

* **`scripts/capture.py`**: script to capture and stream audio from the RPi0-2W to the server
  - options
    * host:           IP address or hostname of the main recording machine/server
    * port:           TCP port to connect to (defaults to 9876)
    * deviceIndex:    sounddevice input device index (None = system default)
    * sampleRate:     Capture sample rate in Hz (defaults to 44100)
    * chunkFrames:    Number of audio frames per chunk sent over the wire (defaults to 4096)
  - uses `src/aircraftAudio/audioStream/piCapture.py`
  - writes sample chunks over persistent TCP connection
    * format: <timeStamp> <chunkLength> <rawPCM_S16LE<mono>
* **`tools/evalMics.py`**: tool to evaluate the quality of various microphone/ADC choices
  - options
    * deviceIndices:        Specific device indices to test; None = all input devices
    * passiveDurationSecs:  How long to record silence for passive metrics
    * activeDurationSecs:   How long to record reference tone for active metrics
    * outputDir:            If set, saves per-device WAV recordings and a JSON report
  - evaluates one or more attached audio input devices and ranks them by noise floor, SNR, and frequency response quality
  - the passive phase runs automatically from silence
  - the active phase prompts you to play a reference tone from a nearby speaker
    * a tone can be played from a browser with `tools/tone.html`
  - computes a set of (both active and passive) metrics to judge quality of input signal path
    * passive more (silence):
      - Noise floor (dBFS)  -- RMS of the captured silence window
      - Self-noise (dBFS)   -- minimum RMS across 1-second windows
      - Clipping headroom   -- peak amplitude (lower = more headroom remaining)
      - Max sample rate     -- highest rate the device accepts
    * active mode (reference tone, optional):
      - SNR (dB)            -- reference RMS minus noise floor
      - Spectral flatness   -- how flat the frequency response is (0–1, 1=perfect)
      - Frequency response  -- per-band RMS across 8 octave bands (125–16kHz)
  - also need run tests on RPi0-2W with cabling and power representative of what will be done in production
  - `src/aircraftAudio/capture/micEval.py` uses the scipy.signal.welch library

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
python scripts/buildDataset.py \
      --recordingsDir ./recordings \
      --outputDir ./dataset \
      --faaDatabaseDir ./data/ReleasableAircraft \
      --autoCorrectClock \
      --maxCoTrackRatio 2.0 \
      --dropUnknown \
      --balanceClasses  # auto-balance to rarest class count
  or
      --maxPerClass 200  # cap each class at a given number

    - options:
      * clipSecs <float>: change clip length in secs (default: 5sec)
      * minDistanceKm <float>: filter out aircraft that are too close (can cause audio clipping)
      * maxDistanceKm <float>: filter out aircraft that are too far away to be heard clearly
      * trainFrac <float>: adjust ratio of train/val split (defaults to 80/20 -- i.e., 0.8)
      * clockCorrection <float>: manual global clock offset
        - only use if --autoCorrectClock option produces uniformly bad alignment
```

  - this produces `dataset/train.csv` and `dataset/val.csv` which plug directly into toolchain.py's VehicleAudioDataset and reference the audio samples in `clips/`
  - `toolchain.py` expects 'filepath' (path to a 5-second clip WAV) and 'vehicle_types' (JSON list, e.g., ["B738"])
  - ????dataset.csv????
  - the generated CSV files contain 'directionClass' (i.e., values 0–7, from 'headingDeg') and 'velocityKts' (for when the direction and speed heads are added to the model)
* **`scripts/inspectDataset.py`**: provides a measure of the quantity, quality, and distribution of collected training/testing samples
  - e.g.,```bash
python3 scripts/inspectDataset.py --recordingsDir ./recordings
```
  - this takes an inventory of the samples in the dataset and prints information about the data dataset described in `<recordingsDir>/../dataset/dataset.csv`.
  - the information provided by this program includes:
    * number of Metadata files with matching WAV files and the number of missing WAV files
    * the number of single-aircraft and the number of multi-aircraft recordings
    * a graphical depiction of the distribution of durations of the recordings
    * a graph of the distribution of distances of the sampled aircraft
    * a graph, the percentages, and absolute number of each class of aircraft
    * 

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
      --nullSampleInterval 180  # saves a background clip every 3 minutes when no aircraft is in range
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
      --maxCoTrackRatio 2.0 \
      --dropUnknown \
      --balanceClasses  # auto-balance to rarest class count, or --maxPerClass 200  # cap each class at a given number
```
    - ?
  * ?balance classes, get ~1000 samples per class, include null cases?
    - ?

6) Set up DGX Spark to train the models
  * ?do everything in containers, install toolchain (preferrably NVIDIA versions)
  * ?
  * ?

7) Verify dataset quality and quantity
    - run test to check dataset
    - check the quality, class distribution (including null cases), and sampling context distribution of the dataset
      * want to be sure we have sufficient labeled examples of each category, under different capture circumstances (e.g., weather, time-of-day, etc.), and that there are approximately the same number of examples for each category
```bash
python scripts/inspectDataset.py --recordingsDir ./recordings --datasetDir ./dataset
```

8) Training
  * Phase 1: classify single aircraft by propulsion type, engine count, and wing type
    - coarse aircraft category labels:
      * piston_single
      * piston_twin
      * turboprop
      * helicopter
      * business_jet
      * regional_jet
      * narrowbody_jet
      * widebody_jet
```bash
python -m aircraftClassifier.training.toolchain \
      --trainCsv dataset/train.csv \
      --valCsv dataset/val.csv \
      --useCategories \           # classify based on coarse type labels
      --bgNoiseDir dataset/clips  # use null clips as background noise source
```

  * Phase 2: ????
    - ?
  * Phase 3: ????
    - ?
    - ?try to classify on engine type (e.g., GE GE90-115B, Lycoming IO-360-L2A, etc.)

9) Validation
  - ?

10) Inference
  - ?

## Design Notes

See DESIGN_NOTES.md <make link>

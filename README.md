# Audio Classification of Aircraft

**WIP**

## Hardware

### ADS-B Capture

--> see ADSBMonitor repo

### Audio Capture

* RPi0-2W + Mic + USB ADC
  - I don't have a good mic + USB ADC combination for measurements
* **TODO**
  - get better mic and ADC combo
  - look into AD ADC eval board with SPI interface
  - design and build rechargeable (Nx 16750?) PSU with solar panel
  - design and build weatherproof enclosure
  - find proper mounting location

### ADS-B/Audio Processing

* Ubuntu machine with 128GB DRAM and GTX2080
* ?

### Model Training and Inference

* DGX Spark

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
  - e.g., ```bash
python scripts/buildDataset.py \
    --recordingsDir ./recordings \
    --outputDir ./dataset \
    --clipSecs 5.0 \
    --maxDistanceKm 8.0
```
  

### Model Training (DGX Sparc)

* ????



* **`?.py`**: 
* 


## Design Notes

See DESIGN_NOTES.md <make link>

# Audio Classification of Aircraft

**WIP**

## Hardware

### ADS-B Capture

--> see ADSBMonitor repo

### Audio Capture

* RPi0-2W + Mic + USB ADC
* ?

### ADS-B/Audio Processing

* Ubuntu machine
* ?

### Model Training and Inference

* DGX Spark

## Software

### ADS-B Capture (RPi Zero 2W)

* evalMics.py: tool to evaluate the quality of various microphone/ADC choices
  - computes a set of (both active and passive) metrics to judge quality of input signal path
  - settled on MCm-1 mic and JSAUX USB ADC
  - uses src/aircraftAudio/micEval.py, which uses scipy.signal.welch



## Design Notes

See DESIGN_NOTES.md <make link>

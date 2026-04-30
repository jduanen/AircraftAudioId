# Audio Capture Device for Aircraft Audio Id Project

## Hardware

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

* 3D-printed enclosure
  - images
    ![](../assets/audioCapCase.jpg)
    ![](../assets/audioCapInternal.jpg)
    ![](../assets/audioCapExternal.jpg)
  - CAD files: TBD
    ![](../cad/top.3mf)
    ![](../assets/top.png)
    ![](../cad/bottom.3mf)
    ![](../assets/bottom.png)

* **TODO**
  - evaluate other mic and ADC eval boards with SPI interface
  - design and build rechargeable (Nx 16750?) PSU with solar panel
  - design and build weatherproof enclosure
  - find proper mounting location

## Software

* OS: Trixie
  - packages: chrony

* Virtual Environment: ?

* Microphone evaluation application
  - run this to help select the microphone to use here
  - **`../tools/evalMics.py`**: tool to evaluate the quality of various microphone/ADC choices
    * options
      - deviceIndices:        Specific device indices to test; None = all input devices
      - passiveDurationSecs:  How long to record silence for passive metrics
      - activeDurationSecs:   How long to record reference tone for active metrics
      - outputDir:            If set, saves per-device WAV recordings and a JSON report
    * evaluates one or more attached audio input devices and ranks them by noise floor, SNR, and frequency response quality
    * the passive phase runs automatically from silence
    * the active phase prompts you to play a reference tone from a nearby speaker
      - a tone can be played from a browser with `tools/tone.html`
    * computes a set of (both active and passive) metrics to judge quality of input signal path
      - passive more (silence):
        * Noise floor (dBFS)  -- RMS of the captured silence window
        * Self-noise (dBFS)   -- minimum RMS across 1-second windows
        * Clipping headroom   -- peak amplitude (lower = more headroom remaining)
        * Max sample rate     -- highest rate the device accepts
      - active mode (reference tone, optional):
        * SNR (dB)            -- reference RMS minus noise floor
        * Spectral flatness   -- how flat the frequency response is (0–1, 1=perfect)
        * Frequency response  -- per-band RMS across 8 octave bands (125–16kHz)
    * also need run tests on RPi0-2W with cabling and power representative of what will be done in production
    * `src/aircraftAudio/capture/micEval.py` uses the scipy.signal.welch library

* Aircraft flyover audio capture program
  - **`../scripts/capture.py`**: script to capture and stream audio from the RPi0-2W to the server
    * options
      - host:           IP address or hostname of the main recording machine/server
      - port:           TCP port to connect to (defaults to 9876)
      - deviceIndex:    sounddevice input device index (None = system default)
      - sampleRate:     Capture sample rate in Hz (defaults to 44100)
      - chunkFrames:    Number of audio frames per chunk sent over the wire (defaults to 4096)
    * uses `src/aircraftAudio/audioStream/piCapture.py`
    * writes sample chunks over persistent TCP connection
      - format: <timeStamp> <chunkLength> <rawPCM_S16LE>

## Operation

See [Data Collection Process](./DataCollection.md) for a description of the data collection and post-processing process.

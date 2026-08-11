## Microphone Selection Process

**TBD**

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

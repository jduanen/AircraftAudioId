# capture

Runs on the **Raspberry Pi Zero 2W**. Captures USB microphone audio and streams
it over TCP to the recording server.

## Modules

**`piCapture.py`** — `PiCapture`  
Captures audio from a USB mic via `sounddevice` and sends framed PCM chunks
over a persistent TCP connection. Reconnects automatically on drop. Checks NTP
clock offset at startup and warns if it exceeds 100 ms (important for alignment
with ADS-B data).

Wire format per chunk:
```
[ 8 bytes: float64 Unix timestamp, big-endian ]
[ 4 bytes: uint32  byte length,    big-endian ]
[ N bytes: raw PCM S16LE mono                 ]
```

**`micEval.py`** — `evaluateDevices`  
Evaluates attached USB microphone/ADC devices and ranks them by noise floor,
SNR, and frequency response. Run this before committing to a mic+ADC
combination for field deployment. Requires no calibrated equipment — passive
metrics come from silence, active metrics from a reference tone played nearby.

## Entry points

```bash
# Stream audio to the recording server
python scripts/capture.py --host <server-ip> --port 9876 [--device 1]

# Evaluate attached microphones
python scripts/evalMics.py [--duration 10] [--outputDir ./mic_eval]
```

## Dependencies

`sounddevice`, `soundfile`, `numpy`. Optionally `ntplib` for NTP offset check.

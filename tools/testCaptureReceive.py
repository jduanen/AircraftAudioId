#!/usr/bin/env python3
"""
Receive a piCapture.py audio stream and save it to a WAV file — for testing
the mic/capture path end-to-end without running the full recording system.

Reuses RemoteAudioStream (the same TCP receiver AircraftRecordingSystem
uses) rather than re-parsing the wire protocol.

Usage:
    # On the receiving machine (dev laptop, Ubuntu server, etc.):
    python tools/testCaptureReceive.py --port 9876 --duration 10 --output test.wav

    # On the capture machine (Pi Zero W, or a CM4 being bench-tested):
    python audioCapture/scripts/capture.py --host <receiver-ip> --port 9876 [--device N]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import soundfile as sf

from aircraftAudio.record.audioStream.remoteStream import RemoteAudioStream


def buildArgParser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=9876, help="TCP port to listen on (default: 9876)")
    p.add_argument("--sampleRate", type=int, default=44100, help="Expected sample rate (default: 44100)")
    p.add_argument("--duration", type=float, default=10.0, help="Seconds of audio to capture (default: 10)")
    p.add_argument("--output", type=str, default="testCapture.wav", help="Output WAV path")
    p.add_argument("--connectTimeoutSecs", type=float, default=30.0,
                   help="Seconds to wait for piCapture.py to connect (default: 30)")
    return p


def main():
    args = buildArgParser().parse_args()

    stream = RemoteAudioStream(port=args.port, sampleRate=args.sampleRate)
    stream.start()

    print(f"Waiting up to {args.connectTimeoutSecs:.0f}s for a connection on port {args.port} ...")
    waited = 0.0
    while not stream.isConnected() and waited < args.connectTimeoutSecs:
        time.sleep(1)
        waited += 1
    if not stream.isConnected():
        print(f"No connection after {args.connectTimeoutSecs:.0f}s — is piCapture.py pointed at this host/port?")
        stream.stop()
        sys.exit(1)
    print("Connected. Capturing audio ...")

    print(f"Waiting for {args.duration:.0f}s of real audio to accumulate ...")
    deadline = time.time() + args.connectTimeoutSecs + args.duration
    while not stream.isStreamHealthy(args.duration) and time.time() < deadline:
        time.sleep(0.5)
    if not stream.isStreamHealthy(args.duration):
        print("Stream did not stay healthy long enough to fill the requested duration — "
              "saving whatever's in the buffer anyway.")

    audio = stream.getBuffer(args.duration)
    stream.stop()

    sf.write(args.output, audio, args.sampleRate)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
    rmsDb = 20 * np.log10(rms) if rms > 0 else float("-inf")
    print(f"Saved {args.output}: {len(audio) / args.sampleRate:.1f}s, peak={peak:.4f}, RMS={rmsDb:.1f} dBFS")
    if peak < 1e-6:
        print("[warning] Buffer is silent — piCapture.py connected but no real audio arrived "
              "(check the mic device index and that the source isn't silent).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stream USB mic audio from the Pi Zero W to the main recording machine.

Usage:
    python scripts/capture.py --host <main-machine-ip> --port 9876
                              [--device 1] [--sampleRate 44100]

Run this on the Pi Zero W.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aircraftAudio.capture.piCapture import PiCapture, buildArgParser


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    PiCapture(
        host=args.host,
        port=args.port,
        deviceIndex=args.deviceIndex,
        sampleRate=args.sampleRate,
        chunkFrames=args.chunkFrames,
    ).start()

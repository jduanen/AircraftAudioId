#!/usr/bin/env python3
"""
Evaluate and compare USB microphones attached to the Pi Zero W.

Usage:
    python scripts/evalMics.py [--duration 10] [--outputDir ./micResults]
                               [--devices 1,2,3]

Run this on the Pi Zero W with all candidate microphones attached.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aircraftAudio.micEval import evaluateDevices, buildArgParser


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    indices = (
        [int(x) for x in args.devices.split(",")]
        if args.devices else None
    )
    evaluateDevices(
        deviceIndices=indices,
        passiveDurationSecs=args.passiveDuration,
        outputDir=args.outputDir,
    )

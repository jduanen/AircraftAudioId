#!/usr/bin/env python3
"""
Start the aircraft recording system on the main machine.

Usage:
    python scripts/record.py --lat 37.5 --lon -122.3 --radiusKm 20 --outputDir ./recordings
                             [--listenPort 9876] [--readsbUrl http://adsbrx.lan]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from aircraftAudio.record.recorder import AircraftRecordingSystem


def main():
    p = argparse.ArgumentParser(description="Record aircraft audio synchronized with ADS-B data.")
    p.add_argument("--lat", type=float, required=True, help="Observer latitude")
    p.add_argument("--lon", type=float, required=True, help="Observer longitude")
    p.add_argument("--radiusKm", type=float, default=20.0, help="Tracking radius in km")
    p.add_argument("--outputDir", type=str, default="recordings", help="Output directory")
    p.add_argument("--minAltitudeFt", type=float, default=500.0, help="Minimum aircraft altitude")
    p.add_argument("--sampleRate", type=int, default=44100, help="Audio sample rate")
    p.add_argument("--listenPort", type=int, default=9876, help="TCP port to receive Pi audio")
    p.add_argument("--readsbUrl", type=str, default="http://adsbrx.lan/data/aircraft.json",
                   help="readsb JSON endpoint URL")
    args = p.parse_args()

    system = AircraftRecordingSystem(
        observerLat=args.lat,
        observerLon=args.lon,
        outputDir=args.outputDir,
        radiusKm=args.radiusKm,
        minAltitudeFt=args.minAltitudeFt,
        sampleRate=args.sampleRate,
        listenPort=args.listenPort,
        readsbUrl=args.readsbUrl,
    )
    system.start()


if __name__ == "__main__":
    main()

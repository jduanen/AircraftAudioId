#!/usr/bin/env python3
"""
Start the Standalone Data Collection unit (runs on the CM4).

Waits for a GPS fix, then records aircraft flyovers using a local USB mic
and a locally-run readsb/dump1090-fa instance — no network dependency.

Usage:
    python scripts/standaloneRecord.py --outputDir ./recordings \
        --faaDatabaseDir /path/to/ReleasableAircraft \
        [--readsbUrl http://localhost/data/aircraft.json] \
        [--gpsPort /dev/serial0] [--ledChip gpiochip0 --ledLine 11]

Send SIGUSR1 to the running process to write a session summary snapshot
(session_<timestamp>.json in outputDir) without stopping the recorder:
    kill -SIGUSR1 <pid>
"""

import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from aircraftAudio.standalone.standaloneRecorder import StandaloneRecorder


def main():
    p = argparse.ArgumentParser(description="Standalone aircraft audio/ADS-B/GPS recording unit.")
    p.add_argument("--outputDir", type=str, default="recordings", help="Output directory")
    p.add_argument("--radiusKm", type=float, default=20.0, help="Tracking radius in km")
    p.add_argument("--minAltitudeFt", type=float, default=500.0, help="Minimum aircraft altitude")
    p.add_argument("--maxAltitudeFt", type=float, default=None,
                   help="Maximum aircraft altitude in feet. Disabled by default.")
    p.add_argument("--sampleRate", type=int, default=44100, help="Audio sample rate")
    p.add_argument("--readsbUrl", type=str, default="http://localhost/data/aircraft.json",
                   help="Local readsb/dump1090-fa JSON endpoint URL")
    p.add_argument("--micDeviceIndex", type=int, default=None,
                   help="sounddevice input device index (omit for system default)")
    p.add_argument("--chunkFrames", type=int, default=4096, help="Audio frames per capture callback")
    p.add_argument("--gpsPort", type=str, default="/dev/serial0", help="GPS receiver serial device")
    p.add_argument("--gpsBaud", type=int, default=9600, help="GPS receiver baud rate")
    p.add_argument("--gpsFixTimeoutSecs", type=float, default=None,
                   help="Give up waiting for a GPS fix after this many seconds (default: wait forever)")
    p.add_argument("--gpsMinSatellites", type=int, default=4,
                   help="Minimum satellites required to accept a GPS fix")
    p.add_argument("--ledChip", type=str, default="gpiochip0", help="gpiod chip name for the status LED")
    p.add_argument("--ledLine", type=int, default=11, help="GPIO line offset driving the status LED (RTS4)")
    p.add_argument("--ledBlinkIntervalSecs", type=float, default=0.5,
                   help="Half-period of the status LED's error/starting-up blink pattern")
    p.add_argument("--minFreeGb", type=float, default=2.0,
                   help="Halt new recordings once free disk space drops below this many GB")
    p.add_argument("--healthPollIntervalSecs", type=float, default=2.0,
                   help="How often to re-check audio/storage health to drive the status LED")
    p.add_argument("--nullSampleInterval", type=float, default=None,
                   help="Save a background (null) clip every N seconds when no aircraft is "
                        "in range. Disabled by default. Recommended: 120-300.")
    p.add_argument("--nullSampleDuration", type=float, default=10.0,
                   help="Duration of null clips in seconds (default: 10)")
    p.add_argument("--maxNullSamples", type=int, default=None,
                   help="Stop saving null clips once this many exist in outputDir.")
    p.add_argument("--postTriggerSecs", type=float, default=10.0,
                   help="Seconds to keep collecting departure states after the save trigger fires.")
    p.add_argument("--faaDatabaseDir", type=Path, required=True,
                   help="Path to unzipped FAA ReleasableAircraft directory. Required — this unit "
                        "is air-gapped, so aircraft-type lookups must resolve offline.")
    p.add_argument("--datasetCsv", type=Path, default=None,
                   help="Path to existing dataset.csv. Used to load current per-class clip counts "
                        "when --maxSamplesPerClass is set.")
    p.add_argument("--maxSamplesPerClass", type=int, default=None,
                   help="Skip recording aircraft whose coarse category already has this many clips "
                        "in dataset.csv. Requires --datasetCsv. Unknown/foreign aircraft are "
                        "always recorded regardless of this cap.")
    p.add_argument("--dropUnknown", action="store_true",
                   help="Skip recording aircraft whose type cannot be resolved to a known coarse category.")
    args = p.parse_args()

    recorder = StandaloneRecorder(
        outputDir=args.outputDir,
        radiusKm=args.radiusKm,
        minAltitudeFt=args.minAltitudeFt,
        maxAltitudeFt=args.maxAltitudeFt,
        sampleRate=args.sampleRate,
        readsbUrl=args.readsbUrl,
        micDeviceIndex=args.micDeviceIndex,
        chunkFrames=args.chunkFrames,
        gpsPort=args.gpsPort,
        gpsBaud=args.gpsBaud,
        gpsFixTimeoutSecs=args.gpsFixTimeoutSecs,
        gpsMinSatellites=args.gpsMinSatellites,
        ledChip=args.ledChip,
        ledLine=args.ledLine,
        ledBlinkIntervalSecs=args.ledBlinkIntervalSecs,
        minFreeBytes=int(args.minFreeGb * 1024 ** 3),
        healthPollIntervalSecs=args.healthPollIntervalSecs,
        nullSampleIntervalSecs=args.nullSampleInterval,
        nullSampleDurationSecs=args.nullSampleDuration,
        maxNullSamples=args.maxNullSamples,
        postTriggerSecs=args.postTriggerSecs,
        faaDatabaseDir=args.faaDatabaseDir,
        datasetCsv=args.datasetCsv,
        maxSamplesPerClass=args.maxSamplesPerClass,
        dropUnknown=args.dropUnknown,
    )

    signal.signal(signal.SIGUSR1, lambda signum, frame: recorder.dumpSessionSummary())

    # Convert SIGTERM (what systemd sends on stop/restart) into the same
    # KeyboardInterrupt-based shutdown path Ctrl-C already uses, so the
    # try/finally below runs recorder.stop() and the status LED is driven
    # off explicitly. Without this, SIGTERM kills the process with no
    # cleanup at all — confirmed on hardware that releasing a GPIO line
    # does not reset it to floating; the pin keeps driving whatever level
    # it was last set to, so an unhandled SIGTERM can leave the LED stuck
    # on or off rather than reading as "crashed".
    def _handleSigterm(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _handleSigterm)

    try:
        recorder.start()
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Watch the Standalone unit's shutdown button (RXD4/GPIO9) and trigger a
graceful `systemctl poweroff` on press. Runs as its own service
(shutdownButton.service), independent of standaloneRecorder.

Usage:
    python scripts/shutdownButtonWatch.py [--ledChip gpiochip0] [--line 9]
                                          [--debounceMs 50]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aircraftAudio.standalone.shutdownButton import ShutdownButton


def main():
    p = argparse.ArgumentParser(description="Standalone unit shutdown button watcher.")
    p.add_argument("--chip", type=str, default="gpiochip0", help="gpiod chip name")
    p.add_argument("--line", type=int, default=9, help="GPIO line offset (RXD4 = 9)")
    p.add_argument("--debounceMs", type=float, default=50.0, help="Debounce period in ms")
    args = p.parse_args()

    button = ShutdownButton(chip=args.chip, line=args.line, debounceMs=args.debounceMs)
    try:
        button.watch()
    except KeyboardInterrupt:
        pass
    finally:
        button.stop()


if __name__ == "__main__":
    main()

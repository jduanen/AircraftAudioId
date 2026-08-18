#!/usr/bin/env python3
"""
Boot-time WiFi gate check for the Standalone unit (runs once via
wifiGate.service, before networking comes up).

Reads input #0 of a PCF8574 I2C I/O expander:
  - not set (pulled low)  -> block the WiFi radio (rfkill block wifi)
  - set (pulled/left high) -> unblock it (rfkill unblock wifi)

Symmetric on purpose: the physical switch fully determines WiFi state each
boot, so there's no way to leave the radio stuck blocked (or stuck
unblocked) from a prior boot after the switch is moved.

Usage:
    python scripts/checkWifiGate.py [--address 0x20] [--busNum 1] [--bit 0]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aircraftAudio.standalone.ioExpander import Pcf8574


def main():
    p = argparse.ArgumentParser(description="Standalone unit boot-time WiFi gate check.")
    p.add_argument("--address", type=lambda x: int(x, 0), default=0x20,
                   help="PCF8574 I2C address (default: 0x20)")
    p.add_argument("--busNum", type=int, default=1, help="I2C bus number (default: 1)")
    p.add_argument("--bit", type=int, default=0, help="Input pin to check (default: 0)")
    args = p.parse_args()

    expander = Pcf8574(address=args.address, busNum=args.busNum)
    enabled = expander.readInput(args.bit)

    if enabled:
        print(f"[wifiGate] Input {args.bit} set — enabling WiFi.")
        subprocess.run(["rfkill", "unblock", "wifi"], check=False)
    else:
        print(f"[wifiGate] Input {args.bit} not set — disabling WiFi.")
        subprocess.run(["rfkill", "block", "wifi"], check=False)


if __name__ == "__main__":
    main()

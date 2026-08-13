#!/usr/bin/env python3
"""
Shutdown button watcher for the Standalone unit.

Watches a momentary switch on RXD4 (GPIO9, muxed to plain GPIO input via
the cm4-shutdown-button.dts overlay) and triggers a full, graceful system
shutdown on press (`systemctl poweroff`) — not just stopping
standaloneRecorder.service.

Runs as its own systemd service (shutdownButton.service), independent of
standaloneRecorder: the button works even if the recording service has
crashed or isn't running, and standaloneRecorder's own clean shutdown (see
scripts/standaloneRecord.py's SIGTERM handling) happens automatically as
part of the normal system-shutdown sequence systemd runs during poweroff —
no coordination needed between the two.

Relies on the CM4 bootloader EEPROM's POWER_OFF_ON_HALT=1 setting (see
StandaloneDataCollection.md) for the module to actually drop to low power
on halt rather than just idling until the next reboot.
"""

import subprocess
import time
from datetime import timedelta
from typing import Sequence

import gpiod
from gpiod.line import Bias, Direction, Edge


class ShutdownButton:
    """
    Args:
        chip:             gpiod chip name.
        line:              GPIO line offset for the button (RXD4 = GPIO9).
        debounceMs:        Kernel-level debounce period (gpiod's
                            LineSettings.debounce_period), filtering
                            mechanical switch contact bounce.
        shutdownCommand:   Command run on a detected press.
    """

    def __init__(
        self,
        chip: str = "gpiochip0",
        line: int = 9,
        debounceMs: float = 50.0,
        shutdownCommand: Sequence[str] = ("systemctl", "poweroff"),
    ):
        self.chip = chip
        self.line = line
        self.shutdownCommand = shutdownCommand

        self._request = gpiod.request_lines(
            f"/dev/{chip}",
            consumer="shutdownButton",
            config={
                line: gpiod.LineSettings(
                    direction=Direction.INPUT,
                    bias=Bias.PULL_UP,          # idle high; switch pulls to GND on press
                    edge_detection=Edge.FALLING,
                    debounce_period=timedelta(milliseconds=debounceMs),
                )
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def watch(self) -> None:
        """Blocks forever; runs shutdownCommand once per debounced press."""
        print(f"[shutdownButton] Watching {self.chip} line {self.line} for button press ...")
        while True:
            for _event in self._request.read_edge_events():
                print("[shutdownButton] Press detected — initiating shutdown")
                subprocess.run(self.shutdownCommand, check=False)

    def stop(self) -> None:
        self._request.release()


if __name__ == "__main__":
    button = ShutdownButton()
    try:
        button.watch()
    except KeyboardInterrupt:
        pass
    finally:
        button.stop()

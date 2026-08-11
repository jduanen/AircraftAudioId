#!/usr/bin/env python3
"""
GPIO status LED for the Standalone unit's illuminated power switch.

Behavior:
    solid on  = healthy (recorder running, storage OK, GPS fix acquired)
    blinking  = software-detected problem, or still starting up (waiting
                for a GPS fix) — "blinking" means "not currently fully
                operational", covering both cases with one state
    off       = no power, or process not running — requires no software,
                since nothing drives the pin in that state

Targets the gpiod v2 (libgpiod 2.x) Python bindings, which is what
`pip install gpiod` provides today and what Raspberry Pi OS Trixie ships.
All actual gpiod calls are isolated in _setLine() so a version mismatch
discovered during hardware bring-up only requires touching one method.
"""

import threading

import gpiod
from gpiod.line import Direction, Value


class StatusLed:
    """
    Args:
        chip:               gpiod chip name (e.g. "gpiochip0").
        line:                GPIO line offset driving the LED. Config value —
                              the exact pin depends on the RTS/CTS-to-GPIO
                              device-tree overlay wiring, not finalized yet.
        blinkIntervalSecs:   Half-period of the error/starting-up blink pattern.
    """

    def __init__(self, chip: str = "gpiochip0", line: int = 17, blinkIntervalSecs: float = 0.5):
        self.chip = chip
        self.line = line
        self.blinkIntervalSecs = blinkIntervalSecs

        self._request = gpiod.request_lines(
            f"/dev/{chip}",
            consumer="standaloneRecorder-statusLed",
            config={line: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)},
        )
        self._blinkThread: threading.Thread | None = None
        self._stopBlink = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setOk(self) -> None:
        """Solid on — fully healthy."""
        self._stopBlinking()
        self._setLine(True)

    def setError(self) -> None:
        """Blinking — software-detected problem, or still starting up."""
        if self._blinkThread is not None and self._blinkThread.is_alive():
            return
        self._stopBlink.clear()
        self._blinkThread = threading.Thread(target=self._blinkLoop, daemon=True)
        self._blinkThread.start()

    def setOff(self) -> None:
        """Drive the line low. Used on clean shutdown."""
        self._stopBlinking()
        self._setLine(False)

    def stop(self) -> None:
        """Release the GPIO line handle."""
        self._stopBlinking()
        self._request.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stopBlinking(self) -> None:
        if self._blinkThread is not None:
            self._stopBlink.set()
            self._blinkThread.join(timeout=self.blinkIntervalSecs * 2)
            self._blinkThread = None

    def _blinkLoop(self) -> None:
        state = False
        while not self._stopBlink.is_set():
            state = not state
            self._setLine(state)
            self._stopBlink.wait(self.blinkIntervalSecs)
        self._setLine(False)

    def _setLine(self, active: bool) -> None:
        """Isolates the actual gpiod call, quarantining any API-version differences."""
        self._request.set_value(self.line, Value.ACTIVE if active else Value.INACTIVE)

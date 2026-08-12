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

Wiring (per audioCapture/StandaloneDataCollection.md): the illuminated power switch's LED sits
between the 3V3 pin on UART0/1 and the RTS pin on USART4 (RTS4) — one leg
fixed high, the other pulled by this GPIO line. That makes it active-low:
driving the pin low sinks current and lights it, driving it high (matching
the 3V3 leg) turns it off. LineSettings(active_low=True) below handles that
inversion at the gpiod level, so setOk()/setError()/setOff() keep the usual
on/off meaning regardless of the physical polarity.

Targets the gpiod v2 (libgpiod 2.x) Python bindings — confirmed as v2.2.1 on
the standalone unit's Trixie image (`python3 -c "import gpiod; print(gpiod.api_version)"`).
All actual gpiod calls are isolated in _setLine() so any future API-version
change only requires touching one method.
"""

import threading

import gpiod
from gpiod.line import Direction, Value


class StatusLed:
    """
    Args:
        chip:               gpiod chip name (e.g. "gpiochip0").
        line:                GPIO line offset driving the LED — RTS4 (USART4),
                              BCM GPIO11 on the Ochin baseboard, muxed to
                              plain GPIO by the cm4-led-gpio.dts overlay.
        blinkIntervalSecs:   Half-period of the error/starting-up blink pattern.
    """

    def __init__(self, chip: str = "gpiochip0", line: int = 11, blinkIntervalSecs: float = 0.5):
        self.chip = chip
        self.line = line
        self.blinkIntervalSecs = blinkIntervalSecs

        self._request = gpiod.request_lines(
            f"/dev/{chip}",
            consumer="standaloneRecorder-statusLed",
            config={line: gpiod.LineSettings(
                direction=Direction.OUTPUT, output_value=Value.INACTIVE, active_low=True,
            )},
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
        """Turn the LED off. Used on clean shutdown."""
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

#!/usr/bin/env python3
"""
GPS fix acquisition for the Standalone unit.

Reads NMEA sentences off a serial GPS receiver and blocks until a valid fix
is obtained. Location is read exactly once at startup — relocating the
physical unit requires a power cycle anyway, so a single fresh fix at boot
is sufficient. There is no runtime location-change detection.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import pynmea2
import serial


@dataclass
class GpsFix:
    latitude: float
    longitude: float
    altitudeM: Optional[float]
    fixTime: float          # Unix timestamp when the fix was acquired
    satellites: Optional[int]


class GpsClient:
    """
    Blocks until a valid NMEA fix (GGA or RMC) is read from a serial GPS
    receiver.

    Args:
        port:            Serial device path for the GPS receiver's UART.
        baudrate:        Serial baud rate (GPS modules commonly default to
                          9600 or 38400).
        readTimeoutSecs: Per-line serial read timeout.
        serialFactory:   Optional callable(port, baudrate, timeout) -> object
                          with a readline() method, used in place of
                          serial.Serial. Overridable for testing without
                          real hardware.
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA3",
        baudrate: int = 9600,
        readTimeoutSecs: float = 2.0,
        serialFactory: Optional[Callable] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.readTimeoutSecs = readTimeoutSecs
        self._serialFactory = serialFactory or (
            lambda port, baudrate, timeout: serial.Serial(port, baudrate, timeout=timeout)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def waitForFix(
        self,
        maxWaitSecs: Optional[float] = None,
        minSatellites: int = 0,
    ) -> GpsFix:
        """
        Block until a valid GGA/RMC fix is received.

        Raises:
            TimeoutError: if maxWaitSecs elapses with no valid fix (None = wait forever).
        """
        stream = self._serialFactory(self.port, self.baudrate, self.readTimeoutSecs)
        startTime = time.time()
        print(f"[gps] Waiting for fix on {self.port} @ {self.baudrate} baud ...")

        try:
            while True:
                if maxWaitSecs is not None and time.time() - startTime > maxWaitSecs:
                    raise TimeoutError(f"No GPS fix within {maxWaitSecs:.0f}s on {self.port}")

                line = stream.readline()
                if not line:
                    continue

                fix = self._parseLine(line, minSatellites)
                if fix is not None:
                    print(
                        f"[gps] Fix acquired: {fix.latitude:.5f}, {fix.longitude:.5f} "
                        f"(sats: {fix.satellites})"
                    )
                    return fix
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parseLine(line: bytes, minSatellites: int) -> Optional[GpsFix]:
        try:
            text = line.decode(errors="ignore").strip()
            msg = pynmea2.parse(text)
        except (pynmea2.ParseError, UnicodeDecodeError):
            return None

        if isinstance(msg, pynmea2.types.talker.GGA):
            if not msg.gps_qual or msg.gps_qual == 0:
                return None
            satellites = int(msg.num_sats) if msg.num_sats else 0
            if satellites < minSatellites:
                return None
            return GpsFix(
                latitude=float(msg.latitude),
                longitude=float(msg.longitude),
                altitudeM=float(msg.altitude) if msg.altitude else None,
                fixTime=time.time(),
                satellites=satellites,
            )

        if isinstance(msg, pynmea2.types.talker.RMC):
            if msg.status != "A":  # A = active/valid, V = void
                return None
            return GpsFix(
                latitude=float(msg.latitude),
                longitude=float(msg.longitude),
                altitudeM=None,
                fixTime=time.time(),
                satellites=None,
            )

        return None

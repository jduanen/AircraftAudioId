#!/usr/bin/env python3
"""
GPS fix acquisition for the Standalone unit.

Reads NMEA sentences off a serial GPS receiver and blocks until a valid fix
is obtained. Location is read exactly once at startup — relocating the
physical unit requires a power cycle anyway, so a single fresh fix at boot
is sufficient. There is no runtime location-change detection.

Two clients are provided:
    GpsClient  — reads /dev/serial0 directly via pyserial + NMEA parsing.
    GpsdClient — queries gpsd's own TCP JSON protocol instead.

StandaloneRecorder uses GpsdClient, not GpsClient: gpsd holds the serial
device open with an exclusive lock (TIOCEXCL) for the chrony PPS/SHM time
bridge (see StandaloneDataCollection.md's GPS time discipline section), so a
second process opening /dev/serial0 directly fails with
`OSError: [Errno 16] Device or resource busy` — confirmed on hardware.
GpsClient is kept for standalone use (e.g. bring-up/testing) on a system
where gpsd isn't already running against the same device.
"""

import json
import socket
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
        port: str = "/dev/serial0",
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


class GpsdClient:
    """
    Blocks until gpsd reports a valid fix, via gpsd's own TCP JSON protocol
    (default port 2947) — used instead of GpsClient when gpsd already holds
    the serial device open (see module docstring).

    Args:
        host:          gpsd host (default: localhost).
        port:          gpsd port (default: 2947, gpsd's standard port).
        socketFactory: Optional callable(host, port) -> socket-like object
                        with makefile("rw"). Overridable for testing without
                        a real gpsd.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2947,
        socketFactory: Optional[Callable] = None,
    ):
        self.host = host
        self.port = port
        self._socketFactory = socketFactory or (
            lambda host, port: socket.create_connection((host, port), timeout=5)
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
        Block until gpsd reports a TPV fix (mode >= 2) with at least
        minSatellites satellites used (per the most recent SKY report).

        Raises:
            TimeoutError: if maxWaitSecs elapses with no valid fix (None = wait forever).
        """
        sock = self._socketFactory(self.host, self.port)
        stream = sock.makefile("rw")
        print(f"[gps] Waiting for fix from gpsd on {self.host}:{self.port} ...")

        try:
            stream.write('?WATCH={"enable":true,"json":true}\n')
            stream.flush()

            startTime = time.time()
            satellitesUsed = 0
            while True:
                if maxWaitSecs is not None and time.time() - startTime > maxWaitSecs:
                    raise TimeoutError(f"No gpsd fix within {maxWaitSecs:.0f}s")

                line = stream.readline()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msgClass = msg.get("class")
                if msgClass == "SKY":
                    # Prefer gpsd's own uSat summary field — present on every
                    # SKY report, full or abbreviated. gpsd alternates between
                    # full reports (with a complete satellites array) and
                    # abbreviated ones (uSat only, no array) — counting the
                    # array directly would silently zero out a correct count
                    # from the previous full report whenever an abbreviated
                    # one arrives (confirmed on hardware: this caused
                    # waitForFix to reject every fix indefinitely, since an
                    # abbreviated report landing right before a TPV always
                    # reset the tracked count to 0).
                    if "uSat" in msg:
                        satellitesUsed = msg["uSat"]
                    else:
                        satellitesUsed = sum(1 for s in msg.get("satellites", []) if s.get("used"))
                elif msgClass == "TPV":
                    if msg.get("mode", 0) < 2 or "lat" not in msg or "lon" not in msg:
                        continue
                    if satellitesUsed < minSatellites:
                        continue
                    fix = GpsFix(
                        latitude=msg["lat"],
                        longitude=msg["lon"],
                        altitudeM=msg.get("altHAE", msg.get("alt")),
                        fixTime=time.time(),
                        satellites=satellitesUsed or None,
                    )
                    print(
                        f"[gps] Fix acquired: {fix.latitude:.5f}, {fix.longitude:.5f} "
                        f"(sats: {fix.satellites})"
                    )
                    return fix
        finally:
            sock.close()

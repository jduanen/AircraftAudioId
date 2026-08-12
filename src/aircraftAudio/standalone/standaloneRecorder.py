#!/usr/bin/env python3
"""
Standalone Data Collection unit orchestrator.

Acquires a single GPS fix at startup (blocking — location is fixed for the
run, since relocating the unit requires a power cycle anyway), then wires
together a local audio stream, a local readsb ADS-B client, an offline
aircraft-type lookup, a disk-space guard, and a GPIO status LED around
AircraftRecordingSystem — mirroring scripts/record.py's structure for the
fully self-contained, offline field unit.
"""

import threading
import time
from pathlib import Path
from typing import Optional

from ..dataset.faaDatabase import FaaDatabase
from ..record.recorder import AircraftRecordingSystem
from ..record.storageGuard import StorageGuard
from ..record.adsb.readsb import ReadsbClient
from ..record.audioStream.localStream import LocalAudioStream
from .gps import GpsClient
from .statusLed import StatusLed


class _FaaTypeLookup:
    """
    Adapts FaaDatabase to the typeDb.getAircraftType(icao24) interface
    AircraftRecordingSystem expects, so recordings resolve an aircraft
    model string from the already-loaded, fully offline FAA registry
    instead of AircraftDatabase's live OpenSky HTTPS call — unreachable,
    and a multi-second stall per save, on this unit's air-gapped network.
    """

    def __init__(self, faaDb: FaaDatabase):
        self._faaDb = faaDb

    def getAircraftType(self, icao24: str) -> Optional[str]:
        info = self._faaDb.infoForIcao24(icao24)
        return info.get("model") if info else None


class StandaloneRecorder:
    """
    Args:
        outputDir, radiusKm, minAltitudeFt, maxAltitudeFt, sampleRate,
        readsbUrl, nullSampleIntervalSecs, nullSampleDurationSecs,
        maxNullSamples, postTriggerSecs, faaDatabaseDir, datasetCsv,
        maxSamplesPerClass, dropUnknown:
            Passed straight through to AircraftRecordingSystem — see
            recorder.py for full docs. faaDatabaseDir is required here
            (unlike record.py) — see the class docstring below.
        micDeviceIndex, chunkFrames:
            LocalAudioStream config.
        gpsPort, gpsBaud, gpsFixTimeoutSecs, gpsMinSatellites:
            GpsClient config. gpsFixTimeoutSecs=None waits forever.
        ledChip, ledLine, ledBlinkIntervalSecs:
            StatusLed config.
        minFreeBytes:
            StorageGuard threshold.
        healthPollIntervalSecs:
            How often the health-loop thread re-checks audio/storage
            health to drive the status LED.

    faaDatabaseDir is required (raises ValueError if omitted): without it,
    AircraftRecordingSystem would fall back to the online AircraftDatabase
    default, silently reintroducing a per-save network stall on a unit
    that's air-gapped by design.
    """

    def __init__(
        self,
        outputDir: str = "recordings",
        radiusKm: float = 20.0,
        minAltitudeFt: float = 500.0,
        maxAltitudeFt: Optional[float] = None,
        sampleRate: int = 44100,
        readsbUrl: str = "http://localhost/data/aircraft.json",
        micDeviceIndex: Optional[int] = None,
        chunkFrames: int = 4096,
        gpsPort: str = "/dev/serial0",
        gpsBaud: int = 9600,
        gpsFixTimeoutSecs: Optional[float] = None,
        gpsMinSatellites: int = 4,
        ledChip: str = "gpiochip0",
        ledLine: int = 11,
        ledBlinkIntervalSecs: float = 0.5,
        minFreeBytes: int = 2 * 1024 ** 3,
        healthPollIntervalSecs: float = 2.0,
        nullSampleIntervalSecs: Optional[float] = None,
        nullSampleDurationSecs: float = 10.0,
        maxNullSamples: Optional[int] = None,
        postTriggerSecs: float = 10.0,
        faaDatabaseDir: Optional[Path] = None,
        datasetCsv: Optional[Path] = None,
        maxSamplesPerClass: Optional[int] = None,
        dropUnknown: bool = False,
    ):
        if faaDatabaseDir is None:
            raise ValueError(
                "faaDatabaseDir is required for StandaloneRecorder — without it, "
                "aircraft-type lookups would fall back to a live OpenSky HTTPS call, "
                "which will stall on every save on this unit's air-gapped network."
            )

        self.outputDir = outputDir
        self.radiusKm = radiusKm
        self.minAltitudeFt = minAltitudeFt
        self.maxAltitudeFt = maxAltitudeFt
        self.sampleRate = sampleRate
        self.readsbUrl = readsbUrl
        self.micDeviceIndex = micDeviceIndex
        self.chunkFrames = chunkFrames
        self.gpsPort = gpsPort
        self.gpsBaud = gpsBaud
        self.gpsFixTimeoutSecs = gpsFixTimeoutSecs
        self.gpsMinSatellites = gpsMinSatellites
        self.ledChip = ledChip
        self.ledLine = ledLine
        self.ledBlinkIntervalSecs = ledBlinkIntervalSecs
        self.minFreeBytes = minFreeBytes
        self.healthPollIntervalSecs = healthPollIntervalSecs
        self.nullSampleIntervalSecs = nullSampleIntervalSecs
        self.nullSampleDurationSecs = nullSampleDurationSecs
        self.maxNullSamples = maxNullSamples
        self.postTriggerSecs = postTriggerSecs
        self.faaDatabaseDir = faaDatabaseDir
        self.datasetCsv = datasetCsv
        self.maxSamplesPerClass = maxSamplesPerClass
        self.dropUnknown = dropUnknown

        self._system: Optional[AircraftRecordingSystem] = None
        self._led: Optional[StatusLed] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Blocks: wait for GPS fix, build components, run the recording loop."""
        self._led = StatusLed(
            chip=self.ledChip, line=self.ledLine, blinkIntervalSecs=self.ledBlinkIntervalSecs
        )
        self._led.setError()  # blink while starting up / waiting for GPS fix

        gps = GpsClient(port=self.gpsPort, baudrate=self.gpsBaud)
        fix = gps.waitForFix(maxWaitSecs=self.gpsFixTimeoutSecs, minSatellites=self.gpsMinSatellites)
        print(f"[standalone] Observer position: {fix.latitude}, {fix.longitude}")

        storageGuard = StorageGuard(self.outputDir, self.minFreeBytes)
        faaDb = FaaDatabase(self.faaDatabaseDir)
        typeDb = _FaaTypeLookup(faaDb)

        adsbClient = ReadsbClient(observerLat=fix.latitude, observerLon=fix.longitude, url=self.readsbUrl)
        audioStream = LocalAudioStream(
            deviceIndex=self.micDeviceIndex, sampleRate=self.sampleRate, chunkFrames=self.chunkFrames
        )

        self._system = AircraftRecordingSystem(
            observerLat=fix.latitude,
            observerLon=fix.longitude,
            outputDir=self.outputDir,
            radiusKm=self.radiusKm,
            minAltitudeFt=self.minAltitudeFt,
            maxAltitudeFt=self.maxAltitudeFt,
            sampleRate=self.sampleRate,
            nullSampleIntervalSecs=self.nullSampleIntervalSecs,
            nullSampleDurationSecs=self.nullSampleDurationSecs,
            maxNullSamples=self.maxNullSamples,
            postTriggerSecs=self.postTriggerSecs,
            faaDatabaseDir=self.faaDatabaseDir,
            datasetCsv=self.datasetCsv,
            maxSamplesPerClass=self.maxSamplesPerClass,
            dropUnknown=self.dropUnknown,
            adsbClient=adsbClient,
            audioStream=audioStream,
            typeDb=typeDb,
            storageGuard=storageGuard,
        )

        self._running = True
        healthThread = threading.Thread(target=self._healthLoop, daemon=True)
        healthThread.start()

        self._system.start()  # blocks until Ctrl+C

    def stop(self) -> None:
        self._running = False
        if self._system is not None:
            self._system.stop()
        if self._led is not None:
            self._led.setOff()
            self._led.stop()

    def dumpSessionSummary(self) -> None:
        if self._system is not None:
            self._system.dumpSessionSummary()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _healthLoop(self) -> None:
        """Daemon thread: drive the LED from audioStream/storage health."""
        storageGuard = self._system.storageGuard
        audioStream = self._system.audioStream
        while self._running:
            healthy = audioStream.isStreamHealthy() and (
                storageGuard is None or storageGuard.hasSpace()
            )
            if healthy:
                self._led.setOk()
            else:
                self._led.setError()
            time.sleep(self.healthPollIntervalSecs)

#!/usr/bin/env python3
"""
Aircraft recording orchestrator.

Coordinates the readsb ADS-B client and the remote audio stream to detect
aircraft flyovers and save synchronized audio + metadata to disk.
"""

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .adsb import AircraftState
from .adsb.readsb import ReadsbClient
from .audioStream.remoteStream import RemoteAudioStream
from .aircraftType import AircraftDatabase


# How often the monitoring loop polls for new ADS-B data (seconds).
POLL_INTERVAL_SECS = 1.0

# Maximum recording duration cap in seconds (catches aircraft that linger).
MAX_RECORDING_SECS = 45.0

# Minimum number of ADS-B states before we consider saving a recording.
MIN_STATES_BEFORE_SAVE = 3


@dataclass
class RecordingMetadata:
    recordingId: str
    startTime: str
    audioStartTime: float        # Pi-side Unix timestamp of sample 0 in the WAV
    clockSkewSecs: Optional[float]   # Pi clock − server clock at recording time
    duration: float
    sampleRate: int
    observerLat: float
    observerLon: float
    aircraftStates: list
    closestAircraft: Optional[dict]
    minDistanceKm: Optional[float]
    aircraftType: Optional[str]
    coTrackedAircraft: list      # other aircraft tracked simultaneously (icao24 + distanceKm)
    isNullSample: bool = False   # True for background-noise-only recordings


class AircraftRecordingSystem:
    """
    Monitors aircraft via readsb and saves audio clips + metadata when a
    flyover is detected.

    Audio comes from a RemoteAudioStream receiving a Pi Zero W capture stream.
    ADS-B data comes from a ReadsbClient polling adsbrx.lan (or a custom URL).

    Args:
        observerLat:       Your latitude (decimal degrees).
        observerLon:       Your longitude (decimal degrees).
        outputDir:         Root directory for saved recordings.
        radiusKm:          Only track aircraft within this distance.
        minAltitudeFt:     Ignore aircraft below this altitude.
        sampleRate:        Audio sample rate (must match Pi capture setting).
        listenPort:        TCP port to receive audio from the Pi.
        readsbUrl:         URL of the readsb JSON endpoint.
    """

    def __init__(
        self,
        observerLat: float,
        observerLon: float,
        outputDir: str = "recordings",
        radiusKm: float = 20.0,
        minAltitudeFt: float = 500.0,
        sampleRate: int = 44100,
        listenPort: int = 9876,
        readsbUrl: str = "http://adsbrx.lan/data/aircraft.json",
        nullSampleIntervalSecs: Optional[float] = None,
        nullSampleDurationSecs: float = 10.0,
    ):
        self.observerLat = observerLat
        self.observerLon = observerLon
        self.radiusKm = radiusKm
        self.minAltitudeFt = minAltitudeFt
        self.sampleRate = sampleRate
        self.nullSampleIntervalSecs = nullSampleIntervalSecs
        self.nullSampleDurationSecs = nullSampleDurationSecs

        self.outputDir = Path(outputDir)
        self.audioDir = self.outputDir / "audio"
        self.metadataDir = self.outputDir / "metadata"
        self.audioDir.mkdir(parents=True, exist_ok=True)
        self.metadataDir.mkdir(parents=True, exist_ok=True)

        self.adsbClient = ReadsbClient(
            observerLat=observerLat,
            observerLon=observerLon,
            url=readsbUrl,
            pollIntervalSecs=POLL_INTERVAL_SECS,
        )
        self.audioStream = RemoteAudioStream(
            port=listenPort,
            sampleRate=sampleRate,
        )
        self.typeDb = AircraftDatabase()

        # icao24 → list of state dicts (from asdict(AircraftState))
        self._trackedAircraft: dict[str, list[dict]] = {}
        # icao24 → time of first detection (for max-duration cap)
        self._firstSeenTime: dict[str, float] = {}
        # icao24 → (capturedAt, distanceKm) of the closest-approach state seen so far
        self._closestApproach: dict[str, tuple[float, float]] = {}
        # Set of icao24 for which we've already saved a recording this pass.
        self._savedIcao: set[str] = set()

        self._running = False
        self._lastNullSampleTime: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the system (blocks until Ctrl+C)."""
        print("=" * 60)
        print("AIRCRAFT RECORDING SYSTEM")
        print("=" * 60)
        print(f"Observer:   {self.observerLat}, {self.observerLon}")
        print(f"Radius:     {self.radiusKm} km")
        print(f"Output:     {self.outputDir}")
        print("=" * 60)

        self.audioStream.start()
        print("Waiting for Pi audio stream to connect ...")

        # Give the audio stream a moment to receive a connection before we
        # start monitoring, but don't block indefinitely.
        for _ in range(30):
            if self.audioStream.isConnected():
                break
            time.sleep(1)

        if not self.audioStream.isConnected():
            print("[warning] No Pi audio stream connected yet — will save audio when available.")

        self._running = True
        monitorThread = threading.Thread(target=self._monitoringLoop, daemon=True)
        monitorThread.start()

        if self.nullSampleIntervalSecs:
            nullThread = threading.Thread(target=self._nullSamplingLoop, daemon=True)
            nullThread.start()
            print(f"Null sampling every {self.nullSampleIntervalSecs:.0f}s when no aircraft in range.")

        print("Running. Press Ctrl+C to stop.\n")
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.audioStream.stop()
        self._saveSummary()
        print("Stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _monitoringLoop(self) -> None:
        while self._running:
            try:
                aircraft = self.adsbClient.getAircraft(
                    radiusKm=self.radiusKm,
                    minAltitudeFt=self.minAltitudeFt,
                )
                if aircraft:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] {len(aircraft)} aircraft within {self.radiusKm} km")
                for state in aircraft:
                    self._processAircraft(state)
            except Exception as e:
                print(f"[recorder] monitoring error: {e}")

    def _processAircraft(self, state: AircraftState) -> None:
        icao24 = state.icao24
        now = time.time()

        if icao24 not in self._trackedAircraft:
            self._trackedAircraft[icao24] = []
            self._firstSeenTime[icao24] = now
            print(
                f"  + {state.callsign or icao24:10s}  "
                f"{state.distanceKm:.1f} km  "
                f"{state.altitudeFt:.0f} ft  "
                f"{state.velocityKts:.0f} kts"
            )

        self._trackedAircraft[icao24].append(asdict(state))

        # Keep _closestApproach updated to the state with minimum distance so far.
        stateDict = self._trackedAircraft[icao24][-1]
        prev = self._closestApproach.get(icao24)
        if prev is None or stateDict["distanceKm"] <= prev[1]:
            self._closestApproach[icao24] = (stateDict["capturedAt"], stateDict["distanceKm"])

        if self._shouldRecord(icao24, now):
            self._saveRecording(icao24)

    def _shouldRecord(self, icao24: str, now: float) -> bool:
        if icao24 in self._savedIcao:
            return False

        states = self._trackedAircraft[icao24]
        if len(states) < MIN_STATES_BEFORE_SAVE:
            return False

        # Trigger 1: symmetric departure window.
        # Wait until the departure phase is as long as the approach phase so
        # clips are evenly split between approaching and departing audio.
        # The departure phase starts at closest approach; the approach phase
        # starts at first detection.
        distances = [s["distanceKm"] for s in states[-3:]]
        if len(distances) == 3 and distances[0] < distances[1] < distances[2]:
            closestEntry = self._closestApproach.get(icao24)
            firstAt = states[0]["capturedAt"]
            if closestEntry is not None:
                closestAt = closestEntry[0]
                if closestAt > firstAt:
                    approachSecs = closestAt - firstAt
                    departureSecs = states[-1]["capturedAt"] - closestAt
                    if departureSecs < approachSecs:
                        return False  # keep waiting for symmetric departure data

            return True

        # Trigger 2: max duration cap.
        elapsed = now - self._firstSeenTime.get(icao24, now)
        if elapsed >= MAX_RECORDING_SECS:
            return True

        return False

    def _nullSamplingLoop(self) -> None:
        while self._running:
            time.sleep(self.nullSampleIntervalSecs)
            if not self._trackedAircraft and self.audioStream.isConnected():
                now = time.time()
                if now - self._lastNullSampleTime >= self.nullSampleIntervalSecs:
                    self._saveNullRecording()
                    self._lastNullSampleTime = now

    def _saveNullRecording(self) -> None:
        audio = self.audioStream.getBuffer(self.nullSampleDurationSecs)
        audioStartTime = self.audioStream.getBufferStartTime(self.nullSampleDurationSecs)
        clockSkewSecs = self.audioStream.getClockSkewSecs()

        recordingId = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_null"
        audioPath = self.audioDir / f"{recordingId}.wav"
        sf.write(str(audioPath), audio, self.sampleRate)

        metadata = RecordingMetadata(
            recordingId=recordingId,
            startTime=datetime.now().isoformat(),
            audioStartTime=audioStartTime,
            clockSkewSecs=clockSkewSecs,
            duration=self.nullSampleDurationSecs,
            sampleRate=self.sampleRate,
            observerLat=self.observerLat,
            observerLon=self.observerLon,
            aircraftStates=[],
            closestAircraft=None,
            minDistanceKm=None,
            aircraftType=None,
            coTrackedAircraft=[],
            isNullSample=True,
        )
        metaPath = self.metadataDir / f"{recordingId}.json"
        with open(metaPath, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        print(f"  [null] Saved background sample {recordingId}")

    def _saveRecording(self, icao24: str) -> None:
        states = self._trackedAircraft[icao24]
        self._savedIcao.add(icao24)

        # Duration: span of tracked states + 2s tail, capped.
        spanSecs = states[-1]["capturedAt"] - states[0]["capturedAt"]
        durationSecs = max(10.0, min(spanSecs + 2.0, MAX_RECORDING_SECS))

        audio = self.audioStream.getBuffer(durationSecs)
        audioStartTime = self.audioStream.getBufferStartTime(durationSecs)
        clockSkewSecs = self.audioStream.getClockSkewSecs()

        closestState = min(states, key=lambda s: s["distanceKm"])
        minDistanceKm = closestState["distanceKm"]

        # Look up aircraft type (non-blocking; may return None).
        aircraftType = self.typeDb.getAircraftType(icao24)

        recordingId = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{icao24}"

        audioPath = self.audioDir / f"{recordingId}.wav"
        sf.write(str(audioPath), audio, self.sampleRate)

        coTracked = [
            {"icao24": other, "distanceKm": self._trackedAircraft[other][-1]["distanceKm"]}
            for other in self._trackedAircraft
            if other != icao24 and self._trackedAircraft[other]
        ]

        metadata = RecordingMetadata(
            recordingId=recordingId,
            startTime=datetime.now().isoformat(),
            audioStartTime=audioStartTime,
            clockSkewSecs=clockSkewSecs,
            duration=durationSecs,
            sampleRate=self.sampleRate,
            observerLat=self.observerLat,
            observerLon=self.observerLon,
            aircraftStates=states,
            closestAircraft=closestState,
            minDistanceKm=minDistanceKm,
            aircraftType=aircraftType,
            coTrackedAircraft=coTracked,
        )

        metaPath = self.metadataDir / f"{recordingId}.json"
        with open(metaPath, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        print(
            f"\n  SAVED {recordingId}\n"
            f"    callsign:  {closestState.get('callsign') or 'unknown'}\n"
            f"    type:      {aircraftType or 'unknown'}\n"
            f"    closest:   {minDistanceKm:.2f} km\n"
            f"    altitude:  {closestState['altitudeFt']:.0f} ft\n"
            f"    speed:     {closestState['velocityKts']:.0f} kts\n"
            f"    duration:  {durationSecs:.1f} s\n"
        )

    def _saveSummary(self) -> None:
        summary = {
            "sessionEnd": datetime.now().isoformat(),
            "totalTracked": len(self._trackedAircraft),
            "recordingsSaved": len(self._savedIcao),
            "observerPosition": {"lat": self.observerLat, "lon": self.observerLon},
            "config": {
                "radiusKm": self.radiusKm,
                "minAltitudeFt": self.minAltitudeFt,
                "sampleRate": self.sampleRate,
            },
        }
        summaryPath = self.outputDir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summaryPath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Session summary saved: {summaryPath}")

#!/usr/bin/env python3
"""
Local audio capture stream for the Standalone unit.

Captures directly from a USB microphone via sounddevice.InputStream and
maintains a circular buffer, exposing the same interface RemoteAudioStream
provides to AircraftRecordingSystem — no TCP/network layer, since capture
happens in-process on the same machine that runs the recorder.

Because audio capture and ADS-B polling share one wall clock, clockSkewSecs
is always 0.0 — unlike RemoteAudioStream, there is no clock-skew estimation
to do.
"""

import threading
import time

import numpy as np
import sounddevice as sd


class LocalAudioStream:
    """
    Captures audio from a USB mic via sounddevice and maintains a circular
    buffer, accessible via getBuffer().

    The interface mirrors RemoteAudioStream so AircraftRecordingSystem can
    be constructed with either one interchangeably (see its audioStream
    constructor arg).

    Args:
        deviceIndex:        sounddevice input device index (None = system default).
        sampleRate:          Capture sample rate (Hz).
        chunkFrames:         Frames per sounddevice callback invocation.
        bufferDurationSecs:  How many seconds of audio the circular buffer holds.
    """

    def __init__(
        self,
        deviceIndex: int | None = None,
        sampleRate: int = 44100,
        chunkFrames: int = 4096,
        bufferDurationSecs: float = 60.0,
    ):
        self.deviceIndex = deviceIndex
        self.sampleRate = sampleRate
        self.chunkFrames = chunkFrames
        self.bufferDurationSecs = bufferDurationSecs

        bufferSamples = int(sampleRate * bufferDurationSecs)

        # Circular buffer: int16 PCM samples (mono → 1-D array for simplicity)
        self._buffer = np.zeros(bufferSamples, dtype=np.int16)
        self._bufferIndex = 0          # next write position
        self._bufferTimestamps: list[tuple[int, float]] = []
        # Each entry is (bufferIndex at write time, Unix timestamp).
        # Used to compute accurate start time for saved recordings.

        self._lock = threading.Lock()
        self._running = False
        self._stream: sd.InputStream | None = None

        # Stream health tracking — mirrors RemoteAudioStream's
        # _lastChunkTime/_streamReadyTime, driven by sounddevice callbacks
        # instead of TCP chunk arrivals.
        self._lastCallbackTime: float | None = None
        self._streamReadyTime: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the input stream (non-blocking — sounddevice runs its own thread)."""
        self._printDeviceInfo()
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sampleRate,
            channels=1,
            dtype="int16",
            device=self.deviceIndex,
            blocksize=self.chunkFrames,
            callback=self._audioCallback,
        )
        self._stream.start()
        print(f"[localStream] Capturing {self.sampleRate} Hz mono")

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        with self._lock:
            self._lastCallbackTime = None
            self._streamReadyTime = None
        print("[localStream] Stopped.")

    def isConnected(self) -> bool:
        """True once the input stream is open and actively capturing."""
        return self._stream is not None and self._stream.active

    def isStreamHealthy(self, requiredDurationSecs: float = 0.0) -> bool:
        """
        Return True if audio callbacks are actively arriving and the buffer
        contains at least `requiredDurationSecs` seconds of real audio.

        A callback must have fired within the last 3 seconds (≈30× the
        nominal 93 ms chunk interval at 44100 Hz / 4096 frames — the same
        threshold RemoteAudioStream uses for TCP chunk arrival). The
        underlying PortAudio stream must also still be active, which
        catches a hard device error (e.g. the USB mic being unplugged)
        even if a `status` flag never fires.
        """
        if self._stream is None or not self._stream.active:
            return False
        with self._lock:
            lastCallback = self._lastCallbackTime
            streamStart = self._streamReadyTime
        if lastCallback is None or time.time() - lastCallback > 3.0:
            return False
        if requiredDurationSecs > 0 and streamStart is not None:
            return time.time() - streamStart >= requiredDurationSecs
        return True

    def getBuffer(self, durationSecs: float) -> np.ndarray:
        """
        Return the last `durationSecs` seconds of audio as a float32 array
        in the range [-1, 1], shaped (N, 1) to match soundfile's write format.
        """
        nSamples = min(int(durationSecs * self.sampleRate), len(self._buffer))

        with self._lock:
            idx = self._bufferIndex
            buf = self._buffer

            if idx >= nSamples:
                chunk = buf[idx - nSamples:idx].copy()
            else:
                first = buf[len(buf) - (nSamples - idx):]
                second = buf[:idx]
                chunk = np.concatenate([first, second])

        return (chunk.astype(np.float32) / 32768.0).reshape(-1, 1)

    def getBufferStartTime(self, durationSecs: float) -> float:
        """
        Return the Unix timestamp corresponding to the start of the last
        `durationSecs` window. Falls back to time.time() - durationSecs if
        no timestamp data is available yet.
        """
        nSamples = int(durationSecs * self.sampleRate)
        bufLen = len(self._buffer)
        targetIdx = (self._bufferIndex - nSamples) % bufLen

        with self._lock:
            stamps = self._bufferTimestamps

        if not stamps:
            return time.time() - durationSecs

        # Circular distance so the nearest stamp is found correctly even when
        # targetIdx and stamp positions straddle the buffer's wraparound point.
        def _circDist(pos: int) -> int:
            d = abs(pos - targetIdx)
            return min(d, bufLen - d)

        best = min(stamps, key=lambda t: _circDist(t[0]))

        # Signed sample offset: positive → target is later than best in time;
        # negative → target is earlier (further back in the buffer).
        rawOffset = (targetIdx - best[0]) % bufLen
        if rawOffset > bufLen // 2:
            rawOffset -= bufLen

        return best[1] + rawOffset / self.sampleRate

    def getClockSkewSecs(self) -> float | None:
        """
        Always 0.0 — audio capture and ADS-B polling run in the same
        process on the same clock here, so there is no skew to estimate
        (unlike RemoteAudioStream, which corrects for TCP-network jitter
        between two separate machines).
        """
        return 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audioCallback(self, indata: np.ndarray, frames: int, timeInfo, status) -> None:
        """sounddevice callback — called from the audio thread."""
        if status:
            print(f"[localStream] audio status: {status}")

        timestamp = time.time()
        with self._lock:
            self._lastCallbackTime = timestamp
            if self._streamReadyTime is None:
                self._streamReadyTime = timestamp

        self._writeSamples(indata[:, 0].copy(), timestamp)

    def _writeSamples(self, samples: np.ndarray, timestamp: float) -> None:
        nSamples = len(samples)
        bufLen = len(self._buffer)

        with self._lock:
            writeIdx = self._bufferIndex
            # Record timestamp → buffer position mapping (keep last 1000 entries).
            self._bufferTimestamps.append((writeIdx, timestamp))
            if len(self._bufferTimestamps) > 1000:
                self._bufferTimestamps = self._bufferTimestamps[-500:]

            end = writeIdx + nSamples
            if end <= bufLen:
                self._buffer[writeIdx:end] = samples
            else:
                firstPart = bufLen - writeIdx
                self._buffer[writeIdx:] = samples[:firstPart]
                self._buffer[:nSamples - firstPart] = samples[firstPart:]

            self._bufferIndex = end % bufLen

    def _printDeviceInfo(self) -> None:
        if self.deviceIndex is not None:
            d = sd.query_devices()[self.deviceIndex]
            print(f"[localStream] Device {self.deviceIndex}: {d['name']}")
        else:
            d = sd.query_devices(kind="input")
            print(f"[localStream] Default input device: {d['name']}")

#!/usr/bin/env python3
"""
Main-machine TCP audio receiver.

Accepts a connection from piCapture.py, receives PCM chunks with timestamps,
and maintains a circular buffer that recorder.py reads from.

Wire protocol (matches piCapture.py):
    [ 8 bytes: float64 Unix timestamp, big-endian ]
    [ 4 bytes: uint32  chunk byte length, big-endian ]
    [ N bytes: raw PCM S16LE mono ]
"""

import socket
import struct
import threading
import time
import numpy as np


HEADER_FMT = ">dI"        # float64 timestamp + uint32 length
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# How long to log a gap before filling it with silence.
GAP_WARN_SECS = 2.0


class RemoteAudioStream:
    """
    Listens on a TCP port for an audio stream from a Pi running piCapture.py.
    Maintains a circular buffer of recent audio, accessible via getBuffer().

    The interface mirrors AudioRecorder from the original aircraftRecorder.py
    so that recorder.py can swap between them without further changes.

    Args:
        port:               TCP port to listen on.
        sampleRate:         Expected sample rate of the incoming stream (Hz).
        bufferDurationSecs: How many seconds of audio the circular buffer holds.
    """

    def __init__(
        self,
        port: int = 9876,
        sampleRate: int = 44100,
        bufferDurationSecs: float = 60.0,
    ):
        self.port = port
        self.sampleRate = sampleRate
        self.bufferDurationSecs = bufferDurationSecs

        bufferSamples = int(sampleRate * bufferDurationSecs)

        # Circular buffer: int16 PCM samples (mono → 1-D array for simplicity)
        self._buffer = np.zeros(bufferSamples, dtype=np.int16)
        self._bufferIndex = 0          # next write position
        self._bufferTimestamps: list[tuple[int, float]] = []
        # Each entry is (bufferIndex at write time, Pi-side Unix timestamp)
        # Used to compute accurate start time for saved recordings.

        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._serverSock: socket.socket | None = None
        self._recvThread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the TCP listener (non-blocking — runs in background thread)."""
        self._running = True
        self._serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._serverSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._serverSock.bind(("", self.port))
        self._serverSock.listen(1)
        self._serverSock.settimeout(1.0)

        self._recvThread = threading.Thread(target=self._acceptLoop, daemon=True)
        self._recvThread.start()
        print(f"[remoteStream] Listening on port {self.port}")

    def stop(self) -> None:
        self._running = False
        if self._serverSock:
            self._serverSock.close()
        print("[remoteStream] Stopped.")

    def isConnected(self) -> bool:
        return self._connected

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

        # Convert int16 → float32 normalised, add channel dim
        return (chunk.astype(np.float32) / 32768.0).reshape(-1, 1)

    def getBufferStartTime(self, durationSecs: float) -> float:
        """
        Return the Pi-side Unix timestamp corresponding to the start of the
        last `durationSecs` window.  Falls back to time.time() - durationSecs
        if no timestamp data is available yet.
        """
        nSamples = int(durationSecs * self.sampleRate)
        targetIdx = (self._bufferIndex - nSamples) % len(self._buffer)

        with self._lock:
            stamps = self._bufferTimestamps

        if not stamps:
            return time.time() - durationSecs

        # Find the timestamp entry whose buffer position is closest to targetIdx.
        best = min(stamps, key=lambda t: abs(t[0] - targetIdx))
        # Adjust for the sample offset between best and targetIdx.
        sampleOffset = (targetIdx - best[0]) % len(self._buffer)
        return best[1] + sampleOffset / self.sampleRate

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _acceptLoop(self) -> None:
        """Accept connections forever; hand each off to _receiveLoop."""
        while self._running:
            try:
                clientSock, addr = self._serverSock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            print(f"[remoteStream] Pi connected from {addr[0]}:{addr[1]}")
            self._connected = True
            self._receiveLoop(clientSock)
            self._connected = False
            print("[remoteStream] Pi disconnected — waiting for reconnect")

    def _receiveLoop(self, sock: socket.socket) -> None:
        """Receive chunks from one connected Pi and write them into the buffer."""
        sock.settimeout(5.0)
        lastTimestamp: float | None = None

        try:
            while self._running:
                header = self._recvExact(sock, HEADER_SIZE)
                if header is None:
                    break

                timestamp, byteLen = struct.unpack(HEADER_FMT, header)

                # Detect gaps (Pi reconnected or clock jump).
                if lastTimestamp is not None:
                    gapSecs = timestamp - (lastTimestamp + byteLen / 2 / self.sampleRate)
                    if gapSecs > GAP_WARN_SECS:
                        silenceSamples = int(gapSecs * self.sampleRate)
                        print(
                            f"[remoteStream] Gap of {gapSecs:.1f}s detected — "
                            f"inserting {silenceSamples} silence samples"
                        )
                        self._writeSamples(np.zeros(silenceSamples, dtype=np.int16), timestamp)

                pcmBytes = self._recvExact(sock, byteLen)
                if pcmBytes is None:
                    break

                samples = np.frombuffer(pcmBytes, dtype=np.int16)
                self._writeSamples(samples, timestamp)
                lastTimestamp = timestamp

        except (OSError, struct.error):
            pass
        finally:
            sock.close()

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

    @staticmethod
    def _recvExact(sock: socket.socket, nBytes: int) -> bytes | None:
        """Read exactly nBytes from sock; return None on disconnect."""
        data = b""
        while len(data) < nBytes:
            try:
                chunk = sock.recv(nBytes - len(data))
            except socket.timeout:
                return None
            if not chunk:
                return None
            data += chunk
        return data

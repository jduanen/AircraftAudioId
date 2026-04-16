#!/usr/bin/env python3
"""
Pi Zero W audio capture and streaming client.

Captures audio from a USB microphone via sounddevice and streams raw PCM chunks
over a persistent TCP connection to the main recording machine.

Wire protocol per chunk:
    [ 8 bytes: float64 Unix timestamp, big-endian ]
    [ 4 bytes: uint32  chunk byte length, big-endian ]
    [ N bytes: raw PCM S16LE mono ]

Run on the Pi Zero W:
    python piCapture.py --host <main-machine-ip> --port 9876 [--device 1]
"""

import struct
import time
import socket
import threading
import queue
import sys
import argparse
import numpy as np
import sounddevice as sd


# Seconds to wait before retrying a failed connection.
RECONNECT_DELAY_SECS = 5

# Outbound send queue max depth — if the network can't keep up, old chunks are dropped.
QUEUE_MAX = 64


def _checkNtpOffset() -> None:
    """Warn if NTP clock offset looks large (best-effort, requires ntplib)."""
    try:
        import ntplib
        c = ntplib.NTPClient()
        resp = c.request("pool.ntp.org", version=3)
        offsetMs = abs(resp.offset) * 1000
        if offsetMs > 100:
            print(
                f"[piCapture] WARNING: NTP offset is {offsetMs:.0f} ms. "
                "Audio timestamps may not align well with ADS-B data. "
                "Run: sudo timedatectl set-ntp true"
            )
        else:
            print(f"[piCapture] NTP offset: {offsetMs:.1f} ms — OK")
    except ImportError:
        print("[piCapture] ntplib not installed — skipping NTP check (pip install ntplib)")
    except Exception as e:
        print(f"[piCapture] NTP check failed: {e}")


class PiCapture:
    """
    Captures audio from a USB mic and streams it over TCP.

    Args:
        host:           IP or hostname of the main recording machine.
        port:           TCP port to connect to.
        deviceIndex:    sounddevice input device index (None = system default).
        sampleRate:     Capture sample rate in Hz.
        chunkFrames:    Number of audio frames per chunk sent over the wire.
    """

    def __init__(
        self,
        host: str,
        port: int,
        deviceIndex: int | None = None,
        sampleRate: int = 44100,
        chunkFrames: int = 4096,
    ):
        self.host = host
        self.port = port
        self.deviceIndex = deviceIndex
        self.sampleRate = sampleRate
        self.chunkFrames = chunkFrames

        self._sendQueue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._running = False
        self._stream = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start capture and streaming (blocks until KeyboardInterrupt or stop())."""
        _checkNtpOffset()
        self._printDeviceInfo()

        self._running = True

        # Network sender runs in a background thread so audio capture never blocks.
        senderThread = threading.Thread(target=self._senderLoop, daemon=True)
        senderThread.start()

        self._stream = sd.InputStream(
            samplerate=self.sampleRate,
            channels=1,
            dtype="int16",
            device=self.deviceIndex,
            blocksize=self.chunkFrames,
            callback=self._audioCallback,
        )
        self._stream.start()

        print(
            f"[piCapture] Streaming {self.sampleRate} Hz mono → {self.host}:{self.port}"
            f"  (Ctrl+C to stop)"
        )

        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        print("[piCapture] Stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audioCallback(self, indata: np.ndarray, frames: int, timeInfo, status) -> None:
        """sounddevice callback — called from the audio thread."""
        if status:
            print(f"[piCapture] audio status: {status}", file=sys.stderr)

        timestamp = time.time()
        # Copy to avoid holding a reference to sounddevice's buffer.
        chunk = indata.copy()

        try:
            self._sendQueue.put_nowait((timestamp, chunk))
        except queue.Full:
            # Network thread is behind; drop oldest and insert newest.
            try:
                self._sendQueue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._sendQueue.put_nowait((timestamp, chunk))
            except queue.Full:
                pass

    def _senderLoop(self) -> None:
        """Background thread: maintain TCP connection and drain the send queue."""
        while self._running:
            sock = self._connect()
            if sock is None:
                continue
            print(f"[piCapture] Connected to {self.host}:{self.port}")
            try:
                while self._running:
                    try:
                        timestamp, chunk = self._sendQueue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    pcmBytes = chunk.astype(np.int16).tobytes()
                    header = struct.pack(">dI", timestamp, len(pcmBytes))
                    sock.sendall(header + pcmBytes)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"[piCapture] Connection lost: {e} — reconnecting in {RECONNECT_DELAY_SECS}s")
            finally:
                sock.close()

    def _connect(self) -> socket.socket | None:
        """Attempt to connect; returns socket on success, None on failure (with sleep)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            return sock
        except (ConnectionRefusedError, OSError) as e:
            print(
                f"[piCapture] Cannot connect to {self.host}:{self.port}: {e}"
                f" — retrying in {RECONNECT_DELAY_SECS}s"
            )
            time.sleep(RECONNECT_DELAY_SECS)
            return None

    def _printDeviceInfo(self) -> None:
        devices = sd.query_devices()
        if self.deviceIndex is not None:
            d = devices[self.deviceIndex]
            print(f"[piCapture] Device {self.deviceIndex}: {d['name']}")
        else:
            d = sd.query_devices(kind="input")
            print(f"[piCapture] Default input device: {d['name']}")


# ------------------------------------------------------------------
# CLI entry point (also used by scripts/capture.py)
# ------------------------------------------------------------------

def buildArgParser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stream USB mic audio over TCP to the main recorder.")
    p.add_argument("--host", required=True, help="Main machine IP or hostname")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--device", type=int, default=None, dest="deviceIndex",
                   help="sounddevice input device index (omit for system default)")
    p.add_argument("--sampleRate", type=int, default=44100)
    p.add_argument("--chunkFrames", type=int, default=4096)
    return p


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    PiCapture(
        host=args.host,
        port=args.port,
        deviceIndex=args.deviceIndex,
        sampleRate=args.sampleRate,
        chunkFrames=args.chunkFrames,
    ).start()

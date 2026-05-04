# record

Runs on the **Ubuntu recording server**. Combines ADS-B telemetry from `readsb`
with the audio stream from the Pi to detect flyovers and save synchronized
recordings to disk.

## Modules

**`recorder.py`** — `AircraftRecordingSystem`  
Top-level orchestrator. Polls ADS-B data via `ReadsbClient`, buffers audio via
`RemoteAudioStream`, and saves a WAV + JSON metadata file when a flyover is
detected. Triggers a save when the aircraft starts moving away or when the
45-second duration cap is reached.

Output layout:
```
<outputDir>/
    audio/<recordingId>.wav
    metadata/<recordingId>.json
    session_<timestamp>.json
```

**`aircraftType.py`** — `AircraftDatabase`  
Looks up an aircraft type string (e.g. `"B738"`) from an ICAO24 hex code.

### adsb/

**`adsb/__init__.py`** — `AircraftState` dataclass  
Holds one ADS-B snapshot: position, altitude, speed, heading, distance from
observer, bearing, and `capturedAt` (server-side Unix timestamp of the poll).

**`adsb/readsb.py`** — `ReadsbClient`  
Polls a `readsb` or `dump1090` JSON endpoint and returns `AircraftState`
objects filtered by radius and altitude. Skips stale entries older than 30
seconds.

### audioStream/

**`audioStream/remoteStream.py`** — `RemoteAudioStream`  
TCP server that receives PCM chunks from `piCapture.py` and writes them into a
60-second circular buffer. Key methods:

- `getBuffer(secs)` — returns the most recent audio as float32
- `getBufferStartTime(secs)` — returns the Pi-side Unix timestamp of the first
  sample in the returned window, used to anchor the WAV to wall time for ADS-B
  alignment. Uses signed circular-buffer arithmetic so the result is correct
  whether or not the nearest timestamp entry straddles the buffer's wraparound
  point.
- `isStreamHealthy(requiredDurationSecs)` — returns `True` only when a real PCM
  chunk has arrived within the last 3 seconds **and** the stream has been running
  long enough for the circular buffer to contain `requiredDurationSecs` worth of
  Pi-originating audio (not the initial silence). Both `_lastChunkTime` and
  `_streamReadyTime` are reset to `None` on Pi disconnect so stale buffer data is
  never treated as valid.

`recorder.py` calls `isStreamHealthy(durationSecs)` before writing any file;
recordings where the Pi is not streaming are silently discarded with a `[skip]`
log line.

## Entry point

```bash
python scripts/record.py --lat <lat> --lon <lon> \
    --radiusKm 8 --outputDir ./recordings \
    --readsbUrl http://adsbrx.lan/tar1090/data/aircraft.json
```

## Dependencies

`soundfile`, `numpy`, `requests`, `geopy`.

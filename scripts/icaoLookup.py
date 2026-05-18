#!/usr/bin/env python3
"""
List unique ICAO24 hex codes seen in recorded metadata.

Usage:
    python scripts/icaoLookup.py --recordingsDir ./recordings
    python scripts/icaoLookup.py --recordingsDir ./recordings --counts --tracks
    python scripts/icaoLookup.py --recordingsDir ./recordings --counts --tracks \\
        --faa --faaDatabaseDir ./data/ReleasableAircraft --sortBy samples
    python scripts/icaoLookup.py --recordingsDir ./recordings --faa \\
        --faaDatabaseDir ./data/ReleasableAircraft \\
        --fields nNumber,manufacturer,model,typeAcft,typeEng,noEngines,noSeats
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


_TYPE_ACFT = {
    "1": "Glider", "2": "Balloon", "3": "Blimp",
    "4": "FW-Single", "5": "FW-Multi", "6": "Rotorcraft",
    "7": "Weight-Shift", "8": "Pwrd-Chute", "9": "Gyroplane",
}
_TYPE_ENG = {
    "0": "None", "1": "Recip", "2": "Turboprop", "3": "Turboshaft",
    "4": "Turbojet", "5": "Turbofan", "6": "Ramjet",
    "7": "2-Cycle", "8": "4-Cycle", "9": "Unknown",
    "10": "Electric", "11": "Rotary",
}

_FAA_FIELDS = ["nNumber", "manufacturer", "model", "category", "typeAcft", "typeEng", "noEngines", "noSeats"]
_FAA_HEADERS = {
    "nNumber":     "N-NUMBER",
    "manufacturer":"MANUFACTURER",
    "model":       "MODEL",
    "category":    "CATEGORY",
    "typeAcft":    "ACFT-TYPE",
    "typeEng":     "ENG-TYPE",
    "noEngines":   "ENGINES",
    "noSeats":     "SEATS",
}


def _countTracks(times: list, intervalSecs: float) -> int:
    if not times:
        return 0
    sortedTimes = sorted(times)
    tracks = 1
    for a, b in zip(sortedTimes, sortedTimes[1:]):
        if b - a > intervalSecs:
            tracks += 1
    return tracks


def main():
    p = argparse.ArgumentParser(description="List unique ICAO24 hex codes from recorded metadata.")
    p.add_argument("--recordingsDir",  required=True, type=Path,
                   help="Root recordings directory (contains metadata/ subdirectory)")
    p.add_argument("--counts",         action="store_true",
                   help="Show raw sample count (number of ADS-B states) per code")
    p.add_argument("--tracks",         action="store_true",
                   help="Show per-track sighting count")
    p.add_argument("--trackInterval",  type=float, default=1.0,
                   help="Minimum hours between recordings to count as a new track (default: 1.0)")
    p.add_argument("--faa",            action="store_true",
                   help="Show FAA registration info (requires --faaDatabaseDir)")
    p.add_argument("--faaDatabaseDir", type=Path, default=None,
                   help="Path to unzipped FAA ReleasableAircraft directory")
    p.add_argument("--fields",         type=str, default="nNumber,manufacturer,model,category",
                   help="Comma-separated FAA fields to show (default: nNumber,manufacturer,model,category). "
                        f"Available: {', '.join(_FAA_FIELDS)}")
    p.add_argument("--sortBy",         choices=["icao24", "samples", "tracks", "callsign"],
                   default="icao24",
                   help="Sort output by this column (default: icao24)")
    args = p.parse_args()

    if args.faa and args.faaDatabaseDir is None:
        sys.exit("--faaDatabaseDir is required when --faa is set")

    faaDb = None
    if args.faa:
        from aircraftAudio.dataset.faaDatabase import FaaDatabase
        faaDb = FaaDatabase(args.faaDatabaseDir)

    metaDir = args.recordingsDir / "metadata"
    if not metaDir.exists():
        sys.exit(f"No metadata directory found at {metaDir}")

    metaPaths = sorted(metaDir.glob("*.json"))
    if not metaPaths:
        sys.exit(f"No metadata files found in {metaDir}")

    # Per icao24: callsign vote counts, sample count, recording start times (epoch)
    callsignVotes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sampleCounts:  dict[str, int]            = defaultdict(int)
    recTimes:      dict[str, list[float]]    = defaultdict(list)

    for metaPath in metaPaths:
        with open(metaPath) as f:
            meta = json.load(f)

        startEpoch = meta.get("audioStartTime")
        if startEpoch is None:
            from datetime import datetime, timezone
            try:
                dt = datetime.fromisoformat(meta["startTime"])
                startEpoch = dt.replace(tzinfo=timezone.utc).timestamp()
            except (KeyError, ValueError):
                startEpoch = 0.0

        seenInFile: set[str] = set()
        for state in meta.get("aircraftStates", []):
            icao = state.get("icao24", "").strip().lower()
            if not icao:
                continue
            sampleCounts[icao] += 1
            callsign = (state.get("callsign") or "").strip()
            if callsign:
                callsignVotes[icao][callsign] += 1
            seenInFile.add(icao)

        for icao in seenInFile:
            recTimes[icao].append(startEpoch)

    if not sampleCounts:
        sys.exit("No ICAO24 codes found in metadata.")

    intervalSecs = args.trackInterval * 3600

    rows = []
    for icao in sorted(sampleCounts):
        topCallsign = (
            max(callsignVotes[icao], key=callsignVotes[icao].get)
            if callsignVotes[icao] else ""
        )
        row = {
            "icao24":   icao,
            "callsign": topCallsign,
            "samples":  sampleCounts[icao],
            "tracks":   _countTracks(recTimes[icao], intervalSecs),
        }
        if faaDb:
            info = faaDb.infoForIcao24(icao) or {}
            for field in _FAA_FIELDS:
                val = info.get(field, "")
                if field == "typeAcft":
                    val = _TYPE_ACFT.get(str(val), str(val))
                elif field == "typeEng":
                    val = _TYPE_ENG.get(str(val), str(val))
                row[field] = str(val) if val else ""
        rows.append(row)

    if args.sortBy == "samples":
        rows.sort(key=lambda r: -r["samples"])
    elif args.sortBy == "tracks":
        rows.sort(key=lambda r: -r["tracks"])
    elif args.sortBy == "callsign":
        rows.sort(key=lambda r: r["callsign"])
    # else icao24: already sorted

    cols = ["icao24", "callsign"]
    if args.counts:
        cols.append("samples")
    if args.tracks:
        cols.append("tracks")
    if faaDb:
        requested = [f.strip() for f in args.fields.split(",")]
        cols.extend(f for f in requested if f in _FAA_FIELDS)

    headers = {
        "icao24":   "ICAO24",
        "callsign": "CALLSIGN",
        "samples":  "SAMPLES",
        "tracks":   f"TRACKS(>{args.trackInterval}h)",
        **_FAA_HEADERS,
    }

    widths = {c: len(headers[c]) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    print("  ".join(headers[c].ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))

    print(f"\n{len(rows)} unique ICAO24 code(s) across {len(metaPaths)} recording(s).")


if __name__ == "__main__":
    main()

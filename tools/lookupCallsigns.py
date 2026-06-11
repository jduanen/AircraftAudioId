#!/usr/bin/env python3
"""Look up airline names for a list of callsigns using the airline codes CSV."""

import csv
import re
import sys


def load_icao_lookup(codes_csv):
    lookup = {}
    with open(codes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            icao = row["ICAO"].strip()
            if icao:
                lookup[icao] = row["Airline"].strip()
    return lookup


def callsign_prefix(callsign):
    """Return the leading alpha characters (ICAO airline designator)."""
    m = re.match(r"^([A-Z]+)", callsign.strip().upper())
    return m.group(1) if m else ""


def main(callsigns_file, codes_csv, output_csv):
    lookup = load_icao_lookup(codes_csv)

    with open(callsigns_file, encoding="utf-8") as f:
        callsigns = [line.strip() for line in f if line.strip()]

    matched = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["callsign", "airline"])
        for cs in callsigns:
            prefix = callsign_prefix(cs)
            airline = lookup.get(prefix, "")
            if airline:
                matched += 1
            writer.writerow([cs, airline])

    print(f"Processed {len(callsigns)} callsigns, {matched} matched ({len(callsigns) - matched} unmatched)")  # noqa: E226
    print(f"Written to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <callsigns.txt> <airline_codes.csv> <output.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

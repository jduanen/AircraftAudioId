"""
FAA Releasable Aircraft Registration database lookup.

Builds an ICAO24 hex → category mapping from the two key files in the FAA
ReleasableAircraft download:
    MASTER.txt   — one row per registered N-number; includes MODE S CODE HEX
    ACFTREF.txt  — aircraft type reference; includes TYPE-ACFT and TYPE-ENG codes

Category derivation:
    TYPE-ACFT codes: 4=Fixed Wing Single Engine, 5=Fixed Wing Multi Engine,
                     6=Rotorcraft, 7=Weight-Shift, 1=Glider, 2=Balloon, 9=Gyroplane
    TYPE-ENG  codes: 1=Reciprocating, 2=Turbo-prop, 3=Turbo-shaft,
                     4=Turbo-jet, 5=Turbo-fan

    For multi-engine jets, NO-SEATS is used to separate:
        <20 seats  → business_jet
        20–100     → regional_jet
        101–220    → narrowbody_jet
        >220       → widebody_jet

Usage:
    db = FaaDatabase("path/to/ReleasableAircraft/")
    category = db.categoryForIcao24("a1fcc5")   # → "narrowbody_jet"
    info = db.infoForIcao24("a1fcc5")           # → full dict
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from .typeCategories import typeToCategory

log = logging.getLogger(__name__)


class FaaDatabase:
    """
    Loads the FAA releasable aircraft database from a directory containing
    MASTER.txt and ACFTREF.txt and provides ICAO24-based lookups.

    Args:
        dataDir: Path to the directory produced by unzipping ReleasableAircraft.zip.
    """

    def __init__(self, dataDir: str | Path):
        self._dataDir = Path(dataDir)
        self._icaoToInfo: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def categoryForIcao24(
        self,
        icao24: str,
        aircraftType: str | None = None,
    ) -> str:
        """
        Return the coarse audio category for the given ICAO24 hex address.

        Lookup order:
            1. FAA database structural derivation (TYPE-ACFT + TYPE-ENG + NO-SEATS).
            2. FAA model-string heuristic (typeCategories.typeToCategory on FAA model).
            3. Caller-supplied aircraftType string (e.g. from OpenSky / metadata JSON).
            4. "unknown".

        The FAA database only covers US-registered (N-number) aircraft.  For
        foreign registrations pass aircraftType so the keyword heuristic fires.
        """
        info = self._icaoToInfo.get(icao24.strip().lower())

        if info is not None:
            cat = _deriveCategory(info)
            if cat != "unknown":
                return cat
            cat = typeToCategory(info.get("model"))
            if cat != "unknown":
                return cat

        # Foreign aircraft or FAA lookup inconclusive — try caller's type string
        return typeToCategory(aircraftType)

    def infoForIcao24(self, icao24: str) -> Optional[dict]:
        """
        Return the full registration record for the given ICAO24 address,
        or None if not found.  Keys: icao24, nNumber, mfrMdlCode, manufacturer,
        model, typeAcft, typeEng, noEngines, noSeats, category.
        """
        return self._icaoToInfo.get(icao24.strip().lower())

    def __len__(self) -> int:
        return len(self._icaoToInfo)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        acftRefPath = self._dataDir / "ACFTREF.txt"
        masterPath  = self._dataDir / "MASTER.txt"

        if not acftRefPath.exists() or not masterPath.exists():
            raise FileNotFoundError(
                f"ACFTREF.txt and MASTER.txt must both be present in {self._dataDir}"
            )

        # Build mfrMdlCode → ACFTREF row
        acftRef: dict[str, dict] = {}
        with open(acftRefPath, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                code = row["CODE"].strip()
                acftRef[code] = {
                    "manufacturer": row["MFR"].strip(),
                    "model":        row["MODEL"].strip(),
                    "typeAcft":     row["TYPE-ACFT"].strip(),
                    "typeEng":      row["TYPE-ENG"].strip(),
                    "noEngines":    _intOrZero(row.get("NO-ENG", "0")),
                    "noSeats":      _intOrZero(row.get("NO-SEATS", "0")),
                }

        # Build icao24 → combined record
        loaded = 0
        with open(masterPath, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                hex_raw = row.get("MODE S CODE HEX", "").strip()
                if not hex_raw:
                    continue
                icao24 = hex_raw.lower()
                code   = row.get("MFR MDL CODE", "").strip()
                ref    = acftRef.get(code, {})
                entry  = {
                    "icao24":      icao24,
                    "nNumber":     row.get("N-NUMBER", "").strip(),
                    "mfrMdlCode":  code,
                    **ref,
                }
                entry["category"] = _deriveCategory(entry) or typeToCategory(ref.get("model"))
                self._icaoToInfo[icao24] = entry
                loaded += 1

        log.info("FaaDatabase loaded %d registrations from %s", loaded, self._dataDir)


# ------------------------------------------------------------------
# Category derivation from FAA type codes
# ------------------------------------------------------------------

def _deriveCategory(info: dict) -> str:
    typeAcft = info.get("typeAcft", "")
    typeEng  = info.get("typeEng", "")
    noSeats  = info.get("noSeats", 0)

    # Rotorcraft
    if typeAcft == "6":
        return "helicopter"

    # Gliders, balloons, powered parachutes, weight-shift, gyroplanes
    if typeAcft in ("1", "2", "3", "7", "8", "9"):
        return "unknown"

    # Fixed-wing single engine
    if typeAcft == "4":
        if typeEng == "1":
            return "piston_single"
        if typeEng in ("2", "3"):
            return "turboprop"
        if typeEng in ("4", "5"):
            return "business_jet"   # single-engine jet (VLJ or unusual)
        return "unknown"

    # Fixed-wing multi engine
    if typeAcft == "5":
        if typeEng == "1":
            return "piston_twin"
        if typeEng in ("2", "3"):
            return "turboprop"
        if typeEng in ("4", "5"):
            # Distinguish jet sub-types by seat count
            if noSeats == 0:
                return "unknown"
            if noSeats < 20:
                return "business_jet"
            if noSeats <= 100:
                return "regional_jet"
            if noSeats <= 220:
                return "narrowbody_jet"
            return "widebody_jet"
        return "unknown"

    return "unknown"


def _intOrZero(s: str) -> int:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return 0

#!/usr/bin/env python3
"""
Aircraft type lookup via the OpenSky Network metadata API.
Results are cached in memory for the duration of the session.
"""

import requests
from typing import Optional


METADATA_URL = "https://opensky-network.org/api/metadata/aircraft/icao/{icao24}"


class AircraftDatabase:
    """Look up aircraft model/type from ICAO24 hex address."""

    def __init__(self):
        self._cache: dict[str, str] = {}
        self._session = requests.Session()

    def getAircraftType(self, icao24: str) -> Optional[str]:
        """
        Return the aircraft model string for the given ICAO24 address,
        or None if the lookup fails or the aircraft is not in the database.
        """
        key = icao24.lower().strip()

        if key in self._cache:
            return self._cache[key]

        url = METADATA_URL.format(icao24=key)
        try:
            resp = self._session.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                model = data.get("model") or data.get("typecode") or None
                self._cache[key] = model
                return model
        except Exception:
            pass

        # Cache the miss so we don't hammer the API on repeated lookups.
        self._cache[key] = None
        return None

from dataclasses import dataclass
from typing import Optional


@dataclass
class AircraftState:
    """Aircraft state derived from ADS-B telemetry."""
    icao24: str
    callsign: Optional[str]
    latitude: float
    longitude: float
    altitudeFt: float
    velocityKts: float
    headingDeg: float
    distanceKm: float
    bearingDeg: float
    seenSecs: float       # seconds since last message from readsb

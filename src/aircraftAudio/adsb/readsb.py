#!/usr/bin/env python3
"""
readsb ADS-B client.
Polls the readsb/dump1090 JSON endpoint and returns nearby aircraft as AircraftState objects.
"""

import math
import time
import requests
from typing import Optional
from geopy.distance import geodesic

from . import AircraftState


DEFAULT_URL = "http://adsbrx.lan/data/aircraft.json"

# Entries with seen_pos older than this are considered stale and skipped.
MAX_SEEN_POS_SECS = 30


def _calculateBearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees (0–360) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


class ReadsbClient:
    """
    Polls a readsb (or dump1090) JSON endpoint and returns aircraft within a radius.

    Usage:
        client = ReadsbClient(observerLat=37.5, observerLon=-122.3)
        aircraft = client.getAircraft(radiusKm=20)
    """

    def __init__(
        self,
        observerLat: float,
        observerLon: float,
        url: str = DEFAULT_URL,
        pollIntervalSecs: float = 1.0,
        maxSeenPosSecs: float = MAX_SEEN_POS_SECS,
    ):
        self.observerLat = observerLat
        self.observerLon = observerLon
        self.url = url
        self.pollIntervalSecs = pollIntervalSecs
        self.maxSeenPosSecs = maxSeenPosSecs
        self._session = requests.Session()
        self._lastPoll: float = 0.0

    def getAircraft(
        self,
        radiusKm: float = 20.0,
        minAltitudeFt: float = 500.0,
        excludeOnGround: bool = True,
    ) -> list[AircraftState]:
        """
        Fetch current aircraft states from readsb, filtered by radius and altitude.
        Respects pollIntervalSecs — will sleep if called faster than that.
        """
        elapsed = time.time() - self._lastPoll
        if elapsed < self.pollIntervalSecs:
            time.sleep(self.pollIntervalSecs - elapsed)

        try:
            resp = self._session.get(self.url, timeout=5)
            resp.raise_for_status()
            self._lastPoll = time.time()
            data = resp.json()
        except Exception as e:
            print(f"[readsb] fetch error: {e}")
            return []

        results = []
        for entry in data.get("aircraft", []):
            lat = entry.get("lat")
            lon = entry.get("lon")
            if lat is None or lon is None:
                continue

            seenPos = entry.get("seen_pos", 999)
            if seenPos > self.maxSeenPosSecs:
                continue

            # readsb reports altitude in feet, speed in knots
            altitudeFt = entry.get("alt_geom", 0) or 0
            if excludeOnGround and altitudeFt < minAltitudeFt:
                continue

            distanceKm = geodesic(
                (self.observerLat, self.observerLon), (lat, lon)
            ).km
            if distanceKm > radiusKm:
                continue

            results.append(AircraftState(
                icao24=entry.get("hex", "").strip(),
                callsign=(entry.get("flight") or "").strip() or None,
                latitude=lat,
                longitude=lon,
                altitudeFt=altitudeFt,
                velocityKts=entry.get("speed", 0) or 0,
                headingDeg=entry.get("track", 0) or 0,
                distanceKm=distanceKm,
                bearingDeg=_calculateBearing(
                    self.observerLat, self.observerLon, lat, lon
                ),
                seenSecs=entry.get("seen", 0) or 0,
                capturedAt=self._lastPoll,
            ))

        return results

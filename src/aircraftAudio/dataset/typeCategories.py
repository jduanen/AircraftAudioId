"""
Maps raw FAA/ADS-B aircraft type strings to coarse audio-relevant categories.

Categories (ordered by typical acoustic signature):
    piston_single   — single-engine piston (Cessna 172, Cirrus SR, Piper PA-24 …)
    piston_twin     — twin-engine piston (Piper Seneca, Beech Baron …)
    turboprop       — turboprop, single or twin (PC-12, TBM, King Air …)
    helicopter      — rotary-wing, any power plant
    business_jet    — light–large private/charter jets (Citation, Gulfstream, Phenom 300 …)
    regional_jet    — small commercial jets up to ~100 seats (CRJ, ERJ-145/175 …)
    narrowbody_jet  — mainline single-aisle jets (737, A320 family, A220 …)
    widebody_jet    — twin-aisle jets (777, 787, A330, A350 …)
    unknown         — fallback when type is missing or unrecognised

Usage:
    from aircraftAudio.dataset.typeCategories import typeToCategory
    category = typeToCategory("172S")   # → "piston_single"
    category = typeToCategory(None)     # → "unknown"
"""

from __future__ import annotations


CATEGORIES = [
    "piston_single",
    "piston_twin",
    "turboprop",
    "helicopter",
    "business_jet",
    "regional_jet",
    "narrowbody_jet",
    "widebody_jet",
    "unknown",
]

# ──────────────────────────────────────────────────────────────────────────────
# Explicit lookup for every observed type string (stripped / lower-cased key)
# ──────────────────────────────────────────────────────────────────────────────

_EXPLICIT: dict[str, str] = {
    # ── Piston singles ────────────────────────────────────────────────────────
    "150g":                         "piston_single",
    "150j":                         "piston_single",
    "150l":                         "piston_single",
    "152":                          "piston_single",
    "172r":                         "piston_single",
    "172s":                         "piston_single",
    "cessna 172s skyhawk sp":       "piston_single",
    "182p":                         "piston_single",
    "182t":                         "piston_single",
    "7eca":                         "piston_single",   # Citabria
    "8kcab":                        "piston_single",   # Decathlon
    "a36":                          "piston_single",   # Bonanza
    "beech b36tc bonanza":          "piston_single",
    "g36":                          "piston_single",   # Bonanza G36
    "aa-5b":                        "piston_single",   # Grumman Tiger
    "de havilland canada dhc-2 beaver": "piston_single",
    "edge 540":                     "piston_single",   # aerobatic
    "pa-24":                        "piston_single",   # Comanche
    "pa-24-260":                    "piston_single",
    "pa-32r-301t":                  "piston_single",   # Saratoga
    "sr20":                         "piston_single",
    "sr22t":                        "piston_single",

    # ── Turboprops ────────────────────────────────────────────────────────────
    "pa 46-350p":                   "turboprop",       # Piper Malibu Meridian
    "pc-12/47e":                    "turboprop",
    "epic e1000":                   "turboprop",       # Epic turboprop single

    # ── Business jets ─────────────────────────────────────────────────────────
    "390":                          "business_jet",    # Raytheon Premier I
    "bd-100-1a10":                  "business_jet",    # Challenger 300
    "bd-500-1a10":                  "business_jet",    # Challenger 350
    "c750":                         "business_jet",    # Citation X
    "emb-505":                      "business_jet",    # Phenom 300
    "giv-x (g450)":                 "business_jet",

    # ── Regional jets ─────────────────────────────────────────────────────────
    "cl-600-2b19":                  "regional_jet",    # CRJ-200
    "cl-600-2c10":                  "regional_jet",    # CRJ-700
    "emb-135lr":                    "regional_jet",    # ERJ-145
    "erj 170-200 ll":               "regional_jet",    # E175
    "erj 170-200 lr":               "regional_jet",

    # ── Narrowbody jets ───────────────────────────────────────────────────────
    "airbus a220-300":              "narrowbody_jet",
    "737-7h4":                      "narrowbody_jet",
    "737-800":                      "narrowbody_jet",
    "737-824":                      "narrowbody_jet",
    "737-890":                      "narrowbody_jet",
    "737-8h4":                      "narrowbody_jet",
    "737-8kn":                      "narrowbody_jet",
    "737-9":                        "narrowbody_jet",
    "737-900er":                    "narrowbody_jet",
    "737-924er":                    "narrowbody_jet",
    "boeing 737 max 8":             "narrowbody_jet",
    "757-2q8":                      "narrowbody_jet",
    "a319-114":                     "narrowbody_jet",
    "a319-131":                     "narrowbody_jet",
    "a319-132":                     "narrowbody_jet",
    "a320-232":                     "narrowbody_jet",
    "a320-251n":                    "narrowbody_jet",
    "a321-211":                     "narrowbody_jet",
    "a321-271n":                    "narrowbody_jet",
    "airbus a321-271nx":            "narrowbody_jet",

    # ── Widebody jets ─────────────────────────────────────────────────────────
    "777-222":                      "widebody_jet",
    "b777-300(er)":                 "widebody_jet",
    "b789":                         "widebody_jet",    # 787-9
    "airbus a330-941":              "widebody_jet",

    # ── Helicopters ───────────────────────────────────────────────────────────
    "mbb-bk 117 c-2":               "helicopter",      # Airbus H145

    # ── Additional piston singles (from observed dataset) ─────────────────────
    "pa-28-181":                    "piston_single",   # Piper Archer
    "pa-28-161":                    "piston_single",   # Piper Warrior
    "pa-28rt-201t":                 "piston_single",   # Piper Arrow T
    "pa-46-310p":                   "piston_single",   # Piper Malibu (piston)
    "m20k":                         "piston_single",   # Mooney M20K
    "t210l":                        "piston_single",   # Cessna 210 Turbo
    "t206h":                        "piston_single",   # Cessna 206 Turbo
    "210":                          "piston_single",   # Cessna 210
    "162":                          "piston_single",   # Cessna 162 Skycatcher
    "f33a":                         "piston_single",   # Bonanza F33A
    "da 40 ng":                     "piston_single",   # Diamond DA40
    "vans rv-7":                    "piston_single",
    "vans rv-6":                    "piston_single",
    "rv-7":                         "piston_single",
    "rv-7a":                        "piston_single",
    "rv-8":                         "piston_single",
    "rv-12is":                      "piston_single",
    "sling tsi":                    "piston_single",   # Sling Aircraft TSI
    "beaver rx 550 plus":           "piston_single",   # ultralight
    "a-24b dauntless":              "piston_single",   # warbird
    "alpha trainer":                "piston_single",   # Pipistrel Alpha
    "114":                          "piston_single",   # Beechcraft/Piper variant
    "a-1":                          "piston_single",

    # ── Piston twins ─────────────────────────────────────────────────────────
    "c340":                         "piston_twin",     # Cessna 340
    "58":                           "piston_twin",     # Beechcraft Baron 58

    # ── Additional turboprops ─────────────────────────────────────────────────
    "700":                          "turboprop",       # TBM 700
    "e1000":                        "turboprop",       # Epic E1000 (alternate form)

    # ── Additional business jets ───────────────────────────────────────────────
    "505":                          "business_jet",    # EMB-505 Phenom 300
    "525a":                         "business_jet",    # Citation CJ2
    "525b":                         "business_jet",    # Citation CJ3
    "560":                          "business_jet",    # Citation V/Ultra/Encore
    "bd-700-1a10":                  "business_jet",    # Global Express
    "gv-sp (g550)":                 "business_jet",    # Gulfstream G550
    "g-iv (g400)":                  "business_jet",    # Gulfstream G400

    # ── ICAO type codes for common jets ───────────────────────────────────────
    "b38m":                         "narrowbody_jet",  # 737 MAX 8
    "b39m":                         "narrowbody_jet",  # 737 MAX 9
    "b3xm":                         "narrowbody_jet",  # 737 MAX 10
    "b77l":                         "widebody_jet",    # 777-200LR
    "b77w":                         "widebody_jet",    # 777-300ER
    "b788":                         "widebody_jet",    # 787-8
    "b789":                         "widebody_jet",    # 787-9 (duplicate, harmless)
    "b78x":                         "widebody_jet",    # 787-10
    "a20n":                         "narrowbody_jet",  # A320neo
    "a21n":                         "narrowbody_jet",  # A321neo
    "a19n":                         "narrowbody_jet",  # A319neo
    "a225":                         "widebody_jet",    # A220-300 (CS300)
    "a333":                         "widebody_jet",    # A330-300
    "a339":                         "widebody_jet",    # A330-900neo
    "a359":                         "widebody_jet",    # A350-900
    "a388":                         "widebody_jet",    # A380-800
    "e75l":                         "regional_jet",    # E175
    "e75s":                         "regional_jet",    # E175
    "e170":                         "regional_jet",
    "e190":                         "regional_jet",
    "e195":                         "regional_jet",
    "crj2":                         "regional_jet",    # CRJ-200
    "crj7":                         "regional_jet",    # CRJ-700
    "crj9":                         "regional_jet",    # CRJ-900
    "crjx":                         "regional_jet",    # CRJ-1000
}


# ──────────────────────────────────────────────────────────────────────────────
# Keyword heuristic for unknown types (applied in order; first match wins)
# ──────────────────────────────────────────────────────────────────────────────

_KEYWORD_RULES: list[tuple[list[str], str]] = [
    # Widebody first (high specificity)
    (["a380", "a350", "a340", "a330", "a300",
      "787", "777", "767", "747",
      "b787", "b777", "b767", "b747",
      "widebody", "wide body"], "widebody_jet"),

    # Narrowbody
    (["a220", "a319", "a320", "a321",
      "737", "757", "717", "md-80", "md-90",
      "narrowbody", "narrow body"], "narrowbody_jet"),

    # Regional jets
    (["crj", "cl-600", "cl600",
      "erj", "embraer 1", "e170", "e175", "e190", "e195",
      "atr 42", "atr 72",
      "regional jet"], "regional_jet"),

    # Business jets
    (["citation", "gulfstream", "learjet", "lear ",
      "global express", "global 6", "global 7",
      "falcon", "hawker", "premier",
      "phenom 3", "phenom 1",
      "challenger", "legacy",
      "honda jet", "hondajet",
      "very light jet", "vlj"], "business_jet"),

    # Helicopters
    (["helicopter", "heli", " h12", " h13", " h14", " h15", " h16",
      "r22", "r44", "r66", "robinson",
      "bell 2", "bell 4", "bell 5", "bell 4", "uh-", "ah-",
      "sikorsky", "agusta", "eurocopter",
      "ec120", "ec130", "ec135", "ec145", "ec155",
      "as350", "as355", "as365",
      "md 500", "md500",
      "rotorcraft", "gyroplane"], "helicopter"),

    # Turboprops
    (["king air", "kingair",
      "tbm ", "tbm-",
      "pc-12", "pc-6",
      "caravan", "c208",
      "pilatus",
      "twin otter", "dhc-6",
      "dash 8", "dhc-8", "q400",
      "atr", "saab 340", "sf340",
      "conquest", "cheyenne",
      "meridian", "malibu meridian",
      "epic e1000",
      "piper pa-46-500", "pa-46-500"], "turboprop"),

    # Piston singles last (very broad fallback keywords)
    (["cessna", "172", "182", "152", "150",
      "cirrus", "sr20", "sr22",
      "bonanza", "musketeer",
      "piper", "cherokee", "archer", "warrior", "comanche",
      "mooney", "grumman aa",
      "citabria", "decathlon",
      "diamond da", "da20", "da40",
      "vans rv", "rv-", "sling",
      "american champion", "pipistrel", "alpha trainer",
      "warbird", "dauntless", "stearman"], "piston_single"),
]


def typeToCategory(aircraftType: str | None) -> str:
    """
    Map a raw aircraft type string to a coarse audio category.

    Lookup order:
        1. Exact match in the explicit table (case-insensitive, stripped).
        2. Keyword scan of the explicit table (each keyword is a substring).
        3. Returns "unknown".

    Args:
        aircraftType: Raw type string from FAA/ADS-B metadata, or None.

    Returns:
        One of the strings in CATEGORIES.
    """
    if not aircraftType:
        return "unknown"

    key = aircraftType.strip().lower()

    # 1. Exact match
    if key in _EXPLICIT:
        return _EXPLICIT[key]

    # 2. Keyword scan
    for keywords, category in _KEYWORD_RULES:
        if any(kw in key for kw in keywords):
            return category

    return "unknown"

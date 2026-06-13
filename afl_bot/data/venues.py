"""
Static AFL venue metadata (plan §1.8).

Maps each venue name as it appears in the Squiggle fixtures to its city,
coordinates (for weather lookups, ``afl_bot.data.weather``) and whether it has a
closable roof. Marvel Stadium (a.k.a. Docklands) is the only AFL ground whose
roof is routinely shut, so rain never affects play there — every other ground
is open-air.

``venue_info`` returns ``None`` for an unknown venue so callers degrade
gracefully (no weather adjustment) rather than raising.
"""

from __future__ import annotations

# venue name -> (city, latitude, longitude, roofed)
# Coordinates are city/ground level — accurate enough for daily rainfall.
VENUE_METADATA: dict[str, dict] = {
    "M.C.G.":                 {"city": "Melbourne",     "lat": -37.820, "lon": 144.983, "roofed": False},
    "Marvel Stadium":         {"city": "Melbourne",     "lat": -37.816, "lon": 144.947, "roofed": True},
    "Docklands":              {"city": "Melbourne",     "lat": -37.816, "lon": 144.947, "roofed": True},
    "Adelaide Oval":          {"city": "Adelaide",      "lat": -34.915, "lon": 138.596, "roofed": False},
    "Norwood Oval":           {"city": "Adelaide",      "lat": -34.918, "lon": 138.633, "roofed": False},
    "Barossa Park":           {"city": "Lyndoch",       "lat": -34.604, "lon": 138.893, "roofed": False},
    "Adelaide Hills":         {"city": "Mount Barker",  "lat": -35.072, "lon": 138.859, "roofed": False},
    "Optus Stadium":          {"city": "Perth",         "lat": -31.951, "lon": 115.889, "roofed": False},
    "Perth Stadium":          {"city": "Perth",         "lat": -31.951, "lon": 115.889, "roofed": False},
    "Hands Oval":             {"city": "Bunbury",       "lat": -33.340, "lon": 115.640, "roofed": False},
    "Gabba":                  {"city": "Brisbane",      "lat": -27.486, "lon": 153.038, "roofed": False},
    "Carrara":                {"city": "Gold Coast",    "lat": -28.006, "lon": 153.366, "roofed": False},
    "Cazaly's Stadium":       {"city": "Cairns",        "lat": -16.936, "lon": 145.749, "roofed": False},
    "Riverway Stadium":       {"city": "Townsville",    "lat": -19.310, "lon": 146.740, "roofed": False},
    "S.C.G.":                 {"city": "Sydney",        "lat": -33.892, "lon": 151.225, "roofed": False},
    "Sydney Showground":      {"city": "Sydney",        "lat": -33.843, "lon": 151.067, "roofed": False},
    "Stadium Australia":      {"city": "Sydney",        "lat": -33.847, "lon": 151.063, "roofed": False},
    "Manuka Oval":            {"city": "Canberra",      "lat": -35.318, "lon": 149.135, "roofed": False},
    "UNSW Canberra Oval":     {"city": "Canberra",      "lat": -35.293, "lon": 149.162, "roofed": False},
    "GMHBA Stadium":          {"city": "Geelong",       "lat": -38.158, "lon": 144.354, "roofed": False},
    "Kardinia Park":          {"city": "Geelong",       "lat": -38.158, "lon": 144.354, "roofed": False},
    "Eureka Stadium":         {"city": "Ballarat",      "lat": -37.511, "lon": 143.830, "roofed": False},
    "Mars Stadium":           {"city": "Ballarat",      "lat": -37.511, "lon": 143.830, "roofed": False},
    "Bellerive Oval":         {"city": "Hobart",        "lat": -42.877, "lon": 147.373, "roofed": False},
    "York Park":              {"city": "Launceston",    "lat": -41.426, "lon": 147.139, "roofed": False},
    "University of Tasmania Stadium": {"city": "Launceston", "lat": -41.426, "lon": 147.139, "roofed": False},
    "Marrara Oval":           {"city": "Darwin",        "lat": -12.399, "lon": 130.887, "roofed": False},
    "Traeger Park":           {"city": "Alice Springs", "lat": -23.706, "lon": 133.876, "roofed": False},
    "Jiangwan Stadium":       {"city": "Shanghai",      "lat":  31.302, "lon": 121.501, "roofed": False},
    "Adelaide Arena at Jiangwan Stadium": {"city": "Shanghai", "lat": 31.302, "lon": 121.501, "roofed": False},
}


def venue_info(venue: str) -> dict | None:
    """Metadata dict for ``venue`` (city/lat/lon/roofed), or ``None`` if unknown."""
    return VENUE_METADATA.get(venue)


def is_roofed(venue: str) -> bool:
    """True if the venue has a roof that's routinely closed (rain irrelevant).
    Unknown venues are treated as open-air (False)."""
    info = VENUE_METADATA.get(venue)
    return bool(info and info["roofed"])

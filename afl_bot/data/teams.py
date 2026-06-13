"""
Canonical team ID mapping (plan §5.1, build-order step 1).

Different sources spell team names differently — AFL Tables uses historical
names ("Footscray", "Kangaroos", "South Melbourne"), Footywire/DFS Australia
add nicknames or abbreviations ("GWS Giants", "Sydney Swans", "ADE"), and
Fryzigg/fitzRoy often uses 3-letter codes. Joining across sources on raw team
names is a silent-bug factory.

This module fixes one canonical name per team — the names already returned by
the Squiggle API, since that's the source the rest of the pipeline is built
around — and provides a lookup table + normaliser so every new data loader can
map its own spelling onto the canonical name before the data hits the rest of
the pipeline.
"""

from __future__ import annotations

# Canonical names, as returned by the Squiggle API (`get_teams` / `get_games`).
CANONICAL_TEAMS: list[str] = [
    "Adelaide",
    "Brisbane Lions",
    "Carlton",
    "Collingwood",
    "Essendon",
    "Fremantle",
    "Geelong",
    "Gold Coast",
    "Greater Western Sydney",
    "Hawthorn",
    "Melbourne",
    "North Melbourne",
    "Port Adelaide",
    "Richmond",
    "St Kilda",
    "Sydney",
    "West Coast",
    "Western Bulldogs",
]

# Standard 3-letter codes (used by Fryzigg/fitzRoy, DFS Australia, Champion Data).
CANONICAL_TO_CODE: dict[str, str] = {
    "Adelaide": "ADE",
    "Brisbane Lions": "BRL",
    "Carlton": "CAR",
    "Collingwood": "COL",
    "Essendon": "ESS",
    "Fremantle": "FRE",
    "Geelong": "GEE",
    "Gold Coast": "GCS",
    "Greater Western Sydney": "GWS",
    "Hawthorn": "HAW",
    "Melbourne": "MEL",
    "North Melbourne": "NTH",
    "Port Adelaide": "POR",
    "Richmond": "RIC",
    "St Kilda": "STK",
    "Sydney": "SYD",
    "West Coast": "WCE",
    "Western Bulldogs": "WBD",
}

CODE_TO_CANONICAL: dict[str, str] = {code: name for name, code in CANONICAL_TO_CODE.items()}

# Aliases seen across AFL Tables, Footywire, DFS Australia, fitzRoy/Fryzigg and
# common abbreviations, including historical names for long-horizon AFL Tables
# data. Keys are matched case-insensitively after stripping whitespace.
_ALIASES: dict[str, str] = {
    # Adelaide
    "adelaide crows": "Adelaide",
    "adel": "Adelaide",
    # Brisbane Lions (post-1997 merger of Fitzroy + Brisbane Bears)
    "brisbane": "Brisbane Lions",
    "lions": "Brisbane Lions",
    "brisbane bears": "Brisbane Lions",
    "fitzroy": "Brisbane Lions",
    # Carlton
    "carlton blues": "Carlton",
    "blues": "Carlton",
    # Collingwood
    "collingwood magpies": "Collingwood",
    "magpies": "Collingwood",
    # Essendon
    "essendon bombers": "Essendon",
    "bombers": "Essendon",
    # Fremantle
    "fremantle dockers": "Fremantle",
    "dockers": "Fremantle",
    # Geelong
    "geelong cats": "Geelong",
    "cats": "Geelong",
    # Gold Coast
    "gold coast suns": "Gold Coast",
    "suns": "Gold Coast",
    "gcfc": "Gold Coast",
    # Greater Western Sydney
    "gws giants": "Greater Western Sydney",
    "greater western sydney giants": "Greater Western Sydney",
    "giants": "Greater Western Sydney",
    "gws": "Greater Western Sydney",
    # Hawthorn
    "hawthorn hawks": "Hawthorn",
    "hawks": "Hawthorn",
    # Melbourne
    "melbourne demons": "Melbourne",
    "demons": "Melbourne",
    # North Melbourne
    "north melbourne kangaroos": "North Melbourne",
    "kangaroos": "North Melbourne",
    "north": "North Melbourne",
    # Port Adelaide
    "port adelaide power": "Port Adelaide",
    "power": "Port Adelaide",
    "port": "Port Adelaide",
    "pta": "Port Adelaide",  # DFS Australia's code for Port Adelaide (vs. our POR)
    # Richmond
    "richmond tigers": "Richmond",
    "tigers": "Richmond",
    # St Kilda
    "st. kilda": "St Kilda",
    "saints": "St Kilda",
    "stkilda": "St Kilda",
    # Sydney (AFL Tables pre-1982 history as South Melbourne)
    "sydney swans": "Sydney",
    "swans": "Sydney",
    "south melbourne": "Sydney",
    # West Coast
    "west coast eagles": "West Coast",
    "eagles": "West Coast",
    # Western Bulldogs (AFL Tables pre-1997 history as Footscray)
    "footscray": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
    "western bulldogs football club": "Western Bulldogs",
}


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in CANONICAL_TEAMS:
        lookup[name.lower()] = name
    for code, name in CODE_TO_CANONICAL.items():
        lookup[code.lower()] = name
    for alias, name in _ALIASES.items():
        lookup[alias.lower()] = name
    return lookup


_LOOKUP = _build_lookup()


def normalize_team_name(name: str) -> str:
    """Map any known spelling/abbreviation/historical name to the canonical
    (Squiggle-style) team name. Raises ``KeyError`` for unrecognised names so
    that join bugs surface immediately rather than silently dropping rows."""
    key = name.strip().lower()
    try:
        return _LOOKUP[key]
    except KeyError:
        raise KeyError(f"Unrecognised team name: {name!r}") from None


def normalize_team_column(series):
    """Vectorised ``normalize_team_name`` for a pandas Series of team names."""
    return series.map(normalize_team_name)


def team_code(canonical_name: str) -> str:
    """Canonical name -> 3-letter code (e.g. 'Geelong' -> 'GEE')."""
    return CANONICAL_TO_CODE[canonical_name]

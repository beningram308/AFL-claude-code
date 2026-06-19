"""
Confirmed team lineups (Fable round-2 §1.2).

A priced player who isn't actually named in the team for a game makes every prop
on him worthless and crowds a real player out of the pool. The official AFL
lineup feed (aflapi.afl.com.au, what fitzRoy's ``fetch_lineup`` wraps) is
media-token-gated — the same wall the detailed stoppage stats hit — so the
reliable free path is the team sheets that drop ~Thursday evening, supplied as a
small JSON (the same hand-entered interchange pattern as ``--odds``):

    {"Geelong": ["Patrick Dangerfield", "Bailey Smith", ...],
     "Carlton": ["Patrick Cripps", ...]}

``load_lineup`` reads that file (keys starting with ``_`` are ignored, e.g. a
future ``_rules`` block) and normalises team names. ``run_round`` then marks any
priced player not named ``confirmed=False`` — the multi builder already excludes
those (``afl_bot.build.multi.usable_legs``). With no file, every player stays
confirmed (unchanged behaviour).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

from afl_bot.data.teams import normalize_team_name

FOOTYWIRE_SELECTIONS_URL = "https://www.footywire.com/afl/footy/afl_team_selections"
_USER_AGENT = "afl-multi-builder (personal use; contact via repo issues)"
_MIN_PLAYERS_FOR_CONFIRMED_SHEET = 18  # teams with fewer named players are treated as not-yet-posted


def load_lineup(path: str | None) -> dict[str, set[str]]:
    """Load ``{team: {player, ...}}`` from a confirmed-lineup JSON file, or
    return ``{}`` when no path is given. Accepts either plain name strings or
    per-player objects — ``{"player": "Name", ...}`` — so old files and new
    TOG-annotated files both work. Unknown team names are skipped."""
    if not path:
        return {}

    data = json.loads(Path(path).read_text())
    lineup: dict[str, set[str]] = {}
    for team, players in data.items():
        if team.startswith("_"):
            continue
        try:
            canonical = normalize_team_name(team)
        except KeyError:
            continue
        confirmed: set[str] = set()
        for entry in players:
            if isinstance(entry, str):
                confirmed.add(entry.strip())
            elif isinstance(entry, dict) and "player" in entry:
                confirmed.add(str(entry["player"]).strip())
        lineup[canonical] = confirmed
    return lineup


def _slug_to_name(slug: str) -> str:
    """Convert a Footywire player slug to a title-cased full name.
    e.g. ``brennan-cox`` → ``"Brennan Cox"``,
         ``judd-mcvee`` → ``"Judd McVee"`` (Mc prefix handled),
         ``luke-davies-uniacke`` → ``"Luke Davies Uniacke"`` (hyphens lost in slug).
    Downstream matching uses :func:`_normalize_name` to tolerate the missing hyphen."""
    parts = []
    for part in slug.split("-"):
        cap = part.capitalize()
        if cap.startswith("Mc") and len(cap) > 2:
            cap = "Mc" + cap[2:].capitalize()
        elif cap.startswith("Mac") and len(cap) > 3:
            cap = "Mac" + cap[3:].capitalize()
        parts.append(cap)
    return " ".join(parts)


def _parse_footywire_selections(html: str) -> dict[str, set[str]]:
    """Parse raw Footywire team-selections HTML into ``{canonical_team: {player_name}}``.
    Uses ``pp-{team-slug}--{player-slug}`` hrefs; teams with fewer than
    ``_MIN_PLAYERS_FOR_CONFIRMED_SHEET`` links are omitted (sheet not yet posted)."""
    from bs4 import BeautifulSoup  # optional dep; already installed per requirements
    soup = BeautifulSoup(html, "html.parser")
    raw: dict[str, set[str]] = {}
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if not href.startswith("pp-") or "--" not in href:
            continue
        # strip "pp-"; split on first "--" to get team-slug and player-slug
        body = href[3:]
        sep = body.index("--")
        team_slug = body[:sep]
        player_slug = body[sep + 2:]
        if not player_slug:
            continue
        try:
            team_name = normalize_team_name(team_slug.replace("-", " "))
        except KeyError:
            continue
        raw.setdefault(team_name, set()).add(_slug_to_name(player_slug))

    # Only include teams whose selection sheet looks complete
    return {t: p for t, p in raw.items() if len(p) >= _MIN_PLAYERS_FOR_CONFIRMED_SHEET}


def fetch_lineup(year: int, round_no: int, *,
                 cache_seconds: float = 3600.0,
                 cache_dir: Path | None = None) -> dict[str, set[str]]:
    """Fetch current AFL team selections from Footywire and return
    ``{canonical_team: {player_name, ...}}``.

    Results are cached for ``cache_seconds`` (default 1 h) so repeated calls
    within the same session don't hammer the site.  Returns ``{}`` on any
    network or parse failure — the caller must handle a missing lineup gracefully.

    Team sheets typically post Thursday night for the upcoming weekend.  Before
    that, most teams will have fewer than ``_MIN_PLAYERS_FOR_CONFIRMED_SHEET``
    named players and will be omitted from the result; the caller can detect this
    by checking whether the returned dict is smaller than expected."""
    from afl_bot.config import CACHE_DIR as _default_cache
    if cache_dir is None:
        cache_dir = _default_cache

    cache_path = Path(cache_dir) / f"lineup_{year}_r{round_no}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_seconds:
        try:
            data = json.loads(cache_path.read_text())
            return {team: set(players) for team, players in data.items()}
        except (json.JSONDecodeError, OSError):
            pass

    try:
        resp = requests.get(FOOTYWIRE_SELECTIONS_URL,
                            headers={"User-Agent": _USER_AGENT}, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"Footywire lineup fetch failed ({exc}); no auto-lineup applied.", file=sys.stderr)
        return {}

    lineup = _parse_footywire_selections(resp.text)
    if lineup:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({t: list(p) for t, p in lineup.items()}))
        except OSError:
            pass
    return lineup


def load_lineup_tog(path: str | None) -> dict[str, float]:
    """Parse per-player projected TOG overrides from a confirmed-lineup JSON.

    The richer per-player object form (mixed freely with plain strings):
        ``{"player": "Adam Cerra", "expected_tog": 0.70}``
        ``{"player": "Sam Walsh", "returning_from_injury": true}``
        ``{"player": "Charlie Curnow", "managed": true}``

    Returns ``{player_name: projected_tog}`` for players with an explicit
    ``expected_tog`` or a returning/managed flag (which applies
    ``TOG_RETURN_DEFAULT``). Plain-string entries and players without a flag are
    omitted. Returns ``{}`` when no path is given."""
    if not path:
        return {}

    from afl_bot.config import TOG_RETURN_DEFAULT  # avoid circular at module load

    data = json.loads(Path(path).read_text())
    tog_map: dict[str, float] = {}
    for team, players in data.items():
        if team.startswith("_"):
            continue
        for entry in players:
            if not isinstance(entry, dict):
                continue
            player_name = str(entry.get("player", "")).strip()
            if not player_name:
                continue
            if "expected_tog" in entry:
                tog_map[player_name] = float(entry["expected_tog"])
            elif entry.get("returning_from_injury") or entry.get("managed"):
                tog_map[player_name] = TOG_RETURN_DEFAULT
    return tog_map

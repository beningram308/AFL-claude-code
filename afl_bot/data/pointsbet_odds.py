"""
PointsBet player-props odds, scraped from the site's undocumented REST API
(modelled on afl_bot/data/sportsbet_odds.py).

PointsBet AU API base: https://api.au.pointsbet.com/api/v2/
AFL competition key  : 7523  (aussie-rules sport, key discovered from
  GET /api/v2/sports/aussie-rules/competitions)

Unlike Sportsbet, PointsBet's event/market endpoints require an authenticated
session (OAuth via https://auth.au.pointsbet.com).  Unauthenticated requests
return HTTP 204 No Content for event data and {"events":[]} for featured
lists -- not a geo-block, but deliberate auth gating.

This module therefore:
  1. Tries to fetch event and market data from the API.
  2. Falls back cleanly (returns {}) when auth is unavailable.
  3. Returns {leg_name: decimal_odds} in the same format as fetch_sportsbet_odds,
     keyed identically to what round_report builds legs in:
       "<player> <line>+ <stat>"  e.g. "Josh Ward 4+ marks"

When the file reports/<year>_r<N>_pointsbet_urls.json exists (a list of
PointsBet event URLs or numeric event keys), the scraper uses those IDs to
fetch event data directly.  Without that file it tries to discover events
from the AFL competition listing.

Because scraping currently requires auth, the recommended workflow is:
  1. Run round-report (generates reports/<year>_r<N>_pointsbet_odds.json
     with all candidate legs as null values).
  2. Fill in real PB prices from the app for lines PB actually offers.
  3. Re-run round-report --sportsbet to pick them up.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

from afl_bot.config import CACHE_DIR
from afl_bot.data.teams import CANONICAL_TEAMS, normalize_team_name

# ── API constants ──────────────────────────────────────────────────────────────
_API_BASE = "https://api.au.pointsbet.com/api/v2"
_SPORT_KEY = "aussie-rules"
_AFL_COMP_KEY = "7523"

_COMPETITIONS_URL = f"{_API_BASE}/sports/{_SPORT_KEY}/competitions"
_EVENTS_URL       = f"{_API_BASE}/events/{{event_key}}"
_MARKETS_URL      = f"{_API_BASE}/events/{{event_key}}/markets"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_CACHE_PREFIX = "pointsbet_event_"

# ── Market name → stat normalisation ──────────────────────────────────────────
_MILESTONE_RE = re.compile(
    r"^(\d+)\+\s*(Disposals?|Goals?|Marks?|Tackles?)$", re.I
)
_STAT_PLURAL = {
    "disposal": "disposals", "goal": "goals",
    "mark": "marks",         "tackle": "tackles",
}

# ── Pull 'Em eligible markets (same gate as build_pull_em_sgm) ─────────────────
_PULL_EM_MARKETS = {
    "player_disposals", "disposals",
    "player_marks",     "marks",
    "player_goals",     "goals",
    "player_tackles",   "tackles",
}


def _normalise_team(name: str) -> str | None:
    try:
        return normalize_team_name(name)
    except KeyError:
        for canon in CANONICAL_TEAMS:
            if name.upper().startswith(canon.upper()) or canon.upper() in name.upper():
                return canon
        return None


def _take_best(best: dict[str, float], leg: str, price) -> None:
    try:
        price = float(price)
    except (TypeError, ValueError):
        return
    if price > best.get(leg, 0.0):
        best[leg] = price


def _is_json(resp: requests.Response) -> bool:
    return "application/json" in resp.headers.get("content-type", "").lower()


def _headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
        "Accept-Language": "en-AU,en;q=0.9",
        "Referer": "https://pointsbet.com.au/",
    }


def parse_markets(markets: list[dict]) -> dict[str, float]:
    """Parse PointsBet market data into {leg_name: best_odds}.

    PointsBet market structure (from API v2):
      market["name"]       : e.g. "20+ Disposals", "4+ Marks", "Head to Head"
      market["selections"] : list of {name: player_name, price: {winPrice: float}}
    Same shape as Sportsbet's Markets endpoint -- parse_markets is
    intentionally symmetric.
    """
    best: dict[str, float] = {}
    for market in markets:
        name = str(market.get("name", "")).strip()
        sels = market.get("selections", [])

        if name == "Head to Head":
            for sel in sels:
                team = _normalise_team(str(sel.get("name", "")))
                if team:
                    price = (sel.get("price") or {}).get("winPrice")
                    _take_best(best, f"{team} to win", price)
            continue

        m = _MILESTONE_RE.match(name)
        if m:
            n, stat_word = m.groups()
            stat = _STAT_PLURAL[stat_word.rstrip("sS").lower()]
            for sel in sels:
                player = str(sel.get("name", "")).strip()
                if not player:
                    continue
                price = (sel.get("price") or {}).get("winPrice")
                _take_best(best, f"{player} {n}+ {stat}", price)
            continue

    return best


def _discover_afl_event_keys(*, timeout: float = 15.0) -> list[str]:
    """Try to discover current AFL event keys from the competition listing.

    Requires auth -- returns [] when the API gates event data (HTTP 204 / empty
    events array).  Documented here so the fallback is transparent.
    """
    try:
        r = requests.get(
            f"{_API_BASE}/sports/{_SPORT_KEY}/competitions/{_AFL_COMP_KEY}/events",
            headers=_headers(), timeout=timeout,
        )
        if r.status_code == 204 or not _is_json(r):
            return []
        r.raise_for_status()
        data = r.json()
        events = data if isinstance(data, list) else data.get("events", [])
        return [str(e.get("key") or e.get("eventKey") or e.get("id") or "")
                for e in events if e]
    except (requests.RequestException, ValueError):
        return []


def fetch_event_odds(
    event_key: str, *, cache_seconds: float = 120.0, cache_dir=CACHE_DIR
) -> dict[str, float]:
    """Odds for ONE PointsBet event, cached for cache_seconds.

    Returns {} when the API requires auth (HTTP 204 / non-JSON) or on any
    network/parse failure -- never raises.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{_CACHE_PREFIX}{event_key}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_seconds:
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    hdrs = _headers()
    try:
        r = requests.get(
            _MARKETS_URL.format(event_key=event_key),
            headers=hdrs, timeout=20,
        )
        if r.status_code == 204:
            # Auth required -- graceful no-op (not a geo-block)
            print(
                f"PointsBet event {event_key}: API returned 204 (auth required). "
                f"Fill in prices manually in the _pointsbet_odds.json template.",
                file=sys.stderr,
            )
            return {}
        if not _is_json(r):
            print(
                f"PointsBet event {event_key}: non-JSON response ({r.status_code}) -- skipping.",
                file=sys.stderr,
            )
            return {}
        r.raise_for_status()
        markets = r.json()
        if not isinstance(markets, list):
            markets = markets.get("markets", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"PointsBet event {event_key}: fetch failed ({exc}) -- skipping.", file=sys.stderr)
        return {}

    odds = parse_markets(markets)
    if odds:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            from afl_bot.io_utils import atomic_write_text
            atomic_write_text(cache_path, json.dumps(odds))
        except OSError:
            pass
    return odds


def fetch_pointsbet_odds(
    event_urls_or_keys: list[str] | None = None,
    *,
    cache_seconds: float = 120.0,
    cache_dir=CACHE_DIR,
) -> dict[str, float]:
    """PointsBet player-props odds for AFL events.

    ``event_urls_or_keys`` -- PointsBet event keys (or URLs containing them).
    When None or empty, falls back to ``_discover_afl_event_keys()`` which
    requires auth and will return {} if auth is unavailable.

    Returns {} on auth failure, network error, or parse error -- never raises.
    Symmetric API to ``fetch_sportsbet_odds``.
    """
    keys_to_try: list[str] = []

    if event_urls_or_keys:
        for entry in event_urls_or_keys:
            entry = str(entry).strip()
            if entry.isdigit():
                keys_to_try.append(entry)
            else:
                # Extract numeric key from URL like:
                # https://pointsbet.com.au/sports/aussie-rules/7523/hawthorn-v-melbourne-12345678
                m = re.search(r"(\d{6,10})/?$", entry.rstrip("/"))
                if m:
                    keys_to_try.append(m.group(1))
                else:
                    print(
                        f"PointsBet: couldn't extract event key from {entry!r} -- skipping.",
                        file=sys.stderr,
                    )
    else:
        keys_to_try = _discover_afl_event_keys()
        if not keys_to_try:
            print(
                "PointsBet: event discovery returned no keys (auth required). "
                "Fill in _pointsbet_odds.json manually.",
                file=sys.stderr,
            )
            return {}

    best: dict[str, float] = {}
    n_ok = 0
    for key in keys_to_try:
        odds = fetch_event_odds(key, cache_seconds=cache_seconds, cache_dir=cache_dir)
        if odds:
            n_ok += 1
        for leg, price in odds.items():
            _take_best(best, leg, price)

    print(
        f"PointsBet: {n_ok}/{len(keys_to_try)} event(s) priced, "
        f"{len(best)} leg(s) matched.",
        file=sys.stderr,
    )
    return best

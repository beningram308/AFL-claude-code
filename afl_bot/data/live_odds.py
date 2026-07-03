"""
Live AFL market odds (MULTI-CHANGES PART A).

Fetches current h2h + totals prices from The Odds API
(https://the-odds-api.com, sport key ``aussierules_afl``) and returns them in
EXACTLY the leg-name format ``afl_bot.cli.round_report`` already uses, so they
drop straight into ``odds_book``:

    "Brisbane Lions to win"      (h2h)
    "Total points 170.5+"        (totals, Over side)

The API key is read from the ``ODDS_API_KEY`` env var (never hard-coded). The
response is cached briefly (odds move) like the historical-odds client. Player
PROP odds are NOT on the free tier, so this returns h2h/totals only — the CLI
merges these with any ``--odds`` file for props (live ⊕ manual). On a missing
key / network failure it returns ``{}`` (the report then falls back to the
manual file), and never pretends props are live.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

from afl_bot.config import CACHE_DIR
from afl_bot.data.teams import CANONICAL_TEAMS, normalize_team_name

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds"
EVENTS_API_URL = "https://api.the-odds-api.com/v4/sports/aussierules_afl/events"
EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/aussierules_afl/events/{event_id}/odds"
USER_AGENT = "afl-multi-builder (https://github.com/; contact via repo issues; personal use)"
CACHE_NAME = "live_odds"
PROPS_CACHE_NAME = "live_props"

# Player-prop market keys -> pipeline stat name
_PROP_MARKET_TO_STAT: dict[str, str] = {
    "player_disposals_over": "disposals",
    "player_goals_scored_over": "goals",
    "player_marks_over": "marks",
    "player_tackles_over": "tackles",
}
_PROP_MARKETS = ",".join(_PROP_MARKET_TO_STAT)


def _normalise_team(name: str) -> str | None:
    """Map an Odds API team string to a canonical name. The feed appends
    mascots ("Geelong Cats", "West Coast Eagles"), so fall back to a
    canonical-name prefix/substring match when the alias table misses."""
    try:
        return normalize_team_name(name)
    except KeyError:
        for canon in CANONICAL_TEAMS:
            if name.startswith(canon) or canon in name:
                return canon
        return None


def _take_best(best: dict[str, float], leg: str, price) -> None:
    """Keep the best (highest) decimal price seen for a leg across bookmakers."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return
    if price > best.get(leg, 0.0):
        best[leg] = price


def parse_events(events: list[dict]) -> dict[str, float]:
    """Reshape The Odds API event list into ``{leg_name: best_decimal_odds}``
    for h2h and totals (Over), in the report's key format."""
    best: dict[str, float] = {}
    for ev in events:
        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                key = market.get("key")
                if key == "h2h":
                    for o in market.get("outcomes", []):
                        team = _normalise_team(o.get("name", ""))
                        if team is not None:
                            _take_best(best, f"{team} to win", o.get("price"))
                elif key == "totals":
                    for o in market.get("outcomes", []):
                        if str(o.get("name", "")).lower() == "over" and o.get("point") is not None:
                            _take_best(best, f"Total points {o['point']}+", o.get("price"))
    return best


def fetch_live_odds(round_no: int | None = None, *, api_key: str | None = None,
                    regions: str = "au", markets: str = "h2h,totals",
                    cache_seconds: float = 180, cache_dir=CACHE_DIR) -> dict[str, float]:
    """Live h2h + totals odds as ``{leg_name: decimal_odds}`` (best price across
    AU books). ``round_no`` is accepted for call-site symmetry but not used to
    filter (the feed returns upcoming games; legs join by name). Returns ``{}``
    if ``ODDS_API_KEY`` is unset or the call fails."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set; skipping live odds (use --odds for a manual file).",
              file=sys.stderr)
        return {}

    cache_path = cache_dir / f"{CACHE_NAME}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_seconds:
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    try:
        resp = requests.get(
            ODDS_API_URL,
            params={"apiKey": api_key, "regions": regions, "markets": markets,
                    "oddsFormat": "decimal"},
            headers={"User-Agent": USER_AGENT}, timeout=30,
        )
        resp.raise_for_status()
        events = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Live odds unavailable ({exc}); falling back to the --odds file.", file=sys.stderr)
        return {}

    odds = parse_events(events)
    cache_dir.mkdir(parents=True, exist_ok=True)
    from afl_bot.io_utils import atomic_write_text
    atomic_write_text(cache_path, json.dumps(odds))
    return odds


def parse_event_props(event_resp: dict) -> dict[str, float]:
    """Reshape a single per-event Odds API props response into
    ``{leg_name: best_decimal_odds}`` in the report's key format.

    Leg names follow the pattern ``"{player_name} {line}+ {stat}"``, where
    ``line`` is derived from the API's ``point`` (Over 24.5 → 25+)."""
    import math

    best: dict[str, float] = {}
    for bm in event_resp.get("bookmakers", []):
        for market in bm.get("markets", []):
            stat = _PROP_MARKET_TO_STAT.get(market.get("key", ""))
            if stat is None:
                continue
            for o in market.get("outcomes", []):
                if str(o.get("name", "")).lower() != "over":
                    continue
                player = str(o.get("description", "")).strip()
                point = o.get("point")
                if not player or point is None:
                    continue
                try:
                    point = float(point)
                except (TypeError, ValueError):
                    continue
                # "Over 24.5" → 25+ (integer threshold matching the pipeline's format)
                int_line = int(point) + 1 if point == int(point) else math.ceil(point)
                leg = f"{player} {int_line}+ {stat}"
                _take_best(best, leg, o.get("price"))
    return best


def fetch_live_props(round_no: int | None = None, *, api_key: str | None = None,
                     regions: str = "au", cache_seconds: float = 300,
                     cache_dir=CACHE_DIR) -> dict[str, float]:
    """Live player-prop odds (disposals/goals/marks/tackles) as
    ``{leg_name: decimal_odds}`` (best price across AU books).

    Requires an Odds API key with the **player props tier** (paid add-on).
    If the key is absent, the props tier is blocked (402/403), or the network
    fails, returns ``{}`` silently — Fix B (bettable-window gate + wider
    PROP_LINES) then handles the no-odds path.

    Fetches the events list first (one call), then one per-event call per
    upcoming AFL game.  Results are cached for ``cache_seconds`` (default 5 min)
    to avoid burning quota on repeated calls in a session."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        return {}

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{PROPS_CACHE_NAME}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_seconds:
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Step 1: get event IDs for upcoming AFL games
    try:
        ev_resp = requests.get(
            EVENTS_API_URL,
            params={"apiKey": api_key, "sport": "aussierules_afl"},
            headers={"User-Agent": USER_AGENT}, timeout=30,
        )
        ev_resp.raise_for_status()
        events = ev_resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Live props unavailable (events fetch: {exc}); using model prices.",
              file=sys.stderr)
        return {}

    # Step 2: per-event prop markets
    all_props: dict[str, float] = {}
    for ev in events:
        event_id = ev.get("id")
        if not event_id:
            continue
        try:
            pr = requests.get(
                EVENT_ODDS_URL.format(event_id=event_id),
                params={"apiKey": api_key, "regions": regions,
                        "markets": _PROP_MARKETS, "oddsFormat": "decimal"},
                headers={"User-Agent": USER_AGENT}, timeout=30,
            )
            if pr.status_code in (402, 403):
                # Props tier not enabled — don't spam errors for every event
                print("Live props require a paid Odds API props tier; "
                      "falling back to model prices.", file=sys.stderr)
                return {}
            pr.raise_for_status()
            props = parse_event_props(pr.json())
            for leg, price in props.items():
                _take_best(all_props, leg, price)
        except (requests.RequestException, ValueError):
            continue   # skip one failed event; keep the others

    if all_props:
        cache_dir.mkdir(parents=True, exist_ok=True)
        from afl_bot.io_utils import atomic_write_text
        atomic_write_text(cache_path, json.dumps(all_props))
    return all_props

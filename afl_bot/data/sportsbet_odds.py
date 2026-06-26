"""
Real Sportsbet odds, scraped from the site's own undocumented JSON API
(FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART A) -- no key, no login, no paid tier.

Sportsbet geo-blocks non-AU IPs (a blocked response isn't JSON), so this only
works run from an Australian IP -- gate it behind ``--sportsbet`` and fall
back cleanly (``{}``) on a block/network/parse failure, the same contract as
``afl_bot.data.live_odds.fetch_live_odds``. This is for personal use against
public, unauthenticated, read-only odds pages: cache aggressively (~2 min per
event) and fetch each market grouping once -- never hammer the site.

Reference endpoints (confirmed working, decimal odds, AFL):
    SportCard   (market groupings for an event):
        https://www.sportsbet.com.au/apigw/sportsbook-sports/Sportsbook/Sports/Events/{event_id}/SportCard
    Markets     (selections + prices within one grouping):
        https://www.sportsbet.com.au/apigw/sportsbook-sports/Sportsbook/Sports/Events/{event_id}/MarketGroupings/{group_id}/Markets

``fetch_sportsbet_odds(event_urls_or_ids)`` returns ``{leg_name: decimal_odds}``
in EXACTLY the key format ``afl_bot.cli.round_report`` already builds legs in
("<team> to win", "Total points <line>+", "<player> <line>+ <stat>"), so it
drops straight into ``odds_book`` alongside any ``--odds`` file.
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

SPORT_CARD_URL = (
    "https://www.sportsbet.com.au/apigw/sportsbook-sports/Sportsbook/Sports/Events/{event_id}/SportCard"
)
MARKETS_URL = (
    "https://www.sportsbet.com.au/apigw/sportsbook-sports/Sportsbook/Sports/Events/"
    "{event_id}/MarketGroupings/{group_id}/Markets"
)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
CACHE_NAME_PREFIX = "sportsbet_event_"

# Only fetch these groupings per event (PART A5: one request per grouping,
# don't fan out across all ~26 -- these five cover every market this bot
# prices: h2h, totals, and the player milestone markets).
_TARGET_GROUPING_NAMES = {
    "Top Markets", "Pick Your Own Disposals", "Pick Your Own Goals",
    "Player Marks", "Player Tackles",
}

# A milestone market is named e.g. "20+ Disposals", "1+ Goal", "2+ Goals",
# "4+ Marks", "3+ Tackles" -- selections are bare player names, no handicap.
_MILESTONE_RE = re.compile(r"^(\d+)\+\s*(Disposals?|Goals?|Marks?|Tackles?)$", re.I)
_STAT_TO_PLURAL = {
    "disposal": "disposals", "goal": "goals", "mark": "marks", "tackle": "tackles",
}

_EVENT_ID_RE = re.compile(r"(\d{6,9})/?\s*$")


def _normalise_team(name: str) -> str | None:
    """Map a Sportsbet team string ("GWS GIANTS", "West Coast Eagles") to a
    canonical name, same fallback pattern as ``live_odds._normalise_team``."""
    try:
        return normalize_team_name(name)
    except KeyError:
        for canon in CANONICAL_TEAMS:
            if name.upper().startswith(canon.upper()) or canon.upper() in name.upper():
                return canon
        return None


def _take_best(best: dict[str, float], leg: str, price) -> None:
    """Keep the best (highest) decimal price seen for a leg."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return
    if price > best.get(leg, 0.0):
        best[leg] = price


def _win_price(selection: dict):
    return (selection.get("price") or {}).get("winPrice")


def parse_markets(markets: list[dict]) -> dict[str, float]:
    """Reshape a list of Sportsbet ``Markets`` entries (h2h, totals, and the
    player milestone markets -- disposals/goals/marks/tackles) into
    ``{leg_name: best_decimal_odds}`` in the report's key format. Selections
    that don't map to a known leg shape are silently skipped (counted by the
    caller, not here -- this is the pure, side-effect-free parser)."""
    best: dict[str, float] = {}
    for market in markets:
        name = str(market.get("name", "")).strip()
        selections = market.get("selections", [])

        if name == "Head to Head":
            for sel in selections:
                team = _normalise_team(str(sel.get("name", "")))
                if team is not None:
                    _take_best(best, f"{team} to win", _win_price(sel))
            continue

        if name == "Total Game Points - Over/Under":
            for sel in selections:
                if str(sel.get("name", "")).strip().lower() != "over":
                    continue
                line = sel.get("displayHandicap") or sel.get("unformattedHandicap")
                if line is None:
                    continue
                line = str(line).lstrip("+")
                try:
                    line = float(line)
                except ValueError:
                    continue
                _take_best(best, f"Total points {line}+", _win_price(sel))
            continue

        m = _MILESTONE_RE.match(name)
        if m:
            n, stat_word = m.groups()
            stat = _STAT_TO_PLURAL[stat_word.rstrip("sS").lower()]
            for sel in selections:
                player = str(sel.get("name", "")).strip()
                if not player:
                    continue
                _take_best(best, f"{player} {n}+ {stat}", _win_price(sel))
            continue

    return best


def _extract_event_id(url_or_id) -> str | None:
    """Pull the 7-digit event ID off a Sportsbet match URL, or pass through a
    bare numeric ID/string."""
    s = str(url_or_id).strip()
    if s.isdigit():
        return s
    m = _EVENT_ID_RE.search(s.rstrip("/"))
    return m.group(1) if m else None


def _is_json_response(resp) -> bool:
    return "application/json" in resp.headers.get("content-type", "").lower()


def fetch_event_odds(event_id: str, *, cache_seconds: float = 120.0,
                     cache_dir=CACHE_DIR) -> dict[str, float]:
    """Real odds for ONE Sportsbet event, cached for ``cache_seconds``
    (default ~2 min). Returns ``{}`` on a geo-block (non-JSON response),
    network failure, or parse error -- never raises, never retries."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{CACHE_NAME_PREFIX}{event_id}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_seconds:
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    try:
        card_resp = requests.get(SPORT_CARD_URL.format(event_id=event_id),
                                 params={"displayWinnersPriceMkt": "true",
                                         "includeLiveMarketGroupings": "true",
                                         "includeCollection": "true"},
                                 headers=headers, timeout=20)
        if not _is_json_response(card_resp):
            print(f"Sportsbet event {event_id}: non-JSON response (geo-blocked? "
                  f"not an AU IP, or rate-limited) -- skipping.", file=sys.stderr)
            return {}
        card_resp.raise_for_status()
        groupings = card_resp.json().get("marketGrouping", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"Sportsbet event {event_id}: SportCard fetch failed ({exc}) -- skipping.",
              file=sys.stderr)
        return {}

    all_markets: list[dict] = []
    for g in groupings:
        if g.get("name") not in _TARGET_GROUPING_NAMES:
            continue
        try:
            resp = requests.get(
                MARKETS_URL.format(event_id=event_id, group_id=g["id"]),
                headers=headers, timeout=20)
            if not _is_json_response(resp):
                continue
            resp.raise_for_status()
            all_markets.extend(resp.json())
        except (requests.RequestException, ValueError):
            continue   # skip one failed grouping; keep the others

    odds = parse_markets(all_markets)
    if odds:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(odds))
        except OSError:
            pass
    return odds


def fetch_sportsbet_odds(event_urls_or_ids: list[str], *, cache_seconds: float = 120.0,
                         cache_dir=CACHE_DIR) -> dict[str, float]:
    """Real Sportsbet odds for every event in ``event_urls_or_ids`` (full
    match URLs or bare event IDs), merged into one ``{leg_name: decimal_odds}``
    dict (best price -- there's only one Sportsbet price per leg, but a leg
    can appear in more than one event's response, e.g. duplicated groupings).

    AU-IP only (Sportsbet geo-blocks everyone else); any event that fails
    (block/network/parse) contributes nothing and is skipped, never raises.
    Empty input returns ``{}``."""
    best: dict[str, float] = {}
    n_events_ok = 0
    for entry in event_urls_or_ids:
        event_id = _extract_event_id(entry)
        if event_id is None:
            print(f"Sportsbet: couldn't extract an event ID from {entry!r} -- skipping.",
                  file=sys.stderr)
            continue
        odds = fetch_event_odds(event_id, cache_seconds=cache_seconds, cache_dir=cache_dir)
        if odds:
            n_events_ok += 1
        for leg, price in odds.items():
            _take_best(best, leg, price)

    print(f"Sportsbet: {n_events_ok}/{len(event_urls_or_ids)} event(s) priced, "
          f"{len(best)} leg(s) matched.", file=sys.stderr)
    return best

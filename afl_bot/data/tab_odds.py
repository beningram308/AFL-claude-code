"""TAB Australia odds — second bookmaker for prop CLV consensus reference.

Public JSON endpoint (no auth, AU IP only):
    https://api.beta.tab.com.au/v1/tab-info-service/sports/Australian%20Rules/
        competitions/AFL/matches?jurisdiction=NSW

Auto-discovers all current AFL matches — no per-round URL list needed.
Returns {leg_name: decimal_odds} in the SAME format as sportsbet_odds.py so
it joins directly into the devig_consensus_single_sided call in capture_close.

Fails gracefully ({}) on geo-block / network / parse error — never raises.
Cache TTL matches sportsbet_odds.py (120 s).
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

TAB_AFL_URL = (
    "https://api.beta.tab.com.au/v1/tab-info-service/sports/"
    "Australian%20Rules/competitions/AFL/matches"
)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# "Player Disposals 25+", "Player Goals 2+", "Player Marks 4+", "Player Tackles 3+"
_PLAYER_MARKET_RE = re.compile(
    r"^Player\s+(Disposals?|Goals?|Marks?|Tackles?)\s+(\d+)\+$", re.I
)
# "To Get 25+ Disposals", "To Get 2+ Goals", etc.
_TO_GET_RE = re.compile(
    r"^To\s+Get\s+(\d+)\+\s*(Disposals?|Goals?|Marks?|Tackles?)$", re.I
)
# Proposition name contains threshold: "Over 170.5"
_OVER_LINE_RE = re.compile(r"^Over\s+([\d.]+)$", re.I)

_STAT_TO_PLURAL = {
    "disposal": "disposals", "goal": "goals", "mark": "marks", "tackle": "tackles",
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
        p = float(price)
    except (TypeError, ValueError):
        return
    if p > best.get(leg, 0.0):
        best[leg] = p


def _stat_plural(word: str) -> str:
    return _STAT_TO_PLURAL[word.rstrip("sS").lower()]


def parse_tab_markets(markets: list[dict]) -> tuple[dict[str, float], int, int]:
    """Parse a list of TAB market dicts (from one match) into {leg_name: decimal_odds}.

    Returns (odds_dict, n_matched, n_dropped).
    Silently skips Draw propositions and Under/Place lines (not bot legs).
    """
    best: dict[str, float] = {}
    n_matched = n_dropped = 0

    for market in markets:
        bet_option = str(market.get("betOption", "")).strip()
        propositions = market.get("propositions", [])

        if bet_option == "Match Odds":
            for prop in propositions:
                pname = str(prop.get("name", "")).strip()
                price = prop.get("returnWin")
                if pname.lower() == "draw":
                    continue
                team = _normalise_team(pname)
                if team is not None and price:
                    _take_best(best, f"{team} to win", price)
                    n_matched += 1
                else:
                    n_dropped += 1
            continue

        if "Total Game Points" in bet_option:
            for prop in propositions:
                pname = str(prop.get("name", "")).strip()
                price = prop.get("returnWin")
                m = _OVER_LINE_RE.match(pname)
                if m and price:
                    _take_best(best, f"Total points {m.group(1)}+", price)
                    n_matched += 1
                # Under lines are not bot legs — silently skip
            continue

        # "Player Disposals 25+" → selections are bare player names
        m = _PLAYER_MARKET_RE.match(bet_option)
        if m:
            stat_word, line = m.groups()
            stat = _stat_plural(stat_word)
            for prop in propositions:
                player = str(prop.get("name", "")).strip()
                price = prop.get("returnWin")
                if player and price:
                    _take_best(best, f"{player} {line}+ {stat}", price)
                    n_matched += 1
                else:
                    n_dropped += 1
            continue

        # "To Get 25+ Disposals" → selections are bare player names
        m = _TO_GET_RE.match(bet_option)
        if m:
            line, stat_word = m.groups()
            stat = _stat_plural(stat_word)
            for prop in propositions:
                player = str(prop.get("name", "")).strip()
                price = prop.get("returnWin")
                if player and price:
                    _take_best(best, f"{player} {line}+ {stat}", price)
                    n_matched += 1
                else:
                    n_dropped += 1
            continue

        # Unrecognised market — count all selections as dropped
        n_dropped += len(propositions)

    return best, n_matched, n_dropped


def _is_json(resp) -> bool:
    return "application/json" in resp.headers.get("content-type", "").lower()


def fetch_tab_odds(
    *,
    jurisdiction: str = "NSW",
    cache_seconds: float = 120.0,
    cache_dir=CACHE_DIR,
) -> dict[str, float]:
    """All current AFL odds from TAB (auto-discovers matches — no URL list needed).

    Returns {leg_name: decimal_odds}, or {} on geo-block / network / parse error.
    AU IP only — TAB geo-blocks non-AU addresses; errors are printed to stderr and
    the function returns {} (never raises).
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"tab_afl_{jurisdiction}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_seconds:
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    try:
        resp = requests.get(
            TAB_AFL_URL,
            params={"jurisdiction": jurisdiction},
            headers=headers,
            timeout=20,
        )
        if not _is_json(resp):
            print(
                "TAB: non-JSON response (geo-blocked or rate-limited — AU IP required) — skipping.",
                file=sys.stderr,
            )
            return {}
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"TAB: fetch failed ({exc}) — skipping.", file=sys.stderr)
        return {}

    all_odds: dict[str, float] = {}
    total_matched = total_dropped = 0

    for match in data.get("matches", []):
        match_odds, nm, nd = parse_tab_markets(match.get("markets", []))
        for leg, price in match_odds.items():
            _take_best(all_odds, leg, price)
        total_matched += nm
        total_dropped += nd

    n_matches = len(data.get("matches", []))
    print(
        f"TAB: {n_matches} match(es), {total_matched} leg(s) matched, "
        f"{total_dropped} selection(s) dropped.",
        file=sys.stderr,
    )

    if all_odds:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            from afl_bot.io_utils import atomic_write_text
            atomic_write_text(cache_path, json.dumps(all_odds))
        except OSError:
            pass

    return all_odds

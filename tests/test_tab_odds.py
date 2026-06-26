"""Tests for the TAB odds scraper (afl_bot/data/tab_odds.py)."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from afl_bot.data.tab_odds import fetch_tab_odds, parse_tab_markets


# ── fixture helpers ───────────────────────────────────────────────────────────

def _json_response(payload: dict) -> MagicMock:
    m = MagicMock()
    m.headers = {"content-type": "application/json"}
    m.json.return_value = payload
    m.raise_for_status.return_value = None
    return m


def _blocked_response() -> MagicMock:
    m = MagicMock()
    m.headers = {"content-type": "text/html; charset=utf-8"}
    return m


_MATCH_ODDS_MARKET = {
    "betOption": "Match Odds",
    "propositions": [
        {"name": "Hawthorn", "returnWin": 1.36},
        {"name": "GWS Giants", "returnWin": 3.25},
        {"name": "Draw", "returnWin": 34.0},
    ],
}

_TOTAL_POINTS_MARKET = {
    "betOption": "Total Game Points - Over/Under",
    "propositions": [
        {"name": "Over 170.5", "returnWin": 1.85},
        {"name": "Under 170.5", "returnWin": 1.90},
    ],
}

_PLAYER_DISPOSALS_25_MARKET = {
    "betOption": "Player Disposals 25+",
    "propositions": [
        {"name": "Jai Newcombe", "returnWin": 1.65},
        {"name": "James Worpel", "returnWin": 2.10},
    ],
}

_PLAYER_GOALS_2_MARKET = {
    "betOption": "Player Goals 2+",
    "propositions": [
        {"name": "Jack Ginnivan", "returnWin": 1.80},
    ],
}

_PLAYER_MARKS_4_MARKET = {
    "betOption": "Player Marks 4+",
    "propositions": [
        {"name": "James Sicily", "returnWin": 1.95},
    ],
}

_PLAYER_TACKLES_3_MARKET = {
    "betOption": "Player Tackles 3+",
    "propositions": [
        {"name": "Will Day", "returnWin": 1.50},
    ],
}

_TO_GET_DISPOSALS_MARKET = {
    "betOption": "To Get 25+ Disposals",
    "propositions": [
        {"name": "Jai Newcombe", "returnWin": 1.70},
    ],
}

_UNKNOWN_MARKET = {
    "betOption": "Brownlow Votes",
    "propositions": [
        {"name": "Player X", "returnWin": 3.50},
        {"name": "Player Y", "returnWin": 4.00},
    ],
}

_SAMPLE_RESPONSE = {
    "matches": [
        {
            "homeTeam": {"name": "Hawthorn"},
            "awayTeam": {"name": "GWS Giants"},
            "markets": [
                _MATCH_ODDS_MARKET,
                _TOTAL_POINTS_MARKET,
                _PLAYER_DISPOSALS_25_MARKET,
            ],
        },
        {
            "homeTeam": {"name": "Collingwood"},
            "awayTeam": {"name": "Melbourne"},
            "markets": [
                _PLAYER_GOALS_2_MARKET,
            ],
        },
    ]
}


# ── parse_tab_markets ─────────────────────────────────────────────────────────

def test_parse_match_odds_maps_to_team_to_win():
    odds, n_matched, n_dropped = parse_tab_markets([_MATCH_ODDS_MARKET])
    assert "Hawthorn to win" in odds
    assert odds["Hawthorn to win"] == pytest.approx(1.36)
    assert "Greater Western Sydney to win" in odds
    assert n_matched == 2  # Draw is silently skipped, not dropped


def test_parse_match_odds_draw_not_dropped():
    """Draw is silently ignored (not a bot leg), should NOT inflate n_dropped."""
    _, n_matched, n_dropped = parse_tab_markets([_MATCH_ODDS_MARKET])
    assert n_dropped == 0


def test_parse_total_points_extracts_over_line():
    odds, n_matched, _ = parse_tab_markets([_TOTAL_POINTS_MARKET])
    assert "Total points 170.5+" in odds
    assert odds["Total points 170.5+"] == pytest.approx(1.85)
    assert "Total points 170.5+ under" not in odds  # Under is not a bot leg
    assert n_matched == 1  # only Over counts


def test_parse_player_disposals_25():
    odds, n_matched, _ = parse_tab_markets([_PLAYER_DISPOSALS_25_MARKET])
    assert "Jai Newcombe 25+ disposals" in odds
    assert odds["Jai Newcombe 25+ disposals"] == pytest.approx(1.65)
    assert "James Worpel 25+ disposals" in odds
    assert n_matched == 2


def test_parse_player_goals_2():
    odds, n_matched, _ = parse_tab_markets([_PLAYER_GOALS_2_MARKET])
    assert "Jack Ginnivan 2+ goals" in odds
    assert n_matched == 1


def test_parse_player_marks_4():
    odds, _, _ = parse_tab_markets([_PLAYER_MARKS_4_MARKET])
    assert "James Sicily 4+ marks" in odds


def test_parse_player_tackles_3():
    odds, _, _ = parse_tab_markets([_PLAYER_TACKLES_3_MARKET])
    assert "Will Day 3+ tackles" in odds


def test_parse_to_get_format():
    """TAB alternate format: 'To Get 25+ Disposals' maps to same key."""
    odds, n_matched, _ = parse_tab_markets([_TO_GET_DISPOSALS_MARKET])
    assert "Jai Newcombe 25+ disposals" in odds
    assert n_matched == 1


def test_parse_to_get_format_same_key_as_player_format():
    """Both betOption formats produce the same leg key — price deduplication works."""
    # Player format
    odds_a, _, _ = parse_tab_markets([_PLAYER_DISPOSALS_25_MARKET])
    # To-Get format (same player)
    odds_b, _, _ = parse_tab_markets([_TO_GET_DISPOSALS_MARKET])
    # Both produce "Jai Newcombe 25+ disposals"
    assert "Jai Newcombe 25+ disposals" in odds_a
    assert "Jai Newcombe 25+ disposals" in odds_b


def test_parse_unrecognised_market_counts_dropped():
    _, _, n_dropped = parse_tab_markets([_UNKNOWN_MARKET])
    assert n_dropped == 2


def test_parse_empty_markets_returns_empty():
    odds, n_matched, n_dropped = parse_tab_markets([])
    assert odds == {}
    assert n_matched == 0
    assert n_dropped == 0


def test_parse_unknown_team_dropped():
    market = {
        "betOption": "Match Odds",
        "propositions": [
            {"name": "UNKNOWNTEAMXYZ", "returnWin": 1.80},
            {"name": "Collingwood", "returnWin": 2.10},
        ],
    }
    odds, n_matched, n_dropped = parse_tab_markets([market])
    assert n_dropped == 1
    assert n_matched == 1
    assert "Collingwood to win" in odds


# ── fetch_tab_odds ────────────────────────────────────────────────────────────

def test_fetch_tab_odds_geo_blocked_returns_empty(tmp_path):
    """Non-JSON (HTML) response means geo-blocked — returns {}."""
    with patch("afl_bot.data.tab_odds.requests.get", return_value=_blocked_response()):
        result = fetch_tab_odds(cache_dir=tmp_path)
    assert result == {}


def test_fetch_tab_odds_network_error_returns_empty(tmp_path):
    """Network exception returns {} without raising."""
    import requests as _requests
    with patch("afl_bot.data.tab_odds.requests.get",
               side_effect=_requests.ConnectionError("timeout")):
        result = fetch_tab_odds(cache_dir=tmp_path)
    assert result == {}


def test_fetch_tab_odds_returns_merged_legs(tmp_path):
    """All matches are merged into a single flat dict."""
    with patch("afl_bot.data.tab_odds.requests.get",
               return_value=_json_response(_SAMPLE_RESPONSE)):
        result = fetch_tab_odds(cache_dir=tmp_path)
    assert "Hawthorn to win" in result
    assert "Total points 170.5+" in result
    assert "Jai Newcombe 25+ disposals" in result
    assert "Jack Ginnivan 2+ goals" in result


def test_fetch_tab_odds_cache_hit_skips_network(tmp_path):
    """A fresh cache file is returned without making a network request."""
    cached = {"Hawthorn to win": 1.36}
    cache_path = tmp_path / "tab_afl_NSW.json"
    cache_path.write_text(json.dumps(cached))
    # mtime is already "now" so cache is fresh

    with patch("afl_bot.data.tab_odds.requests.get") as mock_get:
        result = fetch_tab_odds(cache_dir=tmp_path, cache_seconds=120.0)
    mock_get.assert_not_called()
    assert result == cached


def test_fetch_tab_odds_stale_cache_refetches(tmp_path):
    """A stale cache file triggers a real (mocked) fetch."""
    cached = {"stale key": 1.0}
    cache_path = tmp_path / "tab_afl_NSW.json"
    cache_path.write_text(json.dumps(cached))
    # Make the file appear old (> 120 s)
    old_time = time.time() - 200
    import os
    os.utime(str(cache_path), (old_time, old_time))

    with patch("afl_bot.data.tab_odds.requests.get",
               return_value=_json_response(_SAMPLE_RESPONSE)):
        result = fetch_tab_odds(cache_dir=tmp_path, cache_seconds=120.0)

    assert "Hawthorn to win" in result
    assert "stale key" not in result

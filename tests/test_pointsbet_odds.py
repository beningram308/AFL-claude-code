"""PointsBet odds scraper tests.

HTTP is stubbed (no network). Market JSON shapes are consistent with
the PointsBet API v2 format discovered at api.au.pointsbet.com.
The parse_markets function is symmetric with Sportsbet's -- same leg-name
format, same stat normalisation.
"""
from unittest.mock import Mock, patch

import requests

from afl_bot.data.pointsbet_odds import (
    fetch_event_odds,
    fetch_pointsbet_odds,
    parse_markets,
)

# ── Sample market payloads (API v2 shape) ─────────────────────────────────────

HEAD_TO_HEAD = {
    "name": "Head to Head",
    "selections": [
        {"name": "Hawthorn",  "price": {"winPrice": 1.35}},
        {"name": "Melbourne", "price": {"winPrice": 3.25}},
    ],
}
DISPOSALS_MILESTONE = {
    "name": "20+ Disposals",
    "selections": [
        {"name": "Jai Newcombe", "price": {"winPrice": 1.09}},
        {"name": "Josh Ward",    "price": {"winPrice": 1.22}},
    ],
}
GOALS_SINGULAR = {
    "name": "1+ Goal",
    "selections": [{"name": "Will Day", "price": {"winPrice": 1.70}}],
}
MARKS_MILESTONE = {
    "name": "4+ Marks",
    "selections": [{"name": "James Sicily", "price": {"winPrice": 1.90}}],
}
MARKS_6_MILESTONE = {
    "name": "6+ Marks",
    "selections": [{"name": "James Sicily", "price": {"winPrice": 4.20}}],
}
TACKLES_MILESTONE = {
    "name": "3+ Tackles",
    "selections": [{"name": "Cam Mackenzie", "price": {"winPrice": 1.50}}],
}
UNRELATED_MARKET = {
    "name": "Anytime Goal Scorer",
    "selections": [{"name": "Nick Watson", "price": {"winPrice": 6.50}}],
}


# ── parse_markets unit tests ──────────────────────────────────────────────────

def test_parse_markets_h2h_normalises_teams():
    odds = parse_markets([HEAD_TO_HEAD])
    assert odds == {"Hawthorn to win": 1.35, "Melbourne to win": 3.25}


def test_parse_markets_disposals_plural():
    odds = parse_markets([DISPOSALS_MILESTONE])
    assert odds == {
        "Jai Newcombe 20+ disposals": 1.09,
        "Josh Ward 20+ disposals": 1.22,
    }


def test_parse_markets_goals_singular_name():
    odds = parse_markets([GOALS_SINGULAR])
    assert odds == {"Will Day 1+ goals": 1.70}


def test_parse_markets_marks_and_tackles():
    odds = parse_markets([MARKS_MILESTONE, TACKLES_MILESTONE])
    assert odds == {
        "James Sicily 4+ marks": 1.90,
        "Cam Mackenzie 3+ tackles": 1.50,
    }


def test_parse_markets_unrecognised_market_dropped():
    odds = parse_markets([UNRELATED_MARKET])
    assert odds == {}


def test_parse_markets_full_set():
    all_mkts = [
        HEAD_TO_HEAD, DISPOSALS_MILESTONE, GOALS_SINGULAR,
        MARKS_MILESTONE, MARKS_6_MILESTONE, TACKLES_MILESTONE, UNRELATED_MARKET,
    ]
    odds = parse_markets(all_mkts)
    assert odds == {
        "Hawthorn to win": 1.35,
        "Melbourne to win": 3.25,
        "Jai Newcombe 20+ disposals": 1.09,
        "Josh Ward 20+ disposals": 1.22,
        "Will Day 1+ goals": 1.70,
        "James Sicily 4+ marks": 1.90,
        "James Sicily 6+ marks": 4.20,
        "Cam Mackenzie 3+ tackles": 1.50,
    }


def test_parse_markets_takes_best_when_player_appears_twice():
    mkt_low = {"name": "4+ Marks", "selections": [
        {"name": "James Sicily", "price": {"winPrice": 1.85}}]}
    mkt_high = {"name": "4+ Marks", "selections": [
        {"name": "James Sicily", "price": {"winPrice": 1.95}}]}
    odds = parse_markets([mkt_low, mkt_high])
    assert odds["James Sicily 4+ marks"] == 1.95


def test_parse_markets_missing_price_skipped():
    mkt = {"name": "4+ Marks", "selections": [
        {"name": "James Sicily", "price": None},
        {"name": "Josh Ward",    "price": {"winPrice": 2.10}},
    ]}
    odds = parse_markets([mkt])
    assert "James Sicily 4+ marks" not in odds
    assert odds["Josh Ward 4+ marks"] == 2.10


# ── fetch_event_odds tests (HTTP stubbed) ─────────────────────────────────────

def _json_resp(payload, status=200):
    resp = Mock()
    resp.headers = {"content-type": "application/json; charset=utf-8"}
    resp.status_code = status
    resp.json = Mock(return_value=payload)
    resp.raise_for_status = Mock()
    return resp


def _auth_gate_resp():
    """204 No Content -- PointsBet's auth-required signal."""
    resp = Mock()
    resp.headers = {"content-type": ""}
    resp.status_code = 204
    return resp


def _network_error():
    raise requests.RequestException("timeout")


def test_fetch_event_odds_success_and_caches(tmp_path):
    with patch("afl_bot.data.pointsbet_odds.requests.get",
               return_value=_json_resp([DISPOSALS_MILESTONE, MARKS_MILESTONE])):
        odds = fetch_event_odds("12345678", cache_dir=tmp_path)
    assert odds["Jai Newcombe 20+ disposals"] == 1.09
    assert odds["James Sicily 4+ marks"] == 1.90
    # Cache hit: no further network call
    with patch("afl_bot.data.pointsbet_odds.requests.get") as mock_get2:
        odds2 = fetch_event_odds("12345678", cache_dir=tmp_path)
    assert odds2 == odds
    mock_get2.assert_not_called()


def test_fetch_event_odds_auth_gate_returns_empty(tmp_path):
    with patch("afl_bot.data.pointsbet_odds.requests.get",
               return_value=_auth_gate_resp()):
        assert fetch_event_odds("12345678", cache_dir=tmp_path) == {}


def test_fetch_event_odds_network_error_returns_empty(tmp_path):
    with patch("afl_bot.data.pointsbet_odds.requests.get",
               side_effect=requests.RequestException("boom")):
        assert fetch_event_odds("12345678", cache_dir=tmp_path) == {}


def test_fetch_event_odds_non_json_returns_empty(tmp_path):
    resp = Mock()
    resp.headers = {"content-type": "text/html"}
    resp.status_code = 200
    with patch("afl_bot.data.pointsbet_odds.requests.get", return_value=resp):
        assert fetch_event_odds("12345678", cache_dir=tmp_path) == {}


# ── fetch_pointsbet_odds tests ────────────────────────────────────────────────

def test_fetch_pointsbet_odds_with_explicit_key(tmp_path):
    with patch("afl_bot.data.pointsbet_odds.requests.get",
               return_value=_json_resp([DISPOSALS_MILESTONE])):
        odds = fetch_pointsbet_odds(["12345678"], cache_dir=tmp_path)
    assert "Jai Newcombe 20+ disposals" in odds


def test_fetch_pointsbet_odds_extracts_key_from_url(tmp_path):
    url = "https://pointsbet.com.au/sports/aussie-rules/7523/hawthorn-v-melbourne-12345678"
    with patch("afl_bot.data.pointsbet_odds.requests.get",
               return_value=_json_resp([MARKS_MILESTONE])):
        odds = fetch_pointsbet_odds([url], cache_dir=tmp_path)
    assert "James Sicily 4+ marks" in odds


def test_fetch_pointsbet_odds_unparseable_entry_skipped(tmp_path):
    with patch("afl_bot.data.pointsbet_odds.requests.get") as mock_get:
        odds = fetch_pointsbet_odds(["not-a-url"], cache_dir=tmp_path)
    assert odds == {}
    mock_get.assert_not_called()


def test_fetch_pointsbet_odds_auth_gate_returns_empty(tmp_path):
    with patch("afl_bot.data.pointsbet_odds.requests.get",
               return_value=_auth_gate_resp()):
        assert fetch_pointsbet_odds(["12345678"], cache_dir=tmp_path) == {}


def test_fetch_pointsbet_odds_merges_multiple_events(tmp_path):
    def fake_get(url, **kw):
        if "11111" in url:
            return _json_resp([DISPOSALS_MILESTONE])
        if "22222" in url:
            return _json_resp([MARKS_MILESTONE])
        return _auth_gate_resp()

    with patch("afl_bot.data.pointsbet_odds.requests.get", side_effect=fake_get):
        odds = fetch_pointsbet_odds(["11111", "22222"], cache_dir=tmp_path)
    assert "Jai Newcombe 20+ disposals" in odds
    assert "James Sicily 4+ marks" in odds


def test_fetch_pointsbet_odds_no_keys_no_discovery_returns_empty(tmp_path):
    """When no keys provided and discovery returns [] (auth required) → {}."""
    with patch("afl_bot.data.pointsbet_odds._discover_afl_event_keys", return_value=[]):
        odds = fetch_pointsbet_odds(None, cache_dir=tmp_path)
    assert odds == {}

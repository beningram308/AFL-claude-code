"""Real Sportsbet odds scraper (FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART A).

HTTP is stubbed (no network), following tests/test_live_odds.py. Market JSON
shapes below are trimmed from real live responses captured against
sportsbet.com.au's own SportCard/Markets endpoints."""

from unittest.mock import Mock, patch

import requests

from afl_bot.data.sportsbet_odds import (
    _extract_event_id,
    fetch_event_odds,
    fetch_sportsbet_odds,
    parse_markets,
)

HEAD_TO_HEAD = {
    "name": "Head to Head",
    "selections": [
        {"name": "Hawthorn", "displayHandicap": None, "price": {"winPrice": 1.36}},
        {"name": "GWS GIANTS", "displayHandicap": None, "price": {"winPrice": 3.18}},
    ],
}
TOTAL_POINTS = {
    "name": "Total Game Points - Over/Under",
    "selections": [
        {"name": "Over", "displayHandicap": "+170.5", "unformattedHandicap": "170.5",
         "price": {"winPrice": 1.87}},
        {"name": "Under", "displayHandicap": "+170.5", "unformattedHandicap": "170.5",
         "price": {"winPrice": 1.89}},
    ],
}
DISPOSALS_MILESTONE = {
    "name": "20+ Disposals",
    "selections": [
        {"name": "Jai Newcombe", "displayHandicap": None, "price": {"winPrice": 1.08}},
        {"name": "Karl Amon", "displayHandicap": None, "price": {"winPrice": 1.20}},
    ],
}
GOALS_MILESTONE_SINGULAR = {
    "name": "1+ Goal",
    "selections": [{"name": "Jack Ginnivan", "displayHandicap": None, "price": {"winPrice": 1.65}}],
}
MARKS_MILESTONE = {
    "name": "4+ Marks",
    "selections": [{"name": "James Sicily", "displayHandicap": None, "price": {"winPrice": 1.95}}],
}
TACKLES_MILESTONE = {
    "name": "3+ Tackles",
    "selections": [{"name": "Will Day", "displayHandicap": None, "price": {"winPrice": 1.47}}],
}
UNRELATED_MARKET = {
    "name": "1st Goal",   # "who scores the very first goal" -- not a leg we price
    "selections": [{"name": "Nick Watson", "displayHandicap": None, "price": {"winPrice": 7.0}}],
}

ALL_MARKETS = [HEAD_TO_HEAD, TOTAL_POINTS, DISPOSALS_MILESTONE, GOALS_MILESTONE_SINGULAR,
              MARKS_MILESTONE, TACKLES_MILESTONE, UNRELATED_MARKET]


def test_parse_markets_head_to_head_normalises_team_names():
    odds = parse_markets([HEAD_TO_HEAD])
    assert odds == {"Hawthorn to win": 1.36, "Greater Western Sydney to win": 3.18}


def test_parse_markets_total_points_over_side_only():
    odds = parse_markets([TOTAL_POINTS])
    assert odds == {"Total points 170.5+": 1.87}


def test_parse_markets_milestone_disposals():
    odds = parse_markets([DISPOSALS_MILESTONE])
    assert odds == {"Jai Newcombe 20+ disposals": 1.08, "Karl Amon 20+ disposals": 1.20}


def test_parse_markets_milestone_goal_singular_market_name():
    odds = parse_markets([GOALS_MILESTONE_SINGULAR])
    assert odds == {"Jack Ginnivan 1+ goals": 1.65}


def test_parse_markets_milestone_marks_and_tackles():
    odds = parse_markets([MARKS_MILESTONE, TACKLES_MILESTONE])
    assert odds == {"James Sicily 4+ marks": 1.95, "Will Day 3+ tackles": 1.47}


def test_parse_markets_unmatched_market_dropped():
    odds = parse_markets([UNRELATED_MARKET])
    assert odds == {}


def test_parse_markets_full_set_matches_every_known_shape():
    odds = parse_markets(ALL_MARKETS)
    assert odds == {
        "Hawthorn to win": 1.36, "Greater Western Sydney to win": 3.18,
        "Total points 170.5+": 1.87,
        "Jai Newcombe 20+ disposals": 1.08, "Karl Amon 20+ disposals": 1.20,
        "Jack Ginnivan 1+ goals": 1.65,
        "James Sicily 4+ marks": 1.95,
        "Will Day 3+ tackles": 1.47,
    }


def test_extract_event_id_from_full_url():
    url = "https://www.sportsbet.com.au/betting/australian-rules/afl/hawthorn-v-gws-giants-10599881"
    assert _extract_event_id(url) == "10599881"


def test_extract_event_id_from_bare_id():
    assert _extract_event_id("10599881") == "10599881"
    assert _extract_event_id(10599881) == "10599881"


def test_extract_event_id_unparseable_returns_none():
    assert _extract_event_id("not-a-url") is None


def _json_response(payload, status=200):
    resp = Mock()
    resp.headers = {"content-type": "application/json; charset=utf-8"}
    resp.status_code = status
    resp.json = Mock(return_value=payload)
    resp.raise_for_status = Mock()
    return resp


def _blocked_response():
    resp = Mock()
    resp.headers = {"content-type": "text/html"}
    resp.status_code = 403
    resp.text = "<html>blocked</html>"
    return resp


SPORT_CARD = {
    "marketGrouping": [
        {"id": 125, "name": "Top Markets"},
        {"id": 972, "name": "Pick Your Own Disposals"},
        {"id": 9999, "name": "Some Unwanted Grouping"},
    ],
}


def test_fetch_event_odds_merges_groupings_and_caches(tmp_path):
    responses = {
        "SportCard": _json_response(SPORT_CARD),
        125: _json_response([HEAD_TO_HEAD, TOTAL_POINTS]),
        972: _json_response([DISPOSALS_MILESTONE]),
    }

    def fake_get(url, **kwargs):
        if "SportCard" in url:
            return responses["SportCard"]
        if "/MarketGroupings/125/" in url:
            return responses[125]
        if "/MarketGroupings/972/" in url:
            return responses[972]
        raise AssertionError(f"unexpected grouping fetched: {url}")

    with patch("afl_bot.data.sportsbet_odds.requests.get", side_effect=fake_get) as mock_get:
        odds = fetch_event_odds("10599881", cache_dir=tmp_path)
    assert odds["Hawthorn to win"] == 1.36
    assert odds["Total points 170.5+"] == 1.87
    assert odds["Jai Newcombe 20+ disposals"] == 1.08
    # "Some Unwanted Grouping" never fetched -- only target groupings hit the network
    n_grouping_calls = mock_get.call_count - 1   # minus the SportCard call itself
    assert n_grouping_calls == 2

    # Second call within the cache window must not hit the network again.
    with patch("afl_bot.data.sportsbet_odds.requests.get", side_effect=fake_get) as mock_get2:
        odds2 = fetch_event_odds("10599881", cache_dir=tmp_path)
    assert odds2 == odds
    mock_get2.assert_not_called()


def test_fetch_event_odds_geo_blocked_returns_empty(tmp_path):
    with patch("afl_bot.data.sportsbet_odds.requests.get", return_value=_blocked_response()):
        assert fetch_event_odds("10599881", cache_dir=tmp_path) == {}


def test_fetch_event_odds_network_failure_returns_empty(tmp_path):
    with patch("afl_bot.data.sportsbet_odds.requests.get",
               side_effect=requests.RequestException("timeout")):
        assert fetch_event_odds("10599881", cache_dir=tmp_path) == {}


def test_fetch_event_odds_one_grouping_failing_doesnt_drop_the_others(tmp_path):
    def fake_get(url, **kwargs):
        if "SportCard" in url:
            return _json_response(SPORT_CARD)
        if "/MarketGroupings/125/" in url:
            return _json_response([HEAD_TO_HEAD])
        if "/MarketGroupings/972/" in url:
            raise requests.RequestException("boom")
        raise AssertionError("unexpected URL")

    with patch("afl_bot.data.sportsbet_odds.requests.get", side_effect=fake_get):
        odds = fetch_event_odds("10599881", cache_dir=tmp_path)
    assert odds == {"Hawthorn to win": 1.36, "Greater Western Sydney to win": 3.18}


def test_fetch_sportsbet_odds_merges_multiple_events_and_skips_blocked(tmp_path):
    def fake_get(url, **kwargs):
        if "/Events/111/SportCard" in url:
            return _json_response({"marketGrouping": [{"id": 1, "name": "Top Markets"}]})
        if "/Events/111/MarketGroupings/1/" in url:
            return _json_response([HEAD_TO_HEAD])
        if "/Events/222/SportCard" in url:
            return _blocked_response()
        raise AssertionError(f"unexpected URL: {url}")

    with patch("afl_bot.data.sportsbet_odds.requests.get", side_effect=fake_get):
        odds = fetch_sportsbet_odds(["111", "222"], cache_dir=tmp_path)
    assert odds == {"Hawthorn to win": 1.36, "Greater Western Sydney to win": 3.18}


def test_fetch_sportsbet_odds_empty_input_returns_empty(tmp_path):
    assert fetch_sportsbet_odds([], cache_dir=tmp_path) == {}


def test_fetch_sportsbet_odds_unparseable_entry_skipped(tmp_path):
    with patch("afl_bot.data.sportsbet_odds.requests.get") as mock_get:
        odds = fetch_sportsbet_odds(["not-a-url"], cache_dir=tmp_path)
    assert odds == {}
    mock_get.assert_not_called()

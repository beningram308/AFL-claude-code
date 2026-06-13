"""Live odds intake (MULTI-CHANGES PART A): The Odds API -> report key format.

HTTP is stubbed (no network), following tests/test_odds.py."""

from unittest.mock import Mock, patch

from afl_bot.data.live_odds import fetch_live_odds, parse_events

# Trimmed Odds API v4 /odds response: one event, one AU book, h2h + totals.
EVENTS = [{
    "home_team": "Brisbane Lions",
    "away_team": "Geelong Cats",
    "bookmakers": [{
        "key": "sportsbet",
        "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Brisbane Lions", "price": 1.65},
                {"name": "Geelong Cats", "price": 2.30},
            ]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.90, "point": 170.5},
                {"name": "Under", "price": 1.90, "point": 170.5},
            ]},
        ],
    }, {
        "key": "tab",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "Brisbane Lions", "price": 1.70},   # better price -> should win
            {"name": "Geelong Cats", "price": 2.25},
        ]}],
    }],
}]


def _mock_response(payload):
    resp = Mock()
    resp.json = Mock(return_value=payload)
    resp.raise_for_status = Mock()
    return resp


def test_parse_events_produces_report_key_format():
    odds = parse_events(EVENTS)
    # H2H legs use the CANONICAL team name + " to win" (the feed's "Geelong Cats"
    # normalises to "Geelong", matching how the report names the leg).
    assert "Brisbane Lions to win" in odds
    assert "Geelong to win" in odds
    assert "Total points 170.5+" in odds
    # Under side is dropped (we only price the Over leg the report uses)
    assert not any("Under" in k for k in odds)


def test_parse_events_keeps_best_price_across_books():
    odds = parse_events(EVENTS)
    assert odds["Brisbane Lions to win"] == 1.70   # max of 1.65 / 1.70


def test_fetch_live_odds_uses_key_and_caches(tmp_path):
    with patch("afl_bot.data.live_odds.requests.get") as mock_get:
        mock_get.return_value = _mock_response(EVENTS)

        odds = fetch_live_odds(api_key="k", cache_dir=tmp_path)
        assert odds["Brisbane Lions to win"] == 1.70
        assert mock_get.call_count == 1

        odds2 = fetch_live_odds(api_key="k", cache_dir=tmp_path)   # within cache window
        assert odds2 == odds
        assert mock_get.call_count == 1


def test_fetch_live_odds_no_key_returns_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with patch("afl_bot.data.live_odds.requests.get") as mock_get:
        assert fetch_live_odds(cache_dir=tmp_path) == {}
        mock_get.assert_not_called()   # never hits the network without a key


def test_fetch_live_odds_network_failure_falls_back_to_empty(tmp_path):
    import requests
    with patch("afl_bot.data.live_odds.requests.get",
               side_effect=requests.RequestException("boom")):
        assert fetch_live_odds(api_key="k", cache_dir=tmp_path) == {}

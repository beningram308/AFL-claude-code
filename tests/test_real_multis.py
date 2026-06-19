"""REAL-MULTIS changes: leg-gate bounds, bettable PROP_LINES, live-props parsing."""

import math
from unittest.mock import Mock, patch

import numpy as np

from afl_bot.cli import PROP_LINES
from afl_bot.config import LEG_PROB_MAX, LEG_PROB_MIN
from afl_bot.data.live_odds import parse_event_props


# --------------------------------------------------------------------------- #
# Leg gate (Fix B)
# --------------------------------------------------------------------------- #

def test_leg_gate_is_tighter_than_old_bounds():
    assert LEG_PROB_MIN > 0.05   # old minimum
    assert LEG_PROB_MAX < 0.97   # old maximum


def test_leg_gate_excludes_near_lock():
    # prob > LEG_PROB_MAX (e.g. a star player's 15+ line at ~0.99)
    assert not (LEG_PROB_MIN < 0.99 < LEG_PROB_MAX)


def test_leg_gate_excludes_very_low_prob():
    # prob < LEG_PROB_MIN (e.g. a 35+ disposals line at ~0.05)
    assert not (LEG_PROB_MIN < 0.05 < LEG_PROB_MAX)


def test_leg_gate_admits_mid_range():
    assert LEG_PROB_MIN < 0.55 < LEG_PROB_MAX
    assert LEG_PROB_MIN < 0.40 < LEG_PROB_MAX
    assert LEG_PROB_MIN < 0.70 < LEG_PROB_MAX


def test_prop_lines_extended_to_cover_ball_magnets():
    # disposals goes to 35 so a star (~28 proj) has 30+ and 35+ as candidates
    assert 30 in PROP_LINES["disposals"]
    assert 35 in PROP_LINES["disposals"]
    # goals goes to 3+
    assert 3 in PROP_LINES["goals"]
    # marks and tackles both go to 7+
    assert 7 in PROP_LINES["marks"]
    assert 7 in PROP_LINES["tackles"]


def test_star_player_gets_mid_range_disposals_legs_not_15_plus():
    """A player projecting ~28 disposals: 15+ (~99%) is outside the gate;
    25+ (~72%) and 30+ (~40%) are within it."""
    rng = np.random.default_rng(42)
    samples = rng.normal(28, 5, 50_000).clip(0)

    bettable = []
    for line in PROP_LINES["disposals"]:
        prob = float((samples >= line).mean())
        if LEG_PROB_MIN < prob < LEG_PROB_MAX:
            bettable.append(line)

    assert 15 not in bettable, "15+ should be ~99% — outside gate"
    assert 20 not in bettable, "20+ should be ~95% — outside gate"
    assert any(ln in bettable for ln in (25, 30)), "25+ or 30+ must be bettable"


# --------------------------------------------------------------------------- #
# Live props parsing (Fix A)
# --------------------------------------------------------------------------- #

# Minimal per-event Odds API props response
_EVENT_RESP = {
    "id": "abc",
    "home_team": "Brisbane Lions",
    "away_team": "Geelong",
    "bookmakers": [{
        "key": "sportsbet",
        "markets": [
            {
                "key": "player_disposals_over",
                "outcomes": [
                    {"description": "Caleb Serong", "name": "Over",
                     "price": 1.85, "point": 24.5},
                    {"description": "Caleb Serong", "name": "Under",
                     "price": 2.00, "point": 24.5},   # Under side dropped
                    {"description": "Max Holmes", "name": "Over",
                     "price": 1.70, "point": 19.5},
                ],
            },
            {
                "key": "player_goals_scored_over",
                "outcomes": [
                    {"description": "Jeremy Cameron", "name": "Over",
                     "price": 2.10, "point": 1.5},
                ],
            },
        ],
    }, {
        "key": "tab",
        "markets": [{
            "key": "player_disposals_over",
            "outcomes": [
                {"description": "Caleb Serong", "name": "Over",
                 "price": 1.90, "point": 24.5},   # better price
            ],
        }],
    }],
}


def test_parse_event_props_leg_name_format():
    props = parse_event_props(_EVENT_RESP)
    # 24.5 -> ceil -> 25; format "{player} {line}+ {stat}"
    assert "Caleb Serong 25+ disposals" in props
    assert "Max Holmes 20+ disposals" in props
    assert "Jeremy Cameron 2+ goals" in props


def test_parse_event_props_keeps_best_price_across_books():
    props = parse_event_props(_EVENT_RESP)
    assert props["Caleb Serong 25+ disposals"] == pytest.approx(1.90)  # TAB beats Sportsbet


def test_parse_event_props_drops_under_side():
    props = parse_event_props(_EVENT_RESP)
    assert not any("Under" in k for k in props)


def test_parse_event_props_integer_point_gives_next_line():
    resp = {"bookmakers": [{"key": "sb", "markets": [{
        "key": "player_disposals_over",
        "outcomes": [{"description": "P", "name": "Over", "price": 2.0, "point": 25.0}],
    }]}]}
    props = parse_event_props(resp)
    # "Over 25.0" strictly = 26+
    assert "P 26+ disposals" in props


def test_parse_event_props_half_line_uses_ceil():
    resp = {"bookmakers": [{"key": "sb", "markets": [{
        "key": "player_tackles_over",
        "outcomes": [{"description": "P", "name": "Over", "price": 1.8, "point": 4.5}],
    }]}]}
    props = parse_event_props(resp)
    assert "P 5+ tackles" in props


def test_fetch_live_props_no_key_returns_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    from afl_bot.data.live_odds import fetch_live_props
    with patch("afl_bot.data.live_odds.requests.get") as mock_get:
        result = fetch_live_props(15, cache_dir=tmp_path)
    assert result == {}
    mock_get.assert_not_called()


def test_fetch_live_props_props_tier_blocked_returns_empty(tmp_path):
    import requests as req
    # Events call succeeds, per-event call returns 402 (props tier not enabled)
    events_resp = Mock()
    events_resp.json = Mock(return_value=[{"id": "ev1"}])
    events_resp.raise_for_status = Mock()
    props_resp = Mock()
    props_resp.status_code = 402

    call_count = {"n": 0}
    def _side_effect(*a, **kw):
        call_count["n"] += 1
        return events_resp if call_count["n"] == 1 else props_resp

    from afl_bot.data.live_odds import fetch_live_props
    with patch("afl_bot.data.live_odds.requests.get", side_effect=_side_effect):
        result = fetch_live_props(15, api_key="k", cache_dir=tmp_path)
    assert result == {}


import pytest

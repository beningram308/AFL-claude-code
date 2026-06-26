"""Tests for Stage 2: multis JSON emitter, bets ledger, and auto-settlement."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from afl_bot.build.multi import LegCandidate
from afl_bot.build.report import search_match_sgms
from afl_bot.cli import _rung_to_json
from afl_bot.dashboard.ledger import (
    add_bet,
    cumulative_profit,
    load_ledger,
    pnl_summary,
    save_ledger,
)
from afl_bot.dashboard.settle import settle_bets


# ── fixtures ────────────────────────────────────────────────────────────────

def _make_legs(seed=1):
    rng = np.random.default_rng(seed)
    n = 40_000
    probs = {"A": 0.77, "B": 0.70, "C": 0.62, "D": 0.55, "E": 0.47, "F": 0.41}
    legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        legs.append(LegCandidate(
            f"{name} 20+ disposals", "m1", "player_disposals", name,
            mask.mean(), 1 / mask.mean(), mask=mask))
    return legs


def _make_multi_record(ladder="model", band=2.10):
    return {
        "id": f"2026-r16-Hawthorn-GWS-{ladder}-{band:.2f}",
        "year": 2026, "round": 16,
        "game": "Hawthorn vs Greater Western Sydney",
        "ladder": ladder,
        "band": band,
        "legs": [
            {"player": "Will Day", "market": "disposals", "line": 20,
             "name": "Will Day 20+ disposals", "book_odds": 1.42},
            {"player": "Karl Amon", "market": "disposals", "line": 25,
             "name": "Karl Amon 25+ disposals", "book_odds": 1.85},
            {"player": "Jack Gunston", "market": "goals", "line": 1,
             "name": "Jack Gunston 1+ goals", "book_odds": 1.38},
        ],
        "model_joint": 0.48, "model_fair": 2.10,
        "book_combo": 3.32, "edge": 0.03, "value_pick": False,
    }


# ── Stage 2A: multis JSON emitter ───────────────────────────────────────────

def test_rung_to_json_stable_id():
    legs = _make_legs()
    leg_by_name = {l.name: l for l in legs}
    odds_book = {l.name: l.market_odds for l in legs}
    rungs = search_match_sgms(legs)
    record = _rung_to_json(rungs[0], "model", 2026, 16, "Hawthorn", "GWS",
                           leg_by_name, odds_book)
    assert record["id"].startswith("2026-r16-Hawthorn-GWS-model-")
    assert record["year"] == 2026
    assert record["round"] == 16
    assert record["game"] == "Hawthorn vs GWS"
    assert record["ladder"] == "model"


def test_rung_to_json_leg_metadata():
    """Each leg record carries player, market, line, name, book_odds."""
    legs = _make_legs()
    leg_by_name = {l.name: l for l in legs}
    odds_book = {l.name: l.market_odds for l in legs}
    rungs = search_match_sgms(legs, odds_book=odds_book)
    record = _rung_to_json(rungs[0], "model", 2026, 16, "Home", "Away",
                           leg_by_name, odds_book)
    for leg_json in record["legs"]:
        assert "name" in leg_json
        assert "player" in leg_json
        assert "market" in leg_json
        assert "line" in leg_json
        assert "book_odds" in leg_json
        # line should be an integer parsed from the name
        if leg_json["market"] == "disposals":
            assert leg_json["line"] == 20


def test_rung_to_json_numbers_match_rung():
    """The JSON record's model_joint/model_fair match the rung exactly."""
    legs = _make_legs()
    leg_by_name = {l.name: l for l in legs}
    odds_book = {l.name: l.market_odds for l in legs}
    rungs = search_match_sgms(legs, odds_book=odds_book)
    for rung in rungs:
        record = _rung_to_json(rung, "model", 2026, 16, "Home", "Away",
                               leg_by_name, odds_book)
        assert record["model_joint"] == pytest.approx(rung["joint_prob"])
        assert record["model_fair"] == pytest.approx(rung["fair_odds"])
        if "book_odds" in rung:
            assert record["book_combo"] == pytest.approx(rung["book_odds"])
            assert record["edge"] == pytest.approx(rung["edge"])


def test_rung_to_json_distinct_stable_ids_per_rung():
    """Every rung in the ladder gets a unique stable ID."""
    legs = _make_legs()
    leg_by_name = {l.name: l for l in legs}
    odds_book = {l.name: l.market_odds for l in legs}
    rungs = search_match_sgms(legs, odds_book=odds_book)
    ids = [_rung_to_json(r, "model", 2026, 16, "Home", "Away",
                         leg_by_name, odds_book)["id"]
           for r in rungs]
    assert len(ids) == len(set(ids))


# ── Stage 2B: bets ledger ───────────────────────────────────────────────────

def test_add_bet_appends_pending_record(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    multi = _make_multi_record()
    bet = add_bet(ledger_path, multi, stake=25.0, taken_odds=3.10)
    assert bet["status"] == "pending"
    assert bet["stake"] == 25.0
    assert bet["taken_odds"] == 3.10
    assert bet["payout"] is None
    assert bet["settled_at"] is None
    assert len(bet["legs"]) == 3   # snapshot
    # snapshot is independent of original record
    assert bet["legs"] is not multi["legs"]


def test_add_bet_persists_and_is_reloadable(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    add_bet(ledger_path, _make_multi_record(), 25.0, 3.10)
    add_bet(ledger_path, _make_multi_record(band=3.50), 10.0, 4.20)
    bets = load_ledger(ledger_path)
    assert len(bets) == 2
    assert bets[0]["stake"] == 25.0
    assert bets[1]["stake"] == 10.0


def test_add_bet_leg_snapshot_is_independent(tmp_path):
    """Modifying the original multi record after placement does not alter the snapshot."""
    ledger_path = tmp_path / "bets_ledger.json"
    multi = _make_multi_record()
    bet = add_bet(ledger_path, multi, 25.0, 3.10)
    multi["legs"][0]["book_odds"] = 999.0
    assert bet["legs"][0]["book_odds"] != 999.0


def test_pnl_summary_correct(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    bets = [
        {"bet_id": "a", "status": "won", "stake": 25.0, "taken_odds": 3.10,
         "payout": 77.5, "settled_at": "2026-06-20T12:00:00+10:00", "legs": []},
        {"bet_id": "b", "status": "lost", "stake": 25.0, "taken_odds": 4.50,
         "payout": 0.0, "settled_at": "2026-06-20T13:00:00+10:00", "legs": []},
        {"bet_id": "c", "status": "pending", "stake": 15.0, "taken_odds": 2.75,
         "payout": None, "settled_at": None, "legs": []},
    ]
    save_ledger(ledger_path, bets)
    loaded = load_ledger(ledger_path)
    s = pnl_summary(loaded)
    assert s["total_staked"] == pytest.approx(50.0)    # only settled
    assert s["total_returned"] == pytest.approx(77.5)
    assert s["net_profit"] == pytest.approx(27.5)
    assert s["roi_pct"] == pytest.approx(55.0)
    assert s["n_settled"] == 2
    assert s["n_won"] == 1
    assert s["strike_rate"] == pytest.approx(0.5)


def test_pnl_summary_void_counts_as_settled_but_not_won(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    bets = [{"bet_id": "v", "status": "void", "stake": 10.0, "taken_odds": 2.0,
              "payout": 10.0, "settled_at": "2026-06-20T12:00:00+10:00", "legs": []}]
    save_ledger(ledger_path, bets)
    s = pnl_summary(load_ledger(ledger_path))
    assert s["n_settled"] == 1
    assert s["n_won"] == 0
    assert s["net_profit"] == pytest.approx(0.0)


def test_cumulative_profit_ordered_and_running(tmp_path):
    bets = [
        {"bet_id": "a", "status": "won", "stake": 10.0, "taken_odds": 3.0,
         "payout": 30.0, "settled_at": "2026-06-15T12:00:00+10:00"},
        {"bet_id": "b", "status": "lost", "stake": 10.0, "taken_odds": 3.0,
         "payout": 0.0, "settled_at": "2026-06-20T12:00:00+10:00"},
    ]
    cp = cumulative_profit(bets)
    assert len(cp) == 2
    assert cp[0]["cumulative_profit"] == pytest.approx(20.0)   # 30 - 10
    assert cp[1]["cumulative_profit"] == pytest.approx(10.0)   # 20 - 10


# ── Stage 2C: auto-settlement ────────────────────────────────────────────────

def _make_bet(bet_id, year, round_no, legs, stake=25.0, taken_odds=3.0):
    return {
        "bet_id": bet_id,
        "multi_id": f"multi-{bet_id}",
        "year": year, "round": round_no,
        "game": "Home vs Away",
        "ladder": "model",
        "legs": legs,
        "stake": stake,
        "taken_odds": taken_odds,
        "placed_at": "2026-06-20T12:00:00+10:00",
        "status": "pending",
        "settled_at": None,
        "payout": None,
        "leg_results": None,
    }


def _mock_actuals(h2h=None, total=None, player=None):
    """Patch _load_actuals to return fixed actuals without hitting network."""
    return (h2h or {}, total or {}, player or {})


_LEGS_ALL_HIT = [
    {"player": "Will Day", "market": "disposals", "line": 20,
     "name": "Will Day 20+ disposals", "book_odds": 1.42},
    {"player": "Karl Amon", "market": "disposals", "line": 15,
     "name": "Karl Amon 15+ disposals", "book_odds": 1.30},
    {"player": "Jack Gunston", "market": "goals", "line": 1,
     "name": "Jack Gunston 1+ goals", "book_odds": 1.38},
]
_PLAYER_STAT_ALL_HIT = {
    ("Will Day", "disposals"): 24,
    ("Karl Amon", "disposals"): 18,
    ("Jack Gunston", "goals"): 2,
}
_PLAYER_STAT_ONE_MISS = {
    ("Will Day", "disposals"): 24,
    ("Karl Amon", "disposals"): 12,   # misses 15+
    ("Jack Gunston", "goals"): 2,
}


def test_settle_all_legs_hit_marks_won(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    bet = _make_bet("b1", 2026, 16, _LEGS_ALL_HIT, stake=25.0, taken_odds=3.10)
    save_ledger(ledger_path, [bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=_PLAYER_STAT_ALL_HIT)):
        n = settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert n == 1
    assert bets[0]["status"] == "won"
    assert bets[0]["payout"] == pytest.approx(25.0 * 3.10)
    assert bets[0]["settled_at"] is not None
    for lr in bets[0]["leg_results"]:
        assert lr["hit"] is True


def test_settle_one_leg_miss_marks_lost(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    bet = _make_bet("b2", 2026, 16, _LEGS_ALL_HIT, stake=25.0, taken_odds=3.10)
    save_ledger(ledger_path, [bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=_PLAYER_STAT_ONE_MISS)):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "lost"
    assert bets[0]["payout"] == 0.0
    miss = [lr for lr in bets[0]["leg_results"] if lr["hit"] is False]
    assert len(miss) == 1
    assert miss[0]["name"] == "Karl Amon 15+ disposals"


def test_settle_non_playing_player_voids_leg(tmp_path):
    """If a player has no stat entry, that leg is voided; re-settle remaining legs."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "Injured Player", "market": "disposals", "line": 15,
         "name": "Injured Player 15+ disposals", "book_odds": 1.30},
        {"player": "Will Day", "market": "disposals", "line": 20,
         "name": "Will Day 20+ disposals", "book_odds": 1.42},
        {"player": "Jack Gunston", "market": "goals", "line": 1,
         "name": "Jack Gunston 1+ goals", "book_odds": 1.38},
    ]
    player_stat = {
        # "Injured Player" missing → void that leg
        ("Will Day", "disposals"): 22,
        ("Jack Gunston", "goals"): 1,
    }
    bet = _make_bet("b3", 2026, 16, legs, stake=25.0, taken_odds=3.50)
    save_ledger(ledger_path, [bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=player_stat)):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    void_legs = [lr for lr in bets[0]["leg_results"] if lr["hit"] is None]
    assert len(void_legs) == 1
    # Non-void legs all hit → won
    assert bets[0]["status"] == "won"


def test_settle_all_legs_void_returns_stake(tmp_path):
    """If every leg is void, the bet is void and stake is returned."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "P1", "market": "disposals", "line": 20,
         "name": "P1 20+ disposals", "book_odds": 1.40},
        {"player": "P2", "market": "disposals", "line": 20,
         "name": "P2 20+ disposals", "book_odds": 1.40},
        {"player": "P3", "market": "disposals", "line": 20,
         "name": "P3 20+ disposals", "book_odds": 1.40},
    ]
    bet = _make_bet("b4", 2026, 16, legs, stake=25.0, taken_odds=2.80)
    save_ledger(ledger_path, [bet])

    # h2h populated → round is complete; player_stat empty → all legs void
    h2h = {"Home": 1, "Away": 0}
    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(h2h=h2h, player={})):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "void"
    assert bets[0]["payout"] == pytest.approx(25.0)


def test_settle_no_data_leaves_pending(tmp_path):
    """If actuals are unavailable (round not complete), bet stays pending."""
    ledger_path = tmp_path / "bets_ledger.json"
    bet = _make_bet("b5", 2026, 17, _LEGS_ALL_HIT)
    save_ledger(ledger_path, [bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals()):  # all empty
        settle_bets(ledger_path, year=2026, round_no=17)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "pending"


def test_settle_only_settles_matching_round(tmp_path):
    """settle_bets(year, round_no) only touches bets for that round."""
    ledger_path = tmp_path / "bets_ledger.json"
    r16_bet = _make_bet("r16", 2026, 16, _LEGS_ALL_HIT)
    r17_bet = _make_bet("r17", 2026, 17, _LEGS_ALL_HIT)
    save_ledger(ledger_path, [r16_bet, r17_bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=_PLAYER_STAT_ALL_HIT)):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = {b["bet_id"]: b for b in load_ledger(ledger_path)}
    assert bets["r16"]["status"] == "won"
    assert bets["r17"]["status"] == "pending"

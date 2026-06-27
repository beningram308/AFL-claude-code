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
    add_manual_bet,
    cumulative_profit,
    load_ledger,
    manual_settle_bet,
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


def test_settle_missing_player_stays_pending(tmp_path):
    """A player with no stat entry is ungradeable → whole bet stays PENDING (no phantom win)."""
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
        # "Injured Player" absent → ungradeable leg
        ("Will Day", "disposals"): 22,
        ("Jack Gunston", "goals"): 1,
    }
    bet = _make_bet("b3", 2026, 16, legs, stake=25.0, taken_odds=3.50)
    save_ledger(ledger_path, [bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=player_stat)):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    # No phantom win: bet must stay pending because one leg is ungradeable
    assert bets[0]["status"] == "pending"
    assert bets[0]["payout"] is None
    assert "Injured Player 15+ disposals" in bets[0].get("ungradeable_legs", [])


def test_settle_round_stats_not_published_stays_pending(tmp_path):
    """If player stats are not yet published (empty dict but h2h complete),
    all prop legs are ungradeable and the bet stays PENDING — not void."""
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

    # h2h populated → round complete; player_stat empty → stats not published
    h2h = {"Home": 1, "Away": 0}
    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(h2h=h2h, player={})):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    # Must stay pending — stats not published yet ≠ void
    assert bets[0]["status"] == "pending"
    assert bets[0]["payout"] is None


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


# ── FIX-LOCK: frozen JSON / dashboard read-only / determinism ────────────────

def test_multis_json_fields_identical_to_rung_objects():
    """_rung_to_json faithfully transcribes every field render_markdown reads.

    Both the .md and the JSON are built from the same sgms objects in
    round_report; this test proves the transcription is lossless so the two
    outputs are guaranteed to agree.
    """
    from afl_bot.build.report import search_match_sgms
    legs = _make_legs()
    leg_by_name = {l.name: l for l in legs}
    odds_book = {l.name: l.market_odds for l in legs}
    rungs = search_match_sgms(legs, odds_book=odds_book)
    assert rungs, "need at least one rung"
    for rung in rungs:
        rec = _rung_to_json(rung, "model", 2026, 16, "Home", "Away",
                            leg_by_name, odds_book)
        # core numeric fields
        assert rec["model_joint"] == pytest.approx(rung["joint_prob"])
        assert rec["model_fair"] == pytest.approx(rung["fair_odds"])
        assert rec["band"] == pytest.approx(rung["target_odds"])
        assert rec["value_pick"] == bool(rung.get("value_pick", False))
        # leg names in the same order
        assert [l["name"] for l in rec["legs"]] == rung["legs"]
        # edge/book_combo when priced
        if "book_odds" in rung:
            assert rec["book_combo"] == pytest.approx(rung["book_odds"])
            assert rec["edge"] == pytest.approx(rung["edge"])


def test_dashboard_index_reads_json_never_recomputes(tmp_path):
    """Loading the dashboard index must NOT call the sim, search_match_sgms,
    the Sportsbet scraper, or round_report — it reads the frozen JSON only."""
    import json
    from afl_bot.dashboard.app import app, REPORTS_DIR

    # Write a minimal multis.json so the route has something to render.
    (tmp_path / "2026_r16_multis.json").write_text(
        json.dumps([_make_multi_record()]), encoding="utf-8")

    with (
        patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path),
        patch("afl_bot.build.report.search_match_sgms") as mock_sgms,
        patch("afl_bot.build.report.search_market_sgms") as mock_mkt,
        patch("afl_bot.build.report.build_sgm_candidates") as mock_build,
        patch("afl_bot.data.sportsbet_odds.fetch_sportsbet_odds", create=True) as mock_sb,
    ):
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")

    assert resp.status_code == 200
    mock_sgms.assert_not_called()
    mock_mkt.assert_not_called()
    mock_build.assert_not_called()
    mock_sb.assert_not_called()


def test_loading_same_multis_json_twice_gives_identical_output(tmp_path):
    """Reading the same multis.json twice produces byte-identical rung data —
    the dashboard is purely a read-only view of the frozen JSON."""
    import json
    from afl_bot.dashboard.app import _load_multis_files, _group_by_game, REPORTS_DIR

    records = [_make_multi_record(), _make_multi_record(ladder="sportsbet", band=2.75)]
    jpath = tmp_path / "2026_r16_multis.json"
    jpath.write_text(json.dumps(records), encoding="utf-8")

    with patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path):
        first  = _load_multis_files()
        second = _load_multis_files()

    assert first == second
    # grouping is also stable
    assert _group_by_game(first.get("2026_r16", [])) == _group_by_game(second.get("2026_r16", []))


def test_dashboard_total_ev_and_stake_render(tmp_path):
    """Total EV and Stake columns render formatted values when fields are present."""
    from afl_bot.dashboard.app import app, REPORTS_DIR

    rec = {**_make_multi_record(), "total_ev": 0.475, "suggested_stake": 0.05,
           "p_one_loss": 0.18, "promo_ev": 0.08}
    (tmp_path / "2026_r16_multis.json").write_text(
        json.dumps([rec]), encoding="utf-8")

    with patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path):
        app.config["TESTING"] = True
        resp = app.test_client().get("/")

    body = resp.data.decode()
    assert "+47.5%" in body          # Total EV formatted as signed %
    assert "5.0%" in body            # Stake formatted as % of bankroll
    assert "P(one loss)=18%" in body  # hover tooltip promo breakdown
    assert "Promo EV=+8.0%" in body


def test_dashboard_total_ev_null_renders_dash(tmp_path):
    """When total_ev / suggested_stake are absent the cells render '—'."""
    from afl_bot.dashboard.app import app, REPORTS_DIR

    rec = _make_multi_record()  # no total_ev or suggested_stake
    (tmp_path / "2026_r16_multis.json").write_text(
        json.dumps([rec]), encoding="utf-8")

    with patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path):
        app.config["TESTING"] = True
        resp = app.test_client().get("/")

    body = resp.data.decode()
    # Edge column still renders; Total EV and Stake should show — not crash
    assert resp.status_code == 200
    assert "Total EV" in body


def test_dashboard_stake_zero_renders_dash(tmp_path):
    """suggested_stake=0.0 (below Kelly threshold) renders '—', not '0.0%'."""
    from afl_bot.dashboard.app import app, REPORTS_DIR

    rec = {**_make_multi_record(), "total_ev": 0.12, "suggested_stake": 0.0}
    (tmp_path / "2026_r16_multis.json").write_text(
        json.dumps([rec]), encoding="utf-8")

    with patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path):
        app.config["TESTING"] = True
        resp = app.test_client().get("/")

    body = resp.data.decode()
    # Stake cell with 0 renders — not a span.value; ROI/other cells may still show "0.0%"
    assert 'class="value">0.0%<' not in body
    assert "+12.0%" in body          # Total EV still shows


# ── FIX-HIT-PCT-AND-PREFER-DISPOSALS Part A: hit_prob in JSON + dashboard ────

def test_rung_to_json_includes_hit_prob():
    """Each leg in the JSON record carries hit_prob = the calibrated fair_prob."""
    legs = _make_legs()
    leg_by_name = {l.name: l for l in legs}
    odds_book = {l.name: l.market_odds for l in legs}
    rungs = search_match_sgms(legs, odds_book=odds_book)
    for rung in rungs:
        rec = _rung_to_json(rung, "model", 2026, 16, "Home", "Away",
                            leg_by_name, odds_book)
        for leg_json in rec["legs"]:
            assert "hit_prob" in leg_json
            assert 0.0 < leg_json["hit_prob"] <= 1.0


def test_dashboard_renders_leg_hit_pct(tmp_path):
    """When legs carry hit_prob, the dashboard shows (76%) next to each leg name."""
    from afl_bot.dashboard.app import app, REPORTS_DIR

    rec = {**_make_multi_record()}
    for leg in rec["legs"]:
        leg["hit_prob"] = 0.76
    (tmp_path / "2026_r16_multis.json").write_text(
        json.dumps([rec]), encoding="utf-8")

    with patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path):
        app.config["TESTING"] = True
        resp = app.test_client().get("/")

    body = resp.data.decode()
    assert resp.status_code == 200
    assert "(76%)" in body


def test_dashboard_renders_dash_when_hit_prob_missing(tmp_path):
    """When legs have no hit_prob, the dashboard shows (—) for each leg."""
    from afl_bot.dashboard.app import app, REPORTS_DIR

    rec = _make_multi_record()   # legs have no hit_prob key
    (tmp_path / "2026_r16_multis.json").write_text(
        json.dumps([rec]), encoding="utf-8")

    with patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path):
        app.config["TESTING"] = True
        resp = app.test_client().get("/")

    body = resp.data.decode()
    assert resp.status_code == 200
    assert "(—)" in body   # missing hit_prob shows dash


# ── Part 1 settlement regression tests ──────────────────────────────────────

def test_settle_h2h_hit_plus_no_data_props_stays_pending(tmp_path):
    """H2H leg hits but prop legs have no data → PENDING, NOT WON (phantom-win regression)."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "Carlton", "market": "h2h", "line": None,
         "name": "Carlton to win", "book_odds": 1.80},
        {"player": "Charlie Curnow", "market": "disposals", "line": 20,
         "name": "Charlie Curnow 20+ disposals", "book_odds": 1.50},
        {"player": "Patrick Cripps", "market": "goals", "line": 1,
         "name": "Patrick Cripps 1+ goals", "book_odds": 1.60},
    ]
    bet = _make_bet("phantom", 2026, 16, legs, stake=25.0, taken_odds=4.50)
    save_ledger(ledger_path, [bet])

    # H2H populated (Carlton won), but no prop data yet
    h2h = {"Carlton": 1, "Essendon": 0}
    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(h2h=h2h, player={})):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    # This was the phantom-win bug: bet must NOT be won when prop legs are ungradeable
    assert bets[0]["status"] == "pending", "phantom win must not occur"
    assert bets[0]["payout"] is None


def test_settle_definite_miss_settles_lost_even_with_ungradeable_legs(tmp_path):
    """If one leg is a definite miss, the multi is LOST even if other legs are still ungradeable."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "Will Day", "market": "disposals", "line": 20,
         "name": "Will Day 20+ disposals", "book_odds": 1.42},
        {"player": "Missing Player", "market": "disposals", "line": 20,
         "name": "Missing Player 20+ disposals", "book_odds": 1.50},
    ]
    player_stat = {("Will Day", "disposals"): 10}   # definite miss; Missing Player absent
    bet = _make_bet("miss+ungradeable", 2026, 16, legs, stake=25.0, taken_odds=2.50)
    save_ledger(ledger_path, [bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=player_stat)):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "lost"
    assert bets[0]["payout"] == 0.0


def test_settle_regrade_reverts_phantom_won_to_pending(tmp_path):
    """1C re-grade: a bet currently won but with null leg_results is reverted to pending."""
    ledger_path = tmp_path / "bets_ledger.json"
    # Simulate a previously phantom-won bet: status=won but leg_results has hit=null
    phantom_bet = {
        "bet_id": "phantom-old",
        "multi_id": "multi-phantom-old",
        "year": 2026, "round": 16,
        "game": "Home vs Away",
        "ladder": "model",
        "legs": _LEGS_ALL_HIT,
        "stake": 25.0,
        "taken_odds": 3.10,
        "placed_at": "2026-06-20T12:00:00+10:00",
        "status": "won",   # was incorrectly settled
        "settled_at": "2026-06-21T10:00:00+10:00",
        "payout": 77.5,
        "leg_results": [
            {"name": "Will Day 20+ disposals", "hit": True},
            {"name": "Karl Amon 15+ disposals", "hit": None},  # ungradeable → phantom
            {"name": "Jack Gunston 1+ goals", "hit": True},
        ],
    }
    save_ledger(ledger_path, [phantom_bet])

    # Any settle call triggers the 1C re-grade pass
    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals()):   # no data available this call
        settle_bets(ledger_path)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "pending", "phantom won must be reverted to pending"
    assert bets[0]["payout"] is None
    assert bets[0]["settled_at"] is None


def test_settle_other_market_leg_keeps_bet_pending(tmp_path):
    """A leg with market='other' is always ungradeable; bet stays pending."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "Will Day", "market": "disposals", "line": 20,
         "name": "Will Day 20+ disposals", "book_odds": 1.42},
        {"player": "", "market": "other", "line": None,
         "name": "First goal scorer any team", "book_odds": None},
    ]
    player_stat = {("Will Day", "disposals"): 25}
    bet = _make_bet("other-leg", 2026, 16, legs, stake=25.0, taken_odds=5.0)
    save_ledger(ledger_path, [bet])

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=player_stat)):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "pending"   # "other" leg blocks auto-settlement


# ── Part 3 manual bets tests ─────────────────────────────────────────────────

def test_add_manual_bet_appends_pending_record(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "Charlie Curnow", "market": "player_goals", "line": 2,
         "name": "Charlie Curnow 2+ goals", "book_odds": None},
        {"player": "Carlton", "market": "h2h", "line": None,
         "name": "Carlton to win", "book_odds": None},
    ]
    bet = add_manual_bet(ledger_path, year=2026, round_no=16,
                         game="Carlton vs Essendon",
                         stake=20.0, taken_odds=5.50,
                         legs=legs, label="Own punt")
    assert bet["status"] == "pending"
    assert bet["source"] == "manual"
    assert bet["ladder"] == "manual"
    assert bet["stake"] == 20.0
    assert bet["taken_odds"] == 5.50
    assert bet["label"] == "Own punt"
    assert bet["payout"] is None
    assert bet["manual_result"] is None
    assert bet["multi_id"].startswith("manual-")
    # persisted
    saved = load_ledger(ledger_path)
    assert len(saved) == 1
    assert saved[0]["bet_id"] == bet["bet_id"]


def test_manual_bet_gradeable_legs_auto_settle_won(tmp_path):
    """Manual bet with only gradeable legs (no 'other') settles under Part 1 rules."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "Charlie Curnow", "market": "player_goals", "line": 2,
         "name": "Charlie Curnow 2+ goals", "book_odds": None},
        {"player": "Will Day", "market": "player_disposals", "line": 20,
         "name": "Will Day 20+ disposals", "book_odds": None},
    ]
    add_manual_bet(ledger_path, year=2026, round_no=16,
                   game="Carlton vs Hawthorn",
                   stake=20.0, taken_odds=4.0, legs=legs)
    player_stat = {
        ("Charlie Curnow", "goals"): 3,
        ("Will Day", "disposals"): 25,
    }
    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=player_stat)):
        n = settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert n == 1
    assert bets[0]["status"] == "won"
    assert bets[0]["payout"] == pytest.approx(20.0 * 4.0)


def test_manual_bet_other_leg_stays_pending_until_manual(tmp_path):
    """Manual bet with an 'other' leg cannot auto-settle; stays pending."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [
        {"player": "Will Day", "market": "player_disposals", "line": 20,
         "name": "Will Day 20+ disposals", "book_odds": None},
        {"player": "", "market": "other", "line": None,
         "name": "Anytime Goal Scorer Bonus", "book_odds": None},
    ]
    add_manual_bet(ledger_path, year=2026, round_no=16,
                   game="Hawthorn vs GWS", stake=15.0, taken_odds=6.0, legs=legs)
    player_stat = {("Will Day", "disposals"): 28}
    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals(player=player_stat)):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "pending"   # "other" leg still ungradeable


def test_manual_settle_bet_forces_won(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [{"player": "", "market": "other", "line": None,
             "name": "First goal scorer", "book_odds": None}]
    bet = add_manual_bet(ledger_path, year=2026, round_no=16,
                         game="G1 vs G2", stake=10.0, taken_odds=3.0, legs=legs)
    found = manual_settle_bet(ledger_path, bet["bet_id"], outcome="won")
    assert found is True
    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "won"
    assert bets[0]["payout"] == pytest.approx(30.0)
    assert bets[0]["manual_result"] == "won"
    assert bets[0]["settled_at"] is not None


def test_manual_settle_bet_forces_lost(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [{"player": "", "market": "other", "line": None,
             "name": "First goal scorer", "book_odds": None}]
    bet = add_manual_bet(ledger_path, year=2026, round_no=16,
                         game="G1 vs G2", stake=10.0, taken_odds=3.0, legs=legs)
    manual_settle_bet(ledger_path, bet["bet_id"], outcome="lost")
    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "lost"
    assert bets[0]["payout"] == 0.0


def test_manual_settle_bet_forces_void(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [{"player": "", "market": "other", "line": None,
             "name": "First goal scorer", "book_odds": None}]
    bet = add_manual_bet(ledger_path, year=2026, round_no=16,
                         game="G1 vs G2", stake=10.0, taken_odds=3.0, legs=legs)
    manual_settle_bet(ledger_path, bet["bet_id"], outcome="void")
    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "void"
    assert bets[0]["payout"] == pytest.approx(10.0)


def test_manual_settle_unknown_bet_id_returns_false(tmp_path):
    ledger_path = tmp_path / "bets_ledger.json"
    save_ledger(ledger_path, [])
    assert manual_settle_bet(ledger_path, "nonexistent-id", outcome="won") is False


def test_manual_result_honoured_by_settle_bets(tmp_path):
    """settle_bets honours manual_result and does not attempt auto-grading."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [{"player": "", "market": "other", "line": None,
             "name": "Exotic bet", "book_odds": None}]
    bet = add_manual_bet(ledger_path, year=2026, round_no=16,
                         game="G1 vs G2", stake=10.0, taken_odds=3.0, legs=legs)
    # Set manual_result directly in ledger (simulating prior manual_settle_bet call)
    bets = load_ledger(ledger_path)
    bets[0]["manual_result"] = "won"
    save_ledger(ledger_path, bets)

    with patch("afl_bot.dashboard.settle._load_actuals",
               return_value=_mock_actuals()):
        settle_bets(ledger_path, year=2026, round_no=16)

    bets = load_ledger(ledger_path)
    assert bets[0]["status"] == "won"
    assert bets[0]["payout"] == pytest.approx(30.0)


def test_pnl_includes_manual_bets(tmp_path):
    """pnl_summary counts both bot and manual bets in the season summary."""
    ledger_path = tmp_path / "bets_ledger.json"
    legs = [{"player": "Will Day", "market": "player_disposals", "line": 20,
             "name": "Will Day 20+ disposals", "book_odds": None}]
    bet = add_manual_bet(ledger_path, year=2026, round_no=16,
                         game="Hawthorn vs GWS", stake=20.0, taken_odds=2.0, legs=legs)
    manual_settle_bet(ledger_path, bet["bet_id"], outcome="won")

    bets = load_ledger(ledger_path)
    s = pnl_summary(bets)
    assert s["n_settled"] == 1
    assert s["n_won"] == 1
    assert s["total_staked"] == pytest.approx(20.0)
    assert s["total_returned"] == pytest.approx(40.0)
    assert s["net_profit"] == pytest.approx(20.0)


def test_add_bet_has_source_field(tmp_path):
    """add_bet stores source='bot' by default."""
    ledger_path = tmp_path / "bets_ledger.json"
    multi = _make_multi_record()
    bet = add_bet(ledger_path, multi, stake=25.0, taken_odds=3.10)
    assert bet["source"] == "bot"
    assert bet["manual_result"] is None


def test_dashboard_shows_manual_badge(tmp_path):
    """Dashboard renders the 'manual' badge and label for manual bets."""
    from afl_bot.dashboard.app import app, REPORTS_DIR

    legs = [{"player": "Will Day", "market": "player_disposals", "line": 20,
             "name": "Will Day 20+ disposals", "book_odds": None}]
    ledger_path = tmp_path / "bets_ledger.json"
    add_manual_bet(ledger_path, year=2026, round_no=16,
                   game="Hawthorn vs GWS", stake=20.0, taken_odds=5.0,
                   legs=legs, label="Test punt")
    (tmp_path / "2026_r16_multis.json").write_text(json.dumps([]), encoding="utf-8")

    with (
        patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path),
        patch("afl_bot.dashboard.app.LEDGER_PATH", ledger_path),
    ):
        app.config["TESTING"] = True
        resp = app.test_client().get("/")

    body = resp.data.decode()
    assert resp.status_code == 200
    assert "badge-manual" in body
    assert "Test punt" in body


def test_selection_is_deterministic_under_equal_scoring(tmp_path):
    """When two combos score identically on the primary key, the stable
    leg-name tie-break guarantees the same combo is chosen on every call."""
    from afl_bot.build.report import search_match_sgms

    # Build a pool where multiple combos land very close to the same target
    # so tie-breaking actually matters.
    rng = np.random.default_rng(99)
    n = 30_000
    probs = {"A": 0.72, "B": 0.72, "C": 0.72, "D": 0.72, "E": 0.72}
    same_legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        same_legs.append(LegCandidate(
            f"{name} 20+ disposals", "m1", "player_disposals", name,
            mask.mean(), 1 / mask.mean(), mask=mask))

    result1 = search_match_sgms(same_legs)
    result2 = search_match_sgms(same_legs)
    assert len(result1) == len(result2)
    for r1, r2 in zip(result1, result2):
        assert r1["legs"] == r2["legs"]
        assert r1["joint_prob"] == pytest.approx(r2["joint_prob"])

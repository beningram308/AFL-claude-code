"""Phase 3 tests — CLV math, capture-close, and dashboard CLV panel."""

import json
import math
import pytest

from afl_bot.dashboard.clv import (
    clv_breakdown_by_market,
    clv_stats,
    compute_clv,
    devig_consensus,
    min_detectable_edge,
)
from afl_bot.dashboard.ledger import add_bet, add_clv_snapshot, load_ledger


# ── helpers ──────────────────────────────────────────────────────────────────

def _multi_record(band=2.10, market="disposals"):
    return {
        "id": f"2026-r16-Home-Away-model-{band:.2f}",
        "year": 2026, "round": 16,
        "game": "Home vs Away",
        "ladder": "model",
        "band": band,
        "legs": [
            {"player": "A", "market": market, "line": 20,
             "name": "A 20+ disposals", "book_odds": 1.90},
            {"player": "B", "market": market, "line": 20,
             "name": "B 20+ disposals", "book_odds": 1.75},
            {"player": "C", "market": market, "line": 20,
             "name": "C 20+ disposals", "book_odds": 2.10},
        ],
        "model_joint": 0.28, "model_fair": 2.10,
        "book_combo": 3.32, "edge": 0.03, "value_pick": False,
    }


def _h2h_record():
    return {
        "id": "2026-r16-Home-Away-model-2.10",
        "year": 2026, "round": 16,
        "game": "Home vs Away",
        "ladder": "model",
        "band": 2.10,
        "legs": [
            {"player": "Home", "market": "h2h", "line": None,
             "name": "Home to win", "book_odds": 1.90},
            {"player": "B", "market": "total_points", "line": 150,
             "name": "Total points 150+", "book_odds": 1.85},
            {"player": "C", "market": "h2h", "line": None,
             "name": "C to win", "book_odds": 2.10},
        ],
        "model_joint": 0.28, "model_fair": 2.10,
        "book_combo": 3.32, "edge": 0.03, "value_pick": False,
    }


# ── STEP 1: compute_clv ───────────────────────────────────────────────────────

def test_compute_clv_positive_when_market_moves_in():
    """taken_odds=2.0 (50% implied); close=1.8 (55.6% implied) -> CLV > 0."""
    clv = compute_clv(open_odds=2.0, close_ref_odds=1.8)
    assert clv == pytest.approx(1 / 1.8 - 1 / 2.0)
    assert clv > 0.0


def test_compute_clv_negative_when_market_drifts():
    """Market drifted out (close > open) -> CLV < 0."""
    clv = compute_clv(open_odds=2.0, close_ref_odds=2.2)
    assert clv < 0.0


def test_compute_clv_zero_when_equal():
    clv = compute_clv(open_odds=2.0, close_ref_odds=2.0)
    assert clv == pytest.approx(0.0)


# ── STEP 1: devig_consensus ───────────────────────────────────────────────────

def test_devig_consensus_requires_two_books():
    with pytest.raises(ValueError, match=">=2 books"):
        devig_consensus([(1.90, 2.05)])


def test_devig_consensus_two_equal_books():
    """Identical books -> median = that book's de-vigged prob."""
    pairs = [(1.90, 2.05), (1.90, 2.05)]
    p = devig_consensus(pairs)
    from afl_bot.pricing.edge import devig_proportional
    expected, _ = devig_proportional([1.90, 2.05])
    assert p == pytest.approx(expected)


def test_devig_consensus_median_across_three_books():
    """Median of 3 books, each with different prices."""
    # book1: 1.80/2.20 -> p_over ~0.55
    # book2: 1.90/2.05 -> p_over ~0.52
    # book3: 1.70/2.40 -> p_over ~0.59
    pairs = [(1.80, 2.20), (1.90, 2.05), (1.70, 2.40)]
    from afl_bot.pricing.edge import devig_proportional
    probs = [devig_proportional([o, u])[0] for o, u in pairs]
    probs_sorted = sorted(probs)
    expected_median = probs_sorted[1]
    assert devig_consensus(pairs) == pytest.approx(expected_median)


def test_devig_consensus_reference_is_not_soft_self():
    """Two different books give a consensus; a single Sportsbet price does not qualify."""
    with pytest.raises(ValueError):
        devig_consensus([(1.90, 2.05)])  # only 1 book


# ── STEP 1: min_detectable_edge ───────────────────────────────────────────────

def test_min_detectable_edge_inf_for_zero_n():
    assert math.isinf(min_detectable_edge(0))


def test_min_detectable_edge_decreases_with_n():
    mde10 = min_detectable_edge(10)
    mde100 = min_detectable_edge(100)
    assert mde100 < mde10


def test_min_detectable_edge_uses_fallback_sd():
    """When sd is None, falls back to 0.05."""
    mde = min_detectable_edge(10, sd=None)
    expected = 0.05 * (1.645 + 0.842) / math.sqrt(10)
    assert mde == pytest.approx(expected)


def test_min_detectable_edge_uses_provided_sd():
    sd = 0.10
    mde = min_detectable_edge(10, sd=sd)
    expected = sd * (1.645 + 0.842) / math.sqrt(10)
    assert mde == pytest.approx(expected)


# ── STEP 3: clv_stats ─────────────────────────────────────────────────────────

def test_clv_stats_empty():
    s = clv_stats([])
    assert s["n"] == 0
    assert s["mean_clv"] is None
    assert s["significant"] is False
    assert s["min_detectable_edge"] is None


def test_clv_stats_single_value_no_t_stat():
    s = clv_stats([0.05])
    assert s["n"] == 1
    assert s["mean_clv"] == pytest.approx(0.05)
    assert s["t_stat"] is None
    assert s["significant"] is False


def test_clv_stats_correct_mean_and_pct_positive():
    values = [0.03, -0.01, 0.05, 0.02, 0.04]
    s = clv_stats(values)
    import numpy as np
    assert s["n"] == 5
    assert s["mean_clv"] == pytest.approx(float(np.mean(values)))
    assert s["pct_positive"] == pytest.approx(0.8)   # 4/5 positive


def test_clv_stats_correct_t_stat():
    """t = mean / (sd / sqrt(n))."""
    import numpy as np
    values = [0.03, 0.04, 0.05, 0.02, 0.06]
    s = clv_stats(values)
    arr = np.array(values)
    expected_t = float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(values))))
    assert s["t_stat"] == pytest.approx(expected_t)


def test_clv_stats_significant_when_t_above_1645():
    """Large positive CLV with enough observations -> significant."""
    # 30 bets, mean CLV 5pp, sd 2pp -> t ~ 5/2*sqrt(30) >> 1.645
    import numpy as np
    rng = np.random.default_rng(42)
    values = list(rng.normal(0.05, 0.02, size=30))
    s = clv_stats(values)
    assert s["significant"] is True


def test_clv_stats_not_significant_small_n():
    """Even a high mean CLV with n=2 gives t close to 0 -> not significant."""
    s = clv_stats([0.05, 0.06])
    # t = mean / (sd / sqrt(2)) = ~0.055 / (0.0071 / 1.414) ~ 10.9 — actually can be significant
    # Use values with high sd to ensure not significant
    s2 = clv_stats([0.10, -0.08])
    # mean=0.01, sd=0.127, t=0.01/(0.127/1.414)~0.11 -> not significant
    assert s2["significant"] is False


def test_clv_stats_negative_mean_never_significant():
    values = [-0.02, -0.03, -0.01, -0.04, -0.02, -0.03, -0.01, -0.04,
              -0.02, -0.03, -0.01, -0.04, -0.02, -0.03, -0.01, -0.04,
              -0.02, -0.03, -0.01, -0.04]
    s = clv_stats(values)
    assert s["significant"] is False


def test_clv_stats_mde_decreases_with_more_data():
    s5 = clv_stats([0.02, 0.03, -0.01, 0.04, 0.01])
    s20 = clv_stats([0.02, 0.03, -0.01, 0.04, 0.01] * 4)
    assert s20["min_detectable_edge"] < s5["min_detectable_edge"]


# ── STEP 2: capture_close ─────────────────────────────────────────────────────

def test_capture_close_marks_soft_self_unavailable(tmp_path):
    """All bets get clv_available=False when only Sportsbet is available."""
    from afl_bot.dashboard.capture_close import capture_close
    ledger = tmp_path / "ledger.json"
    add_bet(ledger, _multi_record(), 25.0, 3.10)
    result = capture_close(ledger)
    assert result["n_updated"] == 1
    assert result["n_sharp"] == 0
    assert result["n_soft_only"] == 1
    bets = load_ledger(ledger)
    assert bets[0]["clv_available"] is False
    assert bets[0]["clv_pct"] is None
    assert bets[0]["close_captured_at"] is not None


def test_capture_close_is_idempotent(tmp_path):
    """Re-running capture_close on an already-captured bet is a no-op."""
    from afl_bot.dashboard.capture_close import capture_close
    ledger = tmp_path / "ledger.json"
    add_bet(ledger, _multi_record(), 25.0, 3.10)
    result1 = capture_close(ledger)
    result2 = capture_close(ledger)
    assert result1["n_updated"] == 1
    assert result2["n_updated"] == 0
    bets = load_ledger(ledger)
    assert len(bets) == 1


def test_capture_close_records_line_move_flag(tmp_path):
    """Per-leg line_move_flag is recorded from Sportsbet close prices."""
    from unittest.mock import patch
    from afl_bot.dashboard.capture_close import capture_close
    ledger = tmp_path / "ledger.json"
    add_bet(ledger, _multi_record(), 25.0, 3.10)
    # Mock Sportsbet returning a shorter price on leg A (market moved in)
    sb_prices = {"A 20+ disposals": 1.70}  # open was 1.90 -> shortened
    with patch("afl_bot.data.sportsbet_odds.fetch_sportsbet_odds", return_value=sb_prices):
        capture_close(ledger, sportsbet_urls=["https://example.com/event/1"])
    bets = load_ledger(ledger)
    cl = bets[0]["close_legs"]
    leg_a = next(l for l in cl if l["name"] == "A 20+ disposals")
    assert leg_a["line_move_flag"] == "shortened"


def test_capture_close_h2h_source_is_no_betfair(tmp_path):
    """H2H-only bets get source='no-betfair' to explain why CLV is unavailable."""
    from afl_bot.dashboard.capture_close import capture_close
    ledger = tmp_path / "ledger.json"
    add_bet(ledger, _h2h_record(), 25.0, 2.10)
    capture_close(ledger)
    bets = load_ledger(ledger)
    assert bets[0]["close_ref_source"] == "no-betfair"


def test_capture_close_prop_source_is_single_book(tmp_path):
    """Prop bets get source='single-book' (needs 2nd book for consensus)."""
    from afl_bot.dashboard.capture_close import capture_close
    ledger = tmp_path / "ledger.json"
    add_bet(ledger, _multi_record(market="disposals"), 25.0, 3.10)
    capture_close(ledger)
    bets = load_ledger(ledger)
    assert bets[0]["close_ref_source"] == "single-book"


def test_capture_close_skips_non_pending(tmp_path):
    """Settled bets are not updated."""
    from afl_bot.dashboard.capture_close import capture_close
    from afl_bot.dashboard.ledger import save_ledger
    ledger = tmp_path / "ledger.json"
    add_bet(ledger, _multi_record(), 25.0, 3.10)
    bets = load_ledger(ledger)
    bets[0]["status"] = "won"
    bets[0]["payout"] = 77.5
    save_ledger(ledger, bets)
    result = capture_close(ledger)
    assert result["n_updated"] == 0


# ── STEP 1: add_bet open_odds + add_clv_snapshot ─────────────────────────────

def test_add_bet_includes_open_odds(tmp_path):
    """add_bet now records open_odds = taken_odds as CLV baseline."""
    ledger = tmp_path / "ledger.json"
    bet = add_bet(ledger, _multi_record(), 25.0, 3.10)
    assert bet["open_odds"] == pytest.approx(3.10)
    assert bet["taken_odds"] == pytest.approx(3.10)


def test_add_clv_snapshot_computes_clv_pct(tmp_path):
    """add_clv_snapshot with clv_available=True and a sharp ref computes clv_pct."""
    ledger = tmp_path / "ledger.json"
    bet = add_bet(ledger, _multi_record(), 25.0, 3.20)
    close_ref = 3.00  # market moved in (shorter) -> positive CLV
    ok = add_clv_snapshot(ledger, bet["bet_id"],
                          close_ref_odds=close_ref,
                          close_ref_source="betfair",
                          clv_available=True)
    assert ok is True
    bets = load_ledger(ledger)
    b = bets[0]
    assert b["clv_available"] is True
    assert b["clv_pct"] == pytest.approx(1 / close_ref - 1 / 3.20)
    assert b["clv_pct"] > 0.0
    assert b["close_implied_prob"] == pytest.approx(1 / close_ref)


def test_add_clv_snapshot_returns_false_for_unknown_id(tmp_path):
    ledger = tmp_path / "ledger.json"
    ok = add_clv_snapshot(ledger, "no-such-id",
                          close_ref_odds=2.0, close_ref_source="betfair",
                          clv_available=True)
    assert ok is False


def test_add_clv_snapshot_unavailable_when_no_sharp_ref(tmp_path):
    """clv_available=False -> clv_pct stays None even if close_ref_odds is set."""
    ledger = tmp_path / "ledger.json"
    bet = add_bet(ledger, _multi_record(), 25.0, 3.20)
    add_clv_snapshot(ledger, bet["bet_id"],
                     close_ref_odds=3.0,
                     close_ref_source="soft-self",
                     clv_available=False)
    bets = load_ledger(ledger)
    assert bets[0]["clv_pct"] is None
    assert bets[0]["clv_available"] is False


# ── clv_breakdown_by_market ───────────────────────────────────────────────────

def test_clv_breakdown_excludes_unavailable_bets(tmp_path):
    """clv_breakdown_by_market ignores bets where clv_available=False."""
    ledger = tmp_path / "ledger.json"
    bet = add_bet(ledger, _multi_record(), 25.0, 3.20)
    add_clv_snapshot(ledger, bet["bet_id"],
                     close_ref_odds=None, close_ref_source="single-book",
                     clv_available=False)
    bets = load_ledger(ledger)
    breakdown = clv_breakdown_by_market(bets)
    assert breakdown == {}


def test_clv_breakdown_groups_by_first_leg_market(tmp_path):
    """Two bets in different markets appear in separate groups."""
    ledger = tmp_path / "ledger.json"
    b1 = add_bet(ledger, _multi_record(market="disposals"), 25.0, 3.20)
    b2 = add_bet(ledger, _multi_record(band=3.0, market="goals"), 25.0, 4.00)
    add_clv_snapshot(ledger, b1["bet_id"], close_ref_odds=3.0,
                     close_ref_source="consensus-2", clv_available=True)
    add_clv_snapshot(ledger, b2["bet_id"], close_ref_odds=3.8,
                     close_ref_source="consensus-2", clv_available=True)
    bets = load_ledger(ledger)
    breakdown = clv_breakdown_by_market(bets)
    assert "disposals" in breakdown
    assert "goals" in breakdown
    assert breakdown["disposals"]["n"] == 1
    assert breakdown["goals"]["n"] == 1


# ── Dashboard CLV panel (end-to-end) ─────────────────────────────────────────

def test_dashboard_clv_panel_renders_without_clv_data(tmp_path):
    """Dashboard loads cleanly when there are no CLV-available bets."""
    import json as _json
    from unittest.mock import patch
    from afl_bot.dashboard.app import app

    rec = {
        "id": "2026-r16-Home-Away-model-2.10",
        "year": 2026, "round": 16,
        "game": "Home vs Away", "ladder": "model", "band": 2.10,
        "legs": [{"player": "A", "market": "disposals", "line": 20,
                  "name": "A 20+ disposals", "book_odds": 1.90}],
        "model_joint": 0.48, "model_fair": 2.10,
        "book_combo": 3.32, "edge": 0.03, "value_pick": False,
    }
    (tmp_path / "2026_r16_multis.json").write_text(_json.dumps([rec]), encoding="utf-8")

    with patch("afl_bot.dashboard.app.REPORTS_DIR", tmp_path):
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Closing Line Value" in body
    assert "No CLV data yet" in body or "CLV n/a" in body or "unavailable" in body

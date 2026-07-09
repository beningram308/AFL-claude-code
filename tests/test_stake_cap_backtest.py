"""Stake-cap backtest (DO-STAKE-CAP-BACKTEST) — diagnostic tool, tested with
synthetic rungs/actuals so no network is needed. Covers the settlement
branches (all-win / one-miss-refund / one-miss-no-refund / 2+ miss), the
"bigger cap never gives fewer units" monotonicity property, the read-only
reimplementation of the removed round cap, and that the report renders."""

from __future__ import annotations

import pytest

from afl_bot.backtest.stake_cap import (
    GradedRung,
    SizedBet,
    apply_old_round_cap,
    grade_leg_outcomes,
    hit_rate_cross_check,
    probabilistic_sim,
    realized_replay,
    settle_dollar,
    size_rungs,
    _render_report,
)
from afl_bot.config import BANKROLL, UNIT_MAX, UNIT_SIZE


def _rung(outcome="win", book_odds=3.0, joint_prob=0.30, promo_ev=0.30, total_ev=0.15,
         p_win=0.35, p_one_loss=0.35, p_dead=0.30, year=2026, round_no=16, band=2.10) -> GradedRung:
    return GradedRung(
        year=year, round_no=round_no, game="Home vs Away", ladder="model", band=band,
        joint_prob=joint_prob, book_odds=book_odds, promo_ev=promo_ev, total_ev=total_ev,
        p_win=p_win, p_one_loss=p_one_loss, p_dead=p_dead, outcome=outcome,
    )


# ── grade_leg_outcomes ──────────────────────────────────────────────────────

def test_grade_leg_outcomes_all_hit_is_win():
    legs = [
        {"player": "A", "market": "disposals", "line": 20, "name": "A 20+ disposals"},
        {"player": "B", "market": "goals", "line": 1, "name": "B 1+ goals"},
    ]
    player_stat = {("A", "disposals"): 25.0, ("B", "goals"): 2.0}
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, {}, player_stat, 2026, 16)
    assert outcome == "win"


def test_grade_leg_outcomes_one_miss():
    legs = [
        {"player": "A", "market": "disposals", "line": 20, "name": "A 20+ disposals"},
        {"player": "B", "market": "goals", "line": 1, "name": "B 1+ goals"},
        {"player": "C", "market": "tackles", "line": 3, "name": "C 3+ tackles"},
    ]
    player_stat = {("A", "disposals"): 25.0, ("B", "goals"): 0.0, ("C", "tackles"): 5.0}
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, {}, player_stat, 2026, 16)
    assert outcome == "one_miss"


def test_grade_leg_outcomes_two_miss_is_dead():
    legs = [
        {"player": "A", "market": "disposals", "line": 20, "name": "A 20+ disposals"},
        {"player": "B", "market": "goals", "line": 1, "name": "B 1+ goals"},
        {"player": "C", "market": "tackles", "line": 3, "name": "C 3+ tackles"},
    ]
    player_stat = {("A", "disposals"): 10.0, ("B", "goals"): 0.0, ("C", "tackles"): 5.0}
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, {}, player_stat, 2026, 16)
    assert outcome == "dead"


def test_grade_leg_outcomes_ungradeable_leg_returns_none():
    legs = [
        {"player": "A", "market": "disposals", "line": 20, "name": "A 20+ disposals"},
        {"player": "Unknown", "market": "disposals", "line": 20, "name": "Unknown 20+ disposals"},
    ]
    player_stat = {("A", "disposals"): 25.0}  # "Unknown" has no stat -> ungradeable
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, {}, player_stat, 2026, 16)
    assert outcome is None


# ── settle_dollar: the four settlement branches ─────────────────────────────

def test_settle_dollar_all_win():
    bet = SizedBet(rung=_rung(outcome="win", book_odds=3.0), units=2.0, tag="2u PROMO KELLY",
                   stake=2.0 * UNIT_SIZE)
    net = settle_dollar(bet)
    assert net == pytest.approx(bet.stake * (3.0 - 1.0))


def test_settle_dollar_one_miss_promo_partial_refund():
    bet = SizedBet(rung=_rung(outcome="one_miss"), units=2.0, tag="2u PROMO KELLY",
                   stake=2.0 * UNIT_SIZE)
    net = settle_dollar(bet, refund_factor=0.75)
    assert net == pytest.approx(-(1.0 - 0.75) * bet.stake)
    assert net < 0  # still a net loss, just a smaller one than a straight loss


def test_settle_dollar_one_miss_straight_bet_full_loss():
    # A straight (non-promo) bet has no stake-back refund at all -> full loss on any miss.
    bet = SizedBet(rung=_rung(outcome="one_miss"), units=2.0, tag="2u", stake=2.0 * UNIT_SIZE)
    net = settle_dollar(bet, refund_factor=0.75)
    assert net == pytest.approx(-bet.stake)


def test_settle_dollar_two_plus_miss_is_dead_full_loss():
    bet = SizedBet(rung=_rung(outcome="dead"), units=2.0, tag="2u PROMO KELLY",
                   stake=2.0 * UNIT_SIZE)
    net = settle_dollar(bet)
    assert net == pytest.approx(-bet.stake)


# ── size_rungs: bigger cap never gives fewer units ──────────────────────────

def test_bigger_unit_max_never_reduces_stake():
    # A strong promo-eligible rung whose formula wants more than any of these caps.
    graded = [_rung(book_odds=3.5, joint_prob=0.20, promo_ev=0.30, total_ev=0.30,
                    p_win=0.30, p_one_loss=0.35, p_dead=0.35)]
    caps = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    units_by_cap = {}
    for cap in caps:
        sized = size_rungs(graded, cap)
        units_by_cap[cap] = sized[0].units if sized else 0.0

    ordered_units = [units_by_cap[c] for c in caps]
    assert ordered_units == sorted(ordered_units), (
        f"units must be non-decreasing as unit_max grows: {units_by_cap}"
    )
    # And a bigger cap must never STRICTLY reduce it.
    for a, b in zip(caps, caps[1:]):
        assert units_by_cap[b] >= units_by_cap[a] - 1e-9


def test_size_rungs_skips_unstaked_rungs():
    # A rung with no positive edge and no promo eligibility (bad probs) stakes nothing.
    graded = [_rung(book_odds=1.20, joint_prob=0.10, promo_ev=0.0, total_ev=-0.5,
                    p_win=0.05, p_one_loss=0.05, p_dead=0.90)]
    sized = size_rungs(graded, unit_max=3.0)
    assert sized == []


# ── apply_old_round_cap (read-only reimplementation for comparison only) ────

def test_apply_old_round_cap_allocates_top_ev_first():
    r_a = _rung(round_no=16)
    r_b = _rung(round_no=16)
    r_c = _rung(round_no=16)
    r_a.total_ev, r_b.total_ev, r_c.total_ev = 0.30, 0.20, 0.05
    sized = [
        SizedBet(rung=r_a, units=8.0, tag="8u PROMO KELLY", stake=8.0 * UNIT_SIZE),
        SizedBet(rung=r_b, units=7.0, tag="7u PROMO KELLY", stake=7.0 * UNIT_SIZE),
        SizedBet(rung=r_c, units=4.0, tag="4u PROMO KELLY", stake=4.0 * UNIT_SIZE),
    ]
    kept = apply_old_round_cap(sized, cap_units=15.0)
    kept_units = {id(s.rung): s.units for s in kept}
    assert kept_units.get(id(r_a)) == 8.0, "highest-EV rung keeps full units"
    assert kept_units.get(id(r_b)) == 7.0, "second-EV rung keeps full units"
    assert id(r_c) not in kept_units, "overflow rung is dropped entirely"
    assert sum(s.units for s in kept) <= 15.0 + 1e-9


def test_apply_old_round_cap_within_budget_no_change():
    r_a = _rung(round_no=16)
    r_a.total_ev = 0.20
    sized = [SizedBet(rung=r_a, units=3.0, tag="3u PROMO KELLY", stake=3.0 * UNIT_SIZE)]
    kept = apply_old_round_cap(sized, cap_units=15.0)
    assert len(kept) == 1
    assert kept[0].units == 3.0


# ── realized_replay ──────────────────────────────────────────────────────────

def test_realized_replay_chains_bets_and_computes_roi():
    r_win = _rung(outcome="win", book_odds=3.0, round_no=16)
    r_lose = _rung(outcome="dead", book_odds=3.0, round_no=17)
    sized = [
        SizedBet(rung=r_win, units=1.0, tag="1u PROMO KELLY", stake=1.0 * UNIT_SIZE),
        SizedBet(rung=r_lose, units=1.0, tag="1u PROMO KELLY", stake=1.0 * UNIT_SIZE),
    ]
    result = realized_replay(sized, bankroll0=1500.0)
    assert result.n_bets == 2
    expected_net = (UNIT_SIZE * 2.0) - UNIT_SIZE  # +2u profit on win, -1u on loss
    assert result.net_profit == pytest.approx(expected_net)
    assert result.end_bankroll == pytest.approx(1500.0 + expected_net)
    assert result.total_staked == pytest.approx(2 * UNIT_SIZE)


def test_realized_replay_empty_is_flat():
    result = realized_replay([], bankroll0=1500.0)
    assert result.n_bets == 0
    assert result.end_bankroll == 1500.0
    assert result.net_profit == 0.0
    assert result.max_drawdown_pct == 0.0


# ── probabilistic_sim ────────────────────────────────────────────────────────

def test_probabilistic_sim_certain_win_grows_bankroll_both_modes():
    # p_win=1.0 -> every path wins -> ending bankroll strictly above start in both modes.
    r = _rung(book_odds=3.0, p_win=1.0, p_one_loss=0.0, p_dead=0.0)
    sized = [SizedBet(rung=r, units=1.0, tag="1u PROMO KELLY", stake=1.0 * UNIT_SIZE)]
    results = probabilistic_sim(sized, n_sims=500, bankroll0=1500.0, seed=1)
    for mode in ("fixed", "compounding"):
        assert results[mode].median_end > 1500.0
        assert results[mode].p_down == 0.0


def test_probabilistic_sim_certain_dead_shrinks_bankroll():
    r = _rung(book_odds=3.0, p_win=0.0, p_one_loss=0.0, p_dead=1.0)
    sized = [SizedBet(rung=r, units=1.0, tag="1u PROMO KELLY", stake=1.0 * UNIT_SIZE)]
    results = probabilistic_sim(sized, n_sims=500, bankroll0=1500.0, seed=1)
    for mode in ("fixed", "compounding"):
        assert results[mode].median_end < 1500.0
        assert results[mode].p_down == 1.0


def test_probabilistic_sim_empty_returns_flat_bankroll():
    results = probabilistic_sim([], n_sims=100, bankroll0=1500.0)
    assert results["fixed"].median_end == 1500.0
    assert results["compounding"].median_end == 1500.0


# ── hit_rate_cross_check ─────────────────────────────────────────────────────

def test_hit_rate_cross_check_basic():
    graded = [
        _rung(outcome="win", p_win=0.6),
        _rung(outcome="dead", p_win=0.6),
        _rung(outcome="win", p_win=0.6),
    ]
    result = hit_rate_cross_check(graded)
    assert result["n"] == 3
    assert result["modelled_hit_rate"] == pytest.approx(0.6)
    assert result["actual_hit_rate"] == pytest.approx(2 / 3)


def test_hit_rate_cross_check_empty():
    result = hit_rate_cross_check([])
    assert result["n"] == 0
    assert result["modelled_hit_rate"] is None
    assert result["actual_hit_rate"] is None


# ── report renders without crashing ──────────────────────────────────────────

def test_render_report_smoke():
    graded = [_rung(outcome="win", round_no=16), _rung(outcome="dead", round_no=18)]
    version_a = []
    version_b = {}
    for cap in (1.5, 2.0, 3.0, 4.0):
        sized = size_rungs(graded, cap)
        rep = realized_replay(sized)
        rep.unit_max = cap
        version_a.append(rep)
        version_b[cap] = probabilistic_sim(sized, n_sims=200, seed=1)

    live_sized = size_rungs(graded, UNIT_MAX)
    cap_off = realized_replay(live_sized)
    cap_off.unit_max = UNIT_MAX
    cap_on = realized_replay(apply_old_round_cap(live_sized))
    cap_on.unit_max = UNIT_MAX

    md = _render_report(
        rounds=[(2026, 16), (2026, 18)], graded=graded, n_excluded=0,
        version_a=version_a, version_b=version_b, cap_off=cap_off, cap_on=cap_on,
        live_unit_max=UNIT_MAX, cross_check=hit_rate_cross_check(graded),
        excluded_rounds=[("2026_r17", "no real book_combo prices (model-only run)")],
        n_sims=200,
    )
    assert "# Stake-cap backtest" in md
    assert "Version A" in md
    assert "Version B" in md
    assert "Verdict" in md
    assert "2026 R16, 2026 R18" in md
    assert "2026_r17" in md  # excluded round is disclosed, not silently dropped


def test_render_report_flags_when_caps_never_bind():
    # A weak rung whose formula never wants more than 0.25u, uncapped -- every
    # candidate cap in the sweep produces identical sizing. The report must say
    # so explicitly rather than silently showing 4 identical rows.
    graded = [_rung(outcome="win", round_no=16, book_odds=2.5, joint_prob=0.10,
                    promo_ev=0.20, total_ev=0.11, p_win=0.20, p_one_loss=0.30, p_dead=0.50)]
    version_a = []
    version_b = {}
    for cap in (1.5, 2.0, 3.0, 4.0):
        sized = size_rungs(graded, cap)
        rep = realized_replay(sized)
        rep.unit_max = cap
        version_a.append(rep)
        version_b[cap] = probabilistic_sim(sized, n_sims=200, seed=1)

    uncapped = size_rungs(graded, unit_max=1e6, unit_max_longshot=1e6)
    max_uncapped = max((s.units for s in uncapped), default=0.0)
    assert max_uncapped <= 1.5  # confirms the fixture actually exercises this case

    live_sized = size_rungs(graded, UNIT_MAX)
    cap_off = realized_replay(live_sized)
    cap_off.unit_max = UNIT_MAX
    cap_on = realized_replay(apply_old_round_cap(live_sized))
    cap_on.unit_max = UNIT_MAX

    md = _render_report(
        rounds=[(2026, 16)], graded=graded, n_excluded=0,
        version_a=version_a, version_b=version_b, cap_off=cap_off, cap_on=cap_on,
        live_unit_max=UNIT_MAX, cross_check=hit_rate_cross_check(graded),
        excluded_rounds=[], n_sims=200,
        max_uncapped_units=max_uncapped, caps_never_bind=True,
    )
    assert "not a bug" in md
    assert "No cap comparison is possible this run" in md

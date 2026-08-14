"""DO-EDGE-FLOOR-VIABILITY-TEST (2026-08-14) -- policies B/C, promo-assumption
settlement, and the grade_total_points flag. Synthetic fixtures only (no real
report files / network) except test_reconciliation_gate_matches_audit_numbers,
which is the one test that has to touch real data to mean anything -- marked
slow to keep the fast suite instant, same convention as
test_stake_cap_backtest.py."""

from __future__ import annotations

import pytest

from afl_bot.backtest.stake_cap import (
    GradedRung,
    SizedBet,
    apply_per_player_cap,
    grade_leg_outcomes,
    grade_policy,
    settle_round_assumption,
    size_rungs_edge_floor,
)
from afl_bot.config import UNIT_MAX, UNIT_SIZE


def _rung(*, year=2026, round_no=16, band=5.0, joint_prob=0.30, book_odds=3.0,
         promo_ev=0.30, total_ev=0.15, p_win=0.35, p_one_loss=0.35, p_dead=0.30,
         outcome="win", players=()) -> GradedRung:
    return GradedRung(
        year=year, round_no=round_no, game="Home vs Away", ladder="model", band=band,
        joint_prob=joint_prob, book_odds=book_odds, promo_ev=promo_ev, total_ev=total_ev,
        p_win=p_win, p_one_loss=p_one_loss, p_dead=p_dead, outcome=outcome, players=players,
    )


def _bet(rung: GradedRung, units: float, tag: str = "") -> SizedBet:
    return SizedBet(rung=rung, units=units, tag=tag, stake=units * UNIT_SIZE)


# ── grade_leg_outcomes: grade_total_points flag ─────────────────────────────

def test_total_points_threshold_read_from_name_not_line():
    # `line` is a decoy (5, the placeholder value every total_points leg in
    # saved multis.json actually carries); the real threshold (200.5) only
    # lives in `name`. Total actual is 105 -- if the code used `line` it
    # would (wrongly) compare against 5 and hit; reading `name` correctly
    # makes it a miss (single leg -> n_miss=1 of 1 -> "one_miss").
    legs = [{"market": "total_points", "line": 5, "name": "Total points 200.5+"}]
    total_actual = {"2026_r16_Home_v_Away": 105.0}
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, total_actual, {}, 2026, 16)
    assert outcome == "one_miss"


def test_total_points_hits_when_name_threshold_cleared():
    legs = [
        {"market": "total_points", "line": 5, "name": "Total points 100.5+"},
        {"market": "goals", "player": "A", "line": 1, "name": "A 1+ goals"},
    ]
    total_actual = {"2026_r16_Home_v_Away": 105.0}
    player_stat = {("A", "goals"): 2.0}
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, total_actual, player_stat, 2026, 16)
    assert outcome == "win"


def test_grade_total_points_false_forces_ungradeable_even_with_real_data():
    legs = [
        {"market": "total_points", "line": 5, "name": "Total points 100.5+"},
        {"market": "goals", "player": "A", "line": 1, "name": "A 1+ goals"},
    ]
    total_actual = {"2026_r16_Home_v_Away": 105.0}  # data IS available
    player_stat = {("A", "goals"): 2.0}
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, total_actual, player_stat, 2026, 16,
                                 grade_total_points=False)
    assert outcome is None  # forced ungradeable regardless of available data


def test_grade_total_points_default_is_true_backward_compatible():
    # Default (no kwarg passed) must match pre-existing behaviour exactly --
    # this is the branch every other caller in the module relies on.
    legs = [{"market": "goals", "player": "A", "line": 1, "name": "A 1+ goals"}]
    outcome = grade_leg_outcomes(legs, "Home vs Away", {}, {}, {("A", "goals"): 2.0}, 2026, 16)
    assert outcome == "win"


# ── GradedRung.raw_edge ──────────────────────────────────────────────────────

def test_raw_edge_positive_and_negative():
    assert _rung(joint_prob=0.40, book_odds=3.0).raw_edge == pytest.approx(0.40 * 3.0 - 1.0)
    assert _rung(joint_prob=0.20, book_odds=3.0).raw_edge == pytest.approx(0.20 * 3.0 - 1.0)


def test_raw_edge_none_when_missing_inputs():
    assert _rung(joint_prob=None).raw_edge is None


# ── size_rungs_edge_floor (policy B): eligibility gate ──────────────────────

def test_edge_floor_excludes_non_positive_raw_edge_despite_strong_promo():
    # Exactly the audit's Finding 1 shape: -18% raw edge, but total_ev/promo_ev
    # would clear PROMO_EV_MIN and get staked under the LIVE bot. Policy B must
    # refuse it -- promo cannot create eligibility.
    r = _rung(joint_prob=0.20, book_odds=2.0,  # raw_edge = 0.20*2.0-1 = -0.60
             promo_ev=0.50, total_ev=0.30, p_win=0.20, p_one_loss=0.50, p_dead=0.30)
    assert r.raw_edge < 0.0
    sized = size_rungs_edge_floor([r], unit_max=UNIT_MAX)
    assert sized == []


def test_edge_floor_includes_positive_raw_edge_rung():
    r = _rung(joint_prob=0.40, book_odds=3.0, p_win=0.40, p_one_loss=0.30, p_dead=0.30)
    assert r.raw_edge > 0.0
    sized = size_rungs_edge_floor([r], unit_max=UNIT_MAX)
    assert len(sized) == 1
    assert sized[0].units > 0.0


def test_edge_floor_promo_only_ever_increases_size_never_decreases():
    # Same eligible (positive raw edge) rung, with vs without promo branch
    # probabilities available. Adding promo info must never SHRINK the stake.
    base_kwargs = dict(joint_prob=0.40, book_odds=3.0)
    plain = _rung(**base_kwargs, p_win=None, p_one_loss=None, p_dead=None)
    with_promo = _rung(**base_kwargs, p_win=0.40, p_one_loss=0.45, p_dead=0.15)  # big one-miss refund value

    units_plain = size_rungs_edge_floor([plain], unit_max=UNIT_MAX)[0].units
    units_promo = size_rungs_edge_floor([with_promo], unit_max=UNIT_MAX)[0].units
    assert units_promo >= units_plain


# ── apply_per_player_cap (policy C) ──────────────────────────────────────────

def test_per_player_cap_keeps_highest_edge_drops_other():
    strong = _rung(round_no=16, joint_prob=0.50, book_odds=3.0, players=("Alice",))   # raw_edge=0.50
    weak = _rung(round_no=16, joint_prob=0.34, book_odds=3.0, players=("Alice", "Bob"))  # raw_edge=0.02
    sized = [_bet(strong, 2.0), _bet(weak, 1.0)]
    kept = apply_per_player_cap(sized)
    assert len(kept) == 1
    assert kept[0].rung is strong


def test_per_player_cap_no_shared_players_keeps_both():
    r1 = _rung(round_no=16, players=("Alice",))
    r2 = _rung(round_no=16, players=("Bob",))
    sized = [_bet(r1, 1.0), _bet(r2, 1.0)]
    kept = apply_per_player_cap(sized)
    assert len(kept) == 2


def test_per_player_cap_is_per_round_not_global():
    r1 = _rung(round_no=16, joint_prob=0.50, book_odds=3.0, players=("Alice",))
    r2 = _rung(round_no=17, joint_prob=0.50, book_odds=3.0, players=("Alice",))
    sized = [_bet(r1, 1.0), _bet(r2, 1.0)]
    kept = apply_per_player_cap(sized)
    assert len(kept) == 2  # different rounds -- both survive


# ── settle_round_assumption: three promo modes ──────────────────────────────

def test_promo_mode_none_full_loss_regardless_of_tag():
    win = _bet(_rung(outcome="win", book_odds=3.0), 1.0)
    one_miss = _bet(_rung(outcome="one_miss"), 1.0, tag="1u PROMO KELLY")
    dead = _bet(_rung(outcome="dead"), 1.0)
    results = settle_round_assumption([win, one_miss, dead], "none")
    assert results[0] == pytest.approx(win.stake * 2.0)
    assert results[1] == pytest.approx(-one_miss.stake)  # no refund despite PROMO tag
    assert results[2] == pytest.approx(-dead.stake)


def test_promo_mode_all_refunds_every_one_miss_regardless_of_tag():
    plain_one_miss = _bet(_rung(outcome="one_miss"), 1.0, tag="1u")  # NOT tagged PROMO
    results = settle_round_assumption([plain_one_miss], "all", refund_factor=0.75)
    assert results[0] == pytest.approx(-(1.0 - 0.75) * plain_one_miss.stake)


def test_promo_mode_one_per_round_only_largest_stake_bet_refunded():
    small = _bet(_rung(round_no=16, outcome="one_miss"), 1.0)   # $15
    large = _bet(_rung(round_no=16, outcome="one_miss"), 3.0)   # $45
    results = settle_round_assumption([small, large], "one_per_round", refund_factor=0.75,
                                      refund_dollar_cap=50.0)
    assert results[0] == pytest.approx(-small.stake)  # not the largest -- full loss
    assert results[1] == pytest.approx(-(large.stake - large.stake * 0.75))  # refunded (under $50 cap)


def test_promo_mode_one_per_round_refund_capped_at_dollar_cap():
    # 6u stake = $90 > $50 cap -- only $50 of stake is eligible for the refund.
    big = _bet(_rung(round_no=16, outcome="one_miss"), 6.0)
    results = settle_round_assumption([big], "one_per_round", refund_factor=0.75, refund_dollar_cap=50.0)
    expected = -(big.stake - 50.0 * 0.75)
    assert results[0] == pytest.approx(expected)
    assert results[0] != pytest.approx(-(big.stake - big.stake * 0.75))  # would be wrong if uncapped


def test_promo_mode_one_per_round_no_one_miss_bets_is_a_noop():
    win = _bet(_rung(round_no=16, outcome="win", book_odds=2.0), 1.0)
    results = settle_round_assumption([win], "one_per_round")
    assert results[0] == pytest.approx(win.stake * 1.0)


def test_settle_round_assumption_rejects_unknown_mode():
    with pytest.raises(ValueError):
        settle_round_assumption([], "made_up_mode")


# ── grade_policy aggregation ─────────────────────────────────────────────────

def test_grade_policy_aggregates_n_units_and_pl():
    win = _bet(_rung(round_no=16, outcome="win", book_odds=3.0), 2.0)
    dead = _bet(_rung(round_no=17, outcome="dead", book_odds=3.0), 1.0)
    result = grade_policy([win, dead], policy="B", promo_mode="none")
    assert result.n == 2
    assert result.n_rounds == 2
    assert result.units_staked == pytest.approx(3.0)
    assert result.units_per_round == pytest.approx(1.5)
    expected_pl_dollars = win.stake * 2.0 - dead.stake
    assert result.pl_dollars == pytest.approx(expected_pl_dollars)
    assert result.pl_units == pytest.approx(expected_pl_dollars / UNIT_SIZE)
    assert result.roi_pct == pytest.approx(expected_pl_dollars / (win.stake + dead.stake) * 100.0)


def test_grade_policy_empty_input_is_flat():
    result = grade_policy([], policy="C", promo_mode="all")
    assert result.n == 0
    assert result.units_staked == 0.0
    assert result.units_per_round == 0.0
    assert result.pl_dollars == 0.0
    assert result.roi_pct == 0.0


# ── real-data regression: the reconciliation gate itself ────────────────────

# NOTE (2026-08-14, same day, later in the session): this test used to pin
# check_r18_20_reconciliation(grade_total_points=False) to the audit's own
# 119 / 214.75u / -87.33u for 2026 R18-R20 policy A. That reconciliation
# genuinely happened -- it's what cleared the gate to proceed -- and is
# preserved in reports/edge_floor_viability.md and this module's git history.
# It can no longer be re-asserted here as a live regression test because
# POLICY C SHIPPED INTO THE LIVE afl_bot.build.staking.recommend_units THIS
# SAME SESSION (the whole point of TASK 1's measurement): "policy A" in this
# module intentionally always calls the LIVE recommend_units (its own module
# docstring: "goes through the EXACT live ... recommend_units -- its body is
# never touched" by THIS module), so it now reflects Policy C's raw-edge gate
# too. Re-running check_r18_20_reconciliation(grade_total_points=False) today
# correctly returns n=17, not 119 -- that's the edge floor working as shipped,
# not a broken harness. A test asserting the pre-port number against the
# live path would be pinning a policy Ben explicitly asked to replace.

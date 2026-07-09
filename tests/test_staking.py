import numpy as np
import pytest

from afl_bot.config import BONUS_BET_FACTOR, KELLY_PER_BET_CAP, UNIT_SIZE, UNIT_STEP
from afl_bot.build.staking import (
    bankroll_report,
    fractional_kelly_fraction,
    kelly_fraction,
    multi_outcome_kelly,
    recommend_units,
    simulate_bankroll,
    simulate_bankroll_joint,
    stake_bets,
)


def test_kelly_fraction_math_and_no_edge():
    # p=0.55 @ 2.0 -> f* = (0.55*2 - 1)/(2 - 1) = 0.10
    assert abs(kelly_fraction(0.55, 2.0) - 0.10) < 1e-9
    assert kelly_fraction(0.45, 2.0) == 0.0       # no edge -> no bet
    assert kelly_fraction(0.9, 1.0) == 0.0        # odds <= 1 -> no bet


def test_fractional_kelly_applies_fraction_and_cap():
    # quarter of 0.10 = 0.025, under the per-bet cap
    assert abs(fractional_kelly_fraction(0.55, 2.0, fraction=0.25) - 0.025) < 1e-9
    # a huge edge is capped per bet
    assert fractional_kelly_fraction(0.95, 5.0) == KELLY_PER_BET_CAP


def test_stake_bets_sizes_each_bet_independently():
    # stake_bets must NOT enforce any round-level cap — each bet gets its own Kelly,
    # with no cross-rung influence anywhere in the pipeline (round cap removed 2026-07-10).
    bets = [(f"b{i}", 0.6, 2.0) for i in range(10)]  # each capped at 5%
    staked = stake_bets(bets, 1000.0)
    assert all(abs(s.fraction - KELLY_PER_BET_CAP) < 1e-9 for s in staked), (
        "Each bet must get its independent per-bet-capped Kelly fraction"
    )
    total_frac = sum(s.fraction for s in staked)
    assert total_frac > 0.15, (
        "stake_bets must NOT scale down to any round-level total — no round cap exists"
    )
    assert all(s.stake == s.fraction * 1000.0 for s in staked)


def test_stake_bets_drops_negative_edge():
    staked = stake_bets([("good", 0.6, 2.0), ("bad", 0.40, 2.0)], 1000.0)
    by_name = {s.name: s for s in staked}
    assert by_name["good"].stake > 0
    assert by_name["bad"].stake == 0.0


def test_simulate_bankroll_positive_edge_grows_negative_shrinks():
    rng = np.random.default_rng(0)
    pos = simulate_bankroll([(0.55, 2.0, 0.025)], 1000.0, rounds=24, n_sims=20_000, rng=rng)
    neg = simulate_bankroll([(0.48, 2.0, 0.025)], 1000.0, rounds=24, n_sims=20_000, rng=rng)

    pos_rep = bankroll_report(pos, 1000.0)
    neg_rep = bankroll_report(neg, 1000.0)
    assert pos_rep["median_terminal"] > 1000.0
    assert neg_rep["median_terminal"] < 1000.0
    assert pos_rep["p_profit"] > 0.5
    # drawdown reporting is well-formed
    assert 0.0 <= pos_rep["median_max_drawdown"] <= 1.0
    assert (pos["terminal"] >= 0).all()


def test_simulate_bankroll_joint_correlation_widens_outcomes():
    rng = np.random.default_rng(7)
    n_iter = 50_000
    base = rng.random(n_iter) < 0.55
    # two identical (perfectly correlated) winning bets vs two independent ones
    corr_masks = np.vstack([base, base])
    indep_masks = np.vstack([rng.random(n_iter) < 0.55, rng.random(n_iter) < 0.55])
    bets = [(2.0, 0.05), (2.0, 0.05)]

    corr = simulate_bankroll_joint(bets, corr_masks, 1000.0, rounds=20, n_sims=20_000,
                                   rng=np.random.default_rng(1))
    indep = simulate_bankroll_joint(bets, indep_masks, 1000.0, rounds=20, n_sims=20_000,
                                    rng=np.random.default_rng(1))
    # stacking correlated bets has higher terminal variance than independent ones
    assert corr["terminal"].std() > indep["terminal"].std()
    assert (corr["terminal"] >= 0).all()


def test_bankroll_report_keys():
    rng = np.random.default_rng(1)
    sim = simulate_bankroll([(0.55, 2.0, 0.025)], 500.0, rounds=10, n_sims=2000, rng=rng)
    rep = bankroll_report(sim, 500.0)
    assert {"median_terminal", "p5_terminal", "p95_terminal", "p_profit", "p_bust",
            "median_max_drawdown", "p_drawdown_over_50pct"} == set(rep)


# ── recommend_units ───────────────────────────────────────────────────────────

def test_recommend_units_positive_edge_returns_kelly_units():
    # Good edge bet: prob=0.55, odds=2.10 -> fractional Kelly > 0
    units, tag = recommend_units(0.55, 2.10)
    assert units > 0
    assert tag.endswith("u")
    # Result must be a multiple of UNIT_STEP
    assert abs(units % UNIT_STEP) < 1e-9 or abs(units % UNIT_STEP - UNIT_STEP) < 1e-9


def test_recommend_units_rounding_down():
    # Force a known raw_units and verify floor rounding
    # kelly(0.52, 2.0) = (0.52*2-1)/(2-1) = 0.04; fraction=0.25*0.04=0.01
    # raw_units = 0.01 * 1500 / 15 = 1.0 exactly -> 1.0u
    units, tag = recommend_units(0.52, 2.0)
    assert units == 1.0
    assert tag == "1u"


def test_recommend_units_longshot_cap():
    # book_odds >= 5.0 -> UNIT_MAX_LONGSHOT cap (1.0u)
    units, tag = recommend_units(0.35, 5.50)
    assert units <= 1.0


def test_recommend_units_standard_cap():
    # book_odds < 5.0 -> UNIT_MAX cap (3.0u); huge prob will hit it
    units, tag = recommend_units(0.90, 2.50)
    assert units <= 3.0
    assert units > 0


def test_recommend_units_no_edge_returns_no_bet():
    units, tag = recommend_units(0.40, 2.10)
    assert units == 0.0
    assert tag == "NO BET"


def test_recommend_units_promo_kelly_positive():
    # No base edge (joint=0.20 × odds=3.5 = 0.70 < 1), but promo gives positive Kelly
    # and promo_ev clears PROMO_EV_MIN (0.10).
    # p_win=0.25, p_one_loss=0.40, p_dead=0.35, odds=3.5, R=0.75
    # g'(0) = 0.25*2.5 + 0.40*(-0.25) - 0.35 = 0.625 - 0.10 - 0.35 = 0.175 > 0
    units, tag = recommend_units(
        0.20, 3.5, promo_ev=0.15,
        p_win=0.25, p_one_loss=0.40, p_dead=0.35,
    )
    assert units > 0.0
    assert "PROMO KELLY" in tag
    assert abs(units % UNIT_STEP) < 1e-9 or abs(units % UNIT_STEP - UNIT_STEP) < 1e-9


def test_recommend_units_promo_kelly_below_ev_floor_no_bet():
    # No total_ev supplied -> gate falls back to promo_ev, which sits below
    # PROMO_EV_MIN (0.10) -> must NOT stake even though multi_outcome_kelly
    # itself would say f*>0.
    units, tag = recommend_units(
        0.20, 3.5, promo_ev=0.05,
        p_win=0.25, p_one_loss=0.40, p_dead=0.35,
    )
    assert units == 0.0
    assert tag == "NO BET"


def test_recommend_units_promo_kelly_gates_on_total_ev_not_promo_ev_alone():
    # The 2026-07-09 real-world case: promo_ev alone (0.234) is large -- it would
    # pass even the >0.10 floor on its own -- but total_ev = edge + promo_ev nets
    # out to only +0.056 because the raw edge underneath is deeply negative. The
    # gate must use total_ev (the number actually shown to the user as
    # "Total EV -- that's the number to bet on"), not the isolated promo_ev, so
    # this must be NO BET.
    units, tag = recommend_units(
        0.20, 3.5, promo_ev=0.234, total_ev=0.056,
        p_win=0.25, p_one_loss=0.40, p_dead=0.35,
    )
    assert units == 0.0
    assert tag == "NO BET"


def test_recommend_units_promo_kelly_total_ev_above_floor_stakes():
    # Same large promo_ev, but this time total_ev clears the floor -> stakeable.
    units, tag = recommend_units(
        0.20, 3.5, promo_ev=0.234, total_ev=0.15,
        p_win=0.25, p_one_loss=0.40, p_dead=0.35,
    )
    assert units > 0.0
    assert "PROMO KELLY" in tag


def test_recommend_units_promo_kelly_neg_ev_no_bet():
    # Even with promo, total EV is negative → NO BET.
    # joint=0.05, odds=6.0: base Kelly < 0; g'(0) = 0.05*5 + 0.10*(-0.25) - 0.85 = -0.625 < 0
    units, tag = recommend_units(
        0.05, 6.0, promo_ev=0.075,
        p_win=0.05, p_one_loss=0.10, p_dead=0.85,
    )
    assert units == 0.0
    assert tag == "NO BET"


def test_recommend_units_promo_kelly_refund_cap():
    # Verify the dollar cap: units × unit_size must not exceed promo_refund_cap.
    # Use no base edge (joint=0.30, odds=3.0: kelly=(0.9-1)/2<0), large bankroll,
    # small unit_size and large unit_max to let raw_units run high before dollar cap.
    from afl_bot.config import PROMO_REFUND_CAP
    units, tag = recommend_units(
        0.30, 3.0, promo_ev=0.30,
        p_win=0.40, p_one_loss=0.35, p_dead=0.25,
        bankroll=100_000, unit_size=1.0, unit_step=1.0, unit_max=10_000.0,
        promo_refund_cap=PROMO_REFUND_CAP,
    )
    assert units * 1.0 <= PROMO_REFUND_CAP + 1e-9
    assert "PROMO KELLY" in tag
    assert "capped by promo refund limit" in tag


def test_recommend_units_promo_kelly_without_branch_probs_no_bet():
    # promo_ev > 0 but branch probs not supplied → NO BET (old PROMO ONLY path removed).
    units, tag = recommend_units(0.40, 2.10, promo_ev=0.05)
    assert units == 0.0
    assert tag == "NO BET"


def test_promo_flat_units_fully_removed():
    # Grep-style test: PROMO_FLAT_UNITS must not appear in any source file.
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import afl_bot.config as c; assert not hasattr(c, 'PROMO_FLAT_UNITS'),"
         " 'PROMO_FLAT_UNITS still in config'"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_multi_outcome_kelly_hand_computable_case():
    # p_win=0.5, p_one_miss=0.3, p_dead=0.2, odds=2.5, R=0.75
    # g'(0) = 0.5*1.5 + 0.3*(-0.25) - 0.2 = 0.75 - 0.075 - 0.2 = 0.475 > 0
    # Full Kelly f* ≈ 0.449 (verified numerically); fractional = 0.25*0.449 ≈ 0.112
    # Capped at KELLY_PER_BET_CAP = 0.05.
    f = multi_outcome_kelly(0.5, 0.3, 0.2, 2.5, BONUS_BET_FACTOR)
    assert f == pytest.approx(KELLY_PER_BET_CAP, abs=1e-9)
    # Verify full Kelly is in a plausible range (brute-force check).
    f_full = multi_outcome_kelly(0.5, 0.3, 0.2, 2.5, BONUS_BET_FACTOR, fraction=1.0, cap=1.0)
    assert 0.40 < f_full < 0.55


def test_recommend_units_model_only_no_book_odds():
    # No book price at all -> MODEL-ONLY, never staked
    units, tag = recommend_units(0.70, None)
    assert units == 0.0
    assert tag == "MODEL-ONLY"


def test_recommend_units_minimum_unit_step():
    # Any positive Kelly (even tiny) must return at least UNIT_STEP, not 0
    # prob slightly above breakeven: kelly(0.505, 2.0) = 0.01 -> frac=0.0025
    # raw_units = 0.0025 * 1500 / 15 = 0.25 -> exactly UNIT_STEP
    units, tag = recommend_units(0.505, 2.0)
    assert units == UNIT_STEP
    assert tag == f"{UNIT_STEP}u"


def test_model_only_rungs_excluded_from_monotonicity():
    """MODEL-ONLY rungs (no book price → units=0) must NOT drag staked rungs
    down to 0 via monotonicity enforcement.

    Scenario: two rungs in a model ladder.
      - Rung A: ev=0.30, units=0 (MODEL-ONLY, highest EV but unstakeable)
      - Rung B: ev=0.15, units=0.5 (positive edge, staked)
    Without the fix, A's 0 units would cap B to 0. With the fix, B keeps 0.5u.
    """
    from afl_bot.cli import _enforce_ladder_monotonicity

    rungs = [
        {"total_ev": 0.30, "units": 0.0, "units_tag": "MODEL-ONLY", "no_bet": False},
        {"total_ev": 0.15, "units": 0.5, "units_tag": "0.5u PROMO KELLY", "no_bet": False},
    ]
    _enforce_ladder_monotonicity(rungs)

    assert rungs[0]["units"] == 0.0, "MODEL-ONLY rung should stay at 0"
    assert rungs[1]["units"] == 0.5, (
        "Staked rung must not be dragged to 0 by MODEL-ONLY rung's 0 units"
    )


# ── Part B: monotonicity tag update + round cap (FIX-PULLEM-MENU-AND-STAKE-COLUMNS) ──


def test_monotonicity_updates_units_tag_when_reducing():
    """When _enforce_ladder_monotonicity reduces units, units_tag leading number must sync."""
    import re
    from afl_bot.cli import _enforce_ladder_monotonicity

    # Rung B has higher EV (gets set first), rung A has lower EV with higher units.
    # Sorted descending by total_ev → B first (2u), then A (3u) → A must cap to 2u.
    rungs = [
        {"total_ev": 0.25, "units": 2.0, "units_tag": "2u KELLY",       "no_bet": False},
        {"total_ev": 0.15, "units": 3.0, "units_tag": "3u PROMO KELLY", "no_bet": False},
    ]
    _enforce_ladder_monotonicity(rungs)

    assert rungs[1]["units"] == 2.0, "lower-EV rung must be capped to 2u"
    tag = rungs[1]["units_tag"]
    m = re.match(r"^([\d.]+)u", tag)
    assert m and float(m.group(1)) == 2.0, (
        f"units_tag leading number must update to 2u, got: {tag!r}"
    )
    assert "PROMO KELLY" in tag, "flavor suffix must be preserved"


def test_monotonicity_preserves_tag_when_no_reduction():
    """units_tag must not change when no reduction is needed."""
    from afl_bot.cli import _enforce_ladder_monotonicity

    rungs = [
        {"total_ev": 0.30, "units": 3.0, "units_tag": "3u KELLY",       "no_bet": False},
        {"total_ev": 0.15, "units": 2.0, "units_tag": "2u PROMO KELLY", "no_bet": False},
    ]
    _enforce_ladder_monotonicity(rungs)

    assert rungs[0]["units"] == 3.0
    assert rungs[0]["units_tag"] == "3u KELLY"
    assert rungs[1]["units"] == 2.0
    assert rungs[1]["units_tag"] == "2u PROMO KELLY"


# ── FIX-RESTORE-PROMO-KELLY-UNITS: per-bet formula, no round-level cap ────────


def test_promo_kelly_high_frac_gives_3u():
    """Promo Kelly formula implying 3u+ raw → exactly 3u (UNIT_MAX), $45, correct tag."""
    from afl_bot.build.staking import recommend_units
    from afl_bot.config import BANKROLL, UNIT_SIZE

    # book_odds=3.0 (< 5.0, UNIT_MAX=3u cap); negative direct edge (joint_prob=0.30 < 1/3).
    # Promo branch (p_win=0.35, p_one_loss=0.35, p_dead=0.30) yields large frac → hits cap.
    units, tag = recommend_units(
        joint_prob=0.30,
        book_odds=3.0,
        promo_ev=0.30,
        p_win=0.35,
        p_one_loss=0.35,
        p_dead=0.30,
    )
    assert units == 3.0, f"Expected 3u, got {units}u"
    assert tag == "3u PROMO KELLY", f"Expected '3u PROMO KELLY', got {tag!r}"
    assert units * UNIT_SIZE == 45.0, "3u × $15 must equal $45"
    assert abs(units * UNIT_SIZE / BANKROLL - 0.03) < 1e-9, "stake% must be 45/1500 = 3%"


def test_apply_round_cap_fully_removed():
    # Grep-style test: the round-level budget allocator must not exist anywhere
    # (2026-07-10: removed per Ben's request -- every rung that clears
    # PROMO_EV_MIN now shows its own per-bet Kelly units, nothing gets
    # crowded out into "NO BET (round cap)" by a round-wide 15u ceiling).
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import afl_bot.cli as c\n"
         "assert not hasattr(c, '_apply_round_cap'), '_apply_round_cap still in cli.py'\n"
         "import afl_bot.config as cfg\n"
         "assert not hasattr(cfg, 'KELLY_PER_ROUND_CAP'), 'KELLY_PER_ROUND_CAP still in config'\n"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


# ── FIX-BUDGET-IS-CEILING-NOT-TARGET ──────────────────────────────────────────


def test_same_rung_gets_identical_units_regardless_of_round_size():
    """A rung with fixed probs/price must produce the same units in a 1-game round
    and a 6-game round. The number of rungs in the round must have zero influence
    (no round-level cap exists to make one rung's units depend on any other's)."""
    # A well-edged rung: joint_prob=0.55, odds=2.10
    units_solo, _ = recommend_units(0.55, 2.10)
    assert units_solo > 0.0, "Reference rung must have positive edge"

    # Simulate _units_fields for a 6-game round via recommend_units — each call
    # is independent, so each must return the same value as the solo call.
    for _ in range(6):
        units, _ = recommend_units(0.55, 2.10)
        assert units == units_solo, (
            f"Same rung must give {units_solo}u regardless of round size, got {units}u"
        )

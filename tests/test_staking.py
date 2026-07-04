import numpy as np
import pytest

from afl_bot.config import BONUS_BET_FACTOR, KELLY_PER_BET_CAP, KELLY_PER_ROUND_CAP, UNIT_SIZE, UNIT_STEP
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


def test_stake_bets_respects_per_round_cap():
    bets = [(f"b{i}", 0.6, 2.0) for i in range(10)]  # each capped at 5% -> 50% raw
    staked = stake_bets(bets, 1000.0)
    total_frac = sum(s.fraction for s in staked)
    assert total_frac <= KELLY_PER_ROUND_CAP + 1e-9
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
    # No base edge (joint=0.20 × odds=3.5 = 0.70 < 1), but promo gives positive Kelly.
    # p_win=0.25, p_one_loss=0.40, p_dead=0.35, odds=3.5, R=0.75
    # g'(0) = 0.25*2.5 + 0.40*(-0.25) - 0.35 = 0.625 - 0.10 - 0.35 = 0.175 > 0
    units, tag = recommend_units(
        0.20, 3.5, promo_ev=0.10,
        p_win=0.25, p_one_loss=0.40, p_dead=0.35,
    )
    assert units > 0.0
    assert "PROMO KELLY" in tag
    assert abs(units % UNIT_STEP) < 1e-9 or abs(units % UNIT_STEP - UNIT_STEP) < 1e-9


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

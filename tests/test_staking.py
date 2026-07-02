import numpy as np
import pytest

from afl_bot.config import KELLY_PER_BET_CAP, KELLY_PER_ROUND_CAP, UNIT_SIZE, UNIT_STEP
from afl_bot.build.staking import (
    bankroll_report,
    fractional_kelly_fraction,
    kelly_fraction,
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


def test_recommend_units_promo_only():
    # No edge on base but positive promo_ev
    units, tag = recommend_units(0.40, 2.10, promo_ev=0.05)
    assert units == 0.5
    assert tag == "PROMO ONLY"


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

"""Output-honesty helpers (round-2 §8): multi edge, market shrink, MC SE."""

import numpy as np

from afl_bot.build.multi import LegCandidate, MultiResult
from afl_bot.cli import _min_sims_for_anchor_se, _multi_anchored_prob
from afl_bot.config import ANCHOR_MIN_PROB, MC_SE_TARGET
from afl_bot.pricing.edge import market_anchored_prob, mc_standard_error


def test_combined_edge_is_prob_times_odds_minus_one():
    legs = [LegCandidate("A", "m", "h2h", "A", 0.9, 1.05)]
    multi = MultiResult(legs, combined_fair_prob=0.81, combined_market_odds=1.10)
    assert abs(multi.combined_edge - (0.81 * 1.10 - 1.0)) < 1e-9
    assert multi.combined_edge < 0           # 0.90 leg at 1.05 stacks to -EV


def test_market_anchored_prob_pulls_toward_market():
    # model 0.80 above market-implied 0.50 -> anchored between, lower than model
    anchored = market_anchored_prob(0.80, 2.0, weight=0.25)
    assert 0.50 < anchored < 0.80
    assert abs(anchored - (0.75 * 0.80 + 0.25 * 0.50)) < 1e-9
    assert market_anchored_prob(0.80, 2.0, weight=0.0) == 0.80   # no shrink
    assert market_anchored_prob(0.80, 1.0, weight=0.25) == 0.80  # degenerate odds


def test_mc_standard_error():
    assert abs(mc_standard_error(0.5, 10000) - (0.25 / 10000) ** 0.5) < 1e-12
    assert mc_standard_error(0.9, 0) == float("inf")


def test_min_sims_clears_anchor_se():
    n = _min_sims_for_anchor_se()
    assert mc_standard_error(ANCHOR_MIN_PROB, n) <= MC_SE_TARGET + 1e-9
    assert mc_standard_error(ANCHOR_MIN_PROB, n - 1000) > MC_SE_TARGET   # n is roughly minimal


def test_multi_anchored_prob_haircuts_overestimates():
    # model probs ABOVE their market-implied (1/odds) -> the overestimate case
    legs = [
        LegCandidate("A", "m", "player_disposals", "A", 0.90, 1.30, mask=None),   # mkt 0.77
        LegCandidate("B", "m2", "player_disposals", "B", 0.85, 1.40, mask=None),  # mkt 0.71
    ]
    multi = MultiResult(legs, combined_fair_prob=0.90 * 0.85, combined_market_odds=1.30 * 1.40)
    anchored = _multi_anchored_prob(multi)
    assert anchored < multi.combined_fair_prob   # legs above market -> shrunk down

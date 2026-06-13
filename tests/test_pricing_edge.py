import numpy as np

from afl_bot.pricing.edge import (
    classify_leg,
    devig_proportional,
    edge,
    fair_odds,
    prob_over,
)


def test_prob_over_and_fair_odds():
    samples = np.array([10, 12, 15, 20, 25, 30])
    assert prob_over(samples, 15) == 4 / 6
    assert abs(fair_odds(0.5) - 2.0) < 1e-9


def test_edge_positive_negative():
    assert edge(0.6, 2.0) > 0      # fair odds 1.67, paying 2.0 -> +EV
    assert edge(0.4, 2.0) < 0      # fair odds 2.5, paying 2.0 -> -EV


def test_devig_proportional_sums_to_one():
    probs = devig_proportional([1.9, 1.9])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert abs(probs[0] - probs[1]) < 1e-9


def test_classify_leg_anchor_value_skip():
    assert classify_leg(0.9, 1.10) == "ANCHOR"
    assert classify_leg(0.55, 2.20) == "VALUE"   # edge = 0.55*2.2-1 = 0.21 >= 0.08
    assert classify_leg(0.55, 1.50) == "SKIP"    # edge = -0.175

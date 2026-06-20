import numpy as np

from afl_bot.pricing.edge import (
    classify_leg,
    devig_prop_leg,
    devig_proportional,
    edge,
    fair_odds,
    implied_prob,
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


def test_devig_prop_leg_two_way_matches_devig_proportional():
    result = devig_prop_leg(1.90, 1.95)
    expected = devig_proportional([1.90, 1.95])[0]
    assert result == (expected, "two-way devig")


def test_devig_prop_leg_single_sided_over_only_uses_overround():
    result = devig_prop_leg(1.80, None)
    prob, label = result
    assert label == "single-sided (approx)"
    assert abs(prob - implied_prob(1.80) / 1.06) < 1e-9


def test_devig_prop_leg_single_sided_under_only_complements():
    result = devig_prop_leg(None, 1.80)
    prob, label = result
    assert label == "single-sided (approx)"
    assert abs(prob - (1.0 - implied_prob(1.80) / 1.06)) < 1e-9


def test_devig_prop_leg_neither_side_is_none():
    assert devig_prop_leg(None, None) is None


def test_devig_prop_leg_custom_overround():
    prob, _ = devig_prop_leg(2.00, None, assumed_overround=1.10)
    assert abs(prob - implied_prob(2.00) / 1.10) < 1e-9

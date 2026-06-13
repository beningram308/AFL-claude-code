"""SGM joint pricing (round-2 §3): mask-based joint probability + multinomial
goal allocation."""

import numpy as np

from afl_bot.build.multi import LegCandidate, joint_prob_from_masks
from afl_bot.sim.engine import allocate_player_goals, make_rng

N = 100_000


def _leg(name, prob, mask=None):
    return LegCandidate(name=name, match_id="m1", market="player_disposals",
                        subject=name, fair_prob=prob, market_odds=2.0, mask=mask)


def test_joint_prob_uses_correlation_not_product():
    rng = np.random.default_rng(0)
    base = rng.random(N)
    # two positively-correlated events (both keyed off the same latent)
    a = base < 0.5
    b = base < 0.55          # strongly correlated with a
    joint = joint_prob_from_masks([_leg("a", a.mean(), a), _leg("b", b.mean(), b)])
    product = a.mean() * b.mean()
    assert joint > product + 0.05          # correlation gain over naive multiply


def test_joint_prob_independent_approximates_product():
    rng = np.random.default_rng(1)
    a = rng.random(N) < 0.5
    b = rng.random(N) < 0.4                # independent draws
    joint = joint_prob_from_masks([_leg("a", a.mean(), a), _leg("b", b.mean(), b)])
    assert abs(joint - a.mean() * b.mean()) < 0.01


def test_joint_prob_falls_back_to_product_without_masks():
    joint = joint_prob_from_masks([_leg("a", 0.6, None), _leg("b", 0.5, None)])
    assert abs(joint - 0.30) < 1e-9


def test_allocate_player_goals_sum_constrained_and_matches_shares():
    rng = make_rng(3)
    team_goals = rng.integers(6, 18, size=N)          # per-iteration team goals
    shares = np.array([0.22, 0.15, 0.08])             # 3 forwards
    alloc = allocate_player_goals(team_goals, shares, rng)

    assert alloc.shape == (3, N)
    assert (alloc.sum(axis=0) <= team_goals).all()    # never exceed the team total
    assert (alloc >= 0).all()
    for i, s in enumerate(shares):
        assert abs(alloc[i].mean() - s * team_goals.mean()) < 0.3   # marginal ~ share*team

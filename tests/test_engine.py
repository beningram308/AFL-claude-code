import numpy as np

from afl_bot.config import DEFAULT_SHOT_ACCURACY
from afl_bot.sim.engine import (
    Team,
    allocate_player_stats,
    draw_pace,
    make_rng,
    simulate_match,
    simulate_team_score,
    simulate_team_stat_total,
)

N = 100_000


def test_simulate_team_score_returns_integer_scores_and_consistent_points():
    rng = make_rng(seed=1)
    out = simulate_team_score(mu_shots=22.0, accuracy=0.52, n=N, rng=rng)

    assert (out["shots"] >= 0).all()
    assert (out["goals"] <= out["shots"]).all()
    assert (out["behinds"] == out["shots"] - out["goals"]).all()
    assert (out["points"] == 6 * out["goals"] + out["behinds"]).all()

    assert abs(out["shots"].mean() - 22.0) < 0.5


def test_simulate_match_recovers_margin_and_total_in_expectation():
    rng = make_rng(seed=2)
    out = simulate_match(
        Team("Carlton", True), Team("Richmond"),
        mu_margin=10.0, mu_total=170.0,
        home_accuracy=0.55, away_accuracy=0.50,
        n=N, rng=rng,
    )

    assert abs(out["margin"].mean() - 10.0) < 1.0
    assert abs((out["home_pts"] + out["away_pts"]).mean() - 170.0) < 1.5

    # integer scorelines
    assert np.allclose(out["home_pts"], np.round(out["home_pts"]))
    assert np.allclose(out["away_pts"], np.round(out["away_pts"]))

    # outcome probabilities partition exactly
    assert np.allclose(out["home_win"] + out["away_win"] + out["draw"], 1.0)

    # real (non-zero) draw probability, unlike a continuous Normal margin
    assert out["draw"].mean() > 0.0


def test_simulate_match_handles_missing_accuracy_profile():
    rng = make_rng(seed=3)
    out = simulate_match(
        Team("NewTeam", True), Team("OtherNewTeam"),
        mu_margin=0.0, mu_total=160.0,
        home_accuracy=float("nan"), away_accuracy=float("nan"),
        n=1000, rng=rng,
    )
    # falls back to DEFAULT_SHOT_ACCURACY without raising, scores still produced
    assert out["home_pts"].mean() > 0
    assert out["away_pts"].mean() > 0
    assert 0 < DEFAULT_SHOT_ACCURACY < 1


def test_simulate_match_higher_total_increases_variance_heteroscedastic():
    rng = make_rng(seed=4)
    low = simulate_match(
        Team("A", True), Team("B"), mu_margin=0.0, mu_total=120.0,
        home_accuracy=0.50, away_accuracy=0.50, n=N, rng=rng,
    )
    high = simulate_match(
        Team("A", True), Team("B"), mu_margin=0.0, mu_total=200.0,
        home_accuracy=0.50, away_accuracy=0.50, n=N, rng=rng,
    )
    # NB variance grows with the mean -> higher-scoring games have more
    # shot-count (and therefore points) variance (plan §2.2).
    assert high["home_pts"].std() > low["home_pts"].std()


# --------------------------------------------------------------------------- #
# Score copula (plan §3.3): negative home/away correlation, preserved marginals
# --------------------------------------------------------------------------- #
def test_simulate_match_wet_lowers_total_and_goals_preserves_margin():
    rng = make_rng(seed=11)
    kw = dict(mu_margin=8.0, mu_total=170.0, home_accuracy=0.525, away_accuracy=0.525, n=N)
    dry = simulate_match(Team("H", True), Team("A"), rng=rng, greasiness=0.0, **kw)
    wet = simulate_match(Team("H", True), Team("A"), rng=rng, greasiness=1.0, **kw)

    dry_total = (dry["home_pts"] + dry["away_pts"]).mean()
    wet_total = (wet["home_pts"] + wet["away_pts"]).mean()
    dry_goals = (dry["home_goals"] + dry["away_goals"]).mean()
    wet_goals = (wet["home_goals"] + wet["away_goals"]).mean()

    assert 0.90 < wet_total / dry_total < 0.96            # ~0.93 total multiplier
    assert wet_goals < dry_goals                          # lower conversion -> fewer goals
    assert abs(wet["margin"].mean() - dry["margin"].mean()) < 1.5  # margin unaffected


def test_simulate_match_score_correlation_is_negative_by_default():
    rng = make_rng(seed=5)
    out = simulate_match(
        Team("A", True), Team("B"), mu_margin=0.0, mu_total=162.0,
        home_accuracy=0.525, away_accuracy=0.525, n=200_000, rng=rng,
    )
    corr = np.corrcoef(out["home_pts"], out["away_pts"])[0, 1]
    assert -0.30 < corr < -0.12  # calibrated around -0.22


def test_score_correlation_zero_recovers_independence_and_preserves_means():
    rng = make_rng(seed=6)
    corr_on = simulate_match(
        Team("A", True), Team("B"), mu_margin=10.0, mu_total=170.0,
        home_accuracy=0.52, away_accuracy=0.52, n=200_000, rng=rng,
    )
    rng = make_rng(seed=6)
    corr_off = simulate_match(
        Team("A", True), Team("B"), mu_margin=10.0, mu_total=170.0,
        home_accuracy=0.52, away_accuracy=0.52, n=200_000, rng=rng,
        score_correlation=0.0,
    )
    # independence -> ~zero correlation
    assert abs(np.corrcoef(corr_off["home_pts"], corr_off["away_pts"])[0, 1]) < 0.02
    # copula preserves the marginals: per-team means/variance unchanged, only
    # the cross-correlation (and hence margin/total split) differs.
    assert abs(corr_on["home_pts"].mean() - corr_off["home_pts"].mean()) < 0.5
    assert abs(corr_on["home_pts"].std() - corr_off["home_pts"].std()) < 0.5
    # negative score correlation widens the margin and tightens the total.
    assert corr_on["margin"].std() > corr_off["margin"].std()
    assert (corr_on["home_pts"] + corr_on["away_pts"]).std() < \
           (corr_off["home_pts"] + corr_off["away_pts"]).std()


# --------------------------------------------------------------------------- #
# Pace latent factor + within-team Dirichlet allocation (plan §2.5, §3.3)
# --------------------------------------------------------------------------- #
def test_draw_pace_mean_one_and_positive():
    rng = make_rng(seed=7)
    pace = draw_pace(200_000, rng, pace_sigma=0.07)
    assert (pace > 0).all()
    assert abs(pace.mean() - 1.0) < 0.01
    assert abs(pace.std() - 0.07) < 0.01


def test_shared_pace_couples_team_totals():
    rng = make_rng(seed=8)
    pace = draw_pace(N, rng)
    home = simulate_team_stat_total(385.0, pace, rng)
    away = simulate_team_stat_total(370.0, pace, rng)
    # same pace array -> positively correlated team disposal totals
    assert np.corrcoef(home, away)[0, 1] > 0.15
    assert abs(home.mean() - 385.0) < 5.0  # pace has mean 1 -> total mean preserved

    # independent pace draws -> ~no correlation (control)
    h2 = simulate_team_stat_total(385.0, draw_pace(N, rng), rng)
    a2 = simulate_team_stat_total(370.0, draw_pace(N, rng), rng)
    assert abs(np.corrcoef(h2, a2)[0, 1]) < 0.05


def test_allocate_player_stats_sums_within_total_and_matches_shares():
    rng = make_rng(seed=9)
    pace = draw_pace(N, rng)
    team_total = simulate_team_stat_total(385.0, pace, rng)
    shares = np.array([0.085, 0.075, 0.065, 0.05])
    alloc = allocate_player_stats(team_total, shares, rng)

    assert alloc.shape == (len(shares), N)
    # priced players never exceed the team total in any iteration
    assert (alloc.sum(axis=0) <= team_total).all()
    # each player's marginal mean ~= expected share * E[team total]
    for i, s in enumerate(shares):
        assert abs(alloc[i].mean() - s * 385.0) < 2.0
    # a player moves with the team total (shared pace + their share)
    assert np.corrcoef(alloc[0], team_total)[0, 1] > 0.2

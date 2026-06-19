"""Phase-2 correlation/dispersion fitting (model-upgrade audit): each
estimator is checked against data simulated from the *same* model functions
at a known ground-truth parameter, so a correct estimator should recover
something close to the value used to generate the data."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from afl_bot.backtest.correlations import (
    CORRELATION_PARAMS_ARTIFACT,
    fit_correlation_params,
    fit_pace_and_dispersion,
    fit_score_shot_correlation,
    fit_share_concentration,
    fit_shot_dispersion,
    load_fitted_correlation_params,
)
from afl_bot.sim.engine import allocate_player_stats, draw_pace, make_rng, simulate_team_stat_total


def test_fit_shot_dispersion_recovers_known_value():
    rng = np.random.default_rng(0)
    true_r, mu = 42.5, 22.3
    p = true_r / (true_r + mu)
    shots = rng.negative_binomial(true_r, p, size=20_000).astype(float)
    accuracy = 0.525
    goals = rng.binomial(shots.astype(int), accuracy)
    behinds = shots - goals
    half = len(shots) // 2
    games = pd.DataFrame({
        "hgoals": goals[:half], "hbehinds": behinds[:half],
        "agoals": goals[half:half * 2], "abehinds": behinds[half:half * 2],
    })
    result = fit_shot_dispersion(games)
    assert abs(result["value"] - true_r) / true_r < 0.15


def _synth_team_totals(true_pace_sigma, true_dispersion, mu, n_games, seed):
    rng = make_rng(seed)
    pace = draw_pace(n_games, rng, pace_sigma=true_pace_sigma)
    home = simulate_team_stat_total(mu, pace, rng, dispersion=true_dispersion)
    away = simulate_team_stat_total(mu, pace, rng, dispersion=true_dispersion)
    rows = []
    for g in range(n_games):
        rows.append({"year": 2024, "round": g, "team": "H", "opponent": "A",
                     "is_home": True, "disposals": int(home[g])})
        rows.append({"year": 2024, "round": g, "team": "A", "opponent": "H",
                     "is_home": False, "disposals": int(away[g])})
    return pd.DataFrame(rows), pace, home, away


def test_fit_pace_and_dispersion_recovers_known_values():
    true_sigma, true_r, mu = 0.07, 150.0, 165.0
    player_log, *_ = _synth_team_totals(true_sigma, true_r, mu, n_games=4000, seed=1)
    result = fit_pace_and_dispersion(player_log, stat="disposals")
    assert abs(result["pace_sigma"] - true_sigma) < 0.03
    assert abs(result["team_stat_dispersion"] - true_r) / true_r < 0.35


def test_fit_share_concentration_recovers_known_value():
    true_sigma, true_dispersion, mu = 0.07, 150.0, 165.0
    true_concentration = 200.0
    n_games = 3000
    player_log, pace, home, away = _synth_team_totals(true_sigma, true_dispersion, mu, n_games, seed=2)

    rng = make_rng(3)
    shares = np.array([0.18, 0.16, 0.14])  # mean ~29.7/26.4/23.1, all above ROLE_MID_DISPOSALS_MIN
    alloc = allocate_player_stats(home, shares, rng, concentration=true_concentration)

    rows = []
    for i, share in enumerate(shares):
        for g in range(n_games):
            rows.append({"player": f"Mid{i}", "disposals": int(alloc[i, g])})
    cohort_log = pd.DataFrame(rows)

    pace_result = fit_pace_and_dispersion(player_log, stat="disposals")
    result = fit_share_concentration(
        cohort_log, pace_result["mean"], pace_result["var"], stat="disposals", min_games=15,
    )
    assert result["n_players"] == 3
    assert np.isfinite(result["value"])
    assert abs(result["value"] - true_concentration) / true_concentration < 0.5


def test_fit_score_shot_correlation_recovers_known_rho():
    from afl_bot.backtest.correlations import _shot_correlation_at_rho

    true_rho, mu_shots, dispersion = -0.32, 22.3, 42.5
    n = 200_000
    rng = np.random.default_rng(5)
    z_home = rng.standard_normal(n)
    z_indep = rng.standard_normal(n)
    z_away = true_rho * z_home + np.sqrt(1 - true_rho ** 2) * z_indep
    from scipy.stats import nbinom
    p = dispersion / (dispersion + mu_shots)
    u_home, u_away = norm.cdf(z_home), norm.cdf(z_away)
    home_shots = nbinom.ppf(u_home, dispersion, p)
    away_shots = nbinom.ppf(u_away, dispersion, p)
    goals_h = (home_shots * 0.525).astype(int)
    goals_a = (away_shots * 0.525).astype(int)
    games = pd.DataFrame({
        "hgoals": goals_h, "hbehinds": home_shots - goals_h,
        "agoals": goals_a, "abehinds": away_shots - goals_a,
    })

    result = fit_score_shot_correlation(games, shot_dispersion=dispersion, n_sims=200_000)
    assert not result["bracket_failed"]
    assert abs(result["value"] - true_rho) < 0.05


def test_fit_correlation_params_writes_artifact_and_round_trips(tmp_path):
    rng = np.random.default_rng(7)
    n = 2000
    shots_h = rng.negative_binomial(42, 42 / (42 + 22), n)
    shots_a = rng.negative_binomial(42, 42 / (42 + 22), n)
    goals_h = (shots_h * 0.525).astype(int)
    goals_a = (shots_a * 0.525).astype(int)
    games = pd.DataFrame({
        "year": 2020, "round": np.arange(n) % 23 + 1,
        "hgoals": goals_h, "hbehinds": shots_h - goals_h,
        "agoals": goals_a, "abehinds": shots_a - goals_a,
    })

    player_log, pace, home, away = _synth_team_totals(0.07, 150.0, 165.0, n_games=2000, seed=8)
    player_log["year"] = 2020
    player_log["player"] = player_log["team"]  # only fit_pace_and_dispersion needs team-level rows

    artifact = fit_correlation_params(games, player_log, train_end_year=2020, cache_dir=tmp_path)
    assert (tmp_path / f"{CORRELATION_PARAMS_ARTIFACT}.json").exists()
    assert artifact["train_end_year"] == 2020
    assert "params" in artifact and "diagnostics" in artifact

    loaded = load_fitted_correlation_params(cache_dir=tmp_path)
    assert set(loaded) <= {
        "SHOT_DISPERSION", "PACE_SIGMA", "TEAM_STAT_DISPERSION",
        "SHARE_CONCENTRATION", "SCORE_SHOT_CORRELATION",
    }
    assert all(np.isfinite(v) for v in loaded.values())

    # round-trips through plain JSON
    raw = json.loads((tmp_path / f"{CORRELATION_PARAMS_ARTIFACT}.json").read_text())
    assert raw["params"] == artifact["params"]


def test_load_fitted_correlation_params_missing_artifact_returns_empty(tmp_path):
    assert load_fitted_correlation_params(cache_dir=tmp_path) == {}


def test_fit_pace_and_dispersion_empty_input():
    empty = pd.DataFrame(columns=["year", "round", "team", "opponent", "is_home", "disposals"])
    result = fit_pace_and_dispersion(empty, stat="disposals")
    assert result["n"] == 0
    assert np.isnan(result["pace_sigma"])


def test_fit_share_concentration_no_cohort_returns_nan():
    log = pd.DataFrame({"player": ["Role1"] * 20, "disposals": [10] * 20})
    result = fit_share_concentration(log, team_stat_mean=165.0, team_stat_var=200.0)
    assert result["n_players"] == 0
    assert np.isnan(result["value"])

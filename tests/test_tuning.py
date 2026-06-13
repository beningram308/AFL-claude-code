import numpy as np
import pandas as pd
import pytest

from afl_bot.backtest.tuning import (
    DEFAULT_ELO_PARAM_RANGES,
    elo_objective,
    grid_search_elo,
    optuna_search_elo,
)


def _synthetic_games(n_per_season=40, seasons=(2018, 2019, 2020, 2021), seed=0):
    """A small synthetic fixture where the home team is genuinely stronger, so
    Elo has signal to fit and metrics are well-defined."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 30, "B": 10, "C": -10, "D": -30}
    rows = []
    ut = 0
    for year in seasons:
        for rnd in range(1, n_per_season + 1):
            h, a = rng.choice(teams, size=2, replace=False)
            ut += 1
            margin = strength[h] - strength[a] + 8 + rng.normal(0, 30)  # +8 home edge
            hs, as_ = 90, 90
            if margin >= 0:
                hs = int(90 + margin / 2)
                as_ = int(90 - margin / 2)
            else:
                hs = int(90 + margin / 2)
                as_ = int(90 - margin / 2)
            rows.append({"year": year, "round": rnd, "unixtime": ut,
                         "hteam": h, "ateam": a, "hscore": hs, "ascore": as_})
    return pd.DataFrame(rows)


GAMES = _synthetic_games()


def test_elo_objective_returns_finite_components():
    res = elo_objective(GAMES, k=30.0, home_advantage=8.0)
    assert np.isfinite(res["objective"])
    assert res["objective"] == pytest.approx(res["log_loss"] + 0.01 * res["margin_mae"], rel=1e-6)
    assert res["n_games"] == len(GAMES)


def test_elo_objective_eval_window_filters_games():
    full = elo_objective(GAMES)
    windowed = elo_objective(GAMES, eval_start_year=2021)
    assert windowed["n_games"] < full["n_games"]
    assert windowed["n_games"] == (GAMES["year"] >= 2021).sum()


def test_grid_search_returns_sorted_results():
    grid = {"k": [20.0, 40.0], "home_advantage": [5.0, 10.0]}
    out = grid_search_elo(GAMES, grid, eval_start_year=2021)

    assert len(out) == 4  # 2 x 2 combinations
    assert list(out.columns[:2]) == ["k", "home_advantage"]
    assert {"objective", "log_loss", "margin_mae", "n_games"}.issubset(out.columns)
    # sorted best-first (objective ascending)
    assert out["objective"].is_monotonic_increasing


def test_grid_search_fixed_params_applied():
    grid = {"k": [20.0, 40.0]}
    out = grid_search_elo(GAMES, grid, fixed_params={"update_mode": "mov", "mov_correction": 2.0})
    assert np.isfinite(out["objective"]).all()


def test_optuna_search_runs_and_returns_best():
    optuna = pytest.importorskip("optuna")
    study = optuna_search_elo(GAMES, n_trials=5, eval_start_year=2021, seed=0)
    assert set(study.best_params) == set(DEFAULT_ELO_PARAM_RANGES)
    assert np.isfinite(study.best_value)
    assert "log_loss" in study.best_trial.user_attrs
    assert "margin_mae" in study.best_trial.user_attrs

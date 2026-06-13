"""Section-6 pieces: margin calibration (6.3), per-venue scoring (6.4),
the Elo `fit` artifact (6.2)."""

import numpy as np
import pandas as pd

from afl_bot.backtest.tuning import fit_elo_params, load_fitted_elo_params
from afl_bot.backtest.walkforward import margin_calibration
from afl_bot.models.scoring import venue_scoring_factors


def _synth_games(seasons=(2021, 2022, 2023, 2024), n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 30, "B": 10, "C": -10, "D": -30}
    rows, ut = [], 0
    venues = ["BigGround", "SmallGround"]
    for y in seasons:
        for r in range(1, n_per + 1):
            h, a = rng.choice(teams, 2, replace=False)
            ut += 1
            margin = strength[h] - strength[a] + 8 + rng.normal(0, 30)
            venue = venues[ut % 2]
            bump = 20 if venue == "BigGround" else -20      # venue scoring effect
            hs = int(95 + margin / 2 + bump / 2)
            as_ = int(95 - margin / 2 + bump / 2)
            rows.append({"year": y, "round": r, "unixtime": ut, "hteam": h, "ateam": a,
                         "hscore": hs, "ascore": as_, "venue": venue})
    return pd.DataFrame(rows)


GAMES = _synth_games()


def test_margin_calibration_reports_slope():
    from afl_bot.backtest.walkforward import evaluate_elo
    mc = margin_calibration(evaluate_elo(GAMES))
    assert mc["n"] == len(GAMES)
    assert 0.3 < mc["slope"] < 2.5          # finite, sensible
    assert 0.0 <= mc["r2"] <= 1.0


def test_venue_scoring_factors_orders_high_low():
    factors = venue_scoring_factors(GAMES)
    assert factors["BigGround"] > 1.0 > factors["SmallGround"]
    assert abs(np.mean(list(factors.values())) - 1.0) < 0.05   # centred on league


def test_venue_scoring_factors_shrinks_small_samples():
    g = GAMES.copy()
    # a one-off venue with an extreme total should stay near 1.0 (shrunk)
    g = pd.concat([g, pd.DataFrame([{"year": 2024, "round": 99, "unixtime": 9e6,
                                     "hteam": "A", "ateam": "B", "hscore": 200, "ascore": 200,
                                     "venue": "Freak"}])], ignore_index=True)
    factors = venue_scoring_factors(g, strength=30.0)
    assert factors["Freak"] < 1.3            # heavily shrunk despite a 400-point game


def test_fit_elo_params_writes_artifact_and_loads(tmp_path):
    art = fit_elo_params(GAMES, train_end_year=2022, eval_start_year=2023, cache_dir=tmp_path)
    assert set(art["params"]) == {"k", "season_carryover", "home_advantage"}
    assert "log_loss" in art["metrics"] and art["search"] == "grid"
    assert (tmp_path / "elo_params.json").exists()
    assert load_fitted_elo_params(cache_dir=tmp_path) == art["params"]


def test_load_fitted_elo_params_absent_returns_empty(tmp_path):
    assert load_fitted_elo_params(cache_dir=tmp_path) == {}

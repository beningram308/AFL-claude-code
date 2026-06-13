"""
Full-pipeline CI backtest (plan §5.3): a golden-file walk-forward metric check
so model changes surface their accuracy delta, plus distribution-sanity tests
on the Monte Carlo sim. Both run on deterministic, seeded data — no network.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from afl_bot.backtest.props import walk_forward_prop_predictions
from afl_bot.backtest.walkforward import brier_score, evaluate_elo, log_loss, season_by_season_report
from afl_bot.data.player_stats import synthetic_player_log
from afl_bot.sim.engine import Team, make_rng, simulate_match

GOLDEN = Path(__file__).parent / "golden" / "backtest_metrics.json"


def _synth_season(seasons=(2020, 2021, 2022, 2023), n_per=90, seed=42):
    """Deterministic season: persistent team strengths + Normal margin noise so
    the Elo backtest has real signal and reproducible metrics."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strength = {t: s for t, s in zip(teams, np.linspace(45, -45, len(teams)))}
    rows = []
    ut = 0
    for year in seasons:
        for rnd in range(1, n_per + 1):
            h, a = rng.choice(teams, 2, replace=False)
            ut += 1
            margin = strength[h] - strength[a] + 8 + rng.normal(0, 36)
            rows.append({"year": year, "round": rnd, "unixtime": ut, "hteam": h,
                         "ateam": a, "hscore": int(90 + margin / 2), "ascore": int(90 - margin / 2)})
    return pd.DataFrame(rows)


def _weighted(rep, col):
    return float((rep[col] * rep["n_games"]).sum() / rep["n_games"].sum())


def test_golden_backtest_metrics():
    rep = season_by_season_report(evaluate_elo(_synth_season()))
    golden = json.loads(GOLDEN.read_text())["overall"]

    assert int(rep["n_games"].sum()) == golden["n_games"]
    # Tolerances are wide enough to ignore numerical noise but tight enough that
    # a real model change trips the test and shows its delta (plan §5.3).
    assert abs(_weighted(rep, "log_loss") - golden["log_loss"]) < 0.02, "log loss drifted"
    assert abs(_weighted(rep, "brier") - golden["brier"]) < 0.02, "brier drifted"
    assert abs(_weighted(rep, "margin_mae") - golden["margin_mae"]) < 1.0, "margin MAE drifted"


def test_golden_prop_backtest_metrics():
    """Walk-forward prop backtest metrics on a deterministic synthetic log
    (round-2 §2.4) — a prop-model change surfaces its calibration delta in CI."""
    rows, ut = [], 0
    for year in (2021, 2022, 2023, 2024):
        for rnd in range(1, 13):
            ut += 1
            rows.append({"year": year, "round": rnd, "unixtime": ut, "hteam": "A", "ateam": "B",
                         "hscore": 90, "ascore": 80, "hgoals": 13, "agoals": 11})
    log = synthetic_player_log(pd.DataFrame(rows), players_per_team=12, seed=7)
    preds = walk_forward_prop_predictions(log, eval_start_year=2023)
    golden = json.loads(GOLDEN.read_text())["props"]

    assert len(preds) == golden["n"]
    p = preds["prob"].to_numpy()
    a = preds["actual"].to_numpy(dtype=float)
    assert abs(log_loss(p, a) - golden["log_loss"]) < 0.02, "prop log loss drifted"
    assert abs(brier_score(p, a) - golden["brier"]) < 0.02, "prop brier drifted"


def test_sim_margin_total_distribution_sanity():
    """The scoring-shots sim's margin/total spread must stay in the empirically
    calibrated band (plan §2.1/§2.2/§3.3, §5.3 distribution sanity)."""
    out = simulate_match(
        Team("H", True), Team("A"), mu_margin=0.0, mu_total=162.0,
        home_accuracy=0.525, away_accuracy=0.525, n=200_000, rng=make_rng(),
    )
    margin_std = out["margin"].std()
    total_std = (out["home_pts"] + out["away_pts"]).std()

    assert abs(out["margin"].mean()) < 1.5            # symmetric matchup -> ~0
    assert 36.0 < margin_std < 42.0                   # empirical ~39.3
    assert 29.0 < total_std < 34.0                    # empirical ~31.4
    assert out["draw"].mean() > 0.0                   # real draw probability
    # integer scores
    assert np.allclose(out["home_pts"], np.round(out["home_pts"]))

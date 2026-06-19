"""Walk-forward SGM-ladder backtest (model-upgrade audit Phase 1.1) — both run
on deterministic, seeded data with the real player-stat fetchers monkeypatched
out, so no network is needed."""

import numpy as np
import pandas as pd
import pytest

from afl_bot.backtest.multis import (
    multi_calibration_report,
    multi_reliability_curve,
    walk_forward_multi_predictions,
)
from afl_bot.data.player_stats import synthetic_player_log

TEAMS = [f"Team{i}" for i in range(8)]
N_ROUNDS = 10
SEASONS = (2021, 2022, 2023, 2024)


def _synth_games(seed=11):
    """Multi-season fixture list (4 matches/round, round-robin pairing) with
    the full schema `team_scoring_profiles` / `team_shot_accuracy_profiles` /
    `venue_scoring_factors` / `fit_team_hga` need: hscore/ascore, hgoals/
    agoals/hbehinds/abehinds, venue, year/round/unixtime."""
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(TEAMS, np.linspace(40, -40, len(TEAMS)))}
    venues = ["VenueA", "VenueB"]
    rows, ut = [], 0
    for year in SEASONS:
        for rnd in range(1, N_ROUNDS + 1):
            order = rng.permutation(TEAMS)
            for i in range(0, len(order), 2):
                h, a = order[i], order[i + 1]
                ut += 1
                margin = strength[h] - strength[a] + 6 + rng.normal(0, 30)
                total = max(80, 160 + rng.normal(0, 12))
                hscore = int(round((total + margin) / 2))
                ascore = int(round((total - margin) / 2))
                hgoals = max(0, hscore // 6)
                hbehinds = max(0, hscore - hgoals * 6)
                agoals = max(0, ascore // 6)
                abehinds = max(0, ascore - agoals * 6)
                rows.append({
                    "year": year, "round": rnd, "unixtime": ut, "hteam": h, "ateam": a,
                    "venue": rng.choice(venues), "hscore": hscore, "ascore": ascore,
                    "hgoals": hgoals, "hbehinds": hbehinds, "agoals": agoals, "abehinds": abehinds,
                })
    return pd.DataFrame(rows)


def _fryzigg_raw_from_log(player_log: pd.DataFrame) -> pd.DataFrame:
    """Reshape a synthetic player log into the raw Fryzigg schema
    `_fetch_actual_player_log` expects, so the backtest's "actual outcome"
    lookup is self-consistent with the same synthetic data used to build the
    predictions."""
    parts = player_log["player"].str.rpartition(" ")
    first, last = parts[0], parts[2]
    return pd.DataFrame({
        "match_date": pd.to_datetime(player_log["year"], format="%Y"),
        "player_first_name": first,
        "player_last_name": last,
        "match_round": player_log["round"].astype(str),
        "disposals": player_log["disposals"],
        "goals": player_log["goals"],
        "marks": player_log["marks"],
        "tackles": player_log["tackles"],
    })


@pytest.fixture
def synth_world(monkeypatch):
    games = _synth_games()
    player_log = synthetic_player_log(games, players_per_team=12, seed=5)
    fryzigg_raw = _fryzigg_raw_from_log(player_log)
    monkeypatch.setattr(
        "afl_bot.data.fryzigg.fetch_fryzigg_player_stats", lambda: fryzigg_raw,
    )
    return games, player_log


def test_walk_forward_multi_predictions_shape_and_bounds(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
    )
    assert not preds.empty
    assert {"year", "round", "match_id", "legs", "joint_prob", "all_hit", "fair_odds"} <= set(preds.columns)
    assert set(preds["round"]) <= {8, 9}
    assert preds["joint_prob"].between(0.0, 1.0).all()
    assert set(preds["all_hit"].unique()) <= {0, 1}
    # Each graded match contributes at most one rung per target-odds band.
    assert preds.groupby("match_id").size().max() <= 3


def test_walk_forward_multi_predictions_no_leakage_across_rounds(synth_world):
    """Predictions for round 8 must not change when later rounds' results are
    present in the input `games`/`player_log` — the function truncates
    internally, so round 9+ rows are inert for round 8's prediction."""
    games, player_log = synth_world
    full_preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    truncated_games = games[~((games["year"] == 2024) & (games["round"] > 8))]
    truncated_log = player_log[~((player_log["year"] == 2024) & (player_log["round"] > 8))]
    truncated_preds = walk_forward_multi_predictions(
        truncated_games, truncated_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    pd.testing.assert_frame_equal(
        full_preds.sort_values("match_id").reset_index(drop=True),
        truncated_preds.sort_values("match_id").reset_index(drop=True),
    )


def test_multi_calibration_report_and_reliability_curve(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[7, 8, 9, 10], n_sims=3000, seed=2,
    )
    report = multi_calibration_report(preds)
    assert report["n"] == len(preds)
    assert np.isfinite(report["log_loss"])
    assert np.isfinite(report["brier"])
    assert 0.0 <= report["hit_rate"] <= 1.0

    curve = multi_reliability_curve(preds, n_bins=5)
    assert {"mean_pred", "actual_rate", "n"} <= set(curve.columns)
    assert curve["n"].sum() == len(preds)


def test_multi_calibration_report_empty_input():
    report = multi_calibration_report(pd.DataFrame())
    assert report["n"] == 0
    assert np.isnan(report["log_loss"])


def test_walk_forward_multi_predictions_empty_when_round_not_in_games(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[999], n_sims=2000,
    )
    assert preds.empty

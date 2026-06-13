"""Walk-forward prop backtest + per-market calibration (round-2 §2)."""

import numpy as np
import pandas as pd

from afl_bot.backtest.ensemble import IsotonicCalibrator
from afl_bot.backtest.props import (
    fit_prop_calibrators,
    load_or_fit_prop_calibrators,
    prop_calibration_report,
    prop_prob,
    walk_forward_prop_predictions,
)
from afl_bot.backtest.walkforward import log_loss
from afl_bot.data.player_stats import synthetic_player_log


def _synth_games(seasons=(2021, 2022, 2023, 2024), rounds=12):
    rows, ut = [], 0
    for y in seasons:
        for r in range(1, rounds + 1):
            ut += 1
            rows.append({"year": y, "round": r, "unixtime": ut, "hteam": "A", "ateam": "B",
                         "hscore": 90, "ascore": 80, "hgoals": 13, "agoals": 11})
    return pd.DataFrame(rows)


LOG = synthetic_player_log(_synth_games(), players_per_team=12, seed=7)


def test_prop_prob_monotone_and_bounded():
    p15 = prop_prob(20.0, 10.0, 15)
    p25 = prop_prob(20.0, 10.0, 25)
    assert 0.0 <= p25 <= p15 <= 1.0          # higher line -> lower probability
    assert prop_prob(20.0, 10.0, 0) == 1.0   # P(>=0) == 1


def test_walk_forward_predictions_shape_and_walk_forward():
    preds = walk_forward_prop_predictions(LOG, eval_start_year=2023)
    assert not preds.empty
    assert {"year", "round", "player", "stat", "line", "prob", "actual"} <= set(preds.columns)
    assert (preds["year"] >= 2023).all()                 # only the eval window
    assert preds["prob"].between(0, 1).all()
    assert set(preds["actual"].unique()) <= {0, 1}


def test_prop_calibration_report_per_market():
    rep = prop_calibration_report(walk_forward_prop_predictions(LOG, eval_start_year=2023))
    assert {"disposals", "goals", "marks", "tackles"} <= set(rep)
    for m in rep.values():
        assert m["n"] > 0 and m["log_loss"] > 0 and 0 <= m["hit_rate"] <= 1


def test_fit_prop_calibrators_improve_in_sample():
    preds = walk_forward_prop_predictions(LOG, eval_start_year=2023)
    cals = fit_prop_calibrators(preds)
    for stat, grp in preds.groupby("stat"):
        raw = log_loss(grp["prob"].to_numpy(), grp["actual"].to_numpy(float))
        cal = log_loss(cals[stat].predict(grp["prob"].to_numpy()), grp["actual"].to_numpy(float))
        assert cal <= raw + 1e-9             # isotonic can't worsen in-sample log loss


def test_isotonic_calibrator_json_roundtrip():
    cal = IsotonicCalibrator().fit([0.2, 0.5, 0.8, 0.9], [0, 1, 1, 1])
    back = IsotonicCalibrator.from_dict(cal.to_dict())
    assert np.allclose(back.predict([0.3, 0.7]), cal.predict([0.3, 0.7]))
    assert IsotonicCalibrator.from_dict({"x": [], "y": []}).predict([0.4])[0] == 0.4


def test_load_or_fit_prop_calibrators_caches(tmp_path):
    cals = load_or_fit_prop_calibrators(LOG, eval_start_year=2023, cache_dir=tmp_path)
    assert cals and (tmp_path / "prop_calibrators.json").exists()
    again = load_or_fit_prop_calibrators(LOG, eval_start_year=2023, cache_dir=tmp_path)
    assert set(again) == set(cals)

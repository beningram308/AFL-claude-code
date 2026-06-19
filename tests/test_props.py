"""Walk-forward prop backtest + per-market calibration (round-2 §2)."""

import json

import numpy as np
import pandas as pd

from afl_bot.backtest.ensemble import IsotonicCalibrator
from afl_bot.backtest.props import (
    CALIBRATOR_CACHE_VERSION,
    apply_prop_calibration,
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


def test_walk_forward_prop_predictions_defaults_to_live_prop_lines():
    """Model-upgrade audit Phase 3.1: one PROP_LINES definition in config.py,
    backtest exactly the lines actually priced live -- guards against a
    second, narrower, stale line set drifting back in (the old
    `backtest/props.py DEFAULT_PROP_LINES` bug this replaced)."""
    from afl_bot.cli import PROP_LINES as cli_prop_lines
    from afl_bot.config import PROP_LINES as config_prop_lines

    assert cli_prop_lines is config_prop_lines
    preds = walk_forward_prop_predictions(LOG, eval_start_year=2023)
    seen_lines = {(stat, line) for stat, line in preds[["stat", "line"]].itertuples(index=False)}
    expected_lines = {(stat, line) for stat, lines in config_prop_lines.items() for line in lines}
    assert seen_lines == expected_lines


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
        calibrated = np.array([apply_prop_calibration(cals, stat, line, p)
                               for p, line in zip(grp["prob"], grp["line"])])
        cal = log_loss(calibrated, grp["actual"].to_numpy(float))
        assert cal <= raw + 1e-9             # isotonic can't worsen in-sample log loss


def test_fit_prop_calibrators_per_line_with_pooled_fallback():
    preds = walk_forward_prop_predictions(LOG, eval_start_year=2023)
    # A low threshold so at least one (stat, line) cell clears it and gets its
    # own curve, while a deliberately huge threshold forces every cell back to
    # the pooled fallback -- both paths of apply_prop_calibration's lookup.
    per_line = fit_prop_calibrators(preds, min_samples=10)
    all_pooled = fit_prop_calibrators(preds, min_samples=10**9)
    assert any(entry["lines"] for entry in per_line.values())
    assert all(entry["lines"] == {} for entry in all_pooled.values())
    # Pooled fallback should match the pooled curve directly for any row.
    stat, line = preds.iloc[0][["stat", "line"]]
    prob = float(preds.iloc[0]["prob"])
    assert apply_prop_calibration(all_pooled, stat, line, prob) == \
        float(all_pooled[stat]["pooled"].predict([prob])[0])


def test_apply_prop_calibration_missing_stat_is_passthrough():
    assert apply_prop_calibration({}, "disposals", 20, 0.42) == 0.42


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
    for stat in cals:
        assert set(again[stat]) == {"pooled", "lines"} == set(cals[stat])
        p = 0.4
        assert again[stat]["pooled"].predict([p])[0] == cals[stat]["pooled"].predict([p])[0]


def test_load_or_fit_prop_calibrators_refits_stale_cache_version(tmp_path):
    path = tmp_path / "prop_calibrators.json"
    path.write_text(json.dumps({"disposals": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}))  # pre-Phase-3.2 flat shape
    cals = load_or_fit_prop_calibrators(LOG, eval_start_year=2023, cache_dir=tmp_path)
    assert cals and all(set(entry) == {"pooled", "lines"} for entry in cals.values())
    data = json.loads(path.read_text())
    assert data["_version"] == CALIBRATOR_CACHE_VERSION

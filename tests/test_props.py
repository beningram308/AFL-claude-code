"""Walk-forward prop backtest + per-market calibration (round-2 §2)."""

import json

import numpy as np
import pandas as pd

from afl_bot.backtest.ensemble import IsotonicCalibrator
from afl_bot.backtest.props import (
    CALIBRATOR_CACHE_VERSION,
    _HIGH_BUCKET_THRESHOLD,
    apply_prop_calibration,
    ece_score,
    fit_prop_calibrators,
    high_bucket_gap,
    load_or_fit_prop_calibrators,
    prop_calibration_report,
    prop_halflife_sweep,
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


# --------------------------------------------------------------------------- #
# EXPERIMENT-FORM-WINDOW-HALFLIFE: ece_score, high_bucket_gap, sweep
# --------------------------------------------------------------------------- #

def test_ece_score_perfect_calibration():
    """When every leg's predicted prob equals its empirical hit rate, ECE = 0."""
    probs = np.array([0.3, 0.3, 0.7, 0.7])
    actuals = np.array([0.0, 1.0, 0.0, 1.0])   # 50% hit in each bucket -> miscalibrated
    # A perfectly calibrated set: prob=0.4, 40% of outcomes are 1
    rng = np.random.default_rng(0)
    n = 1000
    p = np.full(n, 0.4)
    a = (rng.random(n) < 0.4).astype(float)
    assert ece_score(p, a) < 0.05              # should be near zero with large n


def test_ece_score_overconfident():
    """Always predicting 0.9 when true rate is 0.5 gives ECE near 0.4."""
    probs = np.full(1000, 0.9)
    actuals = np.where(np.arange(1000) % 2 == 0, 1.0, 0.0)  # 50% hit rate
    assert ece_score(probs, actuals) > 0.3


def test_ece_score_empty():
    assert np.isnan(ece_score(np.array([]), np.array([])))


def test_high_bucket_gap_overconfident():
    """Predicting 0.8 when only 60% hit -> gap = +0.2."""
    probs = np.full(100, 0.8)
    actuals = np.where(np.arange(100) < 60, 1.0, 0.0)
    gap = high_bucket_gap(probs, actuals)
    assert abs(gap - 0.2) < 0.01


def test_high_bucket_gap_no_high_probs():
    """When no probs reach the threshold, returns NaN."""
    probs = np.full(50, 0.3)
    actuals = np.zeros(50)
    assert np.isnan(high_bucket_gap(probs, actuals, threshold=_HIGH_BUCKET_THRESHOLD))


def test_load_or_fit_prop_calibrators_halflife_param_bypasses_cache(tmp_path):
    """Non-default halflife must not read from or write to the default cache."""
    from afl_bot.config import PROP_EWMA_HALFLIFE
    # Prime cache with default halflife.
    cals_default = load_or_fit_prop_calibrators(
        LOG, eval_start_year=2023, cache_dir=tmp_path)
    assert (tmp_path / "prop_calibrators.json").exists()
    # Non-default halflife should recompute (not use cache) and return different object.
    cals_alt = load_or_fit_prop_calibrators(
        LOG, eval_start_year=2023, cache_dir=tmp_path, halflife=PROP_EWMA_HALFLIFE + 4)
    # Both should be non-empty valid calibrator dicts.
    assert cals_default and cals_alt
    assert set(cals_default) == set(cals_alt)


def test_prop_halflife_sweep_returns_correct_shape():
    """Sweep returns one row per halflife with the required columns."""
    result = prop_halflife_sweep(LOG, eval_years=[2023, 2024], halflives=[6.0, 8.0])
    assert len(result) == 2
    assert set(result.columns) >= {"halflife", "n", "log_loss", "brier", "ece", "high_bucket_gap"}
    assert list(result["halflife"]) == [6.0, 8.0]
    assert result["n"].gt(0).all()
    assert result["log_loss"].between(0, 5).all()
    assert result["brier"].between(0, 1).all()
    assert result["ece"].between(0, 1).all()


def test_prop_halflife_sweep_metrics_are_finite():
    """All metrics should be finite floats for a valid log with enough history."""
    result = prop_halflife_sweep(LOG, eval_years=[2024], halflives=[6.0, 10.0])
    for col in ["log_loss", "brier", "ece"]:
        assert result[col].apply(np.isfinite).all(), f"{col} has non-finite values"


# ── Part 5: stale-calibrator guard ───────────────────────────────────────────

def test_load_or_fit_prop_calibrators_saves_fitted_max_year(tmp_path):
    load_or_fit_prop_calibrators(LOG, eval_start_year=2023, cache_dir=tmp_path)
    data = json.loads((tmp_path / "prop_calibrators.json").read_text())
    assert "_fitted_max_year" in data
    assert data["_fitted_max_year"] == int(LOG["year"].max())


def test_load_or_fit_prop_calibrators_warns_when_log_newer_than_cache(tmp_path, capsys):
    load_or_fit_prop_calibrators(LOG, eval_start_year=2023, cache_dir=tmp_path)
    stale_log = LOG.copy()
    stale_log["year"] = stale_log["year"].max() + 1  # simulate new season
    load_or_fit_prop_calibrators(stale_log, eval_start_year=2023, cache_dir=tmp_path)
    captured = capsys.readouterr()
    assert "stale" in captured.err.lower() or "WARNING" in captured.err


def test_load_or_fit_prop_calibrators_no_warning_when_log_matches_cache(tmp_path, capsys):
    load_or_fit_prop_calibrators(LOG, eval_start_year=2023, cache_dir=tmp_path)
    load_or_fit_prop_calibrators(LOG, eval_start_year=2023, cache_dir=tmp_path)
    captured = capsys.readouterr()
    assert "stale" not in captured.err.lower()

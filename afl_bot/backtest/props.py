"""
Walk-forward player-prop backtest + per-market calibration (round-2 §2).

Player props were never scored against history: the sim's stated "73% for 20+
disposals" had no evidence behind it, and per-leg biases compound across a
multi. This module rebuilds each player's rate AS OF every historical round
(EWMA shifted to use only prior games — genuine walk-forward, no leakage),
prices the standard lines with a Negative-Binomial marginal, and records the
predicted probability vs whether the player actually hit the line.

From those predictions we report calibration (per market type) and fit an
``IsotonicCalibrator`` per stat, applied in the CLI before legs are classified
and staked. Marginals come from the same shrunk mean + role-pooled dispersion
the live model centres on, so the calibrator corrects the dominant mean bias
that the full Monte-Carlo prop price shares.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from afl_bot.backtest.ensemble import IsotonicCalibrator
from afl_bot.backtest.walkforward import brier_score, log_loss
from afl_bot.config import (
    CACHE_DIR,
    PROP_CALIBRATION_MIN_SAMPLES,
    PROP_EWMA_HALFLIFE,
    PROP_LINES,
    PROP_PRIOR_STRENGTH,
)

_HIGH_BUCKET_THRESHOLD = 0.65  # legs predicted >= this are the "high-confidence" bucket
from afl_bot.models.priors import (
    GLOBAL_ROLE,
    classify_roles,
    estimate_dispersion_hierarchical,
    role_rate_priors,
    shrink,
)

CALIBRATOR_CACHE = "prop_calibrators"
CALIBRATOR_CACHE_VERSION = 2   # bumped for the (stat, line) keying (model-upgrade Phase 3.2)


def prop_prob(mean, dispersion, line):
    """P(stat >= line) under a Negative Binomial with the given mean and
    dispersion r. Vectorised over ``mean`` arrays."""
    mean = np.clip(np.asarray(mean, dtype=float), 1e-6, None)
    r = float(dispersion)
    p = r / (r + mean)
    return nbinom.sf(line - 1, r, p)        # P(X > line-1) = P(X >= line)


def walk_forward_prop_predictions(
    log: pd.DataFrame, *, eval_start_year: int, stats=None, lines=None,
    halflife: float = PROP_EWMA_HALFLIFE, strength: float = PROP_PRIOR_STRENGTH,
) -> pd.DataFrame:
    """One row per (player-game, stat, line) in seasons >= ``eval_start_year``:
    the as-of predicted probability and the actual hit/miss. Profiles use only
    games before each game (EWMA shifted by one); dispersion + role priors are
    fit on pre-eval seasons to keep the test honest.

    ``lines`` defaults to the live ``PROP_LINES`` (model-upgrade audit Phase
    3.1 — backtest exactly the lines actually priced; this used to default to
    a separate, narrower, stale ``DEFAULT_PROP_LINES`` that drifted out of
    sync with what `run-round`/`round-report` price)."""
    lines = lines or PROP_LINES
    stats = stats or list(lines)
    log = log.sort_values(["player", "year", "round", "unixtime"]).reset_index(drop=True)

    roles = classify_roles(log)
    pre = log[log["year"] < eval_start_year]
    if pre.empty:
        return pd.DataFrame(columns=["year", "round", "player", "stat", "line", "prob", "actual"])
    role_of = log["player"].map(lambda p: roles.get(p, GLOBAL_ROLE))
    prior_count = log.groupby("player").cumcount().to_numpy()          # games before this one
    eval_mask = (log["year"] >= eval_start_year).to_numpy()

    frames = []
    for stat in stats:
        disp = estimate_dispersion_hierarchical(pre, stat, roles)
        priors = role_rate_priors(pre, stat, roles)
        prior_mean = role_of.map(lambda r: priors.get(r, priors[GLOBAL_ROLE])["mean_prior"]).to_numpy()
        r_per_row = log["player"].map(lambda p: disp.get(p, np.mean(list(disp.values())) if disp else 4.0)).to_numpy()

        ewma_asof = (
            log.groupby("player")[stat]
            .transform(lambda s: s.ewm(halflife=halflife, adjust=True).mean().shift(1))
            .to_numpy()
        )
        n = np.maximum(prior_count, 0.0)
        means = (n * np.nan_to_num(ewma_asof) + strength * prior_mean) / (n + strength)
        actual_vals = log[stat].to_numpy()

        valid = eval_mask & np.isfinite(ewma_asof)
        if not valid.any():
            continue
        for line in lines[stat]:
            # nbinom.sf is vectorised over both the per-row dispersion and mean.
            p = r_per_row[valid] / (r_per_row[valid] + np.clip(means[valid], 1e-6, None))
            probs = nbinom.sf(line - 1, r_per_row[valid], p)
            frames.append(pd.DataFrame({
                "year": log["year"].to_numpy()[valid],
                "round": log["round"].to_numpy()[valid],
                "player": log["player"].to_numpy()[valid],
                "stat": stat,
                "line": line,
                "prob": probs,
                "actual": (actual_vals[valid] >= line).astype(int),
            }))

    if not frames:
        return pd.DataFrame(columns=["year", "round", "player", "stat", "line", "prob", "actual"])
    return pd.concat(frames, ignore_index=True)


def prop_calibration_report(preds: pd.DataFrame) -> dict:
    """Per-market (stat) log loss, Brier, n, mean predicted prob and actual hit
    rate — the headline calibration numbers (round-2 §2.2)."""
    report = {}
    for stat, grp in preds.groupby("stat"):
        p = grp["prob"].to_numpy()
        a = grp["actual"].to_numpy(dtype=float)
        report[stat] = {
            "n": len(grp),
            "log_loss": log_loss(p, a),
            "brier": brier_score(p, a),
            "mean_pred": float(p.mean()),
            "hit_rate": float(a.mean()),
        }
    return report


def fit_prop_calibrators(
    preds: pd.DataFrame, *, min_samples: int = PROP_CALIBRATION_MIN_SAMPLES,
) -> dict[str, dict]:
    """Per-``(stat, line)`` ``IsotonicCalibrator``, with a pooled per-stat
    fallback for ``(stat, line)`` cells with fewer than ``min_samples``
    walk-forward predictions (model-upgrade audit Phase 3.2 — a single curve
    per stat pools 15+ and 35+ disposals despite very different base rates,
    starving the tail lines that dominate multi value of their own fit).
    Returns ``{stat: {"pooled": IsotonicCalibrator, "lines": {line: IsotonicCalibrator}}}``;
    ``apply_prop_calibration`` does the lookup-with-fallback."""
    calibrators: dict[str, dict] = {}
    for stat, grp in preds.groupby("stat"):
        pooled = IsotonicCalibrator().fit(grp["prob"].to_numpy(), grp["actual"].to_numpy(dtype=float))
        by_line: dict[float, IsotonicCalibrator] = {}
        for line, line_grp in grp.groupby("line"):
            if len(line_grp) < min_samples:
                continue
            by_line[float(line)] = IsotonicCalibrator().fit(
                line_grp["prob"].to_numpy(), line_grp["actual"].to_numpy(dtype=float))
        calibrators[stat] = {"pooled": pooled, "lines": by_line}
    return calibrators


def apply_prop_calibration(calibrators: dict[str, dict], stat: str, line, prob: float) -> float:
    """Calibrate one leg's probability: look up the ``(stat, line)`` curve,
    falling back to the pooled per-stat curve when this exact line wasn't fit
    (too few samples) or has no entry; returns ``prob`` unchanged if ``stat``
    has no calibrator at all (e.g. calibrators is ``{}``)."""
    entry = calibrators.get(stat)
    if entry is None:
        return prob
    cal = entry["lines"].get(float(line), entry["pooled"])
    return float(cal.predict([prob])[0])


def load_or_fit_prop_calibrators(log: pd.DataFrame, *, eval_start_year: int,
                                 cache_dir=CACHE_DIR, force_refresh: bool = False,
                                 cache_name: str = CALIBRATOR_CACHE,
                                 predictions: pd.DataFrame | None = None,
                                 halflife: float = PROP_EWMA_HALFLIFE) -> dict[str, dict]:
    """Cached per-``(stat, line)`` calibrators: load the JSON artifact if
    present, else run the walk-forward backtest, fit, and persist. Returns
    ``{}`` if there isn't enough history to backtest (callers then skip
    calibration via ``apply_prop_calibration``'s no-op fallback).

    ``predictions`` lets a caller supply an already-computed (year, round,
    player, stat, line, prob, actual) frame to fit on instead of this
    module's own ``walk_forward_prop_predictions`` proxy marginal -- e.g.
    `afl_bot.backtest.multis.walk_forward_sim_prop_predictions`'s real-sim
    predictions (model-upgrade audit Phase 3.1). ``cache_name`` then lets
    that alternate source cache to its own artifact instead of colliding with
    the proxy-marginal ``prop_calibrators.json``.

    ``halflife`` overrides ``PROP_EWMA_HALFLIFE`` for the walk-forward pass
    (sweep experiments). Non-default values bypass the on-disk cache to
    avoid returning stale calibrators from a previous run with a different
    halflife."""
    # Non-default halflife must not hit the default cache (wrong calibrators).
    effective_refresh = force_refresh or (halflife != PROP_EWMA_HALFLIFE)
    path = cache_dir / f"{cache_name}.json"
    if path.exists() and not effective_refresh:
        data = json.loads(path.read_text())
        if data.get("_version") == CALIBRATOR_CACHE_VERSION:
            fitted_max_year = data.get("_fitted_max_year")
            if fitted_max_year is not None and "year" in log.columns:
                log_max_year = int(log["year"].max())
                if log_max_year > fitted_max_year:
                    import sys
                    print(
                        f"WARNING: {path.name} was fitted through {fitted_max_year} but "
                        f"player log now has data through {log_max_year}. "
                        f"Run `python -m afl_bot.cli fit` (or rerun round-report) to refit.",
                        file=sys.stderr,
                    )
            return {
                stat: {
                    "pooled": IsotonicCalibrator.from_dict(entry["pooled"]),
                    "lines": {float(line): IsotonicCalibrator.from_dict(d)
                              for line, d in entry["lines"].items()},
                }
                for stat, entry in data.items() if stat not in ("_version", "_fitted_max_year")
            }
        # Stale pre-Phase-3.2 flat {stat: calibrator} cache -- refit below.

    preds = predictions if predictions is not None else walk_forward_prop_predictions(
        log, eval_start_year=eval_start_year, halflife=halflife)
    if preds.empty:
        return {}
    calibrators = fit_prop_calibrators(preds)
    fitted_max_year = int(preds["year"].max()) if not preds.empty and "year" in preds.columns else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_version": CALIBRATOR_CACHE_VERSION,
        "_fitted_max_year": fitted_max_year,
        **{
            stat: {
                "pooled": entry["pooled"].to_dict(),
                "lines": {str(line): c.to_dict() for line, c in entry["lines"].items()},
            }
            for stat, entry in calibrators.items()
        },
    }))
    return calibrators


def ece_score(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean |mean_pred - actual_rate|
    across equal-width probability buckets."""
    probs = np.asarray(probs, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if len(probs) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins, right=True)
    ece = 0.0
    n = len(probs)
    for b in range(1, n_bins + 1):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(probs[mask].mean() - actuals[mask].mean())
    return float(ece)


def high_bucket_gap(probs: np.ndarray, actuals: np.ndarray,
                    threshold: float = _HIGH_BUCKET_THRESHOLD) -> float:
    """Overconfidence gap in the high-probability bucket (legs >= ``threshold``):
    mean(pred) − mean(actual). Positive = overconfident, negative = underconfident.
    Returns NaN when no legs reach the threshold."""
    probs = np.asarray(probs, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = probs >= threshold
    if not mask.any():
        return float("nan")
    return float(probs[mask].mean() - actuals[mask].mean())


def prop_halflife_sweep(
    log: pd.DataFrame, *,
    eval_years: list[int],
    halflives: list[float],
    cal_lookback: int = 4,
    n_bins: int = 10,
    high_threshold: float = _HIGH_BUCKET_THRESHOLD,
) -> pd.DataFrame:
    """Sweep ``PROP_EWMA_HALFLIFE`` candidates and report calibrated prop
    metrics across ``eval_years`` (each year is strictly OOS).

    For each (halflife, eval_year):
    - Fit calibrators on walk-forward prop predictions from the ``cal_lookback``
      seasons immediately before ``eval_year`` (no data leakage).
    - Predict on ``eval_year`` and apply those calibrators.
    - Report log_loss / Brier / ECE / high_bucket_gap aggregated across all
      eval_years so the comparison is on pooled OOS data.

    Returns one row per halflife with columns:
    ``halflife, n, log_loss, brier, ece, high_bucket_gap``."""
    rows = []
    for hl in halflives:
        all_probs: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []
        for eval_year in sorted(eval_years):
            train_log = log[log["year"] < eval_year]
            cal_start = eval_year - cal_lookback
            cal_preds = walk_forward_prop_predictions(
                train_log, eval_start_year=cal_start, halflife=hl)
            cals = fit_prop_calibrators(cal_preds) if not cal_preds.empty else {}

            eval_preds = walk_forward_prop_predictions(
                log, eval_start_year=eval_year, halflife=hl)
            eval_preds = (eval_preds[eval_preds["year"] == eval_year]
                          .reset_index(drop=True))
            if eval_preds.empty:
                continue

            # Vectorised calibration per (stat, line) group.
            # reset_index(drop=True) above ensures grp.index is positional,
            # matching cal_probs's 0-based numpy indexing.
            cal_probs = eval_preds["prob"].to_numpy(dtype=float).copy()
            for (stat, line), grp in eval_preds.groupby(["stat", "line"]):
                entry = cals.get(stat)
                if entry is None:
                    continue
                cal = entry["lines"].get(float(line), entry["pooled"])
                cal_probs[grp.index] = cal.predict(grp["prob"].to_numpy())

            all_probs.append(cal_probs)
            all_acts.append(eval_preds["actual"].to_numpy(dtype=float))

        if not all_probs:
            continue
        p = np.concatenate(all_probs)
        a = np.concatenate(all_acts)
        rows.append({
            "halflife": float(hl),
            "n": int(len(p)),
            "log_loss": log_loss(p, a),
            "brier": brier_score(p, a),
            "ece": ece_score(p, a, n_bins),
            "high_bucket_gap": high_bucket_gap(p, a, high_threshold),
        })
    return pd.DataFrame(rows)

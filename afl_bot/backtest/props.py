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
from afl_bot.config import CACHE_DIR, PROP_EWMA_HALFLIFE, PROP_PRIOR_STRENGTH
from afl_bot.models.priors import (
    GLOBAL_ROLE,
    classify_roles,
    estimate_dispersion_hierarchical,
    role_rate_priors,
    shrink,
)

DEFAULT_PROP_LINES = {
    "disposals": [15, 20, 25],
    "goals": [1, 2],
    "marks": [4, 6],
    "tackles": [3, 5],
}
CALIBRATOR_CACHE = "prop_calibrators"


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
    fit on pre-eval seasons to keep the test honest."""
    lines = lines or DEFAULT_PROP_LINES
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


def fit_prop_calibrators(preds: pd.DataFrame) -> dict[str, IsotonicCalibrator]:
    """An ``IsotonicCalibrator`` per stat, fit on the walk-forward predictions
    (round-2 §2.3)."""
    return {
        stat: IsotonicCalibrator().fit(grp["prob"].to_numpy(), grp["actual"].to_numpy(dtype=float))
        for stat, grp in preds.groupby("stat")
    }


def load_or_fit_prop_calibrators(log: pd.DataFrame, *, eval_start_year: int,
                                 cache_dir=CACHE_DIR, force_refresh: bool = False) -> dict[str, IsotonicCalibrator]:
    """Cached per-stat calibrators: load the JSON artifact if present, else run
    the walk-forward backtest, fit, and persist. Returns ``{}`` if there isn't
    enough history to backtest (callers then skip calibration)."""
    path = cache_dir / f"{CALIBRATOR_CACHE}.json"
    if path.exists() and not force_refresh:
        data = json.loads(path.read_text())
        return {stat: IsotonicCalibrator.from_dict(d) for stat, d in data.items()}

    preds = walk_forward_prop_predictions(log, eval_start_year=eval_start_year)
    if preds.empty:
        return {}
    calibrators = fit_prop_calibrators(preds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({stat: c.to_dict() for stat, c in calibrators.items()}))
    return calibrators

"""
Market-blend ensemble + probability calibration (plan §3.5).

The market is the best single predictor, so a model probability should be
blended toward it (and the Squiggle crowd-model consensus) with weights fit on
out-of-sample log loss, and the edge taken on the *blended* probability — which
kills the false edges a raw model throws off. Separately, raw model
probabilities are calibrated (isotonic regression on walk-forward predictions)
so a stated "85% anchor" actually lands ~85% of the time.

Pieces:
  * ``IsotonicCalibrator`` — monotone, non-parametric calibration via PAVA
    (no sklearn dependency).
  * ``fit_blend_weights`` / ``blend_probabilities`` — a log-loss-optimal convex
    blend (weights on the simplex) of several probability columns.
  * ``squiggle_consensus`` — per-game crowd-model home-win probability from the
    Squiggle tips feed.
  * ``MarketBlend`` + ``fit_market_blend`` / ``ensemble_report`` — fit the
    calibrator + blend weights on a training window and (for the report) score
    every signal and the blend on a held-out window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from afl_bot.backtest.walkforward import devig_h2h_probs, evaluate_elo, log_loss

SIGNAL_ORDER = ("model", "market", "squiggle")


def _clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


# --------------------------------------------------------------------------- #
# Isotonic calibration (Pool Adjacent Violators)
# --------------------------------------------------------------------------- #
def _pava(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: nearest non-decreasing fit to ``y`` (equal
    weights), returned at full length."""
    vy: list[float] = []
    vw: list[float] = []
    vn: list[int] = []
    for val in np.asarray(y, dtype=float):
        cy, cw, cn = float(val), 1.0, 1
        while vy and vy[-1] > cy:
            py, pw, pn = vy.pop(), vw.pop(), vn.pop()
            cy = (pw * py + cw * cy) / (pw + cw)
            cw += pw
            cn += pn
        vy.append(cy)
        vw.append(cw)
        vn.append(cn)
    out = np.empty(sum(vn))
    idx = 0
    for val, cnt in zip(vy, vn):
        out[idx:idx + cnt] = val
        idx += cnt
    return out


@dataclass
class IsotonicCalibrator:
    """Monotone probability calibration. ``fit`` learns a non-decreasing map
    from predicted to empirical probability on (pred, outcome) pairs; ``predict``
    interpolates it (clamped to the trained range)."""
    x_: np.ndarray | None = None
    y_: np.ndarray | None = None

    def fit(self, probs, outcomes) -> "IsotonicCalibrator":
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        order = np.argsort(probs, kind="mergesort")
        self.x_ = probs[order]
        self.y_ = _pava(outcomes[order])
        return self

    def predict(self, probs) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if self.x_ is None or len(self.x_) == 0:
            return probs
        return np.clip(np.interp(probs, self.x_, self.y_), 0.0, 1.0)

    def to_dict(self) -> dict:
        if self.x_ is None or len(self.x_) == 0:
            return {"x": [], "y": []}
        # The PAVA fit is piecewise-constant, so keep only the boundary points of
        # each constant run — np.interp reproduces it exactly, but the artifact
        # shrinks from O(n_samples) to O(n_steps).
        y = self.y_
        keep = {0, len(y) - 1}
        changes = np.nonzero(np.diff(y))[0]
        keep.update(changes.tolist())
        keep.update((changes + 1).tolist())
        idx = sorted(keep)
        return {"x": [float(self.x_[i]) for i in idx], "y": [float(y[i]) for i in idx]}

    @classmethod
    def from_dict(cls, data: dict) -> "IsotonicCalibrator":
        x = data.get("x") or []
        obj = cls()
        if x:
            obj.x_ = np.asarray(x, dtype=float)
            obj.y_ = np.asarray(data["y"], dtype=float)
        return obj


# --------------------------------------------------------------------------- #
# Convex log-loss-optimal blend
# --------------------------------------------------------------------------- #
def fit_blend_weights(prob_matrix: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Weights on the simplex (>=0, sum 1) minimising the blended log loss of
    the probability columns in ``prob_matrix`` (n_samples x k)."""
    X = _clip(np.asarray(prob_matrix, dtype=float))
    y = np.asarray(outcomes, dtype=float)
    k = X.shape[1]

    def nll(w):
        p = _clip(X @ w)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize(
        nll, np.full(k, 1.0 / k), method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else np.full(k, 1.0 / k)


def blend_probabilities(prob_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob_matrix, dtype=float) @ np.asarray(weights, dtype=float), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Squiggle crowd-model consensus
# --------------------------------------------------------------------------- #
def squiggle_consensus(tips: pd.DataFrame) -> pd.DataFrame:
    """Per-game consensus home-win probability from the Squiggle tips feed:
    the mean of each source's ``hconfidence`` (%), excluding Squiggle's own
    pre-aggregated row to avoid double counting. Returns ``[year, round, hteam,
    ateam, squiggle_home_prob]``."""
    if tips.empty or "hconfidence" not in tips.columns:
        return pd.DataFrame(columns=["year", "round", "hteam", "ateam", "squiggle_home_prob"])

    df = tips[tips["source"].str.lower() != "aggregate"].copy()
    df["hconfidence"] = pd.to_numeric(df["hconfidence"], errors="coerce") / 100.0
    grouped = (
        df.groupby(["year", "round", "hteam", "ateam"], as_index=False)["hconfidence"].mean()
        .rename(columns={"hconfidence": "squiggle_home_prob"})
    )
    return grouped


# --------------------------------------------------------------------------- #
# Assemble signals, fit, report
# --------------------------------------------------------------------------- #
def assemble_signals(games: pd.DataFrame, odds: pd.DataFrame | None = None,
                     tips: pd.DataFrame | None = None, **elo_kwargs) -> pd.DataFrame:
    """One row per game with the available H2H home-win signals (``model_p``,
    ``market_p``, ``squiggle_p``) plus ``outcome`` (actual home win). Only the
    model signal is guaranteed; market/squiggle appear when supplied."""
    history = evaluate_elo(games, **elo_kwargs)
    out = history[["year", "round", "hteam", "ateam"]].copy()
    out["model_p"] = history["pred_home_win_prob"].to_numpy()
    out["outcome"] = history["actual_home_win"].to_numpy()

    if odds is not None and not odds.empty:
        # Dedupe on the (year, hteam, ateam) join key so the left-merge can't
        # expand rows (the spreadsheet occasionally lists a matchup twice).
        o = odds.drop_duplicates(["year", "hteam", "ateam"]).copy()
        home_p, _ = devig_h2h_probs(o["home_odds_close"], o["away_odds_close"])
        o = o.assign(market_p=home_p.to_numpy())[["year", "hteam", "ateam", "market_p"]]
        out = out.merge(o, on=["year", "hteam", "ateam"], how="left")

    if tips is not None and not tips.empty:
        cons = squiggle_consensus(tips)
        out = out.merge(cons, on=["year", "round", "hteam", "ateam"], how="left")
        out = out.rename(columns={"squiggle_home_prob": "squiggle_p"})

    return out


@dataclass
class MarketBlend:
    """A fitted calibrator + convex blend over the available signals (in
    ``SIGNAL_ORDER``). ``predict_home_prob`` calibrates the model probability,
    then blends it with whichever of market/squiggle are supplied, renormalising
    the weights over the signals actually present."""
    signals: list[str]
    weights: np.ndarray
    calibrator: IsotonicCalibrator

    def predict_home_prob(self, model_p, market_p=None, squiggle_p=None) -> np.ndarray:
        model_p = np.atleast_1d(np.asarray(model_p, dtype=float))
        provided = {"model": self.calibrator.predict(model_p)}
        if market_p is not None:
            provided["market"] = np.atleast_1d(np.asarray(market_p, dtype=float))
        if squiggle_p is not None:
            provided["squiggle"] = np.atleast_1d(np.asarray(squiggle_p, dtype=float))

        cols, ws = [], []
        for sig, w in zip(self.signals, self.weights):
            if sig in provided:
                cols.append(provided[sig])
                ws.append(w)
        ws = np.asarray(ws, dtype=float)
        if ws.sum() <= 0:
            return provided["model"]
        ws = ws / ws.sum()
        return np.clip(np.column_stack(cols) @ ws, 0.0, 1.0)


def fit_market_blend(signals: pd.DataFrame) -> MarketBlend:
    """Fit an ``IsotonicCalibrator`` (on the model column) and convex blend
    weights over the complete-case rows of an ``assemble_signals`` frame."""
    present = [s for s in SIGNAL_ORDER if f"{s}_p" in signals.columns]
    cols = [f"{s}_p" for s in present]
    complete = signals.dropna(subset=cols + ["outcome"])

    calibrator = IsotonicCalibrator().fit(complete["model_p"], complete["outcome"])
    # blend the *calibrated* model with the other signals
    matrix = complete[cols].to_numpy().copy()
    matrix[:, present.index("model")] = calibrator.predict(complete["model_p"])
    weights = fit_blend_weights(matrix, complete["outcome"].to_numpy())
    return MarketBlend(signals=present, weights=weights, calibrator=calibrator)


def ensemble_report(games: pd.DataFrame, odds: pd.DataFrame | None = None,
                    tips: pd.DataFrame | None = None, *,
                    train_end_year: int, eval_start_year: int, **elo_kwargs) -> dict:
    """Fit the blend on games up to ``train_end_year`` and report held-out
    (year >= ``eval_start_year``) log loss for each signal, the calibrated
    model, and the blend. Demonstrates §3.5's "blend toward market beats the
    raw model out of sample"."""
    signals = assemble_signals(games, odds, tips, **elo_kwargs)
    present = [s for s in SIGNAL_ORDER if f"{s}_p" in signals.columns]
    cols = [f"{s}_p" for s in present]
    complete = signals.dropna(subset=cols + ["outcome"])

    train = complete[complete["year"] <= train_end_year]
    holdout = complete[complete["year"] >= eval_start_year]
    if train.empty or holdout.empty:
        return {"n_train": len(train), "n_holdout": len(holdout)}

    blend = fit_market_blend(train)
    y = holdout["outcome"].to_numpy()

    report = {"n_train": len(train), "n_holdout": len(holdout),
              "weights": dict(zip(blend.signals, np.round(blend.weights, 3)))}
    for sig in present:
        report[f"log_loss_{sig}"] = log_loss(holdout[f"{sig}_p"].to_numpy(), y)
    report["log_loss_model_calibrated"] = log_loss(blend.calibrator.predict(holdout["model_p"]), y)

    blended = blend.predict_home_prob(
        holdout["model_p"].to_numpy(),
        market_p=holdout["market_p"].to_numpy() if "market" in present else None,
        squiggle_p=holdout["squiggle_p"].to_numpy() if "squiggle" in present else None,
    )
    report["log_loss_blend"] = log_loss(blended, y)
    return report

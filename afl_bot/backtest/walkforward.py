"""
Walk-forward backtesting & calibration (plan §5) — the accuracy engine.

``EloRatings.fit`` already processes games strictly in chronological order and
records *pre-match* ratings, which is exactly the walk-forward property we
need (today's prediction never sees today's result). This module turns those
pre-match ratings into the metrics the plan calls non-negotiable:

  * Log loss + Brier score for win probabilities.
  * Margin MAE for the scoring model.
  * Calibration / reliability curves -- "if 80% legs only land 70% of the
    time, your anchors are lies".

No shuffle-split is offered anywhere in this module on purpose -- time-series
CV only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afl_bot.config import VALUE_MIN_EDGE
from afl_bot.data.odds import attach_odds
from afl_bot.ratings.elo import EloRatings


def evaluate_elo(games: pd.DataFrame, **elo_kwargs) -> pd.DataFrame:
    """Run EloRatings.fit (walk-forward by construction) and attach predicted
    margin / win probability / actual outcome columns for scoring.
    """
    elo = EloRatings(**elo_kwargs)
    history = elo.fit(games)

    rating_gap = history["home_elo_pre"] - history["away_elo_pre"]
    # Per-game home advantage (venue/travel/rest) when the games carry it (§6.1),
    # else the flat home advantage.
    hga_pts = history["hga_points"] if "hga_points" in history.columns else elo.home_advantage
    rating_diff = rating_gap + hga_pts / elo.points_per_400 * elo.scale
    history["pred_margin"] = rating_gap / elo.scale * elo.points_per_400 + hga_pts
    # vectorised logistic (matches afl_bot.ratings.elo.expected_result), avoids a
    # per-row Python call so thousands of tuning fits stay cheap.
    history["pred_home_win_prob"] = 1.0 / (1.0 + 10.0 ** (-rating_diff / elo.scale))
    history["actual_margin"] = history["hscore"] - history["ascore"]
    history["actual_home_win"] = (history["actual_margin"] > 0).astype(float)
    return history


# ----------------------------------------------------------------------------- #
# Metrics (plan §5)
# ----------------------------------------------------------------------------- #
def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def margin_mae(pred_margin: np.ndarray, actual_margin: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_margin - actual_margin)))


def calibration_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: for each predicted-probability bucket, the mean
    predicted prob vs the actual hit rate and sample count. A well-calibrated
    model has ``mean_pred ~= actual_rate`` in every row.
    """
    df = pd.DataFrame({"prob": probs, "outcome": outcomes})
    df["bucket"] = pd.cut(df["prob"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = df.groupby("bucket", observed=True).agg(
        mean_pred=("prob", "mean"),
        actual_rate=("outcome", "mean"),
        n=("outcome", "size"),
    )
    return grouped.reset_index()


# ----------------------------------------------------------------------------- #
# Market comparison & CLV (plan §4.2, build-order step 3)
# ----------------------------------------------------------------------------- #
def devig_h2h_probs(home_odds: pd.Series, away_odds: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Proportional devig of H2H decimal odds -> (home_prob, away_prob) that
    sum to 1, vectorised for a column of games."""
    home_implied = 1.0 / home_odds
    away_implied = 1.0 / away_odds
    overround = home_implied + away_implied
    return home_implied / overround, away_implied / overround


def market_metrics(games_with_odds: pd.DataFrame) -> dict:
    """Log loss / Brier of the closing-line devigged market probability --
    the benchmark our model's ``evaluate_elo`` log loss should be compared
    against. Games without closing odds are dropped."""
    df = games_with_odds.dropna(subset=["home_odds_close", "away_odds_close"])
    if df.empty:
        return {"n_games": 0, "log_loss": float("nan"), "brier": float("nan")}

    market_home_prob, _ = devig_h2h_probs(df["home_odds_close"], df["away_odds_close"])
    actual_home_win = (df["hscore"] > df["ascore"]).astype(float)
    return {
        "n_games": len(df),
        "log_loss": log_loss(market_home_prob.to_numpy(), actual_home_win.to_numpy()),
        "brier": brier_score(market_home_prob.to_numpy(), actual_home_win.to_numpy()),
    }


def clv_report(history: pd.DataFrame, odds: pd.DataFrame, value_min_edge: float = VALUE_MIN_EDGE) -> pd.DataFrame:
    """For games where the model's (walk-forward) home-win probability
    disagrees with the closing devigged market by at least ``value_min_edge``,
    report the closing-line value (CLV) of backing the model's side at the
    *opening* price.

    CLV = closing devig prob (model's side) - opening devig prob (model's
    side). Positive CLV means the market moved toward the model's view
    between open and close -- i.e. the opening price was better than the
    price the market settled on, the standard signal that an edge was real
    (plan §4.2: "CLV is the fastest honest signal of real edge").

    ``history`` should come from ``evaluate_elo`` (has ``pred_home_win_prob``,
    ``year``, ``hteam``, ``ateam``); ``odds`` from
    ``afl_bot.data.odds.fetch_historical_odds``.
    """
    df = attach_odds(history, odds)
    df = df.dropna(subset=["home_odds_open", "home_odds_close", "away_odds_open", "away_odds_close"]).copy()
    if df.empty:
        return df

    open_home_p, open_away_p = devig_h2h_probs(df["home_odds_open"], df["away_odds_open"])
    close_home_p, close_away_p = devig_h2h_probs(df["home_odds_close"], df["away_odds_close"])

    edge_home = df["pred_home_win_prob"] - close_home_p
    df["side"] = np.where(edge_home >= 0, "home", "away")
    df["model_edge"] = edge_home.abs()
    df["open_prob"] = np.where(df["side"] == "home", open_home_p, open_away_p)
    df["close_prob"] = np.where(df["side"] == "home", close_home_p, close_away_p)
    df["clv"] = df["close_prob"] - df["open_prob"]

    flagged = df[df["model_edge"] >= value_min_edge]
    return flagged[[
        "year", "round", "hteam", "ateam", "side", "model_edge",
        "open_prob", "close_prob", "clv",
    ]].reset_index(drop=True)


def clv_summary(flagged: pd.DataFrame) -> dict:
    """Aggregate ``clv_report`` output: mean CLV and the fraction of flagged
    legs where the market moved in the model's favour (positive CLV)."""
    if flagged.empty:
        return {"n_legs": 0, "mean_clv": float("nan"), "pct_positive": float("nan")}
    return {
        "n_legs": len(flagged),
        "mean_clv": float(flagged["clv"].mean()),
        "pct_positive": float((flagged["clv"] > 0).mean()),
    }


# ----------------------------------------------------------------------------- #
# Roll-forward season-by-season report
# ----------------------------------------------------------------------------- #
def season_by_season_report(history: pd.DataFrame) -> pd.DataFrame:
    """Per-season log loss / Brier / margin MAE, using each game's pre-match
    (i.e. walk-forward) prediction. Useful for spotting drift or a season where
    Elo hyperparameters need retuning.
    """
    rows = []
    for year, grp in history.groupby("year"):
        rows.append({
            "year": year,
            "n_games": len(grp),
            "log_loss": log_loss(grp["pred_home_win_prob"].to_numpy(), grp["actual_home_win"].to_numpy()),
            "brier": brier_score(grp["pred_home_win_prob"].to_numpy(), grp["actual_home_win"].to_numpy()),
            "margin_mae": margin_mae(grp["pred_margin"].to_numpy(), grp["actual_margin"].to_numpy()),
        })
    return pd.DataFrame(rows)


def margin_calibration(history: pd.DataFrame) -> dict:
    """Regress actual margin on predicted margin (round-2 §6.3). A well-calibrated
    margin model has ``slope ~= 1`` and ``intercept ~= 0``; a slope < 1 means the
    model is over-confident in its margins (``ELO_POINTS_PER_400`` too high), > 1
    under-confident. Reported alongside margin MAE so the points-per-400 mapping
    isn't doing silent work."""
    pred = history["pred_margin"].to_numpy(dtype=float)
    actual = history["actual_margin"].to_numpy(dtype=float)
    if len(pred) < 2 or np.allclose(pred, pred[0]):
        return {"n": len(pred), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(pred, actual, 1)
    r = np.corrcoef(pred, actual)[0, 1]
    return {"n": len(pred), "slope": float(slope), "intercept": float(intercept), "r2": float(r ** 2)}

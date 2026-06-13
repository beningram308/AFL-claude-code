import numpy as np
import pandas as pd

from afl_bot.backtest.walkforward import (
    clv_report,
    clv_summary,
    devig_h2h_probs,
    market_metrics,
)


def test_devig_h2h_probs_sums_to_one_and_removes_overround():
    home_odds = pd.Series([1.80, 2.50])
    away_odds = pd.Series([2.05, 1.55])

    home_p, away_p = devig_h2h_probs(home_odds, away_odds)

    assert np.allclose(home_p + away_p, 1.0)
    # raw implied probs sum to >1 (overround); home is still favourite in row 0
    assert home_p.iloc[0] > away_p.iloc[0]
    assert away_p.iloc[1] > home_p.iloc[1]


def test_market_metrics_drops_rows_without_close_odds_and_scores_remainder():
    games_with_odds = pd.DataFrame([
        {"hscore": 90, "ascore": 80, "home_odds_close": 1.70, "away_odds_close": 2.20},
        {"hscore": 70, "ascore": 100, "home_odds_close": 2.80, "away_odds_close": 1.45},
        {"hscore": 60, "ascore": 50, "home_odds_close": np.nan, "away_odds_close": np.nan},
    ])

    metrics = market_metrics(games_with_odds)

    assert metrics["n_games"] == 2
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["brier"] <= 1


def test_market_metrics_empty_when_no_odds():
    games_with_odds = pd.DataFrame([
        {"hscore": 90, "ascore": 80, "home_odds_close": np.nan, "away_odds_close": np.nan},
    ])

    metrics = market_metrics(games_with_odds)

    assert metrics["n_games"] == 0
    assert np.isnan(metrics["log_loss"])
    assert np.isnan(metrics["brier"])


HISTORY = pd.DataFrame([
    # model strongly favours home (0.80) vs market close (~0.486) -> big edge, flagged
    {"year": 2024, "round": 1, "hteam": "Carlton", "ateam": "Richmond", "pred_home_win_prob": 0.80},
    # model and market close roughly agree -> small edge, not flagged
    {"year": 2024, "round": 2, "hteam": "Adelaide", "ateam": "Geelong", "pred_home_win_prob": 0.36},
])

ODDS = pd.DataFrame([
    {
        "year": 2024, "hteam": "Carlton", "ateam": "Richmond",
        "home_odds_open": 2.20, "home_odds_close": 1.90,
        "away_odds_open": 1.75, "away_odds_close": 2.00,
        "total_open": 170.5, "total_close": 172.5,
        "total_over_odds_close": 1.90, "total_under_odds_close": 1.90,
    },
    {
        "year": 2024, "hteam": "Adelaide", "ateam": "Geelong",
        "home_odds_open": 2.55, "home_odds_close": 2.50,
        "away_odds_open": 1.55, "away_odds_close": 1.57,
        "total_open": 165.5, "total_close": 168.5,
        "total_over_odds_close": 1.90, "total_under_odds_close": 1.90,
    },
])


def test_clv_report_flags_only_games_with_sufficient_edge():
    flagged = clv_report(HISTORY, ODDS, value_min_edge=0.08)

    assert len(flagged) == 1
    row = flagged.iloc[0]
    assert row["hteam"] == "Carlton"
    assert row["side"] == "home"
    assert row["model_edge"] >= 0.08
    assert {"open_prob", "close_prob", "clv"}.issubset(flagged.columns)


def test_clv_report_empty_when_odds_missing():
    odds_missing = ODDS.copy()
    odds_missing.loc[:, ["home_odds_open", "home_odds_close", "away_odds_open", "away_odds_close"]] = np.nan

    flagged = clv_report(HISTORY, odds_missing, value_min_edge=0.08)

    assert flagged.empty


def test_clv_summary_aggregates_flagged_legs():
    flagged = clv_report(HISTORY, ODDS, value_min_edge=0.08)

    summary = clv_summary(flagged)

    assert summary["n_legs"] == 1
    assert -1 <= summary["mean_clv"] <= 1
    assert summary["pct_positive"] in (0.0, 1.0)


def test_clv_summary_empty_input():
    summary = clv_summary(pd.DataFrame())

    assert summary["n_legs"] == 0
    assert np.isnan(summary["mean_clv"])
    assert np.isnan(summary["pct_positive"])

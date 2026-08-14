"""Model-vs-closing-price walk-forward evaluation (2026-08-15). Synthetic
fixtures for the grading/summary/formatting functions (fast); the population
loader and full end-to-end analysis need the real odds file + a Squiggle
fetch, so those are marked slow, matching this codebase's convention for
real-data-dependent tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afl_bot.backtest.model_vs_close import (
    MIN_BETS_FOR_ROI_CLAIM,
    _fmt_label,
    add_commission,
    add_edge_bucket,
    compute_totals_model,
    grade_h2h,
    grade_totals,
    load_population,
    run_analysis,
    summarize,
)


# ── grade_h2h ────────────────────────────────────────────────────────────────

def _h2h_history(**overrides) -> pd.DataFrame:
    base = {
        "year": [2020], "hteam": ["Home"], "ateam": ["Away"],
        "hscore": [100], "ascore": [80],
        "home_odds_close": [1.80], "away_odds_close": [2.10],
        "home_odds_open": [1.90], "away_odds_open": [2.00],
        "model_home_prob": [0.60],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_grade_h2h_home_win_favours_home_when_model_prob_exceeds_close():
    # close devig home prob = (1/1.80)/((1/1.80)+(1/2.10)) ~= 0.5385 < model 0.60
    # -> side "home", model_edge ~ 0.0615, home actually won -> "win".
    df = grade_h2h(_h2h_history())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["side"] == "home"
    assert row["model_edge"] == pytest.approx(0.60 - (1 / 1.80) / (1 / 1.80 + 1 / 2.10), abs=1e-9)
    assert row["outcome"] == "win"
    assert row["gross_profit"] == pytest.approx(1.80 - 1.0)


def test_grade_h2h_away_favoured_and_wins():
    # model_home_prob=0.30 -> close devig home ~0.5385 > model -> side "away".
    # Away actually won (ascore > hscore) -> "win".
    df = grade_h2h(_h2h_history(model_home_prob=[0.30], hscore=[70], ascore=[100]))
    row = df.iloc[0]
    assert row["side"] == "away"
    assert row["outcome"] == "win"
    assert row["gross_profit"] == pytest.approx(2.10 - 1.0)


def test_grade_h2h_side_loses():
    # side "home" favoured (model 0.60) but away actually won.
    df = grade_h2h(_h2h_history(hscore=[70], ascore=[100]))
    row = df.iloc[0]
    assert row["side"] == "home"
    assert row["outcome"] == "loss"
    assert row["gross_profit"] == -1.0


def test_grade_h2h_drops_rows_missing_close_price_or_model_prob():
    df = _h2h_history()
    missing_close = pd.concat([df, _h2h_history(home_odds_close=[np.nan])], ignore_index=True)
    graded = grade_h2h(missing_close)
    assert len(graded) == 1  # the row with NaN close price is excluded, not crashed on


def test_grade_h2h_clv_nan_when_open_price_missing():
    df = _h2h_history(home_odds_open=[np.nan], away_odds_open=[np.nan])
    graded = grade_h2h(df)
    assert pd.isna(graded.iloc[0]["clv"])


def test_grade_h2h_clv_computed_when_open_present():
    df = grade_h2h(_h2h_history())
    row = df.iloc[0]
    open_home_p = (1 / 1.90) / (1 / 1.90 + 1 / 2.00)
    close_home_p = (1 / 1.80) / (1 / 1.80 + 1 / 2.10)
    assert row["clv"] == pytest.approx(close_home_p - open_home_p, abs=1e-9)


def test_grade_h2h_empty_input_returns_empty_frame():
    empty = _h2h_history().iloc[0:0]
    assert grade_h2h(empty).empty


# ── grade_totals ─────────────────────────────────────────────────────────────

def _totals_history(**overrides) -> pd.DataFrame:
    base = {
        "year": [2020], "hteam": ["Home"], "ateam": ["Away"],
        "hscore": [100], "ascore": [80],   # actual total 180
        "total_open": [170.5], "total_close": [175.5],
        "total_over_odds_close": [1.90], "total_under_odds_close": [1.90],
        "model_over_prob": [0.60],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_grade_totals_over_favoured_and_hits():
    # close devig over ~0.5, model 0.60 -> side "over"; actual total 180 > 175.5 -> win.
    df = grade_totals(_totals_history())
    row = df.iloc[0]
    assert row["side"] == "over"
    assert row["outcome"] == "win"
    assert row["gross_profit"] == pytest.approx(0.90)


def test_grade_totals_under_favoured_and_misses():
    df = grade_totals(_totals_history(model_over_prob=[0.30]))
    row = df.iloc[0]
    assert row["side"] == "under"
    assert row["outcome"] == "loss"   # actual total 180 > line 175.5 -> under loses
    assert row["gross_profit"] == -1.0


def test_grade_totals_clv_is_always_nan_not_fabricated():
    df = grade_totals(_totals_history())
    assert pd.isna(df.iloc[0]["clv"])


def test_grade_totals_line_move_reported_in_model_direction():
    # total_open=170.5, total_close=175.5 -> line moved UP by 5.
    # side "over": moving up is BAD for over (harder to clear) -- but we
    # report "model_direction" as the raw signed move on the model's side:
    # over -> -(close-open); here that's -5.
    df = grade_totals(_totals_history(model_over_prob=[0.60]))
    row = df.iloc[0]
    assert row["side"] == "over"
    assert row["line_move_model_direction"] == pytest.approx(-5.0)


def test_grade_totals_drops_rows_missing_prices():
    df = _totals_history()
    missing = pd.concat([df, _totals_history(total_close=[np.nan])], ignore_index=True)
    graded = grade_totals(missing)
    assert len(graded) == 1


def test_grade_totals_empty_input_returns_empty_frame():
    empty = _totals_history().iloc[0:0]
    assert grade_totals(empty).empty


# ── add_commission / add_edge_bucket ────────────────────────────────────────

def test_add_commission_only_taxes_wins():
    df = pd.DataFrame({"gross_profit": [2.0, -1.0, 0.0]})
    out = add_commission(df, commission=0.05)
    assert out["net_profit"].tolist() == pytest.approx([2.0 * 0.95, -1.0, 0.0])


def test_add_edge_bucket_boundaries():
    df = pd.DataFrame({"model_edge": [0.0, 0.019, 0.02, 0.049, 0.05, 0.099, 0.10, 0.5]})
    out = add_edge_bucket(df)
    assert out["edge_bucket"].astype(str).tolist() == [
        "0-2%", "0-2%", "2-5%", "2-5%", "5-10%", "5-10%", "10%+", "10%+",
    ]


# ── summarize ────────────────────────────────────────────────────────────────

def _sized_frame(profits: list[float], edges: list[float] | None = None, year: int = 2020) -> pd.DataFrame:
    n = len(profits)
    return pd.DataFrame({
        "year": [year] * n,
        "model_prob_side": [0.55] * n,
        "close_prob": [0.50] * n,
        "gross_profit": profits,
        "model_edge": edges or [0.03] * n,
        "clv": [0.01] * n,
    })


def test_summarize_overall_row_with_no_group_cols():
    df = _sized_frame([1.0, -1.0, -1.0])
    out = summarize(df, [])
    assert len(out) == 1
    assert out.iloc[0]["n"] == 3
    assert out.iloc[0]["gross_roi_pct"] == pytest.approx((1.0 - 1.0 - 1.0) / 3 * 100)


def test_summarize_by_group_reports_n_per_group():
    df = pd.concat([_sized_frame([1.0, 1.0], year=2020), _sized_frame([-1.0], year=2021)], ignore_index=True)
    out = summarize(df, ["year"])
    by_year = dict(zip(out["year"], out["n"]))
    assert by_year == {2020: 2, 2021: 1}


def test_summarize_empty_input_returns_empty_with_correct_columns():
    out = summarize(_sized_frame([]).iloc[0:0], ["year"])
    assert out.empty
    assert "year" in out.columns and "n" in out.columns


def test_summarize_min_bets_threshold_is_200():
    # Documents the brief's own rule as a constant this module actually uses,
    # not just a number in a docstring somewhere.
    assert MIN_BETS_FOR_ROI_CLAIM == 200


# ── formatting ───────────────────────────────────────────────────────────────

def test_fmt_label_strips_whole_number_float():
    assert _fmt_label(2013.0) == "2013"
    assert _fmt_label(2013) == "2013"
    assert _fmt_label("0-2%") == "0-2%"
    assert _fmt_label(2013.5) == "2013.5"


# ── compute_totals_model: first season has no walk-forward prediction ──────

def test_compute_totals_model_first_season_is_all_nan():
    games = pd.DataFrame({
        "year": [2020, 2020, 2021, 2021],
        "round": [1, 2, 1, 2],
        "unixtime": [1, 2, 3, 4],
        "hteam": ["A", "B", "A", "B"],
        "ateam": ["B", "A", "B", "A"],
        "hscore": [100, 90, 110, 95],
        "ascore": [80, 85, 90, 100],
        "venue": ["V1", "V1", "V1", "V1"],
        "total_close": [175.5, 170.5, 180.5, 175.5],
    })
    out = compute_totals_model(games)
    first_season = out[out["year"] == 2020]
    second_season = out[out["year"] == 2021]
    assert first_season["expected_total"].isna().all(), "no prior seasons -> no walk-forward prediction"
    assert second_season["expected_total"].notna().all(), "2021 has 2020 as prior data"


# ── real-data integration (slow) ─────────────────────────────────────────────

@pytest.mark.slow
def test_load_population_anchored_to_odds_file():
    """The population must be EXACTLY the odds file's own row count, and the
    null-h2h/null-totals counts must be untouched by the Squiggle join --
    this is the core "don't silently drop the 681 rows" guarantee the whole
    report leans on."""
    pop, diag = load_population()
    assert diag["n_odds_raw"] == len(pop)
    assert diag["n_null_h2h_raw"] == int(pop["home_odds_close"].isna().sum())
    assert diag["n_null_totals_raw"] == int(pop["total_close"].isna().sum())
    assert diag["n_odds_raw"] > 3000  # sanity: this is the real multi-thousand-row file


@pytest.mark.slow
def test_run_analysis_end_to_end_produces_well_formed_report():
    result = run_analysis()
    md = result["report_md"]
    assert md.startswith("No seasons excluded, no null-closing rows dropped")
    assert "## Verdict" in md
    assert ("**YES**" in md) or ("**NO**" in md)
    assert len(result["h2h_graded"]) > MIN_BETS_FOR_ROI_CLAIM
    assert len(result["totals_graded"]) > MIN_BETS_FOR_ROI_CLAIM

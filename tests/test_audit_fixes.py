"""Regression tests for the 2026-07-31 external audit fixes.

Covers:
  1. promo_multi_ev — one-loss branch must net (bonus_factor - 1), not +bonus_factor.
  2. attach_odds — date-aware join must not expand rows on rematch keys and must
     attach each meeting's own odds.
  3. grade_round log dedup — (year, round), not round alone (tested at the
     dataframe-filter level, mirroring the cli logic).
  4. venue_scoring_factors — venue aliases pooled; alias lookups still resolve.
  5. days_rest / game_hga_points — rest values aligned to caller's row order
     regardless of input sort order.
"""

import numpy as np
import pandas as pd
import pytest

from afl_bot.build.multi import promo_multi_ev
from afl_bot.data.odds import attach_odds
from afl_bot.models.scoring import venue_scoring_factors
from afl_bot.ratings.hga import days_rest, game_hga_points


# --------------------------------------------------------------------------- #
# 1. promo_multi_ev
# --------------------------------------------------------------------------- #
def test_promo_ev_matches_exact_identity():
    """EV per unit stake must equal p_all*M - 1 + p_one_loss*R exactly."""
    p1, p2, p3, M, R = 0.7, 0.7, 0.7, 2.92, 0.75
    r = promo_multi_ev(p1, p2, p3, M, stake=1.0, bonus_factor=R)
    expected = r["p_all_win"] * M - 1.0 + r["p_exactly_one_loss"] * R
    assert r["ev_pct"] == pytest.approx(expected, abs=1e-12)


def test_promo_ev_one_loss_branch_is_a_net_loss_for_partial_refund():
    """With bonus_factor < 1, a certain one-leg miss must give NEGATIVE EV."""
    # p1=p2=1, p3=0 -> always exactly one loss -> net (R-1) per unit.
    r = promo_multi_ev(1.0, 1.0, 0.0, 3.0, stake=1.0, bonus_factor=0.75)
    assert r["p_exactly_one_loss"] == pytest.approx(1.0)
    assert r["ev_pct"] == pytest.approx(-0.25, abs=1e-12)


def test_promo_ev_consistent_with_multi_outcome_kelly_gate():
    """promo_multi_ev > 0 must agree with multi_outcome_kelly's g'(0) > 0 gate."""
    from afl_bot.build.staking import multi_outcome_kelly
    for probs, M in [((0.8, 0.8, 0.8), 2.2), ((0.6, 0.6, 0.6), 5.0),
                     ((0.9, 0.9, 0.5), 2.8), ((0.5, 0.5, 0.5), 7.0)]:
        r = promo_multi_ev(*probs, M, stake=1.0, bonus_factor=0.75)
        f = multi_outcome_kelly(r["p_all_win"], r["p_exactly_one_loss"],
                                r["p_dead"], M, 0.75)
        if r["ev_pct"] > 1e-9:
            assert f > 0.0, f"EV {r['ev_pct']:.4f} > 0 but Kelly stake 0 for {probs} @ {M}"
        else:
            assert f == 0.0, f"EV {r['ev_pct']:.4f} <= 0 but Kelly stake {f} for {probs} @ {M}"


# --------------------------------------------------------------------------- #
# 2. attach_odds
# --------------------------------------------------------------------------- #
def _games(rows):
    return pd.DataFrame(rows)


def test_attach_odds_does_not_expand_rematch_rows():
    """Two same-key meetings (H&A game + final) must stay two rows, each with
    its own meeting's odds."""
    games = _games([
        {"year": 2016, "hteam": "Geelong", "ateam": "Hawthorn",
         "date": "2016-04-18 19:20:00", "hscore": 100, "ascore": 90},
        {"year": 2016, "hteam": "Geelong", "ateam": "Hawthorn",
         "date": "2016-09-09 19:50:00", "hscore": 80, "ascore": 85},
    ])
    odds = pd.DataFrame([
        {"date": "2016-04-18", "hteam": "Geelong", "ateam": "Hawthorn",
         "year": 2016, "home_odds_open": 1.80, "home_odds_close": 1.85,
         "away_odds_open": 2.00, "away_odds_close": 1.95},
        {"date": "2016-09-09", "hteam": "Geelong", "ateam": "Hawthorn",
         "year": 2016, "home_odds_open": 2.20, "home_odds_close": 2.30,
         "away_odds_open": 1.65, "away_odds_close": 1.60},
    ])
    out = attach_odds(games, odds)
    assert len(out) == 2, "join must not expand rows"
    assert out.iloc[0]["home_odds_close"] == pytest.approx(1.85)
    assert out.iloc[1]["home_odds_close"] == pytest.approx(2.30)


def test_attach_odds_left_join_keeps_unmatched_games():
    games = _games([
        {"year": 2026, "hteam": "Carlton", "ateam": "Essendon",
         "date": "2026-05-01 19:20:00", "hscore": 0, "ascore": 0},
    ])
    odds = pd.DataFrame([
        {"date": "2019-05-01", "hteam": "Carlton", "ateam": "Essendon",
         "year": 2019, "home_odds_open": 2.0, "home_odds_close": 2.0,
         "away_odds_open": 1.8, "away_odds_close": 1.8},
    ])
    out = attach_odds(games, odds)
    assert len(out) == 1
    assert np.isnan(out.iloc[0]["home_odds_close"]), "odds 7 years away must not match"


def test_attach_odds_tolerates_one_day_date_skew():
    games = _games([
        {"year": 2024, "hteam": "Sydney", "ateam": "Carlton",
         "date": "2024-06-07 19:50:00", "hscore": 1, "ascore": 2},
    ])
    odds = pd.DataFrame([
        {"date": "2024-06-08", "hteam": "Sydney", "ateam": "Carlton",
         "year": 2024, "home_odds_open": 1.5, "home_odds_close": 1.55,
         "away_odds_open": 2.6, "away_odds_close": 2.5},
    ])
    out = attach_odds(games, odds)
    assert out.iloc[0]["home_odds_close"] == pytest.approx(1.55)


# --------------------------------------------------------------------------- #
# 3. calibration-log dedup (year, round)
# --------------------------------------------------------------------------- #
def test_calibration_log_dedup_is_year_aware():
    prev = pd.DataFrame([
        {"year": 2025, "round": 1, "market": "h2h", "subject": "A", "line": "",
         "prob": 0.6, "actual": 1},
        {"year": 2026, "round": 1, "market": "h2h", "subject": "B", "line": "",
         "prob": 0.5, "actual": 0},
    ])
    year, round_no = 2026, 1
    kept = prev[~((prev["year"] == year) & (prev["round"] == round_no))]
    assert len(kept) == 1 and kept.iloc[0]["year"] == 2025, (
        "re-grading 2026 R1 must not delete the 2025 R1 record")


# --------------------------------------------------------------------------- #
# 4. venue alias pooling
# --------------------------------------------------------------------------- #
def test_venue_factors_pool_aliases_and_alias_lookup_resolves():
    rng = np.random.default_rng(0)
    rows = []
    # 40 high-scoring games under one alias, 5 under the other; plus a
    # low-scoring reference ground to move the league mean.
    for _ in range(40):
        rows.append({"venue": "Docklands", "hscore": 110, "ascore": 100})
    for _ in range(5):
        rows.append({"venue": "Marvel Stadium", "hscore": 110, "ascore": 100})
    for _ in range(45):
        rows.append({"venue": "Gabba", "hscore": 70, "ascore": 60})
    games = pd.DataFrame(rows)
    f = venue_scoring_factors(games, strength=30.0)
    assert f["Docklands"] == pytest.approx(f["Marvel Stadium"]), "aliases must share one factor"
    # pooled 45-game sample must be less shrunk than a 5-game alias alone would be
    league = games.eval("hscore + ascore").mean()
    pooled = (45 * 210 + 30 * league) / (45 + 30) / league
    assert f["Marvel Stadium"] == pytest.approx(pooled, rel=1e-9)


# --------------------------------------------------------------------------- #
# 5. days_rest alignment
# --------------------------------------------------------------------------- #
def test_days_rest_alignment_is_independent_of_input_order():
    day = 86400.0
    games = pd.DataFrame([
        {"hteam": "A", "ateam": "B", "unixtime": 0 * day, "hscore": 1, "ascore": 0,
         "venue": "M.C.G."},
        {"hteam": "A", "ateam": "C", "unixtime": 6 * day, "hscore": 1, "ascore": 0,
         "venue": "M.C.G."},
        {"hteam": "B", "ateam": "C", "unixtime": 13 * day, "hscore": 1, "ascore": 0,
         "venue": "M.C.G."},
    ])
    shuffled = games.iloc[[2, 0, 1]]
    r1 = days_rest(games).set_index("unixtime")[["home_rest", "away_rest"]]
    r2 = days_rest(shuffled).set_index("unixtime")[["home_rest", "away_rest"]]
    pd.testing.assert_frame_equal(r1.sort_index(), r2.sort_index())
    # spot values: game at t=6d — A rested 6 days; game at t=13d — B rested 13, C 7.
    assert r1.loc[6 * day, "home_rest"] == pytest.approx(6.0)
    assert r1.loc[13 * day, "home_rest"] == pytest.approx(13.0)
    assert r1.loc[13 * day, "away_rest"] == pytest.approx(7.0)


def test_game_hga_points_order_invariant():
    day = 86400.0
    games = pd.DataFrame([
        {"hteam": "Geelong", "ateam": "Sydney", "unixtime": 0 * day,
         "hscore": 100, "ascore": 90, "venue": "GMHBA Stadium"},
        {"hteam": "Sydney", "ateam": "Geelong", "unixtime": 6 * day,
         "hscore": 90, "ascore": 80, "venue": "S.C.G."},
        {"hteam": "Geelong", "ateam": "Sydney", "unixtime": 20 * day,
         "hscore": 95, "ascore": 70, "venue": "GMHBA Stadium"},
    ])
    hga = {"Geelong": 14.0, "Sydney": 9.0}
    a = game_hga_points(games, hga)
    b = game_hga_points(games.iloc[[2, 1, 0]], hga)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b)[::-1])


# --------------------------------------------------------------------------- #
# 6. Phase-2 improvements (2026-07-31): sim-style blend + tuned-params artifact
# --------------------------------------------------------------------------- #
def _synthetic_two_team_season():
    rows = []
    day = 86400
    for i in range(20):
        rows.append({"year": 2020 + i // 10, "round": (i % 10) + 1,
                     "unixtime": i * 7 * day, "date": pd.Timestamp("2020-04-01") + pd.Timedelta(days=7 * i),
                     "hteam": "A" if i % 2 == 0 else "B",
                     "ateam": "B" if i % 2 == 0 else "A",
                     "hscore": 100 + (5 if i % 2 == 0 else -5), "ascore": 90})
    return pd.DataFrame(rows)


def test_assemble_signals_sim_style_uses_margin_scale():
    """sim_style model_p must equal Phi(pred_margin / SIM_MARGIN_SIGMA)."""
    from scipy.stats import norm
    from afl_bot.backtest.ensemble import assemble_signals
    from afl_bot.backtest.walkforward import evaluate_elo
    from afl_bot.config import SIM_MARGIN_SIGMA
    games = _synthetic_two_team_season()
    sig = assemble_signals(games, sim_style=True)
    hist = evaluate_elo(games)
    expected = norm.cdf(hist["pred_margin"].to_numpy() / SIM_MARGIN_SIGMA)
    np.testing.assert_allclose(sig["model_p"].to_numpy(), expected, atol=1e-12)
    # default remains the logistic
    sig0 = assemble_signals(games)
    np.testing.assert_allclose(sig0["model_p"].to_numpy(),
                               hist["pred_home_win_prob"].to_numpy(), atol=1e-12)


def test_assemble_signals_market_join_does_not_expand_rematches():
    from afl_bot.backtest.ensemble import assemble_signals
    games = _synthetic_two_team_season()
    odds = pd.DataFrame([
        {"date": r["date"], "year": r["year"], "hteam": r["hteam"], "ateam": r["ateam"],
         "home_odds_open": 1.9, "home_odds_close": 1.9 + 0.001 * i,
         "away_odds_open": 1.9, "away_odds_close": 1.9}
        for i, (_, r) in enumerate(games.iterrows())
    ])
    sig = assemble_signals(games, odds)
    assert len(sig) == len(games), "market join must not expand rematch rows"
    assert sig["market_p"].notna().all()


def test_fitted_elo_params_artifact_is_loaded_and_applied(tmp_path):
    import json as _json
    from afl_bot.backtest.tuning import load_fitted_elo_params
    (tmp_path / "elo_params.json").write_text(_json.dumps(
        {"params": {"points_per_400": 116.0}, "metrics": {}}))
    params = load_fitted_elo_params(cache_dir=tmp_path)
    assert params == {"points_per_400": 116.0}
    from afl_bot.ratings.elo import EloRatings
    elo = EloRatings(**params)
    assert elo.points_per_400 == 116.0
    # margin prediction scales with the artifact value
    elo.ratings = {"A": 1600.0, "B": 1500.0}
    assert elo.expected_margin("A", "B") == pytest.approx(100 / 400 * 116.0 + elo.home_advantage)


# --------------------------------------------------------------------------- #
# Phase 3 task 2: grade-round calibration breakdown report
# --------------------------------------------------------------------------- #
def _graded_frame(year, round_no, n_h2h=4, n_disposals=6):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_h2h):
        p = float(rng.uniform(0.3, 0.8))
        rows.append({"year": year, "round": round_no, "market": "h2h", "subject": f"T{i}",
                     "line": "", "prob": p, "actual": int(rng.uniform() < p)})
    for i in range(n_disposals):
        p = float(rng.uniform(0.3, 0.8))
        rows.append({"year": year, "round": round_no, "market": "player_disposals",
                     "subject": f"P{i}", "line": 20, "prob": p, "actual": int(rng.uniform() < p)})
    return pd.DataFrame(rows)


def test_calibration_summary_written_and_sectioned(tmp_path):
    from afl_bot.grading import _write_calibration_summary

    graded = _graded_frame(2026, 18)
    summary_path = _write_calibration_summary(tmp_path, 2026, 18, graded)
    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    assert "<!-- calibration-summary:2026:18:start -->" in text
    assert "<!-- calibration-summary:2026:18:end -->" in text
    assert "## 2026 Round 18" in text
    assert "player_disposals" in text
    assert "h2h" in text


def test_calibration_summary_regrade_replaces_not_duplicates(tmp_path):
    from afl_bot.grading import _write_calibration_summary

    _write_calibration_summary(tmp_path, 2026, 18, _graded_frame(2026, 18))
    _write_calibration_summary(tmp_path, 2026, 19, _graded_frame(2026, 19))
    summary_path = _write_calibration_summary(tmp_path, 2026, 18, _graded_frame(2026, 18, n_h2h=2))
    text = summary_path.read_text(encoding="utf-8")
    assert text.count("<!-- calibration-summary:2026:18:start -->") == 1
    assert text.count("<!-- calibration-summary:2026:19:start -->") == 1
    assert "## 2026 Round 19" in text, "re-grading R18 must not disturb R19's section"


def test_calibration_summary_regrade_is_idempotent(tmp_path):
    from afl_bot.grading import _write_calibration_summary

    _write_calibration_summary(tmp_path, 2026, 18, _graded_frame(2026, 18))
    _write_calibration_summary(tmp_path, 2026, 19, _graded_frame(2026, 19))
    summary_path = _write_calibration_summary(tmp_path, 2026, 18, _graded_frame(2026, 18))
    once = summary_path.read_text(encoding="utf-8")
    summary_path = _write_calibration_summary(tmp_path, 2026, 18, _graded_frame(2026, 18))
    twice = summary_path.read_text(encoding="utf-8")
    assert once == twice, "re-grading the same round repeatedly must not accumulate blank lines"

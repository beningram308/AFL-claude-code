"""Walk-forward SGM-ladder backtest (model-upgrade audit Phase 1.1, plus
Phase 2.5's calibration-ON mode) — both run on deterministic, seeded data
with the real player-stat fetchers monkeypatched out, so no network is
needed."""

import numpy as np
import pandas as pd
import pytest

from afl_bot.backtest.ensemble import IsotonicCalibrator
from afl_bot.backtest.multis import (
    corr_gain_diagnostic,
    fit_corr_gain_haircut,
    fit_multi_calibrator,
    haircut_joint_prob,
    load_or_fit_multi_calibrator,
    multi_calibration_report,
    multi_reliability_curve,
    walk_forward_multi_predictions,
    walk_forward_sgm_candidate_predictions,
    walk_forward_sim_prop_predictions,
)
from afl_bot.backtest.props import apply_prop_calibration, fit_prop_calibrators
from afl_bot.backtest.walkforward import log_loss
from afl_bot.build.report import apply_multi_calibration
from afl_bot.config import LEG_PROB_MAX, LEG_PROB_MIN, MULTI_TARGET_ODDS
from afl_bot.data.player_stats import synthetic_player_log

TEAMS = [f"Team{i}" for i in range(8)]
N_ROUNDS = 10
SEASONS = (2021, 2022, 2023, 2024)


def _synth_games(seed=11):
    """Multi-season fixture list (4 matches/round, round-robin pairing) with
    the full schema `team_scoring_profiles` / `team_shot_accuracy_profiles` /
    `venue_scoring_factors` / `fit_team_hga` need: hscore/ascore, hgoals/
    agoals/hbehinds/abehinds, venue, year/round/unixtime."""
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(TEAMS, np.linspace(40, -40, len(TEAMS)))}
    venues = ["VenueA", "VenueB"]
    rows, ut = [], 0
    for year in SEASONS:
        for rnd in range(1, N_ROUNDS + 1):
            order = rng.permutation(TEAMS)
            for i in range(0, len(order), 2):
                h, a = order[i], order[i + 1]
                ut += 1
                margin = strength[h] - strength[a] + 6 + rng.normal(0, 30)
                total = max(80, 160 + rng.normal(0, 12))
                hscore = int(round((total + margin) / 2))
                ascore = int(round((total - margin) / 2))
                hgoals = max(0, hscore // 6)
                hbehinds = max(0, hscore - hgoals * 6)
                agoals = max(0, ascore // 6)
                abehinds = max(0, ascore - agoals * 6)
                rows.append({
                    "year": year, "round": rnd, "unixtime": ut, "hteam": h, "ateam": a,
                    "venue": rng.choice(venues), "hscore": hscore, "ascore": ascore,
                    "hgoals": hgoals, "hbehinds": hbehinds, "agoals": agoals, "abehinds": abehinds,
                })
    return pd.DataFrame(rows)


def _fryzigg_raw_from_log(player_log: pd.DataFrame) -> pd.DataFrame:
    """Reshape a synthetic player log into the raw Fryzigg schema
    `_fetch_actual_player_log` expects, so the backtest's "actual outcome"
    lookup is self-consistent with the same synthetic data used to build the
    predictions."""
    parts = player_log["player"].str.rpartition(" ")
    first, last = parts[0], parts[2]
    return pd.DataFrame({
        "match_date": pd.to_datetime(player_log["year"], format="%Y"),
        "player_first_name": first,
        "player_last_name": last,
        "match_round": player_log["round"].astype(str),
        "disposals": player_log["disposals"],
        "goals": player_log["goals"],
        "marks": player_log["marks"],
        "tackles": player_log["tackles"],
    })


@pytest.fixture
def synth_world(monkeypatch):
    games = _synth_games()
    player_log = synthetic_player_log(games, players_per_team=12, seed=5)
    fryzigg_raw = _fryzigg_raw_from_log(player_log)
    monkeypatch.setattr(
        "afl_bot.data.fryzigg.fetch_fryzigg_player_stats", lambda: fryzigg_raw,
    )
    return games, player_log


def test_walk_forward_multi_predictions_shape_and_bounds(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
    )
    assert not preds.empty
    assert {"year", "round", "match_id", "legs", "joint_prob", "all_hit", "fair_odds"} <= set(preds.columns)
    assert set(preds["round"]) <= {8, 9}
    assert preds["joint_prob"].between(0.0, 1.0).all()
    assert set(preds["all_hit"].unique()) <= {0, 1}
    # Each graded match contributes at most one rung per target-odds band.
    from afl_bot.config import MULTI_TARGET_ODDS
    assert preds.groupby("match_id").size().max() <= len(MULTI_TARGET_ODDS)


def test_walk_forward_multi_predictions_no_leakage_across_rounds(synth_world):
    """Predictions for round 8 must not change when later rounds' results are
    present in the input `games`/`player_log` — the function truncates
    internally, so round 9+ rows are inert for round 8's prediction."""
    games, player_log = synth_world
    full_preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    truncated_games = games[~((games["year"] == 2024) & (games["round"] > 8))]
    truncated_log = player_log[~((player_log["year"] == 2024) & (player_log["round"] > 8))]
    truncated_preds = walk_forward_multi_predictions(
        truncated_games, truncated_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    pd.testing.assert_frame_equal(
        full_preds.sort_values("match_id").reset_index(drop=True),
        truncated_preds.sort_values("match_id").reset_index(drop=True),
    )


def test_multi_calibration_report_and_reliability_curve(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[7, 8, 9, 10], n_sims=3000, seed=2,
    )
    report = multi_calibration_report(preds)
    assert report["n"] == len(preds)
    assert np.isfinite(report["log_loss"])
    assert np.isfinite(report["brier"])
    assert 0.0 <= report["hit_rate"] <= 1.0

    curve = multi_reliability_curve(preds, n_bins=5)
    assert {"mean_pred", "actual_rate", "n"} <= set(curve.columns)
    assert curve["n"].sum() == len(preds)


def test_multi_calibration_report_empty_input():
    report = multi_calibration_report(pd.DataFrame())
    assert report["n"] == 0
    assert np.isnan(report["log_loss"])


def test_walk_forward_multi_predictions_has_n_legs_hit_and_n_legs(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[7, 8, 9, 10], n_sims=3000, seed=2,
    )
    assert {"n_legs_hit", "n_legs"} <= set(preds.columns)
    assert (preds["n_legs"] == 3).all()                       # search_match_sgms is always 3-leg
    assert preds["n_legs_hit"].between(0, 3).all()
    assert (preds.loc[preds["all_hit"] == 1, "n_legs_hit"] == 3).all()


def test_corr_gain_diagnostic_matches_hand_computed_values():
    # One bucket, 2 rungs: sim says joint=0.50/naive=0.30 (corr_gain +0.20) for
    # both. Actual: rung 1 all 3 legs hit (n_legs_hit=3), rung 2 only 1 of 3
    # hits (n_legs_hit=1) -- pooled empirical leg rate = (3+1)/(3+3) = 0.667,
    # empirical_naive = 0.667**3 ~= 0.296, actual_joint = mean([1, 0]) = 0.5.
    preds = pd.DataFrame({
        "joint_prob": [0.50, 0.50], "naive_product": [0.30, 0.30],
        "corr_gain": [0.20, 0.20], "all_hit": [1, 0],
        "n_legs_hit": [3, 1], "n_legs": [3, 3],
    })
    out = corr_gain_diagnostic(preds, n_bins=1)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n"] == 2
    assert abs(row["sim_joint"] - 0.50) < 1e-9
    assert abs(row["sim_naive"] - 0.30) < 1e-9
    assert abs(row["sim_corr_gain"] - 0.20) < 1e-9
    assert abs(row["actual_joint"] - 0.5) < 1e-9
    pooled_rate = 4 / 6
    assert abs(row["empirical_naive"] - pooled_rate ** 3) < 1e-9
    expected_empirical_gain = 0.5 - pooled_rate ** 3
    assert abs(row["empirical_corr_gain"] - expected_empirical_gain) < 1e-9
    assert abs(row["gap"] - (0.20 - expected_empirical_gain)) < 1e-9


def test_corr_gain_diagnostic_empty_input_and_missing_columns():
    assert corr_gain_diagnostic(pd.DataFrame()).empty
    # joint_prob/all_hit present but no n_legs_hit/n_legs (e.g. an older cache) -> empty, not a crash.
    assert corr_gain_diagnostic(pd.DataFrame({"joint_prob": [0.5], "all_hit": [1],
                                              "naive_product": [0.3]})).empty


def test_corr_gain_diagnostic_real_backtest_runs_end_to_end(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[7, 8, 9, 10], n_sims=3000, seed=2,
    )
    out = corr_gain_diagnostic(preds, n_bins=5)
    assert {"sim_corr_gain", "empirical_corr_gain", "gap"} <= set(out.columns)
    assert out["n"].sum() == len(preds)


def test_haircut_joint_prob_matches_naive_plus_haircut_times_corr_gain():
    preds = pd.DataFrame({"naive_product": [0.3, 0.5], "corr_gain": [0.2, -0.1]})
    out = haircut_joint_prob(preds, 0.5)
    np.testing.assert_allclose(out, [0.3 + 0.5 * 0.2, 0.5 + 0.5 * -0.1])


def test_haircut_joint_prob_clips_to_unit_interval():
    preds = pd.DataFrame({"naive_product": [0.9], "corr_gain": [0.5]})
    assert haircut_joint_prob(preds, 1.0)[0] == 1.0
    preds_low = pd.DataFrame({"naive_product": [0.05], "corr_gain": [-0.5]})
    assert haircut_joint_prob(preds_low, 1.0)[0] == 0.0


def test_fit_corr_gain_haircut_zero_lift_wins_when_data_is_independent():
    # all_hit always matches naive_product's implied independence (corr_gain
    # is irrelevant noise added on top of an already-correct naive estimate)
    # -> the log-loss-minimizing haircut should land near 0.0, not 1.0.
    rng = np.random.default_rng(3)
    n = 4000
    naive = rng.uniform(0.2, 0.6, n)
    corr_gain = rng.normal(0.05, 0.02, n)   # sim systematically adds lift...
    all_hit = (rng.random(n) < naive).astype(float)   # ...that the outcomes never show
    preds = pd.DataFrame({"naive_product": naive, "corr_gain": corr_gain, "all_hit": all_hit})
    fitted = fit_corr_gain_haircut(preds)
    assert fitted < 0.5   # should shrink well below the unhaircut 1.0


def test_fit_corr_gain_haircut_improves_in_sample_log_loss():
    rng = np.random.default_rng(4)
    n = 4000
    naive = rng.uniform(0.2, 0.6, n)
    corr_gain = rng.normal(0.05, 0.02, n)
    all_hit = (rng.random(n) < naive).astype(float)
    preds = pd.DataFrame({"naive_product": naive, "corr_gain": corr_gain, "all_hit": all_hit})
    fitted = fit_corr_gain_haircut(preds)
    baseline_loss = log_loss(haircut_joint_prob(preds, 1.0), all_hit)
    fitted_loss = log_loss(haircut_joint_prob(preds, fitted), all_hit)
    assert fitted_loss <= baseline_loss


def test_fit_corr_gain_haircut_empty_input_returns_unhaircut():
    assert fit_corr_gain_haircut(pd.DataFrame()) == 1.0


def test_walk_forward_multi_predictions_corr_gain_haircut_zero_matches_naive_product(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[7, 8, 9, 10], n_sims=3000, seed=2,
        corr_gain_haircut=0.0,
    )
    assert not preds.empty
    np.testing.assert_allclose(preds["joint_prob"].to_numpy(),
                               preds["naive_product"].to_numpy(), atol=1e-9)


def test_walk_forward_multi_predictions_empty_when_round_not_in_games(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[999], n_sims=2000,
    )
    assert preds.empty


def test_walk_forward_multi_predictions_no_calibration_column_by_default(synth_world):
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    assert "calibrated_joint_prob" not in preds.columns


def test_walk_forward_sgm_candidate_predictions_shape_and_bounds(synth_world):
    games, player_log = synth_world
    preds = walk_forward_sgm_candidate_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
    )
    assert not preds.empty
    assert {"year", "round", "match_id", "legs", "joint_prob", "all_hit", "fair_odds"} <= set(preds.columns)
    assert set(preds["round"]) <= {8, 9}
    assert preds["joint_prob"].between(0.0, 1.0).all()
    assert set(preds["all_hit"].unique()) <= {0, 1}
    # Every match can have many qualifying combos, not at most 3 -- the whole point.
    assert preds.groupby("match_id").size().max() > 3


def test_walk_forward_sgm_candidate_predictions_includes_every_selected_rung(synth_world):
    """Phase 3.5's optimizer's-curse check is only meaningful if the selected
    population is genuinely a subset of the candidate population graded from
    the SAME sim draws -- same seed means `_build_match_legs` consumes the
    rng identically in both walk-forward loops, so every selected rung's
    (match_id, legs) pair must appear in the candidate population with the
    exact same joint_prob."""
    games, player_log = synth_world
    selected = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
    )
    candidates = walk_forward_sgm_candidate_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
    )
    cand_lookup = candidates.set_index(["match_id", "legs"])["joint_prob"]
    for _, row in selected.iterrows():
        assert (row["match_id"], row["legs"]) in cand_lookup.index
        assert cand_lookup.loc[(row["match_id"], row["legs"])] == pytest.approx(row["joint_prob"])


def test_walk_forward_sgm_candidate_predictions_no_leakage_across_rounds(synth_world):
    games, player_log = synth_world
    full_preds = walk_forward_sgm_candidate_predictions(
        games, player_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    truncated_games = games[~((games["year"] == 2024) & (games["round"] > 8))]
    truncated_log = player_log[~((player_log["year"] == 2024) & (player_log["round"] > 8))]
    truncated_preds = walk_forward_sgm_candidate_predictions(
        truncated_games, truncated_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    pd.testing.assert_frame_equal(
        full_preds.sort_values(["match_id", "legs"]).reset_index(drop=True),
        truncated_preds.sort_values(["match_id", "legs"]).reset_index(drop=True),
    )


def test_walk_forward_sgm_candidate_predictions_empty_when_round_not_in_games(synth_world):
    games, player_log = synth_world
    preds = walk_forward_sgm_candidate_predictions(
        games, player_log, eval_year=2024, rounds=[999], n_sims=2000,
    )
    assert preds.empty


def test_walk_forward_multi_predictions_lcb_z_and_price_shrink_default_off(synth_world):
    """lcb_z=0, price_shrink=0 (the defaults) must reproduce the unhaircut
    selected population exactly -- both knobs are opt-in."""
    games, player_log = synth_world
    baseline = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
    )
    explicit_off = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
        lcb_z=0.0, price_shrink=0.0,
    )
    pd.testing.assert_frame_equal(baseline, explicit_off)


def test_walk_forward_multi_predictions_price_shrink_lands_exactly_on_target_odds(synth_world):
    """Fully shrunk (price_shrink=1.0): every selected rung's joint_prob is
    set to exactly 1/target for its rung's target -- so its fair_odds lands
    exactly on one of MULTI_TARGET_ODDS (no odds_book is passed anywhere in
    this backtest, so every rung goes through the target-distance selection,
    none through the book-edge value-pick branch)."""
    games, player_log = synth_world
    shrunk = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
        price_shrink=1.0,
    )
    assert not shrunk.empty
    for fo in shrunk["fair_odds"]:
        assert any(fo == pytest.approx(t) for t in MULTI_TARGET_ODDS)


def test_walk_forward_sim_prop_predictions_shape_and_bounds(synth_world):
    games, player_log = synth_world
    preds = walk_forward_sim_prop_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
    )
    assert not preds.empty
    assert list(preds.columns) == ["year", "round", "player", "stat", "line", "prob", "actual"]
    assert set(preds["round"]) <= {8, 9}
    assert set(preds["stat"]) <= {"disposals", "goals", "marks", "tackles"}
    assert set(preds["actual"].unique()) <= {0, 1}
    # _build_match_legs gates every leg to the same bettable window live pricing uses.
    assert preds["prob"].between(LEG_PROB_MIN, LEG_PROB_MAX, inclusive="neither").all()


def test_walk_forward_sim_prop_predictions_no_leakage_across_rounds(synth_world):
    games, player_log = synth_world
    full_preds = walk_forward_sim_prop_predictions(
        games, player_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    truncated_games = games[~((games["year"] == 2024) & (games["round"] > 8))]
    truncated_log = player_log[~((player_log["year"] == 2024) & (player_log["round"] > 8))]
    truncated_preds = walk_forward_sim_prop_predictions(
        truncated_games, truncated_log, eval_year=2024, rounds=[8], n_sims=3000, seed=1,
    )
    pd.testing.assert_frame_equal(
        full_preds.sort_values(["player", "stat", "line"]).reset_index(drop=True),
        truncated_preds.sort_values(["player", "stat", "line"]).reset_index(drop=True),
    )


def test_walk_forward_sim_prop_predictions_empty_when_round_not_in_games(synth_world):
    games, player_log = synth_world
    preds = walk_forward_sim_prop_predictions(
        games, player_log, eval_year=2024, rounds=[999], n_sims=2000,
    )
    assert preds.empty


def test_fit_prop_calibrators_works_on_sim_predictions(synth_world):
    """Phase 3.1's whole point: `fit_prop_calibrators`/`apply_prop_calibration`
    are agnostic to which walk-forward source produced (stat, line, prob,
    actual) rows -- the real-sim predictions slot in unchanged."""
    games, player_log = synth_world
    preds = walk_forward_sim_prop_predictions(
        games, player_log, eval_year=2024, rounds=[7, 8, 9, 10], n_sims=3000, seed=2,
    )
    cals = fit_prop_calibrators(preds, min_samples=5)
    assert cals
    calibrated = np.array([apply_prop_calibration(cals, stat, line, p)
                           for p, stat, line in zip(preds["prob"], preds["stat"], preds["line"])])
    raw = log_loss(preds["prob"].to_numpy(), preds["actual"].to_numpy(dtype=float))
    cal = log_loss(calibrated, preds["actual"].to_numpy(dtype=float))
    assert cal <= raw + 1e-9             # isotonic can't worsen in-sample log loss


def test_walk_forward_multi_predictions_calibration_on_adds_bounded_column(synth_world):
    games, player_log = synth_world
    # Halves every probability: predict(p) = p * 0.5 (linear interp from (0,0) to (1,0.5)).
    halving = IsotonicCalibrator(x_=np.array([0.0, 1.0]), y_=np.array([0.0, 0.5]))
    calibrators = {stat: {"pooled": halving, "lines": {}}
                   for stat in ("disposals", "goals", "marks", "tackles")}

    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9], n_sims=3000, seed=1,
        prop_calibrators=calibrators,
    )
    assert not preds.empty
    assert {"calibrated_joint_prob", "calibrated_naive_product"} <= set(preds.columns)
    assert preds["calibrated_joint_prob"].between(0.0, 1.0).all()

    # calibrated_joint_prob = clip(calibrated_naive_product + corr_gain, 0, 1) exactly.
    expected = (preds["calibrated_naive_product"] + preds["corr_gain"]).clip(0.0, 1.0)
    pd.testing.assert_series_equal(preds["calibrated_joint_prob"], expected, check_names=False)

    # Halving every leg's prob roughly halves the naive product per leg (3 legs ->
    # ~1/8th), so the calibrated naive product should be far below the raw one.
    assert (preds["calibrated_naive_product"] < preds["naive_product"]).all()


def test_multi_calibration_report_and_curve_support_column_param(synth_world):
    games, player_log = synth_world
    halving = IsotonicCalibrator(x_=np.array([0.0, 1.0]), y_=np.array([0.0, 0.5]))
    calibrators = {stat: {"pooled": halving, "lines": {}}
                   for stat in ("disposals", "goals", "marks", "tackles")}
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[8, 9, 10], n_sims=3000, seed=1,
        prop_calibrators=calibrators,
    )

    raw_report = multi_calibration_report(preds)
    cal_report = multi_calibration_report(preds, column="calibrated_joint_prob")
    assert raw_report["n"] == cal_report["n"] == len(preds)
    assert raw_report["mean_pred"] != cal_report["mean_pred"]

    raw_curve = multi_reliability_curve(preds)
    cal_curve = multi_reliability_curve(preds, column="calibrated_joint_prob")
    assert raw_curve["n"].sum() == cal_curve["n"].sum() == len(preds)


def test_multi_calibration_report_missing_column_returns_empty():
    preds = pd.DataFrame({"joint_prob": [0.5], "all_hit": [1]})
    report = multi_calibration_report(preds, column="calibrated_joint_prob")
    assert report["n"] == 0
    assert np.isnan(report["log_loss"])


def test_fit_multi_calibrator_improves_in_sample(synth_world):
    """Model-upgrade audit Phase 3.6: an IsotonicCalibrator fit directly on
    the SELECTED-rung walk-forward predictions (not the all-candidates pool)
    can't worsen in-sample log loss -- the same guarantee
    test_fit_prop_calibrators_improve_in_sample checks at the leg level."""
    games, player_log = synth_world
    preds = walk_forward_multi_predictions(
        games, player_log, eval_year=2024, rounds=[7, 8, 9, 10], n_sims=3000, seed=2,
    )
    cal = fit_multi_calibrator(preds)
    raw = log_loss(preds["joint_prob"].to_numpy(), preds["all_hit"].to_numpy(dtype=float))
    calibrated = log_loss(cal.predict(preds["joint_prob"].to_numpy()), preds["all_hit"].to_numpy(dtype=float))
    assert calibrated <= raw + 1e-9


def test_load_or_fit_multi_calibrator_caches(tmp_path, synth_world):
    games, player_log = synth_world
    cal = load_or_fit_multi_calibrator(
        games, player_log, eval_start_year=2022, eval_end_year=2024, n_sims=3000,
        cache_dir=tmp_path, seed=1,
    )
    assert cal is not None
    assert (tmp_path / "multi_calibrator.json").exists()
    again = load_or_fit_multi_calibrator(
        games, player_log, eval_start_year=2022, eval_end_year=2024, n_sims=3000,
        cache_dir=tmp_path, seed=1,
    )
    p = 0.4
    assert again.predict([p])[0] == cal.predict([p])[0]


def test_load_or_fit_multi_calibrator_reuses_season_preds_cache(tmp_path, synth_world):
    """The shared season_preds_cache must hold every season actually
    backtested (so a caller fitting overlapping windows for adjacent eval
    years, e.g. grade-multis across multiple years, doesn't recompute the
    same season's full sim backtest twice)."""
    games, player_log = synth_world
    cache: dict[int, pd.DataFrame] = {}
    load_or_fit_multi_calibrator(
        games, player_log, eval_start_year=2022, eval_end_year=2024, n_sims=3000,
        cache_dir=tmp_path, force_refresh=True, seed=1, season_preds_cache=cache,
    )
    assert set(cache) == {2022, 2023}
    cached_2022 = cache[2022]
    # Refitting an overlapping window (2023-2024) must reuse the cached 2023
    # predictions and only compute the new season (2024) -- not recompute 2023.
    load_or_fit_multi_calibrator(
        games, player_log, eval_start_year=2023, eval_end_year=2025, n_sims=3000,
        cache_dir=tmp_path, force_refresh=True, seed=1, season_preds_cache=cache,
    )
    assert set(cache) == {2022, 2023, 2024}
    pd.testing.assert_frame_equal(cache[2022], cached_2022)


def test_load_or_fit_multi_calibrator_empty_when_no_history(tmp_path, synth_world):
    games, player_log = synth_world
    cal = load_or_fit_multi_calibrator(
        games, player_log, eval_start_year=2018, eval_end_year=2019, n_sims=2000, seed=1,
        cache_dir=tmp_path,
    )
    assert cal is None


def test_apply_multi_calibration_no_op_when_calibrator_none():
    sgms = [{"joint_prob": 0.3, "fair_odds": 3.33}]
    out = apply_multi_calibration(sgms, None)
    assert out[0]["joint_prob"] == 0.3
    assert out[0]["fair_odds"] == 3.33


def test_apply_multi_calibration_recomputes_fair_odds_and_edge():
    halving = IsotonicCalibrator(x_=np.array([0.0, 1.0]), y_=np.array([0.0, 0.5]))
    sgms = [{"joint_prob": 0.4, "fair_odds": 2.5, "book_odds": 3.0, "raw_edge": 0.2, "edge": 0.1}]
    out = apply_multi_calibration(sgms, halving)
    assert out[0]["joint_prob"] == pytest.approx(0.2)             # halved
    assert out[0]["fair_odds"] == pytest.approx(5.0)              # 1/0.2
    assert out[0]["raw_edge"] == pytest.approx(0.2 * 3.0 - 1.0)   # recomputed off the calibrated prob
    assert out[0]["odds"] == 3.0

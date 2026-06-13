import numpy as np
import pandas as pd

from afl_bot.backtest.ensemble import (
    IsotonicCalibrator,
    MarketBlend,
    assemble_signals,
    blend_probabilities,
    ensemble_report,
    fit_blend_weights,
    fit_market_blend,
    squiggle_consensus,
)
from afl_bot.backtest.walkforward import log_loss


# --------------------------------------------------------------------------- #
# Isotonic calibration
# --------------------------------------------------------------------------- #
def test_isotonic_is_monotone_and_improves_in_sample_log_loss():
    rng = np.random.default_rng(0)
    n = 4000
    raw = rng.uniform(0.5, 1.0, n)
    true_p = 0.5 + 0.4 * (raw - 0.5)        # raw is overconfident vs reality
    outcomes = (rng.random(n) < true_p).astype(float)

    cal = IsotonicCalibrator().fit(raw, outcomes)
    pred = cal.predict(raw)

    # monotone in the input
    order = np.argsort(raw)
    assert np.all(np.diff(pred[order]) >= -1e-9)
    # calibration lowers (in-sample) log loss vs the raw overconfident probs
    assert log_loss(pred, outcomes) < log_loss(raw, outcomes)


def test_isotonic_predict_clamps_and_unfitted_is_identity():
    cal = IsotonicCalibrator().fit([0.2, 0.5, 0.8], [0.0, 1.0, 1.0])
    assert 0.0 <= cal.predict([0.0])[0] <= 1.0
    assert 0.0 <= cal.predict([1.0])[0] <= 1.0
    assert IsotonicCalibrator().predict([0.3])[0] == 0.3   # unfitted -> identity


# --------------------------------------------------------------------------- #
# Convex blend
# --------------------------------------------------------------------------- #
def test_fit_blend_weights_on_simplex_and_favours_better_signal():
    rng = np.random.default_rng(1)
    n = 3000
    true_p = rng.uniform(0.2, 0.8, n)
    outcomes = (rng.random(n) < true_p).astype(float)
    good = true_p                              # near-perfect signal
    useless = np.full(n, 0.5)                  # no information
    w = fit_blend_weights(np.column_stack([good, useless]), outcomes)

    assert abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-9).all()
    assert w[0] > 0.8                          # weight concentrates on the good signal


def test_blend_probabilities_in_range():
    out = blend_probabilities(np.array([[0.2, 0.9], [0.6, 0.4]]), np.array([0.5, 0.5]))
    assert ((out >= 0) & (out <= 1)).all()
    assert np.allclose(out, [0.55, 0.5])


# --------------------------------------------------------------------------- #
# Squiggle consensus
# --------------------------------------------------------------------------- #
def test_squiggle_consensus_averages_excluding_aggregate():
    tips = pd.DataFrame([
        {"year": 2024, "round": 1, "hteam": "A", "ateam": "B", "source": "M1", "hconfidence": 60.0},
        {"year": 2024, "round": 1, "hteam": "A", "ateam": "B", "source": "M2", "hconfidence": 80.0},
        {"year": 2024, "round": 1, "hteam": "A", "ateam": "B", "source": "Aggregate", "hconfidence": 99.0},
    ])
    cons = squiggle_consensus(tips)
    assert len(cons) == 1
    # mean of 60 and 80 (Aggregate excluded) / 100 = 0.70
    assert abs(cons.iloc[0]["squiggle_home_prob"] - 0.70) < 1e-9


# --------------------------------------------------------------------------- #
# Assemble / fit / report on a synthetic season
# --------------------------------------------------------------------------- #
def _synth(seasons=(2021, 2022, 2023), n_per=80, seed=3):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    strength = {t: s for t, s in zip(teams, np.linspace(40, -40, len(teams)))}
    grows, orows, trows = [], [], []
    ut = 0
    for year in seasons:
        for rnd in range(1, n_per + 1):
            h, a = rng.choice(teams, 2, replace=False)
            ut += 1
            edge = strength[h] - strength[a] + 8
            true_p = 1 / (1 + 10 ** (-edge / 80))
            home_win = rng.random() < true_p
            hs, as_ = (100, 80) if home_win else (80, 100)
            grows.append({"year": year, "round": rnd, "unixtime": ut,
                          "hteam": h, "ateam": a, "hscore": hs, "ascore": as_})
            # market closing odds that devig back to ~true_p (5% vig)
            orows.append({"year": year, "hteam": h, "ateam": a,
                          "home_odds_close": 1 / (0.975 * true_p),
                          "away_odds_close": 1 / (0.975 * (1 - true_p))})
            trows.append({"year": year, "round": rnd, "hteam": h, "ateam": a,
                          "source": "Crowd", "hconfidence": true_p * 100 + rng.normal(0, 5)})
    return pd.DataFrame(grows), pd.DataFrame(orows), pd.DataFrame(trows)


def test_assemble_signals_has_all_columns_and_row_count():
    games, odds, tips = _synth()
    sig = assemble_signals(games, odds, tips)
    assert len(sig) == len(games)
    assert {"model_p", "market_p", "squiggle_p", "outcome"} <= set(sig.columns)


def test_market_blend_predict_renormalises_over_available_signals():
    games, odds, tips = _synth()
    blend = fit_market_blend(assemble_signals(games, odds, tips))
    # model-only prediction -> calibrated model (single signal, weight renormalised)
    only_model = blend.predict_home_prob(0.6)
    assert 0.0 <= only_model[0] <= 1.0
    # with market supplied, the blend pulls toward it
    with_market = blend.predict_home_prob(0.6, market_p=0.8)
    assert with_market[0] != only_model[0]


def test_ensemble_report_blend_beats_model_out_of_sample():
    games, odds, tips = _synth()
    rep = ensemble_report(games, odds, tips, train_end_year=2022, eval_start_year=2023)
    assert rep["n_holdout"] > 0
    # the market-anchored blend should not be worse than the raw model
    assert rep["log_loss_blend"] <= rep["log_loss_model"] + 1e-9
    assert abs(sum(rep["weights"].values()) - 1.0) < 1e-6

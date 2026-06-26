"""Tests for Phase-1 weather greasiness factor and continuous multipliers."""

import numpy as np
import pandas as pd
import pytest

from afl_bot.config import (
    GREASINESS_RAIN_MM_MAX,
    GREASINESS_TEMP_COLD_C,
    GREASINESS_TEMP_NEUTRAL_C,
    GREASINESS_WEIGHTS,
)
from afl_bot.data.weather import attach_weather
from afl_bot.models.weather_effects import (
    DEFAULT_RAIN_MULTIPLIERS,
    greasiness_factor,
    greasiness_multiplier,
)
from afl_bot.sim.engine import Team, make_rng, simulate_match

N = 100_000


# --------------------------------------------------------------------------- #
# greasiness_factor — unit tests
# --------------------------------------------------------------------------- #

def test_greasiness_factor_roofed_always_zero():
    assert greasiness_factor(50.0, 0.0, -10.0, 100.0, roofed=True) == 0.0


def test_greasiness_factor_perfectly_dry_warm_no_wind_is_zero():
    # hot, dry, calm
    g = greasiness_factor(0.0, 30.0, 29.0, 0.0, roofed=False)
    assert g == 0.0


def test_greasiness_factor_maxes_at_one():
    # extreme conditions cap at 1.0, not beyond
    g = greasiness_factor(100.0, -5.0, -20.0, 200.0, roofed=False)
    assert g <= 1.0


def test_greasiness_factor_heavy_rain_contributes_rain_weight():
    w_rain = GREASINESS_WEIGHTS[0]
    g = greasiness_factor(
        GREASINESS_RAIN_MM_MAX, GREASINESS_TEMP_NEUTRAL_C, GREASINESS_TEMP_NEUTRAL_C,
        0.0, roofed=False,
    )
    assert abs(g - w_rain) < 1e-9


def test_greasiness_factor_cold_with_no_rain():
    # only cold component active
    w_cold = GREASINESS_WEIGHTS[1]
    g = greasiness_factor(0.0, GREASINESS_TEMP_COLD_C, GREASINESS_TEMP_COLD_C, 0.0, roofed=False)
    assert abs(g - w_cold) < 1e-9


def test_greasiness_factor_nan_temp_gives_zero_cold_and_dew():
    nan = float("nan")
    g_nan_temp = greasiness_factor(5.0, nan, nan, 0.0, roofed=False)
    g_warm_temp = greasiness_factor(5.0, 25.0, 24.0, 0.0, roofed=False)
    # NaN temp → no cold/dew contribution; rain contribution is the same
    assert g_nan_temp == g_warm_temp


def test_greasiness_factor_increases_with_worsening_conditions():
    dry_warm = greasiness_factor(0.0, 22.0, 21.0, 5.0, roofed=False)
    damp_cool = greasiness_factor(3.0, 15.0, 12.0, 20.0, roofed=False)
    heavy_cold = greasiness_factor(12.0, 10.0, 5.0, 40.0, roofed=False)
    assert dry_warm < damp_cool < heavy_cold


def test_greasiness_factor_range():
    for _ in range(200):
        rng = np.random.default_rng(42)
        rain = rng.uniform(0, 20)
        temp = rng.uniform(-5, 35)
        app_temp = temp - rng.uniform(0, 8)
        wind = rng.uniform(0, 80)
        g = greasiness_factor(rain, temp, app_temp, wind, roofed=False)
        assert 0.0 <= g <= 1.0


# --------------------------------------------------------------------------- #
# greasiness_multiplier — unit tests
# --------------------------------------------------------------------------- #

def test_greasiness_multiplier_zero_is_neutral():
    for stat in ("disposals", "marks", "tackles", "goals"):
        assert greasiness_multiplier(stat, 0.0) == 1.0


def test_greasiness_multiplier_roofed_is_neutral():
    assert greasiness_multiplier("marks", 1.0, roofed=True) == 1.0


def test_greasiness_multiplier_full_wet_equals_rain_multiplier_endpoint():
    for stat, endpoint in DEFAULT_RAIN_MULTIPLIERS.items():
        m = greasiness_multiplier(stat, 1.0, roofed=False)
        assert abs(m - endpoint) < 1e-9, f"{stat}: {m} != {endpoint}"


def test_greasiness_multiplier_half_wet_is_midpoint():
    for stat, endpoint in DEFAULT_RAIN_MULTIPLIERS.items():
        expected_mid = 1.0 + 0.5 * (endpoint - 1.0)
        m = greasiness_multiplier(stat, 0.5)
        assert abs(m - expected_mid) < 1e-9


def test_greasiness_multiplier_unknown_stat_returns_one():
    assert greasiness_multiplier("undefined_stat", 1.0) == 1.0


def test_greasiness_multiplier_marks_suppressed_more_than_disposals():
    # marks endpoint 0.85 < disposals endpoint 0.93 → marks more suppressed
    m_marks = greasiness_multiplier("marks", 0.7)
    m_disp = greasiness_multiplier("disposals", 0.7)
    assert m_marks < m_disp


# --------------------------------------------------------------------------- #
# simulate_match greasiness — integration
# --------------------------------------------------------------------------- #

def test_simulate_match_greasiness_zero_matches_dry_baseline():
    rng = make_rng(seed=99)
    out = simulate_match(
        Team("H", True), Team("A"), mu_margin=0.0, mu_total=162.0,
        home_accuracy=0.525, away_accuracy=0.525, n=N, rng=rng, greasiness=0.0,
    )
    total_mean = (out["home_pts"] + out["away_pts"]).mean()
    assert abs(total_mean - 162.0) < 4.0  # within noise


def test_simulate_match_greasiness_scales_total_continuously():
    kw = dict(mu_margin=0.0, mu_total=162.0, home_accuracy=0.525, away_accuracy=0.525, n=N)
    rng = make_rng(seed=7)
    dry = simulate_match(Team("H", True), Team("A"), rng=rng, greasiness=0.0, **kw)
    rng = make_rng(seed=7)
    half = simulate_match(Team("H", True), Team("A"), rng=rng, greasiness=0.5, **kw)
    rng = make_rng(seed=7)
    full = simulate_match(Team("H", True), Team("A"), rng=rng, greasiness=1.0, **kw)

    dry_t = (dry["home_pts"] + dry["away_pts"]).mean()
    half_t = (half["home_pts"] + half["away_pts"]).mean()
    full_t = (full["home_pts"] + full["away_pts"]).mean()

    assert full_t < half_t < dry_t           # monotone suppression
    assert full_t / dry_t < 0.96             # at least ~4% down at full greasiness


# --------------------------------------------------------------------------- #
# attach_weather greasiness column
# --------------------------------------------------------------------------- #

GAMES = pd.DataFrame([
    {"year": 2024, "round": 1, "hteam": "Geelong", "ateam": "Carlton",
     "venue": "M.C.G.", "date": "2024-04-01"},
    {"year": 2024, "round": 1, "hteam": "Essendon", "ateam": "Sydney",
     "venue": "Marvel Stadium", "date": "2024-04-02"},
])


def test_attach_weather_greasiness_column_present():
    weather = pd.DataFrame([
        {"venue": "M.C.G.", "date": "2024-04-01", "rain_mm": 8.0, "wind_kmh": 25.0,
         "temp_c": 10.0, "apparent_temp_c": 6.0},
        {"venue": "Marvel Stadium", "date": "2024-04-02", "rain_mm": 30.0, "wind_kmh": 50.0,
         "temp_c": 8.0, "apparent_temp_c": 2.0},
    ])
    out = attach_weather(GAMES, weather=weather)
    assert "greasiness" in out.columns
    assert out["greasiness"].between(0.0, 1.0).all()


def test_attach_weather_roofed_venue_greasiness_zero():
    weather = pd.DataFrame([
        {"venue": "M.C.G.", "date": "2024-04-01", "rain_mm": 0.0, "wind_kmh": 5.0,
         "temp_c": 20.0, "apparent_temp_c": 19.0},
        {"venue": "Marvel Stadium", "date": "2024-04-02", "rain_mm": 30.0, "wind_kmh": 50.0,
         "temp_c": 8.0, "apparent_temp_c": 2.0},
    ])
    out = attach_weather(GAMES, weather=weather)
    marvel = out[out["venue"] == "Marvel Stadium"].iloc[0]
    assert marvel["greasiness"] == 0.0  # roofed -> always dry regardless of weather


def test_attach_weather_cold_dry_game_has_positive_greasiness():
    weather = pd.DataFrame([
        {"venue": "M.C.G.", "date": "2024-04-01", "rain_mm": 0.0, "wind_kmh": 15.0,
         "temp_c": 9.0, "apparent_temp_c": 4.0},
    ])
    out = attach_weather(GAMES.head(1), weather=weather)
    mcg = out[out["venue"] == "M.C.G."].iloc[0]
    # Dry but cold and windy → greasiness > 0 even though is_wet=False
    assert not mcg["is_wet"]
    assert mcg["greasiness"] > 0.0

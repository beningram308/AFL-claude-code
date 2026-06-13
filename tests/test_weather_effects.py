import numpy as np
import pandas as pd

from afl_bot.models.weather_effects import (
    DEFAULT_RAIN_MULTIPLIERS,
    fit_rain_multipliers,
    fit_wet_total_ratio,
    rain_multiplier,
)


def test_fit_wet_total_ratio_recovers_wet_suppression():
    rng = np.random.default_rng(0)
    n = 300
    is_wet = rng.random(n) < 0.3
    total = np.where(is_wet, rng.normal(150, 10, n), rng.normal(165, 10, n))
    df = pd.DataFrame({"hscore": total / 2, "ascore": total / 2, "is_wet": is_wet})
    ratio = fit_wet_total_ratio(df, min_wet_games=20)
    assert 0.88 < ratio < 0.96            # ~150/165


def test_fit_wet_total_ratio_falls_back_when_thin():
    df = pd.DataFrame({"hscore": [80, 90], "ascore": [80, 90], "is_wet": [False, True]})
    assert fit_wet_total_ratio(df, min_wet_games=20, default=0.93) == 0.93


def test_default_multiplier_directions():
    assert DEFAULT_RAIN_MULTIPLIERS["tackles"] > 1.0   # tackles up in the wet
    assert DEFAULT_RAIN_MULTIPLIERS["disposals"] < 1.0
    assert DEFAULT_RAIN_MULTIPLIERS["marks"] < 1.0
    assert DEFAULT_RAIN_MULTIPLIERS["goals"] < 1.0


def test_rain_multiplier_dry_and_roofed_are_neutral():
    assert rain_multiplier("marks", is_wet=False) == 1.0
    assert rain_multiplier("marks", is_wet=True, roofed=True) == 1.0
    assert rain_multiplier("marks", is_wet=True, roofed=False) == DEFAULT_RAIN_MULTIPLIERS["marks"]
    assert rain_multiplier("unknown_stat", is_wet=True) == 1.0


def test_fit_rain_multipliers_recovers_a_wet_suppression():
    rng = np.random.default_rng(0)
    n = 400
    is_wet = rng.random(n) < 0.4
    # disposals ~20% lower in the wet, tackles ~10% higher
    disposals = np.where(is_wet, rng.normal(320, 20, n), rng.normal(400, 20, n))
    tackles = np.where(is_wet, rng.normal(66, 5, n), rng.normal(60, 5, n))
    df = pd.DataFrame({"disposals": disposals, "tackles": tackles, "is_wet": is_wet})

    fitted = fit_rain_multipliers(df, ["disposals", "tackles"], min_wet_games=20)
    assert fitted["disposals"] < 0.9      # recovered the suppression
    assert fitted["tackles"] > 1.05       # recovered the lift


def test_fit_rain_multipliers_falls_back_when_too_few_wet_games():
    df = pd.DataFrame({"disposals": [400, 390, 410], "is_wet": [False, False, True]})
    fitted = fit_rain_multipliers(df, ["disposals"], min_wet_games=20)
    assert fitted["disposals"] == DEFAULT_RAIN_MULTIPLIERS["disposals"]


def test_fit_rain_multipliers_empty_returns_defaults():
    assert fit_rain_multipliers(pd.DataFrame()) == DEFAULT_RAIN_MULTIPLIERS

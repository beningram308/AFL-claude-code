"""
Wet-weather multipliers for player props (plan §3.4).

Rain suppresses disposals (especially uncontested) and marks, lifts tackles,
and lowers goals/accuracy. This module supplies per-stat multipliers — fit from
history where weather is attached, or sensible published-research defaults
otherwise — that plug into the existing ``context_mult`` hook in
``afl_bot.models.props.expected_stat_mean`` (and the share/goal paths in the
CLI). Roofed grounds and dry games get a neutral 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afl_bot.config import (
    GREASINESS_DEW_SPREAD_C,
    GREASINESS_RAIN_MM_MAX,
    GREASINESS_TEMP_COLD_C,
    GREASINESS_TEMP_NEUTRAL_C,
    GREASINESS_WEIGHTS,
    GREASINESS_WIND_MAX_KMH,
    WET_MARKS_MULTIPLIER,
)

# Direction & rough magnitude from AFL wet-weather research (plan §3.4):
# disposals/marks/goals down, tackles up. These describe GENUINELY wet *play*.
#
# NOTE: fitting these from Open-Meteo *daily* rainfall gives much weaker numbers
# (disposals ~0.99, marks ~0.96, tackles ~1.01 on 2022-25 AFL games) because a
# daily total is a noisy proxy for conditions at the bounce — it can rain all
# morning and be dry by an evening game. Marks are the one stat with a clear
# empirical signal (~4-5% down) even through that noise. So we keep these
# research defaults as the wet-play scenario the user prices (via the CLI
# ``--rain-mm`` flag), rather than the attenuated daily-rainfall fit.
DEFAULT_RAIN_MULTIPLIERS: dict[str, float] = {
    "disposals": 0.93,
    "marks": WET_MARKS_MULTIPLIER,
    "tackles": 1.08,
    "goals": 0.92,
}


def fit_rain_multipliers(
    stat_games: pd.DataFrame, stats: list[str] | None = None, *,
    wet_col: str = "is_wet", min_wet_games: int = 20,
    defaults: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fit per-stat wet/dry multipliers as ``mean(stat | wet) / mean(stat | dry)``.

    ``stat_games`` is one row per observation (team-game or player-game) with a
    boolean ``wet_col`` and the named ``stats`` columns. Stats with fewer than
    ``min_wet_games`` wet observations (or a non-finite ratio) fall back to
    ``defaults`` (``DEFAULT_RAIN_MULTIPLIERS``), so a short or all-dry history
    can't produce a garbage multiplier.
    """
    defaults = defaults or DEFAULT_RAIN_MULTIPLIERS
    stats = stats or list(defaults)
    if stat_games.empty or wet_col not in stat_games.columns:
        return dict(defaults)

    wet = stat_games[stat_games[wet_col]]
    dry = stat_games[~stat_games[wet_col]]

    out: dict[str, float] = {}
    for stat in stats:
        fallback = defaults.get(stat, 1.0)
        if stat not in stat_games.columns or len(wet) < min_wet_games or dry.empty:
            out[stat] = fallback
            continue
        dry_mean = dry[stat].mean()
        wet_mean = wet[stat].mean()
        ratio = wet_mean / dry_mean if dry_mean and np.isfinite(dry_mean) else float("nan")
        out[stat] = float(ratio) if np.isfinite(ratio) and ratio > 0 else fallback
    return out


def fit_wet_total_ratio(games_with_weather: pd.DataFrame, *, wet_col: str = "is_wet",
                        min_wet_games: int = 20, default: float = 0.93) -> float:
    """Fit the match-level wet total multiplier as mean(total | wet) /
    mean(total | dry) from games carrying ``hscore``/``ascore`` + a boolean
    ``wet_col`` (round-2 §4.1). Falls back to ``default`` on a thin sample.
    On 2022-25 daily data this lands ~0.92-0.94; refit on hourly rain (§4.3)."""
    if games_with_weather.empty or wet_col not in games_with_weather.columns:
        return default
    df = games_with_weather.copy()
    df["_total"] = df["hscore"] + df["ascore"]
    wet, dry = df[df[wet_col]], df[~df[wet_col]]
    if len(wet) < min_wet_games or dry.empty:
        return default
    dry_mean = dry["_total"].mean()
    ratio = wet["_total"].mean() / dry_mean if dry_mean else float("nan")
    return float(ratio) if np.isfinite(ratio) and ratio > 0 else default


def rain_multiplier(stat: str, is_wet: bool, roofed: bool = False,
                    multipliers: dict[str, float] | None = None) -> float:
    """Context multiplier for ``stat`` given conditions: 1.0 when the venue is
    roofed or the game is dry, else the (fitted or default) wet multiplier."""
    if roofed or not is_wet:
        return 1.0
    multipliers = multipliers or DEFAULT_RAIN_MULTIPLIERS
    return float(multipliers.get(stat, 1.0))


def greasiness_factor(
    rain_mm: float,
    temp_c: float,
    apparent_temp_c: float,
    wind_kmh: float,
    roofed: bool = False,
) -> float:
    """Continuous 0.0–1.0 greasiness score blending rain, cold, dew proximity,
    and wind. Roofed venues always return 0.0. NaN inputs contribute 0 for their
    component rather than crashing — a missing temperature reading doesn't force
    a greasy classification.

    Intended as the single greasiness signal flowing into ``greasiness_multiplier``
    and ``simulate_match``; replaces the binary ``is_wet`` flag (Phase 1).
    """
    if roofed:
        return 0.0

    r = rain_mm if np.isfinite(rain_mm) else 0.0
    w = wind_kmh if np.isfinite(wind_kmh) else 0.0

    rain_g = min(1.0, r / GREASINESS_RAIN_MM_MAX)

    if np.isfinite(temp_c):
        span = GREASINESS_TEMP_NEUTRAL_C - GREASINESS_TEMP_COLD_C
        cold_g = max(0.0, min(1.0, (GREASINESS_TEMP_NEUTRAL_C - temp_c) / span))
    else:
        cold_g = 0.0

    # Dew/humidity slipperiness only matters in cold conditions; a warm night
    # with minor wind chill doesn't make the ball greasy.
    if np.isfinite(temp_c) and np.isfinite(apparent_temp_c) and temp_c < GREASINESS_TEMP_NEUTRAL_C:
        spread = max(0.0, float(temp_c) - float(apparent_temp_c))
        dew_g = min(1.0, spread / GREASINESS_DEW_SPREAD_C)
    else:
        dew_g = 0.0

    wind_g = min(1.0, w / GREASINESS_WIND_MAX_KMH)

    w_rain, w_cold, w_dew, w_wind = GREASINESS_WEIGHTS
    return w_rain * rain_g + w_cold * cold_g + w_dew * dew_g + w_wind * wind_g


def greasiness_multiplier(
    stat: str,
    greasiness: float,
    roofed: bool = False,
    multipliers: dict[str, float] | None = None,
) -> float:
    """Per-stat multiplier scaled continuously from 1.0 at greasiness=0.0 to the
    heavy-wet endpoint (``DEFAULT_RAIN_MULTIPLIERS``) at greasiness=1.0. Roofed
    venues and zero greasiness always return 1.0."""
    if roofed or greasiness <= 0.0:
        return 1.0
    multipliers = multipliers or DEFAULT_RAIN_MULTIPLIERS
    wet_end = float(multipliers.get(stat, 1.0))
    return 1.0 + float(greasiness) * (wet_end - 1.0)

"""
Player props model (plan §3.3) — usage share -> expected stat -> Negative Binomial.

This module is deliberately data-source agnostic: it operates on a "player game
log" DataFrame with one row per player per game and (at minimum) the columns:

    year, round, unixtime, player, team, opponent, is_home, <stat columns...>

where ``<stat columns>`` are raw counting stats such as ``disposals``, ``goals``,
``marks``, ``tackles``. A ``team_<stat>`` column (the team's total for that stat in
that game) is used to compute usage share; if absent it is derived by summing the
stat across all of a team's players in that game.

Real data should come from fitzRoy / afl_tables / Footywire per plan §2 — wire a
loader in ``afl_bot/data/player_stats.py`` that returns a DataFrame in this shape
and everything below works unchanged. ``afl_bot.data.player_stats`` ships a small
synthetic generator so the rest of the pipeline is testable without that data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afl_bot.config import PROP_EWMA_HALFLIFE, PROP_MIN_DISPERSION, PROP_RECENT_SEASONS


def _ewma_last(series: pd.Series, halflife: float) -> float:
    if series.empty:
        return float("nan")
    return float(series.ewm(halflife=halflife, adjust=True).mean().iloc[-1])


def _ensure_team_totals(log: pd.DataFrame, stat: str) -> pd.DataFrame:
    col = f"team_{stat}"
    if col in log.columns:
        return log
    totals = log.groupby(["year", "round", "team"])[stat].transform("sum")
    log = log.copy()
    log[col] = totals
    return log


def player_rate_profile(
    log: pd.DataFrame, player: str, stat: str,
    as_of_year: int | None = None, as_of_round: int | None = None,
    halflife: float = PROP_EWMA_HALFLIFE, lookback_games: int = 20,
) -> dict[str, float]:
    """EWMA baseline rate + usage share for one player/stat, using only games
    strictly before (as_of_year, as_of_round) — anti-leakage (plan §2)."""
    log = _ensure_team_totals(log, stat)
    rows = log[log["player"] == player].sort_values(["year", "round", "unixtime"])

    if as_of_year is not None:
        rows = rows[
            (rows["year"] < as_of_year)
            | ((rows["year"] == as_of_year) & (rows["round"] < (as_of_round or 0)))
        ]
    rows = rows.tail(lookback_games)

    if rows.empty:
        return {"mean": float("nan"), "share": float("nan"), "n_games": 0}

    mean = _ewma_last(rows[stat], halflife)
    team_totals = rows[f"team_{stat}"].replace(0, np.nan)
    share_series = (rows[stat] / team_totals).dropna()
    share = _ewma_last(share_series, halflife) if not share_series.empty else float("nan")

    return {"mean": mean, "share": share, "n_games": len(rows)}


def estimate_dispersion(log: pd.DataFrame, stat: str, min_games: int = 6) -> dict[str, float]:
    """Per-player Negative Binomial dispersion ``r`` via method of moments:

        variance = mean + mean^2 / r   =>   r = mean^2 / (variance - mean)

    Falls back to a league-wide pooled estimate (or ``PROP_MIN_DISPERSION``) for
    players without enough games, and floors all estimates at
    ``PROP_MIN_DISPERSION`` to avoid degenerate (near-zero) dispersion fits.
    """
    grouped = log.groupby("player")[stat].agg(["mean", "var", "count"])
    pooled_mean = log[stat].mean()
    pooled_var = log[stat].var()
    pooled_r = max(
        PROP_MIN_DISPERSION,
        pooled_mean ** 2 / (pooled_var - pooled_mean) if pooled_var > pooled_mean else PROP_MIN_DISPERSION,
    )

    out: dict[str, float] = {}
    for player, row in grouped.iterrows():
        if row["count"] < min_games or not (row["var"] > row["mean"]):
            out[player] = pooled_r
            continue
        r = row["mean"] ** 2 / (row["var"] - row["mean"])
        out[player] = max(PROP_MIN_DISPERSION, float(r))
    return out


def opponent_matchup_multiplier(
    log: pd.DataFrame, stat: str, opponent: str,
    as_of_year: int | None = None, as_of_round: int | None = None,
    halflife: float = PROP_EWMA_HALFLIFE, recent_seasons: int = PROP_RECENT_SEASONS,
) -> float:
    """How much more/less of ``stat`` the opponent concedes vs the league average,
    as a multiplier (1.0 = league average).

    The league baseline is **era-matched** (round-2 §5.1): only the last
    ``recent_seasons`` seasons, not the 2012+ all-history mean — stat levels
    drift with rule changes, so an all-history baseline biases every multiplier
    when the league average has moved."""
    log = _ensure_team_totals(log, stat)
    by_team_game = (
        log.groupby(["year", "round", "team", "opponent"], as_index=False)[stat].sum()
    )

    if as_of_year is not None:
        by_team_game = by_team_game[
            (by_team_game["year"] < as_of_year)
            | ((by_team_game["year"] == as_of_year) & (by_team_game["round"] < (as_of_round or 0)))
        ]
    if by_team_game.empty:
        return 1.0

    # League baseline over the recent era only.
    latest = by_team_game["year"].max()
    recent = by_team_game[by_team_game["year"] > latest - recent_seasons]
    league_avg = recent[stat].mean()
    if not np.isfinite(league_avg) or league_avg == 0:
        return 1.0

    conceded = by_team_game[by_team_game["opponent"] == opponent].sort_values(
        ["year", "round"]
    )[stat]
    if conceded.empty:
        return 1.0

    conceded_avg = _ewma_last(conceded, halflife)
    return float(conceded_avg / league_avg)


def expected_stat_mean(
    baseline_mean: float, share: float, team_total_mean: float,
    matchup_mult: float = 1.0, context_mult: float = 1.0,
) -> float:
    """Combine usage share with the team's expected scoreline-driven total
    (e.g. expected team disposals / goals from the match sim) and matchup /
    context multipliers (plan §3.3 steps 2-3).

    If ``share`` and ``team_total_mean`` are unavailable (NaN), falls back to
    the player's raw EWMA baseline mean scaled by the multipliers.
    """
    if np.isfinite(share) and np.isfinite(team_total_mean):
        base = share * team_total_mean
    else:
        base = baseline_mean
    return base * matchup_mult * context_mult

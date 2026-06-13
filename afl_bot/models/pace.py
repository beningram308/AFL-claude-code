"""
Team volume-stat profiles for the pace/environment latent factor (plan §2.5).

The scoring model (``afl_bot.models.scoring``) drives *points*; this module
drives the *volume* stats (disposals, marks, tackles, ...) that the player-prop
simulation allocates among players. For each team it fits an anti-leakage EWMA
of the team's per-game total of each stat — e.g. a team's expected disposals
per game — read from the ``team_<stat>`` columns of the player game log
(``afl_bot.data.player_stats`` / ``afl_bot.models.props``).

``afl_bot.sim.engine`` then draws a shared per-iteration ``pace`` multiplier and
scales every team's expected total by it (``simulate_team_stat_total``), so the
two teams' disposal counts move together — the §2.5 "draw the game environment
first, then condition team totals and player props on it" structure.
"""

from __future__ import annotations

import pandas as pd

from afl_bot.config import PROP_EWMA_HALFLIFE

# Volume stats the pace factor governs (points come from the scoring model, not
# from here, so goals are intentionally excluded).
PACE_STATS = ["disposals", "marks", "tackles"]


def _ewma_last(series: pd.Series, halflife: float) -> float:
    if series.empty:
        return float("nan")
    return float(series.ewm(halflife=halflife, adjust=True).mean().iloc[-1])


def _ensure_team_totals(log: pd.DataFrame, stat: str) -> pd.DataFrame:
    col = f"team_{stat}"
    if col in log.columns:
        return log
    log = log.copy()
    log[col] = log.groupby(["year", "round", "team"])[stat].transform("sum")
    return log


def team_stat_total_profiles(
    log: pd.DataFrame, stats: list[str] | None = None,
    as_of_year: int | None = None, as_of_round: int | None = None,
    halflife: float = PROP_EWMA_HALFLIFE,
) -> dict[str, dict[str, float]]:
    """Per-team EWMA expected team total for each stat, using only games
    strictly before ``(as_of_year, as_of_round)`` — anti-leakage (plan §2).

    Returns ``{team: {stat: mu_total}}``. Teams/stats with no prior games are
    simply absent (callers fall back to the league average, see
    ``league_stat_totals``).
    """
    stats = stats or PACE_STATS
    if log.empty:
        return {}

    profiles: dict[str, dict[str, float]] = {}
    for stat in stats:
        df = _ensure_team_totals(log, stat)
        col = f"team_{stat}"
        # one row per team-game (team_<stat> is repeated across a team's players)
        team_games = df.drop_duplicates(["year", "round", "team"])[
            ["year", "round", "unixtime", "team", col]
        ]
        if as_of_year is not None:
            team_games = team_games[
                (team_games["year"] < as_of_year)
                | ((team_games["year"] == as_of_year) & (team_games["round"] < (as_of_round or 0)))
            ]
        for team, grp in team_games.groupby("team"):
            grp = grp.sort_values(["year", "round", "unixtime"])
            profiles.setdefault(team, {})[stat] = _ewma_last(grp[col], halflife)
    return profiles


def league_stat_totals(
    log: pd.DataFrame, stats: list[str] | None = None,
    as_of_year: int | None = None, as_of_round: int | None = None,
) -> dict[str, float]:
    """League-average team total per stat — the fallback expected total for a
    team with no prior games (e.g. a new season's first round)."""
    stats = stats or PACE_STATS
    if log.empty:
        return {stat: float("nan") for stat in stats}

    out: dict[str, float] = {}
    for stat in stats:
        df = _ensure_team_totals(log, stat)
        col = f"team_{stat}"
        team_games = df.drop_duplicates(["year", "round", "team"])[["year", "round", "team", col]]
        if as_of_year is not None:
            team_games = team_games[
                (team_games["year"] < as_of_year)
                | ((team_games["year"] == as_of_year) & (team_games["round"] < (as_of_round or 0)))
            ]
        out[stat] = float(team_games[col].mean()) if not team_games.empty else float("nan")
    return out

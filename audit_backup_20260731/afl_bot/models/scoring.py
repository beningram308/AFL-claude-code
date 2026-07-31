"""
Match scoring model (plan §2.1, §3.2).

Pieces that feed the Monte Carlo engine alongside the Elo margin:

  1. ``team_scoring_profiles`` — for each team, an EWMA of points scored
     ("off_rate") and points conceded ("def_rate") computed *only* from games
     played strictly before a given cutoff (anti-leakage, plan §2).
  2. ``expected_total`` — combines both teams' off/def rates (plus an optional
     venue scoring factor) into a predicted combined score, matching the
     starter engine's ``expected_total``.
  3. ``team_shot_accuracy_profiles`` — for each team, an EWMA goal-conversion
     rate (goals / (goals + behinds)), computed with the same anti-leakage
     cutoff.
  4. ``points_to_shots`` — converts an expected points total into an expected
     scoring-shots count given an accuracy, the inverse of
     ``points = 6*goals + behinds = (5*accuracy + 1) * shots`` (plan §2.1).
     ``afl_bot.sim.engine.simulate_team_score`` uses this to turn
     (mu_total, mu_margin, accuracy) into a scoring-shots simulation:
     shots ~ NB(mu_shots, r), goals ~ Binomial(shots, accuracy), giving
     integer scores, real draw probabilities and accuracy-driven variance
     instead of a cosmetic post-hoc goals/behinds split.
"""

from __future__ import annotations

import pandas as pd

from afl_bot.config import PROP_EWMA_HALFLIFE


def _ewma_last(series: pd.Series, halflife: float) -> float:
    """EWMA of a chronologically-ordered series, returning the value *as of*
    the last observation (i.e. informed by all prior games, including the last)."""
    if series.empty:
        return float("nan")
    return float(series.ewm(halflife=halflife, adjust=True).mean().iloc[-1])


def team_scoring_long(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape completed games into one row per team per game with
    ``points_for`` / ``points_against``, ordered chronologically."""
    games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)

    home = games[["year", "round", "unixtime", "hteam", "hscore", "ascore"]].rename(
        columns={"hteam": "team", "hscore": "points_for", "ascore": "points_against"}
    )
    away = games[["year", "round", "unixtime", "ateam", "ascore", "hscore"]].rename(
        columns={"ateam": "team", "ascore": "points_for", "hscore": "points_against"}
    )
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "year", "round", "unixtime"]).reset_index(drop=True)


def team_scoring_profiles(
    games: pd.DataFrame, as_of_year: int | None = None, as_of_round: int | None = None,
    halflife: float = PROP_EWMA_HALFLIFE,
) -> dict[str, dict[str, float]]:
    """EWMA off/def scoring rate per team using only games strictly before
    (as_of_year, as_of_round). If both are None, uses all games provided
    (i.e. "current form" as of the latest game in ``games``)."""
    long = team_scoring_long(games)

    if as_of_year is not None:
        before = long[
            (long["year"] < as_of_year)
            | ((long["year"] == as_of_year) & (long["round"] < (as_of_round or 0)))
        ]
    else:
        before = long

    profiles: dict[str, dict[str, float]] = {}
    for team, grp in before.groupby("team"):
        grp = grp.sort_values(["year", "round", "unixtime"])
        profiles[team] = {
            "off_rate": _ewma_last(grp["points_for"], halflife),
            "def_rate": _ewma_last(grp["points_against"], halflife),
        }
    return profiles


def expected_total(home_off: float, home_def: float, away_off: float, away_def: float,
                    venue_factor: float = 1.0) -> float:
    """Predicted combined points from both sides' off/def rates (plan §3.2 §1)."""
    base = 0.5 * ((home_off + away_def) + (away_off + home_def))
    return base * venue_factor


def venue_scoring_factors(games: pd.DataFrame, strength: float = 30.0) -> dict[str, float]:
    """Per-venue scoring multiplier vs the league (round-2 §6.4): the venue's
    mean total points, empirical-Bayes shrunk toward the league mean by
    ``strength`` pseudo-games, divided by the league mean. Big grounds (MCG) and
    small/low ones differ; small-sample venues stay near 1.0. Feed into
    ``expected_total(venue_factor=...)``."""
    if games.empty or "venue" not in games.columns:
        return {}
    g = games.copy()
    g["_total"] = g["hscore"] + g["ascore"]
    league_mean = g["_total"].mean()
    if not league_mean:
        return {}

    factors: dict[str, float] = {}
    for venue, grp in g.groupby("venue"):
        n = len(grp)
        shrunk = (n * grp["_total"].mean() + strength * league_mean) / (n + strength)
        factors[venue] = float(shrunk / league_mean)
    return factors


def team_shot_accuracy_profiles(
    games: pd.DataFrame, as_of_year: int | None = None, as_of_round: int | None = None,
    halflife: float = PROP_EWMA_HALFLIFE,
) -> dict[str, float]:
    """EWMA goal-conversion rate (goals / scoring shots) per team, using only
    games strictly before (as_of_year, as_of_round) -- same anti-leakage
    cutoff as ``team_scoring_profiles``. Feeds the scoring-shots model
    (plan §2.1): ``points = (5*accuracy + 1) * shots``.
    """
    games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)

    home = games[["year", "round", "unixtime", "hteam", "hgoals", "hbehinds"]].rename(
        columns={"hteam": "team", "hgoals": "goals", "hbehinds": "behinds"}
    )
    away = games[["year", "round", "unixtime", "ateam", "agoals", "abehinds"]].rename(
        columns={"ateam": "team", "agoals": "goals", "abehinds": "behinds"}
    )
    long = pd.concat([home, away], ignore_index=True)
    long["accuracy"] = long["goals"] / (long["goals"] + long["behinds"])

    if as_of_year is not None:
        long = long[
            (long["year"] < as_of_year)
            | ((long["year"] == as_of_year) & (long["round"] < (as_of_round or 0)))
        ]

    profiles: dict[str, float] = {}
    for team, grp in long.groupby("team"):
        grp = grp.sort_values(["year", "round", "unixtime"])
        profiles[team] = _ewma_last(grp["accuracy"], halflife)
    return profiles


def points_to_shots(points: float, accuracy: float) -> float:
    """Invert ``points = (5*accuracy + 1) * shots`` (plan §2.1: points =
    6*goals + behinds, goals = accuracy * shots, behinds = (1-accuracy) * shots)."""
    return max(points, 0.0) / (5.0 * accuracy + 1.0)

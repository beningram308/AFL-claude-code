"""
Per-game home-ground advantage features for Elo (round-2 §6.1).

A flat ``ELO_HOME_ADVANTAGE=10`` for every game ignores that Geelong at GMHBA is
a very different "home" to a Melbourne club sharing the MCG, that the away team
flying interstate is worth several points, and that a rest-day edge matters. This
module turns the games table into a per-game ``hga_points`` value:

    hga = (home team's own venue HGA, shrunk toward the league 10)
          + interstate penalty   (away team travels to a different state)
          + rest coefficient * (home days rest - away days rest)

``afl_bot.ratings.elo.EloRatings`` reads a ``hga_points`` column when present, so
the ratings update and margin prediction use this instead of the flat constant.
Cross-check fitted HGAs against published AFL Elo write-ups as a sanity band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afl_bot.config import ELO_HOME_ADVANTAGE
from afl_bot.data.venues import VENUE_METADATA

# Each canonical AFL club's home state (for the interstate-travel flag).
TEAM_STATE = {
    "Adelaide": "SA", "Port Adelaide": "SA",
    "Brisbane Lions": "QLD", "Gold Coast": "QLD",
    "Carlton": "VIC", "Collingwood": "VIC", "Essendon": "VIC", "Geelong": "VIC",
    "Hawthorn": "VIC", "Melbourne": "VIC", "North Melbourne": "VIC", "Richmond": "VIC",
    "St Kilda": "VIC", "Western Bulldogs": "VIC",
    "Fremantle": "WA", "West Coast": "WA",
    "Greater Western Sydney": "NSW", "Sydney": "NSW",
}

# Venue city -> state (for venue_state). Canberra grouped with NSW (GWS home base).
_CITY_STATE = {
    "Melbourne": "VIC", "Geelong": "VIC", "Ballarat": "VIC",
    "Adelaide": "SA", "Lyndoch": "SA", "Mount Barker": "SA",
    "Perth": "WA", "Bunbury": "WA",
    "Brisbane": "QLD", "Gold Coast": "QLD", "Cairns": "QLD", "Townsville": "QLD",
    "Sydney": "NSW", "Canberra": "NSW",
    "Hobart": "TAS", "Launceston": "TAS",
    "Darwin": "NT", "Alice Springs": "NT", "Shanghai": "CHN",
}

INTERSTATE_PENALTY = 6.0     # points the home side gains when the away team travels interstate
REST_COEF = 0.6              # points per net day of rest advantage
REST_DAYS_CAP = 10          # cap the rest differential (early-season / bye outliers)
HGA_SHRINK_STRENGTH = 12.0  # pseudo-games shrinking a team's venue HGA toward the league


def venue_state(venue: str) -> str | None:
    info = VENUE_METADATA.get(venue)
    return _CITY_STATE.get(info["city"]) if info else None


def fit_team_hga(games: pd.DataFrame, league_hga: float = ELO_HOME_ADVANTAGE,
                 strength: float = HGA_SHRINK_STRENGTH) -> dict[str, float]:
    """Per-team home-ground advantage (points), as the home-minus-away margin
    swing, empirical-Bayes shrunk toward ``league_hga``. Conflates schedule a
    little, but is a robust standalone estimate and the shrinkage keeps
    small-sample teams near the league value."""
    if games.empty:
        return {}
    home = pd.DataFrame({"team": games["hteam"], "margin": games["hscore"] - games["ascore"], "loc": "home"})
    away = pd.DataFrame({"team": games["ateam"], "margin": games["ascore"] - games["hscore"], "loc": "away"})
    both = pd.concat([home, away], ignore_index=True)
    means = both.groupby(["team", "loc"])["margin"].mean().unstack()
    n_home = home.groupby("team").size()

    out: dict[str, float] = {}
    for team in means.index:
        if "home" not in means or "away" not in means or pd.isna(means.loc[team]).any():
            out[team] = league_hga
            continue
        swing = (means.loc[team, "home"] - means.loc[team, "away"]) / 2.0
        n = float(n_home.get(team, 0))
        out[team] = float((n * swing + strength * league_hga) / (n + strength))
    return out


def days_rest(games: pd.DataFrame) -> pd.DataFrame:
    """Home/away days since each team's previous game, from ``unixtime``.
    Returns the games (sorted) with ``home_rest`` / ``away_rest`` columns."""
    g = games.sort_values("unixtime").reset_index(drop=True)
    last: dict[str, float] = {}
    home_rest, away_rest = [], []
    for _, row in g.iterrows():
        ut = float(row["unixtime"])
        h, a = row["hteam"], row["ateam"]
        home_rest.append((ut - last[h]) / 86400.0 if h in last else np.nan)
        away_rest.append((ut - last[a]) / 86400.0 if a in last else np.nan)
        last[h] = ut
        last[a] = ut
    g["home_rest"] = home_rest
    g["away_rest"] = away_rest
    return g


def game_hga_points(games: pd.DataFrame, team_hga: dict[str, float] | None = None,
                    league_hga: float = ELO_HOME_ADVANTAGE) -> pd.Series:
    """Per-game home-ground advantage in points (team venue HGA + interstate +
    rest), indexed like ``games``."""
    team_hga = team_hga if team_hga is not None else fit_team_hga(games)
    rested = days_rest(games)
    # realign to the input order
    rested = rested.set_index(games.index) if len(rested) == len(games) else rested

    base = games["hteam"].map(lambda t: team_hga.get(t, league_hga))

    v_state = games["venue"].map(venue_state)
    away_state = games["ateam"].map(TEAM_STATE.get)
    interstate = (v_state.notna() & away_state.notna() & (v_state != away_state))
    interstate_adj = interstate.astype(float) * INTERSTATE_PENALTY

    rest_diff = (rested["home_rest"] - rested["away_rest"]).fillna(0.0).clip(-REST_DAYS_CAP, REST_DAYS_CAP)
    rest_adj = rest_diff * REST_COEF

    return (base.to_numpy() + interstate_adj.to_numpy() + rest_adj.to_numpy())


def attach_hga(games: pd.DataFrame, team_hga: dict[str, float] | None = None) -> pd.DataFrame:
    """Return ``games`` with an ``hga_points`` column for EloRatings.fit (§6.1)."""
    out = games.copy()
    out["hga_points"] = game_hga_points(games, team_hga)
    return out

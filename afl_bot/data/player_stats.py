"""
Player box-score loader (plan §2, addendum A1.1).

``load_player_log`` is the entry point the rest of the pipeline should use. It
combines two real sources, both reshaped to the same schema:

- ``afl_bot.data.fryzigg``: full *past* seasons (2012 onwards) -- kicks,
  handballs, behinds, hitouts, ruck contests, frees, TOG%, fantasy scores, ...
- ``afl_bot.data.dfs_australia``: the *current* season only, with the same
  fields plus centre bounce attendances (CBA) which Fryzigg lacks.

The two are concatenated into one player game log. If both are unavailable
(e.g. offline, or ``prefer_real=False``), falls back to
``synthetic_player_log``.

Schema (the contract ``afl_bot.models.props`` operates on):

    year, round, unixtime, player, team, opponent, is_home,
    disposals, goals, marks, tackles, team_disposals, team_goals, ...

plus extra columns where available: behinds, kicks, handballs, hitouts,
ruck_contests, free_kicks_for, free_kicks_against, time_on_ground_percentage,
afl_fantasy_score, supercoach_score, centre_bounce_attendances (DFS only),
kick_ins, kick_ins_play_on (DFS only).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import requests

from afl_bot.data.dfs_australia import fetch_player_stats, to_player_log
from afl_bot.data.fryzigg import fetch_fryzigg_player_stats
from afl_bot.data.fryzigg import to_player_log as fryzigg_to_player_log

STATS = ["disposals", "goals", "marks", "tackles"]


def load_player_log(
    games: pd.DataFrame, prefer_real: bool = True,
    players_per_team: int = 22, seed: int = 7, return_source: bool = False,
):
    """Combined Fryzigg (history) + DFS Australia (current season) player
    game log, with a synthetic fallback.

    Falls back to ``synthetic_player_log`` if ``prefer_real=False``, or if
    both real sources are unavailable/empty (e.g. offline, or ``pyreadr`` not
    installed for the Fryzigg loader).

    With ``return_source=True`` returns ``(log, source)`` where ``source`` is
    ``"real"`` or ``"synthetic"`` -- so callers can refuse to price props off a
    silent synthetic fallback (Fable round-2 §1.4 / §P10a).
    """
    if prefer_real:
        frames: list[pd.DataFrame] = []
        current_year = int(games["year"].max())

        try:
            fz_raw = fetch_fryzigg_player_stats()
            fz_log = fryzigg_to_player_log(fz_raw)
            if not fz_log.empty:
                frames.append(fz_log[fz_log["year"] < current_year])
        except (ImportError, requests.RequestException, ValueError) as exc:
            print(f"Fryzigg player stats unavailable ({exc}); "
                  f"continuing without historical player log.", file=sys.stderr)

        try:
            raw = fetch_player_stats()
            real = to_player_log(raw, games)
            if not real.empty:
                frames.append(real)
        except (requests.RequestException, ValueError) as exc:
            print(f"DFS Australia player stats unavailable ({exc}); "
                  f"continuing without current-season player log.", file=sys.stderr)

        if frames:
            log = pd.concat(frames, ignore_index=True, sort=False)
            return (log, "real") if return_source else log

    log = synthetic_player_log(games, players_per_team=players_per_team, seed=seed)
    return (log, "synthetic") if return_source else log


def synthetic_player_log(
    games: pd.DataFrame, players_per_team: int = 22, seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic per-player game log aligned to a real fixture list
    (``games`` from ``SquiggleClient.get_completed_games``). Each team gets a
    fixed roster of ``players_per_team`` players with stable per-player skill
    levels, so EWMA/usage-share/dispersion calculations behave realistically.
    """
    rng = np.random.default_rng(seed)

    teams = pd.unique(games[["hteam", "ateam"]].values.ravel())
    rosters: dict[str, list[str]] = {}
    skills: dict[str, dict[str, float]] = {}
    for team in teams:
        roster = [f"{team} Player {i+1}" for i in range(players_per_team)]
        rosters[team] = roster
        for i, player in enumerate(roster):
            # A handful of high-usage mids/forwards, the rest role players.
            tier = rng.choice(["elite", "solid", "role"], p=[0.15, 0.35, 0.50])
            base = {"elite": 1.6, "solid": 1.0, "role": 0.6}[tier]
            skills[player] = {
                "disposals": base * rng.uniform(0.8, 1.2),
                "goals": base * rng.uniform(0.4, 1.6),
                "marks": base * rng.uniform(0.7, 1.3),
                "tackles": base * rng.uniform(0.7, 1.3),
            }

    rows = []
    games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)
    for _, g in games.iterrows():
        for is_home, team, opponent, team_pts in (
            (True, g["hteam"], g["ateam"], g["hscore"]),
            (False, g["ateam"], g["hteam"], g["ascore"]),
        ):
            team_goals_target = team_pts // 6  # rough split, fine for synthetic data
            for player in rosters[team]:
                sk = skills[player]
                disposals = max(0, int(rng.normal(18 * sk["disposals"], 5)))
                goals = max(0, int(rng.poisson(0.6 * sk["goals"])))
                marks = max(0, int(rng.normal(4 * sk["marks"], 2)))
                tackles = max(0, int(rng.normal(3.5 * sk["tackles"], 1.8)))
                rows.append({
                    "year": int(g["year"]), "round": g["round"], "unixtime": g["unixtime"],
                    "player": player, "team": team, "opponent": opponent, "is_home": is_home,
                    "disposals": disposals, "goals": goals, "marks": marks, "tackles": tackles,
                })

    log = pd.DataFrame(rows)

    # Rescale each game's player goals so the team total roughly matches the
    # actual scoreline's goal count -- keeps goals correlated with the real result.
    for (year, rnd, team), grp in log.groupby(["year", "round", "team"]):
        game_row = games[
            (games["year"] == year) & (games["round"] == rnd)
            & ((games["hteam"] == team) | (games["ateam"] == team))
        ].iloc[0]
        actual_goals = int(game_row["hgoals"] if game_row["hteam"] == team else game_row["agoals"])
        sim_total = grp["goals"].sum()
        if sim_total > 0 and actual_goals > 0:
            scale = actual_goals / sim_total
            scaled = (grp["goals"] * scale).round().astype(int)
            diff = actual_goals - scaled.sum()
            if diff != 0 and len(scaled) > 0:
                idx = scaled.index[scaled.argmax()]
                scaled.loc[idx] = max(0, scaled.loc[idx] + diff)
            log.loc[grp.index, "goals"] = scaled

    for stat in STATS:
        log[f"team_{stat}"] = log.groupby(["year", "round", "team"])[stat].transform("sum")

    return log

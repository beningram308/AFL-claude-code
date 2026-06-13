"""
DFS Australia per-player game log loader (plan §1.2/§2, build-order step 2).

DFS Australia's free "AFL Stats Download" page (https://dfsaustralia.com/afl-stats-download/)
is backed by a WordPress AJAX endpoint that returns the current season's
per-player per-game box scores as JSON -- kicks, handballs, marks, tackles,
hitouts, centre bounce attendances (CBA), time on ground %, etc. No login or
API key is required.

``fetch_player_stats`` hits that endpoint and caches the raw response.
``to_player_log`` reshapes it into the schema ``afl_bot.models.props`` expects
(see ``afl_bot.data.player_stats`` for the schema contract), joining against a
fixture/result DataFrame (e.g. ``SquiggleClient.get_completed_games``) to
attach ``unixtime`` and ``is_home``.
"""

from __future__ import annotations

import pandas as pd
import requests

from afl_bot.config import CACHE_DIR
from afl_bot.data.storage import read_parquet, write_parquet
from afl_bot.data.teams import normalize_team_name

DFS_AJAX_URL = "https://dfsaustralia.com/wp-admin/admin-ajax.php"
DFS_USER_AGENT = "afl-multi-builder (https://github.com/; contact via repo issues)"
DFS_REQUEST_ACTION = "afl_player_stats_download_call_mysql"

CACHE_NAME = "dfs_australia_player_stats"
SCHEMA_VERSION = 1

# Counting-stat columns the box score reshape produces (matches
# ``afl_bot.data.player_stats.STATS``, kept independent to avoid a circular
# import between the two modules).
STATS = ["disposals", "goals", "marks", "tackles"]

_RAW_NUMERIC_COLS = [
    "year", "round", "kicks", "handballs", "marks", "tackles", "hitouts",
    "ruckContests", "freesFor", "freesAgainst", "goals", "behinds",
    "centreBounceAttendances", "kickIns", "kickInsPlayon",
    "timeOnGroundPercentage", "dreamTeamPoints", "SC",
]

# Renamed to match afl_bot.data.fryzigg's (already snake_case) column names so
# the two sources concatenate onto one schema (plan addendum A1.3).
_RENAME_TO_SHARED_SCHEMA = {
    "ruckContests": "ruck_contests",
    "freesFor": "free_kicks_for",
    "freesAgainst": "free_kicks_against",
    "timeOnGroundPercentage": "time_on_ground_percentage",
    "dreamTeamPoints": "afl_fantasy_score",
    "SC": "supercoach_score",
    "centreBounceAttendances": "centre_bounce_attendances",
    "kickIns": "kick_ins",
    "kickInsPlayon": "kick_ins_play_on",
}

# Extra per-player columns carried through alongside afl_bot.data.fryzigg.STATS
# (kicks/handballs/behinds/hitouts/etc shared with Fryzigg, plus
# centre_bounce_attendances/kick_ins/kick_ins_play_on which are DFS-only --
# NaN for Fryzigg rows once combined).
EXTRA_COLS = [
    "behinds", "kicks", "handballs", "hitouts", "ruck_contests",
    "free_kicks_for", "free_kicks_against", "time_on_ground_percentage",
    "afl_fantasy_score", "supercoach_score",
    "centre_bounce_attendances", "kick_ins", "kick_ins_play_on",
]


def fetch_player_stats(force_refresh: bool = False, cache_dir=CACHE_DIR) -> pd.DataFrame:
    """Per-player per-game box scores for the current season from DFS
    Australia. Cached to parquet; pass ``force_refresh=True`` to re-fetch
    (e.g. once a round completes)."""
    if not force_refresh:
        cached = read_parquet(CACHE_NAME, expected_schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
        if not cached.empty:
            return cached

    resp = requests.post(
        DFS_AJAX_URL,
        data={"action": DFS_REQUEST_ACTION},
        headers={"User-Agent": DFS_USER_AGENT},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    records = payload["data"] if isinstance(payload, dict) else payload

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    for col in _RAW_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    write_parquet(df, CACHE_NAME, schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
    _snapshot_rounds(df, cache_dir)
    return df


def _snapshot_rounds(df: pd.DataFrame, cache_dir) -> None:
    """Archive each completed (year, round) of a DFS pull to its own dated
    parquet (``dfs_<year>_r<round>.parquet``) so the current-season-only
    CBA/kick-in history accumulates permanently — DFS serves only the current
    season and old rounds can't be re-downloaded later (round-2 §7.3 / A1.1).
    The single most important operational habit; called on every live fetch."""
    if "year" not in df.columns or "round" not in df.columns:
        return
    rounds = df.dropna(subset=["year", "round"])
    for (year, rnd), grp in rounds.groupby(["year", "round"]):
        write_parquet(grp, f"dfs_{int(year)}_r{int(rnd)}",
                      schema_version=SCHEMA_VERSION, cache_dir=cache_dir)


def _long_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape a Squiggle-style fixture/result DataFrame (one row per match,
    ``hteam``/``ateam`` columns) into one row per team per match with
    ``is_home``, for joining onto the per-player box scores."""
    home = games[["year", "round", "unixtime", "hteam"]].rename(columns={"hteam": "team"})
    home["is_home"] = True
    away = games[["year", "round", "unixtime", "ateam"]].rename(columns={"ateam": "team"})
    away["is_home"] = False
    return pd.concat([home, away], ignore_index=True)


def to_player_log(raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Reshape DFS Australia's raw box scores (as returned by
    ``fetch_player_stats``) into the player-game-log schema used by
    ``afl_bot.models.props``:

        year, round, unixtime, player, team, opponent, is_home,
        disposals, goals, marks, tackles, team_disposals, team_goals, ...

    ``games`` should be a Squiggle-style fixture/result DataFrame (e.g.
    ``SquiggleClient.get_completed_games``) covering the same years -- it
    supplies ``unixtime`` and ``is_home``, and rows for games not present in
    ``games`` are dropped.
    """
    if raw.empty:
        return raw.copy()

    df = raw.copy()
    df["team"] = df["team"].map(normalize_team_name)
    df["opponent"] = df["opp"].map(normalize_team_name)
    df["disposals"] = df["kicks"] + df["handballs"]
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    if "startingPosition" in df.columns:
        df["position"] = df["startingPosition"]    # real AFL positional code (§5.3)

    schedule = _long_schedule(games)
    df = df.merge(schedule, on=["year", "round", "team"], how="inner")

    for stat in STATS:
        df[f"team_{stat}"] = df.groupby(["year", "round", "team"])[stat].transform("sum")

    df = df.rename(columns=_RENAME_TO_SHARED_SCHEMA)

    keep_cols = [
        "year", "round", "unixtime", "player", "position", "team", "opponent", "is_home",
        *STATS,
        *(f"team_{stat}" for stat in STATS),
        *EXTRA_COLS,
    ]
    return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

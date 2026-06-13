"""
Fryzigg historical player stats loader (plan §1.1, addendum A1.1).

The Fryzigg dataset (the data source behind fitzRoy's ``fetch_player_stats_fryzigg``)
is published as a single RDS file at ``fryziggafl.net/static/fryziggafl.rds`` --
~80 columns per player-game back to 1897, including kicks/handballs split,
behinds, hitouts, ruck contests, frees for/against, time on ground %, AFL
Fantasy and SuperCoach scores. It does NOT include centre bounce attendances
(CBA) -- that's the DFS Australia loader's unique contribution (see
``afl_bot.data.dfs_australia``).

Fryzigg is updated at the end of each season, so it covers full *past*
seasons but not the current one -- ``afl_bot.data.player_stats.load_player_log``
combines this (history) with DFS Australia (current season) into one log.

Reading the RDS file requires the optional ``pyreadr`` dependency
(``pip install pyreadr``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import requests

from afl_bot.config import CACHE_DIR
from afl_bot.data.storage import read_parquet, write_parquet
from afl_bot.data.teams import normalize_team_name

FRYZIGG_RDS_URL = "http://www.fryziggafl.net/static/fryziggafl.rds"
FRYZIGG_USER_AGENT = "afl-multi-builder (https://github.com/; contact via repo issues)"

CACHE_NAME = "fryzigg_player_stats"
SCHEMA_VERSION = 1

# Pre-2012 data mixes teams (Fitzroy, University, South Melbourne, Brisbane
# Bears, ...) that need deeper historical-era handling than `afl_bot.data.teams`
# currently covers (plan addendum A1.1) -- restrict to the GWS/Gold Coast era.
MIN_SEASON = 2012

STATS = ["disposals", "goals", "marks", "tackles"]

# Extra per-player columns carried through alongside STATS (already
# snake_case in the source data -- kept as-is so the combined player log uses
# one naming convention; afl_bot.data.dfs_australia renames its camelCase
# equivalents to match).
EXTRA_COLS = [
    "behinds", "kicks", "handballs", "hitouts", "ruck_contests",
    "free_kicks_for", "free_kicks_against", "time_on_ground_percentage",
    "afl_fantasy_score", "supercoach_score",
]


def fetch_fryzigg_player_stats(
    force_refresh: bool = False, min_season: int = MIN_SEASON, cache_dir=CACHE_DIR,
) -> pd.DataFrame:
    """Per-player per-game box scores for ``min_season`` onwards from the
    Fryzigg dataset. Cached to parquet -- the source is a ~12MB RDS file that
    only changes once a season, so re-fetching isn't needed in normal use.

    Raises ``ImportError`` if ``pyreadr`` isn't installed.
    """
    if not force_refresh:
        cached = read_parquet(CACHE_NAME, expected_schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
        if not cached.empty:
            return cached

    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required to read the Fryzigg RDS dataset; install it "
            "with `pip install pyreadr`."
        ) from exc

    resp = requests.get(FRYZIGG_RDS_URL, headers={"User-Agent": FRYZIGG_USER_AGENT}, timeout=120)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = Path(tmp.name)

    try:
        result = pyreadr.read_r(str(tmp_path))
        df = next(iter(result.values()))
    finally:
        tmp_path.unlink(missing_ok=True)

    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"])
    df["year"] = df["match_date"].dt.year
    df = df[df["year"] >= min_season].reset_index(drop=True)

    write_parquet(df, CACHE_NAME, schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
    return df


def to_player_log(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape Fryzigg's raw box scores (as returned by
    ``fetch_fryzigg_player_stats``) into the player-game-log schema used by
    ``afl_bot.models.props`` (see ``afl_bot.data.player_stats``):

        year, round, unixtime, player, team, opponent, is_home,
        disposals, goals, marks, tackles, team_disposals, team_goals, ...

    plus the ``EXTRA_COLS`` (behinds, kicks, handballs, hitouts,
    ruck_contests, free_kicks_for/against, time_on_ground_percentage,
    afl_fantasy_score, supercoach_score) for the role/scoring-shot upgrades
    in plan addendum A1.3. There is no centre-bounce-attendance equivalent --
    that column is left absent (NaN once combined with DFS Australia rows).
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["team"] = out["player_team"].map(normalize_team_name)
    home = out["match_home_team"].map(normalize_team_name)
    away = out["match_away_team"].map(normalize_team_name)
    out["is_home"] = out["team"] == home
    out["opponent"] = away.where(out["is_home"], home)
    out["player"] = out["player_first_name"].str.strip() + " " + out["player_last_name"].str.strip()
    if "player_position" in out.columns:
        out["position"] = out["player_position"]   # real AFL positional code (§5.3)
    # Epoch seconds, resolution-independent: a cached parquet round-trips
    # match_date as datetime64[us], so `.astype("int64")` (which would give
    # microseconds) // 10**9 lands in 1970. Subtracting the epoch and dividing
    # by a 1s Timedelta is correct for ns/us/s alike (chronological ordering in
    # EWMA / as-of-round comparisons depends on this).
    out["unixtime"] = (
        (pd.to_datetime(out["match_date"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    )

    # Fryzigg's match_round is a label ("1".."24", "Opening Round", "Grand
    # Final", ...). Map it to the REAL round number so (year, round) aligns with
    # Squiggle/DFS (round-2 §7.2): numeric labels keep their int; pre-season
    # rounds slot below round 1 (e.g. "Opening Round" -> 0) and finals continue
    # above the last numeric round, both by chronological order. The pure
    # chronological ordinal is kept as ``round_ordinal`` for stable sorting.
    out["year"] = pd.to_datetime(out["match_date"]).dt.year
    order = (
        out.groupby(["year", "match_round"])["match_date"].min()
        .reset_index().sort_values(["year", "match_date"])
    )
    order["round_ordinal"] = order.groupby("year").cumcount()
    order["_num"] = pd.to_numeric(order["match_round"], errors="coerce")

    real_rows = []
    for year, grp in order.groupby("year", sort=False):
        grp = grp.sort_values("match_date")
        numeric_dates = grp.loc[grp["_num"].notna(), "match_date"]
        first_numeric = numeric_dates.min() if not numeric_dates.empty else None
        pre = int(grp["_num"].min()) if grp["_num"].notna().any() else 1
        finals = int(grp["_num"].max()) if grp["_num"].notna().any() else 0
        for _, r in grp.iterrows():
            if pd.notna(r["_num"]):
                rr = int(r["_num"])
            elif first_numeric is not None and r["match_date"] < first_numeric:
                pre -= 1
                rr = pre                       # pre-season (e.g. Opening Round) -> 0
            else:
                finals += 1
                rr = finals                    # finals after the last numeric round
            real_rows.append((year, r["match_round"], rr, int(r["round_ordinal"])))
    round_map = pd.DataFrame(real_rows, columns=["year", "match_round", "round", "round_ordinal"])
    out = out.merge(round_map, on=["year", "match_round"])

    for stat in STATS:
        out[f"team_{stat}"] = out.groupby(["year", "round", "team"])[stat].transform("sum")

    keep_cols = [
        "year", "round", "round_ordinal", "unixtime", "player", "player_id",
        "position", "team", "opponent", "is_home",
        *STATS,
        *(f"team_{stat}" for stat in STATS),
        *EXTRA_COLS,
    ]
    return out[[c for c in keep_cols if c in out.columns]].reset_index(drop=True)

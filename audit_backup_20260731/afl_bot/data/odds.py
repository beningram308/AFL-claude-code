"""
Historical AFL odds loader (plan §1.7, build-order step 3).

Australia Sports Betting publishes a free historical AFL results + odds
spreadsheet (H2H open/close, line, and totals markets, back to 2009). It's
intended for personal use -- cached locally to parquet, never redistributed.

``fetch_historical_odds`` downloads and reshapes it to one row per match with
canonical team names (``afl_bot.data.teams``), ready to join onto Squiggle
fixtures on ``(year, hteam, ateam)`` for CLV / market-comparison backtests
(``afl_bot.backtest.walkforward``).
"""

from __future__ import annotations

import sys
import time
from io import BytesIO

import pandas as pd
import requests

from afl_bot.config import CACHE_DIR, ODDS_MAX_AGE_DAYS
from afl_bot.data.storage import read_parquet, write_parquet
from afl_bot.data.teams import normalize_team_name

ODDS_URL = "https://www.aussportsbetting.com/historical_data/afl.xlsx"
ODDS_USER_AGENT = "afl-multi-builder (https://github.com/; contact via repo issues; personal use)"

CACHE_NAME = "aussportsbetting_afl_odds"
SCHEMA_VERSION = 1

# Source column -> our column. H2H odds (open/close) drive the CLV backtest;
# totals are kept for future total-points market backtests (plan §4).
_COLUMN_MAP = {
    "Date": "date",
    "Home Team": "hteam",
    "Away Team": "ateam",
    "Home Score": "hscore",
    "Away Score": "ascore",
    "Home Odds Open": "home_odds_open",
    "Home Odds Close": "home_odds_close",
    "Away Odds Open": "away_odds_open",
    "Away Odds Close": "away_odds_close",
    "Total Score Open": "total_open",
    "Total Score Close": "total_close",
    "Total Score Over Close": "total_over_odds_close",
    "Total Score Under Close": "total_under_odds_close",
}


def fetch_historical_odds(force_refresh: bool = False, cache_dir=CACHE_DIR,
                          max_age_days: float | None = ODDS_MAX_AGE_DAYS) -> pd.DataFrame:
    """Historical AFL H2H + totals odds, one row per match, with canonical
    team names and a ``year`` column for joining onto Squiggle fixtures.

    Cached to parquet, but the source workbook updates weekly in-season, so a
    cache older than ``max_age_days`` is treated as stale and re-downloaded
    (round-2 §7.5). ``max_age_days=None`` caches forever; ``force_refresh`` always
    re-fetches."""
    cache_path = cache_dir / f"{CACHE_NAME}.parquet"
    stale = (
        max_age_days is not None and cache_path.exists()
        and (time.time() - cache_path.stat().st_mtime) > max_age_days * 86400
    )
    if not force_refresh and not stale:
        cached = read_parquet(CACHE_NAME, expected_schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
        if not cached.empty:
            return cached
    if stale:
        print("Historical odds cache is stale; re-downloading the weekly workbook.", file=sys.stderr)

    resp = requests.get(ODDS_URL, headers={"User-Agent": ODDS_USER_AGENT}, timeout=60)
    resp.raise_for_status()

    df = pd.read_excel(BytesIO(resp.content), header=1)
    df = df.rename(columns=_COLUMN_MAP)
    df = df[[c for c in _COLUMN_MAP.values() if c in df.columns]].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["hteam"] = df["hteam"].map(normalize_team_name)
    df["ateam"] = df["ateam"].map(normalize_team_name)
    df = df.sort_values("date").reset_index(drop=True)

    write_parquet(df, CACHE_NAME, schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
    return df


def attach_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Left-join historical odds onto a Squiggle-style fixture/result
    DataFrame, matching on ``(year, hteam, ateam)``. Each team plays each
    opponent at most twice a season (once at each venue), so this triple is
    a safe join key. Games without a matching odds row keep NaN odds
    columns."""
    odds_cols = [c for c in _COLUMN_MAP.values() if c not in ("date", "hteam", "ateam", "hscore", "ascore")]
    return games.merge(
        odds[["year", "hteam", "ateam", *odds_cols]],
        on=["year", "hteam", "ateam"],
        how="left",
    )

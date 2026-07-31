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


def attach_odds(games: pd.DataFrame, odds: pd.DataFrame,
                tolerance_days: int = 3) -> pd.DataFrame:
    """Left-join historical odds onto a Squiggle-style fixture/result
    DataFrame, matching on ``(hteam, ateam)`` + nearest match date within
    ``tolerance_days``.

    AUDIT FIX 2026-07-31: the old join key ``(year, hteam, ateam)`` was
    documented as unique ("each team plays each opponent at most twice a
    season, once at each venue") but is NOT — finals rematches and same-venue
    rematches produce 63 duplicate keys in the 2016-26 games table alone, so
    the old merge expanded 2,224 games to 2,354 rows and attached the wrong
    game's odds to every duplicated pair (contaminating the market benchmark,
    the ensemble blend training data and CLV). A date-proximity join is
    unambiguous. Games without a matching odds row keep NaN odds columns."""
    odds_cols = [c for c in _COLUMN_MAP.values()
                 if c not in ("date", "hteam", "ateam", "hscore", "ascore")
                 and c in odds.columns]

    # Fallback for date-less frames (synthetic test seasons): legacy
    # (year, hteam, ateam) join, but with the odds side deduplicated so the
    # merge can never expand rows. Real Squiggle/ASB data always has dates and
    # takes the unambiguous date-proximity path below.
    if "date" not in games.columns or "date" not in odds.columns:
        o = odds.drop_duplicates(["year", "hteam", "ateam"])
        return games.merge(
            o[["year", "hteam", "ateam", *odds_cols]],
            on=["year", "hteam", "ateam"], how="left",
        )

    g = games.copy()
    g["_orig_order"] = range(len(g))
    g["_gdate"] = (pd.to_datetime(g["date"], errors="coerce")
                   .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))

    o = odds[["date", "hteam", "ateam", *odds_cols]].copy()
    o["_odate"] = (pd.to_datetime(o["date"], errors="coerce")
                   .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))
    o = o.drop(columns=["date"]).dropna(subset=["_odate"]).sort_values("_odate")

    joinable = g.dropna(subset=["_gdate"]).sort_values("_gdate")
    merged = pd.merge_asof(
        joinable, o,
        left_on="_gdate", right_on="_odate",
        by=["hteam", "ateam"],
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    # Games with an unparseable date keep NaN odds columns (rare/defensive).
    missing = g[g["_gdate"].isna()].copy()
    for c in odds_cols:
        missing[c] = float("nan")
    out = pd.concat([merged, missing], ignore_index=True).sort_values("_orig_order")
    out = out.drop(columns=["_gdate", "_odate", "_orig_order"], errors="ignore")
    return out.reset_index(drop=True)

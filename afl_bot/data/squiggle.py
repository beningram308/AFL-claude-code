"""
Squiggle API client (plan §2).

Thin wrapper over https://api.squiggle.com.au/ for fixtures, results, ladder and
the free crowd-model "tips" (used as an ensemble signal). Every response is cached
to local parquet under ``data_cache/`` so we never hit the API twice for the same
query — per Squiggle's usage guidelines, cache aggressively and identify your app
with a descriptive User-Agent.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from afl_bot.config import CACHE_DIR, SQUIGGLE_BASE_URL, SQUIGGLE_USER_AGENT
from afl_bot.data.storage import read_parquet, write_parquet

_MIN_REQUEST_INTERVAL = 1.0  # seconds; be polite to the free API

# Bumped whenever the shape of cached Squiggle responses changes.
SCHEMA_VERSION = 1


class SquiggleClient:
    def __init__(self, base_url: str = SQUIGGLE_BASE_URL, cache_dir=CACHE_DIR):
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0

    # ------------------------------------------------------------------ #
    # Low-level fetch
    # ------------------------------------------------------------------ #
    def _get(self, params: dict[str, Any]) -> list[dict]:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        resp = requests.get(
            self.base_url,
            params=params,
            headers={"User-Agent": SQUIGGLE_USER_AGENT},
            timeout=30,
        )
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        payload = resp.json()
        # Squiggle wraps results under a key matching the query, e.g. {"games": [...]}
        for key in ("games", "tips", "ladder", "teams", "sources", "standings"):
            if key in payload:
                return payload[key]
        return payload if isinstance(payload, list) else [payload]

    def _cached(self, cache_name: str, params: dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{cache_name}.parquet"
        if cache_path.exists() and not force_refresh:
            return read_parquet(cache_name, expected_schema_version=SCHEMA_VERSION, cache_dir=self.cache_dir)

        records = self._get(params)
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            write_parquet(df, cache_name, schema_version=SCHEMA_VERSION, cache_dir=self.cache_dir)
        return df

    # ------------------------------------------------------------------ #
    # Public endpoints
    # ------------------------------------------------------------------ #
    def get_games(self, year: int, force_refresh: bool = False) -> pd.DataFrame:
        """Fixtures + results for a season (completed games have non-null scores)."""
        return self._cached(f"games_{year}", {"q": "games", "year": year}, force_refresh)

    def get_tips(self, year: int, round: int | None = None, force_refresh: bool = False) -> pd.DataFrame:
        """Crowd-model tips/probabilities (free ensemble signal)."""
        params: dict[str, Any] = {"q": "tips", "year": year}
        cache_name = f"tips_{year}"
        if round is not None:
            params["round"] = round
            cache_name += f"_r{round}"
        return self._cached(cache_name, params, force_refresh)

    def get_ladder(self, year: int, round: int, force_refresh: bool = False) -> pd.DataFrame:
        return self._cached(
            f"ladder_{year}_r{round}",
            {"q": "ladder", "year": year, "round": round},
            force_refresh,
        )

    def get_teams(self, force_refresh: bool = False) -> pd.DataFrame:
        return self._cached("teams", {"q": "teams"}, force_refresh)

    def get_lineup(self, year: int, round: int, force_refresh: bool = True) -> pd.DataFrame:
        """Confirmed team lineups for a round. Always re-fetched by default — lineups
        change up until close to bounce, so caching them long-term is unsafe."""
        return self._cached(
            f"lineup_{year}_r{round}",
            {"q": "lineup", "year": year, "round": round},
            force_refresh,
        )

    def get_completed_games(self, year: int, force_refresh: bool = False) -> pd.DataFrame:
        """Subset of get_games where both scores are populated (i.e. the match is over)."""
        df = self.get_games(year, force_refresh)
        if df.empty:
            return df
        return df[df["complete"] == 100].reset_index(drop=True)

    def get_upcoming_games(self, year: int, force_refresh: bool = True) -> pd.DataFrame:
        """Subset of get_games that haven't been played yet."""
        df = self.get_games(year, force_refresh)
        if df.empty:
            return df
        return df[df["complete"] < 100].reset_index(drop=True)

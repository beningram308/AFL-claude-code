"""
Per-game weather ingestion (plan §1.8).

Rain is the #1 driver of low-scoring, low-disposal, high-tackle AFL games, so
each match needs a rainfall figure for its venue on game day. The plan names
BOM; in practice BOM's historical access is awkward, so we use Open-Meteo's
free, keyless historical archive (and forecast) API by venue latitude/longitude
(``afl_bot.data.venues``). Swapping in BOM later only means changing
``fetch_venue_weather``.

``attach_weather`` joins daily rainfall + max wind onto a Squiggle-style games
table and flags each game ``is_wet`` (rain >= ``WET_THRESHOLD_MM`` and the venue
is open-air). Roofed grounds (Marvel/Docklands) are always dry. Per-venue daily
weather is cached to parquet; network failures degrade gracefully to NaN
rainfall (and therefore no weather adjustment) rather than raising.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import requests

from afl_bot.config import CACHE_DIR, WET_THRESHOLD_MM
from afl_bot.data.storage import read_parquet, write_parquet
from afl_bot.data.venues import is_roofed, venue_info
from afl_bot.models.weather_effects import greasiness_factor

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "afl-multi-builder (https://github.com/; contact via repo issues)"

CACHE_NAME = "venue_daily_weather"
SCHEMA_VERSION = 2  # bumped: added temp_c + apparent_temp_c columns


def fetch_venue_weather(lat: float, lon: float, start_date: str, end_date: str,
                        timeout: int = 60) -> pd.DataFrame:
    """Daily rainfall (mm), max wind (km/h), min temperature (°C), and apparent
    temperature min (°C) for a venue between two ISO dates, from the Open-Meteo
    historical archive. Returns columns
    ``[date, rain_mm, wind_kmh, temp_c, apparent_temp_c]``."""
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "daily": (
                "precipitation_sum,wind_speed_10m_max,"
                "temperature_2m_min,apparent_temperature_min"
            ),
            "timezone": "auto",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    return pd.DataFrame({
        "date": daily.get("time", []),
        "rain_mm": daily.get("precipitation_sum", []),
        "wind_kmh": daily.get("wind_speed_10m_max", []),
        "temp_c": daily.get("temperature_2m_min", []),
        "apparent_temp_c": daily.get("apparent_temperature_min", []),
    })


def fetch_forecast_rain(lat: float, lon: float, date: str, timeout: int = 30) -> float:
    """Forecast daily rainfall (mm) for an upcoming game date, or NaN if the
    date is outside the forecast horizon / the call fails."""
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat, "longitude": lon,
                "start_date": date, "end_date": date,
                "daily": "precipitation_sum", "timezone": "auto",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        vals = resp.json().get("daily", {}).get("precipitation_sum", [])
        return float(vals[0]) if vals and vals[0] is not None else float("nan")
    except (requests.RequestException, ValueError, IndexError):
        return float("nan")


def forecast_game_conditions(lat: float, lon: float, local_dt, window_h: int = 3,
                             timeout: int = 30) -> dict:
    """Forecast conditions at bounce: rain over the game window plus daily min
    temperature and apparent temperature. Returns a dict with keys
    ``rain_mm``, ``temp_c``, ``apparent_temp_c``, ``wind_kmh``; NaN for any
    value that fails to fetch."""
    nan = float("nan")
    result: dict = {"rain_mm": nan, "temp_c": nan, "apparent_temp_c": nan, "wind_kmh": nan}
    try:
        date = pd.to_datetime(local_dt).strftime("%Y-%m-%d")
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat, "longitude": lon,
                "start_date": date, "end_date": date,
                "hourly": "precipitation",
                "daily": (
                    "temperature_2m_min,apparent_temperature_min,"
                    "wind_speed_10m_max"
                ),
                "timezone": "auto",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        h = data.get("hourly", {})
        hourly = pd.DataFrame({"time": h.get("time", []), "rain_mm": h.get("precipitation", [])})
        result["rain_mm"] = (
            game_window_rain(hourly, local_dt, window_h) if not hourly.empty else nan
        )

        d = data.get("daily", {})
        def _first(key: str) -> float:
            vals = d.get(key, [])
            return float(vals[0]) if vals and vals[0] is not None else nan
        result["temp_c"] = _first("temperature_2m_min")
        result["apparent_temp_c"] = _first("apparent_temperature_min")
        result["wind_kmh"] = _first("wind_speed_10m_max")
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return result


def game_window_rain(hourly: pd.DataFrame, local_dt, window_h: int = 3) -> float:
    """Sum hourly precipitation over a game's ~window (bounce hour + next
    ``window_h``-1 hours). ``hourly`` has columns ``[time, rain_mm]`` in venue
    local time; ``local_dt`` is the bounce time. NaN if no overlapping hours."""
    dt = pd.to_datetime(local_dt)
    times = pd.to_datetime(hourly["time"])
    start = dt.floor("h")
    mask = (times >= start) & (times < start + pd.Timedelta(hours=window_h))
    vals = pd.to_numeric(hourly.loc[mask, "rain_mm"], errors="coerce").dropna()
    return float(vals.sum()) if not vals.empty else float("nan")


def fetch_hourly_archive(lat: float, lon: float, start_date: str, end_date: str,
                         timeout: int = 120) -> pd.DataFrame:
    """Hourly precipitation (mm) for a venue between two ISO dates from the
    Open-Meteo archive — for fitting wet effects on rain AT BOUNCE TIME rather
    than the noisy daily total (round-2 §4.3). Columns ``[time, rain_mm]``."""
    resp = requests.get(
        ARCHIVE_URL,
        params={"latitude": lat, "longitude": lon, "start_date": start_date,
                "end_date": end_date, "hourly": "precipitation", "timezone": "auto"},
        headers={"User-Agent": USER_AGENT}, timeout=timeout,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    return pd.DataFrame({"time": hourly.get("time", []), "rain_mm": hourly.get("precipitation", [])})


def forecast_game_rain(lat: float, lon: float, local_dt, window_h: int = 3,
                       timeout: int = 30) -> float:
    """Forecast rainfall (mm) over an upcoming game's window at bounce time
    (round-2 §4.2/§4.3). NaN outside the forecast horizon / on failure."""
    try:
        date = pd.to_datetime(local_dt).strftime("%Y-%m-%d")
        resp = requests.get(
            FORECAST_URL,
            params={"latitude": lat, "longitude": lon, "start_date": date, "end_date": date,
                    "hourly": "precipitation", "timezone": "auto"},
            headers={"User-Agent": USER_AGENT}, timeout=timeout,
        )
        resp.raise_for_status()
        h = resp.json().get("hourly", {})
        hourly = pd.DataFrame({"time": h.get("time", []), "rain_mm": h.get("precipitation", [])})
        return game_window_rain(hourly, local_dt, window_h) if not hourly.empty else float("nan")
    except (requests.RequestException, ValueError, KeyError):
        return float("nan")


def _gather_weather(games: pd.DataFrame, cache_dir, force_refresh: bool, fetcher) -> pd.DataFrame:
    """Fetch (and cache) daily weather for every open-air venue in ``games``,
    only calling the API for venue/date ranges not already cached."""
    cache = pd.DataFrame()
    if not force_refresh:
        cache = read_parquet(CACHE_NAME, expected_schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
    cached_dates = (
        {v: set(g["date"]) for v, g in cache.groupby("venue")} if not cache.empty else {}
    )

    frames = [cache] if not cache.empty else []
    updated = False
    for venue, grp in games.groupby("venue"):
        info = venue_info(venue)
        if info is None or info["roofed"]:
            continue  # unknown venue or roofed -> no weather needed
        needed = set(grp["_date"].dropna())
        if not needed or (not force_refresh and needed <= cached_dates.get(venue, set())):
            continue
        try:
            fetched = fetcher(info["lat"], info["lon"], min(needed), max(needed))
        except requests.RequestException as exc:
            print(f"Weather unavailable for {venue} ({exc}); leaving it dry.", file=sys.stderr)
            continue
        if fetched.empty:
            continue
        for col in ("temp_c", "apparent_temp_c"):
            if col not in fetched.columns:
                fetched = fetched.copy()
                fetched[col] = float("nan")
        fetched = fetched.assign(venue=venue)[
            ["venue", "date", "rain_mm", "wind_kmh", "temp_c", "apparent_temp_c"]
        ]
        frames = [f[f["venue"] != venue] for f in frames]  # drop stale rows for this venue
        frames.append(fetched)
        updated = True

    if not frames:
        return pd.DataFrame(columns=["venue", "date", "rain_mm", "wind_kmh", "temp_c", "apparent_temp_c"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["venue", "date"])
    if updated:
        write_parquet(out, CACHE_NAME, schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
    return out


def attach_weather(games: pd.DataFrame, weather: pd.DataFrame | None = None, *,
                   cache_dir=CACHE_DIR, force_refresh: bool = False,
                   wet_threshold_mm: float = WET_THRESHOLD_MM, fetcher=fetch_venue_weather) -> pd.DataFrame:
    """Attach ``rain_mm`` / ``wind_kmh`` / ``temp_c`` / ``apparent_temp_c`` /
    ``roofed`` / ``is_wet`` / ``greasiness`` to a games table (which must have
    ``venue`` and ``date`` columns).

    Pass ``weather`` (columns ``[venue, date, rain_mm, wind_kmh, ...]``) to skip
    the network entirely; otherwise it is fetched/cached via ``fetcher``. A game
    is ``is_wet`` when its rainfall is at/above ``wet_threshold_mm`` and the
    venue is open-air. ``greasiness`` is the continuous 0.0-1.0 Phase-1 factor
    (blends rain, cold, dew proximity, wind). Missing weather produces NaN
    weather columns, ``is_wet=False``, and ``greasiness=0.0``.
    """
    games = games.copy()
    if games.empty:
        for col in ("rain_mm", "wind_kmh", "temp_c", "apparent_temp_c", "greasiness"):
            games[col] = pd.Series(dtype=float)
        games["is_wet"] = pd.Series(dtype=bool)
        games["roofed"] = pd.Series(dtype=bool)
        return games

    games["_date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    games["roofed"] = games["venue"].map(is_roofed)

    if weather is None:
        weather = _gather_weather(games, cache_dir, force_refresh, fetcher)

    if weather is None or weather.empty:
        games["rain_mm"] = np.nan
        games["wind_kmh"] = np.nan
        games["temp_c"] = np.nan
        games["apparent_temp_c"] = np.nan
    else:
        # Older cached weather (schema v1) may lack temp columns; fill with NaN.
        for col in ("temp_c", "apparent_temp_c"):
            if col not in weather.columns:
                weather = weather.copy()
                weather[col] = np.nan
        games = games.merge(
            weather.rename(columns={"date": "_wdate"}),
            how="left", left_on=["venue", "_date"], right_on=["venue", "_wdate"],
        ).drop(columns=["_wdate"], errors="ignore")

    games["is_wet"] = (games["rain_mm"].fillna(0.0) >= wet_threshold_mm) & (~games["roofed"])
    games["greasiness"] = games.apply(
        lambda r: greasiness_factor(
            r["rain_mm"], r.get("temp_c", float("nan")),
            r.get("apparent_temp_c", float("nan")),
            r.get("wind_kmh", float("nan")), bool(r["roofed"])
        ),
        axis=1,
    )
    return games.drop(columns=["_date"], errors="ignore")

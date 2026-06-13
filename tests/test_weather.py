from unittest.mock import Mock, patch

import pandas as pd
import requests

from afl_bot.data.weather import (
    attach_weather,
    fetch_forecast_rain,
    forecast_game_rain,
    game_window_rain,
)

GAMES = pd.DataFrame([
    {"year": 2024, "round": 1, "hteam": "Geelong", "ateam": "Carlton",
     "venue": "M.C.G.", "date": "2024-04-01 19:40:00"},
    {"year": 2024, "round": 1, "hteam": "Essendon", "ateam": "Sydney",
     "venue": "Marvel Stadium", "date": "2024-04-02 19:40:00"},
    {"year": 2024, "round": 2, "hteam": "Geelong", "ateam": "Sydney",
     "venue": "M.C.G.", "date": "2024-04-08 13:00:00"},
])


def test_attach_weather_with_supplied_table_flags_wet_dry_roofed():
    weather = pd.DataFrame([
        {"venue": "M.C.G.", "date": "2024-04-01", "rain_mm": 12.0, "wind_kmh": 20.0},
        {"venue": "M.C.G.", "date": "2024-04-08", "rain_mm": 0.5, "wind_kmh": 10.0},
        {"venue": "Marvel Stadium", "date": "2024-04-02", "rain_mm": 30.0, "wind_kmh": 40.0},
    ])
    out = attach_weather(GAMES, weather=weather, wet_threshold_mm=5.0)

    mcg_r1 = out[(out["venue"] == "M.C.G.") & (out["round"] == 1)].iloc[0]
    mcg_r2 = out[(out["venue"] == "M.C.G.") & (out["round"] == 2)].iloc[0]
    marvel = out[out["venue"] == "Marvel Stadium"].iloc[0]

    assert mcg_r1["is_wet"]            # 12mm open-air -> wet
    assert not mcg_r2["is_wet"]        # 0.5mm -> dry
    assert not marvel["is_wet"]        # 30mm but roofed -> dry
    assert marvel["roofed"] and not mcg_r1["roofed"]


def test_attach_weather_fetches_and_caches(tmp_path):
    def fake_fetch(lat, lon, start, end):
        dates = pd.date_range(start, end).strftime("%Y-%m-%d")
        return pd.DataFrame({"date": dates, "rain_mm": [12.0] * len(dates), "wind_kmh": [20.0] * len(dates)})

    fetcher = Mock(side_effect=fake_fetch)
    out = attach_weather(GAMES, cache_dir=tmp_path, fetcher=fetcher)
    # only the open-air venue (M.C.G.) is fetched; roofed Marvel is skipped
    assert fetcher.call_count == 1
    assert out[out["venue"] == "M.C.G."]["is_wet"].all()

    # second call reads the cache -> no new fetch
    attach_weather(GAMES, cache_dir=tmp_path, fetcher=fetcher)
    assert fetcher.call_count == 1


def test_attach_weather_network_failure_degrades_to_dry(tmp_path):
    def boom(*args, **kwargs):
        raise requests.RequestException("offline")

    out = attach_weather(GAMES, cache_dir=tmp_path, fetcher=boom)
    assert out["rain_mm"].isna().all()
    assert not out["is_wet"].any()


def test_attach_weather_empty_games():
    out = attach_weather(pd.DataFrame())
    assert out.empty
    assert {"rain_mm", "wind_kmh", "roofed", "is_wet"} <= set(out.columns)


def test_fetch_forecast_rain_failure_returns_nan():
    with patch("afl_bot.data.weather.requests.get", side_effect=requests.RequestException("x")):
        val = fetch_forecast_rain(-37.82, 144.98, "2026-06-20")
    assert val != val  # NaN


def test_fetch_forecast_rain_parses_value():
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"daily": {"precipitation_sum": [7.3]}}
    with patch("afl_bot.data.weather.requests.get", return_value=resp):
        assert fetch_forecast_rain(-37.82, 144.98, "2026-06-20") == 7.3


def test_game_window_rain_sums_bounce_window():
    hourly = pd.DataFrame({
        "time": ["2025-03-15 18:00", "2025-03-15 19:00", "2025-03-15 20:00", "2025-03-15 21:00"],
        "rain_mm": [9.0, 2.0, 3.0, 1.0],
    })
    # bounce 19:40 floors to 19:00; 3h window [19,20,21] = 2+3+1 = 6 (18:00 excluded)
    assert game_window_rain(hourly, "2025-03-15 19:40", window_h=3) == 6.0


def test_forecast_game_rain_hourly_window_and_failure():
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"hourly": {
        "time": ["2026-06-20 18:00", "2026-06-20 19:00", "2026-06-20 20:00"],
        "precipitation": [0.0, 4.0, 5.0],
    }}
    with patch("afl_bot.data.weather.requests.get", return_value=resp):
        assert forecast_game_rain(-37.82, 144.98, "2026-06-20 19:10", window_h=2) == 9.0
    with patch("afl_bot.data.weather.requests.get", side_effect=requests.RequestException("x")):
        assert forecast_game_rain(-37.82, 144.98, "2026-06-20 19:10") != \
               forecast_game_rain(-37.82, 144.98, "2026-06-20 19:10")  # NaN != NaN

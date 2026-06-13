"""
Loader schema-contract tests (plan §5.3). Network-free: every loader must emit
the columns the downstream models depend on. The DFS / Fryzigg / odds loaders
also assert their output schemas in their own test files; this covers the
unified player-log contract plus the weather / stoppages / storage layers.
"""

import pandas as pd

from afl_bot.data.player_stats import synthetic_player_log
from afl_bot.data.stoppages import BOUNDARY_THROWIN_COL, load_boundary_throwins
from afl_bot.data.storage import read_parquet, write_parquet
from afl_bot.data.weather import attach_weather

GAMES = pd.DataFrame([
    {"year": 2025, "round": 1, "unixtime": 1, "hteam": "Adelaide", "ateam": "Geelong",
     "hscore": 90, "ascore": 80, "hgoals": 13, "agoals": 11, "venue": "Adelaide Oval",
     "date": "2025-03-15 19:40:00"},
])

# The player-log contract every prop/pace model relies on (afl_bot.models.props).
PLAYER_LOG_CONTRACT = {
    "year", "round", "unixtime", "player", "team", "opponent", "is_home",
    "disposals", "goals", "marks", "tackles",
    "team_disposals", "team_goals", "team_marks", "team_tackles",
}


def test_player_log_contract():
    log = synthetic_player_log(GAMES)
    assert PLAYER_LOG_CONTRACT <= set(log.columns)
    assert not log.empty


def test_storage_roundtrip_preserves_columns_and_schema(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    write_parquet(df, "schema_probe", schema_version=3, cache_dir=tmp_path)
    back = read_parquet("schema_probe", expected_schema_version=3, cache_dir=tmp_path)
    assert list(back.columns) == ["a", "b"]
    assert len(back) == 2


def test_weather_attach_schema():
    weather = pd.DataFrame([
        {"venue": "Adelaide Oval", "date": "2025-03-15", "rain_mm": 1.0, "wind_kmh": 10.0},
    ])
    out = attach_weather(GAMES, weather=weather)
    assert {"rain_mm", "wind_kmh", "roofed", "is_wet"} <= set(out.columns)
    assert out["is_wet"].dtype == bool


def test_boundary_throwins_contract(tmp_path):
    src = pd.DataFrame([
        {"year": 2025, "round": 1, "hteam": "Adelaide", "ateam": "Geelong",
         BOUNDARY_THROWIN_COL: 34},
    ])
    out = load_boundary_throwins(src, cache_dir=tmp_path)
    assert list(out.columns) == ["year", "round", "hteam", "ateam", BOUNDARY_THROWIN_COL]

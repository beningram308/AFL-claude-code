from unittest.mock import patch

import pandas as pd
import requests

from afl_bot.data.player_stats import load_player_log, synthetic_player_log

GAMES = pd.DataFrame([
    {
        "year": 2025, "round": 1, "unixtime": 1234567890,
        "hteam": "Adelaide", "ateam": "Geelong",
        "hscore": 90, "ascore": 80, "hgoals": 13, "agoals": 11,
    },
])


def test_load_player_log_falls_back_to_synthetic_when_real_empty():
    with patch("afl_bot.data.player_stats.fetch_player_stats", return_value=pd.DataFrame()), \
         patch("afl_bot.data.player_stats.fetch_fryzigg_player_stats", return_value=pd.DataFrame()):
        log = load_player_log(GAMES)
    expected = synthetic_player_log(GAMES)
    assert not log.empty
    assert set(log["player"]) == set(expected["player"])


def test_load_player_log_falls_back_on_request_exception():
    with patch("afl_bot.data.player_stats.fetch_player_stats",
               side_effect=requests.RequestException("offline")), \
         patch("afl_bot.data.player_stats.fetch_fryzigg_player_stats",
               side_effect=requests.RequestException("offline")):
        log = load_player_log(GAMES)
    assert not log.empty
    assert set(log["player"]) == set(synthetic_player_log(GAMES)["player"])


def test_load_player_log_prefer_real_false_skips_network():
    with patch("afl_bot.data.player_stats.fetch_player_stats") as mock_fetch, \
         patch("afl_bot.data.player_stats.fetch_fryzigg_player_stats") as mock_fz_fetch:
        log = load_player_log(GAMES, prefer_real=False)
    mock_fetch.assert_not_called()
    mock_fz_fetch.assert_not_called()
    assert not log.empty


def test_load_player_log_uses_real_data_when_available():
    real = pd.DataFrame([{
        "year": 2025, "round": 1, "unixtime": 1234567890, "player": "Real Player",
        "team": "Adelaide", "opponent": "Geelong", "is_home": True,
        "disposals": 25, "goals": 2, "marks": 5, "tackles": 4,
        "team_disposals": 25, "team_goals": 2, "team_marks": 5, "team_tackles": 4,
    }])
    with patch("afl_bot.data.player_stats.fetch_player_stats", return_value=pd.DataFrame({"x": [1]})), \
         patch("afl_bot.data.player_stats.to_player_log", return_value=real), \
         patch("afl_bot.data.player_stats.fetch_fryzigg_player_stats", return_value=pd.DataFrame()):
        log = load_player_log(GAMES)
    assert list(log["player"]) == ["Real Player"]


def test_load_player_log_combines_fryzigg_history_and_dfs_current_season():
    fz_log = pd.DataFrame([
        {
            "year": 2024, "round": 1, "unixtime": 1111111111, "player": "History Player",
            "team": "Adelaide", "opponent": "Geelong", "is_home": True,
            "disposals": 20, "goals": 1, "marks": 4, "tackles": 3,
            "team_disposals": 20, "team_goals": 1, "team_marks": 4, "team_tackles": 3,
            "behinds": 1,
        },
    ])
    current_log = pd.DataFrame([
        {
            "year": 2025, "round": 1, "unixtime": 1234567890, "player": "Current Player",
            "team": "Adelaide", "opponent": "Geelong", "is_home": True,
            "disposals": 25, "goals": 2, "marks": 5, "tackles": 4,
            "team_disposals": 25, "team_goals": 2, "team_marks": 5, "team_tackles": 4,
            "centre_bounce_attendances": 10,
        },
    ])
    with patch("afl_bot.data.player_stats.fetch_player_stats", return_value=pd.DataFrame({"x": [1]})), \
         patch("afl_bot.data.player_stats.to_player_log", return_value=current_log), \
         patch("afl_bot.data.player_stats.fetch_fryzigg_player_stats", return_value=pd.DataFrame({"y": [1]})), \
         patch("afl_bot.data.player_stats.fryzigg_to_player_log", return_value=fz_log):
        log = load_player_log(GAMES)

    assert set(log["player"]) == {"History Player", "Current Player"}
    # columns unique to one source are present with NaNs for the other
    assert "behinds" in log.columns and "centre_bounce_attendances" in log.columns
    history_row = log[log["player"] == "History Player"].iloc[0]
    assert pd.isna(history_row["centre_bounce_attendances"])

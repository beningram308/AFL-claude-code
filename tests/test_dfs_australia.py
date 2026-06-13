from unittest.mock import Mock, patch

import pandas as pd
import pytest

from afl_bot.data.dfs_australia import fetch_player_stats, to_player_log

RAW_RECORDS = [
    {
        "player": "Jordan Dawson", "team": "ADE", "opp": "GEE", "year": "2026", "round": "1",
        "kicks": "17", "handballs": "5", "marks": "7", "tackles": "9", "hitouts": "0",
        "ruckContests": "0", "freesFor": "1", "freesAgainst": "2", "goals": "3", "behinds": "0",
        "centreBounceAttendances": "16", "kickIns": "0", "kickInsPlayon": "0",
        "timeOnGroundPercentage": "89", "dreamTeamPoints": "131", "startingPosition": "C", "SC": "114",
    },
    {
        "player": "Jeremy Cameron", "team": "GEE", "opp": "ADE", "year": "2026", "round": "1",
        "kicks": "10", "handballs": "2", "marks": "5", "tackles": "1", "hitouts": "0",
        "ruckContests": "0", "freesFor": "0", "freesAgainst": "0", "goals": "4", "behinds": "1",
        "centreBounceAttendances": "0", "kickIns": "0", "kickInsPlayon": "0",
        "timeOnGroundPercentage": "80", "dreamTeamPoints": "100", "startingPosition": "FF", "SC": "90",
    },
]

GAMES = pd.DataFrame([
    {"year": 2026, "round": 1, "unixtime": 1234567890, "hteam": "Adelaide", "ateam": "Geelong"},
])


def _mock_response(payload):
    resp = Mock()
    resp.json.return_value = payload
    resp.raise_for_status = Mock()
    return resp


def test_fetch_player_stats_caches(tmp_path):
    with patch("afl_bot.data.dfs_australia.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"data": RAW_RECORDS})

        df = fetch_player_stats(cache_dir=tmp_path)
        assert len(df) == 2
        assert mock_post.call_count == 1

        # Second call should hit the cache, not the network.
        df2 = fetch_player_stats(cache_dir=tmp_path)
        assert len(df2) == 2
        assert mock_post.call_count == 1


def test_fetch_player_stats_snapshots_rounds(tmp_path):
    with patch("afl_bot.data.dfs_australia.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"data": RAW_RECORDS})
        fetch_player_stats(cache_dir=tmp_path)
    # per-round dated snapshot written so CBA history accumulates (§7.3)
    assert (tmp_path / "dfs_2026_r1.parquet").exists()


def test_to_player_log_reshapes_and_normalises_teams():
    raw = pd.DataFrame.from_records(RAW_RECORDS)
    for col in ("year", "round", "kicks", "handballs", "marks", "tackles", "goals"):
        raw[col] = pd.to_numeric(raw[col])

    log = to_player_log(raw, GAMES)

    assert set(log["team"]) == {"Adelaide", "Geelong"}
    assert set(log["opponent"]) == {"Adelaide", "Geelong"}

    dawson = log[log["player"] == "Jordan Dawson"].iloc[0]
    assert dawson["disposals"] == 22  # 17 kicks + 5 handballs
    assert dawson["is_home"] is True or dawson["is_home"] == True  # noqa: E712
    assert dawson["unixtime"] == 1234567890
    assert dawson["team_disposals"] == 22  # only player for Adelaide in fixture


def test_to_player_log_drops_unmatched_games():
    raw = pd.DataFrame.from_records(RAW_RECORDS)
    for col in ("year", "round", "kicks", "handballs", "marks", "tackles", "goals"):
        raw[col] = pd.to_numeric(raw[col])
    raw.loc[0, "round"] = 99  # no corresponding fixture in GAMES

    log = to_player_log(raw, GAMES)
    assert "Jordan Dawson" not in set(log["player"])
    assert "Jeremy Cameron" in set(log["player"])


def test_to_player_log_empty_input():
    assert to_player_log(pd.DataFrame(), GAMES).empty

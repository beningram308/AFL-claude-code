from unittest.mock import Mock, patch

import pandas as pd

from afl_bot.data.fryzigg import fetch_fryzigg_player_stats, to_player_log

RAW_DF = pd.DataFrame([
    {
        "match_date": pd.Timestamp("2024-03-15"), "match_round": "1",
        "match_home_team": "Carlton", "match_away_team": "Richmond",
        "player_first_name": "Patrick", "player_last_name": "Cripps",
        "player_team": "Carlton",
        "kicks": 15, "handballs": 10, "marks": 6, "tackles": 7,
        "disposals": 25, "goals": 1, "behinds": 0, "hitouts": 0,
        "ruck_contests": 0, "free_kicks_for": 2, "free_kicks_against": 1,
        "time_on_ground_percentage": 88, "afl_fantasy_score": 120, "supercoach_score": 110,
    },
    {
        "match_date": pd.Timestamp("2024-03-15"), "match_round": "1",
        "match_home_team": "Carlton", "match_away_team": "Richmond",
        "player_first_name": "Dustin", "player_last_name": "Martin",
        "player_team": "Richmond",
        "kicks": 12, "handballs": 8, "marks": 4, "tackles": 3,
        "disposals": 20, "goals": 2, "behinds": 1, "hitouts": 0,
        "ruck_contests": 0, "free_kicks_for": 1, "free_kicks_against": 0,
        "time_on_ground_percentage": 82, "afl_fantasy_score": 105, "supercoach_score": 95,
    },
])


def _mock_response(content: bytes):
    resp = Mock()
    resp.content = content
    resp.raise_for_status = Mock()
    return resp


def test_fetch_fryzigg_player_stats_caches(tmp_path):
    raw_with_year = RAW_DF.copy()
    with patch("afl_bot.data.fryzigg.requests.get") as mock_get, \
         patch("pyreadr.read_r", return_value={"df": raw_with_year}):
        mock_get.return_value = _mock_response(b"fake-rds-bytes")

        df = fetch_fryzigg_player_stats(min_season=2012, cache_dir=tmp_path)
        assert len(df) == 2
        assert "year" in df.columns
        assert mock_get.call_count == 1

        # second call hits cache, no network/pyreadr needed
        df2 = fetch_fryzigg_player_stats(min_season=2012, cache_dir=tmp_path)
        assert len(df2) == 2
        assert mock_get.call_count == 1


def test_fetch_fryzigg_filters_min_season(tmp_path):
    old_row = RAW_DF.iloc[[0]].copy()
    old_row["match_date"] = pd.Timestamp("2005-03-15")
    df = pd.concat([RAW_DF, old_row], ignore_index=True)

    with patch("afl_bot.data.fryzigg.requests.get") as mock_get, \
         patch("pyreadr.read_r", return_value={"df": df}):
        mock_get.return_value = _mock_response(b"fake-rds-bytes")
        out = fetch_fryzigg_player_stats(min_season=2012, cache_dir=tmp_path)

    assert (out["year"] >= 2012).all()
    assert len(out) == 2


def test_to_player_log_reshapes_and_normalises_teams():
    raw = RAW_DF.copy()
    raw["year"] = raw["match_date"].dt.year

    log = to_player_log(raw)

    assert set(log["team"]) == {"Carlton", "Richmond"}
    assert set(log["opponent"]) == {"Carlton", "Richmond"}

    cripps = log[log["player"] == "Patrick Cripps"].iloc[0]
    assert cripps["is_home"] is True or cripps["is_home"] == True  # noqa: E712
    assert cripps["opponent"] == "Richmond"
    assert cripps["disposals"] == 25
    assert cripps["team_disposals"] == 25
    assert cripps["round"] == 1  # match_round "1" -> real round 1 (§7.2)
    assert cripps["round_ordinal"] == 0  # chronological ordinal kept separately


def test_to_player_log_round_ordinal_orders_finals_after_home_and_away():
    raw = pd.concat([RAW_DF, RAW_DF.iloc[[0]].assign(
        match_date=pd.Timestamp("2024-09-28"), match_round="Grand Final",
    )], ignore_index=True)
    raw["year"] = raw["match_date"].dt.year

    log = to_player_log(raw)
    rounds = dict(zip(
        log.assign(label=raw["match_round"])["label"],
        log["round"],
    ))
    assert rounds["1"] < rounds["Grand Final"]


def test_to_player_log_empty_input():
    assert to_player_log(pd.DataFrame()).empty

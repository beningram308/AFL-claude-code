from io import BytesIO
from unittest.mock import Mock, patch

import pandas as pd

from afl_bot.data.odds import attach_odds, fetch_historical_odds

RAW_DF = pd.DataFrame([
    {
        "Date": pd.Timestamp("2024-03-15"), "Home Team": "Carlton", "Away Team": "Richmond",
        "Home Score": 90, "Away Score": 80,
        "Home Odds Open": 1.80, "Home Odds Close": 1.70,
        "Away Odds Open": 2.05, "Away Odds Close": 2.20,
        "Total Score Open": 170.5, "Total Score Close": 172.5,
        "Total Score Over Close": 1.90, "Total Score Under Close": 1.90,
    },
    {
        "Date": pd.Timestamp("2024-03-22"), "Home Team": "Adelaide", "Away Team": "Geelong",
        "Home Score": 70, "Away Score": 100,
        "Home Odds Open": 2.50, "Home Odds Close": 2.80,
        "Away Odds Open": 1.55, "Away Odds Close": 1.45,
        "Total Score Open": 165.5, "Total Score Close": 168.5,
        "Total Score Over Close": 1.90, "Total Score Under Close": 1.90,
    },
])

GAMES = pd.DataFrame([
    {"year": 2024, "round": 1, "hteam": "Carlton", "ateam": "Richmond", "hscore": 90, "ascore": 80},
    {"year": 2024, "round": 2, "hteam": "Adelaide", "ateam": "Geelong", "hscore": 70, "ascore": 100},
    {"year": 2024, "round": 3, "hteam": "Sydney", "ateam": "Essendon", "hscore": 60, "ascore": 50},
])


def _mock_response(df: pd.DataFrame):
    buf = BytesIO()
    # write_excel needs a header row to skip (header=1 in fetch_historical_odds)
    with pd.ExcelWriter(buf) as writer:
        pd.DataFrame([["placeholder"] * len(df.columns)], columns=df.columns).to_excel(
            writer, index=False, header=False, startrow=0,
        )
        df.to_excel(writer, index=False, startrow=1)
    resp = Mock()
    resp.content = buf.getvalue()
    resp.raise_for_status = Mock()
    return resp


def test_fetch_historical_odds_caches(tmp_path):
    with patch("afl_bot.data.odds.requests.get") as mock_get:
        mock_get.return_value = _mock_response(RAW_DF)

        df = fetch_historical_odds(cache_dir=tmp_path)
        assert len(df) == 2
        assert mock_get.call_count == 1
        assert set(df["hteam"]) == {"Carlton", "Adelaide"}
        assert "year" in df.columns

        # second call hits cache, no network
        df2 = fetch_historical_odds(cache_dir=tmp_path)
        assert len(df2) == 2
        assert mock_get.call_count == 1


def test_fetch_historical_odds_refetches_when_stale(tmp_path):
    import os
    import time as _time
    with patch("afl_bot.data.odds.requests.get") as mock_get:
        mock_get.return_value = _mock_response(RAW_DF)
        fetch_historical_odds(cache_dir=tmp_path)
        assert mock_get.call_count == 1
        # age the cache file past the max -> stale -> re-download (§7.5)
        old = _time.time() - 10 * 86400
        os.utime(tmp_path / "aussportsbetting_afl_odds.parquet", (old, old))
        fetch_historical_odds(cache_dir=tmp_path)          # default max_age_days=3
        assert mock_get.call_count == 2
        # max_age_days=None caches forever
        fetch_historical_odds(cache_dir=tmp_path, max_age_days=None)
        assert mock_get.call_count == 2


def test_attach_odds_joins_on_year_hteam_ateam(tmp_path):
    with patch("afl_bot.data.odds.requests.get") as mock_get:
        mock_get.return_value = _mock_response(RAW_DF)
        odds = fetch_historical_odds(cache_dir=tmp_path)

    merged = attach_odds(GAMES, odds)
    assert len(merged) == len(GAMES)

    carlton_row = merged[merged["hteam"] == "Carlton"].iloc[0]
    assert carlton_row["home_odds_close"] == 1.70

    # game with no matching odds row keeps NaN odds columns
    sydney_row = merged[merged["hteam"] == "Sydney"].iloc[0]
    assert pd.isna(sydney_row["home_odds_close"])

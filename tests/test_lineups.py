import json
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from afl_bot.cli import _normalize_name, _select_players
from afl_bot.data.lineups import (
    _parse_footywire_selections,
    _slug_to_name,
    fetch_lineup,
    load_lineup,
    load_lineup_tog,
)


def test_load_lineup_normalises_teams_and_skips_meta(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({
        "Geelong": ["Patrick Dangerfield", "Bailey Smith "],
        "PTA": ["Connor Rozee"],                 # alias -> Port Adelaide
        "_rules": {"h2h_draw": "refund"},        # ignored
        "Not A Team": ["Whoever"],               # unknown -> skipped
    }))
    lineup = load_lineup(str(path))
    assert lineup["Geelong"] == {"Patrick Dangerfield", "Bailey Smith"}
    assert lineup["Port Adelaide"] == {"Connor Rozee"}
    assert "_rules" not in lineup
    assert "Not A Team" not in lineup


def test_load_lineup_none_returns_empty():
    assert load_lineup(None) == {}


def _log():
    rows = []
    ut = 0
    # Carlton: 18 current-season (2026) players + a retired star only in 2012.
    for rnd in range(1, 6):
        for i in range(18):
            ut += 1
            rows.append({"year": 2026, "round": rnd, "unixtime": ut, "player": f"Cur{i}",
                         "team": "Carlton", "opponent": "Geelong", "is_home": True,
                         "disposals": 15 + i, "goals": 0, "marks": 3, "tackles": 3})
    # retired legend: huge career disposals, but ONLY 2012 -> must be excluded
    rows.append({"year": 2012, "round": 1, "unixtime": 999999, "player": "RetiredLegend",
                 "team": "Carlton", "opponent": "Geelong", "is_home": True,
                 "disposals": 40, "goals": 2, "marks": 8, "tackles": 6})
    return pd.DataFrame(rows)


def test_select_players_excludes_non_current_season_player():
    players = _select_players(_log(), "Carlton", current_year=2026, n=5)
    assert "RetiredLegend" not in players           # career avg can't sneak in
    assert len(players) == 5
    assert all(p.startswith("Cur") for p in players)


def test_select_players_confirmed_lineup_gates_and_prices_all():
    confirmed = {"Cur0", "Cur5", "Cur17"}
    players = _select_players(_log(), "Carlton", current_year=2026, n=5, confirmed=confirmed)
    assert set(players) == confirmed                 # only named players, all of them


def test_select_players_falls_back_to_prior_season_early_year():
    # only last-season data present -> still returns players (no current games yet)
    log = _log().assign(year=2025)
    players = _select_players(log, "Carlton", current_year=2026, n=3)
    assert len(players) == 3


# --------------------------------------------------------------------------- #
# load_lineup with rich per-player object format (C2)
# --------------------------------------------------------------------------- #

def test_load_lineup_handles_rich_object_format(tmp_path):
    """Dict-style entries with returning/TOG fields are still included in the confirmed set."""
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({
        "Carlton": [
            "Patrick Cripps",
            {"player": "Sam Walsh", "returning_from_injury": True},
            {"player": "Adam Cerra", "expected_tog": 0.70},
        ],
    }))
    lineup = load_lineup(str(path))
    assert lineup["Carlton"] == {"Patrick Cripps", "Sam Walsh", "Adam Cerra"}


def test_load_lineup_backward_compatible_plain_strings(tmp_path):
    """Old-style plain-string lineup files still load correctly."""
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({"Geelong": ["Patrick Dangerfield", "Tom Hawkins"]}))
    lineup = load_lineup(str(path))
    assert lineup["Geelong"] == {"Patrick Dangerfield", "Tom Hawkins"}


# --------------------------------------------------------------------------- #
# load_lineup_tog (C2)
# --------------------------------------------------------------------------- #

def test_load_lineup_tog_none_returns_empty():
    assert load_lineup_tog(None) == {}


def test_load_lineup_tog_extracts_expected_tog(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({
        "Carlton": [
            "Patrick Cripps",
            {"player": "Adam Cerra", "expected_tog": 0.70},
        ],
    }))
    tog = load_lineup_tog(str(path))
    assert "Adam Cerra" in tog
    assert tog["Adam Cerra"] == pytest.approx(0.70)
    assert "Patrick Cripps" not in tog   # plain string → no override


def test_load_lineup_tog_returning_from_injury_applies_default(tmp_path):
    from afl_bot.config import TOG_RETURN_DEFAULT
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({
        "Carlton": [{"player": "Sam Walsh", "returning_from_injury": True}],
    }))
    tog = load_lineup_tog(str(path))
    assert tog["Sam Walsh"] == pytest.approx(TOG_RETURN_DEFAULT)


def test_load_lineup_tog_managed_flag_applies_default(tmp_path):
    from afl_bot.config import TOG_RETURN_DEFAULT
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({
        "Melbourne": [{"player": "Clayton Oliver", "managed": True}],
    }))
    tog = load_lineup_tog(str(path))
    assert tog["Clayton Oliver"] == pytest.approx(TOG_RETURN_DEFAULT)


def test_load_lineup_tog_skips_meta_keys(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({
        "_rules": {"h2h_draw": "refund"},
        "Carlton": [{"player": "Sam Walsh", "expected_tog": 0.80}],
    }))
    tog = load_lineup_tog(str(path))
    assert "Sam Walsh" in tog
    assert len(tog) == 1


# --------------------------------------------------------------------------- #
# fetch_lineup (REAL-MULTIS Problem 1 / auto-lineup)
# --------------------------------------------------------------------------- #

def _make_footywire_html(teams: dict[str, list[str]]) -> str:
    """Build a minimal Footywire-style HTML page with pp- hrefs."""
    links = []
    for team_slug, player_slugs in teams.items():
        for slug in player_slugs:
            links.append(f'<a href="pp-{team_slug}--{slug}">X</a>')
    return "<html><body>" + "".join(links) + "</body></html>"


def _nkm_players(n: int = 22) -> list[str]:
    """Return n distinct NMK player slugs."""
    pool = [
        "harry-sheezel", "luke-davies-uniacke", "luke-parker", "caleb-daniel",
        "colby-mckercher", "jy-simpkin", "dylan-stephens", "george-wardlaw",
        "finn-o-sullivan", "tristan-xerri", "jason-horne-francis", "tom-powell",
        "will-hayes", "nick-larkey", "charlie-comben", "flynn-perez",
        "cameron-zurhaar", "aidan-corr", "ben-mckay", "dom-tyson",
        "harry-potter", "jack-ziebell",
    ]
    return pool[:n]


def test_slug_to_name_simple():
    assert _slug_to_name("brennan-cox") == "Brennan Cox"
    assert _slug_to_name("harry-sheezel") == "Harry Sheezel"


def test_slug_to_name_mc_prefix():
    assert _slug_to_name("judd-mcvee") == "Judd McVee"
    assert _slug_to_name("brayden-macdonald") == "Brayden MacDonald"


def test_slug_to_name_hyphenated_name_becomes_spaces():
    # hyphens in the slug can't be distinguished from word separators; we output
    # spaces and rely on _normalize_name for matching
    result = _slug_to_name("luke-davies-uniacke")
    assert result == "Luke Davies Uniacke"


def test_normalize_name_handles_hyphenated():
    # "Luke Davies Uniacke" (from slug) and "Luke Davies-Uniacke" (from log) both
    # normalise to the same key, enabling fuzzy match
    assert _normalize_name("Luke Davies Uniacke") == _normalize_name("Luke Davies-Uniacke")
    assert _normalize_name("Jason Horne Francis") == _normalize_name("Jason Horne-Francis")


def test_parse_footywire_selections_extracts_teams_and_players():
    html = _make_footywire_html({
        "north-melbourne-kangaroos": _nkm_players(22),
        "west-coast-eagles": [
            "harley-reid", "tim-kelly", "tom-mccarthy", "liam-duggan",
            "ryan-maric", "jack-graham", "josh-lindsay", "willem-duursma",
            "milan-murdock", "deven-robertson", "jake-watkins", "jamie-cripps",
            "luke-shuey", "elliot-yeo", "nic-naitanui", "oscar-allen",
            "alex-witherden", "brady-hough", "tom-barrass", "shannon-hurn",
            "andrew-gaff", "liam-ryan",
        ],
    })
    result = _parse_footywire_selections(html)
    assert "North Melbourne" in result
    assert "West Coast" in result
    assert "Harry Sheezel" in result["North Melbourne"]
    assert "Harley Reid" in result["West Coast"]


def test_parse_footywire_selections_skips_teams_below_threshold():
    # A team with fewer than 18 player links is treated as "not yet posted"
    html = _make_footywire_html({
        "north-melbourne-kangaroos": _nkm_players(22),
        "west-coast-eagles": ["harley-reid", "tim-kelly"],   # only 2 — skip
    })
    result = _parse_footywire_selections(html)
    assert "North Melbourne" in result
    assert "West Coast" not in result


def test_parse_footywire_selections_empty_page_returns_empty():
    assert _parse_footywire_selections("<html></html>") == {}


def test_fetch_lineup_uses_cache(tmp_path):
    html = _make_footywire_html({"north-melbourne-kangaroos": _nkm_players(22)})
    mock_resp = Mock()
    mock_resp.text = html
    mock_resp.raise_for_status = Mock()
    with patch("afl_bot.data.lineups.requests.get", return_value=mock_resp) as mock_get:
        r1 = fetch_lineup(2026, 15, cache_seconds=3600, cache_dir=tmp_path)
        r2 = fetch_lineup(2026, 15, cache_seconds=3600, cache_dir=tmp_path)
    assert mock_get.call_count == 1    # second call hits cache
    assert r1 == r2
    assert "North Melbourne" in r1


def test_fetch_lineup_network_error_returns_empty(tmp_path):
    import requests as req
    with patch("afl_bot.data.lineups.requests.get",
               side_effect=req.RequestException("timeout")):
        result = fetch_lineup(2026, 15, cache_dir=tmp_path)
    assert result == {}


def test_select_players_normalized_match_for_hyphenated_name():
    """_select_players must include a log-name like "Luke Davies-Uniacke" when the
    confirmed set was built from a slug and contains "Luke Davies Uniacke"."""
    rows = []
    for rnd in range(1, 6):
        rows.append({"year": 2026, "round": rnd, "unixtime": rnd,
                     "player": "Luke Davies-Uniacke", "team": "North Melbourne",
                     "opponent": "Geelong", "is_home": True,
                     "disposals": 25, "goals": 0, "marks": 3, "tackles": 3})
    log = pd.DataFrame(rows)
    confirmed_from_footywire = {"Luke Davies Uniacke"}   # spaces, not hyphens
    players = _select_players(log, "North Melbourne", current_year=2026, n=5,
                              confirmed=confirmed_from_footywire)
    assert "Luke Davies-Uniacke" in players   # matched via normalisation

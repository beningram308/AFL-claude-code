import json
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from afl_bot.cli import _normalize_name, _select_players
from afl_bot.data.lineups import (
    _parse_footywire_selections,
    _slug_to_name,
    apply_outs,
    fetch_lineup,
    load_lineup,
    load_lineup_tog,
    load_outs,
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
    """Build a minimal Footywire-style HTML page: each player link sits in a
    grid-style ``<tr class="lightcolor">`` row -- the on-field position table
    the real-site parser always treats as confirmed, with no section
    headers needed (see ``_make_footywire_team_html`` below for the
    Interchange/Emergencies/Ins/Outs sidebar structure)."""
    rows = []
    for team_slug, player_slugs in teams.items():
        for slug in player_slugs:
            rows.append(f'<tr class="lightcolor"><td><a href="pp-{team_slug}--{slug}">X</a></td></tr>')
    return "<html><body><table>" + "".join(rows) + "</table></body></html>"


def _make_footywire_team_html(team_slug: str, grid: list[str], *,
                              interchange: list[str] = (), emergencies: list[str] = (),
                              ins: list[str] = (), outs: list[str] = ()) -> str:
    """Build a realistic single-team Footywire block: an on-field grid plus
    a sidebar with Interchange/Emergencies/Ins/Outs sections in that order,
    matching the real site's structure (each section a ``<b>`` header row
    followed by ``pp-`` link rows)."""
    grid_rows = "".join(
        f'<tr class="lightcolor"><td><a href="pp-{team_slug}--{slug}">X</a></td></tr>'
        for slug in grid
    )

    def _section(name: str, slugs: list[str]) -> str:
        rows = [f"<tr><td><b>{name}</b></td></tr>"]
        rows += [f'<tr><td><a href="pp-{team_slug}--{slug}">X</a></td></tr>' for slug in slugs]
        return "".join(rows)

    sidebar = (_section("Interchange", list(interchange)) + _section("Emergencies", list(emergencies))
               + _section("Ins", list(ins)) + _section("Outs", list(outs)))
    return (f'<html><body>'
           f'<table cellpadding="2">{sidebar}</table>'
           f'<table>{grid_rows}</table>'
           f'</body></html>')


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


# --------------------------------------------------------------------------- #
# FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART B2: section-aware parsing (kills
# Emergencies/Ins/Outs leaking into the confirmed set)
# --------------------------------------------------------------------------- #

def test_parse_footywire_selections_keeps_interchange_excludes_emergencies_ins_outs():
    html = _make_footywire_team_html(
        "greater-western-sydney-giants",
        grid=[f"grid-{i}" for i in range(18)],
        interchange=["bench-1", "bench-2", "bench-3", "bench-4", "bench-5"],
        emergencies=["jesse-hogan"],
        ins=["new-in-1"],
        outs=["cut-1"],
    )
    result = _parse_footywire_selections(html)
    confirmed = result["Greater Western Sydney"]
    assert len(confirmed) == 23                      # 18 grid + 5 interchange (incl. sub)
    assert "Jesse Hogan" not in confirmed             # Emergencies excluded
    assert "Cut 1" not in confirmed                   # Outs excluded even though linked
    assert "Bench 1" in confirmed                     # Interchange kept


def test_parse_footywire_selections_ins_not_double_counted_if_only_in_sidebar():
    # An "Ins" entry not also in the grid/interchange must NOT be confirmed --
    # it's informational only (the player's actual selection is the grid row).
    html = _make_footywire_team_html(
        "hawthorn", grid=[f"grid-{i}" for i in range(18)],
        interchange=["bench-1"], ins=["sidebar-only-in"],
    )
    confirmed = _parse_footywire_selections(html)["Hawthorn"]
    assert "Sidebar Only In" not in confirmed


def test_fetch_lineup_warns_when_team_has_extended_squad(tmp_path, capsys):
    # No Emergencies section posted yet -> 18 grid + 10 interchange = 28, >24.
    html = _make_footywire_team_html(
        "north-melbourne-kangaroos", grid=[f"grid-{i}" for i in range(18)],
        interchange=[f"bench-{i}" for i in range(10)],
    )
    mock_resp = Mock()
    mock_resp.text = html
    mock_resp.raise_for_status = Mock()
    with patch("afl_bot.data.lineups.requests.get", return_value=mock_resp):
        fetch_lineup(2026, 16, cache_dir=tmp_path)
    err = capsys.readouterr().err
    assert "North Melbourne" in err and "28" in err and "WARNING" in err


def test_fetch_lineup_no_warning_for_normal_squad_size(tmp_path, capsys):
    html = _make_footywire_team_html(
        "hawthorn", grid=[f"grid-{i}" for i in range(18)],
        interchange=["bench-1", "bench-2", "bench-3", "bench-4", "bench-5"],
    )
    mock_resp = Mock()
    mock_resp.text = html
    mock_resp.raise_for_status = Mock()
    with patch("afl_bot.data.lineups.requests.get", return_value=mock_resp):
        fetch_lineup(2026, 16, cache_dir=tmp_path)
    assert "WARNING" not in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART B1: manual outs override
# --------------------------------------------------------------------------- #

def test_load_outs_from_dedicated_file(tmp_path):
    path = tmp_path / "outs.json"
    path.write_text(json.dumps({"_outs": {"GWS": ["Jesse Hogan"]}}))
    outs = load_outs(str(path))
    assert outs == {"Greater Western Sydney": {"Jesse Hogan"}}


def test_load_outs_embedded_in_lineup_file(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({
        "Hawthorn": ["Jai Newcombe"],
        "_outs": {"Hawthorn": ["Some Cut Player"]},
    }))
    outs = load_outs(str(path))
    assert outs == {"Hawthorn": {"Some Cut Player"}}


def test_load_outs_none_path_returns_empty():
    assert load_outs(None) == {}


def test_load_outs_no_outs_key_returns_empty(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({"Hawthorn": ["Jai Newcombe"]}))
    assert load_outs(str(path)) == {}


def test_load_outs_unknown_team_skipped(tmp_path):
    path = tmp_path / "outs.json"
    path.write_text(json.dumps({"_outs": {"Not A Team": ["X"]}}))
    assert load_outs(str(path)) == {}


def test_apply_outs_removes_named_player_normalised():
    lineup = {"Greater Western Sydney": {"Jesse Hogan", "Lachie Whitfield"}}
    outs = {"Greater Western Sydney": {"jesse-hogan"}}    # hyphen, still matches
    new_lineup, n = apply_outs(lineup, outs)
    assert new_lineup == {"Greater Western Sydney": {"Lachie Whitfield"}}
    assert n == 1


def test_apply_outs_team_not_in_lineup_is_noop():
    lineup = {"Hawthorn": {"Jai Newcombe"}}
    new_lineup, n = apply_outs(lineup, {"Richmond": {"X"}})
    assert new_lineup == lineup
    assert n == 0


def test_apply_outs_does_not_mutate_input_lineup():
    lineup = {"Hawthorn": {"Jai Newcombe", "Karl Amon"}}
    apply_outs(lineup, {"Hawthorn": {"Karl Amon"}})
    assert lineup == {"Hawthorn": {"Jai Newcombe", "Karl Amon"}}   # original untouched

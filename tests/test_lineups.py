import json

import pandas as pd

from afl_bot.cli import _select_players
from afl_bot.data.lineups import load_lineup


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

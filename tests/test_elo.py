import json

import pandas as pd

from afl_bot.ratings.elo import EloRatings, mov_multiplier

# Two seasons so season carryover + eval windows are exercised.
GAMES = pd.DataFrame([
    {"year": 2020, "round": 1, "unixtime": 1, "hteam": "A", "ateam": "B", "hscore": 100, "ascore": 60},
    {"year": 2020, "round": 2, "unixtime": 2, "hteam": "B", "ateam": "A", "hscore": 90, "ascore": 80},
    {"year": 2021, "round": 1, "unixtime": 3, "hteam": "A", "ateam": "B", "hscore": 70, "ascore": 70},
])


def test_first_game_uses_initial_pre_ratings():
    elo = EloRatings()
    hist = elo.fit(GAMES)
    first = hist.iloc[0]
    assert first["home_elo_pre"] == elo.initial
    assert first["away_elo_pre"] == elo.initial


def test_margin_update_matches_hand_computation():
    # Single game: A beats B by 40 from level (both 1500).
    g = GAMES.iloc[[0]]
    elo = EloRatings(k=35.0, home_advantage=10.0, points_per_400=92.0, scale=400.0, margin_cap=80.0)
    elo.fit(g)

    hga = 10.0 / 92.0 * 400.0
    expected = 1.0 / (1.0 + 10.0 ** (-hga / 400.0))
    actual = 40.0 / 160.0 + 0.5
    delta = 35.0 * (actual - expected)

    assert abs(elo.ratings["A"] - (1500.0 + delta)) < 1e-9
    assert abs(elo.ratings["B"] - (1500.0 - delta)) < 1e-9


def test_points_per_400_scales_expected_margin():
    elo = EloRatings(points_per_400=92.0, home_advantage=0.0)
    elo.ratings = {"A": 1600.0, "B": 1500.0}
    m92 = elo.expected_margin("A", "B")

    elo2 = EloRatings(points_per_400=46.0, home_advantage=0.0)
    elo2.ratings = {"A": 1600.0, "B": 1500.0}
    m46 = elo2.expected_margin("A", "B")

    assert abs(m92 - 2 * m46) < 1e-9  # half the points_per_400 -> half the margin


def test_mov_mode_differs_from_margin_mode():
    margin_elo = EloRatings(update_mode="margin")
    mov_elo = EloRatings(update_mode="mov")
    margin_elo.fit(GAMES)
    mov_elo.fit(GAMES)
    assert margin_elo.ratings != mov_elo.ratings


def test_mov_mode_draw_does_not_update():
    draw = pd.DataFrame([
        {"year": 2020, "round": 1, "unixtime": 1, "hteam": "A", "ateam": "B", "hscore": 80, "ascore": 80},
    ])
    elo = EloRatings(update_mode="mov")
    elo.fit(draw)
    # ln(|0|+1) == 0 -> zero multiplier -> no rating movement
    assert elo.ratings["A"] == elo.initial
    assert elo.ratings["B"] == elo.initial


def test_mov_multiplier_upset_exceeds_favourite_and_grows_with_margin():
    # Same 40-pt margin: an upset (winner was 200 behind) moves ratings more
    # than a favourite (winner was 200 ahead) winning by the same margin.
    assert mov_multiplier(40, -200) > mov_multiplier(40, 200)
    # Monotonic in the margin.
    assert mov_multiplier(80, 0) > mov_multiplier(20, 0)


def test_fit_sorts_chronologically_regardless_of_input_order():
    shuffled = GAMES.iloc[[2, 0, 1]].reset_index(drop=True)
    a = EloRatings().fit(GAMES)
    b = EloRatings().fit(shuffled)
    # Same chronological order -> identical pre-match ratings sequence.
    assert a[["home_elo_pre", "away_elo_pre"]].round(9).equals(
        b[["home_elo_pre", "away_elo_pre"]].round(9)
    )


def test_save_load_round_trips_new_fields(tmp_path):
    elo = EloRatings(k=42.0, points_per_400=70.0, update_mode="mov", mov_correction=3.1)
    elo.fit(GAMES)
    path = tmp_path / "elo.json"
    elo.save(path)

    loaded = EloRatings.load(path)
    assert loaded.k == 42.0
    assert loaded.points_per_400 == 70.0
    assert loaded.update_mode == "mov"
    assert loaded.mov_correction == 3.1
    assert loaded.ratings == elo.ratings

    # forward-compatible: an older file missing the new keys still loads
    old = json.loads(path.read_text())
    for key in ("points_per_400", "update_mode", "mov_correction"):
        old.pop(key)
    path.write_text(json.dumps(old))
    legacy = EloRatings.load(path)
    assert legacy.points_per_400 == EloRatings().points_per_400
    assert legacy.update_mode == EloRatings().update_mode

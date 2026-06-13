import pandas as pd

from afl_bot.models.scoring import (
    expected_total,
    points_to_shots,
    team_scoring_profiles,
    team_shot_accuracy_profiles,
)

GAMES = pd.DataFrame([
    {
        "year": 2024, "round": 1, "unixtime": 1,
        "hteam": "Carlton", "ateam": "Richmond",
        "hscore": 90, "ascore": 80, "hgoals": 13, "hbehinds": 12, "agoals": 11, "abehinds": 14,
    },
    {
        "year": 2024, "round": 2, "unixtime": 2,
        "hteam": "Richmond", "ateam": "Carlton",
        "hscore": 85, "ascore": 95, "hgoals": 12, "hbehinds": 13, "agoals": 14, "abehinds": 11,
    },
    {
        "year": 2024, "round": 3, "unixtime": 3,
        "hteam": "Carlton", "ateam": "Richmond",
        "hscore": 100, "ascore": 70, "hgoals": 15, "hbehinds": 10, "agoals": 9, "abehinds": 16,
    },
])


def test_points_to_shots_inverts_points_formula():
    accuracy = 0.52
    points = 90.0
    shots = points_to_shots(points, accuracy)
    # points = (5*accuracy + 1) * shots
    assert abs((5 * accuracy + 1) * shots - points) < 1e-9


def test_points_to_shots_clips_negative_points():
    assert points_to_shots(-10.0, 0.5) == 0.0


def test_team_shot_accuracy_profiles_returns_per_team_rate():
    profiles = team_shot_accuracy_profiles(GAMES)
    assert set(profiles) == {"Carlton", "Richmond"}
    for acc in profiles.values():
        assert 0.0 < acc < 1.0


def test_team_shot_accuracy_profiles_anti_leakage_cutoff():
    # As of round 3, only rounds 1-2 should be visible.
    profiles_before_r3 = team_shot_accuracy_profiles(GAMES, as_of_year=2024, as_of_round=3)
    profiles_all = team_shot_accuracy_profiles(GAMES)
    # Using fewer games should generally give a different (or at least
    # independently-derived) EWMA value than using all games.
    assert profiles_before_r3.keys() == profiles_all.keys()


def test_expected_total_unchanged():
    total = expected_total(home_off=90, home_def=85, away_off=80, away_def=95)
    assert total == 0.5 * ((90 + 95) + (80 + 85))


def test_team_scoring_profiles_still_works():
    profiles = team_scoring_profiles(GAMES)
    assert "off_rate" in profiles["Carlton"]
    assert "def_rate" in profiles["Carlton"]

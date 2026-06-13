import pytest

from afl_bot.data.teams import (
    CANONICAL_TEAMS,
    normalize_team_name,
    team_code,
)


def test_canonical_names_are_idempotent():
    for name in CANONICAL_TEAMS:
        assert normalize_team_name(name) == name


@pytest.mark.parametrize("alias, canonical", [
    ("GWS Giants", "Greater Western Sydney"),
    ("gws", "Greater Western Sydney"),
    ("Sydney Swans", "Sydney"),
    ("South Melbourne", "Sydney"),
    ("Footscray", "Western Bulldogs"),
    ("Kangaroos", "North Melbourne"),
    ("Brisbane Bears", "Brisbane Lions"),
    ("Fitzroy", "Brisbane Lions"),
    ("Port Adelaide Power", "Port Adelaide"),
    ("West Coast Eagles", "West Coast"),
    (" geelong cats ", "Geelong"),
    ("GEE", "Geelong"),
    ("ADE", "Adelaide"),
])
def test_known_aliases_map_to_canonical(alias, canonical):
    assert normalize_team_name(alias) == canonical


def test_unknown_team_name_raises():
    with pytest.raises(KeyError):
        normalize_team_name("Not A Real Team")


def test_team_code_round_trip():
    for name in CANONICAL_TEAMS:
        code = team_code(name)
        assert normalize_team_name(code) == name

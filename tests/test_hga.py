"""Per-team venue HGA + interstate + rest (round-2 §6.1)."""

import numpy as np
import pandas as pd

from afl_bot.ratings.elo import EloRatings
from afl_bot.ratings.hga import (
    INTERSTATE_PENALTY,
    attach_hga,
    days_rest,
    fit_team_hga,
    game_hga_points,
    venue_state,
)


def test_venue_state():
    assert venue_state("M.C.G.") == "VIC"
    assert venue_state("Adelaide Oval") == "SA"
    assert venue_state("Optus Stadium") == "WA"
    assert venue_state("Unknown Ground") is None


def _games(home_margin=40, n=30):
    rows, ut = [], 0
    for r in range(n):
        ut += 7 * 86400
        # FORT always strong at home, weak away -> big home swing
        rows.append({"year": 2024, "round": r + 1, "unixtime": ut,
                     "hteam": "Fortress", "ateam": "B", "hscore": 100 + home_margin // 2,
                     "ascore": 100 - home_margin // 2, "venue": "M.C.G."})
        ut += 3 * 86400
        rows.append({"year": 2024, "round": r + 1, "unixtime": ut,
                     "hteam": "C", "ateam": "Fortress", "hscore": 100 + home_margin // 2,
                     "ascore": 100 - home_margin // 2, "venue": "Adelaide Oval"})
    return pd.DataFrame(rows)


def test_fit_team_hga_strong_home_above_league_and_shrinks():
    hga = fit_team_hga(_games(home_margin=60), league_hga=10.0)
    assert hga["Fortress"] > 12.0                 # genuine home-ground edge > league
    # a team with one game stays near the league value (shrinkage)
    one = pd.concat([_games(), pd.DataFrame([{"year": 2024, "round": 99, "unixtime": 9e9,
        "hteam": "OneOff", "ateam": "B", "hscore": 200, "ascore": 50, "venue": "M.C.G."}])],
        ignore_index=True)
    assert abs(fit_team_hga(one)["OneOff"] - 10.0) < 6.0


def test_game_hga_points_applies_interstate_penalty():
    # B (VIC) travels to Adelaide Oval (SA) -> interstate penalty added
    games = pd.DataFrame([
        {"year": 2024, "round": 1, "unixtime": 1_000_000, "hteam": "Adelaide", "ateam": "Carlton",
         "hscore": 90, "ascore": 80, "venue": "Adelaide Oval"},
        {"year": 2024, "round": 1, "unixtime": 1_000_000, "hteam": "Carlton", "ateam": "Richmond",
         "hscore": 90, "ascore": 80, "venue": "M.C.G."},
    ])
    team_hga = {"Adelaide": 10.0, "Carlton": 10.0}
    pts = game_hga_points(games, team_hga)
    assert pts[0] > pts[1]                          # interstate game has higher HGA
    assert abs((pts[0] - pts[1]) - INTERSTATE_PENALTY) < 1e-6   # Richmond(VIC)@MCG not interstate


def test_days_rest_computes_gaps():
    games = pd.DataFrame([
        {"year": 2024, "round": 1, "unixtime": 0, "hteam": "A", "ateam": "B", "hscore": 1, "ascore": 0},
        {"year": 2024, "round": 2, "unixtime": 7 * 86400, "hteam": "A", "ateam": "C", "hscore": 1, "ascore": 0},
    ])
    rested = days_rest(games)
    assert np.isnan(rested.loc[0, "home_rest"])     # first game, no prior
    assert abs(rested.loc[1, "home_rest"] - 7.0) < 1e-6


def test_elo_uses_per_game_hga_column():
    games = _games()
    flat = EloRatings().fit(games)
    perg = EloRatings().fit(attach_hga(games))
    # the hga_points column changes the ratings path
    assert not np.allclose(flat["home_elo_pre"], perg["home_elo_pre"])

import numpy as np
import pandas as pd

from afl_bot.models.pace import (
    PACE_STATS,
    league_stat_totals,
    team_stat_total_profiles,
)


def _log_row(year, rnd, ut, player, team, opp, is_home, disp, team_disp):
    return {
        "year": year, "round": rnd, "unixtime": ut, "player": player,
        "team": team, "opponent": opp, "is_home": is_home,
        "disposals": disp, "marks": 5, "tackles": 4,
        "team_disposals": team_disp, "team_marks": 80, "team_tackles": 60,
    }


# Team A totals 380 (r1) then 400 (r2); Team B totals 340 then 360.
LOG = pd.DataFrame([
    _log_row(2024, 1, 1, "a1", "A", "B", True, 200, 380),
    _log_row(2024, 1, 1, "a2", "A", "B", True, 180, 380),
    _log_row(2024, 1, 1, "b1", "B", "A", False, 170, 340),
    _log_row(2024, 1, 1, "b2", "B", "A", False, 170, 340),
    _log_row(2024, 2, 2, "a1", "A", "B", False, 210, 400),
    _log_row(2024, 2, 2, "a2", "A", "B", False, 190, 400),
    _log_row(2024, 2, 2, "b1", "B", "A", True, 180, 360),
    _log_row(2024, 2, 2, "b2", "B", "A", True, 180, 360),
])


def test_team_stat_total_profiles_per_team_ewma():
    prof = team_stat_total_profiles(LOG, stats=["disposals"])
    assert set(prof) == {"A", "B"}
    # EWMA of [380, 400] sits between the two, weighted toward the latest.
    assert 380 < prof["A"]["disposals"] <= 400
    assert 340 < prof["B"]["disposals"] <= 360
    assert prof["A"]["disposals"] > prof["B"]["disposals"]


def test_team_stat_total_profiles_anti_leakage_cutoff():
    # As of round 2, only round 1 is visible -> single value, EWMA == 380.
    prof = team_stat_total_profiles(LOG, stats=["disposals"], as_of_year=2024, as_of_round=2)
    assert prof["A"]["disposals"] == 380.0


def test_team_stat_total_profiles_defaults_to_pace_stats():
    prof = team_stat_total_profiles(LOG)
    assert set(prof["A"]) == set(PACE_STATS)


def test_team_stat_total_profiles_derives_team_total_if_absent():
    raw = LOG.drop(columns=["team_disposals"])
    derived = team_stat_total_profiles(raw, stats=["disposals"])
    # team_disposals reconstructed (200+180=380, 210+190=400) matches the
    # version computed from the explicit column.
    explicit = team_stat_total_profiles(LOG, stats=["disposals"])
    assert np.isclose(derived["A"]["disposals"], explicit["A"]["disposals"])


def test_league_stat_totals_average():
    league = league_stat_totals(LOG, stats=["disposals"])
    # mean of team-game totals: (380+340+400+360)/4 = 370
    assert league["disposals"] == 370.0


def test_empty_inputs():
    assert team_stat_total_profiles(pd.DataFrame()) == {}
    assert np.isnan(league_stat_totals(pd.DataFrame(), stats=["disposals"])["disposals"])

import numpy as np
import pandas as pd
import pytest

from afl_bot.config import PLAYER_FORM_WINDOW, PROP_MIN_DISPERSION
from afl_bot.models.priors import (
    _player_means_and_shares,
    cba_role_multiplier,
    classify_roles,
    estimate_dispersion_hierarchical,
    player_cba,
    player_tog,
    role_rate_priors,
    shrink,
    tog_multiplier,
)
from afl_bot.models.props import estimate_dispersion, player_rate_profile

rng = np.random.default_rng(0)


def _player_rows(player, team, role_stats, n_games=12, start_ut=0):
    """role_stats: dict of base per-game values for disposals/goals/marks/tackles/
    hitouts/tog/cba. Adds mild noise so var > mean (dispersion is estimable)."""
    rows = []
    for g in range(n_games):
        rows.append({
            "year": 2024, "round": g + 1, "unixtime": start_ut + g,
            "player": player, "team": team, "opponent": "X", "is_home": True,
            "disposals": max(int(rng.poisson(role_stats["disposals"])), 0),
            "goals": max(int(rng.poisson(role_stats["goals"])), 0),
            "marks": max(int(rng.poisson(role_stats["marks"])), 0),
            "tackles": max(int(rng.poisson(role_stats["tackles"])), 0),
            "hitouts": max(int(rng.poisson(role_stats["hitouts"])), 0),
            "time_on_ground_percentage": role_stats["tog"],
            "centre_bounce_attendances": role_stats["cba"],
        })
    return rows


LOG = pd.DataFrame(
    _player_rows("Ruck", "A", {"disposals": 12, "goals": 0, "marks": 3, "tackles": 2, "hitouts": 30, "tog": 80, "cba": 8}, start_ut=0)
    + _player_rows("Fwd", "A", {"disposals": 10, "goals": 2, "marks": 5, "tackles": 1, "hitouts": 0, "tog": 75, "cba": 1}, start_ut=100)
    + _player_rows("Mid", "A", {"disposals": 28, "goals": 0, "marks": 4, "tackles": 6, "hitouts": 0, "tog": 88, "cba": 22}, start_ut=200)
    + _player_rows("Def", "A", {"disposals": 15, "goals": 0, "marks": 6, "tackles": 3, "hitouts": 0, "tog": 82, "cba": 0}, start_ut=300)
)
# team_<stat> totals so shares are well-defined.
for stat in ("disposals", "goals", "marks", "tackles"):
    LOG[f"team_{stat}"] = LOG.groupby(["year", "round", "team"])[stat].transform("sum")


# --------------------------------------------------------------------------- #
# §3.1 shrinkage
# --------------------------------------------------------------------------- #
def test_shrink_debutant_near_prior_veteran_near_own():
    prior = 0.065
    debutant = shrink(0.12, 2, prior, strength=8)
    veteran = shrink(0.12, 60, prior, strength=8)
    assert abs(debutant - prior) < abs(debutant - 0.12)   # debutant pulled to prior
    assert abs(veteran - 0.12) < abs(veteran - prior)     # veteran stays near own
    assert prior < debutant < veteran < 0.12


def test_shrink_handles_missing_values():
    assert shrink(float("nan"), 5, 0.065) == 0.065   # no own data -> prior
    assert shrink(0.12, 5, float("nan")) == 0.12      # no prior -> own


def test_classify_roles():
    roles = classify_roles(LOG)
    assert roles["Ruck"] == "ruck"
    assert roles["Fwd"] == "forward"
    assert roles["Mid"] == "midfielder"
    assert roles["Def"] == "general"


def test_classify_roles_without_hitouts_column_has_no_rucks():
    roles = classify_roles(LOG.drop(columns=["hitouts"]))
    assert "ruck" not in set(roles.values())


def test_classify_roles_prefers_real_position_labels():
    # a low-disposal player box-score-inferred as "general" but positioned at
    # centre -> midfielder via the real label (round-2 §5.3)
    rows = []
    for r in range(1, 6):
        rows.append({"year": 2024, "round": r, "unixtime": r, "player": "Tagger",
                     "team": "A", "opponent": "B", "is_home": True, "position": "C",
                     "disposals": 12, "goals": 0, "marks": 2, "tackles": 8, "hitouts": 0})
        rows.append({"year": 2024, "round": r, "unixtime": r, "player": "Sub",
                     "team": "A", "opponent": "B", "is_home": True, "position": "INT",
                     "disposals": 10, "goals": 0, "marks": 2, "tackles": 2, "hitouts": 0})
    df = pd.DataFrame(rows)
    roles = classify_roles(df)
    assert roles["Tagger"] == "midfielder"      # real "C" label beats box-score "general"
    assert roles["Sub"] == "general"            # INT carries no signal -> box-score fallback


def test_opponent_matchup_multiplier_era_matched_baseline():
    from afl_bot.models.props import opponent_matchup_multiplier
    rows = []
    ut = 0
    # old era (2018-2020): league disposals ~50/team-game
    for year in (2018, 2019, 2020):
        for rnd in range(1, 6):
            for team, opp in (("A", "OPP"), ("OPP", "A")):
                ut += 1
                rows.append({"year": year, "round": rnd, "unixtime": ut, "player": "p",
                             "team": team, "opponent": opp, "disposals": 50})
    # recent era (2024-2026): league ~100/team-game; OPP concedes ~100
    for year in (2024, 2025, 2026):
        for rnd in range(1, 6):
            for team, opp in (("A", "OPP"), ("OPP", "A")):
                ut += 1
                rows.append({"year": year, "round": rnd, "unixtime": ut, "player": "p",
                             "team": team, "opponent": opp, "disposals": 100})
    log = pd.DataFrame(rows)
    mult = opponent_matchup_multiplier(log, "disposals", "OPP", recent_seasons=3)
    # era-matched baseline (recent ~100) -> ~1.0; an all-history baseline (~75)
    # would inflate this to ~1.33
    assert 0.9 < mult < 1.1


def test_role_rate_priors_orders_midfield_above_forward():
    roles = classify_roles(LOG)
    priors = role_rate_priors(LOG, "disposals", roles)
    assert "_global" in priors
    assert priors["midfielder"]["share_prior"] > priors["forward"]["share_prior"]
    assert priors["midfielder"]["mean_prior"] > priors["general"]["mean_prior"]


def test_role_rate_priors_empty_log():
    priors = role_rate_priors(pd.DataFrame(), "disposals", {})
    assert np.isnan(priors["_global"]["mean_prior"])


def test_estimate_dispersion_hierarchical_floors_and_covers_all_players():
    roles = classify_roles(LOG)
    disp = estimate_dispersion_hierarchical(LOG, "disposals", roles)
    assert set(disp) == {"Ruck", "Fwd", "Mid", "Def"}
    assert all(r >= PROP_MIN_DISPERSION for r in disp.values())


def test_estimate_dispersion_low_game_player_takes_role_prior():
    roles = classify_roles(LOG)
    # min_games huge -> nobody has a usable own estimate -> all take role/league prior
    disp = estimate_dispersion_hierarchical(LOG, "disposals", roles, min_games=999)
    assert all(r >= PROP_MIN_DISPERSION for r in disp.values())


# --------------------------------------------------------------------------- #
# §3.2 role & minutes adjustments
# --------------------------------------------------------------------------- #
def test_tog_multiplier_scales_and_clips():
    assert tog_multiplier(88.0, 80.0) == 1.10           # within bounds
    assert tog_multiplier(50.0, 100.0) == 0.70          # clipped to lower bound
    assert tog_multiplier(200.0, 80.0) == 1.15          # clipped to upper bound


def test_tog_multiplier_neutral_when_missing():
    assert tog_multiplier(float("nan"), 80.0) == 1.0
    assert tog_multiplier(80.0, 0.0) == 1.0


def test_cba_role_multiplier_jump_and_clip():
    assert cba_role_multiplier(20.0, 5.0) == 1.15        # +15 CBA -> +15%
    assert cba_role_multiplier(60.0, 5.0) == 1.25        # clipped to upper bound
    assert cba_role_multiplier(float("nan"), 5.0) == 1.0


def test_player_tog_and_cba_recent_baseline():
    recent, baseline = player_tog(LOG, "Mid")
    assert np.isfinite(recent) and np.isfinite(baseline)
    recent_cba, baseline_cba = player_cba(LOG, "Mid")
    assert recent_cba > 0


def test_player_tog_missing_column_returns_nan():
    recent, baseline = player_tog(LOG.drop(columns=["time_on_ground_percentage"]), "Mid")
    assert np.isnan(recent) and np.isnan(baseline)


# --------------------------------------------------------------------------- #
# Form-window tests (FORM-WINDOW-INSTRUCTIONS Parts B, D, E)
# --------------------------------------------------------------------------- #

def _level_shift_log(old_val, new_val, n_old=20, n_new=40):
    """Player log with a level shift: first n_old games at old_val, next n_new at new_val."""
    rows = []
    for g in range(n_old + n_new):
        val = old_val if g < n_old else new_val
        rows.append({
            "year": 2024, "round": g + 1, "unixtime": g + 1,
            "player": "P", "team": "A", "opponent": "B", "is_home": True,
            "disposals": val, "goals": 0, "marks": 0, "tackles": 0,
        })
    return pd.DataFrame(rows)


def test_player_rate_profile_uses_form_window():
    """Last-40 projection reflects the new level, not the blended all-history mean (B1)."""
    log = _level_shift_log(old_val=5, new_val=20, n_old=20, n_new=40)
    profile = player_rate_profile(log, "P", "disposals")
    assert profile["n_games"] == PLAYER_FORM_WINDOW  # windowed to 40
    assert profile["mean"] > 15  # well above the old-era 5; EWMA of 40 x 20 = 20


def test_player_rate_profile_thin_history_not_dropped():
    """A player with only 5 games still gets a finite, non-NaN projection (Part D)."""
    rows = [{"year": 2024, "round": r, "unixtime": r, "player": "Debut",
             "team": "A", "opponent": "B", "is_home": True,
             "disposals": 15, "goals": 0, "marks": 0, "tackles": 0}
            for r in range(1, 6)]
    log = pd.DataFrame(rows)
    profile = player_rate_profile(log, "Debut", "disposals")
    assert profile["n_games"] == 5
    assert np.isfinite(profile["mean"])


def test_player_rate_profile_sort_robustness_bad_unixtime():
    """A corrupted unixtime on a recent game doesn't mis-order the window (Part E).
    year/round are primary sort keys; unixtime is just a tiebreak."""
    rows = []
    for g in range(45):
        ut = 1 if g == 40 else g + 1000   # game 41 has a near-1970 unixtime
        rows.append({"year": 2024, "round": g + 1, "unixtime": ut,
                     "player": "P", "team": "A", "opponent": "B", "is_home": True,
                     "disposals": 20 if g >= 5 else 5})
    log = pd.DataFrame(rows)
    profile = player_rate_profile(log, "P", "disposals")
    # Last 40 by year/round = games 5-44 → all 20. Bad unixtime for g=40
    # doesn't change ordering (round 41 still sorts after round 40).
    assert profile["mean"] > 15
    assert profile["n_games"] == PLAYER_FORM_WINDOW


def test_estimate_dispersion_uses_form_window():
    """estimate_dispersion windows each player to their last 40 games (B2).
    Old era (games 0-19): extreme alternation 1/39 → high variance ≈ 361.
    Recent era (games 20-59): moderate alternation 15/25 → variance ≈ 25.
    Both eras have var > mean, so NB method-of-moments is valid for each window.
    Recent era is LESS overdispersed → higher r; all-history is pulled down by old era."""
    rows = []
    for g in range(60):
        val = (1 if g % 2 == 0 else 39) if g < 20 else (15 if g % 2 == 0 else 25)
        rows.append({"year": 2024, "round": g + 1, "unixtime": g + 1,
                     "player": "P", "team": "A", "opponent": "B", "is_home": True,
                     "disposals": val})
    log = pd.DataFrame(rows)
    disp_windowed = estimate_dispersion(log, "disposals")          # window=40: recent era
    disp_all = estimate_dispersion(log, "disposals", window=60)   # all 60 games
    # Windowed to recent era (var≈25): r ≈ 71; all-history (var≈139): r ≈ 3.4
    assert disp_windowed["P"] > disp_all["P"]


def test_player_means_and_shares_uses_form_window():
    """_player_means_and_shares ignores games outside the 40-game window (B3)."""
    rows = []
    for g in range(50):
        rows.append({
            "year": 2024, "round": g + 1, "unixtime": g + 1,
            "player": "P", "team": "A", "opponent": "B", "is_home": True,
            "disposals": 5 if g < 10 else 20,  # last 40 all 20
        })
    log = pd.DataFrame(rows)
    # team_disposals column so shares are defined
    log["team_disposals"] = log["disposals"]
    means, _ = _player_means_and_shares(log, "disposals")
    # Windowed (last 40) mean should be 20; all-history mean would be ~17
    assert means["P"] == pytest.approx(20.0, abs=0.1)


def test_estimate_dispersion_hierarchical_uses_form_window():
    """estimate_dispersion_hierarchical windows each player to last 40 games (B4).
    Same alternating pattern as the estimate_dispersion test above."""
    rows = []
    for g in range(60):
        val = (1 if g % 2 == 0 else 39) if g < 20 else (15 if g % 2 == 0 else 25)
        rows.append({"year": 2024, "round": g + 1, "unixtime": g + 1,
                     "player": "P", "team": "A", "opponent": "B", "is_home": True,
                     "disposals": val})
    log = pd.DataFrame(rows)
    roles = {"P": "midfielder"}
    disp_windowed = estimate_dispersion_hierarchical(log, "disposals", roles)
    disp_all = estimate_dispersion_hierarchical(log, "disposals", roles, window=60)
    # Recent era (var≈25) → higher r than all-history (var≈139)
    assert disp_windowed["P"] > disp_all["P"]


def test_player_tog_baseline_uses_form_window():
    """player_tog baseline EWMA is computed over the last PLAYER_FORM_WINDOW games (B5)."""
    rows = []
    for g in range(50):
        tog = 50.0 if g < 10 else 90.0   # last 40 games: TOG = 90
        rows.append({"year": 2024, "round": g + 1, "unixtime": g + 1,
                     "player": "P", "team": "A", "opponent": "B", "is_home": True,
                     "disposals": 15, "time_on_ground_percentage": tog})
    log = pd.DataFrame(rows)
    _, baseline = player_tog(log, "P")
    # Windowed to last 40 (all 90 TOG) → baseline near 90
    assert baseline > 80


def test_shrunk_projection_finite_for_thin_history():
    """A 5-game player gets a finite, shrunk projection — not NaN (Part D)."""
    from afl_bot.models.priors import role_rate_priors, shrink
    thin_rows = [{"year": 2024, "round": r, "unixtime": r, "player": "New",
                  "team": "A", "opponent": "B", "is_home": True, "disposals": 20,
                  "goals": 1, "marks": 3, "tackles": 2}
                 for r in range(1, 6)]
    thin_log = pd.DataFrame(thin_rows)
    for stat in ("disposals", "goals", "marks", "tackles"):
        thin_log[f"team_{stat}"] = thin_log[stat]
    priors_dict = role_rate_priors(thin_log, "disposals", {"New": "midfielder"})
    profile = player_rate_profile(thin_log, "New", "disposals")
    prior_mean = priors_dict.get("midfielder", priors_dict["_global"])["mean_prior"]
    result = shrink(profile["mean"], profile["n_games"], prior_mean)
    assert np.isfinite(result) and result > 0

import numpy as np
import pandas as pd

from afl_bot.config import PROP_MIN_DISPERSION
from afl_bot.models.priors import (
    cba_role_multiplier,
    classify_roles,
    estimate_dispersion_hierarchical,
    player_cba,
    player_tog,
    role_rate_priors,
    shrink,
    tog_multiplier,
)

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

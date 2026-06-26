"""Phase 2 tests — promo-aware Total EV on the ladder + multi-outcome Kelly."""

import numpy as np
import pytest

from afl_bot.build.multi import LegCandidate
from afl_bot.build.report import search_market_sgms, search_match_sgms
from afl_bot.build.staking import fractional_kelly_fraction, multi_outcome_kelly
from afl_bot.config import BONUS_BET_FACTOR, KELLY_FRACTION, KELLY_PER_BET_CAP, PROMO_MIN_LEGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leg(name, prob, mask, subject, odds=None):
    return LegCandidate(
        name=name, match_id="m1", market="player_disposals", subject=subject,
        fair_prob=prob, market_odds=odds if odds is not None else 1 / prob, mask=mask,
    )


def _correlated_legs(n=50_000, seed=42):
    """Three legs from a single shared mask so all pairwise correlations are 1.
    p_all_win = p_A = p_B = p_C (all fire together).
    p_one_loss is exactly 0 — no 2-leg subset fires without the third."""
    rng = np.random.default_rng(seed)
    joint_mask = rng.random(n) < 0.55
    p = joint_mask.mean()
    legs = [_leg(f"{name} 15+", p, joint_mask, name) for name in ("A", "B", "C")]
    return legs, float(joint_mask.mean())


def _independent_legs(n=50_000, probs=(0.75, 0.70, 0.65), seed=7):
    rng = np.random.default_rng(seed)
    legs = []
    masks = []
    for name, p in zip(("A", "B", "C"), probs):
        m = rng.random(n) < p
        legs.append(_leg(f"{name} 15+", float(m.mean()), m, name))
        masks.append(m)
    return legs, masks


# ---------------------------------------------------------------------------
# STEP 1: Promo branch probabilities counted from masks
# ---------------------------------------------------------------------------

def test_promo_p_all_win_matches_joint_from_masks_for_perfect_corr():
    """Perfectly correlated legs: p_all_win == raw joint prob (every sim either
    all-win or all-lose); p_one_loss == 0 by construction."""
    legs, p_joint = _correlated_legs()
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    assert r["p_all_win"] is not None
    assert abs(r["p_all_win"] - p_joint) < 0.01
    assert abs(r["p_one_loss"]) < 0.01        # perfect corr -> no 2-win-1-loss path


def test_promo_branch_probs_sum_to_one():
    legs, _ = _independent_legs()
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    total = r["p_all_win"] + r["p_one_loss"] + r["p_two_plus_loss"]
    assert abs(total - 1.0) < 1e-9


def test_promo_p_one_loss_differs_from_independence_formula_for_correlated_legs():
    """For perfectly correlated legs the independence formula gives
    p_one_loss = 3 * p * (1-p)^2 + ... (non-zero), but the sim mask gives 0."""
    legs, p = _correlated_legs(n=50_000)
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    # Independence formula for 3 equal-prob legs (approx, for p ~= 0.55):
    indep_p_one_loss = 3 * p * p * (1 - p)
    # Correlated (sim mask): p_one_loss ~ 0
    assert r["p_one_loss"] < indep_p_one_loss / 2


def test_promo_stats_on_market_sgms_rung():
    legs, _ = _independent_legs()
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    assert out
    r = out[0]
    assert r["p_all_win"] is not None
    assert r["p_one_loss"] is not None
    total = r["p_all_win"] + r["p_one_loss"] + r["p_two_plus_loss"]
    assert abs(total - 1.0) < 1e-9


def test_promo_stats_none_when_masks_unavailable():
    """Legs with mask=None: promo stats should be None (not a crash)."""
    legs = [_leg(f"{name} 15+", 0.65, None, name) for name in ("A", "B", "C")]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    assert r["p_all_win"] is None
    assert r["p_one_loss"] is None
    assert r["promo_ev"] is None


def test_promo_ev_equals_p_one_loss_times_R():
    legs, _ = _independent_legs()
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    if r["p_one_loss"] is not None:
        assert abs(r["promo_ev"] - r["p_one_loss"] * BONUS_BET_FACTOR) < 1e-9


def test_total_ev_equals_base_plus_promo():
    legs, _ = _independent_legs()
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    if r.get("edge") is not None and r.get("promo_ev") is not None:
        assert abs(r["total_ev"] - (r["edge"] + r["promo_ev"])) < 1e-9


def test_total_ev_equals_edge_when_no_masks():
    """Without masks, total_ev falls back to base edge (no promo term)."""
    legs = [_leg(f"{name} 15+", 0.65, None, name, odds=0.65 * 1.05 / 0.65) for name in ("A", "B", "C")]
    # Re-build: model_odds above fair
    rng = np.random.default_rng(0)
    n = 1000
    legs2 = []
    for name in ("A", "B", "C"):
        p = 0.65
        odds = 1.0 / p * 1.02
        legs2.append(_leg(f"{name} 15+", p, None, name, odds=odds))
    odds_book = {l.name: l.market_odds for l in legs2}
    out = search_match_sgms(legs2, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    assert r["promo_ev"] is None
    # total_ev should equal the base edge (or None when no book odds)
    if r.get("edge") is not None:
        assert r["total_ev"] == pytest.approx(r["edge"])


def test_promo_min_legs_respected():
    """A 2-leg combo (below PROMO_MIN_LEGS=3) must have no promo stats."""
    legs, _ = _independent_legs()
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, min_legs=2, max_legs=2,
                            target_odds=(1.5,), min_joint_prob=0.0)
    if out and out[0]["p_all_win"] is not None:
        # only valid if PROMO_MIN_LEGS > 2; check it's enforced
        assert PROMO_MIN_LEGS <= 2 or out[0]["p_all_win"] is None


def test_promo_min_legs_default_is_3():
    assert PROMO_MIN_LEGS == 3


# ---------------------------------------------------------------------------
# STEP 2: multi_outcome_kelly
# ---------------------------------------------------------------------------

def test_multi_outcome_kelly_zero_when_ev_negative():
    # Binary Kelly also returns 0 on -EV bets; multi-outcome Kelly must agree.
    p_win, p_one, p_dead = 0.10, 0.20, 0.70
    odds = 3.0
    assert multi_outcome_kelly(p_win, p_one, p_dead, odds, BONUS_BET_FACTOR) == 0.0


def test_multi_outcome_kelly_positive_when_total_ev_positive():
    """A bet that's slightly -EV on a standalone basis but has positive total EV
    via the promo refund must return a positive (capped) stake."""
    # Set up: base edge mildly negative, promo pushes total EV positive.
    # p_win=0.20, p_one=0.40, p_dead=0.40, odds=4.0, R=0.75
    # base EV = 0.20*(4-1) + 0.40*(0.75-1) - 0.40 = 0.60 - 0.10 - 0.40 = +0.10 > 0
    p_win, p_one, p_dead = 0.20, 0.40, 0.40
    odds = 4.0
    f = multi_outcome_kelly(p_win, p_one, p_dead, odds, BONUS_BET_FACTOR)
    assert f > 0.0


def test_multi_outcome_kelly_capped_at_kelly_per_bet_cap():
    # Extreme edge — full Kelly would be very large; must still cap.
    p_win, p_one, p_dead = 0.50, 0.30, 0.20
    odds = 5.0
    f = multi_outcome_kelly(p_win, p_one, p_dead, odds, BONUS_BET_FACTOR)
    assert f <= KELLY_PER_BET_CAP


def test_multi_outcome_kelly_monotone_in_R():
    """Higher refund factor R -> more value from one-loss branch -> larger stake."""
    p_win, p_one, p_dead = 0.25, 0.35, 0.40
    odds = 5.0
    f_low = multi_outcome_kelly(p_win, p_one, p_dead, odds, refund_factor=0.50)
    f_high = multi_outcome_kelly(p_win, p_one, p_dead, odds, refund_factor=0.90)
    assert f_high >= f_low


def test_multi_outcome_kelly_greater_than_binary_kelly_when_promo_adds_value():
    """multi_outcome_kelly >= fractional_kelly when a meaningful promo refund
    exists, because the extra one-loss branch improves the log-growth."""
    p_win, p_one, p_dead = 0.30, 0.40, 0.30
    odds = 4.5
    R = BONUS_BET_FACTOR
    f_multi = multi_outcome_kelly(p_win, p_one, p_dead, odds, R)
    # Binary Kelly treats all non-win outcomes as total loss:
    f_binary = fractional_kelly_fraction(p_win, odds)
    assert f_multi >= f_binary


def test_multi_outcome_kelly_zero_when_truly_neg_ev():
    # No edge even with promo.
    p_win, p_one, p_dead = 0.05, 0.10, 0.85
    odds = 6.0   # fair would need p=1/6~0.167; we have 0.05 — deeply -EV
    f = multi_outcome_kelly(p_win, p_one, p_dead, odds, BONUS_BET_FACTOR)
    assert f == 0.0


def test_multi_outcome_kelly_fraction_and_cap_applied():
    p_win, p_one, p_dead = 0.40, 0.35, 0.25
    odds = 4.0
    # With fraction=1.0 and cap=1.0 we get the uncapped full Kelly.
    f_full = multi_outcome_kelly(p_win, p_one, p_dead, odds, BONUS_BET_FACTOR,
                                 fraction=1.0, cap=1.0)
    # With fraction=KELLY_FRACTION and cap=KELLY_PER_BET_CAP we get the standard fractional.
    f_frac = multi_outcome_kelly(p_win, p_one, p_dead, odds, BONUS_BET_FACTOR)
    assert f_frac == pytest.approx(min(KELLY_FRACTION * f_full, KELLY_PER_BET_CAP), abs=1e-6)


# ---------------------------------------------------------------------------
# STEP 2: suggested_stake on ladder rungs
# ---------------------------------------------------------------------------

def test_suggested_stake_is_positive_on_value_rung():
    legs, _ = _independent_legs(probs=(0.75, 0.70, 0.65))
    # Make book odds slightly above fair for positive edge
    odds_book = {leg.name: leg.market_odds * 1.04 for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    r = out[0]
    if r.get("total_ev") is not None and r["total_ev"] > 0 and r.get("p_all_win") is not None:
        assert r["suggested_stake"] is not None
        assert r["suggested_stake"] > 0.0
        assert r["suggested_stake"] <= KELLY_PER_BET_CAP


def test_suggested_stake_none_when_no_book_odds():
    legs, _ = _independent_legs()
    out = search_match_sgms(legs, target_odds=(1.5,), min_joint_prob=0.0)  # no odds_book
    for r in out:
        assert r["suggested_stake"] is None


def test_suggested_stake_on_market_sgms():
    legs, _ = _independent_legs(probs=(0.75, 0.70, 0.65))
    odds_book = {leg.name: leg.market_odds * 1.04 for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(1.5,), min_joint_prob=0.0)
    assert out
    r = out[0]
    assert "suggested_stake" in r
    if r.get("total_ev") is not None and r["total_ev"] > 0:
        assert r["suggested_stake"] is not None
        assert r["suggested_stake"] <= KELLY_PER_BET_CAP


# ---------------------------------------------------------------------------
# STEP 1 end-to-end: value pick ranks by total EV
# ---------------------------------------------------------------------------

def test_value_pick_present_on_top_band_with_odds():
    """With book odds and positive edge, the top band should be tagged VALUE PICK."""
    rng = np.random.default_rng(1)
    n = 40_000
    probs = {"A": 0.90, "B": 0.85, "C": 0.78, "D": 0.68, "E": 0.55, "F": 0.42}
    legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        prob = float(mask.mean())
        legs.append(_leg(f"{name} 15+", prob, mask, name, odds=(1.0 / prob) * 1.05))
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book)
    picks = [r for r in out if r.get("value_pick")]
    assert len(picks) == 1
    vp = picks[0]
    assert vp.get("total_ev") is not None
    assert vp["total_ev"] > 0.0
    # total_ev >= base edge (promo only adds)
    if vp.get("edge") is not None and vp.get("promo_ev") is not None:
        assert vp["total_ev"] >= vp["edge"]


def test_promo_stats_present_on_all_rungs_in_full_ladder():
    rng = np.random.default_rng(77)
    n = 40_000
    probs9 = {"A": 0.77, "B": 0.70, "C": 0.62, "D": 0.55, "E": 0.47,
              "F": 0.41, "G": 0.36, "H": 0.33, "I": 0.30}
    legs = []
    for name, p in probs9.items():
        mask = rng.random(n) < p
        legs.append(_leg(f"{name} 20+ disp", float(mask.mean()), mask, name,
                         odds=(1.0 / float(mask.mean())) * 1.03))
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book)
    for r in out:
        assert r["p_all_win"] is not None, f"promo stats missing for rung {r['legs']}"
        assert r["p_one_loss"] is not None
        assert r["total_ev"] is not None
        # total_ev >= edge (promo adds non-negative value)
        if r.get("edge") is not None:
            assert r["total_ev"] >= r["edge"] - 1e-9

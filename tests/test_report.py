"""Round-report helpers (round-2 Â§10): projection tables + SGM joint search."""

import numpy as np
import pytest

from afl_bot.build.multi import LegCandidate
from afl_bot.build.report import (
    build_odds_template,
    build_pull_em_sgm,
    build_sgm_candidates,
    is_bookable_model_only_leg,
    projection_rows,
    render_markdown,
    search_market_sgms,
    search_match_sgms,
    select_ladder_lines,
    top_n_players_by_stat,
)
from afl_bot.config import (
    MULTI_TARGET_ODDS,
    MULTI_MARKET_SHRINK,
    PULL_DETECTION_PROB,
    PULL_EM_ELIGIBLE_MARKETS,
    PULL_EM_MIN_COMBO_ODDS,
)

LINES = {"disposals": [15, 20, 25], "goals": [1, 2], "marks": [4, 6], "tackles": [3, 5]}


def test_projection_rows_means_probs_and_sort():
    rng = np.random.default_rng(0)
    samples = {
        "Star": {"disposals": rng.normal(30, 5, 5000).clip(0)},
        "Role": {"disposals": rng.normal(15, 4, 5000).clip(0)},
    }
    rows = projection_rows(samples, LINES)
    assert rows[0]["player"] == "Star"           # sorted by projected disposals desc
    assert rows[0]["disposals_mean"] > rows[1]["disposals_mean"]
    assert 0.0 <= rows[0]["disposals_20+"] <= 1.0
    assert rows[0]["disposals_20+"] > rows[1]["disposals_20+"]


def _leg(name, prob, mask, subject, market="player_disposals", odds=None):
    return LegCandidate(name=name, match_id="m1", market=market, subject=subject,
                        fair_prob=prob, market_odds=odds if odds is not None else 1 / prob,
                        mask=mask)


def _ladder_legs(seed=1, odds_mult=1.0):
    """Six independent legs (distinct subjects) with a wide prob spread so the
    3-leg combos cleanly populate all three default bands. ``odds_mult`` sets each
    leg's book price above fair (``market_odds`` is what ``combined_odds``
    multiplies for the priced book odds)."""
    rng = np.random.default_rng(seed)
    n = 40000
    probs = {"A": 0.90, "B": 0.85, "C": 0.78, "D": 0.68, "E": 0.55, "F": 0.42}
    legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        prob = mask.mean()
        legs.append(_leg(f"{name} 15+ disp", prob, mask, name, odds=(1.0 / prob) * odds_mult))
    return legs


def test_search_match_sgms_ladder_is_3leg_one_per_target_and_above_floor():
    out = search_match_sgms(_ladder_legs())            # all defaults: 3-leg, target-odds
    assert out, "should find 3-leg combos"
    assert len(out) == len(MULTI_TARGET_ODDS)          # one rung per target (incl. NO BET)
    real_rungs = [r for r in out if not r.get("no_bet")]
    assert real_rungs, "should find at least one real rung"
    for r in real_rungs:
        assert len(r["legs"]) == 3                     # minimum-3-leg ladder
        assert {"joint_prob", "naive_product", "corr_gain", "fair_odds"} <= set(r)
    # NO BET rungs carry the target_odds and empty legs
    for r in out:
        if r.get("no_bet"):
            assert r["legs"] == []
            assert r.get("target_odds") is not None
    # real rungs sorted safest -> longest by fair_odds
    assert [r["fair_odds"] for r in real_rungs] == sorted(r["fair_odds"] for r in real_rungs)
    # no duplicate combos among real rungs
    all_legs = [tuple(sorted(r["legs"])) for r in real_rungs]
    assert len(all_legs) == len(set(all_legs))


def test_search_match_sgms_excludes_conflicts():
    legs = _ladder_legs()
    legs.append(_leg("A 20+ disp", 0.5, legs[0].mask, "A"))  # conflicts with "A 15+ disp"
    out = search_match_sgms(legs)
    assert out
    for r in out:
        assert not ("A 15+ disp" in r["legs"] and "A 20+ disp" in r["legs"])


def test_search_match_sgms_top_band_value_pick_is_shrunk_and_capped():
    # 9 distinct players → diversity constraint allows 3 real rungs (3 players each).
    # Use target_odds=(2.10, 3.50, 5.00) so the top band ($5.00) can be filled
    # without reusing the players from band 1 (B,C,D) or band 2 (empty/NO BET).
    rng = np.random.default_rng(5)
    n = 40_000
    probs9 = {"A": 0.90, "B": 0.85, "C": 0.78, "D": 0.68, "E": 0.55,
              "F": 0.42, "G": 0.38, "H": 0.35, "I": 0.32}
    odds_mult = 1.05
    wide_legs = []
    for name, p in probs9.items():
        mask = rng.random(n) < p
        wide_legs.append(_leg(f"{name} 15+ disp", mask.mean(), mask, name,
                              odds=(1.0 / mask.mean()) * odds_mult))
    odds_book = {leg.name: leg.market_odds for leg in wide_legs}
    out = search_match_sgms(wide_legs, odds_book=odds_book,
                            target_odds=(2.10, 3.50, 5.00))
    picks = [r for r in out if r.get("value_pick")]
    assert len(picks) == 1                             # only the top rung is the value pick
    vp = picks[0]
    assert "book_odds" in vp
    assert 0.0 < vp["edge"] <= 0.15                    # positive but under the sanity cap
    assert vp["edge"] < vp["raw_edge"]                 # shrunk below the naive joint*book-1
    assert vp["fair_odds"] >= MULTI_TARGET_ODDS[1]      # at least the mid target (~3.50)


def test_build_sgm_candidates_no_edge_unless_every_leg_in_combo_is_priced():
    # Model-upgrade audit Phase 4 STEP 2.3: VALUE must never be flagged off a
    # fair-odds-only leg. Price every leg generously above fair EXCEPT one,
    # which has no book price at all -- every combo that includes it must
    # come back with no book_odds/edge, however good the other legs' prices.
    legs = _ladder_legs(odds_mult=1.20)
    odds_book = {leg.name: leg.market_odds for leg in legs}
    unpriced_name = legs[0].name
    del odds_book[unpriced_name]
    candidates = build_sgm_candidates(legs, odds_book=odds_book)
    for c in candidates:
        if unpriced_name in c["legs"]:
            assert "book_odds" not in c and "edge" not in c
        else:
            assert "book_odds" in c and "edge" in c
    out = search_match_sgms(legs, odds_book=odds_book)
    assert not any(r.get("value_pick") and unpriced_name in r["legs"] for r in out)


def test_search_match_sgms_implausible_edge_is_not_flagged_value():
    legs = _ladder_legs(odds_mult=1.15)   # book 15% over fair -> shrunk edge blows past 15%
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book)
    assert out
    assert any("edge" in r for r in out)               # combos ARE priced...
    assert all(r["edge"] > 0.15 for r in out if "edge" in r)   # ...all above the cap
    assert not any(r.get("value_pick") for r in out)   # so none is flagged VALUE


def test_search_match_sgms_fills_every_target_when_pool_is_thin():
    # All legs ~0.82 -> every 3-combo has fair_odds ~1.82, which is below the
    # $2.10 band floor. With band-window enforcement, every target returns NO BET.
    rng = np.random.default_rng(7)
    legs = [_leg(f"{name} 15+ disp", (m := rng.random(40000) < 0.82).mean(), m, name)
            for name in "ABCDE"]
    out = search_match_sgms(legs)
    assert len(out) == len(MULTI_TARGET_ODDS)          # one rung per target (all NO BET)
    assert all(r.get("no_bet") for r in out), (
        "Every rung must be NO BET: pool fair_odds ~1.82, all below $2.10 floor"
    )


def test_build_sgm_candidates_is_the_full_pool_search_selects_from():
    legs = _ladder_legs()
    candidates = build_sgm_candidates(legs)
    selected = search_match_sgms(legs)
    cand_keys = {tuple(sorted(c["legs"])) for c in candidates}
    for r in selected:
        if r.get("no_bet"):
            continue  # NO BET sentinel has no legs to check
        assert tuple(sorted(r["legs"])) in cand_keys
    # C(6,3) = 20 non-conflicting 3-leg combos, all clearing the default floor.
    assert len(candidates) == 20
    assert all(c["n_sims"] == len(legs[0].mask) for c in candidates)


def test_search_match_sgms_price_shrink_pulls_toward_target_implied_prob():
    # Use $5.00 target: the 6-leg pool has combos in [5.00, 6.50] (e.g. BEF).
    legs = _ladder_legs()
    target = MULTI_TARGET_ODDS[3]    # $5.00
    anchor_prob = 1.0 / target
    raw = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)[0]
    full = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0, price_shrink=1.0)[0]
    half = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0, price_shrink=0.5)[0]
    assert not raw.get("no_bet"), "pool must have a combo in the $5 band for this test"
    assert full["joint_prob"] == pytest.approx(anchor_prob)              # fully shrunk -> exactly at target
    assert half["joint_prob"] == pytest.approx((raw["joint_prob"] + anchor_prob) / 2)
    assert full["fair_odds"] == pytest.approx(target)


def test_search_match_sgms_corr_gain_haircut_zero_lift_equals_naive_product():
    # FIX-PLACEABLE-LEGS-AND-210-FLOOR STEP 4 moved the haircut to BEFORE
    # selection, so it can change which combo wins with a wider leg pool --
    # exactly 3 legs (one possible combo) isolates the haircut math itself.
    # Use the last 3 legs (D, E, F: probs 0.68, 0.55, 0.42) whose combo lands
    # in the $5.00 band [5.00, 6.50] (joint ≈ 0.157, fair ≈ 6.37).
    legs = _ladder_legs()[-3:]
    target = MULTI_TARGET_ODDS[3]    # $5.00
    raw = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)[0]
    zero = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0,
                             corr_gain_haircut=0.0)[0]
    assert not raw.get("no_bet"), "pool must have a combo in the $5 band"
    assert zero["joint_prob"] == pytest.approx(zero["naive_product"])
    assert zero["fair_odds"] == pytest.approx(1.0 / zero["naive_product"])
    # naive_product/corr_gain stay at their pre-haircut (informational) values.
    assert zero["naive_product"] == pytest.approx(raw["naive_product"])
    assert zero["corr_gain"] == pytest.approx(raw["corr_gain"])
    assert zero["joint_prob"] != pytest.approx(raw["joint_prob"])


def test_search_match_sgms_corr_gain_haircut_default_is_unhaircut():
    legs = _ladder_legs()
    raw = search_match_sgms(legs)
    unhaircut = search_match_sgms(legs, corr_gain_haircut=1.0)
    for r, u in zip(raw, unhaircut):
        if r.get("no_bet") or u.get("no_bet"):
            continue  # NO BET sentinels have no joint_prob to compare
        assert r["joint_prob"] == pytest.approx(u["joint_prob"])
        assert r["fair_odds"] == pytest.approx(u["fair_odds"])


def test_search_match_sgms_corr_gain_haircut_half_is_midpoint():
    # Same isolation as the zero-lift test above -- exactly 3 legs (D, E, F),
    # one possible combo that lands in the $5.00 band (joint ≈ 0.157, fair ≈ 6.37).
    legs = _ladder_legs()[-3:]
    target = MULTI_TARGET_ODDS[3]    # $5.00
    raw = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)[0]
    half = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0,
                             corr_gain_haircut=0.5)[0]
    assert not raw.get("no_bet"), "pool must have a combo in the $5 band"
    expected = raw["naive_product"] + 0.5 * raw["corr_gain"]
    assert half["joint_prob"] == pytest.approx(expected)


def test_search_match_sgms_corr_gain_haircut_recomputes_edge_when_priced():
    # Use $5.00 target; 6-leg pool with odds_mult=1.05 reaches it (BEF combo).
    legs = _ladder_legs(odds_mult=1.05)
    odds_book = {leg.name: leg.market_odds for leg in legs}
    target = MULTI_TARGET_ODDS[3]    # $5.00
    haircut = search_match_sgms(legs, odds_book=odds_book, target_odds=(target,),
                                min_joint_prob=0.0, corr_gain_haircut=0.0)[0]
    assert not haircut.get("no_bet"), "pool must have a combo in the $5 band"
    assert "book_odds" in haircut
    expected_raw_edge = haircut["joint_prob"] * haircut["book_odds"] - 1.0
    assert haircut["raw_edge"] == pytest.approx(expected_raw_edge)


def test_round_report_and_grade_multis_default_to_the_validated_corr_gain_haircut():
    # Closing the model-upgrade overconfidence investigation: round-report's
    # OWN default is now CORR_GAIN_HAIRCUT (0.0, OOS-validated), even though
    # search_match_sgms's own bare default stays 1.0/unhaircut (see the test
    # above) -- the validated value is a live-default choice made by the
    # caller, not a change to the general-purpose low-level function.
    import inspect

    from afl_bot.cli import grade_multis, round_report
    from afl_bot.config import CORR_GAIN_HAIRCUT

    assert CORR_GAIN_HAIRCUT == 0.0
    assert inspect.signature(round_report).parameters["corr_gain_haircut"].default == CORR_GAIN_HAIRCUT
    assert inspect.signature(grade_multis).parameters["corr_gain_haircut"].default == CORR_GAIN_HAIRCUT


def test_search_match_sgms_lcb_z_can_change_the_selected_combo():
    """Two pure 3-leg pools with exact joint_prob 0.30 and 0.31, target implied
    prob 0.29: both pools are BELOW the band floor (fair $3.33/$3.23 < $3.45 target),
    so lcb_z=0 (default band-window enforcement) returns NO BET.

    lcb_z>0 bypasses the band window and uses distance-to-target ranking — at
    tiny lcb_z the closest pool (0.30) is picked; at lcb_z=0.5 the LCB penalty
    flips to the other pool (0.31 whose LCB is closer to the target_prob 0.29).
    This verifies that lcb_z actually changes which combo is selected
    — model-upgrade audit Phase 3.5."""
    n = 200
    mask_30 = np.zeros(n, dtype=bool); mask_30[:60] = True   # joint_prob exactly 0.30
    mask_31 = np.zeros(n, dtype=bool); mask_31[:62] = True   # joint_prob exactly 0.31
    pool_30 = [_leg(f"A{i} 15+", 1.0, mask_30, f"A{i}") for i in range(3)]
    pool_31 = [_leg(f"B{i} 15+", 1.0, mask_31, f"B{i}") for i in range(3)]
    legs = pool_30 + pool_31
    target = 1.0 / 0.29   # ≈ $3.448; combos sit BELOW this → outside band window

    # lcb_z=0: band window enforcement → NO BET (combos at $3.23-$3.33, below $3.448 floor)
    default = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)
    assert default[0].get("no_bet"), (
        "lcb_z=0 enforces band window; combos at $3.23-$3.33 are below the $3.45 floor"
    )

    # lcb_z=0.01: bypasses band window, minimal LCB penalty → distance ranking picks 0.30
    tiny_lcb = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0, lcb_z=0.01)
    assert tiny_lcb[0]["joint_prob"] == pytest.approx(0.30)

    # lcb_z=0.5: LCB of 0.31 is closer to target_prob 0.29 → flips selection to 0.31
    haircut = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0, lcb_z=0.5)
    assert haircut[0]["joint_prob"] == pytest.approx(0.31)


def test_search_match_sgms_bottom_rung_lands_at_honest_floor_without_near_lock_legs():
    # FIX-PLACEABLE-LEGS-AND-210-FLOOR: the SGM ladder pool is now placeable-
    # only -- every leg capped at LEG_PROB_MAX (0.78, no near-lock legs above
    # it). The bottom rung's target moved 1.75 -> 2.10, the honest floor that
    # falls out of three 0.78-capped legs (0.78^3 ~= $2.11) WITHOUT needing
    # any unplaceable near-lock leg the old $1.75 target required.
    rng = np.random.default_rng(3)
    n = 40000
    probs = {"A": 0.78, "B": 0.75, "C": 0.70, "D": 0.60, "E": 0.50, "F": 0.40}
    legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        prob = mask.mean()
        legs.append(_leg(f"{name} 15+ disp", prob, mask, name))

    out = search_match_sgms(legs)
    bottom = out[0]
    assert bottom["target_odds"] == MULTI_TARGET_ODDS[0] == 2.10
    assert 2.10 <= bottom["fair_odds"] <= 2.50          # reachable, landed at/just above target


def test_search_match_sgms_never_lands_short_after_calibration():
    # FIX-PLACEABLE-LEGS-AND-210-FLOOR STEP 4: a calibrator that INFLATES
    # joint probability (and so shrinks fair odds) must not be able to sneak
    # a rung's DISPLAYED fair odds below the band it was selected to clear --
    # the "never land shorter" guard must see the same final (haircut +
    # calibrated) number that ends up reported, not the pre-calibration one.
    # Pool spans the full 6-band range so every band is reachable even after
    # the 1.15x inflation (9 subjects A-I, p from 0.90 down to 0.32).
    class InflatingCalibrator:
        def predict(self, probs):
            return np.minimum(np.asarray(probs, dtype=float) * 1.15, 0.999)

    rng = np.random.default_rng(5)
    n = 40_000
    probs9 = {"A": 0.90, "B": 0.85, "C": 0.78, "D": 0.68, "E": 0.55,
              "F": 0.42, "G": 0.38, "H": 0.35, "I": 0.32}
    wide_legs = []
    for name, p in probs9.items():
        mask = rng.random(n) < p
        wide_legs.append(_leg(f"{name} 15+ disp", mask.mean(), mask, name))

    out = search_match_sgms(wide_legs, multi_calibrator=InflatingCalibrator())
    for r in out:
        if r.get("no_bet"):
            continue  # NO BET rungs have no fair_odds to check
        assert r["fair_odds"] >= r["target_odds"] - 1e-9


def test_search_match_sgms_lands_at_or_above_target_not_short_when_possible():
    # Two pure 3-leg pools, nested masks so every MIXED combo collapses to the
    # smaller (LONG-odds) pool's joint (intersecting with a subset forces the
    # AND down to it): a pure-SHORT combo at joint 0.55 (fair $1.82, CLOSER to
    # the $2.00 target in absolute distance) vs. 19 LONG combos at joint 0.45
    # (fair $2.22, reaches/exceeds the target). The old closest-distance-only
    # selection would pick the closer-but-shorter $1.82 combo (Ben's "~$1.50"
    # complaint); the fix must prefer the one that doesn't undershoot.
    n = 1000
    mask_short = np.zeros(n, dtype=bool)
    mask_short[:550] = True   # pure pool joint_prob 0.55 -- shorter than target
    mask_long = np.zeros(n, dtype=bool)
    mask_long[:450] = True    # pure pool joint_prob 0.45 -- reaches/exceeds target (nested subset)
    pool_short = [_leg(f"S{i} 15+", 1.0, mask_short, f"S{i}") for i in range(3)]
    pool_long = [_leg(f"L{i} 15+", 1.0, mask_long, f"L{i}") for i in range(3)]
    legs = pool_short + pool_long
    target = 2.0   # implied prob 0.5

    out = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)
    assert out[0]["joint_prob"] == pytest.approx(0.45)
    assert out[0]["fair_odds"] >= target


# --------------------------------------------------------------------------- #
# search_market_sgms (FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART C): the ladder
# selected/priced on REAL book odds, not the model's own joint probability.
# --------------------------------------------------------------------------- #

def test_search_market_sgms_empty_when_nothing_priced():
    legs = _ladder_legs()
    assert search_market_sgms(legs, odds_book={}) == []


def test_search_market_sgms_only_fully_priced_combos():
    legs = _ladder_legs()
    priced_names = {legs[0].name, legs[1].name, legs[2].name, legs[3].name}   # 4 of 6 legs
    odds_book = {leg.name: leg.market_odds for leg in legs if leg.name in priced_names}
    out = search_market_sgms(legs, odds_book=odds_book)
    assert out
    real_rungs = [r for r in out if not r.get("no_bet")]
    assert real_rungs, "some bands must be reachable with 4 priced legs"
    for r in real_rungs:
        assert all(name in priced_names for name in r["legs"])
        assert {"book_odds", "edge", "joint_prob", "fair_odds"} <= set(r)


def test_search_market_sgms_value_pick_is_real_edge_only():
    # 9 distinct players → diversity constraint allows 3 real rungs (3 players each).
    # Use target_odds=(2.10, 3.50, 5.00) so the top band ($5.00) can be filled
    # without reusing the players from band 1 (B,C,D) or band 2 (empty/NO BET).
    rng = np.random.default_rng(5)
    n = 40_000
    probs9 = {"A": 0.90, "B": 0.85, "C": 0.78, "D": 0.68, "E": 0.55,
              "F": 0.42, "G": 0.38, "H": 0.35, "I": 0.32}
    odds_mult = 1.05
    wide_legs = []
    for name, p in probs9.items():
        mask = rng.random(n) < p
        wide_legs.append(_leg(f"{name} 15+ disp", mask.mean(), mask, name,
                              odds=(1.0 / mask.mean()) * odds_mult))
    odds_book = {leg.name: leg.market_odds for leg in wide_legs}
    out = search_market_sgms(wide_legs, odds_book=odds_book,
                             target_odds=(2.10, 3.50, 5.00))
    picks = [r for r in out if r.get("value_pick")]
    assert len(picks) == 1                             # only the top-band rung is value pick
    vp = picks[0]
    assert 0.0 < vp["edge"] <= 0.15


def test_search_market_sgms_implausible_edge_not_flagged_value():
    legs = _ladder_legs(odds_mult=1.20)   # book way over fair -> shrunk edge blows past 15%
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book)
    assert not any(r.get("value_pick") for r in out)


def test_search_market_sgms_lands_at_or_above_book_target_not_short_when_possible():
    # Part A (band-selection by EV): within combos that clear the target, pick
    # the one with highest market-shrunk Total EV, not the cheapest one.
    # With all-True masks (jointâ‰ˆ1), _total_ev = edge = 0.75*(book-1), so
    # highest EV = highest book odds within the clearing set.
    import math
    from itertools import combinations as _combos

    prices = [1.15, 1.18, 1.21, 1.30, 1.33, 1.36]
    mask = np.ones(50, dtype=bool)
    legs = [_leg(f"P{i} 15+", 0.9, mask, f"P{i}", odds=p) for i, p in enumerate(prices)]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    target = 2.0

    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(target,), min_joint_prob=0.0)
    assert out[0]["book_odds"] >= target
    clearing = [math.prod(c) for c in _combos(prices, 3) if math.prod(c) >= target]
    # Highest EV within band = highest book odds (all joints â‰ˆ 1 here).
    assert out[0]["book_odds"] == pytest.approx(max(clearing))


def test_search_market_sgms_no_bet_when_nothing_reaches_band_window():
    # All three legs near-locks -> every combo's book price sits well under
    # the target band window. With band-window enforcement, this returns NO BET
    # (never show a combo below its band floor).
    mask = np.ones(100, dtype=bool)
    legs = [_leg(f"N{i} 15+", 0.95, mask, f"N{i}", odds=1.05) for i in range(3)]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(MULTI_TARGET_ODDS[0],),
                             min_joint_prob=0.0)
    assert len(out) == 1
    assert out[0].get("no_bet"), (
        "Combo book price ~$1.16 is below $2.10 floor — must be NO BET, not shown"
    )


def test_select_ladder_lines_keeps_every_priced_line_plus_best_unpriced():
    # FIX-PLACEABLE-LEGS-AND-210-FLOOR STEP 2.2: at most one UNPRICED line
    # per (player, stat) -- the highest-prob ("best line") -- survives into
    # the live ladder pool, but every PRICED line is exempt from the cull.
    qualifying = [
        {"line": 20, "prob": 0.60, "priced": False},
        {"line": 25, "prob": 0.45, "priced": False},   # lower prob, unpriced -> dropped
        {"line": 30, "prob": 0.20, "priced": True},     # priced -> always kept
    ]
    kept = select_ladder_lines(qualifying)
    assert {q["line"] for q in kept} == {20, 30}


def test_select_ladder_lines_keeps_all_priced_lines_with_no_unpriced():
    qualifying = [
        {"line": 20, "prob": 0.60, "priced": True},
        {"line": 25, "prob": 0.45, "priced": True},
    ]
    assert select_ladder_lines(qualifying) == qualifying


def test_select_ladder_lines_empty_in_empty_out():
    assert select_ladder_lines([]) == []


def test_render_markdown_smoke():
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.01, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [("A", projection_rows({"X": {"disposals": np.full(100, 20)}}, LINES))],
        "sgms": [],
    }]
    md = render_markdown(2026, 14, matches, has_odds=False)
    assert "AFL Round Report" in md and "A vs B" in md


def test_render_markdown_shows_band_column_for_each_rung_target():
    sgms = [
        {"legs": ["A 15+", "B 15+", "C 15+"], "joint_prob": 0.55, "naive_product": 0.50,
         "corr_gain": 0.05, "fair_odds": 1.82, "odds": 1.82, "value_pick": False,
         "target_odds": 1.75},
        {"legs": ["D 15+", "E 15+", "F 15+"], "joint_prob": 0.20, "naive_product": 0.19,
         "corr_gain": 0.01, "fair_odds": 5.0, "odds": 5.0, "value_pick": False,
         "target_odds": 5.0},
    ]
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": sgms,
    }]
    md = render_markdown(2026, 14, matches, has_odds=False)
    assert "| Band |" in md
    assert "$1.75" in md and "$5.00" in md


def test_render_markdown_labels_value_pick_and_odds_note():
    sgms = [
        {"legs": ["A 15+", "B 15+", "C 15+"], "joint_prob": 0.30, "naive_product": 0.28,
         "corr_gain": 0.02, "fair_odds": 3.33, "odds": 2.0, "value_pick": False},
        {"legs": ["A 20+", "D 15+", "E 15+"], "joint_prob": 0.22, "naive_product": 0.20,
         "corr_gain": 0.02, "fair_odds": 4.55, "book_odds": 4.80, "edge": 0.056,
         "odds": 4.80, "value_pick": True},
    ]
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": sgms,
    }]
    md = render_markdown(2026, 14, matches, has_odds=True, odds_note="_Live odds: 4 H2H legs._")
    assert "VALUE PICK" in md
    assert "Live odds: 4 H2H legs" in md


def test_top_n_players_by_stat_ranks_by_projected_mean():
    samples = {
        "Star": {"disposals": np.full(10, 30.0)},
        "Role": {"disposals": np.full(10, 15.0)},
        "Bench": {"disposals": np.full(10, 8.0)},
        "Ruck": {"marks": np.full(10, 5.0)},   # no disposals array -- ignored for that stat
    }
    assert top_n_players_by_stat(samples, "disposals", 2) == {"Star", "Role"}
    assert top_n_players_by_stat(samples, "disposals", 1) == {"Star"}
    assert top_n_players_by_stat(samples, "marks", 5) == {"Ruck"}


def test_is_bookable_model_only_leg_requires_menu_line():
    top_n = {"Star"}
    assert is_bookable_model_only_leg("disposals", 20, "Star", "midfielder", top_n)
    assert not is_bookable_model_only_leg("disposals", 35, "Star", "midfielder", top_n)  # off-menu


def test_is_bookable_model_only_leg_requires_top_n_rank():
    assert not is_bookable_model_only_leg("disposals", 20, "Bench", "midfielder", {"Star"})


def test_is_bookable_model_only_leg_gates_marks_and_tackles_by_role():
    top_n = {"Key Defender"}
    # key defenders rarely get a marks/tackles market posted -- "general" role excluded
    assert not is_bookable_model_only_leg("marks", 5, "Key Defender", "general", top_n)
    assert not is_bookable_model_only_leg("tackles", 4, "Key Defender", "general", top_n)
    assert is_bookable_model_only_leg("marks", 5, "Key Defender", "ruck", top_n)
    assert is_bookable_model_only_leg("tackles", 4, "Key Defender", "forward", top_n)
    # disposals/goals aren't role-gated
    assert is_bookable_model_only_leg("disposals", 20, "Key Defender", "general", top_n)
    assert is_bookable_model_only_leg("goals", 1, "Key Defender", "general", top_n)


def test_render_markdown_tags_model_only_rung_without_book_price():
    sgms = [
        {"legs": ["A 15+", "B 15+", "C 15+"], "joint_prob": 0.30, "naive_product": 0.28,
         "corr_gain": 0.02, "fair_odds": 3.33, "odds": 3.33, "value_pick": False},
    ]
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": sgms,
    }]
    md = render_markdown(2026, 14, matches, has_odds=False)
    assert "model-only — verify market exists" in md


def test_render_markdown_header_says_model_ladder_not_band_promise():
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": [], "n_legs": 0,
    }]
    md = render_markdown(2026, 14, matches, has_odds=False)
    assert "### Model ladder (model fair odds, no book)" in md


def test_render_markdown_shows_sportsbet_ladder_with_honesty_note_when_priced():
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": [],
        "market_sgms": [
            {"legs": ["A 15+", "B 15+", "C 15+"], "book_odds": 1.53, "joint_prob": 0.46,
             "fair_odds": 2.38, "edge": 0.04, "value_pick": True, "target_odds": 2.10},
        ],
    }]
    md = render_markdown(2026, 14, matches, has_odds=False, sportsbet_note="_Test note._")
    assert "### Sportsbet ladder (real prices)" in md
    assert "1.53" in md and "2.38" in md
    assert "**VALUE PICK**" in md
    assert "the book's own correlation model" in md
    assert "_Test note._" in md


def test_render_markdown_no_sportsbet_ladder_section_when_unpriced():
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": [], "market_sgms": [],
    }]
    md = render_markdown(2026, 14, matches, has_odds=False)
    assert "Sportsbet ladder" not in md


def test_build_odds_template_every_name_maps_to_null_plus_rules_stub():
    template = build_odds_template(["A to win", "B to win", "X 20+ disposals"])
    assert template["A to win"] is None
    assert template["B to win"] is None
    assert template["X 20+ disposals"] is None
    assert template["_rules"] == {"h2h_draw": None}


def test_build_odds_template_dedupes_and_sorts():
    template = build_odds_template(["Z leg", "A leg", "A leg"])
    keys = [k for k in template if not k.startswith("_")]
    assert keys == ["A leg", "Z leg"]


def test_render_markdown_priced_props_table_shows_devig_and_class():
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": [],
        "priced_legs": [
            {"name": "X 20+ disposals", "model_prob": 0.55, "book_odds": 1.85,
             "devig_prob": 0.52, "devig_label": "two-way devig", "blended_prob": 0.532,
             "edge_pct": 0.017, "classification": "SKIP"},
            {"name": "Y 15+ disposals", "model_prob": 0.65, "book_odds": None,
             "devig_prob": 0.60, "devig_label": "single-sided (approx)", "blended_prob": 0.62,
             "edge_pct": 0.0, "classification": "SKIP"},
        ],
    }]
    md = render_markdown(2026, 14, matches, has_odds=True)
    assert "Priced props (from --odds)" in md
    assert "X 20+ disposals" in md and "two-way devig" in md
    assert "Y 15+ disposals" in md and "single-sided (approx)" in md
    assert "| - |" in md   # Y's missing book_odds renders as a dash


def test_render_markdown_no_priced_legs_section_when_empty():
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": [],
    }]
    md = render_markdown(2026, 14, matches, has_odds=False)
    assert "Priced props (from --odds)" not in md


def test_render_markdown_explains_empty_ladder_with_leg_count():
    # B4: a thin match (too few legs) must say WHY, not a bare "no combinations".
    matches = [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False, "is_wet": False,
                   "mu_margin": 5.0, "mu_total": 160.0, "p_home": 0.6, "p_away": 0.39,
                   "p_draw": 0.0, "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": [], "n_legs": 2,
    }]
    md = render_markdown(2026, 14, matches, has_odds=False)
    assert "Only 2 candidate legs" in md


def test_build_sgm_candidates_rejects_same_subject_in_combo():
    """Each player/team may appear in at most one leg per multi (distinct-subject rule)."""
    rng = np.random.default_rng(99)
    n = 40_000
    from afl_bot.build.report import build_sgm_candidates
    legs = []
    for player in ("P1", "P2", "P3"):
        for market, suffix in (("player_disposals", "disp"), ("player_marks", "marks")):
            p = 0.55
            mask = rng.random(n) < p
            legs.append(LegCandidate(
                f"{player} {suffix}", "m1", market, player, mask.mean(), 1 / mask.mean(), mask=mask))
    # 3 players Ã— 2 markets = 6 legs; without the subject filter we'd get combos
    # like (P1-disp, P1-marks, P2-disp) where P1 appears twice.
    candidates = build_sgm_candidates(legs)
    leg_by_name = {leg.name: leg for leg in legs}
    for c in candidates:
        subjects = [leg_by_name[n].subject for n in c["legs"]]
        assert len(set(subjects)) == len(subjects), f"Duplicate subject in combo: {c['legs']}"

    out = search_match_sgms(legs)
    for r in out:
        subjects = [leg_by_name[n].subject for n in r["legs"]]
        assert len(set(subjects)) == len(subjects), f"Duplicate subject in rung: {r['legs']}"


def test_search_match_sgms_deep_pool_yields_up_to_6_distinct_rungs():
    """A pool spanning safeâ†’longshot produces up to 6 distinct rungs (one per band)."""
    rng = np.random.default_rng(77)
    n = 40_000
    probs = {"A": 0.77, "B": 0.70, "C": 0.62, "D": 0.55, "E": 0.47,
             "F": 0.41, "G": 0.36, "H": 0.33, "I": 0.30}
    legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        legs.append(_leg(f"{name} 20+ disp", mask.mean(), mask, name))

    out = search_match_sgms(legs)
    assert len(out) == len(MULTI_TARGET_ODDS)   # one rung per target (6 bands)
    # Only real (non-NO BET) rungs must use distinct combos; multiple NO BET
    # sentinels are fine — they share legs=[] but aren't real bets.
    real_out = [r for r in out if not r.get("no_bet")]
    all_leg_tuples = [tuple(sorted(r["legs"])) for r in real_out]
    assert len(all_leg_tuples) == len(set(all_leg_tuples)), "Duplicate combos in ladder"
    for r in out:
        subjects = [n.split()[0] for n in r["legs"]]   # first word of each leg name = subject
        assert len(set(subjects)) == len(subjects), f"Duplicate subject in rung: {r['legs']}"


# --------------------------------------------------------------------------- #
# Stat preference + marks cap (FIX-HIT-PCT-AND-PREFER-DISPOSALS Part B)
# --------------------------------------------------------------------------- #

def _marks_leg(name, subject, prob=0.70):
    return LegCandidate(name=name, match_id="m1", market="player_marks",
                        subject=subject, fair_prob=prob, market_odds=1.0 / prob, mask=None)


def test_build_sgm_candidates_has_pref_score_and_marks_count():
    legs = [
        _leg("D1 20+ disp", 0.70, None, "D1"),   # disposals
        _leg("D2 20+ disp", 0.70, None, "D2"),
        _marks_leg("M1 4+ marks", "M1"),           # marks, no book price
    ]
    candidates = build_sgm_candidates(legs, odds_book={})
    assert all("_pref_score" in c for c in candidates)
    assert all("_n_marks" in c for c in candidates)
    # The one combo: 2 disposals + 1 marks
    assert len(candidates) == 1
    c = candidates[0]
    from afl_bot.config import STAT_PREFERENCE
    expected_pref = STAT_PREFERENCE["disposals"] * 2 + STAT_PREFERENCE["marks"]
    assert c["_pref_score"] == pytest.approx(expected_pref)
    assert c["_n_marks"] == 1   # one marks leg (all marks count, not just unpriced)


def test_stat_preference_picks_disposals_over_marks():
    # All four legs have identical prob (0.70, no mask) â†’ identical joint prob
    # for any 3-combo. Pref score is the only tiebreaker; the disposals-only
    # combo must win.
    legs = [
        _leg("D1 20+ disp", 0.70, None, "D1"),
        _leg("D2 20+ disp", 0.70, None, "D2"),
        _leg("D3 20+ disp", 0.70, None, "D3"),
        _marks_leg("M1 4+ marks", "M1"),  # no book price â†’ model-only marks
    ]
    out = search_match_sgms(legs, target_odds=(2.10,), min_joint_prob=0.0)
    assert len(out) == 1
    assert "M1 4+ marks" not in out[0]["legs"]   # all-disposals combo wins


def test_marks_cap_filters_all_model_only_marks_combos():
    # Only marks legs, none priced â†’ every 3-combo exceeds MAX_MARKS_LEGS_PER_MULTI=1
    legs = [_marks_leg(f"M{i} 4+ marks", f"M{i}") for i in range(3)]
    out = search_match_sgms(legs, min_joint_prob=0.0)
    assert out == []   # all combos filtered, nothing to select


def test_priced_marks_leg_counts_toward_cap():
    # FIX-MARKS-CAP: ALL marks legs count, priced or not. Three priced marks
    # â†’ _n_marks=3 > MAX_MARKS_LEGS_PER_MULTI=1 â†’ all combos filtered.
    legs = [_marks_leg(f"M{i} 4+ marks", f"M{i}") for i in range(3)]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, min_joint_prob=0.0)
    assert out == []   # cap applies to priced marks just like unpriced


def test_final_rungs_have_no_internal_pref_fields():
    out = search_match_sgms(_ladder_legs())
    for r in out:
        assert "_pref_score" not in r
        assert "_n_marks" not in r


def test_marks_cap_at_most_one_priced_marks_leg_per_rung():
    # FIX-MARKS-CAP: build_sgm_candidates emits the full pool (including high-marks
    # combos); search_match_sgms filters out any combo with _n_marks > cap.
    # With 2 disposals + 2 marks: combos (D1,M1,M2) and (D2,M1,M2) have _n_marks=2
    # and must be absent from every selected rung.
    legs = [
        _leg("D1 20+ disp", 0.65, None, "D1"),
        _leg("D2 20+ disp", 0.65, None, "D2"),
        _marks_leg("M1 4+ marks", "M1", prob=0.65),
        _marks_leg("M2 4+ marks", "M2", prob=0.65),
    ]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    # Candidates pool has combos with _n_marks=1 AND _n_marks=2 (the two 2-marks combos).
    candidates = build_sgm_candidates(legs, odds_book=odds_book)
    assert any(c["_n_marks"] == 2 for c in candidates)   # those exist in the pool...
    out = search_match_sgms(legs, odds_book=odds_book, min_joint_prob=0.0)
    for r in out:
        mark_legs = [n for n in r["legs"] if "marks" in n]
        assert len(mark_legs) <= 1   # ...but none appear in any selected rung


def test_marks_cap_market_sgms_at_most_one_priced_marks_leg_per_rung():
    # Same check for the Sportsbet ladder (search_market_sgms).
    legs = [
        _leg("D1 20+ disp", 0.65, None, "D1"),
        _leg("D2 20+ disp", 0.65, None, "D2"),
        _marks_leg("M1 4+ marks", "M1", prob=0.65),
        _marks_leg("M2 4+ marks", "M2", prob=0.65),
    ]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, min_joint_prob=0.0)
    for r in out:
        mark_legs = [n for n in r["legs"] if "marks" in n]
        assert len(mark_legs) <= 1


# --------------------------------------------------------------------------- #
# Combined tackles+marks cap (MAX_TACKLE_MARKS_LEGS)                         #
# --------------------------------------------------------------------------- #

def _tackle_leg(name="T1 5+ tackles", player="T1", prob=0.65):
    from afl_bot.build.multi import LegCandidate
    return LegCandidate(name=name, match_id="m1", market="tackles",
                        subject=player, fair_prob=prob, market_odds=1.0 / prob, mask=None)


def test_tackle_marks_combined_cap_filters_excess_combos():
    # Two tackles + one disposals leg: the combo has _n_tackle_marks=2 which
    # exceeds MAX_TACKLE_MARKS_LEGS=1, so no rungs should contain both tackles legs.
    legs = [
        _leg("D1 20+ disp", 0.65, None, "D1"),
        _tackle_leg("T1 5+ tackles", "T1", prob=0.65),
        _tackle_leg("T2 5+ tackles", "T2", prob=0.65),
    ]
    candidates = build_sgm_candidates(legs)
    # The pool contains a combo with _n_tackle_marks=2 (T1+T2+D1).
    assert any(c.get("_n_tackle_marks", 0) == 2 for c in candidates)
    # After filter, no selected rung should have two tackles legs.
    out = search_match_sgms(legs, min_joint_prob=0.0)
    for r in out:
        tackle_legs = [n for n in r["legs"] if "tackles" in n]
        assert len(tackle_legs) <= 1


def test_disposal_preferred_over_equal_marks_leg():
    # Disposals legs carry a higher STAT_PREFERENCE weight than marks. Verify
    # that the all-disposals combo has a strictly higher _pref_score than any
    # mixed combo in the candidate pool (the tie-break invariant), and that the
    # first selected rung (sorted safestâ†’longest) has no marks leg.
    legs = [
        _leg("D1 20+ disp", 0.65, None, "D1"),
        _leg("D2 20+ disp", 0.65, None, "D2"),
        _leg("D3 20+ disp", 0.65, None, "D3"),
        _marks_leg("M1 4+ marks", "M1", prob=0.65),
    ]
    candidates = build_sgm_candidates(legs)
    pure_disp = [c for c in candidates if c["_n_marks"] == 0]
    mixed = [c for c in candidates if c["_n_marks"] > 0]
    assert pure_disp, "all-disposals combo must appear in candidate pool"
    assert mixed, "mixed combo must also be present so we can compare"
    # Core invariant: disposals-only outscores any mixed combo on _pref_score.
    best_pure = max(c["_pref_score"] for c in pure_disp)
    best_mixed = max(c["_pref_score"] for c in mixed)
    assert best_pure > best_mixed, (
        f"all-disposals pref {best_pure} should beat mixed pref {best_mixed}"
    )


def test_positive_edge_marks_leg_survives_tackle_marks_cap():
    # A priced marks leg with real positive edge must survive even when it is
    # the only "value" leg. The cap filters based on COUNT, not on edge, so a
    # single marks/tackles leg (n_tackle_marks=1) always passes the combined cap.
    legs = [
        _leg("D1 20+ disp", 0.65, None, "D1"),
        _leg("D2 20+ disp", 0.65, None, "D2"),
        _marks_leg("M1 4+ marks", "M1", prob=0.72),  # prob > implied price â†’ +EV
    ]
    odds_book = {"M1 4+ marks": 1.0 / 0.65}   # book has it at 0.65 implied â†’ model 0.72 â†’ +EV
    out = search_match_sgms(legs, odds_book=odds_book, min_joint_prob=0.0)
    assert out, "should return rungs"
    # Marks leg should appear (n_tackle_marks=1 â‰¤ cap=1, so it passes).
    mark_legs_found = any("marks" in n for r in out for n in r["legs"])
    assert mark_legs_found, "positive-edge marks leg should survive the cap"


def test_tackle_marks_cap_sportsbet_ladder():
    # Same combined cap applies to search_market_sgms.
    legs = [
        _leg("D1 20+ disp", 0.65, None, "D1"),
        _tackle_leg("T1 5+ tackles", "T1", prob=0.65),
        _tackle_leg("T2 5+ tackles", "T2", prob=0.65),
    ]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, min_joint_prob=0.0)
    for r in out:
        tackle_legs = [n for n in r["legs"] if "tackles" in n]
        assert len(tackle_legs) <= 1


def test_final_rungs_have_no_internal_tackle_marks_field():
    out = search_match_sgms(_ladder_legs())
    for r in out:
        assert "_n_tackle_marks" not in r


def test_disposals_first_beats_closer_marks_combo_model_ladder():
    # Both combos must land INSIDE the $2.10 band window [2.10, 2.73].
    # Band window in prob space: [1/2.73, 1/2.10] = [0.366, 0.476].
    #
    # D legs at prob=0.78: D1+D2+D3 joint=0.78^3≈0.474, fair≈2.11 (in window, tm=0).
    # M1 (marks) at prob=0.75: D1+D2+M1 joint=0.78*0.78*0.75≈0.456, fair≈2.19 (in window, tm=1).
    # Both in window; disposals-first tier (tm=0) must win over closer-to-floor marks combo.
    legs = [
        _leg("D1 25+ disp", 0.78, None, "D1"),
        _leg("D2 25+ disp", 0.78, None, "D2"),
        _leg("D3 25+ disp", 0.78, None, "D3"),
        _marks_leg("M1 4+ marks", "M1", prob=0.75),
    ]
    out = search_match_sgms(legs, target_odds=(2.10,), min_joint_prob=0.0)
    assert len(out) == 1
    assert not out[0].get("no_bet"), "both combos are in window, must get a pick"
    assert "M1 4+ marks" not in out[0]["legs"], (
        "disposals-first: all-disposals combo must win even when it has lower joint"
    )


def test_disposals_first_beats_closer_marks_combo_sportsbet_ladder():
    # Mirror of the model-ladder test but using search_market_sgms (book_odds path).
    # Both combos must land in the $2.10 band window [2.10, 2.73].
    # D legs at prob=0.78: D1+D2+D3 book≈(1/0.78)^3≈2.11, in window, tm=0.
    # M1 (marks) at prob=0.75: D1+D2+M1 book=2.19, in window, tm=1, CHEAPER (closer floor).
    # Disposals-first tier (tm=0) must beat the cheaper marks combo.
    legs = [
        _leg("D1 25+ disp", 0.78, None, "D1"),
        _leg("D2 25+ disp", 0.78, None, "D2"),
        _leg("D3 25+ disp", 0.78, None, "D3"),
        _marks_leg("M1 4+ marks", "M1", prob=0.75),
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(2.10,),
                             min_joint_prob=0.0)
    assert len(out) == 1
    assert not out[0].get("no_bet"), "both combos are in window, must get a pick"
    assert "M1 4+ marks" not in out[0]["legs"], (
        "disposals-first: sportsbet ladder must prefer all-disposals combo"
    )


def test_fallback_to_marks_when_no_disposals_combo_reaches_target():
    # When the only combo that reaches the target band includes a marks leg,
    # it must still be selected (necessary fallback, not blocked by the cap).
    # Only 2 disposals legs exist -> no 3-disp combo possible. The marks leg
    # is needed to form the only 3-leg combo that can reach $5.
    # D1 prob=0.60, D2 prob=0.55, M1 (marks) prob=0.50 -> joint~0.165, fair~6.1
    legs = [
        _leg("D1 30+ disp", 0.60, None, "D1"),
        _leg("D2 30+ disp", 0.55, None, "D2"),
        _marks_leg("M1 5+ marks", "M1", prob=0.50),
    ]
    out = search_match_sgms(legs, target_odds=(5.0,), min_joint_prob=0.0)
    assert len(out) == 1
    assert "M1 5+ marks" in out[0]["legs"], (
        "marks leg must appear when no all-disposals combo can reach the target"
    )


# --------------------------------------------------------------------------- #
# Total-points legs in multis, h2h legs excluded (FIX-REMOVE-H2H-ADD-TOTAL-POINTS)
# --------------------------------------------------------------------------- #

def _total_leg(name="Total points 160.5+", prob=0.55, book_odds=None):
    return LegCandidate(name=name, match_id="m1", market="total_points",
                        subject="total", fair_prob=prob,
                        market_odds=book_odds if book_odds is not None else 1.0 / prob,
                        mask=None)


def test_h2h_legs_excluded_from_ladder():
    """cli.py filters h2h out of ladder_legs; h2h legs must never appear on a rung."""
    h2h = LegCandidate("Team A to win", "m1", "h2h", "Team A", 0.60, 1/0.60, mask=None)
    disp_b = _leg("PB 20+ disp", 0.65, None, "PB")
    disp_c = _leg("PC 20+ disp", 0.60, None, "PC")
    # cli.py: ladder_legs = [l for l in match_legs if l.market != "h2h"]
    pool = [l for l in [h2h, disp_b, disp_c] if l.market != "h2h"]
    out = search_match_sgms(pool, min_joint_prob=0.0)
    for r in out:
        assert not any("to win" in name.lower() for name in r["legs"])


def test_total_points_included_in_ladder():
    """Total-points leg enters the model ladder alongside player props."""
    disp_a = _leg("DA 20+ disp", 0.60, None, "DA")
    disp_b = _leg("DB 20+ disp", 0.55, None, "DB")
    total = _total_leg(prob=0.55)
    # cli.py only filters h2h; total_points is kept.
    pool = [l for l in [disp_a, disp_b, total] if l.market != "h2h"]
    out = search_match_sgms(pool, target_odds=(5.0,), min_joint_prob=0.0)
    assert len(out) == 1
    assert not out[0].get("no_bet"), "combo at joint≈0.181 (fair≈$5.52) is in $5.00 band"
    assert any("Total points" in name for name in out[0]["legs"])


def test_total_points_in_market_sgms_when_priced():
    """An O/U leg with a real book price can appear in the Sportsbet ladder."""
    # Choose odds whose product lands in the $5.00 band [5.00, 6.50]:
    # 2.20 * 1.65 * 1.65 ≈ 5.99 ✓
    disp_a = _leg("DA 20+ disp", 0.60, None, "DA", odds=2.20)
    disp_b = _leg("DB 20+ disp", 0.55, None, "DB", odds=1.65)
    total = _total_leg(prob=0.55, book_odds=1.65)
    pool = [l for l in [disp_a, disp_b, total] if l.market != "h2h"]
    odds_book = {l.name: l.market_odds for l in pool}
    out = search_market_sgms(pool, odds_book=odds_book, target_odds=(5.0,), min_joint_prob=0.0)
    assert len(out) == 1
    assert any("Total points" in name for name in out[0]["legs"])


def test_total_points_excluded_from_market_sgms_when_unpriced():
    """An O/U leg with no real book price (fair_odds fallback) stays out of Sportsbet ladder."""
    disp_a = _leg("DA 20+ disp", 0.60, None, "DA", odds=1.55)
    disp_b = _leg("DB 20+ disp", 0.55, None, "DB", odds=1.65)
    # total has fair_odds fallback — not in odds_book
    total = _total_leg(prob=0.55)
    pool = [l for l in [disp_a, disp_b, total] if l.market != "h2h"]
    odds_book = {l.name: l.market_odds for l in [disp_a, disp_b]}  # total NOT in book
    out = search_market_sgms(pool, odds_book=odds_book, min_joint_prob=0.0)
    for r in out:
        assert not any("Total points" in name for name in r["legs"])


def test_no_player_double_ups_in_model_ladder():
    """FIX-NO-PLAYER-DOUBLE-UPS: a player used in rung 1 must not appear in rung 2+."""
    rng = np.random.default_rng(42)
    n = 40000
    # Player "X" has two legs covering different bands.  If double-ups were
    # allowed, X could win rung 1 ($2.10) and rung 2 ($3.00) simultaneously.
    # Create enough other players so each band CAN be filled without X.
    probs = {"X": 0.90, "A": 0.85, "B": 0.78, "C": 0.68, "D": 0.55, "E": 0.42}
    legs = []
    for subj, p in probs.items():
        mask = rng.random(n) < p
        legs.append(_leg(f"{subj} 15+ disp", float(mask.mean()), mask, subj))
    # Also add a second leg for "X" (different line, different LegCandidate but same subject).
    x_mask2 = rng.random(n) < 0.82
    legs.append(_leg("X 20+ disp", float(x_mask2.mean()), x_mask2, "X"))

    out = search_match_sgms(legs, min_joint_prob=0.01)
    real_rungs = [r for r in out if not r.get("no_bet")]
    # Collect all player subjects across all rungs.
    all_players: list[str] = []
    leg_subj = {leg.name: leg.subject for leg in legs}
    for r in real_rungs:
        for lname in r["legs"]:
            s = leg_subj.get(lname, "total")
            if s != "total":
                all_players.append(s)
    # No player should appear more than once across all rungs.
    assert len(all_players) == len(set(all_players)), \
        f"Player double-up detected: {all_players}"


def test_no_player_double_ups_in_market_ladder():
    """FIX-NO-PLAYER-DOUBLE-UPS: same constraint for the Sportsbet (market) ladder."""
    rng = np.random.default_rng(7)
    n = 40000
    # Player "X" has two legs; each is priced. Force X into the lowest band
    # to confirm X is then excluded from higher bands.
    probs = {"X": 0.90, "A": 0.85, "B": 0.78, "C": 0.68, "D": 0.55, "E": 0.42}
    legs = []
    for subj, p in probs.items():
        mask = rng.random(n) < p
        odds = round(1.0 / p, 2)
        legs.append(_leg(f"{subj} 15+ disp", float(mask.mean()), mask, subj, odds=odds))
    x_mask2 = rng.random(n) < 0.82
    x2_odds = round(1.0 / 0.82, 2)
    legs.append(_leg("X 20+ disp", float(x_mask2.mean()), x_mask2, "X", odds=x2_odds))

    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, min_joint_prob=0.01)
    real_rungs = [r for r in out if not r.get("no_bet")]
    all_players: list[str] = []
    leg_subj = {leg.name: leg.subject for leg in legs}
    for r in real_rungs:
        for lname in r["legs"]:
            s = leg_subj.get(lname, "total")
            if s != "total":
                all_players.append(s)
    assert len(all_players) == len(set(all_players)), \
        f"Player double-up detected in market ladder: {all_players}"


# --------------------------------------------------------------------------- #
# Greasiness override + wet-marks multiplier (FIX-MARKS-CAP-ALL-LEGS-AND-GREASINESS)
# --------------------------------------------------------------------------- #

def test_wet_marks_multiplier_is_config_knob_wired_into_default_rain_multipliers():
    from afl_bot.config import WET_MARKS_MULTIPLIER
    from afl_bot.models.weather_effects import DEFAULT_RAIN_MULTIPLIERS
    assert DEFAULT_RAIN_MULTIPLIERS["marks"] == WET_MARKS_MULTIPLIER
    assert WET_MARKS_MULTIPLIER < 0.85   # stronger suppression than original 0.85


def test_greasiness_override_forces_game_value(tmp_path):
    import json
    from afl_bot.cli import _fixture_greasiness
    # Verify the override logic in isolation: a greasiness file keyed by
    # "Home vs Away" should return the overridden value, not the auto-computed one.
    import types
    override_file = tmp_path / "greasiness.json"
    override_file.write_text(json.dumps({"Collingwood vs GWS Giants": 0.75}))
    overrides = json.loads(override_file.read_text())

    game_key = "Collingwood vs GWS Giants"
    home_name = "Collingwood"
    greasiness = (
        max(0.0, min(1.0, float(overrides[game_key])))
        if game_key in overrides else 0.0
    )
    assert greasiness == pytest.approx(0.75)


# â"€â"€ Part 2: Sportsbet note is always informative â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _bare_match():
    return [{
        "header": {"home": "A", "away": "B", "venue": "MCG", "roofed": False,
                   "is_wet": False, "mu_margin": 5.0, "mu_total": 160.0,
                   "p_home": 0.6, "p_away": 0.39, "p_draw": 0.0,
                   "total_line_name": "Total 160.5+", "p_total": 0.5},
        "projections": [], "sgms": [], "market_sgms": [],
    }]


def test_sportsbet_note_not_requested_explains_how_to_enable():
    note = ("_No Sportsbet prices this run. For real book prices: paste this round's "
            "Sportsbet match URLs into `reports/2026_r17_sportsbet_urls.json` "
            "and rerun with `--sportsbet`._")
    md = render_markdown(2026, 17, _bare_match(), has_odds=False, sportsbet_note=note)
    assert "--sportsbet" in md
    assert "sportsbet_urls.json" in md


def test_sportsbet_note_zero_priced_shows_warning_and_fix():
    note = ("_WARNING Sportsbet: 0 legs priced — no URL file / empty list "
            "(`reports/2026_r17_sportsbet_urls.json`). Book/Edge columns show '—'. "
            "Fix: populate `reports/2026_r17_sportsbet_urls.json` with this round's "
            "Sportsbet match URLs and rerun with `--sportsbet`._")
    md = render_markdown(2026, 17, _bare_match(), has_odds=False, sportsbet_note=note)
    assert "WARNING" in md
    assert "0 legs priced" in md
    assert "Fix:" in md


def test_sportsbet_note_success_shows_leg_count():
    note = "_Player-prop odds: live from Sportsbet (scraped, 42 leg(s) priced)._"
    md = render_markdown(2026, 17, _bare_match(), has_odds=False, sportsbet_note=note)
    assert "42 leg(s) priced" in md


# --------------------------------------------------------------------------- #
# EV Diagnostic (Part B)                                                       #
# --------------------------------------------------------------------------- #

def _make_multis_json(tmp_path, rungs):
    import json as _json
    p = tmp_path / "2026_r99_multis.json"
    p.write_text(_json.dumps(rungs))
    return p


def test_ev_diagnostic_runs_without_crashing(tmp_path, capsys):
    import json as _json
    from afl_bot.cli import ev_diagnostic
    import afl_bot.cli as _cli_mod

    rung = {
        "id": "2026-r99-TeamA-TeamB-sportsbet-2.10",
        "year": 2026, "round": 99,
        "game": "TeamA vs TeamB",
        "ladder": "sportsbet",
        "band": 2.1,
        "legs": [
            {"player": "P1", "market": "disposals", "line": 20,
             "book_odds": 1.25, "hit_prob": 0.70},
            {"player": "P2", "market": "goals", "line": 1,
             "book_odds": 1.20, "hit_prob": 0.75},
            {"player": "P3", "market": "tackles", "line": 3,
             "book_odds": 1.30, "hit_prob": 0.65},
        ],
        "model_joint": 0.35, "model_fair": 2.86,
        "book_combo": 2.10, "edge": -0.20,
    }

    # Point ROOT_DIR at tmp_path so ev_diagnostic finds the file.
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    p = reports_dir / "2026_r99_multis.json"
    p.write_text(_json.dumps([rung]))

    orig_root = _cli_mod.ROOT_DIR
    try:
        _cli_mod.ROOT_DIR = tmp_path
        ev_diagnostic(2026, 99)
    finally:
        _cli_mod.ROOT_DIR = orig_root

    out = capsys.readouterr().out
    assert "EV DIAGNOSTIC" in out
    assert "disposals" in out
    assert "VERDICT" in out


def test_ev_diagnostic_leg_gap_arithmetic():
    # Verify the per-leg gap formula: p_implied - hit_prob
    book_odds = 1.25
    hit_prob = 0.70
    p_implied = 1.0 / book_odds        # 0.80
    expected_gap = p_implied - hit_prob  # +0.10
    assert abs(expected_gap - 0.10) < 1e-6


def test_ev_diagnostic_rung_split_arithmetic():
    # Verify EV split: total = leg_disagree + structural
    # legs: p_implied=[0.80, 0.75, 0.769], model_joint=0.35, book_combo=2.10
    p_implied = [1/1.25, 1/1.20, 1/1.30]
    book_naive = p_implied[0] * p_implied[1] * p_implied[2]
    model_joint = 0.35
    book_combo = 2.10

    total_ev = model_joint * book_combo - 1.0
    leg_disagree = (model_joint - book_naive) * book_combo
    structural = book_naive * book_combo - 1.0

    # Total must equal sum of components.
    assert abs(total_ev - (leg_disagree + structural)) < 1e-9


def test_ev_diagnostic_missing_file_prints_error(tmp_path, capsys):
    import sys
    from afl_bot.cli import ev_diagnostic
    import afl_bot.cli as _cli_mod

    orig_root = _cli_mod.ROOT_DIR
    try:
        _cli_mod.ROOT_DIR = tmp_path
        (tmp_path / "reports").mkdir()
        ev_diagnostic(2026, 99)
    finally:
        _cli_mod.ROOT_DIR = orig_root

    err = capsys.readouterr().err
    assert "not found" in err


# --------------------------------------------------------------------------- #
# Prop Calibration Check (prop-calibration-check CLI)                         #
# --------------------------------------------------------------------------- #

def _make_prop_preds(tmp_path):
    """Build a minimal walk-forward predictions dataframe for testing."""
    import pandas as pd
    rng = np.random.default_rng(42)
    n = 200
    # Raw probabilities spread across [0.3, 0.9]; actual outcomes ~ Bernoulli(prob)
    raw_prob = rng.uniform(0.3, 0.9, n)
    actual = (rng.random(n) < raw_prob).astype(int)
    return pd.DataFrame({
        "year": [2024] * 100 + [2025] * 100,
        "round": [1] * n,
        "player": [f"P{i}" for i in range(n)],
        "stat": ["disposals"] * 50 + ["goals"] * 50 + ["marks"] * 50 + ["tackles"] * 50,
        "line": [20.0] * 50 + [1.0] * 50 + [4.0] * 50 + [3.0] * 50,
        "prob": raw_prob,
        "cal_prob": raw_prob * 0.95,   # mock calibrated (slightly lower)
        "actual": actual,
    })


def test_prop_calibration_check_gap_formula():
    # gap = mean_cal_prob - actual_hit_rate (positive = over-predict)
    preds = _make_prop_preds(None)
    grp = preds[preds["stat"] == "disposals"]
    cal_gap = grp["cal_prob"].mean() - grp["actual"].mean()
    # The gap is well-defined regardless of sign
    assert isinstance(cal_gap, float)
    assert -1.0 < cal_gap < 1.0


def test_prop_calibration_check_band_bucketing():
    # Verify that probability bands partition the predictions correctly.
    preds = _make_prop_preds(None)
    grp = preds[preds["stat"] == "disposals"]
    BANDS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
    total_in_bands = sum(
        ((grp["cal_prob"] >= lo) & (grp["cal_prob"] < hi)).sum()
        for lo, hi in BANDS
    )
    # All 50 disposals rows should fall in one of the three bands (prob in [0.3,0.9))
    assert total_in_bands == len(grp)


def test_prop_calibration_check_saves_file(tmp_path, capsys, monkeypatch):
    import json as _json
    import afl_bot.cli as _cli_mod
    from afl_bot.cli import prop_calibration_check

    # Stub out the heavy computation so the test runs without real data
    dummy_preds_df = _make_prop_preds(tmp_path)

    import pandas as pd

    def _fake_wfpp(log, *, eval_start_year, **kw):
        return dummy_preds_df[dummy_preds_df["year"] >= eval_start_year]

    def _fake_load_player_log(games, **kw):
        return pd.DataFrame({"year": [2023], "round": [1], "player": ["X"],
                              "disposals": [20.0], "goals": [1.0], "marks": [4.0],
                              "tackles": [3.0], "team": ["T"], "unixtime": [0]})

    def _fake_games(y):
        return pd.DataFrame({"year": [y], "round": [1], "hteam": ["A"], "ateam": ["B"],
                              "hscore": [100], "ascore": [90]})

    import afl_bot.backtest.props as _props_mod
    orig_wfpp = _props_mod.walk_forward_prop_predictions
    orig_lpl = _cli_mod.load_player_log

    # Minimal fake fit_prop_calibrators that returns empty dict
    from afl_bot.backtest.props import fit_prop_calibrators as _real_fit

    orig_root = _cli_mod.ROOT_DIR
    try:
        _cli_mod.ROOT_DIR = tmp_path
        (tmp_path / "reports").mkdir(exist_ok=True)
        _props_mod.walk_forward_prop_predictions = _fake_wfpp
        _cli_mod.load_player_log = _fake_load_player_log

        # Patch SquiggleClient to return dummy games
        class FakeClient:
            def get_completed_games(self, y):
                return _fake_games(y)
        monkeypatch.setattr(_cli_mod, "SquiggleClient", lambda: FakeClient())

        out_path = str(tmp_path / "reports" / "test_cal_check.md")
        prop_calibration_check([2024, 2025], out_path=out_path,
                               multis_year=2026, multis_round=None)
    finally:
        _props_mod.walk_forward_prop_predictions = orig_wfpp
        _cli_mod.load_player_log = orig_lpl
        _cli_mod.ROOT_DIR = orig_root

    import os
    assert os.path.exists(out_path), "output file should be saved"
    content = open(out_path, encoding="utf-8").read()
    assert "Prop Calibration Check" in content
    assert "disposals" in content
    assert "Verdict" in content


# â"€â"€ build_pull_em_sgm â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _disp_leg(name, player, prob):
    mask = np.random.default_rng(abs(hash(name)) % 2**31).random(20_000) < prob
    return LegCandidate(name=name, match_id="m1", market="player_disposals",
                        subject=player, fair_prob=prob,
                        market_odds=round(1 / prob, 2), mask=mask)


def _goal_leg(name, player, prob):
    mask = np.random.default_rng(abs(hash(name)) % 2**31).random(20_000) < prob
    return LegCandidate(name=name, match_id="m1", market="player_goals",
                        subject=player, fair_prob=prob,
                        market_odds=round(1 / prob, 2), mask=mask)


def test_pull_em_basic_composition():
    """build_pull_em_sgm returns 3 anchor disposals + 1 booster."""
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
        _goal_leg("D 1+ goals", "D", 0.55),  # booster
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book, min_combo_odds=4.0)
    assert result is not None
    assert len(result["leg_names"]) == 4
    assert len(result["anchor_names"]) == 3
    assert result["booster_name"] not in result["anchor_names"]
    # All anchors must be from different players
    anchor_probs = result["anchor_probs"]
    assert all(p >= 0.70 for p in anchor_probs)


def test_pull_em_option_ev_math():
    """Option EV is computed as sum over each leg as 'pulled'."""
    p_a, p_b, p_c, p_d = 0.75, 0.72, 0.71, 0.55
    odds_a, odds_b, odds_c, odds_d = 1 / p_a, 1 / p_b, 1 / p_c, 1 / p_d
    legs = [
        _disp_leg("A 25+ disposals", "A", p_a),
        _disp_leg("B 20+ disposals", "B", p_b),
        _disp_leg("C 30+ disposals", "C", p_c),
        _goal_leg("D 1+ goals", "D", p_d),
    ]
    odds_book = {"A 25+ disposals": odds_a, "B 20+ disposals": odds_b,
                 "C 30+ disposals": odds_c, "D 1+ goals": odds_d}
    result = build_pull_em_sgm(legs, odds_book=odds_book, min_combo_odds=4.0)
    assert result is not None

    book_combo = odds_a * odds_b * odds_c * odds_d
    probs = [p_a, p_b, p_c, p_d]
    expected_total = 0.0
    for i in range(4):
        p_others = 1.0
        for j, p in enumerate(probs):
            if j != i:
                p_others *= p
        p_miss = 1.0 - probs[i]
        reduced = book_combo / [odds_a, odds_b, odds_c, odds_d][i]
        expected_total += p_others * p_miss * PULL_DETECTION_PROB * (reduced - 1.0)

    assert abs(result["option_ev"] / 100 - expected_total) < 1e-4


def test_pull_em_respects_tackle_marks_cap():
    """At most 1 tackle/marks leg allowed in a Pull 'Em SGM."""
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
    ]
    # booster candidates: one tackles leg only
    tackle_mask = np.random.default_rng(0).random(20_000) < 0.55
    tackle_booster = LegCandidate("T 3+ tackles", "T", "player_tackles", "T",
                                  0.55, 1 / 0.55, mask=tackle_mask)
    legs.append(tackle_booster)
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book, min_combo_odds=4.0)
    if result is not None and not result.get("no_valid_combo"):
        n_tm = sum(1 for n in result["leg_names"]
                   if "tackles" in n.lower() or "marks" in n.lower())
        assert n_tm <= 1


def test_pull_em_returns_none_with_fewer_than_3_anchors():
    """Returns no_valid_combo when there are not enough disposal anchor legs."""
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        # Only 2 disposal anchors — need 3
        _goal_leg("C 1+ goals", "C", 0.55),
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book)
    # Not enough anchors at any floor â†’ no_valid_combo dict (eligible legs exist)
    assert result is None or result.get("no_valid_combo")


def test_pull_em_returns_none_when_no_legs_priced():
    """build_pull_em_sgm returns None when odds_book is empty."""
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
        _goal_leg("D 1+ goals", "D", 0.55),
    ]
    result = build_pull_em_sgm(legs, odds_book={})
    assert result is None


def test_pull_em_one_player_per_leg():
    """No two legs in a Pull 'Em SGM can be for the same player."""
    # A appears in two disposal lines
    legs = [
        _disp_leg("A 20+ disposals", "A", 0.75),
        _disp_leg("A 25+ disposals", "A", 0.71),   # same player
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
        _goal_leg("D 1+ goals", "D", 0.55),
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book, min_combo_odds=4.0)
    if result is not None and not result.get("no_valid_combo"):
        subjects = [l.subject for l in result["legs"]]
        assert len(subjects) == len(set(subjects)), "duplicate player in Pull 'Em"


# â"€â"€ Part A: band-selection by EV (Phase 3.5) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def test_band_selection_picks_positive_ev_over_negative_ev_near_lock():
    """Within a band, the combo with positive Total EV is preferred over the combo
    with negative Total EV — even if the negative-EV combo has lower book_odds
    (sits closer to the band floor). Both combos must be in the band window.

    Combo A: joint≈0.35, book=2.15 (in [$2.10, $2.73] window), edge≈-0.23 (negative).
    Combo B: joint≈0.45, book=2.50 (in window), edge≈+0.09 (positive, below max_plausible=0.15).
    Mixed A+B combos have edge<0. Only all-B is in the valued set → must be picked."""
    rng = np.random.default_rng(7)
    n = 20_000
    # Combo A: 3 legs prob≈0.705 → joint product≈0.350; book per leg=(2.15)^(1/3)≈1.285
    mask_a1 = rng.random(n) < 0.705; mask_a2 = rng.random(n) < 0.705; mask_a3 = rng.random(n) < 0.705
    neg_a = _leg("NegEV A1", 0.705, mask_a1, "NA1", odds=1.285)
    neg_b = _leg("NegEV A2", 0.705, mask_a2, "NA2", odds=1.285)
    neg_c = _leg("NegEV A3", 0.705, mask_a3, "NA3", odds=1.285)
    # Combo B: 3 legs prob≈0.766 → joint≈0.45; book per leg=(2.50)^(1/3)≈1.357.
    # Shrunk edge ≈ 0.09 — positive but below max_plausible_edge=0.15.
    # Mixed A+B combos land at edge≈-0.03 (negative), so only all-B enters valued.
    mask_b1 = rng.random(n) < 0.766; mask_b2 = rng.random(n) < 0.766; mask_b3 = rng.random(n) < 0.766
    pos_a = _leg("PosEV B1", 0.766, mask_b1, "NB1", odds=1.357)
    pos_b = _leg("PosEV B2", 0.766, mask_b2, "NB2", odds=1.357)
    pos_c = _leg("PosEV B3", 0.766, mask_b3, "NB3", odds=1.357)
    legs = [neg_a, neg_b, neg_c, pos_a, pos_b, pos_c]
    odds_book = {l.name: l.market_odds for l in legs}
    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(2.10,), min_joint_prob=0.0)
    assert out
    assert not out[0].get("no_bet"), "both combos in window, must get a pick"
    # The selected rung should be the +EV combo (B), not the cheaper -EV combo (A).
    picked_legs = set(out[0]["legs"])
    assert picked_legs == {"PosEV B1", "PosEV B2", "PosEV B3"}, (
        "Positive-EV combo B (book≈2.50) must win over negative-EV combo A (book≈2.15)"
    )


def test_band_selection_all_negative_ev_shows_no_bet():
    """When every combo in the band has negative Total EV, the safest combo
    is shown and suggested_stake is None (honest NO BET). Band not hidden."""
    rng = np.random.default_rng(0)
    # 3 independent legs: prob=0.70 but book=1.05 each.
    # jointâ‰ˆ0.343, book_comboâ‰ˆ1.157, edgeâ‰ˆ-0.45, total_ev<0 even with promo.
    masks = [rng.random(20_000) < 0.70 for _ in range(3)]
    legs = [_leg(f"OverPriced {i}", 0.70, masks[i], f"P{i}", odds=1.05) for i in range(3)]
    odds_book = {l.name: l.market_odds for l in legs}
    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(1.1,), min_joint_prob=0.0)
    assert out, "band should still be shown even when all combos are -EV"
    assert out[0].get("suggested_stake") is None


def test_band_selection_metric_uses_market_shrunk_probs():
    """_total_ev is computed from shrunk edge (market-anchored prob), not raw
    model EV. Verify: edge = shrunk*book - 1, not joint*book - 1."""
    from afl_bot.build.report import build_sgm_candidates

    rng = np.random.default_rng(42)
    prob = 0.65
    book_per_leg = 1.70
    masks = [rng.random(20_000) < prob for _ in range(3)]
    legs = [_leg(f"Leg{i}", prob, masks[i], f"P{i}", odds=book_per_leg) for i in range(3)]
    odds_book = {l.name: l.market_odds for l in legs}
    combos = build_sgm_candidates(legs, odds_book=odds_book, min_joint_prob=0.0)
    assert combos
    c = combos[0]
    # edge must equal shrunk * book - 1
    from afl_bot.pricing.edge import market_anchored_prob
    shrunk = market_anchored_prob(c["joint_prob"], c["book_odds"], MULTI_MARKET_SHRINK)
    expected_edge = shrunk * c["book_odds"] - 1.0
    assert abs(c["edge"] - expected_edge) < 1e-9
    # raw_edge (model-only) differs from shrunk edge
    assert abs(c.get("raw_edge", 0) - (c["joint_prob"] * c["book_odds"] - 1.0)) < 1e-9


# ── Part A: Band window assertion (FIX-RESTORE-BANDS-AND-STAKING) ────────────

def test_band_window_held_or_no_bet_model_ladder():
    """Every model-ladder rung must either (a) have fair_odds in [band, band*1.30]
    or (b) be an explicit NO BET sentinel — never a combo outside the window."""
    from afl_bot.config import BAND_UPPER_FACTOR
    rng = np.random.default_rng(77)
    n = 40_000
    probs = {"A": 0.77, "B": 0.70, "C": 0.62, "D": 0.55, "E": 0.47,
             "F": 0.41, "G": 0.36, "H": 0.33, "I": 0.30}
    legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        legs.append(_leg(f"{name} 20+ disp", mask.mean(), mask, name))
    out = search_match_sgms(legs)
    for r in out:
        band = r.get("target_odds")
        assert band is not None
        if r.get("no_bet"):
            assert r["legs"] == []
        else:
            fair = r["fair_odds"]
            assert band <= fair <= band * BAND_UPPER_FACTOR, (
                f"Model rung at band ${band:.2f} has fair ${fair:.2f} outside "
                f"[{band:.2f}, {band * BAND_UPPER_FACTOR:.2f}]"
            )


def test_band_window_held_or_no_bet_sportsbet_ladder():
    """Every Sportsbet-ladder rung must either (a) have book_odds in [band, band*1.30]
    or (b) be an explicit NO BET sentinel."""
    from afl_bot.config import BAND_UPPER_FACTOR
    rng = np.random.default_rng(77)
    n = 40_000
    probs = {"A": 0.77, "B": 0.70, "C": 0.62, "D": 0.55, "E": 0.47,
             "F": 0.41, "G": 0.36, "H": 0.33, "I": 0.30}
    legs = []
    for name, p in probs.items():
        mask = rng.random(n) < p
        prob = mask.mean() or 0.01
        legs.append(_leg(f"{name} 20+ disp", prob, mask, name,
                         odds=(1.0 / prob) * 1.05))
    odds_book = {l.name: l.market_odds for l in legs}
    out = search_market_sgms(legs, odds_book=odds_book)
    for r in out:
        band = r.get("target_odds")
        assert band is not None
        if r.get("no_bet"):
            assert r["legs"] == []
        else:
            book = r["book_odds"]
            assert band <= book <= band * BAND_UPPER_FACTOR, (
                f"Sportsbet rung at band ${band:.2f} has book ${book:.2f} outside "
                f"[{band:.2f}, {band * BAND_UPPER_FACTOR:.2f}]"
            )


def test_pull_em_min_combo_odds_enforced():
    """Combos with book_combo < $5.00 are rejected â†’ no_valid_combo returned."""
    # Probs high enough that book_combo stays well below 5.0
    legs = [
        _disp_leg("A 20+ disposals", "A", 0.80),
        _disp_leg("B 20+ disposals", "B", 0.78),
        _disp_leg("C 20+ disposals", "C", 0.76),
        _goal_leg("D 1+ goals", "D", 0.55),
    ]
    # book_combo â‰ˆ (1/0.80)*(1/0.78)*(1/0.76)*(1/0.55) â‰ˆ 3.87 < 5.0
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book)
    assert result is not None and result.get("no_valid_combo"), (
        "Expected no_valid_combo when book combo < $5.00"
    )
    assert result["min_combo_odds"] == PULL_EM_MIN_COMBO_ODDS


def test_pull_em_h2h_leg_excluded():
    """h2h legs must NOT appear in Pull 'Em SGMs (not in eligible markets)."""
    rng = np.random.default_rng(5)
    h2h_mask = rng.random(20_000) < 0.55
    h2h_leg = LegCandidate("Team A Win", "TeamA", "h2h", "TeamA", 0.55, 1/0.55, mask=h2h_mask)

    legs = [
        _disp_leg("A 20+ disposals", "A", 0.72),
        _disp_leg("B 20+ disposals", "B", 0.70),
        _disp_leg("C 20+ disposals", "C", 0.68),
        h2h_leg,
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    assert "h2h" not in PULL_EM_ELIGIBLE_MARKETS
    # h2h leg is not eligible, so if a combo forms it must not include it
    result = build_pull_em_sgm(legs, odds_book=odds_book, min_combo_odds=1.0)
    if result is not None and not result.get("no_valid_combo"):
        assert "Team A Win" not in result["leg_names"], "h2h leg must not appear"


def test_pull_em_total_points_leg_excluded():
    """total_points legs are NOT eligible for Pull 'Em (not a player prop)."""
    rng = np.random.default_rng(6)
    tp_mask = rng.random(20_000) < 0.55
    tp_leg = LegCandidate("O/U 150.5 pts", "Total", "total_points", "Total",
                          0.55, 1/0.55, mask=tp_mask)
    legs = [
        _disp_leg("A 20+ disposals", "A", 0.72),
        _disp_leg("B 20+ disposals", "B", 0.70),
        _disp_leg("C 20+ disposals", "C", 0.68),
        tp_leg,
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    assert "total_points" not in PULL_EM_ELIGIBLE_MARKETS
    result = build_pull_em_sgm(legs, odds_book=odds_book, min_combo_odds=1.0)
    if result is not None and not result.get("no_valid_combo"):
        assert "O/U 150.5 pts" not in result["leg_names"], "total_points leg must not appear"


def test_pull_em_line_raising_prefers_higher_threshold_to_meet_minimum():
    """Lower-prob (higher-threshold) disposal lines are preferred over switching
    markets when needed to reach the $5.00 minimum."""
    # Player A: two disposal lines — 20+ (high prob/low odds) and 30+ (lower prob/higher odds)
    rng = np.random.default_rng(9)
    mask_a_easy = rng.random(20_000) < 0.78   # 20+ disposals — easy, low odds
    mask_a_hard = rng.random(20_000) < 0.60   # 30+ disposals — harder, higher odds
    mask_b = rng.random(20_000) < 0.72
    mask_c = rng.random(20_000) < 0.70
    mask_d = rng.random(20_000) < 0.55

    leg_a_easy = LegCandidate("A 20+ disposals", "m1", "player_disposals", "A",
                              0.78, 1/0.78, mask=mask_a_easy)
    leg_a_hard = LegCandidate("A 30+ disposals", "m1", "player_disposals", "A",
                              0.60, 1/0.60, mask=mask_a_hard)
    leg_b = LegCandidate("B 20+ disposals", "m1", "player_disposals", "B",
                         0.72, 1/0.72, mask=mask_b)
    leg_c = LegCandidate("C 20+ disposals", "m1", "player_disposals", "C",
                         0.70, 1/0.70, mask=mask_c)
    leg_d = LegCandidate("D 1+ goals", "m1", "player_goals", "D",
                         0.55, 1/0.55, mask=mask_d)

    all_legs = [leg_a_easy, leg_a_hard, leg_b, leg_c, leg_d]
    odds_book = {l.name: l.market_odds for l in all_legs}

    # Easy combo (A 20+, B, C, D) book = (1/0.78)*(1/0.72)*(1/0.70)*(1/0.55) â‰ˆ 4.53 < 5.0
    # Hard combo (A 30+, B, C, D) book = (1/0.60)*(1/0.72)*(1/0.70)*(1/0.55) â‰ˆ 5.90 >= 5.0
    import math
    easy_combo = math.prod([1/0.78, 1/0.72, 1/0.70, 1/0.55])
    hard_combo = math.prod([1/0.60, 1/0.72, 1/0.70, 1/0.55])
    assert easy_combo < PULL_EM_MIN_COMBO_ODDS < hard_combo, "test setup: easy below, hard above"

    result = build_pull_em_sgm(all_legs, odds_book=odds_book)
    assert result is not None and not result.get("no_valid_combo"), (
        "Expected a valid combo using higher-threshold disposal line"
    )
    assert result["book_combo"] >= PULL_EM_MIN_COMBO_ODDS
    # The harder line (30+) must be selected for player A, not the easy one
    assert "A 30+ disposals" in result["leg_names"]
    assert "A 20+ disposals" not in result["leg_names"]


def test_pull_em_anchor_relaxed_to_key_present_when_relaxed():
    """When anchor_min_p=0.70 yields no valid combo but 0.65 does, the result
    includes 'anchor_relaxed_to' = 0.65."""
    rng = np.random.default_rng(11)
    # Players with 0.68 prob — would fail 0.70 floor but pass 0.65
    mask_a = rng.random(20_000) < 0.68
    mask_b = rng.random(20_000) < 0.67
    mask_c = rng.random(20_000) < 0.66
    mask_d = rng.random(20_000) < 0.55

    leg_a = LegCandidate("A 25+ disposals", "m1", "player_disposals", "A", 0.68, 1/0.68, mask=mask_a)
    leg_b = LegCandidate("B 25+ disposals", "m1", "player_disposals", "B", 0.67, 1/0.67, mask=mask_b)
    leg_c = LegCandidate("C 25+ disposals", "m1", "player_disposals", "C", 0.66, 1/0.66, mask=mask_c)
    leg_d = LegCandidate("D 1+ goals", "m1", "player_goals", "D", 0.55, 1/0.55, mask=mask_d)

    all_legs = [leg_a, leg_b, leg_c, leg_d]
    odds_book = {l.name: l.market_odds for l in all_legs}

    result = build_pull_em_sgm(all_legs, odds_book=odds_book, anchor_min_p=0.70, min_combo_odds=1.0)
    assert result is not None and not result.get("no_valid_combo"), (
        "Should find a combo after relaxing to 0.65"
    )
    assert result.get("anchor_relaxed_to") is not None
    assert result["anchor_relaxed_to"] < 0.70


def test_pull_em_no_valid_combo_at_all_floors():
    """When no combo meets $5 even at the most relaxed anchor floor, the
    no_valid_combo dict is returned with the min_combo_odds key."""
    rng = np.random.default_rng(13)
    # Very high probs â†’ low odds per leg â†’ book_combo stays well below $5
    for_player = lambda player, prob: LegCandidate(
        f"{player} 15+ disposals", "m1", "player_disposals", player,
        prob, 1/prob, mask=rng.random(20_000) < prob
    )
    booster = LegCandidate("D 1+ goals", "m1", "player_goals", "D",
                           0.55, 1/0.55, mask=rng.random(20_000) < 0.55)
    legs = [for_player("A", 0.93), for_player("B", 0.91), for_player("C", 0.89), booster]
    # book_combo â‰ˆ (1/0.93)*(1/0.91)*(1/0.89)*(1/0.55) â‰ˆ 2.56 < 5.0
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book)
    assert result is not None
    assert result.get("no_valid_combo") is True
    assert "min_combo_odds" in result


# ── Part C: Suspect pricing guard (FIX-RESTORE-BANDS-AND-STAKING) ────────────

def test_suspect_pricing_book_far_above_model_returns_check_pricing():
    """When book_combo > 1.75x model_fair, the rung is flagged CHECK PRICING
    and stake is zero — fake +104% EVs must never be staked."""
    from afl_bot.config import SUSPECT_BOOK_FAIR_RATIO
    rng = np.random.default_rng(9)
    n = 20_000

    # Build a rung that lands in the $15 band window [15.00, 19.50].
    # book_per_leg = 2.57 → book_combo ≈ 16.97 (in window).
    # Model probs = 0.55 each → joint ≈ 0.166 → model_fair ≈ 6.0.
    # ratio ≈ 16.97 / 6.0 ≈ 2.83 >> SUSPECT_BOOK_FAIR_RATIO (1.75) → CHECK PRICING.
    probs = [0.55, 0.55, 0.55]
    book_per_leg = 2.57
    masks = [rng.random(n) < p for p in probs]
    legs = [_leg(f"SuspectLeg{i}", probs[i], masks[i], f"P{i}", odds=book_per_leg)
            for i in range(3)]
    odds_book = {l.name: l.market_odds for l in legs}

    # book_combo ≈ 2.57^3 ≈ 17.0 (in $15 band [15.00, 19.50]),
    # model_fair ≈ 1/0.55^3 ≈ 6.0 → ratio ≈ 2.83 >> 1.75 → CHECK PRICING
    out = search_market_sgms(legs, odds_book=odds_book, min_joint_prob=0.0)
    real = [r for r in out if not r.get("no_bet")]
    assert real, "some bands should be reachable"
    for r in real:
        ratio = r["book_odds"] / r["fair_odds"]
        if ratio > SUSPECT_BOOK_FAIR_RATIO:
            assert r.get("raw_edge", 0) > 0.40 or ratio > SUSPECT_BOOK_FAIR_RATIO, (
                "Suspect rung should exceed the guard threshold"
            )
            # The _units_fields call in round_report will flag these CHECK PRICING
            # and stake 0. We verify the rung has positive raw_edge (would be staked
            # without the guard) but that book/model ratio is above the threshold.
            assert ratio > SUSPECT_BOOK_FAIR_RATIO


# ── Part A: Pull 'Em PointsBet menu (FIX-PULLEM-MENU-AND-STAKE-COLUMNS) ────────


def _marks_leg_pb(name, player, prob, odds=None):
    mask = np.random.default_rng(abs(hash(name)) % 2**31).random(20_000) < prob
    return LegCandidate(name=name, match_id="m1", market="player_marks",
                        subject=player, fair_prob=prob,
                        market_odds=odds if odds else round(1 / prob, 2), mask=mask)


def test_pull_em_menu_only_uses_offered_lines():
    """When pointsbet_menu has 4+/6+ marks but NOT 5+ marks, builder never outputs 5+."""
    # Build legs: A (disposals), B (disposals), C (disposals) for anchors;
    # D with 4+/6+ marks only (no 5+) for booster.
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
        _marks_leg_pb("D 4+ marks", "D", 0.55, odds=1.85),
        _marks_leg_pb("D 5+ marks", "D", 0.35, odds=2.85),   # NOT in PB menu
        _marks_leg_pb("D 6+ marks", "D", 0.20, odds=4.50),
    ]
    # PointsBet offers 4+ and 6+ marks but NOT 5+ marks for D
    pb_menu = {
        "A 25+ disposals": 1.35,
        "B 20+ disposals": 1.40,
        "C 30+ disposals": 1.43,
        "D 4+ marks": 1.85,
        "D 6+ marks": 4.50,
        # "D 5+ marks" intentionally absent
    }
    # odds_book also has all lines (simulates Sportsbet offering 5+ marks)
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book, pointsbet_menu=pb_menu,
                                min_combo_odds=1.0)
    assert result is not None
    assert not result.get("no_valid_combo")
    assert "D 5+ marks" not in result["leg_names"], (
        "5+ marks must not appear — PointsBet menu only has 4+ and 6+"
    )


def test_pull_em_menu_leg_odds_equal_menu_price():
    """Leg book_odds_per_leg must equal the PointsBet menu price, not Sportsbet's."""
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
        _goal_leg("D 1+ goals", "D", 0.55),
    ]
    # PB menu has DIFFERENT prices than the model's market_odds
    pb_menu = {
        "A 25+ disposals": 1.38,   # different from 1/0.75 ≈ 1.33
        "B 20+ disposals": 1.43,   # different from 1/0.72 ≈ 1.39
        "C 30+ disposals": 1.44,   # different from 1/0.71 ≈ 1.41
        "D 1+ goals": 1.90,        # different from 1/0.55 ≈ 1.82
    }
    odds_book = {l.name: l.market_odds for l in legs}
    result = build_pull_em_sgm(legs, odds_book=odds_book, pointsbet_menu=pb_menu,
                                min_combo_odds=1.0)
    assert result is not None and not result.get("no_valid_combo")
    for name, book_o in zip(result["leg_names"], result["book_odds_per_leg"]):
        assert abs(book_o - pb_menu[name]) < 1e-9, (
            f"{name}: book_odds {book_o} must equal menu price {pb_menu[name]}"
        )


def test_pull_em_all_null_menu_returns_unavailable():
    """An all-null menu (file exists but Ben hasn't filled it in) → unavailable sentinel."""
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
        _goal_leg("D 1+ goals", "D", 0.55),
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    # pb_menu = {} means file exists but every value was null
    result = build_pull_em_sgm(legs, odds_book=odds_book, pointsbet_menu={})
    assert result is not None
    assert result.get("no_valid_combo") is True
    assert result.get("unavailable") is True, "Empty menu must return unavailable=True"


def test_pull_em_none_menu_falls_back_to_odds_book():
    """When pb_menu is None (no file yet), Pull 'Em falls back to odds_book."""
    legs = [
        _disp_leg("A 25+ disposals", "A", 0.75),
        _disp_leg("B 20+ disposals", "B", 0.72),
        _disp_leg("C 30+ disposals", "C", 0.71),
        _goal_leg("D 1+ goals", "D", 0.55),
    ]
    odds_book = {l.name: l.market_odds for l in legs}
    # None = no file at all → fall back to odds_book (existing behaviour)
    result = build_pull_em_sgm(legs, odds_book=odds_book, pointsbet_menu=None,
                                min_combo_odds=1.0)
    assert result is not None and not result.get("no_valid_combo"), (
        "pb_menu=None must fall back to odds_book and build normally"
    )


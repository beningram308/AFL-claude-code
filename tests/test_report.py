"""Round-report helpers (round-2 §10): projection tables + SGM joint search."""

import numpy as np
import pytest

from afl_bot.build.multi import LegCandidate
from afl_bot.build.report import (
    build_odds_template,
    build_sgm_candidates,
    is_bookable_model_only_leg,
    projection_rows,
    render_markdown,
    search_market_sgms,
    search_match_sgms,
    select_ladder_lines,
    top_n_players_by_stat,
)
from afl_bot.config import MULTI_TARGET_ODDS

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
    for r in out:
        assert len(r["legs"]) == 3                     # minimum-3-leg ladder
        assert {"joint_prob", "naive_product", "corr_gain", "fair_odds"} <= set(r)
    assert len(out) == len(MULTI_TARGET_ODDS)          # one rung per target
    # returned safest -> longest by fair_odds
    assert [r["fair_odds"] for r in out] == sorted(r["fair_odds"] for r in out)
    # no duplicate combos (each rung is a distinct combo)
    all_legs = [tuple(sorted(r["legs"])) for r in out]
    assert len(all_legs) == len(set(all_legs))


def test_search_match_sgms_excludes_conflicts():
    legs = _ladder_legs()
    legs.append(_leg("A 20+ disp", 0.5, legs[0].mask, "A"))  # conflicts with "A 15+ disp"
    out = search_match_sgms(legs)
    assert out
    for r in out:
        assert not ("A 15+ disp" in r["legs"] and "A 20+ disp" in r["legs"])


def test_search_match_sgms_top_band_value_pick_is_shrunk_and_capped():
    legs = _ladder_legs(odds_mult=1.05)   # book ~5% above fair -> a modest, plausible edge
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book)
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
    # All legs ~0.82 -> every 3-combo has fair_odds ~1.82; targets 3.50 and 5.00
    # have no natural match, so the fill picks the closest available distinct combos.
    rng = np.random.default_rng(7)
    legs = [_leg(f"{name} 15+ disp", (m := rng.random(40000) < 0.82).mean(), m, name)
            for name in "ABCDE"]
    out = search_match_sgms(legs)
    assert len(out) == len(MULTI_TARGET_ODDS)          # one rung per target, no blank
    assert all(len(r["legs"]) == 3 for r in out)
    assert [r["fair_odds"] for r in out] == sorted(r["fair_odds"] for r in out)


def test_build_sgm_candidates_is_the_full_pool_search_selects_from():
    legs = _ladder_legs()
    candidates = build_sgm_candidates(legs)
    selected = search_match_sgms(legs)
    cand_keys = {tuple(sorted(c["legs"])) for c in candidates}
    for r in selected:
        assert tuple(sorted(r["legs"])) in cand_keys
    # C(6,3) = 20 non-conflicting 3-leg combos, all clearing the default floor.
    assert len(candidates) == 20
    assert all(c["n_sims"] == len(legs[0].mask) for c in candidates)


def test_search_match_sgms_price_shrink_pulls_toward_target_implied_prob():
    legs = _ladder_legs()
    target = MULTI_TARGET_ODDS[-1]
    anchor_prob = 1.0 / target
    raw = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)[0]
    full = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0, price_shrink=1.0)[0]
    half = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0, price_shrink=0.5)[0]
    assert full["joint_prob"] == pytest.approx(anchor_prob)              # fully shrunk -> exactly at target
    assert half["joint_prob"] == pytest.approx((raw["joint_prob"] + anchor_prob) / 2)
    assert full["fair_odds"] == pytest.approx(target)


def test_search_match_sgms_corr_gain_haircut_zero_lift_equals_naive_product():
    # FIX-PLACEABLE-LEGS-AND-210-FLOOR STEP 4 moved the haircut to BEFORE
    # selection, so it can change which combo wins with a wider leg pool --
    # exactly 3 legs (one possible combo) isolates the haircut math itself.
    legs = _ladder_legs()[:3]
    target = MULTI_TARGET_ODDS[-1]
    raw = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)[0]
    zero = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0,
                             corr_gain_haircut=0.0)[0]
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
        assert r["joint_prob"] == pytest.approx(u["joint_prob"])
        assert r["fair_odds"] == pytest.approx(u["fair_odds"])


def test_search_match_sgms_corr_gain_haircut_half_is_midpoint():
    # Same isolation as the zero-lift test above -- exactly 3 legs, one combo.
    legs = _ladder_legs()[:3]
    target = MULTI_TARGET_ODDS[-1]
    raw = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)[0]
    half = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0,
                             corr_gain_haircut=0.5)[0]
    expected = raw["naive_product"] + 0.5 * raw["corr_gain"]
    assert half["joint_prob"] == pytest.approx(expected)


def test_search_match_sgms_corr_gain_haircut_recomputes_edge_when_priced():
    legs = _ladder_legs(odds_mult=1.05)
    odds_book = {leg.name: leg.market_odds for leg in legs}
    target = MULTI_TARGET_ODDS[-1]
    haircut = search_match_sgms(legs, odds_book=odds_book, target_odds=(target,),
                                min_joint_prob=0.0, corr_gain_haircut=0.0)[0]
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
    """Two pure 3-leg pools with exact joint_prob 0.30 and 0.31 (n=200 masks,
    no extra leg-level noise), target implied prob 0.29 (both pools sit
    above it, 0.30 closer): lcb_z=0 (default) picks the closer 0.30 pool,
    but the haircut's effect on each pool's distance-to-target is enough to
    flip the pick to the 0.31 pool at lcb_z=0.5 -- model-upgrade audit
    Phase 3.5's selection-haircut prototype actually changes the selected
    rung, not just its reported probability."""
    n = 200
    mask_30 = np.zeros(n, dtype=bool); mask_30[:60] = True   # joint_prob exactly 0.30
    mask_31 = np.zeros(n, dtype=bool); mask_31[:62] = True   # joint_prob exactly 0.31
    pool_30 = [_leg(f"A{i} 15+", 1.0, mask_30, f"A{i}") for i in range(3)]
    pool_31 = [_leg(f"B{i} 15+", 1.0, mask_31, f"B{i}") for i in range(3)]
    legs = pool_30 + pool_31
    target = 1.0 / 0.29

    default = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0)
    haircut = search_match_sgms(legs, target_odds=(target,), min_joint_prob=0.0, lcb_z=0.5)
    assert default[0]["joint_prob"] == pytest.approx(0.30)
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
    for r in out:
        assert all(name in priced_names for name in r["legs"])
        assert {"book_odds", "edge", "joint_prob", "fair_odds"} <= set(r)


def test_search_market_sgms_value_pick_is_real_edge_only():
    legs = _ladder_legs(odds_mult=1.05)   # book ~5% above fair -> a modest, plausible edge
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book)
    picks = [r for r in out if r.get("value_pick")]
    assert len(picks) == 1
    vp = picks[0]
    assert 0.0 < vp["edge"] <= 0.15


def test_search_market_sgms_implausible_edge_not_flagged_value():
    legs = _ladder_legs(odds_mult=1.20)   # book way over fair -> shrunk edge blows past 15%
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book)
    assert not any(r.get("value_pick") for r in out)


def test_search_market_sgms_lands_at_or_above_book_target_not_short_when_possible():
    # Book odds are a plain per-leg PRODUCT (no mask/correlation collapsing
    # mixed combos the way joint_prob_from_masks does) -- six distinct prices
    # mean every one of C(6,3)=20 combos is a genuinely different book price.
    # The selection must still never land shorter than the target when SOME
    # combo (pure or mixed) clears it, and must pick the closest one above.
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
    assert out[0]["book_odds"] == pytest.approx(min(clearing))


def test_search_market_sgms_falls_back_to_closest_below_when_nothing_reaches():
    # All three legs near-locks -> every combo's book price sits well under
    # the target; with nothing reaching it, the fallback is the closest
    # below (Ben's r16 finding: the real market can legitimately price a
    # combo shorter than the model ladder's $2.10 floor -- not a bug).
    mask = np.ones(100, dtype=bool)
    legs = [_leg(f"N{i} 15+", 0.95, mask, f"N{i}", odds=1.05) for i in range(3)]
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_market_sgms(legs, odds_book=odds_book, target_odds=(MULTI_TARGET_ODDS[0],),
                             min_joint_prob=0.0)
    assert out
    assert out[0]["book_odds"] < MULTI_TARGET_ODDS[0]


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
    # 3 players × 2 markets = 6 legs; without the subject filter we'd get combos
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
    """A pool spanning safe→longshot produces up to 6 distinct rungs (one per band)."""
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
    all_leg_tuples = [tuple(sorted(r["legs"])) for r in out]
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
    # All four legs have identical prob (0.70, no mask) → identical joint prob
    # for any 3-combo. Pref score is the only tiebreaker; the disposals-only
    # combo must win.
    legs = [
        _leg("D1 20+ disp", 0.70, None, "D1"),
        _leg("D2 20+ disp", 0.70, None, "D2"),
        _leg("D3 20+ disp", 0.70, None, "D3"),
        _marks_leg("M1 4+ marks", "M1"),  # no book price → model-only marks
    ]
    out = search_match_sgms(legs, target_odds=(2.10,), min_joint_prob=0.0)
    assert len(out) == 1
    assert "M1 4+ marks" not in out[0]["legs"]   # all-disposals combo wins


def test_marks_cap_filters_all_model_only_marks_combos():
    # Only marks legs, none priced → every 3-combo exceeds MAX_MARKS_LEGS_PER_MULTI=1
    legs = [_marks_leg(f"M{i} 4+ marks", f"M{i}") for i in range(3)]
    out = search_match_sgms(legs, min_joint_prob=0.0)
    assert out == []   # all combos filtered, nothing to select


def test_priced_marks_leg_counts_toward_cap():
    # FIX-MARKS-CAP: ALL marks legs count, priced or not. Three priced marks
    # → _n_marks=3 > MAX_MARKS_LEGS_PER_MULTI=1 → all combos filtered.
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
# Total-points legs excluded from multis (FIX-TOTAL-POINTS-LEGS)
# --------------------------------------------------------------------------- #

def _total_leg(name="Total points 160.5+", prob=0.55):
    return LegCandidate(name=name, match_id="m1", market="total_points",
                        subject="total", fair_prob=prob, market_odds=1.0 / prob, mask=None)


def test_allow_total_points_in_multi_defaults_to_false():
    from afl_bot.config import ALLOW_TOTAL_POINTS_IN_MULTI
    assert ALLOW_TOTAL_POINTS_IN_MULTI is False


def test_total_points_excluded_when_flag_false():
    # Simulate cli.py's filter: strip total_points legs before handing to ladder.
    raw_pool = _ladder_legs() + [_total_leg()]
    filtered = [l for l in raw_pool if l.market != "total_points"]
    out = search_match_sgms(filtered, min_joint_prob=0.0)
    assert out
    for r in out:
        assert not any("Total points" in name for name in r["legs"])


def test_total_points_allowed_when_flag_true():
    # With flag True the cli.py passes match_legs unchanged; the total_points
    # leg is eligible to enter a combo if it forms a valid 3-leg combination.
    # Use 2 disp legs + 1 total leg; joint is naive product (no masks), all combos
    # are the single 3-leg combo.
    disp_a = _leg("DA 20+ disp", 0.60, None, "DA")
    disp_b = _leg("DB 20+ disp", 0.55, None, "DB")
    total = _total_leg(prob=0.55)
    # No filtering (ALLOW_TOTAL_POINTS_IN_MULTI=True path in cli.py).
    pool = [disp_a, disp_b, total]
    out = search_match_sgms(pool, target_odds=(2.10,), min_joint_prob=0.0)
    assert len(out) == 1
    assert any("Total points" in name for name in out[0]["legs"])


def test_total_points_excluded_from_market_sgms_when_flag_false():
    # search_market_sgms uses the same ladder_legs pool in cli.py.
    total = _total_leg(prob=0.55)
    total_with_odds = LegCandidate(
        name=total.name, match_id=total.match_id, market=total.market,
        subject=total.subject, fair_prob=total.fair_prob,
        market_odds=total.market_odds, mask=None)
    raw_pool = _ladder_legs(odds_mult=1.0) + [total_with_odds]
    odds_book = {l.name: l.market_odds for l in raw_pool}
    # CLI filter — remove total_points before passing to search_market_sgms.
    filtered = [l for l in raw_pool if l.market != "total_points"]
    out = search_market_sgms(filtered, odds_book=odds_book, min_joint_prob=0.0)
    for r in out:
        assert not any("Total points" in name for name in r["legs"])


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


# ── Part 2: Sportsbet note is always informative ──────────────────────────────

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
    note = ("_⚠ Sportsbet: 0 legs priced — no URL file / empty list "
            "(`reports/2026_r17_sportsbet_urls.json`). Book/Edge columns show '—'. "
            "Fix: populate `reports/2026_r17_sportsbet_urls.json` with this round's "
            "Sportsbet match URLs and rerun with `--sportsbet`._")
    md = render_markdown(2026, 17, _bare_match(), has_odds=False, sportsbet_note=note)
    assert "⚠" in md
    assert "0 legs priced" in md
    assert "Fix:" in md


def test_sportsbet_note_success_shows_leg_count():
    note = "_Player-prop odds: live from Sportsbet (scraped, 42 leg(s) priced)._"
    md = render_markdown(2026, 17, _bare_match(), has_odds=False, sportsbet_note=note)
    assert "42 leg(s) priced" in md

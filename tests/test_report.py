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
    search_match_sgms,
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
    legs = _ladder_legs()
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
    legs = _ladder_legs()
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

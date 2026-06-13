"""Round-report helpers (round-2 §10): projection tables + SGM joint search."""

import numpy as np

from afl_bot.build.multi import LegCandidate
from afl_bot.build.report import (
    DEFAULT_ODDS_BANDS,
    projection_rows,
    render_markdown,
    search_match_sgms,
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


def test_search_match_sgms_ladder_is_3leg_one_per_band_and_above_floor():
    out = search_match_sgms(_ladder_legs())            # all defaults: 3-leg, banded
    assert out, "should find 3-leg combos"
    bands = DEFAULT_ODDS_BANDS
    for r in out:
        assert len(r["legs"]) == 3                     # minimum-3-leg ladder
        assert r["odds"] >= bands[0][0]                # banding odds above the 1.75 floor
        assert {"joint_prob", "naive_product", "corr_gain", "fair_odds"} <= set(r)
    # this spread populates every band, so we get exactly one rung per distinct band
    assert len(out) == len(bands)
    band_of = [next(i for i, (lo, hi) in enumerate(bands) if lo <= r["odds"] < hi) for r in out]
    assert band_of == sorted(band_of)                  # ordered safest -> longest
    assert len(band_of) == len(set(band_of))           # one per band, no overlap


def test_search_match_sgms_excludes_conflicts():
    legs = _ladder_legs()
    legs.append(_leg("A 20+ disp", 0.5, legs[0].mask, "A"))  # conflicts with "A 15+ disp"
    # single wide band, many per band, so every surviving combo is returned
    out = search_match_sgms(legs, odds_bands=((1.0, 50.0),), per_band=50)
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
    assert vp["odds"] >= DEFAULT_ODDS_BANDS[-1][0]     # sits in the ~3.5-5.5 value band


def test_search_match_sgms_implausible_edge_is_not_flagged_value():
    legs = _ladder_legs(odds_mult=1.15)   # book 15% over fair -> shrunk edge blows past 15%
    odds_book = {leg.name: leg.market_odds for leg in legs}
    out = search_match_sgms(legs, odds_book=odds_book)
    assert out
    assert any("edge" in r for r in out)               # combos ARE priced...
    assert all(r["edge"] > 0.15 for r in out if "edge" in r)   # ...all above the cap
    assert not any(r.get("value_pick") for r in out)   # so none is flagged VALUE


def test_search_match_sgms_fills_every_band_when_some_are_empty():
    # All legs ~0.82 -> every 3-combo lands in band 1; bands 2 and 3 are empty and
    # must be filled (B4) so the game still shows a full ladder.
    rng = np.random.default_rng(7)
    legs = [_leg(f"{name} 15+ disp", (m := rng.random(40000) < 0.82).mean(), m, name)
            for name in "ABCDE"]
    out = search_match_sgms(legs)
    assert len(out) == len(DEFAULT_ODDS_BANDS)         # no blank rung
    assert all(len(r["legs"]) == 3 for r in out)
    assert [r["odds"] for r in out] == sorted(r["odds"] for r in out)


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

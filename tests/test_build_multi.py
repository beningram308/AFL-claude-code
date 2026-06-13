from afl_bot.build.multi import LegCandidate, build_anchor_multis, build_promo_multi


def make_anchor(name, match_id, subject, prob=0.90, odds=1.05):
    return LegCandidate(name=name, match_id=match_id, market="h2h",
                         subject=subject, fair_prob=prob, market_odds=odds)


def make_value(name, match_id, subject, prob=0.55, odds=2.20):
    return LegCandidate(name=name, match_id=match_id, market="player_disposals",
                         subject=subject, fair_prob=prob, market_odds=odds)


def test_promo_multi_picks_two_anchors_and_value():
    candidates = [
        make_anchor("Team A win", "m1", "Team A"),
        make_anchor("Team B win", "m2", "Team B"),
        make_anchor("Team C win", "m3", "Team C", prob=0.86, odds=1.10),
        make_value("Player X 2+ goals", "m4", "Player X"),
    ]
    result = build_promo_multi(candidates)
    assert result is not None
    classes = sorted(leg.classification for leg in result.legs)
    assert classes == ["ANCHOR", "ANCHOR", "VALUE"]
    assert result.promo["ev_dollars"] > 0


def test_promo_multi_returns_none_without_value_leg():
    candidates = [
        make_anchor("Team A win", "m1", "Team A"),
        make_anchor("Team B win", "m2", "Team B"),
    ]
    assert build_promo_multi(candidates) is None


def test_anchor_multis_exclude_conflicting_legs():
    candidates = [
        make_anchor("Team A win", "m1", "Team A", prob=0.90),
        make_anchor("Team A win again (dup)", "m1", "Team A", prob=0.88),
        make_anchor("Team B win", "m2", "Team B", prob=0.86),
    ]
    multis = build_anchor_multis(candidates, n_multis=3, legs_per_multi=2)
    for multi in multis:
        subjects = [leg.subject for leg in multi.legs]
        assert len(subjects) == len(set(subjects)) or len({(leg.match_id, leg.market, leg.subject) for leg in multi.legs}) == len(multi.legs)


def test_combined_prob_independent_legs():
    candidates = [
        make_anchor("Team A win", "m1", "Team A", prob=0.9),
        make_anchor("Team B win", "m2", "Team B", prob=0.85),
    ]
    multis = build_anchor_multis(candidates, legs_per_multi=2)
    assert len(multis) == 1
    assert abs(multis[0].combined_fair_prob - 0.9 * 0.85) < 1e-9


def test_anchor_multis_default_is_three_legs():
    # MULTI-CHANGES PART B3: default legs_per_multi bumped 2 -> 3.
    candidates = [   # all prob >= ANCHOR_MIN_PROB (0.85) so they classify ANCHOR
        make_anchor("Team A win", "m1", "Team A", prob=0.92, odds=1.10),
        make_anchor("Team B win", "m2", "Team B", prob=0.88, odds=1.15),
        make_anchor("Team C win", "m3", "Team C", prob=0.86, odds=1.18),
    ]
    multis = build_anchor_multis(candidates)            # no legs_per_multi -> default 3
    assert len(multis) == 1
    assert len(multis[0].legs) == 3
    assert abs(multis[0].combined_fair_prob - 0.92 * 0.88 * 0.86) < 1e-9
    # natural combined odds for the safe rung sit below the 1.75 ladder floor
    assert multis[0].combined_market_odds < 1.75

"""
Multi assembler (plan §4.3, §4.4) — stage 7.

Takes a pool of priced ``LegCandidate``s (one per market the sim has priced)
and assembles:

  * The **3-leg promo multi**: 2 ANCHOR legs (as independent as possible) +
    1 VALUE leg, surfaced only if the promo-aware EV is positive.
  * **3x "very highly likely" multis**: all-ANCHOR combinations ranked by
    combined probability, for the low-variance "safe" builds.

Hard constraints enforced throughout:
  * No contradictory legs (over vs under the same line, or two different
    lines on the same player/team/stat -- books void/down-price stacks).
  * No duplicate underlying outcomes across legs in one multi.
  * For SGMs (legs sharing a match_id), the *joint* sim probability must be
    supplied via ``joint_prob_fn`` rather than multiplying independents
    (plan §3.4).
  * Any leg with ``confirmed=False`` (player not in the confirmed lineup) is
    excluded entirely and reported as DO NOT BET.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from afl_bot.config import BONUS_BET_FACTOR, DEFAULT_STAKE
from afl_bot.pricing.edge import classify_leg, edge, fair_odds


def joint_prob_from_masks(legs: list["LegCandidate"]) -> float:
    """True joint probability of a set of same-match legs from their aligned
    per-iteration sim masks (AND then mean). Falls back to the independent
    product if any leg has no mask. Use as ``joint_prob_fn`` so same-game combos
    price off the correlated sim, not naive multiplication (round-2 §3)."""
    masks = [leg.mask for leg in legs]
    if any(m is None for m in masks):
        prob = 1.0
        for leg in legs:
            prob *= leg.fair_prob
        return prob
    combined = np.logical_and.reduce([np.asarray(m, dtype=bool) for m in masks])
    return float(combined.mean())


@dataclass
class LegCandidate:
    name: str                # human-readable, e.g. "Brisbane Lions to win"
    match_id: str            # groups legs from the same game (for SGM detection)
    market: str              # e.g. "h2h", "player_disposals", "total_points"
    subject: str             # team or player name -- used for conflict detection
    fair_prob: float
    market_odds: float
    confirmed: bool = True   # False if player not in confirmed lineup -> DO NOT BET
    # Per-iteration boolean outcome from the match sim (samples >= line, margin
    # > 0, ...). Same-match legs share an aligned simulation, so ANDing masks
    # gives the true correlated SGM joint probability (plan §3.4 / round-2 §3).
    mask: object = field(default=None, compare=False, repr=False)

    classification: str = field(init=False)
    edge_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.edge_pct = edge(self.fair_prob, self.market_odds)
        self.classification = classify_leg(self.fair_prob, self.market_odds)

    @property
    def fair_odds(self) -> float:
        return fair_odds(self.fair_prob)

    @property
    def conflict_key(self) -> tuple[str, str, str]:
        """Two legs sharing this key are the 'same underlying outcome' (possibly
        at different lines/directions) and cannot coexist in one multi."""
        return (self.match_id, self.market, self.subject)


def usable_legs(candidates: list[LegCandidate]) -> tuple[list[LegCandidate], list[LegCandidate]]:
    """Split into (usable, do_not_bet) based on lineup confirmation."""
    usable = [c for c in candidates if c.confirmed]
    do_not_bet = [c for c in candidates if not c.confirmed]
    return usable, do_not_bet


def _no_conflicts(legs: list[LegCandidate]) -> bool:
    keys = [leg.conflict_key for leg in legs]
    return len(keys) == len(set(keys))


def combined_prob(legs: list[LegCandidate], joint_prob_fn=None) -> float:
    """Combined probability across legs.

    Cross-game legs (different match_id) are independent -- multiply.
    Legs sharing a match_id are an SGM and MUST be priced via
    ``joint_prob_fn(legs_in_that_match) -> float`` (the true joint sim
    probability, plan §3.4) rather than the product of independents.
    """
    by_match: dict[str, list[LegCandidate]] = {}
    for leg in legs:
        by_match.setdefault(leg.match_id, []).append(leg)

    prob = 1.0
    for match_id, match_legs in by_match.items():
        if len(match_legs) == 1:
            prob *= match_legs[0].fair_prob
        else:
            if joint_prob_fn is None:
                raise ValueError(
                    f"Multiple legs from match {match_id!r} require joint_prob_fn "
                    "(plan §3.4: never multiply correlated SGM legs as independent)"
                )
            prob *= joint_prob_fn(match_legs)
    return prob


def combined_odds(legs: list[LegCandidate]) -> float:
    odds = 1.0
    for leg in legs:
        odds *= leg.market_odds
    return odds


# ----------------------------------------------------------------------------- #
# Promo-aware EV (plan §4.4)
# ----------------------------------------------------------------------------- #
def promo_multi_ev(
    p1: float, p2: float, p3: float, multi_odds: float,
    stake: float = DEFAULT_STAKE, bonus_factor: float = BONUS_BET_FACTOR,
) -> dict:
    """EV of a 3-leg multi where the book refunds the stake as a bonus bet if
    the multi fails by exactly one leg. Assumes independent legs (cross-game);
    for SGM legs pre-combine via ``combined_prob`` and pass the result.
    """
    p_all = p1 * p2 * p3
    p_one_loss = (
        p1 * p2 * (1 - p3)
        + p1 * (1 - p2) * p3
        + (1 - p1) * p2 * p3
    )
    p_dead = 1 - p_all - p_one_loss

    ev = (
        p_all * stake * (multi_odds - 1)
        + p_one_loss * stake * bonus_factor
        - p_dead * stake
    )
    return {
        "p_all_win": p_all,
        "p_exactly_one_loss": p_one_loss,
        "p_dead": p_dead,
        "ev_dollars": ev,
        "ev_pct": ev / stake,
    }


# ----------------------------------------------------------------------------- #
# Multi assembly
# ----------------------------------------------------------------------------- #
@dataclass
class MultiResult:
    legs: list[LegCandidate]
    combined_fair_prob: float
    combined_market_odds: float
    promo: dict | None = None

    @property
    def combined_fair_odds(self) -> float:
        return fair_odds(self.combined_fair_prob)

    @property
    def combined_edge(self) -> float:
        """Model EV per $1 on the multi at book odds (round-2 §8.1): high
        combined probability is not the same as value."""
        return self.combined_fair_prob * self.combined_market_odds - 1.0


def build_promo_multi(
    candidates: list[LegCandidate],
    stake: float = DEFAULT_STAKE,
    bonus_factor: float = BONUS_BET_FACTOR,
    joint_prob_fn=None,
) -> MultiResult | None:
    """2 ANCHOR legs (preferring different matches for independence) + 1 VALUE
    leg, surfaced only if promo EV > 0. Returns None if no qualifying
    combination exists.
    """
    usable, _ = usable_legs(candidates)
    anchors = sorted(
        (c for c in usable if c.classification == "ANCHOR"),
        key=lambda c: c.fair_prob, reverse=True,
    )
    values = sorted(
        (c for c in usable if c.classification == "VALUE"),
        key=lambda c: c.edge_pct, reverse=True,
    )

    best: MultiResult | None = None
    for value_leg in values:
        # Prefer two anchors from different matches, and from a different match
        # to the value leg, so "fails by exactly one leg" is dominated by the
        # value leg (plan §4.3).
        anchor_pool = [a for a in anchors if a.match_id != value_leg.match_id]
        if len(anchor_pool) < 2:
            anchor_pool = anchors  # fall back if not enough cross-match anchors

        for a1, a2 in combinations(anchor_pool, 2):
            legs = [a1, a2, value_leg]
            if not _no_conflicts(legs):
                continue

            try:
                p_combo = combined_prob(legs, joint_prob_fn)
            except ValueError:
                continue

            odds = combined_odds(legs)
            promo = promo_multi_ev(a1.fair_prob, a2.fair_prob, value_leg.fair_prob,
                                    odds, stake, bonus_factor)
            if promo["ev_dollars"] <= 0:
                continue

            candidate_result = MultiResult(legs, p_combo, odds, promo)
            if best is None or promo["ev_dollars"] > best.promo["ev_dollars"]:
                best = candidate_result

    return best


def build_anchor_multis(
    candidates: list[LegCandidate], n_multis: int = 3, legs_per_multi: int = 3,
    joint_prob_fn=None,
) -> list[MultiResult]:
    """All-ANCHOR multis ranked by combined probability -- the low-variance
    "safe" builds (plan §4.3). Default 3 legs (MULTI-CHANGES PART B3): their
    natural combined odds (~1.2-1.6) sit below the 1.75 ladder floor, so they're
    the bottom rung of the multi ladder."""
    usable, _ = usable_legs(candidates)
    anchors = [c for c in usable if c.classification == "ANCHOR"]

    results: list[MultiResult] = []
    for combo in combinations(anchors, legs_per_multi):
        legs = list(combo)
        if not _no_conflicts(legs):
            continue
        try:
            p_combo = combined_prob(legs, joint_prob_fn)
        except ValueError:
            continue
        results.append(MultiResult(legs, p_combo, combined_odds(legs)))

    results.sort(key=lambda r: r.combined_fair_prob, reverse=True)
    return results[:n_multis]

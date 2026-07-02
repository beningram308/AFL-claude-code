"""
Weekly round report (round-2 §10) - the actual product: per match, real-player
projection tables and same-game multis ranked by the *joint* sim probability,
saved to a file.

Pure rendering/search helpers here; the simulation orchestration lives in
``afl_bot.cli.round_report`` (it reuses the same per-team sample engine as
run-round). SGM joint probabilities come from the per-iteration leg masks
(``afl_bot.build.multi.joint_prob_from_masks``), so combos are priced off the
correlated sim, not the naive product.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from afl_bot.backtest.props import apply_prop_calibration
from afl_bot.build.multi import LegCandidate, _no_conflicts, combined_odds, joint_prob_from_masks
from afl_bot.build.staking import multi_outcome_kelly
from afl_bot.config import (
    BOOKABLE_MARKS_ROLES,
    BOOKABLE_PROP_MENU,
    BOOKABLE_TACKLES_ROLES,
    BOOKABLE_TOP_N_BY_STAT,
    BONUS_BET_FACTOR,
    MAX_MARKS_LEGS_PER_MULTI,
    MAX_TACKLE_MARKS_LEGS,
    MULTI_MARKET_SHRINK,
    MULTI_TARGET_ODDS,
    PROMO_MIN_LEGS,
    PULL_DETECTION_PROB,
    PULL_EM_ANCHOR_MIN_P,
    PULL_EM_BOOSTER_MAX_P,
    PULL_EM_BOOSTER_MIN_P,
    STAT_PREFERENCE,
    UNIT_SIZE,
)
from afl_bot.pricing.edge import fair_odds, market_anchored_prob, mc_standard_error


def projection_rows(player_samples: dict[str, dict], lines: dict[str, list],
                    calibrators: dict | None = None) -> list[dict]:
    """One row per player: projected mean + P(line) for every stat/line, sorted
    by projected disposals (descending). ``calibrators`` (per ``(stat, line)``,
    with a per-stat pooled fallback — see ``apply_prop_calibration``) apply
    the §2.3 / Phase 3.2 calibration to the printed probabilities."""
    calibrators = calibrators or {}
    rows = []
    for player, samples in player_samples.items():
        row: dict = {"player": player}
        for stat, stat_lines in lines.items():
            arr = samples.get(stat)
            if arr is None:
                continue
            row[f"{stat}_mean"] = float(arr.mean())
            for line in stat_lines:
                p = float((arr >= line).mean())
                p = apply_prop_calibration(calibrators, stat, line, p)
                row[f"{stat}_{line}+"] = p
        rows.append(row)
    rows.sort(key=lambda r: r.get("disposals_mean", 0.0), reverse=True)
    return rows


def top_n_players_by_stat(player_samples: dict[str, dict], stat: str, n: int) -> set[str]:
    """The ``n`` players (of those carrying ``stat``) with the highest projected
    mean for it -- "books price the obvious names" (FIX-BETTABLE-LEGS), used to
    gate which players can get a MODEL-ONLY prop leg for that stat."""
    ranked = sorted(
        (p for p, s in player_samples.items() if s.get(stat) is not None),
        key=lambda p: float(player_samples[p][stat].mean()), reverse=True)
    return set(ranked[:n])


def is_bookable_model_only_leg(stat: str, line: int, player: str, role: str | None,
                               team_top_n: set[str]) -> bool:
    """Whether a MODEL-ONLY prop leg (no real book price entered) is realistic
    enough to post in the live ladder (FIX-BETTABLE-LEGS-AND-PRICING STEP 1):
    the line must be on the ``BOOKABLE_PROP_MENU``, the player must be a
    top-projected name for that stat on their team, and -- for marks/tackles,
    where books are picky about which roles get a market -- the player's
    inferred role must be one books actually post. A leg WITH a real book
    price always bypasses this (a posted market is bettable by definition);
    this only gates legs the model invented with nothing behind them."""
    if line not in BOOKABLE_PROP_MENU.get(stat, ()):
        return False
    if player not in team_top_n:
        return False
    if stat == "marks" and role not in BOOKABLE_MARKS_ROLES:
        return False
    if stat == "tackles" and role not in BOOKABLE_TACKLES_ROLES:
        return False
    return True


def select_ladder_lines(qualifying: list[dict]) -> list[dict]:
    """From every line for ONE (player, stat) that already cleared the
    LEG_PROB_MIN/MAX gate and (if unpriced) the ``BOOKABLE_PROP_MENU`` gate
    (FIX-PLACEABLE-LEGS-AND-210-FLOOR STEP 2.2), pick which ones become live
    ladder legs: every PRICED line (each ``{"priced": True, ...}`` entry is a
    confirmed real market, so all are kept) plus, if any UNPRICED lines
    qualify, only the single highest-``"prob"`` one -- a book doesn't post
    both a near-lock 15+ line and a 25+ line on the same gun mid, so the
    model-only ladder pool shouldn't either. Returns ``[]`` if ``qualifying``
    is empty."""
    priced = [q for q in qualifying if q["priced"]]
    unpriced = [q for q in qualifying if not q["priced"]]
    return priced + ([max(unpriced, key=lambda q: q["prob"])] if unpriced else [])


# Kept for backward compatibility — no longer used in search_match_sgms.
DEFAULT_ODDS_BANDS = ((1.75, 2.50), (2.50, 3.50), (3.50, 5.50))


def build_odds_template(leg_names: list[str]) -> dict:
    """Template for the manual ``--odds`` JSON (model-upgrade audit Phase 4
    STEP 1.2): every priceable leg's exact name -> ``null``, plus the
    ``_rules`` stub (``h2h_draw``, consumed by ``run-round``'s draw-refund
    handling, so the same filled-in file works for both CLIs). Write it,
    fill in the numbers off the bookie, pass the file back via ``--odds`` --
    this kills the leg-name-typo problem at the source since the keys are
    copy-pasted, not retyped from scratch."""
    template: dict = {name: None for name in sorted(set(leg_names))}
    template["_rules"] = {"h2h_draw": None}
    return template


def build_sgm_candidates(legs: list[LegCandidate], *, min_legs: int = 3, max_legs: int = 3,
                         odds_book: dict | None = None,
                         min_joint_prob: float = 0.05) -> list[dict]:
    """Every non-conflicting ``min_legs``..``max_legs``-leg combo from ``legs``
    clearing ``min_joint_prob``, each with its joint sim probability (from
    masks), naive product, correlation gain, fair odds, MC sample size
    (``n_sims``, from a leg's mask length — ``None`` if masks aren't
    available), and — when every leg has a book price — book odds + a
    market-shrunk edge. This is the full candidate population
    ``search_match_sgms`` selects ladder rungs from, split out so a backtest
    can grade calibration over EVERY candidate, not just the selected ones
    (model-upgrade audit Phase 3.5 — optimizer's-curse / selection-bias
    check: does picking "closest fair odds to a target" out of many noisy
    estimates systematically pick the one that got there via upward noise,
    not genuine accuracy?)."""
    odds_book = odds_book or {}
    combos: list[dict] = []
    for r in range(min_legs, max_legs + 1):
        for combo in combinations(legs, r):
            legs_list = list(combo)
            if not _no_conflicts(legs_list):
                continue
            if len({leg.subject for leg in legs_list}) != len(legs_list):
                continue
            joint = joint_prob_from_masks(legs_list)
            if joint < min_joint_prob:
                continue
            naive = 1.0
            for leg in legs_list:
                naive *= leg.fair_prob

            entry = {
                "legs": [leg.name for leg in legs_list],
                "joint_prob": joint,
                "naive_product": naive,
                "corr_gain": joint - naive,
                "fair_odds": fair_odds(joint),
                "n_sims": len(legs_list[0].mask) if legs_list[0].mask is not None else None,
                "value_pick": False,
                "_leg_masks": (
                    [leg.mask for leg in legs_list]
                    if all(leg.mask is not None for leg in legs_list) else None
                ),
            }
            if all(odds_book.get(leg.name) is not None for leg in legs_list):
                book = combined_odds(legs_list)
                shrunk = market_anchored_prob(joint, book, MULTI_MARKET_SHRINK)
                entry["book_odds"] = book
                entry["raw_edge"] = joint * book - 1.0
                entry["edge"] = shrunk * book - 1.0
            entry["odds"] = entry.get("book_odds", entry["fair_odds"])
            # Preference score (secondary sort key), marks leg count, and
            # combined tackles+marks count (for the combined cap filter).
            pref = 0.0
            n_marks = 0
            n_tackle_marks = 0
            for leg in legs_list:
                mstat = (leg.market.replace("player_", "")
                         if leg.market.startswith("player_") else leg.market)
                pref += STAT_PREFERENCE.get(mstat, 0.5)
                if mstat == "marks":
                    n_marks += 1
                if mstat in ("marks", "tackles"):
                    n_tackle_marks += 1
            entry["_pref_score"] = pref
            entry["_n_marks"] = n_marks
            entry["_n_tackle_marks"] = n_tackle_marks
            combos.append(entry)
    return combos


def search_match_sgms(legs: list[LegCandidate], *, min_legs: int = 3, max_legs: int = 3,
                      odds_book: dict | None = None, target_odds: tuple | None = None,
                      min_joint_prob: float = 0.05,
                      max_plausible_edge: float = 0.15,
                      lcb_z: float = 0.0, price_shrink: float = 0.0,
                      corr_gain_haircut: float = 1.0,
                      multi_calibrator=None) -> list[dict]:
    """Build a *ladder* of same-game multis with one rung per target odds
    (REAL-MULTIS ADDENDUM 1), selected from ``build_sgm_candidates``'s full
    candidate pool. For each combo: the JOINT sim prob (from masks), naive
    product, correlation gain, fair odds, and — when every leg has a book
    price — book odds + a SHRUNK edge.

    Edge is *not* the raw ``joint*book-1``: per-leg model overestimates compound
    multiplicatively across a multi, so the joint is first pulled toward the
    book's implied multi prob via ``market_anchored_prob`` (``MULTI_MARKET_SHRINK``,
    round-2 §8.2) before the edge. A shrunk edge above ``max_plausible_edge``
    (default 15%) is treated as "model is wrong, not the book".

    Rung selection: for each target in ``target_odds`` (default ``MULTI_TARGET_ODDS``
    = 2.10 / 3.00 / 5.00), prefer the combo whose FINAL fair odds -- i.e.
    after ``corr_gain_haircut`` and ``multi_calibrator`` are applied, the same
    number that gets reported -- is closest to the target FROM ABOVE: land
    *at or longer than* the target, never shorter (FIX-PLACEABLE-LEGS-AND-
    210-FLOOR: a bottom rung quietly printing $1.50 when $2.10 was promised
    is a worse surprise for the bet slip than printing $2.20; checking the
    guard against a PRE-haircut/PRE-calibration joint, as a now-retired
    version of this function did, let the printed number drift below the
    band it supposedly cleared). Only when no combo reaches the target at
    all does selection fall back to the closest one below it. Highest FINAL
    joint prob breaks ties. The top rung (highest target) is promoted to
    ``value_pick=True`` when a qualifying FINAL shrunk edge (0,
    max_plausible_edge] exists among the available combos. De-duplication:
    each combo can fill only one rung; if the pool is exhausted, remaining
    rungs are filled from the full pool. Each selected rung is tagged with
    its own ``target_odds`` (the band it filled) for display.

    ``lcb_z`` (model-upgrade audit Phase 3.5, opt-in, default 0.0 = off, the
    existing behaviour): rank combos for selection by a lower-confidence-bound
    estimate, ``max(0, joint_prob - lcb_z * mc_standard_error(joint_prob,
    n_sims))``, instead of the raw point estimate -- the standard fix for
    "argmax/closest-match over many noisy estimates over-selects upward
    noise" (the optimizer's curse). The closest-to-target comparison is done
    in PROBABILITY space (``|lcb_value - 1/target|``) rather than odds space
    when ``lcb_z>0``: odds are ``1/p``, so the same absolute probability
    haircut swings a long-shot combo's odds far more than a near-even
    combo's, which would swamp any genuine precision difference between
    combos if compared in odds space. Combos with no ``n_sims`` (no mask)
    rank on the raw point estimate regardless. Only *which* combo wins
    changes; the winner's reported ``joint_prob``/``fair_odds`` stay its own
    raw point estimate unless ``price_shrink`` is also set.

    ``price_shrink`` (opt-in, default 0.0 = off): after selection, shrinks
    the WINNING combo's reported ``joint_prob`` toward *that rung's target's*
    implied probability (``1 / target``) by this factor (0 = no shrink, 1 =
    fully at the target), recomputing ``fair_odds``/``edge`` from the shrunk
    value — the same `market_anchored_prob` mechanic the live book-odds path
    already uses for market anchoring, anchored to the TARGET odds instead
    since this backtest has no book. ``corr_gain``/``naive_product`` are
    left at their pre-shrink values (informational: what the sim's
    correlation structure contributed before any haircut).

    ``corr_gain_haircut`` (model-upgrade audit Phase 4 corr_gain-diagnostic
    follow-up, opt-in, default 1.0 = unhaircut/current behaviour): reprices
    EVERY combo in the pool as ``naive_product + corr_gain_haircut *
    corr_gain`` instead of the raw sim ``joint_prob``, clipped to ``[0, 1]``
    -- 0.0 prices purely off the naive/independent product. The diagnostic
    (README "corr_gain diagnostic" section) found the sim's correlation lift
    is systematically larger than the empirical one, so this shrinks that
    specific term rather than the selection mechanism (`lcb_z`/`price_shrink`,
    which failed). Applied BEFORE selection (FIX-PLACEABLE-LEGS-AND-210-
    FLOOR STEP 4) so the "never land shorter" guard above sees the same
    number it reports, not the pre-haircut one. ``corr_gain``/
    ``naive_product`` on the returned, SELECTED rungs are left at their
    pre-haircut (raw sim) values -- informational, same convention as
    ``price_shrink``.

    ``multi_calibrator`` (model-upgrade audit Phase 3.6, opt-in, default
    ``None``, e.g. an ``afl_bot.backtest.ensemble.IsotonicCalibrator`` from
    `afl_bot.backtest.multis.load_or_fit_multi_calibrator`): applied to every
    combo's ``corr_gain_haircut``-ed joint prob, ALSO before selection, for
    the same reason -- the old two-step pattern of selecting here and then
    calibrating the winners afterwards (`apply_multi_calibration`, now a
    standalone no-op-when-None convenience left for other callers) let
    calibration inflate a rung's joint prob (e.g. 57% -> 63%) AFTER the
    band-clearing guard had already passed on the uncalibrated value,
    printing fair odds below the band it supposedly cleared. Folding it in
    here closes that gap: the guard and the printed number are now always
    the same final, calibrated value.

    Both haircut and calibrator are applied before ``price_shrink`` if that
    is also set (fix the model's own estimate first, then any book-anchoring
    shrink on top).

    Returned safest -> longest (by fair odds)."""
    target_odds = tuple(sorted(target_odds)) if target_odds is not None else MULTI_TARGET_ODDS
    combos = build_sgm_candidates(legs, min_legs=min_legs, max_legs=max_legs,
                                  odds_book=odds_book, min_joint_prob=min_joint_prob)
    # Drop combos exceeding the marks cap (ALL marks legs count, priced or not).
    combos = [c for c in combos if c.get("_n_marks", 0) <= MAX_MARKS_LEGS_PER_MULTI]
    # Drop combos where combined tackles+marks legs exceed the combined cap.
    combos = [c for c in combos if c.get("_n_tackle_marks", 0) <= MAX_TACKLE_MARKS_LEGS]

    # Stable tie-break key: sorted leg names guarantee the same combo always
    # wins when two candidates score identically on the primary sort key.
    def _leg_key(c: dict) -> tuple:
        return tuple(sorted(c["legs"]))
    if not combos:
        return []

    # FIX-PLACEABLE-LEGS-AND-210-FLOOR STEP 4: price every combo (haircut +
    # calibrator) BEFORE selection, so the guards below and the final report
    # both work off the same number -- see docstring.
    for c in combos:
        if corr_gain_haircut != 1.0:
            priced_joint = min(max(c["naive_product"] + corr_gain_haircut * c["corr_gain"], 0.0), 1.0)
        else:
            priced_joint = c["joint_prob"]
        if multi_calibrator is not None:
            priced_joint = float(multi_calibrator.predict([priced_joint])[0])
        c["_priced_joint"] = priced_joint
        c["_priced_fair_odds"] = fair_odds(priced_joint)
        if "book_odds" in c:
            book = c["book_odds"]
            c["_priced_raw_edge"] = priced_joint * book - 1.0
            c["_priced_edge"] = market_anchored_prob(priced_joint, book, MULTI_MARKET_SHRINK) * book - 1.0

    # PHASE 2 STEP 1: Compute promo branch probabilities from sim masks for every
    # candidate, so VALUE PICK selection (below) can rank by Total EV.
    for c in combos:
        leg_masks = c.get("_leg_masks")
        n_legs = len(c["legs"])
        priced_edge = c.get("_priced_edge")
        if leg_masks is not None and n_legs >= PROMO_MIN_LEGS:
            masks_arr = np.vstack([np.asarray(m, dtype=bool) for m in leg_masks])
            n_win = masks_arr.sum(axis=0)
            p_all = float((n_win == n_legs).mean())
            p_one = float((n_win == n_legs - 1).mean())
            p_dead = max(0.0, 1.0 - p_all - p_one)
            promo = p_one * BONUS_BET_FACTOR
            c["_p_all_win"] = p_all
            c["_p_one_loss"] = p_one
            c["_p_two_plus_loss"] = p_dead
            c["_promo_ev"] = promo
            c["_total_ev"] = (priced_edge if priced_edge is not None else 0.0) + promo
        else:
            c["_p_all_win"] = None
            c["_p_one_loss"] = None
            c["_p_two_plus_loss"] = None
            c["_promo_ev"] = None
            c["_total_ev"] = priced_edge  # no promo: total EV = base edge

    def _lcb_value(c: dict) -> float:
        if lcb_z <= 0.0 or c["n_sims"] is None:
            return c["_priced_joint"]
        return max(0.0, c["joint_prob"] - lcb_z * mc_standard_error(c["joint_prob"], c["n_sims"]))

    def _lcb_active(c: dict) -> bool:
        return lcb_z > 0.0 and c["n_sims"] is not None

    def _distance(c: dict, target: float) -> float:
        # Probability-space distance under the haircut (see docstring: odds
        # space's 1/p nonlinearity would swamp genuine precision differences);
        # odds-space distance (the original behaviour) otherwise.
        if _lcb_active(c):
            return abs(_lcb_value(c) - 1.0 / target)
        return abs(c["_priced_fair_odds"] - target)

    def _select_for_target(available: list[dict], target: float) -> dict:
        # Prefer combos whose FINAL (priced) odds are AT OR LONGER than the
        # target (i.e. priced joint <= 1/target -- never overshoot short);
        # among those, the closest to the target is the one with the HIGHEST
        # priced joint prob (shortest odds that still clear the target). Only
        # fall back to the closest combo below the target if none reach it
        # at all. Scoped to ``lcb_z<=0`` (round-report's own path, lcb_z
        # always 0) -- ``lcb_z>0`` is an opt-in Phase 3.5 selection-haircut
        # diagnostic whose whole point is to flip the pick via the
        # lcb-adjusted distance, which this "land at/above" preference would
        # short-circuit.
        # DISPOSALS-FIRST: within the reachable pool, prefer combos with zero
        # tackles/marks legs (tier 0) before any combo that includes a tackle
        # or marks leg (tier 1). This makes the disposals preference dominate
        # the selection, not just break ties. Only fall back to tier 1 when
        # no tier-0 combo can reach the target at all. Within each tier the
        # existing rule applies: highest priced joint (closest from above),
        # then pref_score, then stable leg-key.
        if lcb_z <= 0.0:
            reaches_target = [c for c in available if c["_priced_joint"] <= 1.0 / target]
            if reaches_target:
                best_tm = min(c.get("_n_tackle_marks", 0) for c in reaches_target)
                tier = [c for c in reaches_target if c.get("_n_tackle_marks", 0) == best_tm]
                return max(tier, key=lambda c: (
                    c["_priced_joint"], c.get("_pref_score", 0.0),
                    -_distance(c, target), _leg_key(c)))
            # Fallback: nothing reaches target — still prefer fewer tackle/marks.
            best_tm = min(c.get("_n_tackle_marks", 0) for c in available)
            tier = [c for c in available if c.get("_n_tackle_marks", 0) == best_tm]
            return min(tier, key=lambda c: (
                _distance(c, target), -c.get("_pref_score", 0.0),
                -_lcb_value(c), _leg_key(c)))
        return min(available, key=lambda c: (
            _distance(c, target), -c.get("_pref_score", 0.0),
            -_lcb_value(c), _leg_key(c)))

    selected: list[dict] = []
    chosen: set[int] = set()
    for i, target in enumerate(target_odds):
        available = [c for c in combos if id(c) not in chosen] or list(combos)
        is_top = (i == len(target_odds) - 1)
        if is_top:
            # Sanity gate: base edge within plausible range; rank by total EV (promo-aware).
            pool = [c for c in available
                    if c.get("_priced_edge") is not None
                    and c["_priced_edge"] <= max_plausible_edge]
            def _tv(c: dict) -> float:
                tv = c.get("_total_ev")
                return tv if tv is not None else 0.0
            valued = [c for c in pool if _tv(c) > 0.0]
            if valued:
                valued.sort(key=lambda c: (_tv(c), _leg_key(c)), reverse=True)
                pick = valued[0]
                pick["value_pick"] = True
            else:
                pick = _select_for_target(available, target)
        else:
            pick = _select_for_target(available, target)
        pick["target_odds"] = target
        chosen.add(id(pick))
        selected.append(pick)

    for pick in selected:
        if "_priced_joint" not in pick:
            continue  # reused combo (pool exhausted, filled from list(combos)) — already cleaned
        pick["joint_prob"] = pick.pop("_priced_joint")
        pick["fair_odds"] = pick.pop("_priced_fair_odds")
        if "book_odds" in pick:
            pick["raw_edge"] = pick.pop("_priced_raw_edge")
            pick["edge"] = pick.pop("_priced_edge")
            pick["odds"] = pick["book_odds"]
        # Rename promo stats (private -> public) and strip masks.
        pick["p_all_win"] = pick.pop("_p_all_win", None)
        pick["p_one_loss"] = pick.pop("_p_one_loss", None)
        pick["p_two_plus_loss"] = pick.pop("_p_two_plus_loss", None)
        pick["promo_ev"] = pick.pop("_promo_ev", None)
        pick["total_ev"] = pick.pop("_total_ev", None)
        pick.pop("_leg_masks", None)
        pick.pop("_pref_score", None)
        pick.pop("_n_marks", None)
        pick.pop("_n_tackle_marks", None)
        # Suggested stake via multi-outcome Kelly (promo-eligible rungs only).
        if (pick.get("p_all_win") is not None and pick.get("book_odds")
                and pick.get("total_ev") is not None and pick["total_ev"] > 0):
            pick["suggested_stake"] = multi_outcome_kelly(
                pick["p_all_win"], pick["p_one_loss"], pick["p_two_plus_loss"],
                pick["book_odds"], BONUS_BET_FACTOR,
            )
        else:
            pick["suggested_stake"] = None

    if price_shrink > 0.0:
        for pick, target in zip(selected, target_odds):
            anchor_prob = 1.0 / target
            shrunk = pick["joint_prob"] - price_shrink * (pick["joint_prob"] - anchor_prob)
            pick["joint_prob"] = shrunk
            pick["fair_odds"] = fair_odds(shrunk)
            if "book_odds" in pick:
                book = pick["book_odds"]
                pick["raw_edge"] = shrunk * book - 1.0
                pick["edge"] = market_anchored_prob(shrunk, book, MULTI_MARKET_SHRINK) * book - 1.0
                pick["odds"] = book

    selected.sort(key=lambda c: c["fair_odds"])
    return selected


def search_market_sgms(legs: list[LegCandidate], *, min_legs: int = 3, max_legs: int = 3,
                       odds_book: dict, target_odds: tuple | None = None,
                       min_joint_prob: float = 0.05,
                       max_plausible_edge: float = 0.15) -> list[dict]:
    """A same-game multi ladder selected and priced on REAL BOOK odds, not the
    model's own joint probability (FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART C):
    every leg in every returned rung has a real price in ``odds_book`` (from
    ``--sportsbet`` or ``--odds``). This is the ladder Ben actually sees on
    the bookmaker -- the model's own ``joint_prob``/``fair_odds``/``edge``
    stay attached to each rung so the two can be read side by side (where the
    model disagrees with the market is the real signal).

    Built from the SAME candidate pool ``search_match_sgms`` searches
    (``build_sgm_candidates``), restricted to combos where every leg is
    priced (``"book_odds" in c``) -- returns ``[]`` if none qualify (no
    Sportsbet/--odds prices this run, or none happen to cover a full combo).

    Rung selection mirrors ``search_match_sgms``'s "land at-or-above the
    target, never short" rule and its top-rung VALUE-by-edge promotion (see
    its docstring), just keyed on ``book_odds`` (the naive product of each
    leg's real price -- ``combined_odds``, NOT the book's own same-game-multi
    special, which prices its own correlation and isn't scraped here) instead
    of the model's ``fair_odds``. Real markets often price a near-lock combo
    shorter than the model's 0.78-leg-capped floor (e.g. ~$1.50 vs the
    model's ~$2.10) -- expected, not a bug; that gap IS the point of this
    ladder. No haircut/calibrator transform is applied here: book odds are
    already real prices, not a model estimate to correct.

    Returned safest -> longest (by book odds)."""
    target_odds = tuple(sorted(target_odds)) if target_odds is not None else MULTI_TARGET_ODDS
    combos = build_sgm_candidates(legs, min_legs=min_legs, max_legs=max_legs,
                                  odds_book=odds_book, min_joint_prob=min_joint_prob)
    # Drop combos exceeding the marks cap (ALL marks legs count, priced or not).
    combos = [c for c in combos if c.get("_n_marks", 0) <= MAX_MARKS_LEGS_PER_MULTI]
    # Drop combos where combined tackles+marks legs exceed the combined cap.
    combos = [c for c in combos if c.get("_n_tackle_marks", 0) <= MAX_TACKLE_MARKS_LEGS]
    priced = [c for c in combos if "book_odds" in c]
    if not priced:
        return []

    def _leg_key(c: dict) -> tuple:
        return tuple(sorted(c["legs"]))

    def _select_for_target(available: list[dict], target: float) -> dict:
        # DISPOSALS-FIRST: among combos that reach the book-odds target, prefer
        # those with zero tackles/marks legs (tier 0) before falling back to
        # combos with a tackle/marks leg (tier 1). Within the best available
        # tier, pick the cheapest book_odds that still clears the target (closest
        # from above), with pref_score as the secondary key so disposals beat
        # goals/h2h within the same tier. Fall back to the closest combo below
        # the target only when nothing reaches, again tiering by n_tackle_marks.
        reaches = [c for c in available if c["book_odds"] >= target]
        if reaches:
            best_tm = min(c.get("_n_tackle_marks", 0) for c in reaches)
            tier = [c for c in reaches if c.get("_n_tackle_marks", 0) == best_tm]
            return min(tier, key=lambda c: (c["book_odds"], -c.get("_pref_score", 0.0), _leg_key(c)))
        best_tm = min(c.get("_n_tackle_marks", 0) for c in available)
        tier = [c for c in available if c.get("_n_tackle_marks", 0) == best_tm]
        return max(tier, key=lambda c: (c["book_odds"], c.get("_pref_score", 0.0), _leg_key(c)))

    # PHASE 2 STEP 1: Compute promo branch probabilities from sim masks.
    for c in priced:
        leg_masks = c.get("_leg_masks")
        n_legs = len(c["legs"])
        if leg_masks is not None and n_legs >= PROMO_MIN_LEGS:
            masks_arr = np.vstack([np.asarray(m, dtype=bool) for m in leg_masks])
            n_win = masks_arr.sum(axis=0)
            p_all = float((n_win == n_legs).mean())
            p_one = float((n_win == n_legs - 1).mean())
            p_dead = max(0.0, 1.0 - p_all - p_one)
            promo = p_one * BONUS_BET_FACTOR
            c["_p_all_win"] = p_all
            c["_p_one_loss"] = p_one
            c["_p_two_plus_loss"] = p_dead
            c["_promo_ev"] = promo
            c["_total_ev"] = c["edge"] + promo
        else:
            c["_p_all_win"] = None
            c["_p_one_loss"] = None
            c["_p_two_plus_loss"] = None
            c["_promo_ev"] = None
            c["_total_ev"] = c["edge"]

    selected: list[dict] = []
    chosen: set[int] = set()
    for i, target in enumerate(target_odds):
        available = [c for c in priced if id(c) not in chosen] or list(priced)
        is_top = (i == len(target_odds) - 1)
        if is_top:
            valued = [c for c in available if 0.0 < c["edge"] <= max_plausible_edge]
            if valued:
                # Rank by total EV (promo-aware) rather than base edge alone.
                pick = max(valued, key=lambda c: (c.get("_total_ev") or c["edge"], _leg_key(c)))
                pick["value_pick"] = True
            else:
                pick = _select_for_target(available, target)
        else:
            pick = _select_for_target(available, target)
        pick["target_odds"] = target
        chosen.add(id(pick))
        selected.append(pick)

    # Rename promo stats (private -> public), strip masks, add suggested stakes.
    for pick in selected:
        pick["p_all_win"] = pick.pop("_p_all_win", None)
        pick["p_one_loss"] = pick.pop("_p_one_loss", None)
        pick["p_two_plus_loss"] = pick.pop("_p_two_plus_loss", None)
        pick["promo_ev"] = pick.pop("_promo_ev", None)
        pick["total_ev"] = pick.pop("_total_ev", None)
        pick.pop("_leg_masks", None)
        pick.pop("_pref_score", None)
        pick.pop("_n_marks", None)
        pick.pop("_n_tackle_marks", None)
        if (pick.get("p_all_win") is not None
                and pick.get("total_ev") is not None and pick["total_ev"] > 0):
            pick["suggested_stake"] = multi_outcome_kelly(
                pick["p_all_win"], pick["p_one_loss"], pick["p_two_plus_loss"],
                pick["book_odds"], BONUS_BET_FACTOR,
            )
        else:
            pick["suggested_stake"] = None

    selected.sort(key=lambda c: c["book_odds"])
    return selected


def apply_multi_calibration(sgms: list[dict], calibrator) -> list[dict]:
    """Apply a selection-level ``IsotonicCalibrator`` (model-upgrade audit
    Phase 3.6, e.g. from `afl_bot.backtest.multis.load_or_fit_multi_calibrator`)
    to each rung's joint probability **in place**, recomputing `fair_odds`/
    edge from the calibrated value. Phase 3.5 found `search_match_sgms`'s
    closest-to-target selection is itself a biased estimator (the
    optimizer's curse) and that two prototyped fixes to the selection
    mechanism itself (`lcb_z`, `price_shrink`) didn't help — this instead
    corrects the OUTPUT, fit directly on the walk-forward SELECTED rungs' own
    track record, the same way every other calibrator in this codebase
    works. No-op (returns ``sgms`` unchanged) when ``calibrator`` is ``None``
    -- the opt-in default.

    `search_match_sgms`'s own ``multi_calibrator`` param (FIX-PLACEABLE-LEGS-
    AND-210-FLOOR STEP 4) now folds this same transform in BEFORE selection
    instead, so the round-report live path doesn't call this function
    anymore -- calibrating already-selected rungs after the fact let the
    band-clearing guard pass on the uncalibrated joint, then print a
    calibrated fair odds below the band. This standalone version stays for
    any caller that wants to calibrate an already-built rung list directly
    (e.g. ad-hoc analysis), and for the test below pinning its own
    behaviour."""
    if calibrator is None:
        return sgms
    for s in sgms:
        s["joint_prob"] = float(calibrator.predict([s["joint_prob"]])[0])
        s["fair_odds"] = fair_odds(s["joint_prob"])
        if "book_odds" in s:
            book = s["book_odds"]
            s["raw_edge"] = s["joint_prob"] * book - 1.0
            s["edge"] = market_anchored_prob(s["joint_prob"], book, MULTI_MARKET_SHRINK) * book - 1.0
            s["odds"] = book
    return sgms


def _fmt_pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def render_markdown(year: int, round_no: int, matches: list[dict], *,
                    has_odds: bool, multis_section: str = "", odds_note: str = "",
                    sportsbet_note: str = "", proj_note: str = "",
                    multis_only: bool = False) -> str:
    """Render the round report to markdown. ``matches`` is a list of dicts from
    ``afl_bot.cli.round_report`` (header, team projection rows, sgms,
    market_sgms).

    ``sportsbet_note`` (FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART A6) states the
    ACTUAL player-prop odds source this run -- live Sportsbet scrape, or why
    not (not run from AU / blocked / not requested) -- distinct from
    ``odds_note``'s existing live-h2h/totals-via-Odds-API note.

    When ``multis_only`` is True only the match heading and same-game multi ladder(s)
    are emitted for each fixture — all player-projection tables, margin/win-prob
    bullets, and header notes are skipped."""
    out: list[str] = [f"# AFL Round Report - {year} Round {round_no}", ""]
    if not multis_only:
        out.append("_Real-player projections + same-game multis priced off the "
                   "correlated Monte Carlo sim (joint probability, not naive product)._")
        if proj_note:
            out.append("")
            out.append(proj_note)
        if odds_note:
            out.append("")
            out.append(odds_note)
        if sportsbet_note:
            out.append("")
            out.append(sportsbet_note)
        out.append("")

    for m in matches:
        h = m["header"]
        wet = " |**WET**" if h["is_wet"] else (" |roofed" if h["roofed"] else "")
        out.append(f"## {h['home']} vs {h['away']} - {h['venue']}{wet}")
        if not multis_only:
            out.append(f"- Margin (home): **{h['mu_margin']:+.1f}** |Total: **{h['mu_total']:.0f}**")
            out.append(f"- P({h['home']}) = **{_fmt_pct(h['p_home'])}** |"
                       f"P({h['away']}) = **{_fmt_pct(h['p_away'])}**"
                       + (f" |P(draw) {_fmt_pct(h['p_draw'])}" if h["p_draw"] > 0 else ""))
            out.append(f"- {h['total_line_name']} = **{_fmt_pct(h['p_total'])}**")
            out.append("")

            for team, rows in m["projections"]:
                if not rows:
                    continue
                out.append(f"### {team} - player projections")
                out.append("| Player | Disp | 15+ | 20+ | 25+ | Goals | 1+ | 2+ | Marks | 4+ | Tackles | 3+ |")
                out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
                for row in rows:
                    out.append(
                        f"| {row['player']} "
                        f"| {row.get('disposals_mean', 0):.1f} | {_fmt_pct(row.get('disposals_15+', 0))} "
                        f"| {_fmt_pct(row.get('disposals_20+', 0))} | {_fmt_pct(row.get('disposals_25+', 0))} "
                        f"| {row.get('goals_mean', 0):.1f} | {_fmt_pct(row.get('goals_1+', 0))} "
                        f"| {_fmt_pct(row.get('goals_2+', 0))} "
                        f"| {row.get('marks_mean', 0):.1f} | {_fmt_pct(row.get('marks_4+', 0))} "
                        f"| {row.get('tackles_mean', 0):.1f} | {_fmt_pct(row.get('tackles_3+', 0))} |"
                    )
                out.append("")

            priced = m.get("priced_legs") or []
            if priced:
                out.append("### Priced props (from --odds)")
                out.append("| Leg | Model | Book | Devig | Blended | Edge | Class |")
                out.append("|---|--:|--:|--:|--:|--:|---|")
                for p in priced:
                    book_str = f"{p['book_odds']:.2f}" if p["book_odds"] else "-"
                    devig_str = (f"{_fmt_pct(p['devig_prob'])} ({p['devig_label']})"
                                 if p["devig_prob"] is not None else "-")
                    blended_str = _fmt_pct(p["blended_prob"]) if p.get("blended_prob") is not None else "-"
                    out.append(
                        f"| {p['name']} | {_fmt_pct(p['model_prob'])} | {book_str} "
                        f"| {devig_str} | {blended_str} "
                        f"| {p['edge_pct'] * 100:+.1f}% | {p['classification']} |"
                    )
                out.append("")

        out.append("### Model ladder (model fair odds, no book)")
        if not m["sgms"]:
            n_legs = m.get("n_legs")
            if n_legs is not None and n_legs < 3:
                out.append(f"_Only {n_legs} candidate legs for this match - need >=3 "
                           "non-conflicting for a multi (likely missing lineup/odds, not a drop)._")
            else:
                out.append("_No 3-leg combination cleared the probability floor for this match._")
        else:
            header = "| Legs | Band | Joint prob | Fair odds | Corr gain |"
            sep = "|---|--:|--:|--:|--:|"
            if has_odds:
                header += " Book | Edge | Total EV | Stake | Units | $ |"
                sep += "--:|--:|--:|--:|--:|--:|"
            header += " Pick |"
            sep += "---|"
            out.append(header)
            out.append(sep)
            for s in m["sgms"]:
                band = f"${s['target_odds']:.2f}" if "target_odds" in s else "-"
                line = (f"| {' + '.join(s['legs'])} | {band} | {_fmt_pct(s['joint_prob'])} "
                        f"| {s['fair_odds']:.2f} | {s['corr_gain'] * 100:+.1f}pp |")
                if has_odds:
                    if "book_odds" in s:
                        tev = s.get("total_ev")
                        tev_str = f"{tev * 100:+.1f}%" if tev is not None else "—"
                        stk = s.get("suggested_stake")
                        stk_str = f"{stk * 100:.1f}%" if stk is not None else "—"
                        units_tag = s.get("units_tag", "—")
                        units_val = s.get("units", 0.0)
                        dollar_str = f"${units_val * UNIT_SIZE:.2f}" if units_val > 0 else "—"
                        line += (f" {s['book_odds']:.2f} | {s['edge'] * 100:+.1f}%"
                                 f" | {tev_str} | {stk_str} | {units_tag} | {dollar_str} |")
                    else:
                        line += " - | - | - | - | — | — |"
                if s.get("value_pick"):
                    line += " **VALUE PICK** |"
                elif "book_odds" not in s:
                    line += " _(model-only — verify market exists)_ |"
                else:
                    line += "  |"
                out.append(line)
        out.append("")

        # FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART C2/C4: only when this
        # match has at least one fully-priced combo (real Sportsbet/--odds
        # prices) -- otherwise stay silent, the model ladder above already
        # carries the "(model-only — verify market exists)" tags.
        market_sgms = m.get("market_sgms") or []
        if market_sgms:
            out.append("### Sportsbet ladder (real prices)")
            out.append("_Sportsbet same-game multi specials are priced with the book's own "
                       "correlation model and will differ from the leg-product shown here "
                       "(only individual legs are scraped, not the SGM special itself) — "
                       "use the per-leg book prices as the source of truth._")
            out.append("")
            out.append("| Legs | Book odds (combo) | Model joint % | Model fair | Edge | Total EV | Stake | Units | $ | Pick |")
            out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
            for s in market_sgms:
                tev = s.get("total_ev")
                tev_str = f"{tev * 100:+.1f}%" if tev is not None else "—"
                stk = s.get("suggested_stake")
                stk_str = f"{stk * 100:.1f}%" if stk is not None else "—"
                units_tag = s.get("units_tag", "—")
                units_val = s.get("units", 0.0)
                dollar_str = f"${units_val * UNIT_SIZE:.2f}" if units_val > 0 else "—"
                line = (f"| {' + '.join(s['legs'])} | {s['book_odds']:.2f} "
                        f"| {_fmt_pct(s['joint_prob'])} | {s['fair_odds']:.2f} "
                        f"| {s['edge'] * 100:+.1f}% | {tev_str} | {stk_str} "
                        f"| {units_tag} | {dollar_str} |")
                line += " **VALUE PICK** |" if s.get("value_pick") else "  |"
                out.append(line)
            out.append("")

        # ── PointsBet Pull 'Em block ──────────────────────────────────────
        pull_em = m.get("pull_em")
        if pull_em:
            out.append("### PointsBet Pull 'Em")
            pe_units_tag = pull_em.get("units_tag", "—")
            pe_units_val = pull_em.get("units", 0.0)
            pe_dollar = f"${pe_units_val * UNIT_SIZE:.2f}" if pe_units_val > 0 else "—"
            out.append(f"_Book combo: ${pull_em['book_combo']:.2f} · "
                       f"Option EV (assumed prior): **{pull_em['option_ev']:+.2f}%** · "
                       f"Stake: **{pe_units_tag}** ({pe_dollar})_")
            out.append("")
            out.append("| Leg | Role | Prob | Leg odds |")
            out.append("|---|---|--:|--:|")
            anchor_set = set(pull_em["anchor_names"])
            all_probs = pull_em["anchor_probs"] + [pull_em["booster_prob"]]
            for name, prob, book_o in zip(
                pull_em["leg_names"], all_probs, pull_em["book_odds_per_leg"]
            ):
                role = "Anchor" if name in anchor_set else "Booster"
                out.append(f"| {name} | {role} | {prob * 100:.0f}% | {book_o:.2f} |")
            out.append("")
            out.append("**Option EV breakdown** _(PULL_DETECTION_PROB=0.70 — assumed prior, not fitted)_:")
            out.append("")
            out.append("| Pulled leg | P(others hit) | P(miss) | Reduced odds | EV contrib |")
            out.append("|---|--:|--:|--:|--:|")
            for b in pull_em["option_ev_breakdown"]:
                out.append(f"| {b['leg']} | {b['p_others_hit'] * 100:.1f}%"
                           f" | {b['p_miss'] * 100:.1f}%"
                           f" | {b['reduced_odds']:.2f}"
                           f" | {b['option_ev_contrib']:+.3f}% |")
            out.append("")
            out.append(f"**Pull decision rule:** {pull_em['pull_decision_rule']}")
            out.append("")

    if multis_section:
        out.append(multis_section)
    out.append("\n_Modelling tool only - even a well-calibrated model loses regularly. "
               "Gambling Help Online 1800 858 858._")
    return "\n".join(out)


def build_pull_em_sgm(
    legs: list[LegCandidate],
    *,
    odds_book: dict[str, float],
    pull_detection_prob: float = PULL_DETECTION_PROB,
    anchor_min_p: float = PULL_EM_ANCHOR_MIN_P,
    booster_min_p: float = PULL_EM_BOOSTER_MIN_P,
    booster_max_p: float = PULL_EM_BOOSTER_MAX_P,
) -> dict | None:
    """Build the best PointsBet Pull 'Em SGM for a match.

    Selects 3 disposal anchor legs (model_prob >= anchor_min_p, one per player,
    must be priced in odds_book) and 1 booster leg (booster_min_p <= prob <=
    booster_max_p, different player, respects <=1 tackle/marks cap).

    Option EV per leg = sum over each leg as the "pulled" one:
        P(other 3 win) × P(that leg misses) × PULL_DETECTION_PROB
            × (book_combo / missed_leg_book_odds - 1)

    where book_combo = product of all 4 leg book_odds.

    Returns a dict with keys:
        legs, leg_names, anchor_probs, booster_prob,
        book_combo, option_ev, option_ev_breakdown,
        pull_decision_rule
    or None if a valid 4-leg combo cannot be assembled.

    PULL_DETECTION_PROB is an ASSUMED PRIOR (not fitted). All EV figures
    are labelled as estimates under this prior.
    """
    # Candidate pools: only priced legs
    priced = [l for l in legs if l.name in odds_book]

    disposals = [
        l for l in priced
        if l.market in ("player_disposals", "disposals")
        and l.fair_prob >= anchor_min_p
        and l.fair_prob <= 0.78  # LEG_PROB_MAX
    ]
    # One anchor per player (pick highest-prob line per player)
    by_player: dict[str, LegCandidate] = {}
    for l in disposals:
        if l.subject not in by_player or l.fair_prob > by_player[l.subject].fair_prob:
            by_player[l.subject] = l
    anchor_pool = list(by_player.values())

    if len(anchor_pool) < 3:
        return None

    boosters = [
        l for l in priced
        if booster_min_p <= l.fair_prob <= booster_max_p
        and l.fair_prob <= 0.78
    ]

    best: dict | None = None

    for anchor_combo in combinations(anchor_pool, 3):
        # One player per leg
        if len({l.subject for l in anchor_combo}) < 3:
            continue
        anchor_names = [l.name for l in anchor_combo]

        for booster in boosters:
            # Different player from all anchors
            if booster.subject in {l.subject for l in anchor_combo}:
                continue
            # Tackle/marks cap: at most 1 of the 4 legs can be tackles/marks
            n_tm = sum(
                1 for l in (*anchor_combo, booster)
                if l.market in ("player_tackles", "player_marks", "tackles", "marks")
            )
            if n_tm > 1:
                continue

            all_legs = [*anchor_combo, booster]
            book_odds_vals = [odds_book[l.name] for l in all_legs]
            book_combo = 1.0
            for o in book_odds_vals:
                book_combo *= o

            # Option EV: for each leg as "miss", compute the partial EV
            option_ev_breakdown = []
            total_option_ev = 0.0
            for i, miss_leg in enumerate(all_legs):
                others = [l for j, l in enumerate(all_legs) if j != i]
                # P(all others hit) × P(miss_leg misses)
                p_others_hit = 1.0
                for ol in others:
                    p_others_hit *= ol.fair_prob
                p_miss = 1.0 - miss_leg.fair_prob
                miss_book = book_odds_vals[i]
                reduced_odds = book_combo / miss_book  # payout if that leg is "pulled"
                leg_option_ev = (p_others_hit * p_miss
                                 * pull_detection_prob
                                 * (reduced_odds - 1.0))
                option_ev_breakdown.append({
                    "leg": miss_leg.name,
                    "p_others_hit": round(p_others_hit, 4),
                    "p_miss": round(p_miss, 4),
                    "reduced_odds": round(reduced_odds, 2),
                    "option_ev_contrib": round(leg_option_ev * 100, 3),
                })
                total_option_ev += leg_option_ev

            # Promo branch probs for multi-outcome Kelly sizing.
            # p_win = all 4 legs hit (independence of model probs).
            p_win_pe = booster.fair_prob
            for al in anchor_combo:
                p_win_pe *= al.fair_prob
            # p_one_miss = Σ P(exactly one anchor misses AND pull triggered).
            p_one_miss_pe = sum(
                b["p_others_hit"] * b["p_miss"] * pull_detection_prob
                for b in option_ev_breakdown
            )
            p_dead_pe = max(0.0, 1.0 - p_win_pe - p_one_miss_pe)
            # Weighted average recovery odds (R_eff): expected return per unit when pulled.
            if p_one_miss_pe > 0:
                R_eff = sum(
                    b["p_others_hit"] * b["p_miss"] * pull_detection_prob * b["reduced_odds"]
                    for b in option_ev_breakdown
                ) / p_one_miss_pe
            else:
                R_eff = 1.0

            # Score by option_ev (primary) then sum of anchor probs (secondary)
            score = (total_option_ev, sum(l.fair_prob for l in anchor_combo))
            if best is None or score > best["_score"]:
                best = {
                    "legs": all_legs,
                    "leg_names": [l.name for l in all_legs],
                    "anchor_names": anchor_names,
                    "booster_name": booster.name,
                    "anchor_probs": [round(l.fair_prob, 4) for l in anchor_combo],
                    "booster_prob": round(booster.fair_prob, 4),
                    "book_odds_per_leg": book_odds_vals,
                    "book_combo": round(book_combo, 2),
                    "option_ev": round(total_option_ev * 100, 3),
                    "option_ev_breakdown": option_ev_breakdown,
                    "promo_p_win": round(p_win_pe, 6),
                    "promo_p_one_miss": round(p_one_miss_pe, 6),
                    "promo_p_dead": round(p_dead_pe, 6),
                    "promo_R_eff": round(R_eff, 4),
                    "pull_decision_rule": (
                        f"Pull if >=1 anchor misses AND the other 3 are all winning "
                        f"(assumed P(pull triggered)={pull_detection_prob:.0%} — PRIOR, not fitted)."
                    ),
                    "_score": score,
                }

    if best:
        del best["_score"]
    return best

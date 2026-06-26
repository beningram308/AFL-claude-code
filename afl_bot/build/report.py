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

from afl_bot.backtest.props import apply_prop_calibration
from afl_bot.build.multi import LegCandidate, _no_conflicts, combined_odds, joint_prob_from_masks
from afl_bot.config import (
    BOOKABLE_MARKS_ROLES,
    BOOKABLE_PROP_MENU,
    BOOKABLE_TACKLES_ROLES,
    BOOKABLE_TOP_N_BY_STAT,
    MULTI_MARKET_SHRINK,
    MULTI_TARGET_ODDS,
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
            }
            if all(odds_book.get(leg.name) is not None for leg in legs_list):
                book = combined_odds(legs_list)
                shrunk = market_anchored_prob(joint, book, MULTI_MARKET_SHRINK)
                entry["book_odds"] = book
                entry["raw_edge"] = joint * book - 1.0
                entry["edge"] = shrunk * book - 1.0
            entry["odds"] = entry.get("book_odds", entry["fair_odds"])
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
        if lcb_z <= 0.0:
            reaches_target = [c for c in available if c["_priced_joint"] <= 1.0 / target]
            if reaches_target:
                return max(reaches_target, key=lambda c: (c["_priced_joint"], -_distance(c, target)))
        return min(available, key=lambda c: (_distance(c, target), -_lcb_value(c)))

    selected: list[dict] = []
    chosen: set[int] = set()
    for i, target in enumerate(target_odds):
        available = [c for c in combos if id(c) not in chosen] or list(combos)
        is_top = (i == len(target_odds) - 1)
        if is_top:
            valued = [c for c in available
                      if c.get("_priced_edge") is not None and 0.0 < c["_priced_edge"] <= max_plausible_edge]
            if valued:
                valued.sort(key=lambda c: c["_priced_edge"], reverse=True)
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
        pick["joint_prob"] = pick.pop("_priced_joint")
        pick["fair_odds"] = pick.pop("_priced_fair_odds")
        if "book_odds" in pick:
            pick["raw_edge"] = pick.pop("_priced_raw_edge")
            pick["edge"] = pick.pop("_priced_edge")
            pick["odds"] = pick["book_odds"]

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
    priced = [c for c in combos if "book_odds" in c]
    if not priced:
        return []

    def _select_for_target(available: list[dict], target: float) -> dict:
        reaches = [c for c in available if c["book_odds"] >= target]
        if reaches:
            return min(reaches, key=lambda c: c["book_odds"])
        return max(available, key=lambda c: c["book_odds"])

    selected: list[dict] = []
    chosen: set[int] = set()
    for i, target in enumerate(target_odds):
        available = [c for c in priced if id(c) not in chosen] or list(priced)
        is_top = (i == len(target_odds) - 1)
        if is_top:
            valued = [c for c in available if 0.0 < c["edge"] <= max_plausible_edge]
            if valued:
                pick = max(valued, key=lambda c: c["edge"])
                pick["value_pick"] = True
            else:
                pick = _select_for_target(available, target)
        else:
            pick = _select_for_target(available, target)
        pick["target_odds"] = target
        chosen.add(id(pick))
        selected.append(pick)

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
                header += " Book | Edge |"
                sep += "--:|--:|"
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
                        line += f" {s['book_odds']:.2f} | {s['edge'] * 100:+.1f}% |"
                    else:
                        line += " - | - |"
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
            out.append("| Legs | Book odds (combo) | Model joint % | Model fair | Edge | Pick |")
            out.append("|---|--:|--:|--:|--:|---|")
            for s in market_sgms:
                line = (f"| {' + '.join(s['legs'])} | {s['book_odds']:.2f} "
                        f"| {_fmt_pct(s['joint_prob'])} | {s['fair_odds']:.2f} "
                        f"| {s['edge'] * 100:+.1f}% |")
                line += " **VALUE PICK** |" if s.get("value_pick") else "  |"
                out.append(line)
            out.append("")

    if multis_section:
        out.append(multis_section)
    out.append("\n_Modelling tool only - even a well-calibrated model loses regularly. "
               "Gambling Help Online 1800 858 858._")
    return "\n".join(out)

"""
Stake-cap backtest (DO-STAKE-CAP-BACKTEST, 2026-07-10) -- diagnostic only.

Question: Ben's staking is quarter-Kelly with a hard per-bet cap
UNIT_MAX = 3u. On the strongest promo multis, quarter-Kelly wants MORE than
3u and clips to the cap, so the strongest bets all look identical. Is 3u the
right ceiling, or would 1.5u / 2u / 4u have grown the bankroll more (and at
what drawdown)?

This module is read-only and reimplements NOTHING that the live bot uses to
price, size, or settle a bet:
  * Re-sizing at a candidate UNIT_MAX goes through the EXACT live
    `afl_bot.build.staking.recommend_units` with overridden kwargs -- its body
    is never touched.
  * Leg grading (did a prop/total hit?) reuses `afl_bot.dashboard.settle`'s
    `_load_actuals` / `_settle_leg` -- the same code `settle-bets` uses on the
    real ledger -- so there are not two drifting versions of "did this leg
    win" in the codebase.
  * It only reads saved `reports/*_multis.json` files (each one a point-in-time
    snapshot of what the model knew when `round-report` ran that week) and
    real Squiggle/Fryzigg/DFS results. It writes exactly one new report file
    and touches no config, staking, or pricing code.

Two ways of asking "was 3u right":
  Version A -- REALIZED REPLAY: what literally happened, resettled at each
    candidate cap. Small-sample and luck-heavy by nature (see `discover_rounds`
    -- as of 2026-07-10 there are only two usable rounds).
  Version B -- PROBABILISTIC BANKROLL SIM: Monte Carlo off each bet's own
    modelled branch probabilities (p_win / p_one_loss / p_dead), so it reflects
    the underlying edge rather than which few bets happened to land -- valid
    only to the extent the model is calibrated (see the hit-rate cross-check).

Explicitly OUT of scope (see DO-STAKE-CAP-BACKTEST.txt's "optional" tag):
  * A KELLY_FRACTION sweep. `recommend_units` does not expose the Kelly
    fraction as a parameter (it's baked into `fractional_kelly_fraction` /
    `multi_outcome_kelly`'s own defaults), and reimplementing the frac-to-units
    conversion with a different fraction would be exactly the kind of parallel
    settlement/sizing logic the hard rules for this backtest forbid. Skipped.
  * Pull 'Em. Zero rounds in `reports/` currently carry a real (non-null
    book_combo) Pull 'Em record -- there is nothing to backtest.
"""

from __future__ import annotations

import glob
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from afl_bot.build.staking import fractional_kelly_fraction, multi_outcome_kelly, recommend_units
from afl_bot.config import (
    BANKROLL, BONUS_BET_FACTOR, PROMO_REFUND_CAP, ROOT_DIR, UNIT_MAX,
    UNIT_MAX_LONGSHOT, UNIT_SIZE, UNIT_STEP,
)
from afl_bot.dashboard.settle import _load_actuals, _settle_leg
from afl_bot.io_utils import atomic_write_text

REPORTS_DIR = ROOT_DIR / "reports"

DEFAULT_UNIT_MAX_CANDIDATES: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0)

# The removed cli.py::_apply_round_cap's budget, reimplemented HERE ONLY for
# the optional "what does removing the round cap cost/earn" comparison in the
# report -- the live bot no longer has any round-level cap (2026-07-10).
_OLD_KELLY_PER_ROUND_CAP = 0.15
OLD_ROUND_CAP_UNITS = _OLD_KELLY_PER_ROUND_CAP * BANKROLL / UNIT_SIZE  # 15u


def _load_multis_records(path: Path) -> list[dict]:
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("records", [])


def discover_rounds() -> list[tuple[int, int]]:
    """Sorted (year, round) pairs with a saved multis.json carrying at least
    one real book_combo AND a completed (gradeable) game result."""
    from afl_bot.data.squiggle import SquiggleClient

    client = SquiggleClient()
    completed_cache: dict[int, set[int]] = {}
    found: list[tuple[int, int]] = []

    for p in sorted(glob.glob(str(REPORTS_DIR / "*_multis.json"))):
        stem = Path(p).stem.replace("_multis", "")  # e.g. "2026_r18"
        if "_r" not in stem:
            continue
        year_s, _, round_s = stem.partition("_r")
        try:
            year, round_no = int(year_s), int(round_s)
        except ValueError:
            continue
        records = _load_multis_records(Path(p))
        if not any(r.get("book_combo") is not None for r in records):
            continue
        if year not in completed_cache:
            games = client.get_completed_games(year)
            completed_cache[year] = set(int(r) for r in games["round"].unique())
        if round_no not in completed_cache[year]:
            continue
        found.append((year, round_no))
    return sorted(found)


@dataclass
class GradedRung:
    year: int
    round_no: int
    game: str
    ladder: str
    band: float | None
    joint_prob: float | None
    book_odds: float | None
    promo_ev: float | None
    total_ev: float | None
    p_win: float | None
    p_one_loss: float | None
    p_dead: float | None
    outcome: str  # "win" | "one_miss" | "dead"
    players: tuple[str, ...] = ()  # distinct player-prop players in this rung's legs
    # (DO-EDGE-FLOOR-VIABILITY-TEST 2026-08-14) -- excludes h2h (player field is a team
    # name there) and total_points (player field is the placeholder "total") legs, since
    # the per-player staking cap (policy C) is about not stacking the same PERSON's props
    # across multiple staked multis in a round, not team/match-level exposure.

    @property
    def date_key(self) -> tuple[int, int]:
        return (self.year, self.round_no)

    @property
    def raw_edge(self) -> float | None:
        """joint_prob * book_odds - 1 -- exactly the quantity `recommend_units`'s
        plain-Kelly branch (`fractional_kelly_fraction`) is positive/zero/negative on.
        This is "raw edge" throughout this module: positive iff the rung would be
        staked WITHOUT any promo/refund reasoning at all."""
        if self.joint_prob is None or self.book_odds is None:
            return None
        return self.joint_prob * self.book_odds - 1.0


def grade_leg_outcomes(legs: list[dict], game: str, h2h_actual: dict,
                       total_actual: dict, player_stat: dict,
                       year: int, round_no: int, *,
                       grade_total_points: bool = True) -> str | None:
    """Grade every leg of one rung via the live `_settle_leg`. Returns
    'win' / 'one_miss' / 'dead', or None if any leg is still ungradeable
    (excluded from the backtest -- never guessed).

    `grade_total_points` (DO-EDGE-FLOOR-VIABILITY-TEST 2026-08-14, default True):
    when False, any leg with market "total_points" is forced ungradeable
    regardless of whether `total_actual` actually has the data -- this
    reproduces the exact blind spot in `../audit_2026-08-14`'s independent
    re-grading script, which (per its own methodology note) only read player
    box scores and never sourced match-total actuals at all, so every
    total_points leg came back "no game total yet" there even on completed
    games. `_settle_leg` itself parses the real threshold out of the leg's
    `name` string (e.g. "Total points 175.5+") via regex, NOT the `line` key
    -- `line` is a placeholder (always 5) for this market in the saved
    multis.json, so the threshold literally isn't available anywhere else."""
    n_miss = 0
    for leg in legs:
        if not grade_total_points and leg.get("market") == "total_points":
            return None
        leg_with_game = {**leg, "game": game}
        hit, _reason = _settle_leg(leg_with_game, h2h_actual, total_actual, player_stat,
                                   year, round_no)
        if hit is None:
            return None
        if hit is False:
            n_miss += 1
    if n_miss == 0:
        return "win"
    if n_miss == 1:
        return "one_miss"
    return "dead"


# Player-prop markets a rung's legs can carry -- used to pull out the distinct
# players a rung exposes for the policy-C per-player cap. Deliberately excludes
# h2h (the "player" field holds a team name) and total_points (placeholder
# player "total") -- see GradedRung.players.
_PLAYER_PROP_MARKETS = frozenset({
    "disposals", "goals", "marks", "tackles",
    "player_disposals", "player_goals", "player_marks", "player_tackles",
    "player_kicks", "player_fantasy",
})


def _extract_players(legs: list[dict]) -> tuple[str, ...]:
    names: list[str] = []
    for leg in legs:
        if leg.get("market") in _PLAYER_PROP_MARKETS:
            player = (leg.get("player") or "").strip()
            if player and player not in names:
                names.append(player)
    return tuple(names)


def grade_rounds(rounds: list[tuple[int, int]], *,
                 grade_total_points: bool = True) -> tuple[list[GradedRung], int]:
    """Grade every real-priced, staked-by-the-live-formula-eligible rung
    (model + sportsbet ladders; pull_em has no real data to grade, see module
    docstring) across the given rounds. Returns (graded rungs in chronological
    order, n_excluded_for_an_unresolved_leg).

    `grade_total_points` -- see `grade_leg_outcomes`; threaded through here so
    a caller can reproduce the audit's re-grading scope (False) alongside the
    full live-grading scope (True, the default)."""
    graded: list[GradedRung] = []
    n_excluded = 0

    for year, round_no in rounds:
        path = REPORTS_DIR / f"{year}_r{round_no}_multis.json"
        records = _load_multis_records(path)
        h2h_actual, total_actual, player_stat = _load_actuals(year, round_no)

        for r in records:
            if r.get("ladder") not in ("model", "sportsbet"):
                continue
            if r.get("no_bet") or r.get("book_combo") is None:
                continue
            legs = r.get("legs", [])
            outcome = grade_leg_outcomes(
                legs, r.get("game", ""), h2h_actual, total_actual,
                player_stat, year, round_no, grade_total_points=grade_total_points,
            )
            if outcome is None:
                n_excluded += 1
                continue
            p_win = r.get("p_all_win")
            p_one_loss = r.get("p_one_loss")
            p_dead = (1.0 - p_win - p_one_loss) if (p_win is not None and p_one_loss is not None) else None
            graded.append(GradedRung(
                year=year, round_no=round_no, game=r.get("game", ""), ladder=r["ladder"],
                band=r.get("band"), joint_prob=r.get("model_joint"), book_odds=r.get("book_combo"),
                promo_ev=r.get("promo_ev"), total_ev=r.get("total_ev"),
                p_win=p_win, p_one_loss=p_one_loss, p_dead=p_dead, outcome=outcome,
                players=_extract_players(legs),
            ))
    return graded, n_excluded


@dataclass
class SizedBet:
    rung: GradedRung
    units: float
    tag: str
    stake: float  # dollars

    @property
    def is_promo(self) -> bool:
        return "PROMO KELLY" in self.tag


def size_rungs(graded: list[GradedRung], unit_max: float, *,
              unit_max_longshot: float = UNIT_MAX_LONGSHOT,
              promo_refund_cap: float = PROMO_REFUND_CAP,
              bankroll: float = BANKROLL, unit_size: float = UNIT_SIZE) -> list[SizedBet]:
    """Re-size every graded rung at ``unit_max`` via the LIVE `recommend_units`
    (no reimplementation) and keep only the ones that actually stake."""
    sized = []
    for g in graded:
        units, tag = recommend_units(
            g.joint_prob, g.book_odds, g.promo_ev, total_ev=g.total_ev,
            p_win=g.p_win, p_one_loss=g.p_one_loss, p_dead=g.p_dead,
            bankroll=bankroll, unit_size=unit_size, unit_max=unit_max,
            unit_max_longshot=unit_max_longshot, promo_refund_cap=promo_refund_cap,
        )
        if units > 0:
            sized.append(SizedBet(rung=g, units=units, tag=tag, stake=units * unit_size))
    return sized


def apply_old_round_cap(sized: list[SizedBet], cap_units: float = OLD_ROUND_CAP_UNITS) -> list[SizedBet]:
    """Optional comparison only: reimplements the REMOVED
    cli.py::_apply_round_cap allocator (rank by total_ev desc, fill the round
    budget, trim the overflow rung) per (year, round). This function is not
    called anywhere in the live bot -- it exists solely so this backtest can
    show what the 2026-07-10 round-cap removal costs or earns."""
    by_round: dict[tuple[int, int], list[SizedBet]] = defaultdict(list)
    for s in sized:
        by_round[s.rung.date_key].append(s)

    kept: list[SizedBet] = []
    for bets in by_round.values():
        bets_sorted = sorted(bets, key=lambda s: (s.rung.total_ev or 0.0), reverse=True)
        budget = cap_units
        for s in bets_sorted:
            if budget <= 1e-9:
                continue
            if s.units <= budget + 1e-9:
                kept.append(s)
                budget -= s.units
            else:
                trimmed = math.floor(budget / UNIT_STEP) * UNIT_STEP
                if trimmed >= UNIT_STEP:
                    kept.append(SizedBet(rung=s.rung, units=trimmed, tag=s.tag, stake=trimmed * UNIT_SIZE))
                budget = 0.0
    return kept


def settle_dollar(bet: SizedBet, refund_factor: float = BONUS_BET_FACTOR) -> float:
    """Net profit ($) for one settled bet. Promo (stake-back) rungs get a
    partial refund on exactly-one-leg-missed; straight edge rungs and any
    2+-leg miss just lose the stake."""
    if bet.rung.outcome == "win":
        return bet.stake * (bet.rung.book_odds - 1.0)
    if bet.rung.outcome == "one_miss" and bet.is_promo:
        return -(1.0 - refund_factor) * bet.stake
    return -bet.stake


@dataclass
class ReplayResult:
    unit_max: float
    n_bets: int
    total_staked: float
    total_returned: float
    net_profit: float
    roi_pct: float
    end_bankroll: float
    max_drawdown_pct: float


def realized_replay(sized: list[SizedBet], *, bankroll0: float = BANKROLL) -> ReplayResult:
    """Version A: chain the given (already-sized) bets in date order into one
    bankroll curve and settle each against what actually happened."""
    ordered = sorted(sized, key=lambda s: s.rung.date_key)
    bankroll = bankroll0
    peak = bankroll0
    max_dd = 0.0
    total_staked = 0.0
    total_returned = 0.0
    for s in ordered:
        net = settle_dollar(s)
        returned = s.stake + net
        total_staked += s.stake
        total_returned += returned
        bankroll += net
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    net_profit = total_returned - total_staked
    roi = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    return ReplayResult(
        unit_max=float("nan"), n_bets=len(ordered), total_staked=total_staked,
        total_returned=total_returned, net_profit=net_profit, roi_pct=roi,
        end_bankroll=bankroll, max_drawdown_pct=max_dd * 100.0,
    )


@dataclass
class SimResult:
    mode: str  # "fixed" | "compounding"
    median_end: float
    mean_end: float
    p5_end: float
    p95_end: float
    median_max_dd_pct: float
    p_down: float
    p_dd_over_20: float


def probabilistic_sim(sized: list[SizedBet], *, n_sims: int = 10_000,
                      bankroll0: float = BANKROLL, seed: int = 42) -> dict[str, SimResult]:
    """Version B: Monte Carlo bankroll paths off each bet's OWN modelled
    branch probabilities (not what actually happened). Two staking modes:
    'fixed' (constant dollar stake, matching live behaviour) and
    'compounding' (stake the same % of bankroll, recomputed off the running
    balance each bet -- a truer Kelly-growth read)."""
    ordered = sorted(sized, key=lambda s: s.rung.date_key)
    if not ordered:
        empty = SimResult("fixed", bankroll0, bankroll0, bankroll0, bankroll0, 0.0, 0.0, 0.0)
        return {"fixed": empty, "compounding": SimResult("compounding", bankroll0, bankroll0,
                                                           bankroll0, bankroll0, 0.0, 0.0, 0.0)}

    rng = np.random.default_rng(seed)
    n_bets = len(ordered)
    p_win = np.array([s.rung.p_win if s.rung.p_win is not None else 0.0 for s in ordered])
    p_one_loss = np.array([s.rung.p_one_loss if s.rung.p_one_loss is not None else 0.0 for s in ordered])
    book_odds = np.array([s.rung.book_odds for s in ordered])
    is_promo = [s.is_promo for s in ordered]
    stake_dollars = np.array([s.stake for s in ordered])
    stake_fraction = stake_dollars / bankroll0

    results: dict[str, SimResult] = {}
    for mode in ("fixed", "compounding"):
        bankroll = np.full(n_sims, float(bankroll0))
        peak = bankroll.copy()
        max_dd = np.zeros(n_sims)
        for i in range(n_bets):
            draw = rng.random(n_sims)
            win_mask = draw < p_win[i]
            one_loss_mask = (~win_mask) & (draw < p_win[i] + p_one_loss[i])

            stake = np.full(n_sims, stake_dollars[i]) if mode == "fixed" else stake_fraction[i] * bankroll
            one_loss_net = -(1.0 - BONUS_BET_FACTOR) * stake if is_promo[i] else -stake
            net = np.where(win_mask, stake * (book_odds[i] - 1.0),
                          np.where(one_loss_mask, one_loss_net, -stake))

            bankroll = np.clip(bankroll + net, 0.0, None)
            peak = np.maximum(peak, bankroll)
            dd = np.where(peak > 0, (peak - bankroll) / peak, 0.0)
            max_dd = np.maximum(max_dd, dd)

        results[mode] = SimResult(
            mode=mode, median_end=float(np.median(bankroll)), mean_end=float(np.mean(bankroll)),
            p5_end=float(np.percentile(bankroll, 5)), p95_end=float(np.percentile(bankroll, 95)),
            median_max_dd_pct=float(np.median(max_dd) * 100.0),
            p_down=float(np.mean(bankroll < bankroll0)),
            p_dd_over_20=float(np.mean(max_dd > 0.20)),
        )
    return results


def hit_rate_cross_check(graded: list[GradedRung]) -> dict:
    """Modelled hit-rate (mean p_win over gradeable bets) vs the ACTUAL
    realized hit-rate (fraction that landed 'win'). The bet SET (which rungs
    are eligible) doesn't depend on unit_max, only stake size does -- so this
    is computed once, not per cap."""
    with_probs = [g for g in graded if g.p_win is not None]
    return {
        "n": len(graded),
        "n_with_probs": len(with_probs),
        "modelled_hit_rate": float(np.mean([g.p_win for g in with_probs])) if with_probs else None,
        "actual_hit_rate": float(np.mean([g.outcome == "win" for g in graded])) if graded else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# DO-EDGE-FLOOR-VIABILITY-TEST (2026-08-14) -- TASK 1, measurement only.
# Everything below reuses `recommend_units` / `fractional_kelly_fraction` /
# `multi_outcome_kelly` from `afl_bot.build.staking` verbatim for every Kelly
# calculation -- no sizing math is reimplemented, only which rungs are
# ELIGIBLE and how a round's stakes get deduped/trimmed/settled. Nothing here
# is imported by afl_bot/build/staking.py, afl_bot/build/report.py, or
# afl_bot/cli.py -- the live staking path is untouched.
# ══════════════════════════════════════════════════════════════════════════

def size_rungs_edge_floor(graded: list[GradedRung], unit_max: float, *,
                          unit_max_longshot: float = UNIT_MAX_LONGSHOT,
                          bankroll: float = BANKROLL, unit_size: float = UNIT_SIZE,
                          refund_factor: float = BONUS_BET_FACTOR) -> list[SizedBet]:
    """POLICY B -- EDGE FLOOR. A rung is eligible only if `raw_edge > 0` (the
    same `joint_prob*book_odds > 1` test `recommend_units`'s plain-Kelly
    branch uses). Promo/refund value may only ever SIZE an eligible rung
    larger -- it can never be the reason a rung is eligible; the live
    `recommend_units`'s total_ev > PROMO_EV_MIN "NO BET" -> "PROMO KELLY"
    branch is never reached here for a raw_edge <= 0 rung, full stop.

    For eligible rungs the stake is the LARGER of:
      - the plain fractional-Kelly stake on the outright win/lose outcome
        (`fractional_kelly_fraction(joint_prob, book_odds)` -- ignores promo
        entirely; this is exactly `recommend_units`'s "Nu" branch), and
      - the promo-aware multi-outcome Kelly stake (`multi_outcome_kelly`,
        which prices in the one-miss stake-back refund) when branch
        probabilities are available.
    Rounding/caps (unit_step, unit_max/unit_max_longshot) match
    `recommend_units` exactly. `PROMO_REFUND_CAP` is deliberately NOT applied
    here -- in the live bot that $-cap only bites the promo-ONLY "PROMO
    KELLY" branch for rungs with no plain edge at all; these rungs already
    have one."""
    sized = []
    for g in graded:
        edge = g.raw_edge
        if edge is None or edge <= 0.0:
            continue
        frac = fractional_kelly_fraction(g.joint_prob, g.book_odds)
        promo_boosted = False
        if g.p_win is not None and g.p_one_loss is not None and g.p_dead is not None:
            frac_promo = multi_outcome_kelly(g.p_win, g.p_one_loss, g.p_dead, g.book_odds, refund_factor)
            if frac_promo > frac:
                frac = frac_promo
                promo_boosted = True
        raw_units = frac * bankroll / unit_size
        if raw_units <= 0.0:
            continue
        cap = unit_max_longshot if g.book_odds >= 5.0 else unit_max
        units = min(math.floor(raw_units / UNIT_STEP) * UNIT_STEP, cap)
        units = max(units, UNIT_STEP)
        tag = f"{units:g}u (edge floor, +promo)" if promo_boosted else f"{units:g}u (edge floor)"
        sized.append(SizedBet(rung=g, units=units, tag=tag, stake=units * unit_size))
    return sized


def apply_per_player_cap(sized: list[SizedBet]) -> list[SizedBet]:
    """POLICY C's per-player cap: at most one staked multi per player per
    round. When a player appears in more than one staked rung in the same
    round, keep only the rung with the highest raw edge and drop every other
    rung that shares ANY of its players -- greedy, single pass, highest-edge
    rung first (a dropped rung's OTHER players don't get a second chance to
    anchor a different kept rung)."""
    by_round: dict[tuple[int, int], list[SizedBet]] = defaultdict(list)
    for s in sized:
        by_round[s.rung.date_key].append(s)

    kept: list[SizedBet] = []
    for round_bets in by_round.values():
        ordered = sorted(
            round_bets,
            key=lambda s: (s.rung.raw_edge if s.rung.raw_edge is not None else float("-inf")),
            reverse=True,
        )
        used_players: set[str] = set()
        for s in ordered:
            if used_players.intersection(s.rung.players):
                continue
            kept.append(s)
            used_players.update(s.rung.players)
    return kept


def size_rungs_policy_c(graded: list[GradedRung], unit_max: float, *,
                        unit_max_longshot: float = UNIT_MAX_LONGSHOT,
                        bankroll: float = BANKROLL, unit_size: float = UNIT_SIZE,
                        refund_factor: float = BONUS_BET_FACTOR,
                        round_cap_units: float = OLD_ROUND_CAP_UNITS) -> list[SizedBet]:
    """POLICY C -- EDGE FLOOR + CAPS: policy B's eligible/sized rungs, then
    the per-player cap, then the (reused, unmodified) 15u round cap. Order:
    per-player cap first, round budget second -- so the round's 15u isn't
    spent filling a rung the per-player dedup would drop anyway."""
    sized = size_rungs_edge_floor(graded, unit_max, unit_max_longshot=unit_max_longshot,
                                  bankroll=bankroll, unit_size=unit_size, refund_factor=refund_factor)
    sized = apply_per_player_cap(sized)
    sized = apply_old_round_cap(sized, cap_units=round_cap_units)
    return sized


def settle_round_assumption(round_bets: list[SizedBet], mode: str, *,
                            refund_factor: float = BONUS_BET_FACTOR,
                            refund_dollar_cap: float = 50.0) -> list[float]:
    """Net profit ($) for every bet in ONE round, under a promo-refund
    assumption. Callers must group bets by (year, round_no) themselves (see
    `grade_policy`) -- "one_per_round" is only meaningful within a single
    round's bet list.

    mode:
      "none"           -- no refund, ever. Every non-win bet loses its stake
                          in full, regardless of staking tag.
      "all"            -- the bot's OWN current end-to-end assumption: every
                          one-miss bet, regardless of tag, gets a stake-back
                          refund at `refund_factor`. No dollar cap.
      "one_per_round"  -- realistic: real bookie bonus-back promos are per
                          customer/day, capped, and don't pay out on every
                          bet in a round. Only the SINGLE largest-stake
                          one-miss bet in the round gets a refund, and the
                          refunded STAKE (not the bonus value) is capped at
                          `refund_dollar_cap` before the refund_factor is
                          applied (a bookie "up to $50 back" cap) -- every
                          other one-miss bet in the round is a full loss.
    """
    if mode not in ("none", "all", "one_per_round"):
        raise ValueError(f"unknown promo assumption: {mode!r}")

    refund_bet = None
    if mode == "one_per_round":
        one_miss = [s for s in round_bets if s.rung.outcome == "one_miss"]
        if one_miss:
            refund_bet = max(one_miss, key=lambda s: s.stake)

    results = []
    for s in round_bets:
        if s.rung.outcome == "win":
            results.append(s.stake * (s.rung.book_odds - 1.0))
        elif mode == "none":
            results.append(-s.stake)
        elif mode == "all" and s.rung.outcome == "one_miss":
            results.append(-(1.0 - refund_factor) * s.stake)
        elif mode == "one_per_round" and s is refund_bet:
            refunded_stake = min(s.stake, refund_dollar_cap)
            results.append(-(s.stake - refunded_stake * refund_factor))
        else:
            results.append(-s.stake)
    return results


@dataclass
class PolicyGrade:
    policy: str
    promo_mode: str
    n: int
    n_rounds: int
    units_staked: float
    units_per_round: float
    pl_units: float
    pl_dollars: float
    roi_pct: float


def grade_policy(sized: list[SizedBet], policy: str, promo_mode: str, *,
                 unit_size: float = UNIT_SIZE,
                 refund_factor: float = BONUS_BET_FACTOR,
                 refund_dollar_cap: float = 50.0) -> PolicyGrade:
    """Aggregate P&L for an already-sized bet list under one promo
    assumption, grouping by (year, round_no) so "one_per_round" resolves
    correctly (other modes are round-independent but share the same
    grouping/settlement code path)."""
    by_round: dict[tuple[int, int], list[SizedBet]] = defaultdict(list)
    for s in sized:
        by_round[s.rung.date_key].append(s)

    n = 0
    total_staked = 0.0
    total_pl = 0.0
    for round_bets in by_round.values():
        pls = settle_round_assumption(round_bets, promo_mode, refund_factor=refund_factor,
                                      refund_dollar_cap=refund_dollar_cap)
        for s, pl in zip(round_bets, pls):
            n += 1
            total_staked += s.stake
            total_pl += pl

    n_rounds = len(by_round)
    units_staked = total_staked / unit_size
    return PolicyGrade(
        policy=policy, promo_mode=promo_mode, n=n, n_rounds=n_rounds,
        units_staked=units_staked,
        units_per_round=(units_staked / n_rounds) if n_rounds else 0.0,
        pl_units=total_pl / unit_size, pl_dollars=total_pl,
        roi_pct=(total_pl / total_staked * 100.0) if total_staked > 0 else 0.0,
    )


# Audit's own headline P&L numbers (AUDIT-ROUNDS-16-20-BET-LOSS-AUTOPSY.md,
# "no promo refunds" row): 2026 R18-R20 only, policy A (live), no refunds.
AUDIT_R18_20_TARGET = {"n": 119, "units_staked": 214.75, "pl_units": -87.33}
_RECONCILIATION_TOLERANCE_UNITS = 1.0
_R18_20 = [(2026, 18), (2026, 19), (2026, 20)]
# 2026 R16/R18-R20 have real Sportsbet/model book prices; R17 is a model-only
# run (0 gradeable rungs, see stake_cap_backtest's own scope notes); R21/R22
# are excluded -- not fully gradeable (AUDIT-ROUNDS-16-20-BET-LOSS-AUTOPSY.md
# Finding 0: stats cache stalled again, R21 has 1 game graded, R22 has none).
VIABILITY_ROUNDS = [(2026, 16), (2026, 18), (2026, 19), (2026, 20)]


def check_r18_20_reconciliation(*, grade_total_points: bool) -> PolicyGrade:
    """Policy A ("live" -- plain `size_rungs`/`recommend_units`, no round
    cap), assumption "none", restricted to 2026 R18-R20 only -- the exact
    slice `AUDIT-ROUNDS-16-20-BET-LOSS-AUTOPSY.md`'s headline P&L table
    reports (119 bets / 214.8u / -87.3u under "no promo refunds").
    Audit-scope (`grade_total_points=False`) reproduces those numbers;
    full-scope (`True`) additionally grades the one total_points-bearing rung
    the audit's own re-grader couldn't resolve (see `grade_leg_outcomes`
    docstring)."""
    graded, _ = grade_rounds(_R18_20, grade_total_points=grade_total_points)
    sized = size_rungs(graded, unit_max=UNIT_MAX)
    return grade_policy(sized, policy="A", promo_mode="none")


def _reconciliation_report(audit_scope: PolicyGrade, full_scope: PolicyGrade) -> tuple[bool, str]:
    t = AUDIT_R18_20_TARGET
    ok = (audit_scope.n == t["n"]
          and abs(audit_scope.units_staked - t["units_staked"]) <= _RECONCILIATION_TOLERANCE_UNITS
          and abs(audit_scope.pl_units - t["pl_units"]) <= _RECONCILIATION_TOLERANCE_UNITS)
    lines = []
    lines.append('## Reconciliation gate — Policy A, assumption "none", 2026 R18–R20 only')
    lines.append("")
    lines.append("| Scope | n bets | units staked | P&L (units) |")
    lines.append("|---|--:|--:|--:|")
    lines.append(f"| Audit target (`AUDIT-ROUNDS-16-20-BET-LOSS-AUTOPSY.md`) | {t['n']} | "
                 f"{t['units_staked']:.2f}u | {t['pl_units']:.2f}u |")
    lines.append(f"| **Audit-scope** (`grade_total_points=False` — reproduces the audit's own blind "
                 f"spot) | {audit_scope.n} | {audit_scope.units_staked:.2f}u | {audit_scope.pl_units:.2f}u |")
    lines.append(f"| **Full-scope** (`grade_total_points=True` — basis for policies B/C below) | "
                 f"{full_scope.n} | {full_scope.units_staked:.2f}u | {full_scope.pl_units:.2f}u |")
    lines.append("")
    if ok:
        lines.append(f"**GATE PASSED** — audit-scope lands within "
                     f"{_RECONCILIATION_TOLERANCE_UNITS:.0f}u of the audit's own numbers on all three "
                     "metrics.")
    else:
        lines.append("**GATE FAILED** — audit-scope does not reproduce the audit's numbers within "
                     f"{_RECONCILIATION_TOLERANCE_UNITS:.0f}u. Nothing past this section should be "
                     "trusted.")
    lines.append("")
    lines.append("The full-scope/audit-scope delta is exactly one rung: "
                 "`2026-r20-Western_Bulldogs-Richmond-model-5.00` (Total points 175.5+ / Rhylee West "
                 "1+ goals / Dion Prestia 20+ disposals, 2u PROMO KELLY). Final score Western Bulldogs "
                 "105–48 Richmond = 153 total, under the 175.5 line — a clean miss "
                 "(`data_cache/games_2026.parquet`, round 20, complete). The audit's own "
                 "`legs_graded.csv` shows this leg's `hit`/`actual` fields empty for that row: its "
                 "independent re-grading script never sourced match-total actuals at all (it read "
                 "player box scores only, per its own methodology note), so the whole rung came back "
                 "\"ungradeable\" even though the game is complete and the other two legs graded — and "
                 "matched this harness's leg-level results exactly (both hit). `_settle_leg` reads the "
                 "real 175.5 threshold out of the leg's `name` string via regex "
                 "(`\"Total points 175.5+\"`); the `line` key on every total_points leg in the saved "
                 "multis.json is a placeholder value of 5, not the real line — both live callers "
                 "(`dashboard/settle.py`, this module) pass `game` into `_settle_leg` so it can look up "
                 "the actual score; the audit's separate script apparently didn't. 2.0u of 214.75u — "
                 "0.9% — this cannot move the answer the rest of this report exists to give.")
    lines.append("")
    return ok, "\n".join(lines)


_PROMO_LABELS = {
    "none": "1. No refunds at all",
    "one_per_round": "2. One refund/book/round ($50 cap, 0.75 bonus, largest qualifying one-miss bet only)",
    "all": "3. Bot's current assumption (every one-miss bet refunded @0.75)",
}
_POLICY_LABELS = {
    "A": "A — LIVE (PROMO_EV_MIN gate on total_ev, no round cap)",
    "B": "B — EDGE FLOOR (raw edge > 0 required; promo may only up-size)",
    "C": "C — EDGE FLOOR + CAPS (B + 15u round cap + 1 multi/player/round)",
}


def run_edge_floor_viability() -> dict:
    """Full TASK 1 orchestration: reconciliation gate, then the 3-policy x
    3-promo-assumption grid over 2026 R16/R18-R20 (full scope), plus the
    policy-C/assumption-2 by-round breakdown and the policy-C units/round
    answer. Returns a dict (used by the report renderer and by tests). Per
    DO-EDGE-FLOOR-VIABILITY-TEST's "show me the discrepancy" instruction this
    does NOT raise on gate failure -- it renders the failure into the report
    and stops there (see `_render_viability_report`)."""
    audit_scope = check_r18_20_reconciliation(grade_total_points=False)
    full_scope = check_r18_20_reconciliation(grade_total_points=True)
    gate_passed, gate_md = _reconciliation_report(audit_scope, full_scope)

    if not gate_passed:
        report = _render_viability_report(gate_passed=False, gate_md=gate_md, rounds=VIABILITY_ROUNDS,
                                          graded=[], n_excluded=0, grid={}, promo_modes=[],
                                          c_by_round=[], c_units_per_round=[], c_mean_units_per_round=0.0)
        return {"gate_passed": False, "audit_scope": audit_scope, "full_scope": full_scope,
                "report_md": report}

    graded, n_excluded = grade_rounds(VIABILITY_ROUNDS, grade_total_points=True)

    sized_a = size_rungs(graded, unit_max=UNIT_MAX)
    sized_b = size_rungs_edge_floor(graded, unit_max=UNIT_MAX)
    eligible_c_pre_cap = apply_per_player_cap(size_rungs_edge_floor(graded, unit_max=UNIT_MAX))
    sized_c = apply_old_round_cap(eligible_c_pre_cap, cap_units=OLD_ROUND_CAP_UNITS)

    policies = {"A": sized_a, "B": sized_b, "C": sized_c}
    promo_modes = ["none", "one_per_round", "all"]

    grid: dict[tuple[str, str], PolicyGrade] = {}
    for pname, sized in policies.items():
        for mode in promo_modes:
            grid[(pname, mode)] = grade_policy(sized, pname, mode)

    # Explains the B/C rows' -100% "no refunds" ROI at a glance instead of
    # leaving a round number unexplained: state the actual band set and
    # whether any eligible bet won outright (it's 0 in this sample).
    b_bands = sorted({s.rung.band for s in sized_b if s.rung.band is not None})
    b_n_wins = sum(1 for s in sized_b if s.rung.outcome == "win")

    # Policy C / assumption "one_per_round", by round -- bets staked (post
    # round-cap) vs eligible (pre round-cap) vs the round's total gradeable
    # records, for every scoped round (zero rows included, not dropped).
    graded_by_round: dict[tuple[int, int], int] = defaultdict(int)
    for g in graded:
        graded_by_round[g.date_key] += 1
    eligible_by_round: dict[tuple[int, int], int] = defaultdict(int)
    for s in eligible_c_pre_cap:
        eligible_by_round[s.rung.date_key] += 1
    staked_by_round: dict[tuple[int, int], list[SizedBet]] = defaultdict(list)
    for s in sized_c:
        staked_by_round[s.rung.date_key].append(s)

    c_by_round = []
    for key in VIABILITY_ROUNDS:
        round_bets = staked_by_round.get(key, [])
        c_by_round.append({
            "year": key[0], "round_no": key[1],
            "n_bets": len(round_bets), "units": sum(s.units for s in round_bets),
            "n_eligible": eligible_by_round.get(key, 0),
            "n_total_records": graded_by_round.get(key, 0),
        })

    # Policy-C staking-only answer -- staking happens before/independent of
    # the promo-assumption sweep (assumptions only change SETTLEMENT of
    # already-placed stakes), so this is one number per round, not one per
    # assumption. Mean is over ALL scoped rounds, including any at 0u.
    c_units_per_round = [(key, sum(s.units for s in sized_c if s.rung.date_key == key))
                         for key in VIABILITY_ROUNDS]
    c_mean_units_per_round = (sum(u for _, u in c_units_per_round) / len(c_units_per_round)
                              if c_units_per_round else 0.0)

    report = _render_viability_report(
        gate_passed=True, gate_md=gate_md, rounds=VIABILITY_ROUNDS,
        graded=graded, n_excluded=n_excluded, grid=grid, promo_modes=promo_modes,
        c_by_round=c_by_round, c_units_per_round=c_units_per_round,
        c_mean_units_per_round=c_mean_units_per_round,
        b_bands=b_bands, b_n_wins=b_n_wins,
    )
    return {
        "gate_passed": True, "audit_scope": audit_scope, "full_scope": full_scope,
        "graded": graded, "n_excluded": n_excluded, "grid": grid,
        "c_by_round": c_by_round, "c_units_per_round": c_units_per_round,
        "c_mean_units_per_round": c_mean_units_per_round, "report_md": report,
    }


def _render_viability_report(*, gate_passed, gate_md, rounds, graded, n_excluded, grid,
                             promo_modes, c_by_round, c_units_per_round,
                             c_mean_units_per_round, b_bands=(), b_n_wins=None) -> str:
    lines = []
    lines.append("# Edge-floor viability test — staking policy, not modelling")
    lines.append("")
    lines.append("_TASK 1 of `fable tweaks/DO-EDGE-FLOOR-VIABILITY-TEST.txt` — measurement only. Does "
                 "not touch `afl_bot/build/staking.py`, `afl_bot/build/report.py`, `afl_bot/cli.py`, or "
                 "`afl_bot/config.py`; every number below comes from re-sizing/re-settling the SAME "
                 "saved `reports/*_multis.json` rungs through `afl_bot.build.staking`'s own Kelly "
                 "functions, never a reimplementation._")
    lines.append("")
    lines.append(gate_md)
    if not gate_passed:
        lines.append("**STOPPING HERE per the reconciliation gate rule — the policy grid is not "
                     "computed and nothing below this line should be trusted or acted on.**")
        lines.append("")
        return "\n".join(lines)

    round_list = ", ".join(f"{y} R{r}" for y, r in rounds)
    lines.append("## Scope")
    lines.append("")
    lines.append(f"Rounds: **{round_list}**. 2026 R17 excluded (model-only, no real book prices, 0 "
                 "gradeable rungs). R21/R22 excluded (not fully gradeable — "
                 "`AUDIT-ROUNDS-16-20-BET-LOSS-AUTOPSY.md` Finding 0: the stats-cache pipeline stalled "
                 "again, R21 has 1 of 9 games graded, R22 has none). "
                 f"Full-scope grading (`grade_total_points=True`): **{len(graded)}** gradeable rungs "
                 f"across these 4 rounds, {n_excluded} excluded for a still-ungradeable leg (unmatched "
                 "player name, etc. — never guessed).")
    lines.append("")
    lines.append("## Main grid — one row per (policy × promo assumption)")
    lines.append("")
    lines.append("| Policy | Promo assumption | n bets | units staked | units/round | P&L (units) | "
                 "P&L ($) | ROI% |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for pname in ("A", "B", "C"):
        for mode in promo_modes:
            g = grid[(pname, mode)]
            n_flag = " ⚠️" if g.n < 30 else ""
            lines.append(f"| {_POLICY_LABELS[pname]} | {_PROMO_LABELS[mode]} | {g.n}{n_flag} | "
                         f"{g.units_staked:.2f}u | {g.units_per_round:.2f}u | {g.pl_units:+.2f}u | "
                         f"${g.pl_dollars:+,.2f} | {g.roi_pct:+.1f}% |")
    lines.append("")
    lines.append("**⚠️ = n < 30.** Every row so flagged is computed on fewer than 30 graded bets — at "
                 "that sample size the ROI is NOT evidence of edge in either direction (a single multi "
                 "can swing it by several points). Report the n; do not call a ⚠️ row \"profitable\" or "
                 "\"working\", and do not call any row profitable off this alone without also looking "
                 "at its worst round.")
    lines.append("")
    if b_n_wins == 0 and b_bands:
        bands_str = ", ".join(f"{b:g}" for b in b_bands)
        n_b = grid[("B", "none")].n
        lines.append(f"> **Why B/C's \"no refunds\" ROI is exactly -100%:** every raw-edge-positive "
                     f"rung in this {len(rounds)}-round sample sits at band {bands_str} of the ladder "
                     "(`MULTI_TARGET_ODDS` = 2.10/2.75/3.50/5.00/8.00/15.00), and none of them won "
                     f"outright here (0 of {n_b} policy-B bets — win probabilities are small by "
                     "construction at these odds). With no refund, a bet that doesn't win outright "
                     "loses its full stake, so 0 wins mechanically means -100% ROI on this row. Not a "
                     "bug; a real property of where this bot's positive raw edge currently lives — see "
                     "Finding 3 of the autopsy on short-band overconfidence vs long-band "
                     "small-sample variance.")
        lines.append("")

    lines.append('## Policy C, assumption 2 ("one refund/book/round") — by round')
    lines.append("")
    lines.append("\"Eligible\" = rungs passing policy C's raw-edge floor + per-player dedup, BEFORE the "
                 "15u round cap trims or drops the overflow. \"Bets staked\" = what's actually staked "
                 "after the round cap. \"Total records\" = all gradeable rungs in that round (full "
                 "scope, model + sportsbet ladders) — the denominator the eligible count is \"out of\".")
    lines.append("")
    lines.append("| Round | Bets staked | Units staked | Eligible (pre-cap) | Total records |")
    lines.append("|--|--:|--:|--:|--:|")
    for row in c_by_round:
        lines.append(f"| {row['year']} R{row['round_no']} | {row['n_bets']} | {row['units']:.2f}u | "
                     f"{row['n_eligible']} | {row['n_total_records']} |")
    lines.append("")

    lines.append("## The question this test exists to answer")
    lines.append("")
    lines.append("**Under Policy C, how many units per round does the bot actually stake?**")
    lines.append("")
    lines.append("Staking (eligibility + sizing) happens before, and independently of, the promo-"
                 "assumption sweep above — the three promo assumptions only change how an "
                 "already-placed stake SETTLES, not whether or how much gets staked. So this is one "
                 "number per round, not one per promo assumption.")
    lines.append("")
    lines.append(f"Mean: **{c_mean_units_per_round:.2f}u/round**, averaged over all "
                 f"{len(c_units_per_round)} scoped rounds (including any at 0u — not dropped).")
    lines.append("")
    lines.append("Per-round list:")
    for key, units in c_units_per_round:
        lines.append(f"- {key[0]} R{key[1]}: {units:.2f}u")
    lines.append("")
    return "\n".join(lines)


def edge_floor_viability_command(out_path: str | None = None) -> dict:
    """Run TASK 1 end to end and write `reports/edge_floor_viability.md`
    (atomic write, matching `stake_cap_backtest_command`'s pattern). Prints
    the gate's pass/fail and returns the full result dict."""
    result = run_edge_floor_viability()
    out = Path(out_path) if out_path else (REPORTS_DIR / "edge_floor_viability.md")
    atomic_write_text(out, result["report_md"])
    print("GATE PASSED" if result["gate_passed"] else "GATE FAILED")
    print(f"[saved to {out}]")
    return result


def run_backtest(*, unit_max_candidates: tuple[float, ...] = DEFAULT_UNIT_MAX_CANDIDATES,
                 n_sims: int = 10_000, live_unit_max: float = UNIT_MAX) -> dict:
    """Full orchestration: discover rounds, grade, sweep caps both ways,
    round-cap on/off comparison, cross-check. Returns everything as a dict
    (also used directly by tests) plus the rendered markdown report string."""
    rounds = discover_rounds()
    graded, n_excluded = grade_rounds(rounds)

    version_a: list[ReplayResult] = []
    version_b: dict[float, dict[str, SimResult]] = {}
    for cap in unit_max_candidates:
        sized = size_rungs(graded, cap)
        rep = realized_replay(sized)
        rep.unit_max = cap
        version_a.append(rep)
        version_b[cap] = probabilistic_sim(sized, n_sims=n_sims)

    # Round-cap ON vs OFF, at the live UNIT_MAX only.
    live_sized = size_rungs(graded, live_unit_max)
    cap_off = realized_replay(live_sized)
    cap_off.unit_max = live_unit_max
    cap_on_sized = apply_old_round_cap(live_sized)
    cap_on = realized_replay(cap_on_sized)
    cap_on.unit_max = live_unit_max

    cross_check = hit_rate_cross_check(graded)

    excluded_rounds = _find_excluded_rounds()

    # If the formula never wants more than the SMALLEST swept cap even when
    # uncapped, every candidate produces identical results -- not a bug, but
    # worth calling out explicitly so identical rows don't read as one.
    uncapped = size_rungs(graded, unit_max=1e6, unit_max_longshot=1e6)
    max_uncapped_units = max((s.units for s in uncapped), default=0.0)
    min_swept_cap = min(unit_max_candidates) if unit_max_candidates else None
    caps_never_bind = (min_swept_cap is not None and max_uncapped_units <= min_swept_cap + 1e-9)

    report = _render_report(
        rounds=rounds, graded=graded, n_excluded=n_excluded,
        version_a=version_a, version_b=version_b,
        cap_off=cap_off, cap_on=cap_on, live_unit_max=live_unit_max,
        cross_check=cross_check, excluded_rounds=excluded_rounds,
        n_sims=n_sims, max_uncapped_units=max_uncapped_units,
        caps_never_bind=caps_never_bind,
    )
    return {
        "rounds": rounds, "graded": graded, "n_excluded": n_excluded,
        "version_a": version_a, "version_b": version_b,
        "cap_off": cap_off, "cap_on": cap_on, "cross_check": cross_check,
        "excluded_rounds": excluded_rounds, "report_md": report,
    }


def _find_excluded_rounds() -> list[tuple[str, str]]:
    """(round_stem, reason) for every reports/*_multis.json NOT used, so the
    report can state its scope honestly instead of silently narrowing it."""
    used = set(discover_rounds())
    excluded = []
    for p in sorted(glob.glob(str(REPORTS_DIR / "*_multis.json"))):
        stem = Path(p).stem.replace("_multis", "")
        if "_r" not in stem:
            continue
        year_s, _, round_s = stem.partition("_r")
        try:
            key = (int(year_s), int(round_s))
        except ValueError:
            continue
        if key in used:
            continue
        records = _load_multis_records(Path(p))
        if not any(r.get("book_combo") is not None for r in records):
            excluded.append((stem, "no real book_combo prices (model-only run)"))
        else:
            excluded.append((stem, "round not completed yet"))
    return excluded


def _fmt(x: float | None, spec: str = ".2f") -> str:
    return "—" if x is None else format(x, spec)


def _render_report(*, rounds, graded, n_excluded, version_a, version_b,
                   cap_off, cap_on, live_unit_max, cross_check, excluded_rounds,
                   n_sims, max_uncapped_units=None, caps_never_bind=False) -> str:
    lines = []
    lines.append("# Stake-cap backtest — which UNIT_MAX grows the bankroll?")
    lines.append("")
    lines.append("_Diagnostic only. Does not change UNIT_MAX, KELLY_FRACTION, or any other "
                  "live config — see DO-STAKE-CAP-BACKTEST.txt._")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    if rounds:
        round_list = ", ".join(f"{y} R{r}" for y, r in rounds)
        lines.append(f"Rounds used (real Sportsbet book prices + completed results): **{round_list}**.")
    else:
        lines.append("**No usable rounds found** — nothing to backtest.")
    lines.append(f"Graded rungs: **{len(graded)}** (model + sportsbet ladders; "
                 f"{n_excluded} rung(s) excluded — at least one leg still ungradeable).")
    if excluded_rounds:
        lines.append("")
        lines.append("Rounds NOT used:")
        for stem, reason in excluded_rounds:
            lines.append(f"- {stem}: {reason}")
    lines.append("")
    n_rounds = len(rounds)
    small_sample_note = (
        f"this is a {n_rounds}-round sample" if n_rounds != 5
        else "this is a 5-round sample"
    )
    lines.append(f"> **Honesty note:** {small_sample_note} (R14/R15 predate the multis.json "
                 "emitter and R17 was a model-only run with no book prices, so this can run "
                 "short of the ~5 rounds a full season-in-progress would offer). A handful of "
                 "multis can dominate the realized P&L at this size. Treat Version A as \"what "
                 "happened\", not \"what's optimal\" — Version B's probabilistic sim is the more "
                 "meaningful signal here, and even that leans entirely on model calibration (see "
                 "the cross-check below).")
    lines.append("")
    lines.append("**Not swept:** KELLY_FRACTION. `recommend_units` doesn't parameterise the Kelly "
                 "fraction (it's a default inside `fractional_kelly_fraction`/`multi_outcome_kelly`), "
                 "and reimplementing that conversion for this backtest would be exactly the kind of "
                 "parallel sizing logic the brief says to avoid. **Not backtested:** Pull 'Em — no "
                 "round in `reports/` currently has a real (priced) Pull 'Em record.")
    lines.append("")

    lines.append("## Version A — realized replay (what literally would have happened)")
    lines.append("")
    lines.append("| UNIT_MAX | n bets | staked | returned | net | ROI% | end bankroll | max DD |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in version_a:
        lines.append(f"| {r.unit_max:g}u | {r.n_bets} | ${r.total_staked:,.2f} | "
                     f"${r.total_returned:,.2f} | ${r.net_profit:+,.2f} | {r.roi_pct:+.1f}% | "
                     f"${r.end_bankroll:,.2f} | {r.max_drawdown_pct:.1f}% |")
    lines.append("")
    if caps_never_bind:
        lines.append(f"> **All rows above are identical — this is real, not a bug.** Even sized "
                     f"with no cap at all, the strongest bet in this sample only ever wanted "
                     f"**{max_uncapped_units:g}u**, below every candidate cap tested "
                     f"({', '.join(f'{c:g}u' for c in sorted({r.unit_max for r in version_a}))}). "
                     "None of them bound in this sample, so this run genuinely cannot answer "
                     "\"is 3u too tight\" yet — it can only confirm 3u hasn't cost anything so "
                     "far. A real answer needs a round where the formula's own uncapped output "
                     "exceeds 3u.")
        lines.append("")

    lines.append(f"### Round-cap ON vs OFF (at the live UNIT_MAX={live_unit_max:g}u)")
    lines.append("")
    lines.append("_The round-level 15u cap (`KELLY_PER_ROUND_CAP`) was removed from the live bot "
                 "2026-07-10. This row shows what keeping it would have cost/earned on the same "
                 "bet set, using a read-only reimplementation of the deleted allocator — the live "
                 "bot does not have this cap anymore regardless of what this shows._")
    lines.append("")
    lines.append("| Round cap | n bets | staked | returned | net | ROI% | end bankroll | max DD |")
    lines.append("|--|--:|--:|--:|--:|--:|--:|--:|")
    lines.append(f"| OFF (live) | {cap_off.n_bets} | ${cap_off.total_staked:,.2f} | "
                 f"${cap_off.total_returned:,.2f} | ${cap_off.net_profit:+,.2f} | "
                 f"{cap_off.roi_pct:+.1f}% | ${cap_off.end_bankroll:,.2f} | {cap_off.max_drawdown_pct:.1f}% |")
    lines.append(f"| ON (15u, removed) | {cap_on.n_bets} | ${cap_on.total_staked:,.2f} | "
                 f"${cap_on.total_returned:,.2f} | ${cap_on.net_profit:+,.2f} | "
                 f"{cap_on.roi_pct:+.1f}% | ${cap_on.end_bankroll:,.2f} | {cap_on.max_drawdown_pct:.1f}% |")
    lines.append("")

    lines.append(f"## Version B — probabilistic bankroll sim (N={n_sims:,} paths per cap)")
    lines.append("")
    lines.append("### Fixed stake (constant $, matches live behaviour)")
    lines.append("")
    lines.append("| UNIT_MAX | median end | mean | p5 | p95 | median maxDD | P(down) | P(DD>20%) |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for cap, modes in version_b.items():
        s = modes["fixed"]
        lines.append(f"| {cap:g}u | ${s.median_end:,.2f} | ${s.mean_end:,.2f} | ${s.p5_end:,.2f} | "
                     f"${s.p95_end:,.2f} | {s.median_max_dd_pct:.1f}% | {s.p_down:.1%} | {s.p_dd_over_20:.1%} |")
    lines.append("")
    lines.append("### Compounding stake (% of running bankroll — truer Kelly-growth read)")
    lines.append("")
    lines.append("| UNIT_MAX | median end | mean | p5 | p95 | median maxDD | P(down) | P(DD>20%) |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for cap, modes in version_b.items():
        s = modes["compounding"]
        lines.append(f"| {cap:g}u | ${s.median_end:,.2f} | ${s.mean_end:,.2f} | ${s.p5_end:,.2f} | "
                     f"${s.p95_end:,.2f} | {s.median_max_dd_pct:.1f}% | {s.p_down:.1%} | {s.p_dd_over_20:.1%} |")
    lines.append("")

    lines.append("## Cross-check: is the modelled edge real, or just what happened to land?")
    lines.append("")
    mh, ah = cross_check["modelled_hit_rate"], cross_check["actual_hit_rate"]
    if ah is None:
        lines.append("No graded bets to cross-check.")
    else:
        lines.append(f"Modelled hit-rate (mean p_all_win over {cross_check['n_with_probs']} bets with "
                     f"promo branch probs): **{_fmt(mh, '.1%')}**. "
                     f"Actual realized hit-rate (fraction of all {cross_check['n']} graded bets that "
                     f"won outright): **{_fmt(ah, '.1%')}**.")
    lines.append("")
    lines.append("`reports/calibration_log.csv` has no 2026 round-level entries yet "
                 "(`grade-round` hasn't been run this season, only historical 2025 R1 rows exist) "
                 "— it can't be used as a third reference point here. This cross-check is Version A "
                 "vs Version B only.")
    lines.append("")
    if mh is not None and ah is not None:
        gap = ah - mh
        if abs(gap) > 0.15:
            lines.append(f"**Gap of {gap:+.1%} is large** for a {cross_check['n']}-bet sample — "
                         "with this few bets that's easily within noise (a single multi swings the "
                         "rate by several points), not necessarily evidence of mis-calibration. "
                         "Don't over-read it, but don't fully trust Version B's ranking either.")
        else:
            lines.append(f"Gap of {gap:+.1%} is small given the sample size — Version B's cap "
                         f"ranking is reasonably credible, for what a {n_rounds}-round sample is worth.")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if caps_never_bind:
        lines.append(f"**No cap comparison is possible this run** — every candidate sized "
                     f"identically because the strongest real bet only ever wanted "
                     f"{max_uncapped_units:g}u uncapped (see the note under Version A). This "
                     f"round's data says nothing about whether {live_unit_max:g}u is too tight, "
                     "too loose, or right — it only confirms it hasn't been the binding "
                     "constraint yet. Re-run this after a round where the formula's uncapped "
                     "output for at least one rung exceeds the smallest candidate cap.")
    elif version_a:
        best_a = max(version_a, key=lambda r: r.net_profit)
        best_b_fixed = max(version_b.items(), key=lambda kv: kv[1]["fixed"].median_end)
        lines.append(f"Realized replay: **{best_a.unit_max:g}u** produced the highest net profit "
                     f"(${best_a.net_profit:+,.2f}, max drawdown {best_a.max_drawdown_pct:.1f}%) — "
                     f"on a {n_rounds}-round sample this is a handful of multis' worth of signal, "
                     "not a verdict.")
        lines.append(f"Probabilistic sim (fixed stake): **{best_b_fixed[0]:g}u** gave the highest "
                     f"median ending bankroll (${best_b_fixed[1]['fixed'].median_end:,.2f}).")
        lines.append("")
        lines.append("**Proposal (not applied):** if the two agree, that's the stronger candidate "
                     "for UNIT_MAX; if they disagree, prefer Version B's ranking only once the "
                     "hit-rate cross-check above shows the model's probabilities are trustworthy, "
                     "and treat this whole report as informative rather than conclusive until "
                     "more real-book rounds accumulate. UNIT_MAX stays at "
                     f"{live_unit_max:g}u in `config.py` — this run changes nothing live.")
    else:
        lines.append("No graded bets — no verdict possible.")
    lines.append("")

    return "\n".join(lines)


def stake_cap_backtest_command(unit_max_candidates: tuple[float, ...] = DEFAULT_UNIT_MAX_CANDIDATES,
                               n_sims: int = 10_000, out_path: str | None = None) -> None:
    """CLI entry point: run the backtest, print the summary, save the report."""
    result = run_backtest(unit_max_candidates=unit_max_candidates, n_sims=n_sims)
    out = Path(out_path) if out_path else (REPORTS_DIR / "stake_cap_backtest.md")
    atomic_write_text(out, result["report_md"])
    print(result["report_md"])
    print(f"\n[saved to {out}]")

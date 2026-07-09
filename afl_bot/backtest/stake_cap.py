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

from afl_bot.build.staking import recommend_units
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

    @property
    def date_key(self) -> tuple[int, int]:
        return (self.year, self.round_no)


def grade_leg_outcomes(legs: list[dict], game: str, h2h_actual: dict,
                       total_actual: dict, player_stat: dict,
                       year: int, round_no: int) -> str | None:
    """Grade every leg of one rung via the live `_settle_leg`. Returns
    'win' / 'one_miss' / 'dead', or None if any leg is still ungradeable
    (excluded from the backtest -- never guessed)."""
    n_miss = 0
    for leg in legs:
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


def grade_rounds(rounds: list[tuple[int, int]]) -> tuple[list[GradedRung], int]:
    """Grade every real-priced, staked-by-the-live-formula-eligible rung
    (model + sportsbet ladders; pull_em has no real data to grade, see module
    docstring) across the given rounds. Returns (graded rungs in chronological
    order, n_excluded_for_an_unresolved_leg)."""
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
            outcome = grade_leg_outcomes(
                r.get("legs", []), r.get("game", ""), h2h_actual, total_actual,
                player_stat, year, round_no,
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

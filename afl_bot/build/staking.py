"""
Fractional-Kelly staking + bankroll Monte Carlo (plan §4.4).

Edge alone doesn't tell you how much to bet. Kelly maximises long-run log growth
but full Kelly is brutally volatile, so we stake a *fraction* (0.25x) of it,
cap any single bet, and cap total exposure per round. ``simulate_bankroll`` then
reuses the same vectorised Monte Carlo idea to project a season of these bets
and report the terminal-bankroll and max-drawdown distributions — the honest
picture of variance behind a positive EV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afl_bot.config import KELLY_FRACTION, KELLY_PER_BET_CAP, KELLY_PER_ROUND_CAP


def kelly_fraction(prob: float, odds: float) -> float:
    """Full-Kelly fraction of bankroll for a single bet at decimal ``odds``:
    ``f* = (p*odds - 1)/(odds - 1)`` = (bp - q)/b. Zero when there's no edge."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max((prob * odds - 1.0) / b, 0.0)


def fractional_kelly_fraction(prob: float, odds: float,
                              fraction: float = KELLY_FRACTION,
                              cap: float = KELLY_PER_BET_CAP) -> float:
    """Capped fractional-Kelly stake fraction for one bet."""
    return min(fraction * kelly_fraction(prob, odds), cap)


@dataclass
class StakedBet:
    name: str
    prob: float
    odds: float
    fraction: float   # fraction of bankroll
    stake: float      # dollars


def stake_bets(bets: list[tuple[str, float, float]], bankroll: float, *,
               fraction: float = KELLY_FRACTION, per_bet_cap: float = KELLY_PER_BET_CAP,
               per_round_cap: float = KELLY_PER_ROUND_CAP,
               mults: list[float] | None = None) -> list[StakedBet]:
    """Size a round's bets by capped fractional Kelly.

    ``bets`` is ``[(name, prob, odds), ...]``. Each gets a per-bet-capped
    fractional-Kelly fraction; if the round total exceeds ``per_round_cap`` the
    whole book is scaled down proportionally so total exposure stays bounded.
    ``mults`` (aligned to ``bets``) scales individual bets' Kelly fraction —
    e.g. 0.5 for noisier prop legs (round-2 §2.5). Zero-edge bets get zero stake.
    """
    mults = mults if mults is not None else [1.0] * len(bets)
    fracs = [fractional_kelly_fraction(p, o, fraction, per_bet_cap) * m
             for (_, p, o), m in zip(bets, mults)]
    total = sum(fracs)
    if total > per_round_cap and total > 0:
        scale = per_round_cap / total
        fracs = [f * scale for f in fracs]
    return [
        StakedBet(name=name, prob=p, odds=o, fraction=f, stake=f * bankroll)
        for (name, p, o), f in zip(bets, fracs)
    ]


def simulate_bankroll(bets: list[tuple[float, float, float]], bankroll0: float, *,
                      rounds: int, n_sims: int, rng: np.random.Generator) -> dict:
    """Vectorised bankroll Monte Carlo over ``rounds`` repetitions of a book of
    bets. ``bets`` is ``[(prob, odds, fraction), ...]`` where ``fraction`` is the
    fraction of the *current* bankroll staked (so stakes compound). Returns
    ``terminal`` bankrolls and per-path ``max_drawdown`` (peak-to-trough)."""
    bankroll = np.full(n_sims, float(bankroll0))
    peak = bankroll.copy()
    max_dd = np.zeros(n_sims)

    for _ in range(rounds):
        for prob, odds, frac in bets:
            stake = frac * bankroll
            win = rng.random(n_sims) < prob
            bankroll = bankroll + np.where(win, stake * (odds - 1.0), -stake)
            bankroll = np.clip(bankroll, 0.0, None)
        peak = np.maximum(peak, bankroll)
        drawdown = (peak - bankroll) / np.where(peak > 0, peak, 1.0)
        max_dd = np.maximum(max_dd, drawdown)

    return {"terminal": bankroll, "max_drawdown": max_dd}


def simulate_bankroll_joint(bets: list[tuple[float, float]], masks: np.ndarray, bankroll0: float, *,
                            rounds: int, n_sims: int, rng: np.random.Generator) -> dict:
    """Bankroll Monte Carlo that resolves the round's bets JOINTLY (round-2
    §3.4 / §P10g). ``bets`` is ``[(odds, fraction), ...]`` and ``masks`` is a
    ``(n_bets, n_iter)`` boolean array of each bet's per-iteration sim outcome.

    Each projected round bootstraps one shared sim-iteration index per path, so
    bets that overlap (a promo multi and the singles it contains) win or lose
    *together* — the correlated exposure that the independent ``simulate_bankroll``
    understates. Cross-match legs are independent in the sim, so sharing the
    index is still a valid joint sample."""
    masks = np.asarray(masks, dtype=bool)
    n_bets, n_iter = masks.shape
    odds = np.array([b[0] for b in bets], dtype=float)
    frac = np.array([b[1] for b in bets], dtype=float)

    bankroll = np.full(n_sims, float(bankroll0))
    peak = bankroll.copy()
    max_dd = np.zeros(n_sims)

    for _ in range(rounds):
        idx = rng.integers(0, n_iter, size=n_sims)
        outcomes = masks[:, idx]                       # (n_bets, n_sims), correlated within round
        for j in range(n_bets):
            stake = frac[j] * bankroll
            bankroll = bankroll + np.where(outcomes[j], stake * (odds[j] - 1.0), -stake)
            bankroll = np.clip(bankroll, 0.0, None)
        peak = np.maximum(peak, bankroll)
        drawdown = (peak - bankroll) / np.where(peak > 0, peak, 1.0)
        max_dd = np.maximum(max_dd, drawdown)

    return {"terminal": bankroll, "max_drawdown": max_dd}


def multi_outcome_kelly(
    p_win: float,
    p_one_loss: float,
    p_dead: float,
    multi_odds: float,
    refund_factor: float,
    fraction: float = KELLY_FRACTION,
    cap: float = KELLY_PER_BET_CAP,
) -> float:
    """Capped fractional-Kelly stake fraction for a 3-outcome stake-back multi.

    Three outcomes per $1 staked:
      - all legs win  -> net +( M - 1 )
      - exactly 1 leg fails -> net +( R - 1 )  (partial refund, R < 1 so still a loss)
      - 2+ legs fail  -> net  -1

    Maximises expected log-growth:
      g(f) = p_win*ln(1+f*(M-1)) + p_one_loss*ln(1+f*(R-1)) + p_dead*ln(1-f)

    Solves g'(f)=0 via scipy brentq on (0, 1), applies ``fraction`` and ``cap``.
    Returns 0.0 when the promo-aware EV is non-positive (no stake warranted).
    """
    from scipy.optimize import brentq

    M = float(multi_odds)
    R = float(refund_factor)

    def _gprime(f: float) -> float:
        a = 1.0 + f * (M - 1.0)
        b = 1.0 + f * (R - 1.0)   # = 1 - f*(1-R) > 0 for f < 1 when 0 < R < 1
        c = 1.0 - f
        if a <= 1e-15 or b <= 1e-15 or c <= 1e-15:
            return float("-inf")
        return (p_win * (M - 1.0) / a
                + p_one_loss * (R - 1.0) / b
                - p_dead / c)

    # g'(0) = p_win*(M-1) + p_one_loss*(R-1) - p_dead
    # No positive stake when total EV is non-positive at f=0.
    if _gprime(0.0) <= 0.0:
        return 0.0

    # g'(f) -> -inf as f -> 1-, so a root exists in (0, 1) when g'(0) > 0.
    try:
        f_star = brentq(_gprime, 1e-9, 1.0 - 1e-9, maxiter=200)
    except ValueError:
        # No sign change in interval: g' stays positive -> optimal f_star -> inf, cap it.
        f_star = 1.0

    return min(fraction * f_star, cap)


def bankroll_report(sim: dict, bankroll0: float) -> dict:
    """Summarise a ``simulate_bankroll`` result: terminal percentiles, P(profit),
    P(effective bust < 10% of start), and the drawdown distribution."""
    terminal = sim["terminal"]
    dd = sim["max_drawdown"]
    return {
        "median_terminal": float(np.median(terminal)),
        "p5_terminal": float(np.percentile(terminal, 5)),
        "p95_terminal": float(np.percentile(terminal, 95)),
        "p_profit": float(np.mean(terminal > bankroll0)),
        "p_bust": float(np.mean(terminal < 0.1 * bankroll0)),
        "median_max_drawdown": float(np.median(dd)),
        "p_drawdown_over_50pct": float(np.mean(dd > 0.5)),
    }

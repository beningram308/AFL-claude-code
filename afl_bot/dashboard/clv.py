"""CLV (Closing Line Value) math and statistics — Phase 3.

Formula: clv_pct = (1/close_ref_odds) - (1/open_odds)
  Positive = you beat the closing line (market moved in, confirming edge).
  Consistent with walkforward.py: clv = close_prob - open_prob.

Sharp reference hierarchy (FIX-PHASE3-CLV.txt):
  H2H/line  : Betfair exchange (not yet connected -> clv_available=False)
  Props      : de-vigged consensus across >=2 books (needs 2nd scraper)
  Currently  : all bets marked clv_available=False.
               Adding Betfair (H2H) or a 2nd book scraper (props) is the unlock.
"""
from __future__ import annotations

import math

import numpy as np

from afl_bot.pricing.edge import devig_proportional


def compute_clv(open_odds: float, close_ref_odds: float) -> float:
    """CLV = (1/close_ref_odds) - (1/open_odds).  Positive when market moves in."""
    return 1.0 / close_ref_odds - 1.0 / open_odds


def devig_consensus(book_price_pairs: list[tuple[float, float]]) -> float:
    """Median de-vigged P(over/favourite) across >=2 independent books.

    Each element is (over_odds, under_odds) from one book.
    Raises ValueError when fewer than 2 books are supplied.
    """
    if len(book_price_pairs) < 2:
        raise ValueError(
            f"Need >=2 books for consensus de-vig; got {len(book_price_pairs)}")
    probs = [devig_proportional([o, u])[0] for o, u in book_price_pairs]
    return float(np.median(probs))


def min_detectable_edge(n: int, alpha: float = 0.05, sd: float | None = None) -> float:
    """MDE at 80% power, one-sided alpha, for n observations.

    MDE = (z_alpha + z_beta) * sd / sqrt(n).
    Uses sd=0.05 as fallback (conservative prior for prop CLV standard deviation)
    when sd is None, NaN, or zero.
    """
    if n <= 0:
        return math.inf
    _sd = sd if (sd is not None and not math.isnan(sd) and sd > 0) else 0.05
    z = 1.645 + 0.842  # z_alpha (5% one-sided) + z_beta (80% power)
    return _sd * z / math.sqrt(n)


def clv_stats(clv_values: list[float]) -> dict:
    """Aggregate CLV stats: n, mean, sd, t_stat, significant (one-sided 5%),
    pct_positive, min_detectable_edge."""
    n = len(clv_values)
    if n == 0:
        return {
            "n": 0, "mean_clv": None, "sd_clv": None, "t_stat": None,
            "significant": False, "pct_positive": None, "min_detectable_edge": None,
        }
    arr = np.array(clv_values, dtype=float)
    mean_clv = float(arr.mean())
    sd_clv = float(arr.std(ddof=1)) if n > 1 else None
    t_stat: float | None = None
    if sd_clv is not None and sd_clv > 0:
        t_stat = float(mean_clv / (sd_clv / math.sqrt(n)))
    significant = t_stat is not None and t_stat > 1.645
    pct_positive = float((arr > 0).mean())
    mde = min_detectable_edge(n, sd=sd_clv)
    return {
        "n": n,
        "mean_clv": mean_clv,
        "sd_clv": sd_clv,
        "t_stat": t_stat,
        "significant": significant,
        "pct_positive": pct_positive,
        "min_detectable_edge": mde,
    }


def clv_breakdown_by_market(bets: list[dict]) -> dict[str, dict]:
    """Per-market-type CLV stats.  Only includes bets where clv_available=True."""
    groups: dict[str, list[float]] = {}
    for b in bets:
        if not b.get("clv_available"):
            continue
        clv = b.get("clv_pct")
        if clv is None:
            continue
        legs = b.get("legs", [])
        mkt = legs[0].get("market", "unknown") if legs else "unknown"
        groups.setdefault(mkt, []).append(float(clv))
    return {mkt: clv_stats(vals) for mkt, vals in groups.items()}

"""CLV (Closing Line Value) math and statistics — Phase 3.

Formula: clv_pct = (1/close_ref_odds) - (1/open_odds)
  Positive = you beat the closing line (market moved in, confirming edge).
  Consistent with walkforward.py: clv = close_prob - open_prob.

Sharp reference hierarchy (FIX-PHASE3-CLV.txt / FIX-SECOND-BOOK-FOR-PROP-CLV.txt):
  H2H/line  : 2-book consensus (Sportsbet + TAB) when both books price the leg.
              Betfair exchange is the future upgrade for a sharper H2H reference.
  Props      : de-vigged consensus across Sportsbet + TAB (now active).
  Single-book: clv_available=False (soft-self comparison is meaningless).
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


def devig_consensus_single_sided(
    over_odds_list: list[float],
    assumed_overround: float | None = None,
) -> float:
    """Median de-vigged P(over/favourite) from >=2 books (over side only).

    Uses single-sided devig: P_fair = (1/odds) / assumed_overround per book,
    then returns the median probability across books.

    Raises ValueError when fewer than 2 books are supplied.
    """
    from afl_bot.config import PROP_ASSUMED_OVERROUND as _DEFAULT
    _or = assumed_overround if assumed_overround is not None else _DEFAULT
    if len(over_odds_list) < 2:
        raise ValueError(
            f"Need >=2 books for single-sided consensus; got {len(over_odds_list)}"
        )
    probs = [(1.0 / o) / _or for o in over_odds_list]
    return float(np.median(probs))


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

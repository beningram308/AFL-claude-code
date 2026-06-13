"""
Pricing & edge detection (plan §4).

Turns simulation samples into fair probabilities/odds, devigs market prices,
computes per-leg edge, and classifies legs as ANCHOR / VALUE / SKIP for the
multi builder (stage 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from afl_bot.config import ANCHOR_MIN_PROB, VALUE_MIN_EDGE, VALUE_PROB_RANGE


# ----------------------------------------------------------------------------- #
# Sim sample -> probability / fair odds
# ----------------------------------------------------------------------------- #
def prob_over(samples: np.ndarray, line: float) -> float:
    """P(stat >= line). Use for 'X+' markets (e.g. '2+ goals' -> line=2)."""
    return float(np.mean(samples >= line))


def prob_under(samples: np.ndarray, line: float) -> float:
    """P(stat < line). Complement of prob_over for the same line."""
    return float(np.mean(samples < line))


def prob_event(mask: np.ndarray) -> float:
    """P(event) from a boolean per-iteration mask, e.g. (margin > 0) for a win,
    or a combined mask across legs for a joint SGM probability (plan §3.4)."""
    return float(np.mean(mask))


def fair_odds(prob: float) -> float:
    return float("inf") if prob <= 0 else 1.0 / prob


# ----------------------------------------------------------------------------- #
# Market odds -> implied / no-vig probabilities (plan §4.1)
# ----------------------------------------------------------------------------- #
def implied_prob(odds: float) -> float:
    return 1.0 / odds


def devig_proportional(odds: list[float]) -> list[float]:
    """Proportional ("multiplicative") devig across a complete market (e.g. all
    outcomes of a H2H, or both sides of an over/under line). Each implied prob
    is scaled down by the market's total overround so the probabilities sum to 1.
    """
    implied = [implied_prob(o) for o in odds]
    overround = sum(implied)
    if overround <= 0:
        raise ValueError("Sum of implied probabilities must be positive")
    return [p / overround for p in implied]


def market_anchored_prob(prob: float, odds: float, weight: float) -> float:
    """Pull a leg's model prob ``weight`` of the way toward its market-implied
    prob (1/odds) — a conservative haircut so per-leg overestimates don't
    compound multiplicatively across a multi (round-2 §8.2)."""
    if odds <= 1.0 or not (0.0 <= weight <= 1.0):
        return prob
    return (1.0 - weight) * prob + weight * implied_prob(odds)


def mc_standard_error(prob: float, n_sims: int) -> float:
    """Binomial standard error of a Monte-Carlo probability estimate
    ``sqrt(p(1-p)/n)`` (round-2 §8.3)."""
    if n_sims <= 0:
        return float("inf")
    return float((prob * (1.0 - prob) / n_sims) ** 0.5)


# ----------------------------------------------------------------------------- #
# Edge (plan §4.1)
# ----------------------------------------------------------------------------- #
def edge(prob: float, market_odds: float) -> float:
    """Model EV per $1 staked at market_odds, before any promo. >0 is +EV."""
    return prob * market_odds - 1.0


def edge_vs_devig(model_prob: float, devig_prob: float) -> float:
    """Difference between the model's probability and the market's no-vig
    probability. A large gap is a signal to double-check the model first --
    the market is usually right (plan §0, §4.1)."""
    return model_prob - devig_prob


# ----------------------------------------------------------------------------- #
# Leg classification (plan §4.2)
# ----------------------------------------------------------------------------- #
@dataclass
class Leg:
    name: str
    fair_prob: float
    market_odds: float
    classification: str = field(init=False)
    edge_pct: float = field(init=False)
    devig_prob: float | None = None

    def __post_init__(self) -> None:
        self.edge_pct = edge(self.fair_prob, self.market_odds)
        self.classification = classify_leg(self.fair_prob, self.market_odds)

    @property
    def fair_odds(self) -> float:
        return fair_odds(self.fair_prob)


def classify_leg(
    prob: float, market_odds: float,
    anchor_p: float = ANCHOR_MIN_PROB,
    value_edge: float = VALUE_MIN_EDGE,
    value_prob_range: tuple[float, float] = VALUE_PROB_RANGE,
) -> str:
    """ANCHOR: very high probability, low-variance "lock" leg.
    VALUE: positive-edge leg in the sweet spot where prop mispricings concentrate.
    SKIP: neither.
    """
    e = edge(prob, market_odds)
    if prob >= anchor_p:
        return "ANCHOR"
    lo, hi = value_prob_range
    if e >= value_edge and lo <= prob <= hi:
        return "VALUE"
    return "SKIP"

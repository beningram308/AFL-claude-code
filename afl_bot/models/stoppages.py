"""
Boundary throw-in / out-of-bounds (OOB) market model (plan §1.6c).

Boundary throw-ins are overdispersed and driven by congestion and weather: a
scrappy, low-scoring, wet game has *more* throw-ins, an open high-scoring game
fewer — i.e. OOB is negatively correlated with the total. This model:

  * takes an expected game OOB count ``mu_oob`` (from real per-game counts via
    ``expected_oob`` when they flow, else the ``LEAGUE_OOB_PER_GAME`` prior);
  * couples it to the match sim's per-iteration total-points draw so OOB rises
    when the total falls (``OOB_TOTAL_COUPLING``); and
  * lifts it in the wet (``OOB_RAIN_MULTIPLIER``), reusing the same wet flag as
    the prop multipliers (plan §3.4);
  * draws the per-iteration count as Negative Binomial, so the market can be
    priced (over/under) coherently with the rest of the simulation.

Until a real boundary-throw-in feed is wired (``afl_bot.data.stoppages``), the
prices are prior-based — flagged as such by the CLI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afl_bot.config import (
    LEAGUE_OOB_PER_GAME,
    OOB_DISPERSION,
    OOB_RAIN_MULTIPLIER,
    OOB_TOTAL_COUPLING,
)
from afl_bot.data.stoppages import BOUNDARY_THROWIN_COL


def expected_oob(stoppage_log: pd.DataFrame | None = None,
                 prior: float = LEAGUE_OOB_PER_GAME) -> float:
    """Expected boundary throw-ins per game: the mean of real per-game counts
    when a feed has been loaded, else the league ``prior``."""
    if stoppage_log is None or stoppage_log.empty or BOUNDARY_THROWIN_COL not in stoppage_log.columns:
        return float(prior)
    mean = stoppage_log[BOUNDARY_THROWIN_COL].mean()
    return float(mean) if np.isfinite(mean) else float(prior)


def simulate_boundary_throwins(
    mu_oob: float, total_points: np.ndarray, rng: np.random.Generator, *,
    is_wet: bool = False, dispersion: float = OOB_DISPERSION,
    total_coupling: float = OOB_TOTAL_COUPLING, rain_multiplier: float = OOB_RAIN_MULTIPLIER,
) -> np.ndarray:
    """Per-iteration boundary-throw-in count, coupled to the match sim's
    ``total_points`` draw (plan §1.6c).

    The per-iteration mean is ``mu_oob * (mean_total / total_iter)**total_coupling``
    (so a low-total, congested iteration carries more throw-ins — a negative
    OOB/total correlation), times ``rain_multiplier`` when the game is wet, then
    drawn as NB(mean, dispersion). ``total_points`` should be the array from
    ``afl_bot.sim.engine.simulate_match`` (``home_pts + away_pts``).
    """
    total_points = np.asarray(total_points, dtype=float)
    mean_total = total_points.mean()
    ratio = np.clip(mean_total / np.clip(total_points, 1.0, None), 0.5, 2.0)

    per_iter_mean = np.clip(mu_oob * ratio ** total_coupling, 1e-6, None)
    if is_wet:
        per_iter_mean = per_iter_mean * rain_multiplier

    r = float(dispersion)
    p = r / (r + per_iter_mean)
    return rng.negative_binomial(r, p)

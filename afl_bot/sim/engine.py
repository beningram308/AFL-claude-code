"""
Monte Carlo match + player-prop engine (plan §2.1, §3, §3.5).

This is the simulation core (stage 4): it turns team ratings + player rate
models into per-iteration scorelines and player stat draws, with the
within-game correlation that makes a simulation worth more than multiplying
independent probabilities.

Key modelling choices:
  * Scoring shots, not Normal margin/total (plan §2.1): each team's scoring
    shots (goals + behinds) ~ NegativeBinomial(mu_shots, SHOT_DISPERSION), and
    goals ~ Binomial(shots, accuracy) where accuracy is drawn per-iteration
    around the team's EWMA goal-conversion rate (SHOT_ACCURACY_SIGMA).
    Points = 6*goals + behinds. ``mu_shots`` is derived from the existing
    Elo margin + scoring-profile total via
    ``afl_bot.models.scoring.points_to_shots``. This gives integer scores,
    real draw probabilities, correct margin tails, and -- because NB variance
    grows with its mean -- naturally heteroscedastic sigma (plan §2.2):
    higher-scoring games carry more shot-count variance.
  * Player counting stats: Negative Binomial, not Poisson -- disposals/marks/
    tackles are overdispersed (variance > mean). Dispersion is estimated per
    player in afl_bot.models.props.estimate_dispersion.
  * Correlation: props are scaled by *this iteration's* team scoreline (goals
    or total points), so a team's forwards/mids move together with its score
    -- the real correlation books price for SGMs.

Pricing (fair odds, edge, leg classification, promo EV) lives in
afl_bot.pricing.edge -- this module only produces simulation samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afl_bot.config import (
    DEFAULT_SHOT_ACCURACY,
    PACE_SIGMA,
    RNG_SEED,
    SCORE_SHOT_CORRELATION,
    SHARE_CONCENTRATION,
    SHOT_ACCURACY_BOUNDS,
    SHOT_ACCURACY_SIGMA,
    SHOT_DISPERSION,
    TEAM_STAT_DISPERSION,
    WET_ACCURACY_PENALTY,
    WET_TOTAL_MULTIPLIER,
)
from afl_bot.models.scoring import points_to_shots


# ----------------------------------------------------------------------------- #
# Inputs
# ----------------------------------------------------------------------------- #
@dataclass
class Team:
    name: str
    is_home: bool = False
    travel_penalty: float = 0.0   # extra points handicap for genuine interstate travel


@dataclass
class Player:
    name: str
    team: str
    stat: str                  # "disposals" | "goals" | "marks" | "tackles" ...
    mean: float                # expected count this game (after form/role/matchup)
    dispersion: float          # NB dispersion r (lower = more overdispersed)
    scales_with: str = "none"  # "goals" | "score" | "none" -> correlation driver


# ----------------------------------------------------------------------------- #
# Match-level simulation
# ----------------------------------------------------------------------------- #
def simulate_team_score(
    mu_shots: float, accuracy: float, n: int, rng: np.random.Generator,
    dispersion: float = SHOT_DISPERSION, accuracy_sigma: float = SHOT_ACCURACY_SIGMA,
    accuracy_bounds: tuple[float, float] = SHOT_ACCURACY_BOUNDS,
    shots_uniform: np.ndarray | None = None,
) -> dict:
    """Vectorised Monte Carlo of one team's scoreline (plan §2.1):
    shots ~ NB(mu_shots, dispersion), goals ~ Binomial(shots, accuracy_draw),
    behinds = shots - goals, points = 6*goals + behinds.

    ``accuracy_draw`` is a per-iteration Normal perturbation of ``accuracy``
    (the team's EWMA goal-conversion rate), clipped to ``accuracy_bounds`` --
    this is the source of "shooting for goal" variance on top of shot-volume
    variance.

    ``shots_uniform`` (optional) is a per-iteration array of U(0,1) draws used
    to invert the NB CDF instead of sampling it directly. ``simulate_match``
    passes *correlated* uniforms here to couple the two teams' shot counts
    (plan §3.3 Gaussian copula) while preserving the NB marginal exactly.
    """
    mu_shots = max(mu_shots, 1e-6)
    p = dispersion / (dispersion + mu_shots)
    if shots_uniform is None:
        shots = rng.negative_binomial(dispersion, p, n)
    else:
        from scipy.stats import nbinom
        shots = nbinom.ppf(shots_uniform, dispersion, p).astype(np.int64)

    accuracy_draw = np.clip(rng.normal(accuracy, accuracy_sigma, n), *accuracy_bounds)
    goals = rng.binomial(shots, accuracy_draw)
    behinds = shots - goals
    points = 6 * goals + behinds

    return {"shots": shots, "goals": goals, "behinds": behinds, "points": points}


def simulate_match(
    home: Team, away: Team, mu_margin: float, mu_total: float,
    home_accuracy: float, away_accuracy: float, n: int,
    rng: np.random.Generator,
    score_correlation: float = SCORE_SHOT_CORRELATION,
    is_wet: bool = False,
    shot_dispersion: float = SHOT_DISPERSION,
) -> dict:
    """Vectorised Monte Carlo of n scorelines via the scoring-shots model
    (plan §2.1, §3.3).

    ``mu_margin`` (predicted home margin, points) and ``mu_total`` (predicted
    combined points) come from afl_bot.ratings.elo / afl_bot.models.scoring;
    ``home_accuracy``/``away_accuracy`` come from
    afl_bot.models.scoring.team_shot_accuracy_profiles (use
    ``DEFAULT_SHOT_ACCURACY`` for teams without history -- pass ``float("nan")``
    and it will be substituted). The expected points split is converted to
    expected scoring shots via afl_bot.models.scoring.points_to_shots.

    The two teams' scoring-shot counts are coupled by a Gaussian copula with
    correlation ``score_correlation`` (negative for AFL: when one team owns
    territory the other scores less, plan §3.3). The copula leaves each team's
    NB marginal untouched, so means and per-team variance are unchanged from
    the independent case -- only the dependence (and hence the margin/total
    sigma split) changes. ``score_correlation=0`` recovers independence.
    """
    if home_accuracy != home_accuracy:  # NaN check without importing math/np scalar
        home_accuracy = DEFAULT_SHOT_ACCURACY
    if away_accuracy != away_accuracy:
        away_accuracy = DEFAULT_SHOT_ACCURACY

    # Wet weather (round-2 §4.1): fewer points overall and lower goal conversion
    # (more behinds), so the total/margin/H2H markets move with the wet props.
    if is_wet:
        mu_total = mu_total * WET_TOTAL_MULTIPLIER
        home_accuracy -= WET_ACCURACY_PENALTY
        away_accuracy -= WET_ACCURACY_PENALTY

    mu_home_pts = max((mu_total + mu_margin) / 2.0, 0.0)
    mu_away_pts = max((mu_total - mu_margin) / 2.0, 0.0)

    mu_home_shots = points_to_shots(mu_home_pts, home_accuracy)
    mu_away_shots = points_to_shots(mu_away_pts, away_accuracy)

    u_home = u_away = None
    if score_correlation:
        from scipy.stats import norm
        z_home = rng.standard_normal(n)
        z_indep = rng.standard_normal(n)
        z_away = score_correlation * z_home + np.sqrt(1.0 - score_correlation ** 2) * z_indep
        u_home = norm.cdf(z_home)
        u_away = norm.cdf(z_away)

    home_score = simulate_team_score(mu_home_shots, home_accuracy, n, rng,
                                     dispersion=shot_dispersion, shots_uniform=u_home)
    away_score = simulate_team_score(mu_away_shots, away_accuracy, n, rng,
                                     dispersion=shot_dispersion, shots_uniform=u_away)

    home_pts = home_score["points"].astype(float)
    away_pts = away_score["points"].astype(float)
    margin = home_pts - away_pts

    return {
        "home_pts": home_pts, "away_pts": away_pts,
        "home_goals": home_score["goals"], "away_goals": away_score["goals"],
        "home_behinds": home_score["behinds"], "away_behinds": away_score["behinds"],
        "home_shots": home_score["shots"], "away_shots": away_score["shots"],
        "margin": margin,
        "home_win": (margin > 0).astype(float),
        "away_win": (margin < 0).astype(float),
        "draw": (margin == 0).astype(float),
    }


# ----------------------------------------------------------------------------- #
# Player props -- Negative Binomial, correlated with the scoreline
# ----------------------------------------------------------------------------- #
def _nb_sample(mean: np.ndarray, dispersion: float, rng: np.random.Generator) -> np.ndarray:
    """Negative Binomial samples with a per-iteration mean array (correlation hook).

    Parameterised by (r, p) for mean m and dispersion r:
        p = r / (r + m)  ->  mean = r(1-p)/p = m,  variance = m + m^2/r  (> m).
    """
    mean = np.clip(mean, 1e-6, None)
    r = float(dispersion)
    p = r / (r + mean)
    return rng.negative_binomial(r, p)


def simulate_player(player: Player, match: dict, home: Team, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw n samples of a player's stat, scaled by THIS iteration's team
    scoreline so the prop is correlated with how the game actually unfolds."""
    is_home = (player.team == home.name)

    if player.scales_with == "goals":
        team_goals = match["home_goals"] if is_home else match["away_goals"]
        base_team_goals = np.mean(team_goals) + 1e-6
        scale = team_goals / base_team_goals
        per_iter_mean = player.mean * scale
    elif player.scales_with == "score":
        team_pts = match["home_pts"] if is_home else match["away_pts"]
        base_pts = np.mean(team_pts) + 1e-6
        scale = team_pts / base_pts
        per_iter_mean = player.mean * scale
    else:
        per_iter_mean = np.full(n, player.mean)

    return _nb_sample(per_iter_mean, player.dispersion, rng)


# ----------------------------------------------------------------------------- #
# Environment latent factor & within-team share allocation (plan §2.5, §3.3)
# ----------------------------------------------------------------------------- #
def draw_pace(n: int, rng: np.random.Generator, pace_sigma: float = PACE_SIGMA) -> np.ndarray:
    """A shared per-iteration "pace" multiplier (mean 1.0) for the whole match.

    Lognormal so it's strictly positive and right-skewed; the location is set
    to ``-sigma^2/2`` so ``E[pace] == 1``. Drawing it once per match and feeding
    the SAME array into both teams' volume-stat totals is what makes disposals
    (and other tempo stats) correlate *across* teams -- a fast, open game lifts
    everyone (plan §2.5: draw the game environment first, then condition team
    totals and player props on it).
    """
    return rng.lognormal(mean=-pace_sigma ** 2 / 2.0, sigma=pace_sigma, size=n)


def simulate_team_stat_total(
    mu_team_stat: float, pace: np.ndarray, rng: np.random.Generator,
    dispersion: float = TEAM_STAT_DISPERSION,
) -> np.ndarray:
    """A team's per-iteration total for a volume stat (e.g. disposals), as an
    NB draw around the pace-scaled expected total ``mu_team_stat * pace``.

    Because ``pace`` is shared across both teams, two calls with the same pace
    array produce positively-correlated team totals (the §2.5 environment
    coupling); the NB term adds the team-specific spread on top.
    """
    mean = np.clip(mu_team_stat * np.asarray(pace, dtype=float), 1e-6, None)
    p = dispersion / (dispersion + mean)
    return rng.negative_binomial(dispersion, p)


def allocate_player_stats(
    team_total: np.ndarray, expected_shares: np.ndarray, rng: np.random.Generator,
    concentration: float = SHARE_CONCENTRATION,
) -> np.ndarray:
    """Allocate a per-iteration team total among players via a Dirichlet draw
    so that, every iteration, the players' counts sum to (a rounded) team total
    (plan §3.3 within-team share constraint).

    ``expected_shares`` are the priced players' expected fractions of the team
    total; any remainder ``1 - sum(expected_shares)`` is held by an implicit
    "other players" bucket so the priced players don't absorb 100% of the team.
    A per-iteration share vector is drawn from ``Dirichlet(concentration * [
    shares..., other])`` -- higher ``concentration`` keeps shares closer to
    expectation. Returns an integer array shaped ``(n_players, n_iter)``.

    The Dirichlet's built-in constraint (shares sum to 1) makes teammates'
    draws *negatively* correlated -- if one mid racks up possessions another
    gets fewer -- which is exactly the structure books exploit in same-game
    multis and that independent NB draws get wrong.
    """
    team_total = np.asarray(team_total, dtype=float)
    n = team_total.shape[0]
    shares = np.asarray(expected_shares, dtype=float)
    other = max(1.0 - float(shares.sum()), 1e-6)

    alpha = concentration * np.append(shares, other)
    draws = rng.dirichlet(alpha, size=n)              # (n, n_players + 1)
    player_fracs = draws[:, :len(shares)]             # drop the "other" bucket
    counts = np.rint(player_fracs * team_total[:, None]).astype(np.int64)
    return counts.T                                   # (n_players, n_iter)


def allocate_player_goals(
    team_goals: np.ndarray, expected_shares: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Allocate a per-iteration team GOAL count among players by a Multinomial
    draw on their goal shares (round-2 §3.3), so goal-stack SGMs are
    sum-constrained the way disposals are by the Dirichlet — two forwards can't
    both kick big bags in the same iteration beyond what the team actually
    kicked.

    Implemented as the standard sequential-conditional-binomial multinomial,
    vectorised over iterations (numpy's multinomial takes only a scalar count).
    Any remainder stays with an implicit "other players" bucket. Returns an
    integer array shaped ``(n_players, n_iter)``.
    """
    team_goals = np.asarray(team_goals, dtype=np.int64)
    shares = np.asarray(expected_shares, dtype=float)
    other = max(1.0 - float(shares.sum()), 1e-9)
    probs = np.append(shares, other)
    probs = probs / probs.sum()

    remaining = team_goals.copy()
    remaining_p = 1.0
    out = np.zeros((len(shares), team_goals.shape[0]), dtype=np.int64)
    for k in range(len(shares)):
        p_k = float(np.clip(probs[k] / remaining_p, 0.0, 1.0)) if remaining_p > 0 else 0.0
        g = rng.binomial(remaining, p_k)
        out[k] = g
        remaining = remaining - g
        remaining_p -= probs[k]
    return out


def make_rng(seed: int = RNG_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)

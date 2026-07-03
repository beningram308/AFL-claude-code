"""
Fit the correlation/dispersion constants that drive SGM `corr_gain` from
history instead of hand-setting them -- model-upgrade audit Phase 2.

`SCORE_SHOT_CORRELATION`, `PACE_SIGMA`, `SHARE_CONCENTRATION`,
`SHOT_DISPERSION`, `TEAM_STAT_DISPERSION` (config.py) are currently hand-set
constants that determine `corr_gain` -- the whole reason to price SGMs off
the sim rather than naive per-leg multiplication. This module estimates each
from real history (pre-eval seasons only, anti-leakage) and writes a
versioned JSON artifact, mirroring `afl_bot.backtest.tuning.fit_elo_params` /
`load_fitted_elo_params`.

Derivations (closed-form where the model's distributional assumptions make
that exact, simulate-and-root-find where they don't):

  * `SHOT_DISPERSION` -- team scoring shots (goals+behinds) per game ~
    NB(mu, r). Method-of-moments: ``r = mu^2 / (var - mu)``.
  * `PACE_SIGMA` / `TEAM_STAT_DISPERSION` -- jointly identified in closed
    form from the empirical (mean, variance, home/away correlation) of a
    team's volume-stat total. Conditional on the shared lognormal pace P
    (E[P]=1, Var[P]=v), the model says (exactly, since the NB draw given P
    is independent across teams):
        Cov(home_total, away_total) = mu_home * mu_away * v
        Var(total)                  = mu + mu^2 * [(1+v)/r + v]
    so pooling both sides to a single representative mu (the same "single
    global approximation" the existing config constants already are):
        v = corr_emp * var_emp / mu_emp^2                     -> PACE_SIGMA = sqrt(ln(1+v))
        r = mu_emp^2*(1+v) / (var_emp - mu_emp - mu_emp^2*v)  -> TEAM_STAT_DISPERSION
  * `SHARE_CONCENTRATION` -- a Dirichlet share's marginal variance is
    `share*(1-share)/(C+1)`. The model draws a priced player's count as
    `team_total * share` with the two factors independent (team_total is
    drawn first, the Dirichlet share second), so `Var(count)` decomposes in
    closed form into the team-total variance (already fitted above) and the
    share variance -- inverting that for a representative top-usage cohort's
    empirical disposal CoV and average team-share gives `C`.
  * `SCORE_SHOT_CORRELATION` -- the Gaussian-copula parameter coupling the
    two teams' NB shot draws in `simulate_match`/`simulate_team_score`. The
    NB quantile transform makes rho -> resulting Pearson correlation
    nonlinear, so this one is found by root-finding the same copula code
    path (with common random numbers, so the search objective is a smooth,
    deterministic function of rho) against the empirical home/away shot
    correlation.

Every estimate is a *single global* constant (matching the existing
config-constant design), not per-team/per-player.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nbinom, norm

from afl_bot.config import CACHE_DIR, ROLE_MID_DISPOSALS_MIN, ROOT_DIR, SHOT_DISPERSION

CORRELATION_PARAMS_ARTIFACT = "correlation_params"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def fit_shot_dispersion(games: pd.DataFrame) -> dict:
    """Method-of-moments NB dispersion for team scoring shots (goals+behinds)
    per game, pooling both sides of every game."""
    shots = pd.concat([
        games["hgoals"] + games["hbehinds"],
        games["agoals"] + games["abehinds"],
    ], ignore_index=True).astype(float)
    mu = float(shots.mean())
    var = float(shots.var(ddof=1))
    r = mu ** 2 / (var - mu) if var > mu else float("nan")
    return {"value": r, "mean": mu, "var": var, "n": int(len(shots))}


def _paired_team_totals(player_log: pd.DataFrame, stat: str) -> pd.DataFrame:
    """One row per game with `total_home`/`total_away` for `stat`, built from
    `team_<stat>` (computed if missing) and the `is_home`/`opponent` columns
    every player-log source carries -- pairs each match's two sides via
    `opponent`, not just `(year, round)`, so simultaneous fixtures in the
    same round can't cross-pair."""
    if player_log.empty:
        return pd.DataFrame(columns=["year", "round", "total_home", "total_away"])
    log = player_log.copy()
    col = f"team_{stat}"
    if col not in log.columns:
        log[col] = log.groupby(["year", "round", "team"])[stat].transform("sum")
    team_games = log.drop_duplicates(["year", "round", "team"])[
        ["year", "round", "team", "opponent", "is_home", col]
    ]
    home = team_games[team_games["is_home"]]
    away = team_games[~team_games["is_home"]]
    merged = home.merge(
        away, left_on=["year", "round", "opponent"], right_on=["year", "round", "team"],
        suffixes=("_home", "_away"),
    )
    return merged.rename(columns={f"{col}_home": "total_home", f"{col}_away": "total_away"})


def fit_pace_and_dispersion(player_log: pd.DataFrame, stat: str = "disposals") -> dict:
    """Closed-form `(PACE_SIGMA, TEAM_STAT_DISPERSION)` from the empirical
    (mean, variance, home/away correlation) of team `stat` totals -- see
    module docstring for the derivation."""
    merged = _paired_team_totals(player_log, stat)
    if merged.empty:
        return {"pace_sigma": float("nan"), "team_stat_dispersion": float("nan"),
                "mean": float("nan"), "var": float("nan"), "corr": float("nan"), "n": 0}

    totals = pd.concat([merged["total_home"], merged["total_away"]], ignore_index=True).astype(float)
    mu = float(totals.mean())
    var = float(totals.var(ddof=1))
    corr = float(merged["total_home"].corr(merged["total_away"]))

    v = corr * var / mu ** 2 if mu > 0 else float("nan")
    pace_sigma = float(np.sqrt(np.log(1.0 + v))) if v > 0 else float("nan")
    denom = var - mu - mu ** 2 * v if np.isfinite(v) else float("nan")
    r = mu ** 2 * (1.0 + v) / denom if np.isfinite(denom) and denom > 0 else float("nan")
    return {
        "pace_sigma": pace_sigma, "team_stat_dispersion": r,
        "mean": mu, "var": var, "corr": corr, "n": int(len(merged)),
    }


def fit_share_concentration(player_log: pd.DataFrame, team_stat_mean: float, team_stat_var: float,
                            stat: str = "disposals", min_games: int = 15,
                            mid_threshold: float = ROLE_MID_DISPOSALS_MIN) -> dict:
    """Closed-form Dirichlet `SHARE_CONCENTRATION` from a representative
    top-usage cohort's (players averaging >= `mid_threshold` `stat`/game,
    with >= `min_games`) empirical CoV, their average team-share, and the
    already-fitted team-total CoV (`team_stat_mean`/`team_stat_var`)."""
    grp = player_log.groupby("player")[stat].agg(["mean", "std", "count"])
    cohort = grp[(grp["mean"] >= mid_threshold) & (grp["count"] >= min_games)]
    if cohort.empty or not (team_stat_mean and team_stat_mean > 0):
        return {"value": float("nan"), "n_players": 0, "share": float("nan"),
                "cov_player2": float("nan"), "cov_team2": float("nan")}

    share = float((cohort["mean"] / team_stat_mean).mean())
    cov_player2 = float(((cohort["std"] / cohort["mean"]) ** 2).mean())
    cov_team2 = team_stat_var / team_stat_mean ** 2

    excess = cov_player2 - cov_team2
    if excess <= 0:
        return {"value": float("nan"), "n_players": int(len(cohort)), "share": share,
                "cov_player2": cov_player2, "cov_team2": cov_team2}
    share_var = excess * share ** 2 / (1.0 + cov_team2)
    concentration = share * (1.0 - share) / share_var - 1.0
    return {
        "value": float(concentration), "n_players": int(len(cohort)), "share": share,
        "cov_player2": cov_player2, "cov_team2": cov_team2,
    }


def _shot_correlation_at_rho(rho: float, z_home: np.ndarray, z_indep: np.ndarray,
                             mu_shots: float, dispersion: float) -> float:
    z_away = rho * z_home + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z_indep
    u_home, u_away = norm.cdf(z_home), norm.cdf(z_away)
    p = dispersion / (dispersion + mu_shots)
    shots_home = nbinom.ppf(u_home, dispersion, p)
    shots_away = nbinom.ppf(u_away, dispersion, p)
    return float(np.corrcoef(shots_home, shots_away)[0, 1])


def fit_score_shot_correlation(games: pd.DataFrame, shot_dispersion: float = SHOT_DISPERSION,
                               n_sims: int = 200_000, seed: int = 2024) -> dict:
    """Root-finds the Gaussian-copula `score_correlation` parameter
    (`simulate_match`) that reproduces the empirical home/away scoring-shot
    correlation, via common random numbers (one fixed `z_home`/`z_indep`
    pair reused across every rho tried) so the search objective is smooth
    and deterministic rather than refreshing Monte Carlo noise each call."""
    home_shots = (games["hgoals"] + games["hbehinds"]).astype(float)
    away_shots = (games["agoals"] + games["abehinds"]).astype(float)
    target = float(home_shots.corr(away_shots))
    mu_shots = float(pd.concat([home_shots, away_shots]).mean())

    rng = np.random.default_rng(seed)
    z_home = rng.standard_normal(n_sims)
    z_indep = rng.standard_normal(n_sims)

    def objective(rho: float) -> float:
        return _shot_correlation_at_rho(rho, z_home, z_indep, mu_shots, shot_dispersion) - target

    lo, hi = -0.995, 0.995
    if objective(lo) * objective(hi) > 0:
        # Target outside what this (mu, dispersion) pair can reach via the
        # copula -- fall back to the empirical correlation itself (a
        # reasonable approximation) and flag it rather than crash.
        return {"value": float(np.clip(target, lo, hi)), "target": target, "bracket_failed": True}
    rho_fit = brentq(objective, lo, hi, xtol=1e-4)
    return {"value": float(rho_fit), "target": target, "bracket_failed": False}


def fit_correlation_params(games: pd.DataFrame, player_log: pd.DataFrame, *,
                           train_end_year: int, cache_dir=CACHE_DIR) -> dict:
    """Fit all five constants on games/player_log up to and including
    `train_end_year` only (anti-leakage -- callers backtest later seasons),
    write a versioned JSON artifact mirroring
    `afl_bot.backtest.tuning.fit_elo_params`, and return it. A value that
    couldn't be estimated (NaN) is simply omitted from `params`, so
    `load_fitted_correlation_params` falls back to the config default for it."""
    train_games = games[games["year"] <= train_end_year]
    train_log = player_log[player_log["year"] <= train_end_year]

    shot_disp = fit_shot_dispersion(train_games)
    pace = fit_pace_and_dispersion(train_log, "disposals")
    share = fit_share_concentration(train_log, pace["mean"], pace["var"], "disposals")
    score_corr = fit_score_shot_correlation(
        train_games,
        shot_dispersion=shot_disp["value"] if np.isfinite(shot_disp["value"]) else SHOT_DISPERSION,
    )

    params = {
        "SHOT_DISPERSION": shot_disp["value"],
        "PACE_SIGMA": pace["pace_sigma"],
        "TEAM_STAT_DISPERSION": pace["team_stat_dispersion"],
        "SHARE_CONCENTRATION": share["value"],
        "SCORE_SHOT_CORRELATION": score_corr["value"],
    }
    artifact = {
        "params": {k: v for k, v in params.items() if np.isfinite(v)},
        "diagnostics": {
            "shot_dispersion": shot_disp, "pace_and_team_dispersion": pace,
            "share_concentration": share, "score_shot_correlation": score_corr,
        },
        "train_end_year": int(train_end_year),
        "n_games": int(len(train_games)),
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    from afl_bot.io_utils import atomic_write_text
    atomic_write_text(cache_dir / f"{CORRELATION_PARAMS_ARTIFACT}.json", json.dumps(artifact, indent=2))
    return artifact


def load_fitted_correlation_params(cache_dir=CACHE_DIR) -> dict:
    """The fitted correlation/dispersion params from the latest
    `fit-correlations` artifact, or `{}` if none (callers then use the config
    defaults). Opt-in, same contract as `load_fitted_elo_params`."""
    path = cache_dir / f"{CORRELATION_PARAMS_ARTIFACT}.json"
    if not path.exists():
        return {}
    try:
        params = json.loads(path.read_text()).get("params", {})
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in params.items() if isinstance(v, (int, float)) and np.isfinite(v)}

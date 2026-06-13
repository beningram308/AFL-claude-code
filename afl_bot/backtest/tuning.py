"""
Elo hyperparameter tuning on the walk-forward harness (plan §2.3, build-order
step 5).

Tunes the ``EloRatings`` hyperparameters (k, season_carryover, home_advantage,
margin_cap, points_per_400, and — for the "mov" update mode — mov_correction)
against a held-out window of recent seasons, minimising a joint objective of
out-of-sample log loss + margin MAE. Two search strategies:

  * ``grid_search_elo``   — exhaustive over an explicit grid (no extra deps).
  * ``optuna_search_elo`` — TPE Bayesian search (optional ``optuna`` dependency).

Both reuse ``afl_bot.backtest.walkforward.evaluate_elo``, so the anti-leakage /
walk-forward property is preserved: every game is still scored on its pre-match
ratings, and the objective is computed only on the held-out evaluation window
(``eval_start_year``). For an honest result, tune on early seasons (passing
games only up to a cutoff, or with an ``eval_start_year`` inside the supplied
range) and report final metrics on a *further* untouched window of later
seasons that tuning never saw.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from afl_bot.backtest.walkforward import evaluate_elo, log_loss, margin_mae
from afl_bot.config import CACHE_DIR, ROOT_DIR

ELO_PARAMS_ARTIFACT = "elo_params"

# Margin MAE is ~25-30 pts and log loss ~0.6; this weight (~0.01) puts a few
# points of MAE on the same scale as a meaningful log-loss move so the joint
# objective trades them off sensibly rather than letting MAE dominate.
DEFAULT_MARGIN_MAE_WEIGHT = 0.01

# (low, high, is_int) search ranges for the Optuna sampler.
DEFAULT_ELO_PARAM_RANGES: dict[str, tuple[float, float, bool]] = {
    "k": (10.0, 60.0, False),
    "season_carryover": (0.30, 0.90, False),
    "home_advantage": (0.0, 25.0, False),
    "margin_cap": (40.0, 120.0, False),
    "points_per_400": (60.0, 120.0, False),
}

# Smaller explicit grid for the dependency-free exhaustive search.
DEFAULT_ELO_GRID: dict[str, list[float]] = {
    "k": [20.0, 30.0, 35.0, 45.0],
    "season_carryover": [0.50, 0.65, 0.80],
    "home_advantage": [5.0, 10.0, 15.0],
}


def elo_objective(
    games: pd.DataFrame, *,
    eval_start_year: int | None = None,
    margin_mae_weight: float = DEFAULT_MARGIN_MAE_WEIGHT,
    **elo_kwargs,
) -> dict:
    """Fit Elo with the given hyperparameters and score it on the held-out
    window (games with ``year >= eval_start_year``; all games if None).

    Returns the scalar ``objective`` (``log_loss + margin_mae_weight*margin_mae``)
    plus its components and the evaluated game count.
    """
    history = evaluate_elo(games, **elo_kwargs)
    if eval_start_year is not None:
        history = history[history["year"] >= eval_start_year]
    if history.empty:
        return {"objective": float("inf"), "log_loss": float("nan"),
                "margin_mae": float("nan"), "n_games": 0}

    ll = log_loss(history["pred_home_win_prob"].to_numpy(), history["actual_home_win"].to_numpy())
    mae = margin_mae(history["pred_margin"].to_numpy(), history["actual_margin"].to_numpy())
    return {
        "objective": ll + margin_mae_weight * mae,
        "log_loss": ll,
        "margin_mae": mae,
        "n_games": len(history),
    }


def grid_search_elo(
    games: pd.DataFrame,
    param_grid: dict[str, list] | None = None, *,
    eval_start_year: int | None = None,
    margin_mae_weight: float = DEFAULT_MARGIN_MAE_WEIGHT,
    fixed_params: dict | None = None,
) -> pd.DataFrame:
    """Exhaustive grid search over ``param_grid`` (defaults to
    ``DEFAULT_ELO_GRID``). Returns one row per parameter combination with its
    objective / log_loss / margin_mae, sorted best-first (lowest objective)."""
    param_grid = param_grid or DEFAULT_ELO_GRID
    fixed_params = fixed_params or {}
    keys = list(param_grid)

    rows = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = {**fixed_params, **dict(zip(keys, combo))}
        res = elo_objective(
            games, eval_start_year=eval_start_year,
            margin_mae_weight=margin_mae_weight, **params,
        )
        rows.append({**dict(zip(keys, combo)), **res})

    return pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)


def optuna_search_elo(
    games: pd.DataFrame, *,
    n_trials: int = 100,
    eval_start_year: int | None = None,
    margin_mae_weight: float = DEFAULT_MARGIN_MAE_WEIGHT,
    param_ranges: dict[str, tuple[float, float, bool]] | None = None,
    fixed_params: dict | None = None,
    seed: int = 42,
):
    """TPE Bayesian search over the Elo hyperparameters (optional ``optuna``
    dependency). Returns the completed optuna ``Study``: best hyperparameters on
    ``study.best_params``, best objective on ``study.best_value``, and the
    log_loss / margin_mae of the best trial in its ``user_attrs``."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised only without optuna
        raise ImportError(
            "optuna is required for optuna_search_elo; install it with "
            "`pip install optuna`."
        ) from exc

    param_ranges = param_ranges or DEFAULT_ELO_PARAM_RANGES
    fixed_params = fixed_params or {}

    def objective(trial: "optuna.Trial") -> float:
        params = dict(fixed_params)
        for name, (lo, hi, is_int) in param_ranges.items():
            params[name] = (
                trial.suggest_int(name, int(lo), int(hi)) if is_int
                else trial.suggest_float(name, lo, hi)
            )
        res = elo_objective(
            games, eval_start_year=eval_start_year,
            margin_mae_weight=margin_mae_weight, **params,
        )
        trial.set_user_attr("log_loss", res["log_loss"])
        trial.set_user_attr("margin_mae", res["margin_mae"])
        return res["objective"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# ----------------------------------------------------------------------------- #
# Versioned fit artifact (round-2 §6.2 / plan §5.2, §5.4)
# ----------------------------------------------------------------------------- #
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def fit_elo_params(games: pd.DataFrame, *, train_end_year: int, eval_start_year: int,
                   use_optuna: bool = False, n_trials: int = 150,
                   cache_dir=CACHE_DIR) -> dict:
    """Re-tune Elo on games up to ``train_end_year`` (scoring the held-out
    ``eval_start_year``+ window) and write a *versioned params artifact* with the
    tuned params, held-out metrics and provenance (git sha, fit date, data
    snapshot) — so config defaults stop being hand-edited (round-2 §6.2). Returns
    the artifact dict and writes ``elo_params.json``."""
    train = games[games["year"] <= train_end_year]
    if use_optuna:
        study = optuna_search_elo(train, n_trials=n_trials, eval_start_year=eval_start_year)
        params = {k: float(v) for k, v in study.best_params.items()}
    else:
        grid = grid_search_elo(train, DEFAULT_ELO_GRID, eval_start_year=eval_start_year)
        params = {k: float(grid.iloc[0][k]) for k in DEFAULT_ELO_GRID}

    metrics = elo_objective(games, eval_start_year=eval_start_year, **params)
    artifact = {
        "params": params,
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        "search": "optuna" if use_optuna else "grid",
        "train_end_year": int(train_end_year),
        "eval_start_year": int(eval_start_year),
        "fitted_through": int(games["year"].max()),
        "n_games": int(len(games)),
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{ELO_PARAMS_ARTIFACT}.json").write_text(json.dumps(artifact, indent=2))
    return artifact


def load_fitted_elo_params(cache_dir=CACHE_DIR) -> dict:
    """The tuned Elo params from the latest ``fit`` artifact, or ``{}`` if none
    (callers then use the config defaults). Opt-in: only a deliberate ``fit``
    run writes the artifact, so auto-overfit isn't silently adopted."""
    path = cache_dir / f"{ELO_PARAMS_ARTIFACT}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("params", {})
    except (json.JSONDecodeError, OSError):
        return {}

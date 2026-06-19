"""
Hierarchical player priors + role/minutes adjustments (plan §3.1, §3.2).

§3.1 — empirical-Bayes shrinkage. ``afl_bot.models.props.player_rate_profile``
gives a raw EWMA mean/share that is noisy for short histories. This module
shrinks those toward a **role prior** (the average for the player's inferred
position group), with the weight growing in games played: a 2-game debutant
sits near the prior, a long-history player near his own number. The same idea
partially pools the Negative-Binomial dispersion ``r`` by role instead of
falling back to one league-wide value.

§3.2 — role & minutes adjustments. Expected counts scale with projected /
historical time-on-ground, and a jump in a player's centre-bounce attendance
(CBA, available in the DFS Australia current-season data) flags a midfield
role change that lifts disposals. Ruck-vs-ruck hitout matchups and opponent
tagger flags are noted in the plan but need data we don't yet carry, so they
are intentionally left out here.

Everything degrades gracefully: missing columns (e.g. a synthetic log without
TOG/CBA/hitouts) simply yield neutral multipliers / a "general" role.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afl_bot.config import (
    CBA_MULT_BOUNDS,
    CBA_ROLE_SENSITIVITY,
    PLAYER_FORM_WINDOW,
    PROP_DISPERSION_PRIOR_STRENGTH,
    PROP_EWMA_HALFLIFE,
    PROP_MIN_DISPERSION,
    PROP_PRIOR_STRENGTH,
    ROLE_FORWARD_GOALS_MIN,
    ROLE_MID_DISPOSALS_MIN,
    ROLE_RUCK_HITOUTS_MIN,
    TOG_MULT_BOUNDS,
)

ROLES = ("ruck", "forward", "midfielder", "general")
GLOBAL_ROLE = "_global"

# Real AFL positional codes (Fryzigg `player_position` / DFS `startingPosition`)
# -> the coarse role scheme. INT/SUB/EMERG carry no positional signal, so those
# rows fall back to box-score inference (round-2 §5.3).
POSITION_TO_ROLE = {
    "RK": "ruck",
    "R": "midfielder", "RR": "midfielder", "C": "midfielder", "WL": "midfielder", "WR": "midfielder",
    "FF": "forward", "CHF": "forward", "FPL": "forward", "FPR": "forward",
    "HFFL": "forward", "HFFR": "forward",
    "FB": "general", "CHB": "general", "BPL": "general", "BPR": "general",
    "HBFL": "general", "HBFR": "general",
}


# --------------------------------------------------------------------------- #
# §3.1 Empirical-Bayes shrinkage toward role priors
# --------------------------------------------------------------------------- #
def shrink(raw: float, n_games: float, prior: float, strength: float = PROP_PRIOR_STRENGTH) -> float:
    """Empirical-Bayes posterior mean: ``(n*raw + strength*prior)/(n+strength)``.

    ``strength`` is a pseudo-game count -- a player with ``n == strength`` games
    is weighted 50/50 with the prior. Falls back to whichever of ``raw`` /
    ``prior`` is finite when the other is not.
    """
    raw_ok = raw == raw and np.isfinite(raw)
    prior_ok = prior == prior and np.isfinite(prior)
    if not raw_ok:
        return float(prior) if prior_ok else float("nan")
    if not prior_ok:
        return float(raw)
    n = max(float(n_games), 0.0)
    return float((n * raw + strength * prior) / (n + strength))


def classify_roles(
    log: pd.DataFrame,
    ruck_hitouts_min: float = ROLE_RUCK_HITOUTS_MIN,
    forward_goals_min: float = ROLE_FORWARD_GOALS_MIN,
    mid_disposals_min: float = ROLE_MID_DISPOSALS_MIN,
) -> dict[str, str]:
    """Assign each player a coarse role. Prefer the REAL position label
    (``position`` column, round-2 §5.3): a player's modal non-bench position
    mapped via ``POSITION_TO_ROLE``. Fall back to box-score averages — ruck
    (hitouts) -> forward (goals) -> midfielder (disposals) -> general — for
    players with no usable label (or when the column is absent). Robust to
    missing columns (e.g. no ``hitouts`` -> no rucks by inference)."""
    if log.empty:
        return {}

    grouped = log.groupby("player")
    disposals = grouped["disposals"].mean() if "disposals" in log.columns else pd.Series(dtype=float)
    goals = grouped["goals"].mean() if "goals" in log.columns else None
    hitouts = grouped["hitouts"].mean() if "hitouts" in log.columns else None
    role_by_position = _roles_from_positions(log)

    roles: dict[str, str] = {}
    for player in disposals.index:
        labelled = role_by_position.get(player)
        if labelled is not None:
            roles[player] = labelled
        elif hitouts is not None and hitouts.get(player, 0.0) >= ruck_hitouts_min:
            roles[player] = "ruck"
        elif goals is not None and goals.get(player, 0.0) >= forward_goals_min:
            roles[player] = "forward"
        elif disposals.get(player, 0.0) >= mid_disposals_min:
            roles[player] = "midfielder"
        else:
            roles[player] = "general"
    return roles


def _roles_from_positions(log: pd.DataFrame) -> dict[str, str]:
    """Per-player role from the modal real position label, ignoring bench codes
    (INT/SUB/EMERG) that carry no positional signal. Empty if no ``position``
    column / no mappable labels."""
    if "position" not in log.columns:
        return {}
    pos = log[["player", "position"]].copy()
    pos["role"] = pos["position"].map(POSITION_TO_ROLE)
    pos = pos.dropna(subset=["role"])
    if pos.empty:
        return {}
    # most common mapped role per player across their games
    modal = pos.groupby("player")["role"].agg(lambda s: s.mode().iloc[0])
    return modal.to_dict()


def _player_means_and_shares(log: pd.DataFrame, stat: str,
                             window: int = PLAYER_FORM_WINDOW) -> tuple[pd.Series, pd.Series]:
    """Per-player mean of ``stat`` and per-player mean usage share, windowed to
    each player's last ``window`` games so old-era data doesn't bias role priors."""
    col = f"team_{stat}"
    work = log
    if col not in log.columns:
        work = log.copy()
        work[col] = work.groupby(["year", "round", "team"])[stat].transform("sum")
    work = (work.sort_values(["year", "round", "unixtime"])
                .groupby("player", group_keys=False)
                .tail(window))
    player_mean = work.groupby("player")[stat].mean()
    share = (work[stat] / work[col].replace(0, np.nan))
    player_share = share.groupby(work["player"]).mean()
    return player_mean, player_share


def role_rate_priors(log: pd.DataFrame, stat: str, roles: dict[str, str]) -> dict[str, dict[str, float]]:
    """Per-role prior mean & usage share for ``stat`` -- the average across the
    players assigned to each role. Includes a ``_global`` entry as the fallback
    for roles with no members."""
    if log.empty:
        return {GLOBAL_ROLE: {"mean_prior": float("nan"), "share_prior": float("nan")}}

    player_mean, player_share = _player_means_and_shares(log, stat)
    role_series = pd.Series(roles)

    priors: dict[str, dict[str, float]] = {
        GLOBAL_ROLE: {
            "mean_prior": float(player_mean.mean()),
            "share_prior": float(player_share.mean()),
        }
    }
    for role in set(roles.values()):
        members = role_series.index[role_series == role]
        pm = player_mean.reindex(members).dropna()
        ps = player_share.reindex(members).dropna()
        priors[role] = {
            "mean_prior": float(pm.mean()) if len(pm) else priors[GLOBAL_ROLE]["mean_prior"],
            "share_prior": float(ps.mean()) if len(ps) else priors[GLOBAL_ROLE]["share_prior"],
        }
    return priors


def estimate_dispersion_hierarchical(
    log: pd.DataFrame, stat: str, roles: dict[str, str],
    min_games: int = 6, strength: float = PROP_DISPERSION_PRIOR_STRENGTH,
    window: int = PLAYER_FORM_WINDOW,
) -> dict[str, float]:
    """Per-player NB dispersion ``r`` (method of moments) partially pooled
    toward a role prior (plan §3.1) instead of one league fallback. Windowed
    to each player's last ``window`` games so old-era variance doesn't dominate.

    Each player's own ``r`` (where they have enough games and are overdispersed)
    is shrunk toward the mean ``r`` of their role; players without a usable own
    estimate take the role prior outright. Everything is floored at
    ``PROP_MIN_DISPERSION``.
    """
    recent = (
        log.sort_values(["year", "round", "unixtime"])
           .groupby("player", group_keys=False)
           .tail(window)
    )
    grouped = recent.groupby("player")[stat].agg(["mean", "var", "count"])

    raw_r: dict[str, float] = {}
    for player, row in grouped.iterrows():
        if row["count"] >= min_games and row["var"] > row["mean"] > 0:
            raw_r[player] = row["mean"] ** 2 / (row["var"] - row["mean"])

    league_prior = float(np.mean(list(raw_r.values()))) if raw_r else PROP_MIN_DISPERSION
    role_prior: dict[str, float] = {}
    for role in set(roles.values()):
        vals = [raw_r[p] for p in raw_r if roles.get(p) == role]
        role_prior[role] = float(np.mean(vals)) if vals else league_prior

    out: dict[str, float] = {}
    for player, row in grouped.iterrows():
        prior = role_prior.get(roles.get(player, GLOBAL_ROLE), league_prior)
        if player in raw_r:
            r = shrink(raw_r[player], row["count"], prior, strength)
        else:
            r = prior
        out[player] = max(PROP_MIN_DISPERSION, float(r))
    return out


# --------------------------------------------------------------------------- #
# §3.2 Role & minutes adjustments
# --------------------------------------------------------------------------- #
def _recent_baseline(
    log: pd.DataFrame, player: str, col: str, recent_games: int, halflife: float,
    window: int | None = None,
) -> tuple[float, float]:
    """(recent mean over last ``recent_games``, EWMA baseline) of ``col`` for a
    player. If ``window`` is given, rows are first windowed to the last ``window``
    games so the baseline EWMA matches the form window. Returns (nan, nan) if the
    column is absent or has no values."""
    if col not in log.columns:
        return float("nan"), float("nan")
    rows = log[log["player"] == player].sort_values(["year", "round", "unixtime"])
    if window is not None:
        rows = rows.tail(window)
    vals = rows[col].dropna()
    if vals.empty:
        return float("nan"), float("nan")
    baseline = float(vals.ewm(halflife=halflife, adjust=True).mean().iloc[-1])
    recent = float(vals.tail(recent_games).mean())
    return recent, baseline


def player_tog(log: pd.DataFrame, player: str, recent_games: int = 4,
               halflife: float = PROP_EWMA_HALFLIFE,
               window: int = PLAYER_FORM_WINDOW) -> tuple[float, float]:
    """(recent TOG%, baseline TOG%) for a player. The baseline EWMA uses the
    last ``window`` games so it matches the form window denominator. Recent form
    (last ``recent_games``) is used when no lineup TOG override is supplied."""
    return _recent_baseline(log, player, "time_on_ground_percentage",
                            recent_games, halflife, window)


def player_cba(log: pd.DataFrame, player: str, recent_games: int = 4,
               halflife: float = PROP_EWMA_HALFLIFE,
               window: int = PLAYER_FORM_WINDOW) -> tuple[float, float]:
    """(recent CBA/game, baseline CBA/game) for a player."""
    return _recent_baseline(log, player, "centre_bounce_attendances",
                            recent_games, halflife, window)


def tog_multiplier(projected_tog: float, historical_tog: float,
                   bounds: tuple[float, float] = TOG_MULT_BOUNDS) -> float:
    """Scale expected counts by projected/historical time on ground (plan §3.2),
    clipped to ``bounds``. Neutral (1.0) when either TOG is missing."""
    if not (np.isfinite(projected_tog) and np.isfinite(historical_tog)) or historical_tog <= 0:
        return 1.0
    return float(np.clip(projected_tog / historical_tog, *bounds))


def cba_role_multiplier(cba_recent: float, cba_baseline: float,
                        sensitivity: float = CBA_ROLE_SENSITIVITY,
                        bounds: tuple[float, float] = CBA_MULT_BOUNDS) -> float:
    """Disposal multiplier for a centre-bounce-attendance role change (plan
    §3.2): each extra CBA/game vs the player's baseline lifts disposals by
    ``sensitivity``, clipped to ``bounds``. Neutral (1.0) when CBA is missing."""
    if not (np.isfinite(cba_recent) and np.isfinite(cba_baseline)):
        return 1.0
    return float(np.clip(1.0 + sensitivity * (cba_recent - cba_baseline), *bounds))

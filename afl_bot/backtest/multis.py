"""
Walk-forward same-game-multi (SGM) backtest -- model-upgrade audit Phase 1.1.

The thing actually bet is never the props-as-singles or the H2H alone; it's
the **3-leg joint probability** of the multi ladder `round_report` builds.
Props are calibrated as singles and H2H is calibrated, but `corr_gain`
(`afl_bot.build.multi.joint_prob_from_masks` vs the naive product) is
asserted from the sim, never scored against history. This module replays,
round by round, the exact match-construction + ladder-selection logic
`round_report` uses live (`afl_bot.build.report.search_match_sgms` over
`LegCandidate`s with sim masks), but fed only data strictly before the round
being predicted -- then grades the selected rungs against what actually
happened.

Leak-avoidance is inherited, not reinvented: every helper this module calls
(`team_scoring_profiles`, `EloRatings.fit` via `build_ratings_from_history`,
`_select_players`, `_team_player_samples`, ...) is the same code
`round_report` already trusts; here the caller simply truncates `games` /
`player_log` to rows strictly before `(eval_year, round_no)` on every
iteration before handing them through, exactly as `round_report` does with
its (already pre-round) `history` and `player_log`.

Simplifications vs the live path (kept honest, not hidden -- see
MODEL-UPGRADE-INSTRUCTIONS.md Phase 1 acceptance):
  * No lineup / odds book -- mirrors the no-confirmed-lineup, no-market-odds
    default (every candidate leg prices off the sim and is "confirmed").
  * No wet-weather flag -- reliable per-fixture historical rainfall isn't
    available, so every match sims dry.

Phase 2.5 added an optional calibration-ON mode: pass `prop_calibrators`
(e.g. from `afl_bot.backtest.props.load_or_fit_prop_calibrators`) and every
prediction also gets a `calibrated_joint_prob` column, so a raw vs
calibrated reliability curve can be compared side by side. Note what
calibration *can't* touch here: `search_match_sgms`'s `joint_prob` (the rung
actually picked, and what `round_report` displays) is computed from the
per-iteration sim MASKS, not from `fair_prob` -- calibration rescales a
probability estimate, it can't rewrite which simulated iterations "hit". So
`calibrated_joint_prob` is built by keeping the sim's correlation lift
(`corr_gain = joint_prob - naive_product`, a function of the copula/pace/
Dirichlet structure that calibration doesn't touch) and rebasing the
marginal naive product onto calibrated per-leg probabilities:
`calibrated_joint_prob = clip(calibrated_naive_product + corr_gain, 0, 1)`.
"""

from __future__ import annotations

import pandas as pd

from afl_bot.backtest.props import apply_prop_calibration
from afl_bot.backtest.walkforward import brier_score, calibration_curve, log_loss
from afl_bot.build.multi import LegCandidate
from afl_bot.build.report import search_match_sgms
from afl_bot.config import (
    LEG_PROB_MAX,
    LEG_PROB_MIN,
    PACE_SIGMA,
    PROP_RECENT_SEASONS,
    SCORE_SHOT_CORRELATION,
    SHARE_CONCENTRATION,
    SHOT_DISPERSION,
    SIM_ITERATIONS,
    TEAM_STAT_DISPERSION,
)
from afl_bot.models.pace import PACE_STATS, league_stat_totals, team_stat_total_profiles
from afl_bot.models.priors import classify_roles, role_rate_priors
from afl_bot.models.scoring import (
    expected_total,
    team_scoring_profiles,
    team_shot_accuracy_profiles,
    venue_scoring_factors,
)
from afl_bot.pricing.edge import fair_odds, prob_event
from afl_bot.ratings.elo import build_ratings_from_history
from afl_bot.ratings.hga import attach_hga, fit_team_hga
from afl_bot.sim.engine import Team, draw_pace, make_rng, simulate_match


def _build_match_legs(history: pd.DataFrame, player_log: pd.DataFrame, fixtures: pd.DataFrame,
                      year: int, round_no: int, n_sims: int, rng,
                      correlation_params: dict | None = None,
                      prop_calibrators: dict | None = None) -> tuple[dict, dict]:
    """Per-match `LegCandidate`s (h2h + player props) for this round's
    fixtures, built from `history`/`player_log` the caller has already
    truncated to strictly-before-this-round. Mirrors `round_report`'s
    no-odds/no-lineup default path. Returns ``(legs_by_match, leg_info_by_match)``
    where `leg_info_by_match[match_id][leg_name]` carries the `(market,
    subject, line, calibrated_prob)` needed to grade that leg against the
    actual result later (`LegCandidate` itself doesn't store the line).

    ``correlation_params`` optionally overrides any of `SCORE_SHOT_CORRELATION`
    / `SHOT_DISPERSION` / `PACE_SIGMA` / `TEAM_STAT_DISPERSION` /
    `SHARE_CONCENTRATION` (e.g. from `afl_bot.backtest.correlations
    .load_fitted_correlation_params`) -- model-upgrade audit Phase 2's
    re-run-the-Phase-1-backtest acceptance check.

    ``prop_calibrators`` (per-``(stat, line)`` with a pooled per-stat
    fallback, e.g. from `afl_bot.backtest.props.load_or_fit_prop_calibrators`,
    applied via `apply_prop_calibration`) optionally calibrates each
    player-prop leg's stored probability (h2h legs pass through uncalibrated
    -- the prop calibrators don't cover H2H) -- Phase 2.5's calibration-ON
    mode."""
    from afl_bot.cli import PLAYERS_PER_TEAM_SAMPLE, _fixture_hga, _select_players, _team_player_samples
    from afl_bot.config import PROP_LINES

    cp = correlation_params or {}
    score_correlation = cp.get("SCORE_SHOT_CORRELATION", SCORE_SHOT_CORRELATION)
    shot_dispersion = cp.get("SHOT_DISPERSION", SHOT_DISPERSION)
    pace_sigma = cp.get("PACE_SIGMA", PACE_SIGMA)
    team_stat_dispersion = cp.get("TEAM_STAT_DISPERSION", TEAM_STAT_DISPERSION)
    share_concentration = cp.get("SHARE_CONCENTRATION", SHARE_CONCENTRATION)

    team_hga = fit_team_hga(history)
    elo, _ = build_ratings_from_history(attach_hga(history, team_hga))
    scoring_profiles = team_scoring_profiles(history)
    accuracy_profiles = team_shot_accuracy_profiles(history)
    venue_factors = venue_scoring_factors(history)
    roles = classify_roles(player_log)
    recent_log = player_log[player_log["year"] > year - PROP_RECENT_SEASONS]
    if recent_log.empty:
        recent_log = player_log
    rate_priors = {stat: role_rate_priors(recent_log, stat, roles) for stat in PROP_LINES}
    team_stat_profiles = team_stat_total_profiles(player_log)
    league_totals = league_stat_totals(player_log)
    volume_stats = [s for s in PROP_LINES if s in PACE_STATS]

    legs_by_match: dict[str, list[LegCandidate]] = {}
    leg_info_by_match: dict[str, dict[str, dict]] = {}

    for _, fx in fixtures.iterrows():
        home_name, away_name = fx["hteam"], fx["ateam"]
        match_id = f"{year}_r{round_no}_{home_name}_v_{away_name}"
        venue = fx["venue"]

        mu_margin = elo.expected_margin(home_name, away_name,
                                        hga=_fixture_hga(home_name, away_name, venue, team_hga))
        hp = scoring_profiles.get(home_name, {"off_rate": 90.0, "def_rate": 90.0})
        ap = scoring_profiles.get(away_name, {"off_rate": 90.0, "def_rate": 90.0})
        mu_total = expected_total(hp["off_rate"], hp["def_rate"], ap["off_rate"], ap["def_rate"],
                                  venue_factor=venue_factors.get(venue, 1.0))
        ha = accuracy_profiles.get(home_name, float("nan"))
        aa = accuracy_profiles.get(away_name, float("nan"))
        match = simulate_match(Team(home_name, True), Team(away_name), mu_margin, mu_total,
                               ha, aa, n_sims, rng,
                               score_correlation=score_correlation, shot_dispersion=shot_dispersion)

        p_home = prob_event(match["home_win"] > 0)
        p_away = prob_event(match["away_win"] > 0)
        match_legs = [
            LegCandidate(f"{home_name} to win", match_id, "h2h", home_name, p_home,
                        fair_odds(p_home), mask=(match["home_win"] > 0)),
            LegCandidate(f"{away_name} to win", match_id, "h2h", away_name, p_away,
                        fair_odds(p_away), mask=(match["away_win"] > 0)),
        ]
        leg_info = {
            f"{home_name} to win": {"market": "h2h", "subject": home_name, "line": "",
                                    "calibrated_prob": p_home},
            f"{away_name} to win": {"market": "h2h", "subject": away_name, "line": "",
                                    "calibrated_prob": p_away},
        }

        pace = draw_pace(n_sims, rng, pace_sigma=pace_sigma)
        for team, is_home_team in ((home_name, True), (away_name, False)):
            opponent = away_name if is_home_team else home_name
            usage = _select_players(player_log, team, year, PLAYERS_PER_TEAM_SAMPLE)
            samples = _team_player_samples(
                usage, team, opponent, is_home_team, match, pace, player_log, roles,
                rate_priors, team_stat_profiles, league_totals, volume_stats,
                False, False, n_sims, rng,
                team_stat_dispersion=team_stat_dispersion, share_concentration=share_concentration)

            for player_name, stats in samples.items():
                for stat, lines in PROP_LINES.items():
                    arr = stats.get(stat)
                    if arr is None:
                        continue
                    for line in lines:
                        mask = arr >= line
                        prob = prob_event(mask)
                        if not (LEG_PROB_MIN < prob < LEG_PROB_MAX):
                            continue
                        calibrated_prob = apply_prop_calibration(prop_calibrators or {}, stat, line, prob)
                        name = f"{player_name} {line}+ {stat}"
                        match_legs.append(LegCandidate(
                            name, match_id, f"player_{stat}", player_name, prob,
                            fair_odds(prob), mask=mask))
                        leg_info[name] = {"market": f"player_{stat}", "subject": player_name, "line": line,
                                          "calibrated_prob": calibrated_prob}

        legs_by_match[match_id] = match_legs
        leg_info_by_match[match_id] = leg_info

    return legs_by_match, leg_info_by_match


def _actual_results(fixtures: pd.DataFrame, player_round: pd.DataFrame) -> tuple[dict, dict]:
    """``(h2h_actual, player_stat_actual)`` for one completed round --
    ``h2h_actual[team] -> 1/0`` win flag, ``player_stat_actual[(player, stat)]
    -> value``. Same lookup `afl_bot.cli.grade_round` uses, kept local here so
    this module doesn't depend on `cli`'s grading internals."""
    h2h_actual: dict[str, int] = {}
    for _, g in fixtures.iterrows():
        h2h_actual[g["hteam"]] = int(g["hscore"] > g["ascore"])
        h2h_actual[g["ateam"]] = int(g["ascore"] > g["hscore"])

    player_stat: dict[tuple[str, str], float] = {}
    if not player_round.empty:
        for _, r in player_round.iterrows():
            for stat in ("disposals", "goals", "marks", "tackles"):
                if stat in r:
                    player_stat[(r["player"], stat)] = r[stat]
    return h2h_actual, player_stat


def _leg_actual_hit(info: dict, h2h_actual: dict, player_stat: dict) -> int | None:
    market, subject, line = info["market"], info["subject"], info["line"]
    if market == "h2h":
        return h2h_actual.get(subject)
    stat = market.split("_", 1)[1]
    val = player_stat.get((subject, stat))
    return int(val >= float(line)) if val is not None else None


def _fetch_actual_player_log(year: int, games_year: pd.DataFrame) -> pd.DataFrame:
    """Real per-player box scores for `year` (player/round/disposals/goals/
    marks/tackles), Fryzigg first (past seasons) then DFS Australia (current
    season) -- the same two-source fallback `cli.grade_round` uses."""
    try:
        from afl_bot.data.fryzigg import fetch_fryzigg_player_stats
        raw = fetch_fryzigg_player_stats()
        raw = raw.assign(_year=pd.to_datetime(raw["match_date"]).dt.year,
                         _player=(raw["player_first_name"].str.strip() + " "
                                  + raw["player_last_name"].str.strip()))
        rows = raw[raw["_year"] == year].rename(columns={"_player": "player", "match_round": "round"})
        if not rows.empty:
            rows = rows.assign(round=rows["round"].astype(str))
            return rows
    except Exception:  # noqa: BLE001
        pass
    try:
        from afl_bot.data.dfs_australia import fetch_player_stats, to_player_log
        dfs = to_player_log(fetch_player_stats(), games_year)
        dfs = dfs[dfs["year"] == year].assign(round=lambda d: d["round"].astype(str))
        return dfs
    except Exception:  # noqa: BLE001
        pass
    return pd.DataFrame(columns=["player", "round", "disposals", "goals", "marks", "tackles"])


def _truncate_before_round(games: pd.DataFrame, player_log: pd.DataFrame,
                           eval_year: int, round_no) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """``(history, log)`` truncated to strictly before ``(eval_year, round_no)``
    -- the anti-leakage contract every walk-forward loop in this module shares.
    Returns ``(None, None)`` when either truncated frame would be empty (not
    enough history yet, e.g. the first eval round of the dataset)."""
    before_games = (games["year"] < eval_year) | (
        (games["year"] == eval_year) & (games["round"] < round_no))
    history = games[before_games]
    if history.empty:
        return None, None
    before_log = (player_log["year"] < eval_year) | (
        (player_log["year"] == eval_year) & (player_log["round"] < round_no))
    log = player_log[before_log]
    if log.empty:
        return None, None
    return history, log


def walk_forward_multi_predictions(
    games: pd.DataFrame, player_log: pd.DataFrame, *, eval_year: int,
    rounds: list[int] | None = None, n_sims: int = SIM_ITERATIONS, seed: int | None = None,
    correlation_params: dict | None = None, prop_calibrators: dict | None = None,
) -> pd.DataFrame:
    """One row per (year, round, match_id, rung) the model would have placed
    on its 3-leg ladder for every round in `rounds` (default: every round of
    `eval_year` present in `games`), with the predicted joint sim probability
    vs whether all 3 legs actually hit.

    `games` should span enough prior seasons for Elo/scoring-profile history
    (e.g. several years before `eval_year`); `player_log` likewise needs
    prior-season rows for EWMA form. Both are truncated internally to
    strictly-before-this-round on every iteration -- no later round or season
    can leak into a prediction.

    ``correlation_params`` optionally overrides the config correlation/
    dispersion constants for every round simulated (see `_build_match_legs`)
    -- pass `afl_bot.backtest.correlations.load_fitted_correlation_params()`
    output to re-run this backtest with fitted-from-history values instead
    of the config defaults (model-upgrade audit Phase 2.2 acceptance check).

    ``prop_calibrators`` optionally adds a `calibrated_joint_prob` column
    alongside the raw `joint_prob` (Phase 2.5's calibration-ON mode -- see
    module docstring for how it's built from the raw `corr_gain`). Fit these
    ONCE on data strictly before `eval_year` and reuse across every round in
    this call -- per-round refitting isn't done here (expensive, and not
    what `round_report`'s own calibrator loading does either).
    """
    games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)
    year_games = games[games["year"] == eval_year]
    if rounds is None:
        rounds = sorted(year_games["round"].unique())

    rng = make_rng(seed) if seed is not None else make_rng()
    actual_log = _fetch_actual_player_log(eval_year, year_games)

    rows = []
    for round_no in rounds:
        fixtures = year_games[year_games["round"] == round_no]
        if fixtures.empty:
            continue
        history, log = _truncate_before_round(games, player_log, eval_year, round_no)
        if history is None:
            continue

        legs_by_match, leg_info_by_match = _build_match_legs(
            history, log, fixtures, eval_year, round_no, n_sims, rng,
            correlation_params=correlation_params, prop_calibrators=prop_calibrators)
        player_round = actual_log[actual_log["round"] == str(round_no)] if not actual_log.empty else actual_log
        h2h_actual, player_stat = _actual_results(fixtures, player_round)

        for match_id, match_legs in legs_by_match.items():
            if len(match_legs) < 3:
                continue
            leg_info = leg_info_by_match[match_id]
            for rung in search_match_sgms(match_legs):
                hits = []
                for leg_name in rung["legs"]:
                    hit = _leg_actual_hit(leg_info[leg_name], h2h_actual, player_stat)
                    if hit is None:
                        hits = None
                        break
                    hits.append(hit)
                if hits is None:
                    continue
                row = {
                    "year": eval_year, "round": round_no, "match_id": match_id,
                    "legs": " + ".join(rung["legs"]), "joint_prob": rung["joint_prob"],
                    "naive_product": rung["naive_product"], "corr_gain": rung["corr_gain"],
                    "fair_odds": rung["fair_odds"], "all_hit": int(all(hits)),
                }
                if prop_calibrators is not None:
                    calibrated_naive = 1.0
                    for leg_name in rung["legs"]:
                        calibrated_naive *= leg_info[leg_name]["calibrated_prob"]
                    row["calibrated_naive_product"] = calibrated_naive
                    row["calibrated_joint_prob"] = float(
                        min(max(calibrated_naive + rung["corr_gain"], 0.0), 1.0))
                rows.append(row)

    return pd.DataFrame(rows)


def walk_forward_sim_prop_predictions(
    games: pd.DataFrame, player_log: pd.DataFrame, *, eval_year: int,
    rounds: list[int] | None = None, n_sims: int = SIM_ITERATIONS, seed: int | None = None,
    correlation_params: dict | None = None,
) -> pd.DataFrame:
    """One row per (year, round, player, stat, line) leg `_build_match_legs`
    would have priced, with the predicted RAW sim probability vs actual hit
    (model-upgrade audit Phase 3.1 -- "calibrate against the real sim
    output"). Same shape as `afl_bot.backtest.props.walk_forward_prop_predictions`
    (year/round/player/stat/line/prob/actual), so `fit_prop_calibrators` works
    unchanged on either source, but the probabilities come from the FULL live
    multiplier stack (TOG/CBA/matchup, shared pace draw, Dirichlet share
    allocation, scoreline correlation via `_team_player_samples`/`simulate_match`)
    instead of `walk_forward_prop_predictions`'s simplified shrunk-EWMA NB
    marginal. Don't build a second sim path: this reuses `_build_match_legs`,
    the same per-round leg construction `walk_forward_multi_predictions`
    already trusts (anti-leakage by the same truncation contract), just
    without the 3-leg combo search -- every gated candidate leg becomes one
    row here, not only the ones that end up in a selected SGM rung.

    Legs are gated to ``LEG_PROB_MIN``/``LEG_PROB_MAX`` inside
    `_build_match_legs`, same as live pricing -- this fits calibrators on
    exactly the probability window that's ever shown to a bettor, not the
    full [0,1] range. ``prop_calibrators=None`` is fixed (passing one in here
    would calibrate the very data used to fit the next one)."""
    games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)
    year_games = games[games["year"] == eval_year]
    if rounds is None:
        rounds = sorted(year_games["round"].unique())

    rng = make_rng(seed) if seed is not None else make_rng()
    actual_log = _fetch_actual_player_log(eval_year, year_games)

    rows = []
    for round_no in rounds:
        fixtures = year_games[year_games["round"] == round_no]
        if fixtures.empty:
            continue
        history, log = _truncate_before_round(games, player_log, eval_year, round_no)
        if history is None:
            continue

        legs_by_match, leg_info_by_match = _build_match_legs(
            history, log, fixtures, eval_year, round_no, n_sims, rng,
            correlation_params=correlation_params)
        player_round = actual_log[actual_log["round"] == str(round_no)] if not actual_log.empty else actual_log
        _, player_stat = _actual_results(fixtures, player_round)

        for match_id, match_legs in legs_by_match.items():
            leg_info = leg_info_by_match[match_id]
            for leg in match_legs:
                info = leg_info[leg.name]
                if info["market"] == "h2h":
                    continue   # H2H is calibrated/blended separately (ensemble.py)
                val = player_stat.get((info["subject"], info["market"].split("_", 1)[1]))
                if val is None:
                    continue
                rows.append({
                    "year": eval_year, "round": round_no, "player": info["subject"],
                    "stat": info["market"].split("_", 1)[1], "line": info["line"],
                    "prob": leg.fair_prob, "actual": int(val >= float(info["line"])),
                })

    return pd.DataFrame(rows, columns=["year", "round", "player", "stat", "line", "prob", "actual"])


def multi_calibration_report(preds: pd.DataFrame, column: str = "joint_prob") -> dict:
    """Log loss / Brier / mean predicted vs actual hit rate for the multi
    ladder's joint probabilities -- the headline number for Phase 1's
    acceptance check (is the $5.00 rung's ~20% actually landing near 20%?).
    Pass ``column="calibrated_joint_prob"`` for Phase 2.5's calibration-ON
    comparison (requires `walk_forward_multi_predictions(prop_calibrators=...)`)."""
    if preds.empty or column not in preds:
        return {"n": 0, "log_loss": float("nan"), "brier": float("nan"),
                "mean_pred": float("nan"), "hit_rate": float("nan")}
    p = preds[column].to_numpy()
    a = preds["all_hit"].to_numpy(dtype=float)
    return {
        "n": len(preds), "log_loss": log_loss(p, a), "brier": brier_score(p, a),
        "mean_pred": float(p.mean()), "hit_rate": float(a.mean()),
    }


def multi_reliability_curve(preds: pd.DataFrame, n_bins: int = 5, column: str = "joint_prob") -> pd.DataFrame:
    """Reliability table for the multi ladder (bucket predicted multi prob vs
    actual hit rate). Default `n_bins=5`, coarser than the H2H/prop curves'
    default 10, since a few rounds of 3-rung-per-match ladders is a much
    smaller sample than per-game or per-player-prop predictions. Pass
    ``column="calibrated_joint_prob"`` for the calibration-ON curve."""
    if preds.empty or column not in preds:
        return pd.DataFrame(columns=["bucket", "mean_pred", "actual_rate", "n"])
    return calibration_curve(preds[column].to_numpy(), preds["all_hit"].to_numpy(dtype=float),
                             n_bins=n_bins)

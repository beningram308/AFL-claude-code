"""
``run-round`` CLI (plan §7, stage 10).

Pulls the next round's fixtures, builds Elo + scoring profiles from history,
runs the Monte Carlo sim for each match and a sample of player props, prices
everything, and prints the candidate ANCHOR/VALUE legs plus (if a market-odds
file is supplied) the assembled multis.

Usage:
    python -m afl_bot.cli run-round --year 2026
    python -m afl_bot.cli run-round --year 2026 --round 14 --odds odds.json

The ``--odds`` JSON maps leg names (as printed in the "no odds" run) to market
decimal odds, e.g.:
    {"Brisbane Lions to win": 1.45, "Hawthorn Player 3 15+ disposals": 1.30}

Player props use real per-player box scores from DFS Australia
(afl_bot.data.dfs_australia), falling back to a synthetic player log
(afl_bot.data.player_stats) if that source is unavailable or
--synthetic-props is passed.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from afl_bot.io_utils import atomic_write_text

from afl_bot.backtest.ensemble import assemble_signals, fit_market_blend, squiggle_consensus
from afl_bot.backtest.props import apply_prop_calibration, load_or_fit_prop_calibrators
from afl_bot.backtest.tuning import fit_elo_params, load_fitted_elo_params
from afl_bot.build.multi import (
    LegCandidate,
    build_anchor_multis,
    build_promo_multi,
    joint_prob_from_masks,
)
from afl_bot.build.report import (
    build_odds_template,
    build_pull_em_sgm,
    is_bookable_model_only_leg,
    projection_rows,
    render_markdown,
    search_market_sgms,
    search_match_sgms,
    select_ladder_lines,
    top_n_players_by_stat,
)
from afl_bot.build.staking import (
    bankroll_report,
    multi_outcome_kelly,
    recommend_units,
    simulate_bankroll,
    simulate_bankroll_joint,
    stake_bets,
)
from afl_bot.config import (
    ANCHOR_MIN_PROB,
    BANKROLL,
    BOOKABLE_TOP_N_BY_STAT,
    CACHE_DIR,
    CORR_GAIN_HAIRCUT,
    DEFAULT_BANKROLL,
    KELLY_PER_ROUND_CAP,
    LEG_PROB_MAX,
    LEG_PROB_MIN,
    MANUALLY_UNAVAILABLE,
    MC_SE_TARGET,
    MULTI_CALIBRATION_LOOKBACK,
    MULTI_MARKET_SHRINK,
    PLAYER_FORM_WINDOW,
    PROP_CALIBRATION_LOOKBACK,
    PROP_EWMA_HALFLIFE,
    PROP_KELLY_MULTIPLIER,
    PROP_LINES,
    PROP_MARKET_BLEND_WEIGHT,
    PROP_RECENT_SEASONS,
    PROMO_REFUND_CAP,
    PULL_DETECTION_PROB,
    ROOT_DIR,
    SHARE_CONCENTRATION,
    SIM_ITERATIONS,
    SUSPECT_BOOK_FAIR_RATIO,
    SUSPECT_MAX_RAW_EDGE,
    TEAM_STAT_DISPERSION,
    TOG_RETURN_DEFAULT,
    UNIT_MAX,
    UNIT_MAX_LONGSHOT,
    UNIT_SIZE,
    UNIT_STEP,
    WET_THRESHOLD_MM,
)
from afl_bot.data.odds import fetch_historical_odds
from afl_bot.data.lineups import (apply_outs, fetch_injury_list, fetch_lineup,
                                  load_lineup, load_lineup_tog, load_outs)
from afl_bot.data.live_odds import fetch_live_odds, fetch_live_props
from afl_bot.data.sportsbet_odds import fetch_sportsbet_odds
from afl_bot.data.player_stats import load_player_log
from afl_bot.data.squiggle import SquiggleClient
from afl_bot.data.stoppages import load_boundary_throwins
from afl_bot.data.venues import is_roofed, venue_info
from afl_bot.data.weather import forecast_game_conditions, forecast_game_rain
from afl_bot.models.pace import PACE_STATS, league_stat_totals, team_stat_total_profiles
from afl_bot.models.priors import (
    cba_role_multiplier,
    classify_roles,
    estimate_dispersion_hierarchical,
    player_cba,
    player_tog,
    role_rate_priors,
    shrink,
    tog_multiplier,
)
from afl_bot.models.props import (
    expected_stat_mean,
    opponent_matchup_multiplier,
    player_rate_profile,
)
from afl_bot.models.stoppages import expected_oob, simulate_boundary_throwins
from afl_bot.models.weather_effects import greasiness_factor, greasiness_multiplier
from afl_bot.models.scoring import (
    expected_total,
    team_scoring_profiles,
    team_shot_accuracy_profiles,
    venue_scoring_factors,
)
from afl_bot.pricing.edge import (
    devig_prop_leg,
    devig_proportional,
    fair_odds,
    market_anchored_prob,
    mc_standard_error,
    prob_event,
    prob_over,
)
from afl_bot.ratings.elo import build_ratings_from_history
from afl_bot.ratings.hga import INTERSTATE_PENALTY, TEAM_STATE, attach_hga, fit_team_hga, venue_state
from afl_bot.sim.engine import (
    Player,
    Team,
    allocate_player_goals,
    allocate_player_stats,
    draw_pace,
    make_rng,
    simulate_match,
    simulate_player,
    simulate_team_stat_total,
)

# Top-usage players per team to price when there's no confirmed lineup. Raised
# from 4 once the pool is gated to current-season/confirmed players so the VALUE
# search has real breadth (Fable round-2 §1.3); a supplied lineup prices all 22.
PLAYERS_PER_TEAM_SAMPLE = 10


def _history_years(target_year: int, lookback: int = 6) -> list[int]:
    return list(range(target_year - lookback, target_year + 1))


def _normalize_name(name: str) -> str:
    """Canonical key for fuzzy player-name matching: lowercase, hyphens/apostrophes stripped.
    Lets Footywire slug names (``"Luke Davies Uniacke"``) match log names
    (``"Luke Davies-Uniacke"``) without an explicit player-ID table."""
    return name.lower().replace("-", " ").replace("'", "").strip()


def _min_sims_for_anchor_se(target_se: float = MC_SE_TARGET) -> int:
    """Iterations so the tightest anchor (p = ANCHOR_MIN_PROB, where the binomial
    SE is largest among anchors) clears ``target_se`` (round-2 §8.3)."""
    p = ANCHOR_MIN_PROB
    return int(np.ceil(p * (1.0 - p) / target_se ** 2))


def _multi_anchored_prob(multi) -> float:
    """A multi's combined probability with each leg pulled toward its market
    price (round-2 §8.2) — the joint sim prob scaled by each leg's market-anchor
    ratio, so per-leg overestimates don't compound across the multi."""
    ratio = 1.0
    for leg in multi.legs:
        if leg.fair_prob > 0:
            ratio *= market_anchored_prob(leg.fair_prob, leg.market_odds, MULTI_MARKET_SHRINK) / leg.fair_prob
    return float(multi.combined_fair_prob * ratio)


def _fixture_hga(home: str, away: str, venue: str, team_hga: dict[str, float]) -> float:
    """Venue + interstate home advantage (points) for an upcoming fixture (§6.1).
    Per-team venue HGA, plus the interstate penalty when the away side travels to
    a different state (rest is used in the ratings fit, not this single margin)."""
    from afl_bot.config import ELO_HOME_ADVANTAGE
    hga = team_hga.get(home, ELO_HOME_ADVANTAGE)
    vs, aws = venue_state(venue), TEAM_STATE.get(away)
    if vs and aws and vs != aws:
        hga += INTERSTATE_PENALTY
    return hga


def _fixture_greasiness(fx, rain_mm: float | None, roofed: bool) -> float:
    """Continuous 0.0-1.0 greasiness for a fixture (Phase 1): roofed -> 0.0;
    an explicit --rain-mm override uses rain-only greasiness (no temperature);
    otherwise auto-fetch forecast conditions from Open-Meteo (best-effort,
    0.0 on any failure / outside the forecast horizon)."""
    if roofed:
        return 0.0
    if rain_mm is not None:
        return greasiness_factor(rain_mm, float("nan"), float("nan"), float("nan"), roofed=False)
    info = venue_info(fx["venue"])
    if info is None:
        return 0.0
    when = fx["localtime"] if ("localtime" in fx and pd.notna(fx.get("localtime"))) else fx.get("date")
    cond = forecast_game_conditions(info["lat"], info["lon"], when)
    return greasiness_factor(
        cond["rain_mm"], cond["temp_c"], cond["apparent_temp_c"], cond["wind_kmh"],
        roofed=False,
    )


def _select_players(player_log: pd.DataFrame, team: str, current_year: int, n: int,
                    confirmed: set[str] | None = None) -> list[str]:
    """Players to price for ``team``, ranked by disposals (Fable round-2 §1.1).

    Pools to the current season when the team has fielded a near-full list there
    (so a retired/delisted player on a big career average can't be priced),
    falling back to last season early in the year, then to all history. If a
    confirmed lineup is supplied, restrict to those players and price all of
    them; otherwise take the top ``n``.
    """
    team_rows = player_log[player_log["team"] == team]
    if team_rows.empty:
        return []

    current = team_rows[team_rows["year"] == current_year]
    if current["player"].nunique() >= 18:           # team has played a full-ish season
        pool = current
    else:
        recent = team_rows[team_rows["year"] >= current_year - 1]
        pool = recent if not recent.empty else team_rows

    ranked = pool.groupby("player")["disposals"].mean().sort_values(ascending=False)
    if confirmed:
        # Exact match first; fall back to normalized (handles hyphen/space variants
        # from Footywire slug extraction, e.g. "Luke Davies Uniacke" → "Luke Davies-Uniacke")
        exact = set(ranked.index) & confirmed
        if len(exact) < len(confirmed):
            norm_map = {_normalize_name(n): n for n in ranked.index}
            matched = set(exact)
            for c in confirmed:
                if c not in ranked.index:
                    actual = norm_map.get(_normalize_name(c))
                    if actual:
                        matched.add(actual)
        else:
            matched = exact
        ranked = ranked[ranked.index.isin(matched)]
        return ranked.index.tolist()
    return ranked.head(n).index.tolist()


def _team_player_samples(usage_players, team, opponent, is_home_team, match, pace,
                         player_log, roles, rate_priors, team_stat_profiles, league_totals,
                         volume_stats, greasiness, roofed, n_sims, rng,
                         lineup_tog: dict[str, float] | None = None,
                         team_stat_dispersion: float = TEAM_STAT_DISPERSION,
                         share_concentration: float = SHARE_CONCENTRATION):
    """Per-iteration stat samples for each priced player on one team: volume
    stats via pace -> opponent/greasy-weather-adjusted team total -> role-shrunk
    Dirichlet shares (§2.5/§3.1-3.3), goals via Multinomial on goal shares
    (§3.3). Returns ``{player: {stat: samples}}``. Shared by run-round and
    round-report so the two can't drift.

    ``greasiness`` (0.0=dry, 1.0=heavy-wet/cold) scales per-stat multipliers
    continuously (Phase 1). ``lineup_tog`` maps player names to their projected
    TOG for this match (from ``load_lineup_tog``). When supplied, it overrides
    the recent-form TOG used as the projected TOG for that player (the historical
    baseline stays the same — it is always the 40-game EWMA)."""
    player_samples: dict[str, dict[str, np.ndarray]] = {p: {} for p in usage_players}

    # Role/minutes multipliers (plan §3.2).
    # projected_tog: lineup override if supplied, else recent form (last 4 games).
    # baseline_tog: always the 40-game EWMA (denominator for the multiplier).
    tog_mult = {}
    for p in usage_players:
        recent_tog, baseline_tog = player_tog(player_log, p)
        projected_tog = lineup_tog.get(p, recent_tog) if lineup_tog else recent_tog
        tog_mult[p] = tog_multiplier(projected_tog, baseline_tog)
    cba_mult = {p: cba_role_multiplier(*player_cba(player_log, p)) for p in usage_players}

    for stat in volume_stats:
        mu_team = team_stat_profiles.get(team, {}).get(stat)
        if mu_team is None or not np.isfinite(mu_team):
            mu_team = league_totals.get(stat, float("nan"))
        if not np.isfinite(mu_team):
            continue
        mu_team *= opponent_matchup_multiplier(player_log, stat, opponent)
        mu_team *= greasiness_multiplier(stat, greasiness, roofed)
        team_total = simulate_team_stat_total(mu_team, pace, rng, dispersion=team_stat_dispersion)

        priced, shares = [], []
        for player_name in usage_players:
            profile = player_rate_profile(player_log, player_name, stat)
            role = roles.get(player_name, "general")
            prior = rate_priors[stat].get(role, rate_priors[stat]["_global"])["share_prior"]
            share = shrink(profile["share"], profile["n_games"], prior)
            if not np.isfinite(share):
                continue
            share *= tog_mult[player_name]
            if stat == "disposals":
                share *= cba_mult[player_name]
            priced.append(player_name)
            shares.append(share)
        if not priced:
            continue
        shares = np.asarray(shares, dtype=float)
        if shares.sum() > 0.95:
            shares = shares * (0.95 / shares.sum())
        alloc = allocate_player_stats(team_total, shares, rng, concentration=share_concentration)
        for player_name, row in zip(priced, alloc):
            player_samples[player_name][stat] = row

    # Goals: Multinomial on (shrunk, minutes/matchup-adjusted) goal shares.
    # No wet multiplier here — team_goals already comes from the wet-adjusted
    # match sim (lower accuracy), so re-applying it would double-count (§4.1).
    team_goals = match["home_goals"] if is_home_team else match["away_goals"]
    goal_matchup = opponent_matchup_multiplier(player_log, "goals", opponent)
    g_priced, g_shares = [], []
    for player_name in usage_players:
        profile = player_rate_profile(player_log, player_name, "goals")
        role = roles.get(player_name, "general")
        prior = rate_priors["goals"].get(role, rate_priors["goals"]["_global"])["share_prior"]
        share = shrink(profile["share"], profile["n_games"], prior)
        if not np.isfinite(share):
            continue
        share *= tog_mult[player_name] * goal_matchup
        g_priced.append(player_name)
        g_shares.append(share)
    if g_priced:
        g_shares = np.asarray(g_shares, dtype=float)
        if g_shares.sum() > 0.95:
            g_shares = g_shares * (0.95 / g_shares.sum())
        g_alloc = allocate_player_goals(team_goals, g_shares, rng)
        for player_name, row in zip(g_priced, g_alloc):
            player_samples[player_name]["goals"] = row

    return player_samples


def run_round(year: int, round_no: int | None, odds_path: str | None, n_sims: int,
              synthetic_props: bool = False, rain_mm: float | None = None,
              bankroll: float = DEFAULT_BANKROLL, lineup_path: str | None = None,
              allow_synthetic_props: bool = False) -> None:
    client = SquiggleClient()

    history = pd.concat(
        [client.get_completed_games(y) for y in _history_years(year)],
        ignore_index=True,
    )
    if history.empty:
        print("No historical data available; cannot build ratings.", file=sys.stderr)
        return

    upcoming = client.get_upcoming_games(year)
    if upcoming.empty:
        print(f"No upcoming games found for {year}.", file=sys.stderr)
        return

    if round_no is None:
        round_no = int(upcoming["round"].iloc[0])
    fixtures = upcoming[upcoming["round"] == round_no]
    if fixtures.empty:
        print(f"No upcoming games found for {year} round {round_no}.", file=sys.stderr)
        return

    # Auto-bump n_sims so an anchor's Monte-Carlo SE clears the target (§8.3).
    min_sims = _min_sims_for_anchor_se()
    if n_sims < min_sims:
        print(f"note: bumping n_sims {n_sims} -> {min_sims} for anchor SE < {MC_SE_TARGET} "
              f"(plan §8.3).", file=sys.stderr)
        n_sims = min_sims

    team_hga = fit_team_hga(history)   # per-team venue HGA (§6.1)
    elo, _ = build_ratings_from_history(attach_hga(history, team_hga), **load_fitted_elo_params())
    scoring_profiles = team_scoring_profiles(history)
    accuracy_profiles = team_shot_accuracy_profiles(history)
    venue_factors = venue_scoring_factors(history)   # per-venue scoring (§6.4)

    # Player props: real DFS Australia box scores for the current season,
    # falling back to a synthetic log if unavailable (or --synthetic-props).
    player_log, log_source = load_player_log(
        history, prefer_real=not synthetic_props, return_source=True)
    current_season = int(player_log["year"].max())

    # Synthetic-data guard (round-2 §1.4 / §P10a): with --odds we refuse to
    # price props off a silent synthetic fallback unless explicitly allowed.
    props_synthetic = log_source == "synthetic"
    skip_props = bool(odds_path) and not synthetic_props and not allow_synthetic_props and props_synthetic
    if skip_props:
        print("WARNING: player log fell back to SYNTHETIC data; refusing to price "
              "player props (pass --allow-synthetic-props to override). "
              "H2H/total/OOB markets below are unaffected.", file=sys.stderr)

    # Confirmed lineups (round-2 §1.2): players not named are excluded from
    # multis. No file -> every player stays confirmed (unchanged behaviour).
    lineup = load_lineup(lineup_path)
    lineup_tog = load_lineup_tog(lineup_path)   # per-player projected TOG overrides

    # Hierarchical priors (plan §3.1): infer coarse roles, then pool dispersion
    # and shrink player rates toward role priors.
    roles = classify_roles(player_log)
    # Dispersion pools the full 2012+ history (needs the sample); the role mean/
    # share priors use only recent seasons to avoid era bias (round-2 §5.2).
    recent_log = player_log[player_log["year"] > current_season - PROP_RECENT_SEASONS]
    if recent_log.empty:
        recent_log = player_log
    dispersion = {
        stat: estimate_dispersion_hierarchical(player_log, stat, roles) for stat in PROP_LINES
    }
    rate_priors = {stat: role_rate_priors(recent_log, stat, roles) for stat in PROP_LINES}
    # Per-market prop calibrators from the walk-forward backtest (round-2 §2.3):
    # corrects systematic over/under-prediction before legs are classified/staked.
    prop_calibrators = {} if (synthetic_props or skip_props) else load_or_fit_prop_calibrators(
        player_log, eval_start_year=current_season - PROP_CALIBRATION_LOOKBACK)
    # Per-team expected volume-stat totals (disposals/marks/tackles) for the
    # pace -> team total -> player share allocation (plan §2.5, §3.3).
    team_stat_profiles = team_stat_total_profiles(player_log)
    league_totals = league_stat_totals(player_log)
    volume_stats = [s for s in PROP_LINES if s in PACE_STATS]
    # Expected boundary throw-ins per game for the OOB market (plan §1.6c) — a
    # real per-game count feed isn't free, so this is the league prior until
    # afl_bot.data.stoppages is fed.
    mu_oob = expected_oob(load_boundary_throwins())

    # H2H market-blend ensemble (plan §3.5): fit a calibrator + convex blend of
    # model / closing-market / Squiggle-consensus win probs on history, then
    # take the edge on the market-anchored blend (kills false edges). Best
    # effort — degrades to the raw model if odds/tips are unavailable.
    blend = None
    squiggle_lookup: dict[tuple[str, str], float] = {}
    try:
        hist_tips = pd.concat([client.get_tips(y) for y in _history_years(year)], ignore_index=True)
        blend = fit_market_blend(assemble_signals(history, fetch_historical_odds(), hist_tips))
        consensus = squiggle_consensus(client.get_tips(year, round_no))
        squiggle_lookup = {(r.hteam, r.ateam): r.squiggle_home_prob for r in consensus.itertuples()}
    except Exception as exc:  # noqa: BLE001 - anchoring is optional, must not break pricing
        print(f"Market-blend anchoring unavailable ({exc}); using raw model probs.", file=sys.stderr)

    odds_book: dict[str, float] = {}
    if odds_path:
        odds_book = json.loads(Path(odds_path).read_text())
    # Per-book house rules (round-2 §8.4), e.g. {"_rules": {"h2h_draw": "refund"}}.
    draw_refund = odds_book.get("_rules", {}).get("h2h_draw") == "refund"

    rng = make_rng()
    candidates: list[LegCandidate] = []

    print(f"=== {year} Round {round_no} ===\n")

    for _, fx in fixtures.iterrows():
        home_name, away_name = fx["hteam"], fx["ateam"]
        match_id = f"{year}_r{round_no}_{home_name}_v_{away_name}"
        venue = fx["venue"]
        roofed = is_roofed(venue)
        greasiness = _fixture_greasiness(fx, rain_mm, roofed)
        is_wet = greasiness >= 0.5
        wx = f" [WET g={greasiness:.2f}]" if is_wet else (f" [greasy g={greasiness:.2f}]" if greasiness > 0.1 else (" [roof]" if roofed else ""))
        print(f"--- {home_name} vs {away_name} ({venue}){wx} ---")

        home = Team(home_name, is_home=True)
        away = Team(away_name, is_home=False)

        mu_margin = elo.expected_margin(home_name, away_name,
                                        hga=_fixture_hga(home_name, away_name, venue, team_hga))
        hp = scoring_profiles.get(home_name, {"off_rate": 90.0, "def_rate": 90.0})
        ap = scoring_profiles.get(away_name, {"off_rate": 90.0, "def_rate": 90.0})
        mu_total = expected_total(hp["off_rate"], hp["def_rate"], ap["off_rate"], ap["def_rate"],
                                  venue_factor=venue_factors.get(venue, 1.0))

        print(f"  Predicted margin (home): {mu_margin:+.1f} pts | total: {mu_total:.0f} pts")

        home_accuracy = accuracy_profiles.get(home_name, float("nan"))
        away_accuracy = accuracy_profiles.get(away_name, float("nan"))
        match = simulate_match(home, away, mu_margin, mu_total, home_accuracy, away_accuracy,
                               n_sims, rng, greasiness=greasiness)
        p_home_win = prob_event(match["home_win"] > 0)
        p_away_win = prob_event(match["away_win"] > 0)
        p_draw = prob_event(match["draw"] > 0)
        if p_draw > 0:
            print(f"  P(draw) = {p_draw:.3f}")
        # Draw refunds the stake -> condition the win prob on a non-draw (§8.4).
        if draw_refund and p_draw > 0:
            p_home_win = p_home_win / (1.0 - p_draw)
            p_away_win = p_away_win / (1.0 - p_draw)

        # Market-anchored H2H prob (plan §3.5): blend the model with the devig
        # market + Squiggle consensus when both book prices are available.
        home_odds = odds_book.get(f"{home_name} to win")
        away_odds = odds_book.get(f"{away_name} to win")
        blended_home = None
        if blend is not None and home_odds and away_odds:
            market_home = devig_proportional([home_odds, away_odds])[0]
            squiggle_home = squiggle_lookup.get((home_name, away_name))
            blended_home = float(blend.predict_home_prob(
                p_home_win, market_p=market_home, squiggle_p=squiggle_home)[0])

        total_pts = match["home_pts"] + match["away_pts"]
        for team_name, model_prob, is_home_side in (
            (home_name, p_home_win, True), (away_name, p_away_win, False),
        ):
            leg_name = f"{team_name} to win"
            odds = odds_book.get(leg_name)
            if blended_home is not None:
                fair_prob = blended_home if is_home_side else 1.0 - blended_home
            else:
                fair_prob = model_prob
            win_mask = (match["home_win"] > 0) if is_home_side else (match["away_win"] > 0)
            blend_note = f" -> blend {fair_prob:.3f}" if blended_home is not None else ""
            print(f"  P({leg_name}) = {model_prob:.3f}{blend_note} -> fair {fair_odds(fair_prob):.2f}"
                  + (f" | book {odds}" if odds else ""))
            if odds:
                candidates.append(LegCandidate(
                    name=leg_name, match_id=match_id, market="h2h",
                    subject=team_name, fair_prob=fair_prob, market_odds=odds, mask=win_mask,
                ))

        # Total points line, e.g. 160.5
        total_line = round(mu_total / 5) * 5 + 0.5
        total_mask = total_pts >= total_line
        p_total_over = prob_event(total_mask)
        leg_name = f"Total points {total_line}+"
        odds = odds_book.get(leg_name)
        print(f"  P({leg_name}) = {p_total_over:.3f} -> fair {fair_odds(p_total_over):.2f}"
              + (f" | book {odds}" if odds else ""))
        if odds:
            candidates.append(LegCandidate(
                name=leg_name, match_id=match_id, market="total_points",
                subject="total", fair_prob=p_total_over, market_odds=odds, mask=total_mask,
            ))

        # Boundary throw-ins (OOB) total market (plan §1.6c): coupled negatively
        # to this match's total draw, lifted in the wet. Prior-based until a real
        # boundary-throw-in feed is wired (afl_bot.data.stoppages).
        oob_samples = simulate_boundary_throwins(mu_oob, total_pts, rng, greasiness=greasiness)
        oob_line = round(mu_oob) + 0.5
        oob_mask = oob_samples >= oob_line
        p_oob_over = prob_event(oob_mask)
        leg_name = f"Total boundary throw-ins {oob_line}+"
        odds = odds_book.get(leg_name)
        print(f"  P({leg_name}) = {p_oob_over:.3f} (prior) -> fair {fair_odds(p_oob_over):.2f}"
              + (f" | book {odds}" if odds else ""))
        if odds:
            candidates.append(LegCandidate(
                name=leg_name, match_id=match_id, market="boundary_throwins",
                subject="oob", fair_prob=p_oob_over, market_odds=odds, mask=oob_mask,
            ))

        # Player props: draw the shared game environment (pace) once, then for
        # each team draw pace-scaled volume-stat totals and allocate them among
        # players via a Dirichlet share (plan §2.5, §3.3). Goals keep the
        # scoreline-correlated NB path.
        if skip_props:
            print()
            continue
        pace = draw_pace(n_sims, rng)

        for team, is_home_team in ((home_name, True), (away_name, False)):
            opponent = away_name if is_home_team else home_name
            # Current-season (or last-season) active pool, gated to the confirmed
            # lineup when one is supplied (round-2 §1.1/§1.2/§1.3).
            confirmed = lineup.get(team)
            usage_players = _select_players(
                player_log, team, current_season, PLAYERS_PER_TEAM_SAMPLE, confirmed=confirmed)
            player_samples = _team_player_samples(
                usage_players, team, opponent, is_home_team, match, pace,
                player_log, roles, rate_priors, team_stat_profiles, league_totals,
                volume_stats, greasiness, roofed, n_sims, rng, lineup_tog=lineup_tog)

            # Price every line from the per-player sample arrays.
            for player_name in usage_players:
                # confirmed unless a lineup is supplied and this player isn't named
                confirmed = (team not in lineup) or (player_name in lineup[team])
                for stat, lines in PROP_LINES.items():
                    samples = player_samples[player_name].get(stat)
                    if samples is None:
                        continue
                    for line in lines:
                        leg_mask = samples >= line
                        prob = prob_event(leg_mask)
                        prob = apply_prop_calibration(prop_calibrators, stat, line, prob)  # §2.3, Phase 3.2
                        if not (LEG_PROB_MIN < prob < LEG_PROB_MAX):
                            continue
                        leg_name = f"{player_name} {line}+ {stat}"
                        odds = odds_book.get(leg_name)
                        if odds:
                            candidates.append(LegCandidate(
                                name=leg_name, match_id=match_id, market=f"player_{stat}",
                                subject=player_name, fair_prob=prob, market_odds=odds,
                                confirmed=confirmed, mask=leg_mask,
                            ))
        print()

    if not odds_book:
        print("No --odds file supplied: showing fair probabilities/odds only.")
        print("Pass a JSON file mapping leg names -> market odds to build multis.")
        return

    # Warn on odds-file keys that never matched a priced leg (a typo = a leg
    # silently dropped, round-2 §7.4).
    priced_names = {c.name for c in candidates}
    unmatched = [k for k in odds_book if not k.startswith("_") and k not in priced_names]
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} odds key(s) matched no priced leg "
              f"(typo? player not in pool/lineup?): {', '.join(unmatched)}", file=sys.stderr)

    print("\n=== Candidate legs (with market odds) ===")
    for c in sorted(candidates, key=lambda c: c.edge_pct, reverse=True):
        se = mc_standard_error(c.fair_prob, n_sims)            # MC precision (§8.3)
        se_flag = " !SE" if c.classification == "ANCHOR" and se > MC_SE_TARGET else ""
        print(f"  [{c.classification:6s}] {c.name:35s} model {c.fair_prob:.3f} "
              f"| book {c.market_odds:.2f} | edge {c.edge_pct*100:+.1f}% | SE {se:.4f}{se_flag}")

    promo = build_promo_multi(candidates, joint_prob_fn=joint_prob_from_masks)
    print("\n=== 3-leg promo multi (2 anchors + 1 value) ===")
    if promo is None:
        print("  No qualifying combination found (no positive promo EV).")
    else:
        for leg in promo.legs:
            print(f"  [{leg.classification:6s}] {leg.name}  model {leg.fair_prob:.3f} "
                  f"| book {leg.market_odds:.2f}")
        print(f"  combined fair prob {promo.combined_fair_prob:.3f} "
              f"-> fair odds {promo.combined_fair_odds:.2f}")
        print(f"  book multi odds    {promo.combined_market_odds:.2f}")
        print(f"  P(all win)         {promo.promo['p_all_win']:.3f}")
        print(f"  P(exactly 1 loss)  {promo.promo['p_exactly_one_loss']:.3f}  (refund branch)")
        print(f"  Promo EV           ${promo.promo['ev_dollars']:+.2f} "
              f"({promo.promo['ev_pct']*100:+.1f}%)")

    print("\n=== 'Very highly likely' anchor multis (ranked by edge) ===")
    anchor_multis = build_anchor_multis(candidates, joint_prob_fn=joint_prob_from_masks)
    if not anchor_multis:
        print("  No qualifying all-ANCHOR combinations found.")
    # High combined probability is not value: rank/flag by the market-anchored
    # combined edge, not by probability (round-2 §8.1/§8.2).
    ranked_anchor = sorted(
        anchor_multis,
        key=lambda m: _multi_anchored_prob(m) * m.combined_market_odds - 1.0, reverse=True,
    )
    for i, multi in enumerate(ranked_anchor, 1):
        anchored = _multi_anchored_prob(multi)
        edge = anchored * multi.combined_market_odds - 1.0
        tag = "VALUE" if edge > 0 else "-EV  "
        names = " + ".join(leg.name for leg in multi.legs)
        print(f"  {i}. [{tag}] {names}")
        print(f"       prob {multi.combined_fair_prob:.3f} (mkt-anchored {anchored:.3f}) "
              f"| book {multi.combined_market_odds:.2f} | edge {edge*100:+.1f}%")

    # --- Staking (plan §4.4): capped fractional Kelly + bankroll Monte Carlo ---
    bet_specs = [(c.name, c.fair_prob, c.market_odds, c.mask)
                 for c in candidates if c.edge_pct > 0]
    if promo is not None:
        promo_masks = [leg.mask for leg in promo.legs]
        promo_mask = (np.logical_and.reduce([np.asarray(m, bool) for m in promo_masks])
                      if all(m is not None for m in promo_masks) else None)
        # Stake the promo on the market-anchored combined prob (§8.2), not the raw.
        bet_specs.append((f"PROMO MULTI ({len(promo.legs)} legs)",
                          _multi_anchored_prob(promo), promo.combined_market_odds, promo_mask))

    # Half-Kelly on prop legs — noisier and compounding in multis (round-2 §2.5).
    prop_names = {c.name for c in candidates if c.market.startswith("player_")}
    mults = [PROP_KELLY_MULTIPLIER if n in prop_names else 1.0 for n, _, _, _ in bet_specs]
    staked = stake_bets([(n, p, o) for n, p, o, _ in bet_specs], bankroll, mults=mults)
    placed = [(s, m) for s, (_, _, _, m) in zip(staked, bet_specs) if s.stake > 0]

    print(f"\n=== Recommended stakes (0.25x Kelly, bankroll ${bankroll:.0f}) ===")
    if not placed:
        print("  No positive-edge bets to stake.")
    else:
        for s, _ in sorted(placed, key=lambda sm: sm[0].stake, reverse=True):
            print(f"  ${s.stake:7.2f} ({s.fraction * 100:4.1f}% bank) on {s.name:35s} "
                  f"@ {s.odds:.2f} (model {s.prob:.3f})")
        total_frac = sum(s.fraction for s, _ in placed)
        print(f"  total staked ${total_frac * bankroll:.2f} ({total_frac * 100:.1f}% of bankroll)")

        # Joint sim off the per-leg masks when every staked bet has one (captures
        # promo/single overlap, round-2 §3.4); else independent fallback.
        masks = [m for _, m in placed]
        if all(m is not None for m in masks):
            sim = simulate_bankroll_joint(
                [(s.odds, s.fraction) for s, _ in placed], np.vstack(masks), bankroll,
                rounds=24, n_sims=20_000, rng=make_rng(),
            )
            joint_note = " (joint/correlated)"
        else:
            sim = simulate_bankroll(
                [(s.prob, s.odds, s.fraction) for s, _ in placed], bankroll,
                rounds=24, n_sims=20_000, rng=make_rng(),
            )
            joint_note = ""
        rep = bankroll_report(sim, bankroll)
        print(f"\n=== Bankroll sim{joint_note} (this edge profile x 24 rounds, ${bankroll:.0f} start) ===")
        print(f"  median end ${rep['median_terminal']:.0f} "
              f"(5th ${rep['p5_terminal']:.0f} / 95th ${rep['p95_terminal']:.0f})")
        print(f"  P(profit) {rep['p_profit']:.2f} | P(bust <10% start) {rep['p_bust']:.2f}")
        print(f"  median max drawdown {rep['median_max_drawdown'] * 100:.0f}% | "
              f"P(drawdown >50%) {rep['p_drawdown_over_50pct']:.2f}")

    print("\nReminder: this is a modelling/analytics tool. No model reliably beats")
    print("the bookies long-term on AFL -- only stake what you can afford to lose.")
    print("Gambling Help Online: gamblinghelponline.org.au | 1800 858 858")


def _enforce_ladder_monotonicity(rungs: list[dict]) -> None:
    """Within a ladder, higher shrunk Total EV must not get fewer units than lower EV.
    Operates in-place on rung dicts that already have 'units'/'units_tag' set."""
    def _ev(r: dict) -> float:
        v = r.get("total_ev")
        if v is None:
            v = r.get("edge", 0.0)
        return v or 0.0

    # Exclude MODEL-ONLY (no book price → 0 units, not a real stake decision) so that
    # unstakeable rungs don't drag every lower-EV staked rung down to 0.
    staked = [(i, r) for i, r in enumerate(rungs)
              if not r.get("no_bet")
              and r.get("units_tag") not in ("NO BET", "CHECK PRICING", "MODEL-ONLY")]
    staked_by_ev = sorted(staked, key=lambda x: _ev(x[1]), reverse=True)
    min_units = float("inf")
    for i, r in staked_by_ev:
        u = r.get("units", 0.0)
        if u > min_units:
            rungs[i]["units"] = min_units
            old_tag = rungs[i].get("units_tag", "")
            if old_tag and old_tag not in ("NO BET", "MODEL-ONLY", "CHECK PRICING"):
                rungs[i]["units_tag"] = re.sub(r"^[\d.]+u", f"{min_units:g}u", old_tag)
        else:
            min_units = u


def _apply_round_cap(matches: list[dict]) -> None:
    """Apply per-round Kelly cap as a budget allocator across all staked rungs.

    Ranks all staked rungs by total_ev descending and gives each its FULL formula
    units until the round budget (15u) is exhausted.  The last rung that partially
    fits is trimmed to the remaining budget (floor to UNIT_STEP, min UNIT_STEP).
    Rungs that don't fit are marked 'NO BET (round cap)'.  A kept rung's units are
    never reduced below its formula output other than that final trim.
    """
    import math as _math
    round_cap_units = KELLY_PER_ROUND_CAP * BANKROLL / UNIT_SIZE

    staked: list[tuple[float, dict]] = []
    for m in matches:
        for r in m.get("sgms", []) + m.get("market_sgms", []):
            if (r.get("units", 0.0) > 0
                    and r.get("units_tag") not in ("NO BET", "MODEL-ONLY", "CHECK PRICING")):
                ev = r.get("total_ev") if r.get("total_ev") is not None else (r.get("edge") or 0.0)
                staked.append((ev, r))
        pe = m.get("pull_em")
        if pe and not pe.get("no_valid_combo") and pe.get("units", 0.0) > 0:
            ev = (pe.get("total_ev") if pe.get("total_ev") is not None
                  else (pe.get("option_ev") or 0.0) / 100.0)
            staked.append((ev, pe))

    total = sum(r.get("units", 0.0) for _, r in staked)
    if total <= round_cap_units + 1e-9:
        return

    # Allocation: rank by EV desc — give each rung its FULL formula units until budget gone.
    staked.sort(key=lambda x: x[0], reverse=True)
    budget = round_cap_units
    for _, r in staked:
        u = r.get("units", 0.0)
        if budget <= 1e-9:
            r["units"] = 0.0
            r["units_tag"] = "NO BET (round cap)"
        elif u <= budget + 1e-9:
            budget = max(0.0, budget - u)
        else:
            # Partial fit: trim to remaining budget, floored to UNIT_STEP.
            trimmed = _math.floor(budget / UNIT_STEP) * UNIT_STEP
            if trimmed >= UNIT_STEP:
                r["units"] = trimmed
                old_tag = r.get("units_tag", "")
                if old_tag and old_tag not in ("NO BET", "MODEL-ONLY", "CHECK PRICING"):
                    r["units_tag"] = re.sub(r"^[\d.]+u", f"{trimmed:g}u", old_tag)
            else:
                r["units"] = 0.0
                r["units_tag"] = "NO BET (round cap)"
            budget = 0.0


def _rung_to_json(rung: dict, ladder: str, year: int, round_no: int,
                  home: str, away: str,
                  leg_by_name: dict, odds_book: dict,
                  greasiness: float = 0.0) -> dict:
    """Convert one selected SGM rung to the multis-JSON schema (Stage 2A).
    Both the .md and this JSON draw from the same rung dict, so they can never disagree."""
    h = home.replace(" ", "_")
    a = away.replace(" ", "_")
    band = rung.get("target_odds") or rung.get("book_odds") or rung.get("fair_odds")
    multi_id = f"{year}-r{round_no}-{h}-{a}-{ladder}-{band:.2f}"

    legs_json = []
    for name in rung["legs"]:
        leg = leg_by_name.get(name)
        if leg is None:
            legs_json.append({"name": name})
            continue
        market_display = (leg.market.replace("player_", "")
                          if leg.market.startswith("player_") else leg.market)
        m = re.search(r"(\d+)\+", name)
        line = int(m.group(1)) if m else None
        legs_json.append({
            "player": leg.subject,
            "market": market_display,
            "line": line,
            "name": name,
            "book_odds": odds_book.get(name),
            "hit_prob": round(leg.fair_prob, 4),
        })

    # Use pre-computed staking (set by _units_fields + monotonicity enforcement
    # at the call site) if present; fall back to fresh computation.
    if "units" in rung and "units_tag" in rung:
        uf = {"units": rung["units"], "units_tag": rung["units_tag"]}
    else:
        uf = _units_fields(rung)
    rec = {
        "id": multi_id,
        "year": year,
        "round": round_no,
        "game": f"{home} vs {away}",
        "ladder": ladder,
        "band": band,
        "legs": legs_json,
        "model_joint": rung.get("joint_prob"),
        "model_fair": rung.get("fair_odds"),
        "book_combo": rung.get("book_odds"),
        "edge": rung.get("edge"),
        "p_all_win": rung.get("p_all_win"),
        "p_one_loss": rung.get("p_one_loss"),
        "promo_ev": rung.get("promo_ev"),
        "total_ev": rung.get("total_ev"),
        "suggested_stake": rung.get("suggested_stake"),
        "value_pick": bool(rung.get("value_pick", False)),
        "greasiness": round(greasiness, 3),
        "no_bet": bool(rung.get("no_bet", False)),
        **uf,
    }
    # Add leg-by-leg detail when flagged suspect (CHECK PRICING).
    if uf.get("units_tag") == "CHECK PRICING":
        rec["suspect_legs"] = [
            {"name": lg.get("name"), "book_odds": lg.get("book_odds"),
             "model_prob": lg.get("hit_prob"),
             "model_fair": round(1.0 / lg["hit_prob"], 2) if lg.get("hit_prob") else None}
            for lg in legs_json
        ]
    return rec


def _units_fields(rung: dict) -> dict:
    """Compute units/tag for a rung and return as a dict to merge into the record.

    Part C sanity guard: if book_combo > SUSPECT_BOOK_FAIR_RATIO * model_fair
    OR raw_edge > SUSPECT_MAX_RAW_EDGE, flag the rung as CHECK PRICING with
    zero stake.  This catches leg-pricing/mapping bugs that produce implausibly
    large edges (e.g. a wrong market mapped to the wrong player) without being
    tripped by legitimate modest edges after the corr_gain haircut is applied.
    """
    if rung.get("no_bet"):
        return {"units": 0.0, "units_tag": "NO BET"}
    book_combo = rung.get("book_odds") or rung.get("book_combo")
    model_fair = rung.get("fair_odds")
    raw_edge = rung.get("raw_edge")
    if (book_combo is not None and model_fair is not None and model_fair > 0
            and (book_combo > SUSPECT_BOOK_FAIR_RATIO * model_fair
                 or (raw_edge is not None and raw_edge > SUSPECT_MAX_RAW_EDGE))):
        return {"units": 0.0, "units_tag": "CHECK PRICING"}
    units, tag = recommend_units(
        rung.get("joint_prob"),
        rung.get("book_odds"),
        rung.get("promo_ev"),
        p_win=rung.get("p_all_win"),
        p_one_loss=rung.get("p_one_loss"),
        p_dead=rung.get("p_two_plus_loss"),
    )
    return {"units": units, "units_tag": tag}


def _pull_em_units_fields(pull_em: dict) -> dict:
    """Compute multi-outcome Kelly units for a Pull 'Em SGM.

    Uses the promo branch probs computed in build_pull_em_sgm (promo_p_win,
    promo_p_one_miss, promo_p_dead, promo_R_eff) which capture the recovery
    probability via PULL_DETECTION_PROB and the weighted average reduced odds.
    """
    import math as _math
    p_win = pull_em.get("promo_p_win")
    p_one_miss = pull_em.get("promo_p_one_miss")
    p_dead = pull_em.get("promo_p_dead")
    R_eff = pull_em.get("promo_R_eff")
    option_ev = pull_em.get("option_ev", 0.0)

    if (p_win is None or p_one_miss is None or p_dead is None
            or R_eff is None or option_ev <= 0 or p_one_miss <= 0):
        return {"units": 0.0, "units_tag": "NO BET"}

    book_combo = pull_em["book_combo"]
    frac = multi_outcome_kelly(p_win, p_one_miss, p_dead, book_combo, R_eff)
    if frac <= 0.0:
        return {"units": 0.0, "units_tag": "NO BET"}

    raw_units = frac * BANKROLL / UNIT_SIZE
    cap = UNIT_MAX_LONGSHOT if book_combo >= 5.0 else UNIT_MAX
    units = min(_math.floor(raw_units / UNIT_STEP) * UNIT_STEP, cap)
    units = max(units, UNIT_STEP)
    capped = False
    if units * UNIT_SIZE > PROMO_REFUND_CAP:
        units = _math.floor(PROMO_REFUND_CAP / UNIT_SIZE / UNIT_STEP) * UNIT_STEP
        units = max(units, UNIT_STEP)
        capped = True
    tag = f"{units:g}u PROMO KELLY"
    if capped:
        tag += " (capped by promo refund limit)"
    return {"units": units, "units_tag": tag}


def round_report(year: int, round_no: int | None, odds_path: str | None, n_sims: int,
                 rain_mm: float | None = None, lineup_path: str | None = None,
                 use_live: bool = False, multis_only: bool = False,
                 auto_lineup: bool = False, multi_calibration: bool = False,
                 corr_gain_haircut: float = CORR_GAIN_HAIRCUT,
                 use_sportsbet: bool = False, sportsbet_urls_path: str | None = None,
                 outs_path: str | None = None,
                 greasiness_overrides_path: str | None = None) -> None:
    """The weekly deliverable (round-2 §10): per-match real-player projection
    tables + same-game multis ranked by joint sim probability, saved to
    reports/<year>_r<N>_report.md. REAL players only — refuses a synthetic log.

    ``multi_calibration=False`` (default, opt-in) -- when True, applies a
    selection-level isotonic calibrator (model-upgrade audit Phase 3.6,
    `afl_bot.backtest.multis.load_or_fit_multi_calibrator`) to every selected
    rung's joint probability before it's displayed. Phase 3.5 confirmed
    `search_match_sgms`'s selection is itself a biased estimator (the
    optimizer's curse: closest-to-target selection over many noisy
    candidates over-selects upward noise); Phase 3.6 corrects that bias
    directly on the population actually bet, since two attempts to fix the
    selection mechanism itself didn't work.

    **Manual prop market-blend (model-upgrade audit Phase 4 --
    PHASE-4-CODE-PLAN.md, replaces the paid-API version of Phase 4).** Every
    run also writes ``reports/<year>_r<N>_odds_template.json`` -- every
    priceable leg's exact name (h2h, totals, every gated prop line) mapped to
    ``null`` -- so filling in book prices from the bookie and passing the
    file back via ``--odds`` is copy-paste, not retyping leg names by hand
    (kills the typo class of bug at the source); the filled file itself is
    snapshotted to ``reports/<year>_r<N>_odds.json`` (STEP 2.4) so the repo
    accumulates its own prop-odds history over time. A "Total points
    {line}+" leg now exists too (previously round-report priced no totals
    leg at all), but ONLY once it has a real price in ``--odds`` -- with no
    ``--odds`` the leg set, and therefore the SGM ladder, is unchanged. Any
    priced prop's CALIBRATED model prob is pulled ``PROP_MARKET_BLEND_WEIGHT``
    of the way toward its devigged book price (STEP 2.1, single-sided
    "approx" devig if only one side is entered) before pricing/classifying
    that leg -- a documented prior (no historical prop-odds archive exists
    to fit it against), not a backtested weight. The blend is surfaced in a
    new "Priced props" table per match (model vs book vs devig vs blended vs
    edge vs class) so its effect is visible at the single-leg level, not
    just inside the SGM ladder -- the SGM ladder's own selected-rung joint
    probabilities (from the correlated sim masks) are untouched by this
    blend, so this mitigates per-leg overconfidence on priced legs only, not
    the joint overconfidence Phase 1-3.6 found. Odds-file keys that never
    matched a priceable leg warn (same as ``run-round``).

    ``corr_gain_haircut`` (corr_gain-diagnostic follow-up, default
    ``CORR_GAIN_HAIRCUT`` = 0.0, now the LIVE DEFAULT) passes through to
    `search_match_sgms`'s same param -- OOS-validated both directions and,
    stacked with the always-on per-leg prop calibration above, the best
    result of the whole model-upgrade overconfidence investigation (log
    loss 0.5757 -> 0.5650, high-bucket gap +0.110 -> +0.051). See the
    README's "corr_gain haircut" section for the closing writeup and the
    accepted, bounded ~+0.05 residual. Pass 1.0 via --corr-gain-haircut for
    the raw/unhaircut sim joint_prob (diagnostics only).

    ``use_sportsbet`` (FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART A, default
    False, opt-in via ``--sportsbet``) -- scrapes REAL odds straight off
    Sportsbet's own JSON API (no key, no paid tier) for every event in
    ``sportsbet_urls_path`` (default ``reports/<year>_r<N>_sportsbet_urls.json``,
    a plain JSON list of Sportsbet match URLs Ben pastes in once per round)
    and merges them into ``odds_book`` (manual ``--odds`` still overrides,
    for hand-fixes). AU-IP ONLY -- Sportsbet geo-blocks everyone else, so this
    must run on Ben's own machine, never CI; a block/missing-file/network
    failure degrades to an empty dict and the report says so. Every match
    then gets a SECOND ladder (`search_market_sgms`) selected and priced on
    these real prices, printed beside the existing model-only ladder.

    ``outs_path`` (PART B1, default None, opt-in via ``--outs``) -- a manual
    override that ALWAYS removes named players from the resolved lineup
    (auto or manual), e.g. a Footywire squad cut that slipped through. Also
    read directly off a ``--lineup`` file's own ``"_outs"`` key if present,
    so one file can carry both. See `afl_bot.data.lineups.load_outs`."""
    client = SquiggleClient()
    history = pd.concat([client.get_completed_games(y) for y in _history_years(year)],
                        ignore_index=True)
    if history.empty:
        print("No historical data available.", file=sys.stderr)
        return
    upcoming = client.get_upcoming_games(year)
    if round_no is None and not upcoming.empty:
        round_no = int(upcoming["round"].iloc[0])
    fixtures = upcoming[upcoming["round"] == round_no]
    if fixtures.empty:
        # Fall back to a completed round (backfill/grading). Profiles then include
        # that round's games, so it's not strictly walk-forward — fine for a
        # backfilled report, but live use should run before the round.
        completed = client.get_completed_games(year)
        fixtures = completed[completed["round"] == round_no]
        if not fixtures.empty:
            history = history[~((history["year"] == year) & (history["round"] == round_no))]
    if fixtures.empty:
        print(f"No games found for {year} round {round_no}.", file=sys.stderr)
        return

    player_log, log_source = load_player_log(history, prefer_real=True, return_source=True)
    if log_source == "synthetic":
        print("REFUSING: player log fell back to SYNTHETIC data — the round report is "
              "real-players-only. Fix the data sources (DFS/Fryzigg) and retry.", file=sys.stderr)
        return

    current_season = int(player_log["year"].max())
    team_hga = fit_team_hga(history)   # per-team venue HGA (§6.1)
    elo, _ = build_ratings_from_history(attach_hga(history, team_hga), **load_fitted_elo_params())
    scoring_profiles = team_scoring_profiles(history)
    accuracy_profiles = team_shot_accuracy_profiles(history)
    venue_factors = venue_scoring_factors(history)   # per-venue scoring (§6.4)
    roles = classify_roles(player_log)
    recent_log = player_log[player_log["year"] > current_season - PROP_RECENT_SEASONS]
    if recent_log.empty:
        recent_log = player_log
    rate_priors = {stat: role_rate_priors(recent_log, stat, roles) for stat in PROP_LINES}  # §5.2
    prop_calibrators = load_or_fit_prop_calibrators(
        player_log, eval_start_year=current_season - PROP_CALIBRATION_LOOKBACK)
    multi_cal = None
    if multi_calibration:
        from afl_bot.backtest.multis import load_or_fit_multi_calibrator
        multi_cal = load_or_fit_multi_calibrator(
            history, player_log, eval_start_year=current_season - MULTI_CALIBRATION_LOOKBACK,
            eval_end_year=current_season, n_sims=n_sims)
    team_stat_profiles = team_stat_total_profiles(player_log)
    league_totals = league_stat_totals(player_log)
    volume_stats = [s for s in PROP_LINES if s in PACE_STATS]
    mu_oob = expected_oob(load_boundary_throwins())
    # Lineup: manual file > auto-fetch > none (REAL-MULTIS Problem 1)
    lineup_source: str | None = None
    if lineup_path:
        lineup = load_lineup(lineup_path)
        lineup_tog = load_lineup_tog(lineup_path)
        lineup_source = "manual file"
    elif auto_lineup:
        lineup = fetch_lineup(year, round_no)
        lineup_tog = {}
        if lineup:
            n_teams = len(lineup)
            lineup_source = f"Footywire team selections ({n_teams} team(s) confirmed)"
            print(f"Auto-lineup: {n_teams} team(s) fetched from Footywire.", file=sys.stderr)
        else:
            lineup_source = "Footywire (sheets not yet posted — no filter applied)"
            print("Auto-lineup: Footywire sheets not yet posted; pricing top-usage players.",
                  file=sys.stderr)
    else:
        lineup = {}
        lineup_tog = {}
    # Manual outs override (FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART B1) --
    # always removes a named player regardless of what the lineup source
    # said, e.g. a Footywire squad cut the section-aware parser (PART B2)
    # still missed because that team's sheet hadn't separated out
    # Emergencies yet. Read from a dedicated --outs file AND/OR a "_outs"
    # key embedded directly in the --lineup file.
    outs: dict[str, set[str]] = {}
    if lineup_path:
        for team, names in load_outs(lineup_path).items():
            outs.setdefault(team, set()).update(names)
    if outs_path:
        for team, names in load_outs(outs_path).items():
            outs.setdefault(team, set()).update(names)
    n_outs_excluded = 0
    if outs:
        lineup, n_outs_excluded = apply_outs(lineup, outs)
    # Availability filter (Part 3 NEXT-STEPS-PLAN): always-on, runs every report.
    # Fetches Footywire injury list for Season/LT/Indefinite players, merges with
    # MANUALLY_UNAVAILABLE from config. apply_outs removes them from confirmed
    # lineup sets (teams whose sheet IS posted); the check at the LegCandidate
    # building step below handles teams whose sheet is NOT yet posted.
    known_outs: dict[str, set[str]] = {}
    _injury_list = fetch_injury_list(year, round_no)
    for _team, _players in _injury_list.items():
        known_outs.setdefault(_team, set()).update(_players)
    from afl_bot.data.teams import normalize_team_name as _norm_team
    for _team_name, _players in MANUALLY_UNAVAILABLE.items():
        try:
            _canonical = _norm_team(_team_name)
            known_outs.setdefault(_canonical, set()).update(_players)
        except KeyError:
            print(f"MANUALLY_UNAVAILABLE: unrecognised team {_team_name!r} — skipping.",
                  file=sys.stderr)
    if known_outs:
        lineup, _n_inj = apply_outs(lineup, known_outs)
        n_outs_excluded += _n_inj
        n_inj_total = sum(len(v) for v in known_outs.values())
        print(f"Availability filter: {n_inj_total} player(s) in injury/config blocklist "
              f"({_n_inj} removed from confirmed lineup sets).", file=sys.stderr)
    # Live h2h/totals + player props (The Odds API) and/or real Sportsbet
    # prices, merged with any --odds file; manual overrides everything for
    # hand-fixes (MULTI-CHANGES PART A; FIX-REAL-SPORTSBET-ODDS-AND-LINEUP
    # PART A6). Sportsbet is the richer, real source (props included, not
    # just h2h/totals), so it sits ahead of the (already-live) Odds API feed.
    live = fetch_live_odds(round_no) if use_live else {}
    live_props = fetch_live_props(round_no) if use_live else {}
    # Always compute the default URLs path — needed for diagnostics even when
    # --sportsbet is not used, so the note tells the user exactly what to populate.
    _sb_urls_path = sportsbet_urls_path or str(ROOT_DIR / "reports" /
                                                f"{year}_r{round_no}_sportsbet_urls.json")
    if use_sportsbet:
        try:
            sb_urls = json.loads(Path(_sb_urls_path).read_text())
        except (OSError, json.JSONDecodeError):
            print(f"Sportsbet: no URL file at {_sb_urls_path} -- skipping scrape.",
                  file=sys.stderr)
            sb_urls = []
        sb = fetch_sportsbet_odds(sb_urls)
        if not sb:
            n_urls = len(sb_urls)
            _reason = (f"no URLs in {_sb_urls_path}" if n_urls == 0
                       else f"all {n_urls} event(s) failed — geo-blocked (AU IP required) or stale URLs")
            print(f"WARNING Sportsbet: 0 legs priced ({_reason}). "
                  f"Book/Edge columns will show '—'. "
                  f"Populate {_sb_urls_path} and rerun with --sportsbet.",
                  file=sys.stderr)
    else:
        sb = {}
        sb_urls = []
        print(f"Sportsbet: not requested. Paste round match URLs into "
              f"{_sb_urls_path} and rerun with --sportsbet for real prices.",
              file=sys.stderr)
    manual = json.loads(Path(odds_path).read_text()) if odds_path else {}
    odds_book = {**live, **live_props, **sb, **manual}
    odds_note = ""
    if use_live:
        prop_keys = [k for k in manual if not k.startswith("_")
                     and " to win" not in k and not k.startswith("Total points")]
        src = f"the --odds file ({len(prop_keys)} prop price(s))" if odds_path else "no prop source"
        # The free/standard live feed carries H2H/totals only, NOT player props -- so
        # most multi rungs (which are props) stay unpriced under --live-odds alone.
        odds_note = (
            f"_Live odds cover **H2H/totals only** ({len(live)} live leg(s) from The Odds API). "
            f"Player-prop legs are priced off {src}; rungs with no prop price show '-' for "
            f"Book/Edge and are NOT flagged VALUE. Edges shown are market-shrunk (capped)._")

    if use_sportsbet:
        if sb:
            sportsbet_note = (
                f"_Player-prop odds: live from Sportsbet (scraped, {len(sb)} leg(s) priced)._"
            )
        else:
            n_urls = len(sb_urls)
            _cause = (f"no URL file / empty list (`{_sb_urls_path}`)" if n_urls == 0
                      else f"all {n_urls} event(s) failed — possibly geo-blocked (AU IP required)")
            sportsbet_note = (
                f"_WARNING Sportsbet: 0 legs priced — {_cause}. Book/Edge columns show '—'. "
                f"Fix: populate `{_sb_urls_path}` with this round's Sportsbet match URLs "
                f"and rerun with `--sportsbet`._"
            )
    else:
        sportsbet_note = (
            f"_No Sportsbet prices this run. For real book prices: paste this round's "
            f"Sportsbet match URLs into `{_sb_urls_path}` and rerun with `--sportsbet`._"
        )

    # Per-game greasiness overrides (Step 3 FIX-MARKS-CAP-ALL-LEGS-AND-GREASINESS).
    # JSON file: {"Home vs Away": 0.75} or {"HomeTeam": 0.75}, float in [0,1].
    greasiness_overrides: dict[str, float] = {}
    if greasiness_overrides_path:
        try:
            greasiness_overrides = json.loads(Path(greasiness_overrides_path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: could not load greasiness overrides: {e}", file=sys.stderr)

    # PointsBet Pull 'Em menu (FIX-PULLEM-MENU-AND-STAKE-COLUMNS).
    # If the per-round JSON exists with real (non-null) prices, use those for
    # Pull 'Em leg selection so we only emit lines PointsBet actually offers.
    # pb_menu = None  -> file missing -> fall back to odds_book (Sportsbet)
    # pb_menu = {}    -> file exists but all-null -> show UNAVAILABLE
    # pb_menu = {...} -> real prices -> use them for eligibility + combo odds
    _pb_menu_path = ROOT_DIR / "reports" / f"{year}_r{round_no}_pointsbet_odds.json"
    if _pb_menu_path.exists():
        try:
            _pb_raw = json.loads(_pb_menu_path.read_text(encoding="utf-8"))
            pb_menu: dict[str, float] | None = {
                k: v for k, v in _pb_raw.items()
                if isinstance(v, (int, float)) and v > 0
            }
        except (OSError, json.JSONDecodeError):
            pb_menu = None
    else:
        pb_menu = None

    rng = make_rng()
    matches, predictions, multis_records = [], [], []
    n_tog_overrides = 0
    n_auto_excluded = n_outs_excluded
    # Every leg name this round COULD be priced for (model-upgrade audit Phase
    # 4 STEP 1.2) -- written to an odds template so hand-entry is copy-paste,
    # not guesswork. `known_input_keys` is the wider set used for the
    # unmatched-key warning below: it also covers the "-" (under) side of
    # each prop, which a user can optionally type for a two-way devig (STEP
    # 1.3) even though the template itself only prompts for the "+" side.
    priceable_names: list[str] = []
    known_input_keys: set[str] = set()

    for _, fx in fixtures.iterrows():
        home_name, away_name = fx["hteam"], fx["ateam"]
        match_id = f"{year}_r{round_no}_{home_name}_v_{away_name}"
        venue = fx["venue"]
        roofed = is_roofed(venue)
        game_key = f"{home_name} vs {away_name}"
        if game_key in greasiness_overrides:
            greasiness = max(0.0, min(1.0, float(greasiness_overrides[game_key])))
        elif home_name in greasiness_overrides:
            greasiness = max(0.0, min(1.0, float(greasiness_overrides[home_name])))
        else:
            greasiness = _fixture_greasiness(fx, rain_mm, roofed)
        is_wet = greasiness >= 0.5

        mu_margin = elo.expected_margin(home_name, away_name,
                                        hga=_fixture_hga(home_name, away_name, venue, team_hga))
        hp = scoring_profiles.get(home_name, {"off_rate": 90.0, "def_rate": 90.0})
        ap = scoring_profiles.get(away_name, {"off_rate": 90.0, "def_rate": 90.0})
        mu_total = expected_total(hp["off_rate"], hp["def_rate"], ap["off_rate"], ap["def_rate"],
                                  venue_factor=venue_factors.get(venue, 1.0))
        ha = accuracy_profiles.get(home_name, float("nan"))
        aa = accuracy_profiles.get(away_name, float("nan"))
        match = simulate_match(Team(home_name, True), Team(away_name), mu_margin, mu_total,
                               ha, aa, n_sims, rng, greasiness=greasiness)
        total_pts = match["home_pts"] + match["away_pts"]
        total_line = round(mu_total / 5) * 5 + 0.5
        total_mask = total_pts >= total_line

        header = {
            "home": home_name, "away": away_name, "venue": venue,
            "roofed": roofed, "is_wet": is_wet, "greasiness": round(greasiness, 3),
            "mu_margin": mu_margin, "mu_total": mu_total,
            "p_home": prob_event(match["home_win"] > 0), "p_away": prob_event(match["away_win"] > 0),
            "p_draw": prob_event(match["draw"] > 0),
            "total_line_name": f"Total {total_line}+", "p_total": prob_event(total_mask),
        }
        predictions.append({"match_id": match_id, "market": "h2h", "subject": home_name,
                            "line": "", "prob": header["p_home"]})
        predictions.append({"match_id": match_id, "market": "h2h", "subject": away_name,
                            "line": "", "prob": header["p_away"]})
        predictions.append({"match_id": match_id, "market": "total_points", "subject": "total",
                            "line": total_line, "prob": header["p_total"]})

        home_win_name = f"{home_name} to win"
        away_win_name = f"{away_name} to win"
        # Bookable name matches run-round's convention exactly (`Total points
        # {line}+`) so one --odds file works for either CLI. Display text in
        # the header bullet stays the shorter "Total {line}+" (cosmetic only).
        total_leg_name = f"Total points {total_line}+"
        priceable_names.extend([home_win_name, away_win_name, total_leg_name])
        known_input_keys.update([home_win_name, away_win_name, total_leg_name])

        match_legs = [
            LegCandidate(home_win_name, match_id, "h2h", home_name, header["p_home"],
                         odds_book.get(home_win_name, fair_odds(header["p_home"])),
                         mask=(match["home_win"] > 0)),
            LegCandidate(away_win_name, match_id, "h2h", away_name, header["p_away"],
                         odds_book.get(away_win_name, fair_odds(header["p_away"])),
                         mask=(match["away_win"] > 0)),
        ]
        # Total-points leg: always in the pool. Uses the book's O/U price when
        # available, fair_odds fallback for model-only runs (same as h2h legs).
        match_legs.append(LegCandidate(total_leg_name, match_id, "total_points", "total",
                                       header["p_total"],
                                       odds_book.get(total_leg_name, fair_odds(header["p_total"])),
                                       mask=total_mask))

        priced_legs: list[dict] = []   # this match's priced props, for the report table
        pace = draw_pace(n_sims, rng)
        projections = []
        for team, is_home_team in ((home_name, True), (away_name, False)):
            opponent = away_name if is_home_team else home_name
            team_confirmed = lineup.get(team)
            usage = _select_players(player_log, team, current_season,
                                    PLAYERS_PER_TEAM_SAMPLE, confirmed=team_confirmed)
            if auto_lineup and team_confirmed:
                unrestricted = _select_players(player_log, team, current_season,
                                               PLAYERS_PER_TEAM_SAMPLE)
                n_auto_excluded += sum(1 for p in unrestricted if p not in team_confirmed)
            n_tog_overrides += sum(1 for p in usage if p in lineup_tog)
            samples = _team_player_samples(
                usage, team, opponent, is_home_team, match, pace, player_log, roles,
                rate_priors, team_stat_profiles, league_totals, volume_stats, greasiness, roofed,
                n_sims, rng, lineup_tog=lineup_tog)
            projections.append((team, projection_rows(samples, PROP_LINES, prop_calibrators)))
            # "Book menu" filter (FIX-BETTABLE-LEGS-AND-PRICING STEP 1): which
            # players are even plausible bookmaker names for each stat on this
            # team, used below to gate MODEL-ONLY legs (no real price entered).
            team_top_by_stat = {
                stat: top_n_players_by_stat(samples, stat, BOOKABLE_TOP_N_BY_STAT.get(stat, 0))
                for stat in PROP_LINES
            }

            for player_name, stats in samples.items():
                confirmed = (team not in lineup) or (player_name in lineup[team])
                # Injury/config blocklist: exclude even when no team sheet is posted.
                if confirmed and _normalize_name(player_name) in {
                    _normalize_name(p) for p in known_outs.get(team, set())
                }:
                    confirmed = False
                for stat, lines in PROP_LINES.items():
                    arr = stats.get(stat)
                    if arr is None:
                        continue
                    # FIX-PLACEABLE-LEGS-AND-210-FLOOR STEP 2.2: collect every
                    # qualifying line for this (player, stat) first, then keep
                    # at most ONE unpriced line (the highest-prob -- "best
                    # line"), since a book doesn't post a near-lock 15+ line
                    # AND a 25+ line on the same gun mid. A line with a real
                    # book price is exempt -- always kept, it's a confirmed
                    # market regardless of how many others are priced too.
                    qualifying = []
                    for line in lines:
                        mask = arr >= line
                        prob = prob_event(mask)
                        prob = apply_prop_calibration(prop_calibrators, stat, line, prob)
                        if not (LEG_PROB_MIN < prob < LEG_PROB_MAX):
                            continue
                        name = f"{player_name} {line}+ {stat}"
                        # "-" (under) side is optional, manual-entry only --
                        # not in the template, but recognised so typing one
                        # in for a two-way devig (STEP 1.3) doesn't trigger
                        # an unmatched-key warning.
                        under_name = f"{player_name} {line}- {stat}"
                        over_odds = odds_book.get(name)
                        under_odds = odds_book.get(under_name)
                        predictions.append({"match_id": match_id, "market": f"player_{stat}",
                                            "subject": player_name, "line": line, "prob": prob})
                        priced = over_odds is not None or under_odds is not None
                        # A leg with NO real price is only kept if it's a
                        # realistic bookmaker market (STEP 1.2) -- a posted
                        # price always wins regardless of the menu, since
                        # it's bettable by definition. Still recorded above
                        # for grading (predictions.csv) even when filtered
                        # out of the live ladder here.
                        if not priced and not is_bookable_model_only_leg(
                                stat, line, player_name, roles.get(player_name), team_top_by_stat[stat]):
                            continue
                        qualifying.append({"line": line, "prob": prob, "mask": mask, "name": name,
                                           "under_name": under_name, "over_odds": over_odds,
                                           "under_odds": under_odds, "priced": priced})
                    for q in select_ladder_lines(qualifying):
                        line, prob, mask = q["line"], q["prob"], q["mask"]
                        name, under_name = q["name"], q["under_name"]
                        over_odds, under_odds = q["over_odds"], q["under_odds"]
                        # Pull the calibrated model prob toward the devigged
                        # market prob (model-upgrade audit Phase 4 STEP 2.1)
                        # -- price/classify the leg on the BLENDED prob, not
                        # the raw model prob, whenever a price exists.
                        blended_prob = prob
                        devig_prob = devig_label = None
                        if over_odds or under_odds:
                            devig_prob, devig_label = devig_prop_leg(over_odds, under_odds)
                            blended_prob = market_anchored_prob(
                                prob, fair_odds(devig_prob), PROP_MARKET_BLEND_WEIGHT)
                        leg = LegCandidate(name, match_id, f"player_{stat}", player_name, blended_prob,
                                           over_odds if over_odds else fair_odds(blended_prob),
                                           confirmed=confirmed, mask=mask)
                        match_legs.append(leg)
                        priceable_names.append(name)
                        known_input_keys.update([name, under_name])
                        if devig_prob is not None:
                            priced_legs.append({
                                "name": name, "model_prob": prob, "book_odds": over_odds,
                                "devig_prob": devig_prob, "devig_label": devig_label,
                                "blended_prob": blended_prob,
                                "edge_pct": leg.edge_pct, "classification": leg.classification,
                            })

        # h2h "to win" legs excluded from the SGM pool: they're near-lock near-fair
        # and dominate combos without adding value. Total-points O/U legs stay in.
        ladder_legs = [l for l in match_legs if l.market != "h2h"]
        sgms = search_match_sgms(ladder_legs, odds_book=odds_book,
                                 corr_gain_haircut=corr_gain_haircut, multi_calibrator=multi_cal)
        # FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART C: a second ladder selected
        # and priced on REAL book odds (Sportsbet/--odds) from the same leg
        # pool -- [] when nothing in this match is priced.
        market_sgms = search_market_sgms(ladder_legs, odds_book=odds_book,
                                         corr_gain_haircut=corr_gain_haircut)
        # Stage 2A: compute per-rung units. multis_records built AFTER the loop,
        # post round-cap so units/tag/$/Stake% always agree.
        for r in sgms + market_sgms:
            r.update(_units_fields(r))
        # Pull 'Em: use PointsBet menu if available, else fall back to odds_book.
        # pb_menu=None -> no file -> use odds_book; pb_menu={} -> all-null -> UNAVAILABLE.
        if pb_menu is not None:
            pull_em = build_pull_em_sgm(
                ladder_legs, odds_book=odds_book, pointsbet_menu=pb_menu
            )
        elif odds_book:
            pull_em = build_pull_em_sgm(ladder_legs, odds_book=odds_book)
        else:
            pull_em = None
        if pull_em and not pull_em.get("no_valid_combo"):
            pe_units = _pull_em_units_fields(pull_em)
            pull_em.update(pe_units)
        # Collect all pull_em-eligible leg names for the per-round template
        # (includes all lines across all players, not just the selected 4).
        from afl_bot.config import PULL_EM_ELIGIBLE_MARKETS as _PE_MKTS
        _pe_eligible_names = [
            l.name for l in ladder_legs
            if l.market in _PE_MKTS or l.market == "disposals"
        ]
        matches.append({
            "header": header, "projections": projections,
            "sgms": sgms, "market_sgms": market_sgms, "priced_legs": priced_legs,
            "n_legs": len(match_legs),     # so the report can explain an empty ladder
            "pull_em": pull_em,
            # Context for post-loop multis_records building + template generation.
            "_home_name": home_name,
            "_away_name": away_name,
            "_leg_by_name": {l.name: l for l in match_legs},
            "_greasiness": greasiness,
            "_pe_eligible_names": _pe_eligible_names,
        })

    # Warn on odds-file keys that never matched a priceable leg (a typo = a
    # leg silently dropped, round-2 §7.4 / model-upgrade audit Phase 4 STEP 1.1).
    unmatched = [k for k in odds_book if not k.startswith("_") and k not in known_input_keys]
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} odds key(s) matched no priceable leg "
              f"(typo? player not in pool/lineup?): {', '.join(unmatched)}", file=sys.stderr)

    # FIX-PULLEM-MENU-AND-STAKE-COLUMNS Part B: apply per-round cap across all
    # staked rungs in all matches (lowest-EV first), then build multis_records from
    # the final units. This ensures units, units_tag, Stake%, and $ always agree.
    _apply_round_cap(matches)
    for m in matches:
        _home = m["_home_name"]
        _away = m["_away_name"]
        _leg_by_name = m["_leg_by_name"]
        _grs = m["_greasiness"]
        for _ladder_label, _rungs in (("model", m["sgms"]), ("sportsbet", m["market_sgms"])):
            for r in _rungs:
                multis_records.append(_rung_to_json(
                    r, _ladder_label, year, round_no, _home, _away, _leg_by_name, odds_book,
                    greasiness=_grs))
        pe = m.get("pull_em")
        if pe and not pe.get("no_valid_combo"):
            h = _home.replace(" ", "_")
            a = _away.replace(" ", "_")
            multis_records.append({
                "id": f"{year}-r{round_no}-{h}-{a}-pull_em",
                "year": year,
                "round": round_no,
                "game": f"{_home} vs {_away}",
                "ladder": "pull_em",
                "book": "pointsbet",
                "leg_names": pe["leg_names"],
                "anchor_names": pe["anchor_names"],
                "booster_name": pe["booster_name"],
                "anchor_probs": pe["anchor_probs"],
                "booster_prob": pe["booster_prob"],
                "book_odds_per_leg": pe["book_odds_per_leg"],
                "book_combo": pe["book_combo"],
                "option_ev": pe["option_ev"],
                "option_ev_breakdown": pe["option_ev_breakdown"],
                "pull_decision_rule": pe["pull_decision_rule"],
                "units": pe.get("units", 0.0),
                "units_tag": pe.get("units_tag", "NO BET"),
            })

    _lineup_note = (
        f" Lineup: {lineup_source} — {n_auto_excluded} player(s) excluded as not named to play."
        if lineup_source else ""
    )
    proj_note = (
        f"_Projections = last {PLAYER_FORM_WINDOW} games (EWMA) × expected TOG. "
        f"Players flagged returning from injury capped at {int(TOG_RETURN_DEFAULT * 100)}% TOG; "
        f"{n_tog_overrides} player(s) had a lineup TOG override this run.{_lineup_note}_"
    )
    md = render_markdown(year, round_no, matches, has_odds=bool(odds_book),
                         odds_note=odds_note, sportsbet_note=sportsbet_note,
                         proj_note=proj_note, multis_only=multis_only)
    out_dir = ROOT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{year}_r{round_no}_report.md"
    atomic_write_text(out_path, md)
    # Machine-readable predictions sidecar so the round can be graded later (§10.5).
    pred_path = out_dir / f"{year}_r{round_no}_predictions.csv"
    _csv_buf = io.StringIO()
    pd.DataFrame(predictions).to_csv(_csv_buf, index=False)
    atomic_write_text(pred_path, _csv_buf.getvalue())
    # Stage 2A: machine-readable multis JSON for the dashboard.
    multis_path = out_dir / f"{year}_r{round_no}_multis.json"
    multis_payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "year": year,
        "round": round_no,
        "records": multis_records,
    }
    atomic_write_text(multis_path, json.dumps(multis_payload, indent=2))
    # PointsBet Pull 'Em template: ALL eligible legs across all matches (not just
    # the selected 4) so Ben can fill in any line PointsBet actually offers.
    # Preserve prices already entered; only add new null entries for new legs.
    pb_template: dict[str, float | None] = {}
    try:
        if _pb_menu_path.exists():
            pb_template = json.loads(_pb_menu_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    _any_pe_eligible = False
    for m in matches:
        for name in m.get("_pe_eligible_names", []):
            _any_pe_eligible = True
            if name not in pb_template:
                pb_template[name] = None
    if _any_pe_eligible:
        atomic_write_text(_pb_menu_path, json.dumps(pb_template, indent=2))
    # Odds template (model-upgrade audit Phase 4 STEP 1.2): every priceable
    # leg's exact name -> null, copy-paste-able into a fresh --odds file.
    template_path = out_dir / f"{year}_r{round_no}_odds_template.json"
    atomic_write_text(template_path, json.dumps(build_odds_template(priceable_names), indent=2))
    # Snapshot the filled --odds file (model-upgrade audit Phase 4 STEP 2.4)
    # so the repo accumulates its own prop-odds history over time -- the
    # exact archive STEP 2.2 says doesn't exist yet, the cheapest path to
    # one day fitting PROP_MARKET_BLEND_WEIGHT for real.
    odds_snapshot_path = None
    if manual:
        odds_snapshot_path = out_dir / f"{year}_r{round_no}_odds.json"
        atomic_write_text(odds_snapshot_path, json.dumps(manual, indent=2))
    sys.stdout.buffer.write(md.encode("utf-8", errors="replace") + b"\n")
    snapshot_note = f" | odds snapshot: {odds_snapshot_path}" if odds_snapshot_path else ""
    print(f"\n[saved to {out_path} | predictions: {pred_path} | "
          f"multis: {multis_path} | odds template: {template_path}{snapshot_note}]")


def grade_round(year: int, round_no: int) -> None:
    """Grade a completed round (§10.5): score every saved prediction against
    what actually happened, append to reports/calibration_log.csv, and print the
    round + cumulative calibration. Feeds Section 2's calibration work."""
    out_dir = ROOT_DIR / "reports"
    pred_path = out_dir / f"{year}_r{round_no}_predictions.csv"
    if not pred_path.exists():
        print(f"No predictions file {pred_path}; run round-report for {year} r{round_no} first.",
              file=sys.stderr)
        return
    preds = pd.read_csv(pred_path)

    client = SquiggleClient()
    games = client.get_completed_games(year)
    games = games[games["round"] == round_no]
    if games.empty:
        print(f"{year} round {round_no} is not completed yet — nothing to grade.", file=sys.stderr)
        return
    # Actual player stats for the round, matched by the REAL round number.
    # Past seasons: Fryzigg raw `match_round` (its to_player_log round is a
    # chronological ordinal, §7.2, and its unixtime is unreliable). Current
    # season: DFS, which carries the real round via the Squiggle join.
    player_round = pd.DataFrame(columns=["player", "disposals", "goals", "marks", "tackles"])
    try:
        from afl_bot.data.fryzigg import fetch_fryzigg_player_stats
        raw = fetch_fryzigg_player_stats()
        raw = raw.assign(_year=pd.to_datetime(raw["match_date"]).dt.year,
                         _player=(raw["player_first_name"].str.strip() + " "
                                  + raw["player_last_name"].str.strip()))
        rnd = raw[(raw["_year"] == year) & (raw["match_round"].astype(str) == str(round_no))]
        if not rnd.empty:
            player_round = rnd.rename(columns={"_player": "player"})
    except Exception:  # noqa: BLE001
        pass
    if player_round.empty:
        try:
            from afl_bot.data.dfs_australia import fetch_player_stats
            from afl_bot.data.dfs_australia import to_player_log as _dfs_to_log
            dfs = _dfs_to_log(fetch_player_stats(), games)
            player_round = dfs[dfs["round"] == round_no]
        except Exception:  # noqa: BLE001
            pass

    # actual H2H/total per match (totals keyed by match_id)
    h2h_actual, total_actual = {}, {}
    for _, g in games.iterrows():
        h2h_actual[g["hteam"]] = int(g["hscore"] > g["ascore"])
        h2h_actual[g["ateam"]] = int(g["ascore"] > g["hscore"])
        total_actual[f"{year}_r{round_no}_{g['hteam']}_v_{g['ateam']}"] = g["hscore"] + g["ascore"]
    player_stat = {  # (player, stat) -> actual value that round
        (r["player"], stat): r[stat]
        for _, r in player_round.iterrows() for stat in ("disposals", "goals", "marks", "tackles")
    }

    graded = []
    for _, p in preds.iterrows():
        market, subject, line = p["market"], p["subject"], p["line"]
        if market == "h2h":
            if subject not in h2h_actual:
                continue
            actual = h2h_actual[subject]
        elif market == "total_points":
            tot = total_actual.get(p["match_id"])
            actual = int(tot >= float(line)) if tot is not None else None
        elif market.startswith("player_"):
            stat = market.split("_", 1)[1]
            val = player_stat.get((subject, stat))
            actual = int(val >= float(line)) if val is not None else None
        else:
            actual = None
        if actual is None:
            continue
        graded.append({"year": year, "round": round_no, "market": market, "subject": subject,
                       "line": line, "prob": float(p["prob"]), "actual": actual})

    if not graded:
        print("No predictions could be matched to actuals (player names/rounds).", file=sys.stderr)
        return
    graded_df = pd.DataFrame(graded)

    log_path = out_dir / "calibration_log.csv"
    if log_path.exists():
        prev = pd.read_csv(log_path)
        combined = pd.concat([prev[prev["round"] != round_no] if "round" in prev else prev, graded_df],
                             ignore_index=True)
    else:
        combined = graded_df
    _cal_buf = io.StringIO()
    combined.to_csv(_cal_buf, index=False)
    atomic_write_text(log_path, _cal_buf.getvalue())

    from afl_bot.backtest.walkforward import brier_score, log_loss
    probs = graded_df["prob"].to_numpy()
    actuals = graded_df["actual"].to_numpy(dtype=float)
    print(f"=== Graded {year} Round {round_no}: {len(graded_df)} predictions ===")
    print(f"  log loss {log_loss(probs, actuals):.4f} | brier {brier_score(probs, actuals):.4f} "
          f"| mean pred {probs.mean():.3f} | hit rate {actuals.mean():.3f}")
    cum_probs = combined["prob"].to_numpy()
    cum_act = combined["actual"].to_numpy(dtype=float)
    print(f"  cumulative ({len(combined)} preds across {combined['round'].nunique()} rounds): "
          f"log loss {log_loss(cum_probs, cum_act):.4f} | brier {brier_score(cum_probs, cum_act):.4f}")
    print(f"  [appended to {log_path}]")


def grade_multis(years: list[int], rounds: list[int] | None, n_sims: int,
                 with_calibration: bool = True, calibration_source: str = "proxy",
                 all_candidates: bool = False, lcb_z: float = 0.0, price_shrink: float = 0.0,
                 multi_calibration: bool = False, corr_gain_diag: bool = False,
                 corr_gain_haircut: float = CORR_GAIN_HAIRCUT,
                 prop_halflife: float = PROP_EWMA_HALFLIFE) -> None:
    """Walk-forward backtest of the 3-leg same-game-multi ladder actually bet
    (model-upgrade audit Phase 1.1, expanded by Phase 2.5 steps 2-3 and Phase
    3.1's calibration-source choice): for each completed round across every
    year in `years`, reconstruct the ladder `round-report` would have built
    from data strictly before that round, and grade the selected rungs'
    joint probability against what actually happened. Until this reliability
    curve is flat, treat multi prices as unproven, not evidence (see
    MODEL-UPGRADE-INSTRUCTIONS.md).

    `with_calibration=True` (default) additionally fits prop calibrators once
    per year (on seasons strictly before it, written to a backtest-only cache
    dir so this never touches the live `prop_calibrators.json` `round-report`
    uses) and reports a second, calibration-ON reliability curve alongside
    the raw one -- Phase 2.5's "is the overconfidence in the legs, not the
    correlation?" check.

    `calibration_source`: `"proxy"` (default) fits on
    `walk_forward_prop_predictions`'s simplified shrunk-EWMA NB marginal --
    Phase 2.5's original calibration-ON mode. `"sim"` instead fits on
    `walk_forward_sim_prop_predictions`'s REAL sim-pipeline probabilities
    (model-upgrade audit Phase 3.1's "calibrate against the real sim
    output") -- much more expensive (a full per-round sim pass per prior
    season, not a closed-form NB marginal), but the acceptance test for
    Phase 3's whole premise: if `"sim"`'s calibrated curve is flatter than
    `"proxy"`'s, calibration fidelity (not just calibration's presence) was
    the missing lever.

    `all_candidates=True` (model-upgrade audit Phase 3.5, default False)
    additionally grades the FULL un-selected SGM candidate population
    (`walk_forward_sgm_candidate_predictions`) and prints its reliability
    curve alongside the selected-rungs one -- the optimizer's-curse check:
    is `search_match_sgms`'s selection (not the sim's calibration) the
    source of the multi-ladder overconfidence? Re-runs the sim a second time
    (its own walk-forward loop), so roughly doubles runtime when set.

    `lcb_z`/`price_shrink` (Phase 3.5, default 0.0 = off) pass through to
    `search_match_sgms`'s selection-haircut params (see its docstring) --
    prototype fixes for the optimizer's curse, opt-in so the unhaircut
    default behaviour is unchanged. Both failed real-data acceptance (see
    README's Phase 3.5 section).

    `multi_calibration=True` (model-upgrade audit Phase 3.6, default False)
    fits a selection-level isotonic calibrator on `MULTI_CALIBRATION_LOOKBACK`
    seasons strictly before each eval year (a backtest-only cache dir, never
    touching the live `multi_calibrator.json` `round-report` uses) and
    reports a third, "MULTI-CALIBRATED" reliability curve -- the acceptance
    test for whether fitting directly on the selected-rung track record (not
    trying to fix the selection mechanism, which Phase 3.5 found doesn't
    work) actually flattens the curve and beats the unhaircut baseline log
    loss.

    `corr_gain_diag=True` (PHASE-4-CODE-PLAN.md's parked, no-odds-needed
    diagnostic, default False) additionally compares the sim's `corr_gain`
    (`joint_prob - naive_product`, both from the correlated sim) to the
    EMPIRICAL corr_gain (actual joint hit-rate minus the product of pooled
    actual per-leg hit-rates), bucketed by predicted `joint_prob`, on the
    SELECTED rungs. Diagnostic only -- reports the gap, applies no fix. Also
    prints the in-sample log-loss-minimizing `corr_gain_haircut` value
    (`fit_corr_gain_haircut`) fit directly on the run's own
    `naive_product`/`corr_gain` columns -- no second sim pass needed.

    `corr_gain_haircut` (default `CORR_GAIN_HAIRCUT` = 0.0, mirroring
    `round-report`'s live default) passes straight through to
    `walk_forward_multi_predictions`/`search_match_sgms`'s same param --
    reprices every SELECTED rung as `naive_product + corr_gain_haircut *
    corr_gain` instead of the raw sim `joint_prob`, so SELECTED's own
    log-loss/reliability-curve numbers reflect the haircut by default. Pass
    1.0 to recover the original raw/unhaircut baseline for comparison --
    see README's "corr_gain haircut" section for the OOS fit/acceptance
    result and the closing writeup."""
    from afl_bot.backtest.multis import (
        corr_gain_diagnostic,
        fit_corr_gain_haircut,
        load_or_fit_multi_calibrator,
        multi_calibration_report,
        multi_reliability_curve,
        walk_forward_multi_predictions,
        walk_forward_sgm_candidate_predictions,
        walk_forward_sim_prop_predictions,
    )
    from afl_bot.backtest.props import fit_prop_calibrators, load_or_fit_prop_calibrators

    client = SquiggleClient()
    fetch_years = sorted(set(_history_years(min(years), lookback=7)) | set(years))
    games = pd.concat([client.get_completed_games(y) for y in fetch_years], ignore_index=True)
    games = games[games["year"] <= max(years)]
    if games.empty:
        print("No historical data available.", file=sys.stderr)
        return
    player_log = load_player_log(games, prefer_real=True)
    backtest_cal_cache = CACHE_DIR / "grade_multis_backtest"
    multi_cal_season_cache: dict[int, pd.DataFrame] = {}

    all_preds = []
    for year in years:
        prop_calibrators = None
        if with_calibration:
            eval_start_year = year - PROP_CALIBRATION_LOOKBACK
            if calibration_source == "sim":
                # One walk_forward_sim_prop_predictions call per prior season
                # (it takes a single eval_year), concatenated -- mirrors the
                # multi-season window load_or_fit_prop_calibrators's proxy
                # path covers via eval_start_year. games/player_log already
                # span back further than eval_start_year, and
                # _truncate_before_round re-enforces strictly-before-round
                # per call, so passing the full frames here leaks nothing
                # into a given cal_year's predictions.
                sim_frames = [
                    walk_forward_sim_prop_predictions(
                        games, player_log, eval_year=cal_year, rounds=None,
                        n_sims=n_sims, seed=1)
                    for cal_year in range(eval_start_year, year)
                ]
                sim_preds = pd.concat(sim_frames, ignore_index=True) if sim_frames else pd.DataFrame()
                prop_calibrators = fit_prop_calibrators(sim_preds) if not sim_preds.empty else None
            else:
                prior_log = player_log[player_log["year"] < year]
                prop_calibrators = load_or_fit_prop_calibrators(
                    prior_log, eval_start_year=eval_start_year,
                    cache_dir=backtest_cal_cache, force_refresh=True,
                    halflife=prop_halflife) or None
        preds = walk_forward_multi_predictions(
            games, player_log, eval_year=year, rounds=rounds, n_sims=n_sims,
            prop_calibrators=prop_calibrators, lcb_z=lcb_z, price_shrink=price_shrink,
            corr_gain_haircut=corr_gain_haircut)
        if multi_calibration and not preds.empty:
            multi_cal = load_or_fit_multi_calibrator(
                games, player_log, eval_start_year=year - MULTI_CALIBRATION_LOOKBACK,
                eval_end_year=year, n_sims=n_sims, cache_dir=backtest_cal_cache,
                force_refresh=True, seed=1, season_preds_cache=multi_cal_season_cache)
            if multi_cal is not None:
                preds = preds.assign(
                    multi_calibrated_joint_prob=multi_cal.predict(preds["joint_prob"].to_numpy()))
        all_preds.append(preds)
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if preds.empty:
        print("No graded multi predictions — rounds not completed yet, or every match had "
              "fewer than 3 candidate legs.", file=sys.stderr)
        return

    haircut_note = f" (lcb_z={lcb_z}, price_shrink={price_shrink})" if (lcb_z or price_shrink) else ""
    report = multi_calibration_report(preds)
    print(f"=== Multi (SGM) walk-forward backtest: years {years}, "
          f"{preds.groupby('year')['round'].nunique().to_dict()} rounds/year ===")
    print(f"  SELECTED{haircut_note}   n={report['n']} | log loss {report['log_loss']:.4f} | "
          f"brier {report['brier']:.4f} | mean pred {report['mean_pred']:.3f} | "
          f"actual hit rate {report['hit_rate']:.3f}")
    print("  SELECTED reliability (mean predicted vs actual hit rate per bucket):")
    for _, row in multi_reliability_curve(preds).iterrows():
        print(f"    {row['bucket']}: pred {row['mean_pred']:.3f} | actual {row['actual_rate']:.3f} "
              f"| n={int(row['n'])}")

    if corr_gain_diag:
        print("\n  CORR-GAIN DIAGNOSTIC (sim vs empirical, per predicted-prob bucket):")
        for _, row in corr_gain_diagnostic(preds).iterrows():
            print(f"    {row['bucket']}: sim corr_gain {row['sim_corr_gain']:+.3f} "
                  f"(joint {row['sim_joint']:.3f} - naive {row['sim_naive']:.3f}) | "
                  f"empirical corr_gain {row['empirical_corr_gain']:+.3f} "
                  f"(actual {row['actual_joint']:.3f} - naive {row['empirical_naive']:.3f}) | "
                  f"gap {row['gap']:+.3f} | n={int(row['n'])}")
        fitted = fit_corr_gain_haircut(preds)
        print(f"  Fitted corr_gain_haircut (in-sample, log-loss-minimizing on THIS run's own "
              f"sample): {fitted:.2f} -- re-run with --corr-gain-haircut for an "
              f"out-of-sample/repriced check.")

    if with_calibration and "calibrated_joint_prob" in preds.columns:
        cal_report = multi_calibration_report(preds, column="calibrated_joint_prob")
        print(f"  CALIBRATED n={cal_report['n']} | log loss {cal_report['log_loss']:.4f} | "
              f"brier {cal_report['brier']:.4f} | mean pred {cal_report['mean_pred']:.3f} | "
              f"actual hit rate {cal_report['hit_rate']:.3f}")
        print("  CALIBRATED reliability (mean predicted vs actual hit rate per bucket):")
        for _, row in multi_reliability_curve(preds, column="calibrated_joint_prob").iterrows():
            print(f"    {row['bucket']}: pred {row['mean_pred']:.3f} | actual {row['actual_rate']:.3f} "
                  f"| n={int(row['n'])}")

    if multi_calibration and "multi_calibrated_joint_prob" in preds.columns:
        mc_report = multi_calibration_report(preds, column="multi_calibrated_joint_prob")
        print(f"  MULTI-CALIBRATED n={mc_report['n']} | log loss {mc_report['log_loss']:.4f} | "
              f"brier {mc_report['brier']:.4f} | mean pred {mc_report['mean_pred']:.3f} | "
              f"actual hit rate {mc_report['hit_rate']:.3f}")
        print("  MULTI-CALIBRATED reliability (mean predicted vs actual hit rate per bucket):")
        for _, row in multi_reliability_curve(preds, column="multi_calibrated_joint_prob").iterrows():
            print(f"    {row['bucket']}: pred {row['mean_pred']:.3f} | actual {row['actual_rate']:.3f} "
                  f"| n={int(row['n'])}")

    if all_candidates:
        cand_preds = pd.concat(
            [walk_forward_sgm_candidate_predictions(games, player_log, eval_year=year, rounds=rounds,
                                                     n_sims=n_sims)
             for year in years],
            ignore_index=True)
        if cand_preds.empty:
            print("\nALL-CANDIDATES: no graded candidate combos.", file=sys.stderr)
        else:
            cand_report = multi_calibration_report(cand_preds)
            print(f"\n  ALL-CANDIDATES n={cand_report['n']} | log loss {cand_report['log_loss']:.4f} | "
                  f"brier {cand_report['brier']:.4f} | mean pred {cand_report['mean_pred']:.3f} | "
                  f"actual hit rate {cand_report['hit_rate']:.3f}")
            print("  ALL-CANDIDATES reliability (mean predicted vs actual hit rate per bucket):")
            for _, row in multi_reliability_curve(cand_preds).iterrows():
                print(f"    {row['bucket']}: pred {row['mean_pred']:.3f} | actual {row['actual_rate']:.3f} "
                      f"| n={int(row['n'])}")


def sweep_halflife_command(years: list[int], halflives: list[float],
                           cal_lookback: int = 4) -> None:
    """Diagnostic sweep of PROP_EWMA_HALFLIFE candidates (EXPERIMENT-FORM-WINDOW-HALFLIFE).

    For each halflife value, runs a fully OOS prop walk-forward on ``years``
    and reports calibrated log loss / Brier / ECE / high-bucket gap. No sim
    is run; this uses the fast closed-form NB marginal. The winner from this
    table should then be verified with ``grade-multis --halflife <HL>`` before
    updating ``PROP_EWMA_HALFLIFE`` in config.py.

    Decision rule: adopt a new halflife only when it improves BOTH prop log
    loss AND calibration (ECE / high-bucket gap), consistently across all
    ``years``. A tiny improvement in one season is not enough — keep HL=6."""
    from afl_bot.backtest.props import prop_halflife_sweep

    client = SquiggleClient()
    fetch_years = sorted(set(_history_years(min(years), lookback=7)) | set(years))
    games = pd.concat([client.get_completed_games(y) for y in fetch_years], ignore_index=True)
    games = games[games["year"] <= max(years)]
    if games.empty:
        print("No historical data available.", file=sys.stderr)
        return
    player_log = load_player_log(games, prefer_real=True)

    print(f"PROP_EWMA_HALFLIFE sweep — halflives {halflives} on years {years}")
    print(f"(calibrators fit on {cal_lookback}-season lookback; each eval year is strictly OOS)\n")

    result = prop_halflife_sweep(
        player_log, eval_years=years, halflives=halflives, cal_lookback=cal_lookback)
    if result.empty:
        print("No predictions generated — check data availability.", file=sys.stderr)
        return

    default_hl = float(PROP_EWMA_HALFLIFE)
    hdr = f"{'HL':>4} | {'n':>7} | {'log_loss':>8} | {'brier':>6} | {'ECE':>6} | {'hi_gap':>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in result.iterrows():
        marker = "  <- CURRENT" if r["halflife"] == default_hl else ""
        hi_gap = f"{r['high_bucket_gap']:+.4f}" if not np.isnan(r["high_bucket_gap"]) else "   nan"
        print(f"{int(r['halflife']):>4} | {int(r['n']):>7} | {r['log_loss']:>8.4f} | "
              f"{r['brier']:>6.4f} | {r['ece']:>6.4f} | {hi_gap:>8}{marker}")

    best_row = result.loc[result["log_loss"].idxmin()]
    best_hl = best_row["halflife"]
    current_row = result[result["halflife"] == default_hl]
    current_ll = float(current_row["log_loss"].iloc[0]) if not current_row.empty else float("nan")
    best_ll = float(best_row["log_loss"])
    delta = current_ll - best_ll

    print(f"\nBest log loss: HL={int(best_hl)} ({best_ll:.4f}, "
          f"delta vs HL={int(default_hl)}: {delta:+.4f})")
    if best_hl == default_hl:
        print(f"FINDING: HL={int(default_hl)} is already optimal — no change needed.")
    elif delta < 0.001:
        print(f"FINDING: improvement too small (delta={delta:.4f}) for both-year consistency — "
              f"keep HL={int(default_hl)}.")
    else:
        print(f"CANDIDATE: HL={int(best_hl)} shows delta={delta:.4f} improvement in prop log loss.")
        print(f"  Next step: verify with grade-multis to confirm no regression:")
        ys = ",".join(str(y) for y in years)
        print(f"    python -m afl_bot grade-multis --year {ys} --halflife {int(best_hl)}")
        print(f"    python -m afl_bot grade-multis --year {ys}  # baseline (HL={int(default_hl)})")
        print(f"  Adopt only if multi log loss does not worsen on BOTH years.")
    print(f"\nReactivity caveat: longer half-life = steadier projections, "
          f"but slower to catch a genuine role change (e.g. Duursma-at-HF).")


def fit_correlations_command(through: int) -> None:
    """Fit `SCORE_SHOT_CORRELATION` / `PACE_SIGMA` / `SHARE_CONCENTRATION` /
    `SHOT_DISPERSION` / `TEAM_STAT_DISPERSION` from history up to and
    including `through` (model-upgrade audit Phase 2.1) and write a versioned
    `correlation_params.json` artifact that `round-report`/`run-round` can
    opt into (mirrors `fit`/`elo_params.json`)."""
    from afl_bot.backtest.correlations import fit_correlation_params

    client = SquiggleClient()
    games = pd.concat([client.get_completed_games(y) for y in _history_years(through, lookback=7)],
                      ignore_index=True)
    games = games[games["year"] <= through]
    if games.empty:
        print("No completed games to fit on.", file=sys.stderr)
        return
    player_log = load_player_log(games, prefer_real=True)
    artifact = fit_correlation_params(games, player_log, train_end_year=through)
    print(json.dumps(artifact, indent=2))
    print("\n[wrote correlation_params.json -> opt into it with "
          "afl_bot.backtest.correlations.load_fitted_correlation_params()]", file=sys.stderr)


def fit_command(through: int, use_optuna: bool, n_trials: int) -> None:
    """Re-tune Elo hyperparameters on completed seasons through ``through`` and
    write a versioned ``elo_params.json`` artifact (round-2 §6.2) that run-round /
    round-report then pick up — instead of hand-editing config defaults."""
    client = SquiggleClient()
    games = pd.concat([client.get_completed_games(y) for y in range(through - 7, through + 1)],
                      ignore_index=True)
    games = games[games["year"] <= through]
    if games.empty:
        print("No completed games to fit on.", file=sys.stderr)
        return
    artifact = fit_elo_params(games, train_end_year=through - 2, eval_start_year=through - 1,
                              use_optuna=use_optuna, n_trials=n_trials)
    print(json.dumps(artifact, indent=2))
    print("\n[wrote elo_params.json -> run-round/round-report will use these tuned params]",
          file=sys.stderr)


def prop_calibration_check(
    eval_years: list[int],
    out_path: str | None = None,
    multis_year: int = 2026,
    multis_round: int | None = None,
) -> None:
    """OOS prop calibration check: model calibrated prob vs actual hit rate.

    For every (player, game, stat, line) in eval_years, generates walk-forward
    NB predictions (no leakage within each game's EWMA), applies year-specific
    calibrators fitted on strictly prior data, then reports per-market and
    per-prob-band calibration.  Overlays book raw-implied from the most recent
    saved sportsbet multis.json to classify each market.

    Diagnostic only -- no model/config change.
    """
    import statistics as _stats
    import textwrap

    from afl_bot.backtest.props import (
        apply_prop_calibration,
        fit_prop_calibrators,
        walk_forward_prop_predictions,
    )
    from afl_bot.config import PROP_CALIBRATION_LOOKBACK

    client = SquiggleClient()
    all_fetch_years = sorted(set(
        y for ey in eval_years for y in _history_years(ey, lookback=8)
    ))
    print(f"[prop-cal-check] loading games {min(all_fetch_years)}-{max(all_fetch_years)} ...",
          file=sys.stderr)
    games = pd.concat(
        [client.get_completed_games(y) for y in all_fetch_years], ignore_index=True)
    games = games[games["year"] <= max(eval_years)]

    print("[prop-cal-check] loading player log ...", file=sys.stderr)
    player_log = load_player_log(games, prefer_real=True)

    # ── Per-eval-year walk-forward predictions (leakage-safe) ───────────────
    print("[prop-cal-check] running walk-forward prop predictions ...", file=sys.stderr)
    raw_all = walk_forward_prop_predictions(player_log, eval_start_year=min(eval_years))
    raw_all = raw_all[raw_all["year"].isin(eval_years)].reset_index(drop=True)
    if raw_all.empty:
        print("[prop-cal-check] no predictions generated for eval years.", file=sys.stderr)
        return

    # ── Year-by-year calibration (calibrators fitted on prior years only) ───
    _prop_cal_cache = CACHE_DIR / "prop_cal_check"
    _prop_cal_cache.mkdir(parents=True, exist_ok=True)

    frames_cal: list[pd.DataFrame] = []
    for eval_year in sorted(eval_years):
        prior_log = player_log[player_log["year"] < eval_year]
        cal_start = eval_year - PROP_CALIBRATION_LOOKBACK
        print(f"[prop-cal-check] fitting calibrators for {eval_year} "
              f"(training on {cal_start}-{eval_year-1}) ...", file=sys.stderr)
        prior_raw = walk_forward_prop_predictions(prior_log, eval_start_year=cal_start)
        calibrators = fit_prop_calibrators(prior_raw) if not prior_raw.empty else {}

        year_preds = raw_all[raw_all["year"] == eval_year].copy()
        year_preds["cal_prob"] = year_preds.apply(
            lambda r: apply_prop_calibration(calibrators, r["stat"], r["line"], r["prob"]),
            axis=1,
        )
        frames_cal.append(year_preds)

    preds = pd.concat(frames_cal, ignore_index=True)

    # ── Book comparison from saved sportsbet multis.json ────────────────────
    # Store per-leg (hit_prob_band, market) -> [book_implied] so we can compare
    # within the same probability band (not overall averages, which are skewed
    # by different lines).  hit_prob is the model calibrated prob at the time of
    # the report, equivalent to our cal_prob in the walk-forward sample.
    BANDS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
    book_by_mkt_band: dict[tuple[str, int], list[float]] = {}
    # Overall per-market book-implied list (for reference, NOT used in classification)
    book_by_market: dict[str, list[float]] = {}
    # Mean per-market gap from ev-diagnostic (model hit_prob vs 1/book_odds at priced lines)
    ev_diag_gap: dict[str, float] = {}

    if multis_round is not None:
        multis_path = ROOT_DIR / "reports" / f"{multis_year}_r{multis_round}_multis.json"
    else:
        import glob as _glob
        candidates = sorted(_glob.glob(
            str(ROOT_DIR / "reports" / f"{multis_year}_r*_multis.json")))
        multis_path = Path(candidates[-1]) if candidates else None  # type: ignore[arg-type]

    if multis_path and multis_path.exists():
        multis_data = json.loads(multis_path.read_text())
        for rung in multis_data:
            if rung.get("ladder") != "sportsbet":
                continue
            for leg in rung.get("legs", []):
                bo = leg.get("book_odds")
                hp = leg.get("hit_prob")
                if bo is None or hp is None:
                    continue
                mkt = leg.get("market", "")
                p_imp = 1.0 / bo
                book_by_market.setdefault(mkt, []).append(p_imp)
                # Gap = book_implied - model_hit_prob (positive = model under book)
                ev_diag_gap.setdefault(mkt, [])           # type: ignore[assignment]
                ev_diag_gap[mkt].append(p_imp - hp)       # type: ignore[index]
                # Bucket by the model hit_prob band
                for band_idx, (lo, hi) in enumerate(BANDS):
                    if lo <= hp < hi:
                        book_by_mkt_band.setdefault((mkt, band_idx), []).append(p_imp)
                        break
        multis_note = str(multis_path.name)
        # Average the gap lists
        ev_diag_gap = {k: float(sum(v) / len(v)) for k, v in ev_diag_gap.items()
                       if isinstance(v, list) and v}  # type: ignore[assignment]
    else:
        multis_note = "no sportsbet multis found"
        ev_diag_gap = {}

    def _band_label(lo, hi):
        return f"{lo:.0%}-{hi:.0%}"

    lines_buf: list[str] = []

    def _w(s):
        lines_buf.append(s)

    _w(f"# Prop Calibration Check: Model vs Actual Hit Rate")
    _w(f"")
    _w(f"**Eval years:** {eval_years}")
    _w(f"**Total legs graded:** {len(preds):,}")
    _w(f"**Book comparison source:** {multis_note}")
    _w(f"")
    _w(f"Calibrators are fitted on data strictly prior to each eval year "
       f"(PROP_CALIBRATION_LOOKBACK={PROP_CALIBRATION_LOOKBACK}), so no leakage.")
    _w(f"")
    _w(f"> **Note on book comparison**: `book_implied` is the average of 1/leg_book_odds "
       f"for legs appearing in sportsbet SGM multis. These are only the specific "
       f"high-probability lines the book prices for SGMs - they are NOT comparable to "
       f"the overall calibration mean (which spans all evaluated lines, including long-shot "
       f"lines with low hit rates). The `ev_diag_gap` column is the correct comparable: "
       f"it shows model calibrated prob vs book implied at the SAME specific priced legs "
       f"(from the ev-diagnostic). Use the per-band table for a fair comparison.")
    _w(f"")

    # Overall summary table
    _w(f"## Per-Market Summary\n")
    _w(f"| Market | n | Cal prob | Hit rate | Cal gap | book_minus_model (priced SGM legs) | Class |")
    _w(f"|--------|---|----------|----------|---------|-----------------------------------|-------|")

    verdicts: list[tuple[str, str]] = []
    for stat in ("disposals", "goals", "marks", "tackles"):
        grp = preds[preds["stat"] == stat]
        if grp.empty:
            continue
        cal_probs = grp["cal_prob"].to_numpy()
        actuals = grp["actual"].to_numpy(dtype=float)
        n = len(grp)
        mean_cal = float(cal_probs.mean())
        hit_rate = float(actuals.mean())
        cal_gap = mean_cal - hit_rate   # positive = model over-predicts

        # ev_diag_gap: model hit_prob - book_implied at priced lines (negative = model under book)
        evg = ev_diag_gap.get(stat, float("nan"))
        evg_str = f"{evg:+.3f}" if evg == evg else "n/a"

        # Classify based ONLY on cal_gap (model vs actual, not vs book)
        if cal_gap < -0.05:
            cls = "GENUINELY CONSERVATIVE"
        elif cal_gap > 0.05:
            cls = "OVER-CONFIDENT"
        elif abs(cal_gap) <= 0.02:
            cls = "WELL CALIBRATED"
        else:
            cls = "CLOSE"
        verdicts.append((stat, cls))

        _w(f"| {stat:<11} | {n:>6,} | {mean_cal:.3f} | {hit_rate:.3f} | "
           f"{cal_gap:+.3f} | {evg_str:>35} | {cls} |")

    _w(f"")
    _w(f"*Cal gap: positive = model over-predicts vs actual; negative = model under-predicts.*")
    _w(f"*book_minus_model: positive = book is ABOVE model (book's structural margin); negative = model above book.*")
    _w(f"")

    # Per-market x prob-band with book comparison per band
    _w(f"## Per-Market x Probability Band (with book implied per band)\n")
    _w(f"The three-way comparison works correctly within each probability band, because "
       f"the book's priced legs and the walk-forward sample are in the same prob range.")
    _w(f"")
    for stat in ("disposals", "goals", "marks", "tackles"):
        grp = preds[preds["stat"] == stat]
        if grp.empty:
            continue
        _w(f"### {stat.capitalize()}\n")
        _w(f"| Band | n | Cal prob | Hit rate | Gap | Book implied (SGM) | Book vs actual |")
        _w(f"|------|---|----------|----------|-----|--------------------|----------------|")
        for band_idx, (lo, hi) in enumerate(BANDS):
            band_mask = (grp["cal_prob"] >= lo) & (grp["cal_prob"] < hi)
            band = grp[band_mask]
            if len(band) < 5:
                continue
            bp = float(band["cal_prob"].mean())
            hr = float(band["actual"].mean())
            book_list = book_by_mkt_band.get((stat, band_idx), [])
            book_mean = float(sum(book_list) / len(book_list)) if book_list else float("nan")
            book_vs_actual = f"{book_mean - hr:+.3f}" if book_mean == book_mean else "n/a"
            book_str = f"{book_mean:.3f}" if book_mean == book_mean else "n/a"
            _w(f"| {_band_label(lo, hi)} | {len(band):>5,} | {bp:.3f} | {hr:.3f} | "
               f"{bp-hr:+.3f} | {book_str:>18} | {book_vs_actual:>14} |")
        _w(f"")

    # Calibration_log cross-check (the earlier 0.42-vs-0.46 finding)
    cal_log_path = ROOT_DIR / "reports" / "calibration_log.csv"
    if cal_log_path.exists():
        import csv as _csv
        cal_log_rows = list(_csv.DictReader(cal_log_path.open()))
        prop_rows = [r for r in cal_log_rows if r["market"].startswith("player_")]
        if prop_rows:
            log_mean_pred = _stats.mean(float(r["prob"]) for r in prop_rows)
            log_hit_rate = _stats.mean(float(r["actual"]) for r in prop_rows)
            _w(f"## Cross-check vs calibration_log.csv\n")
            _w(f"calibration_log ({len(prop_rows)} prop rows): "
               f"mean pred = {log_mean_pred:.3f}, actual hit rate = {log_hit_rate:.3f}, "
               f"gap = {log_mean_pred-log_hit_rate:+.3f}")
            _w(f"*(This log uses the model calibrated probs from each live round-report run, "
               f"not the walk-forward reproduced probs above.)*")
            _w(f"")

    # Verdict per market
    _w(f"## Verdict\n")
    for stat, cls in verdicts:
        grp = preds[preds["stat"] == stat]
        cal_probs = grp["cal_prob"].to_numpy()
        actuals = grp["actual"].to_numpy(dtype=float)
        cal_gap = float(cal_probs.mean() - actuals.mean())
        evg = ev_diag_gap.get(stat, float("nan"))

        if cls == "GENUINELY CONSERVATIVE":
            detail = (f"The model under-predicts {stat} hit rates by {abs(cal_gap)*100:.1f}pp "
                      f"on average (calibrated). This is a genuine calibration gap vs actual outcomes. "
                      f"Proposal: re-fit or loosen the {stat} calibrator "
                      f"(raise the isotonic curve's floor for high-probability legs). "
                      f"DO NOT implement in this run.")
        elif cls == "WELL CALIBRATED":
            evg_part = (f" The book prices its {stat} legs ~{abs(evg)*100:.1f}pp above the model "
                        f"on priced SGM legs (ev_diag_gap={evg:+.3f}), which is the book's structural "
                        f"margin, not model error. Accept it; only back legs with clear model edge."
                        if evg == evg else "")
            detail = (f"Model and actual hit rate are closely aligned for {stat} "
                      f"(cal gap {cal_gap:+.3f}).{evg_part} "
                      f"No calibration adjustment needed for this market.")
        elif cls == "CLOSE":
            evg_part = (f" The book prices its {stat} legs ~{abs(evg)*100:.1f}pp above the model "
                        f"on priced SGM legs (ev_diag_gap={evg:+.3f}), consistent with the book's "
                        f"structural margin."
                        if evg == evg else "")
            detail = (f"Model is close to actual hit rate for {stat} (cal gap {cal_gap:+.3f}).{evg_part} "
                      f"Minor calibration drift of {abs(cal_gap)*100:.1f}pp - within normal variance. "
                      f"No immediate action needed.")
        else:
            detail = (f"The model OVER-predicts {stat} hit rates by {cal_gap*100:.1f}pp "
                      f"(calibrated). Watch this market; the existing calibrator may be "
                      f"insufficient. Consider a tighter per-line calibrator.")
        _w(f"**{stat.upper()} - {cls}:** {detail}")
        _w(f"")

    # Overall call
    all_cls = [cls for _, cls in verdicts]
    n_well = sum(1 for c in all_cls if c in ("WELL CALIBRATED", "CLOSE"))
    n_conserv = sum(1 for c in all_cls if c == "GENUINELY CONSERVATIVE")
    n_over = sum(1 for c in all_cls if c == "OVER-CONFIDENT")
    _w(f"## Overall Call\n")
    if n_conserv > 0 and n_over == 0:
        _w(f"**ACTION NEEDED**: {n_conserv} market(s) show genuine conservatism vs actual "
           f"outcomes. Consider re-fitting their calibrators in a follow-up run.")
    elif n_over > 0 and n_conserv == 0:
        _w(f"**WATCH**: {n_over} market(s) are over-confident vs actual outcomes. "
           f"The calibrator may need tightening.")
    elif n_well == len(all_cls):
        _w(f"**NO CALIBRATION ACTION NEEDED.** All {len(all_cls)} markets are well-calibrated "
           f"vs actual hit rates. The observed -EV in SGM multis is attributable to the book's "
           f"structural margin (8-12pp above model on priced legs), not to model conservatism. "
           f"No calibration tightening is warranted. Focus on leg selection quality (ev > threshold) "
           f"to manage the book's edge.")
    else:
        _w(f"Mixed picture across markets. Review individual verdicts above.")
    _w(f"")

    report_text = "\n".join(lines_buf)

    # Save output
    save_path = Path(out_path) if out_path else ROOT_DIR / "reports" / "prop_calibration_check_2024_2025.md"
    atomic_write_text(save_path, report_text)
    print(report_text)
    print(f"\n[saved to {save_path}]", file=sys.stderr)


def ev_diagnostic(year: int, round_no: int) -> None:
    """Print a negative-EV diagnostic for a saved sportsbet multis ladder.

    For every sportsbet rung, computes:
      - per-leg gap = book_raw_implied (1/book_odds) - model_hit_prob
      - book_naive_joint = product of (1/leg_book_odds) per rung
      - EV split: leg-disagreement component vs structural (margin + SGM loading)

    Diagnostic only — no model change.
    """
    import statistics

    multis_path = ROOT_DIR / "reports" / f"{year}_r{round_no}_multis.json"
    if not multis_path.exists():
        print(f"ev-diagnostic: {multis_path} not found — run round-report first.",
              file=sys.stderr)
        return

    rungs = json.loads(multis_path.read_text())
    sb_rungs = [r for r in rungs if r.get("ladder") == "sportsbet" and r.get("book_combo")]
    if not sb_rungs:
        print("ev-diagnostic: no sportsbet rungs with real book_combo found.", file=sys.stderr)
        return

    # ── Per-leg analysis ─────────────────────────────────────────────────────
    # bucket: {market_type -> list of gaps}
    buckets: dict[str, list[float]] = {}
    under20_count = 0    # model within 20pp of book
    under20_40_count = 0  # model 20-40pp under book (calibration concern)
    over40_count = 0     # model > 40pp under book (extreme)
    total_legs = 0

    for rung in sb_rungs:
        for leg in rung.get("legs", []):
            if "book_odds" not in leg or "hit_prob" not in leg:
                continue
            p_implied = 1.0 / leg["book_odds"]
            gap = p_implied - leg["hit_prob"]   # positive = model under book
            mtype = leg.get("market", "unknown")
            buckets.setdefault(mtype, []).append(gap)
            total_legs += 1
            abs_gap = abs(gap)
            if abs_gap < 0.20:
                under20_count += 1
            elif abs_gap <= 0.40:
                under20_40_count += 1
            else:
                over40_count += 1

    # ── Per-rung split ───────────────────────────────────────────────────────
    rung_rows: list[dict] = []
    for rung in sb_rungs:
        legs = rung.get("legs", [])
        if not legs:
            continue
        book_naive_joint = 1.0
        for leg in legs:
            if "book_odds" in leg:
                book_naive_joint *= (1.0 / leg["book_odds"])

        model_joint = rung.get("model_joint", 0.0)
        book_combo = rung.get("book_combo", 0.0)
        if book_combo <= 0:
            continue

        # EV components (from model's perspective)
        total_ev = model_joint * book_combo - 1.0
        # Leg-disagreement: (model_joint - book_naive_joint) * book_combo
        leg_disagree_ev = (model_joint - book_naive_joint) * book_combo
        # Structural: book_naive_joint * book_combo - 1
        structural_ev = book_naive_joint * book_combo - 1.0

        rung_rows.append({
            "id": rung["id"],
            "band": rung.get("band"),
            "model_joint": model_joint,
            "book_naive_joint": book_naive_joint,
            "book_combo": book_combo,
            "total_ev": total_ev,
            "leg_disagree_ev": leg_disagree_ev,
            "structural_ev": structural_ev,
        })

    # ── Print report ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  EV DIAGNOSTIC  |  {year} R{round_no}  |  {len(sb_rungs)} sportsbet rungs")
    print(f"{'=' * 72}")

    print(f"\n-- PER-LEG GAP (p_book_implied - model_hit_prob)  [{total_legs} legs] --\n")
    print(f"  {'Market':<12}  {'n':>4}  {'mean gap':>10}  {'median gap':>11}  {'stdev':>7}")
    print(f"  {'-'*12}  {'-'*4}  {'-'*10}  {'-'*11}  {'-'*7}")
    for mtype, gaps in sorted(buckets.items()):
        n = len(gaps)
        mean_g = statistics.mean(gaps)
        median_g = statistics.median(gaps)
        stdev_g = statistics.stdev(gaps) if n > 1 else 0.0
        bar = "^" if mean_g > 0.05 else ("~" if abs(mean_g) <= 0.05 else "v")
        print(f"  {mtype:<12}  {n:>4}  {mean_g:>+10.3f}  {median_g:>+11.3f}  {stdev_g:>7.3f}  {bar}")
    print()
    print(f"  Leg gap breakdown: within ±20pp={under20_count}"
          f"  |  20-40pp under book={under20_40_count}"
          f"  |  >40pp under book={over40_count}")

    print(f"\n-- PER-RUNG EV SPLIT --\n")
    print(f"  {'Band':>6}  {'Model%':>8}  {'BookNaive%':>11}  {'Combo':>6}"
          f"  {'TotalEV':>8}  {'LegDisag':>9}  {'Structural':>11}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*11}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*11}")
    for row in rung_rows:
        print(f"  {row['band']:>6.2f}"
              f"  {row['model_joint']*100:>7.1f}%"
              f"  {row['book_naive_joint']*100:>10.1f}%"
              f"  {row['book_combo']:>6.2f}"
              f"  {row['total_ev']:>+8.3f}"
              f"  {row['leg_disagree_ev']:>+9.3f}"
              f"  {row['structural_ev']:>+11.3f}")

    # ── Aggregates ────────────────────────────────────────────────────────────
    all_total_ev = [r["total_ev"] for r in rung_rows]
    all_leg_dis = [r["leg_disagree_ev"] for r in rung_rows]
    all_struct = [r["structural_ev"] for r in rung_rows]
    all_model = [r["model_joint"] for r in rung_rows]
    all_naive = [r["book_naive_joint"] for r in rung_rows]

    print(f"\n  Avg model joint:      {statistics.mean(all_model)*100:.1f}%")
    print(f"  Avg book naive joint: {statistics.mean(all_naive)*100:.1f}%")
    print(f"  Mean total EV:        {statistics.mean(all_total_ev):+.3f}")
    print(f"  -- leg-disagree component: {statistics.mean(all_leg_dis):+.3f}")
    print(f"  -- structural component:   {statistics.mean(all_struct):+.3f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    mean_gap_all = statistics.mean(g for gaps in buckets.values() for g in gaps)
    pct_structural = (abs(statistics.mean(all_struct)) /
                      max(abs(statistics.mean(all_total_ev)), 0.001))

    print(f"\n-- VERDICT --\n")
    if mean_gap_all > 0.12:
        calibration_flag = "CALIBRATION CONCERN"
        detail = (f"The model is on average {mean_gap_all*100:.0f}pp below the book's raw-implied "
                  f"probability across all legs, which accounts for ~{100-pct_structural*100:.0f}% "
                  f"of the -EV on the sportsbet ladder. Structural charges (SGM margin + "
                  f"per-leg book margin) explain only ~{pct_structural*100:.0f}%. "
                  f"The dominant driver is model conservatism - the hit_prob calibration "
                  f"consistently under-rates outcomes the book prices as near-locks. "
                  f"This warrants a follow-up calibration audit (e.g. re-examine LEG_PROB_MAX "
                  f"capping, the PROP_EWMA_HALFLIFE for short-priced legs, or greasiness "
                  f"accumulation across correlated legs). Only back priced legs where the "
                  f"model's edge is clear-cut - the average -EV here is structural + model gap.")
    elif mean_gap_all > 0.05:
        calibration_flag = "MIXED"
        detail = (f"The model is {mean_gap_all*100:.0f}pp below the book's raw-implied average, "
                  f"a moderate gap. Structural book charges (margin + SGM loading) account for "
                  f"~{pct_structural*100:.0f}% of total -EV; model-vs-book leg disagreement "
                  f"accounts for the remainder. Some negative edge is genuine structural book "
                  f"take - accept it and only back the explicit value picks. The calibration gap "
                  f"is real but not extreme; monitor whether it persists across multiple rounds "
                  f"before tuning. No model change in this run.")
    else:
        calibration_flag = "STRUCTURAL"
        detail = (f"The model's per-leg probabilities are close to the book's raw-implied "
                  f"(mean gap {mean_gap_all*100:.0f}pp). The -EV is mostly structural: "
                  f"the book's margin + SGM correlation loading (~{pct_structural*100:.0f}% "
                  f"of total -EV) explains it. This means the model is well-calibrated on "
                  f"these legs and the negative edge is the book's structural take - "
                  f"expected, and only value picks (where model clearly beats book implied) "
                  f"should be backed. No calibration follow-up needed.")
    print(f"  [{calibration_flag}] {detail}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="afl_bot")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run-round", help="Pull a round, simulate, price, and build multis.")
    run_p.add_argument("--year", type=int, required=True)
    run_p.add_argument("--round", type=int, default=None, dest="round_no",
                        help="Defaults to the next incomplete round.")
    run_p.add_argument("--odds", type=str, default=None,
                        help="Path to a JSON file mapping leg names -> market odds.")
    run_p.add_argument("--n-sims", type=int, default=SIM_ITERATIONS, dest="n_sims")
    run_p.add_argument("--synthetic-props", action="store_true", dest="synthetic_props",
                        help="Use a synthetic player log instead of DFS Australia "
                             "(for offline/testing use).")
    run_p.add_argument("--rain-mm", type=float, default=None, dest="rain_mm",
                        help="Price the round as if this much rain (mm) falls; applies "
                             "wet-weather prop multipliers at open-air venues "
                             f"(wet >= {WET_THRESHOLD_MM}mm).")
    run_p.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, dest="bankroll",
                        help="Bankroll ($) for fractional-Kelly stake sizing and the "
                             "bankroll Monte Carlo.")
    run_p.add_argument("--lineup", type=str, default=None, dest="lineup_path",
                        help="Path to a confirmed-lineup JSON {team: [players]}; "
                             "players not named are excluded from multis.")
    run_p.add_argument("--allow-synthetic-props", action="store_true", dest="allow_synthetic_props",
                        help="Permit pricing player props from a synthetic fallback log "
                             "when --odds is supplied (off by default for safety).")

    rep_p = sub.add_parser("round-report",
                           help="Weekly real-player report: per-match projections + SGM multis.")
    rep_p.add_argument("--year", type=int, required=True)
    rep_p.add_argument("--round", type=int, default=None, dest="round_no",
                       help="Defaults to the next incomplete round.")
    rep_p.add_argument("--odds", type=str, default=None,
                       help="Optional JSON of leg names -> market odds for edges.")
    rep_p.add_argument("--n-sims", type=int, default=SIM_ITERATIONS, dest="n_sims")
    rep_p.add_argument("--rain-mm", type=float, default=None, dest="rain_mm",
                       help=f"Price the round as wet (>= {WET_THRESHOLD_MM}mm) at open venues.")
    rep_p.add_argument("--lineup", type=str, default=None, dest="lineup_path",
                       help="Path to a confirmed-lineup JSON {team: [players]}.")
    rep_p.add_argument("--live-odds", action="store_true", dest="use_live",
                       help="Fetch live market odds (The Odds API; needs ODDS_API_KEY). "
                            "Merged with --odds (which overrides, e.g. for prop prices).")
    rep_p.add_argument("--multis-only", action="store_true", dest="multis_only",
                       help="Print only the same-game multi ladder for each match; "
                            "skip player-projection tables and match-summary stats.")
    rep_p.add_argument("--auto-lineup", action="store_true", dest="auto_lineup",
                       help="Fetch team selections from Footywire and exclude players "
                            "not named to play. Overridden by --lineup if both are given.")
    rep_p.add_argument("--multi-calibration", action="store_true", dest="multi_calibration",
                       help="Apply a selection-level isotonic calibrator (model-upgrade "
                            "audit Phase 3.6) to every selected rung's joint probability -- "
                            "corrects search_match_sgms's own selection bias (the "
                            "optimizer's curse, confirmed in Phase 3.5). Opt-in.")
    rep_p.add_argument("--corr-gain-haircut", type=float, default=CORR_GAIN_HAIRCUT,
                       dest="corr_gain_haircut",
                       help="corr_gain-diagnostic follow-up: reprice a selected rung as "
                            "naive_product + haircut*corr_gain instead of the raw sim "
                            "joint_prob (0.0 = naive product only, the live default, "
                            "OOS-validated; 1.0 = raw/unhaircut sim joint_prob). See "
                            "README's 'corr_gain haircut' section.")
    rep_p.add_argument("--sportsbet", action="store_true", dest="use_sportsbet",
                       help="Scrape REAL odds from Sportsbet's own JSON API (no key, no "
                            "paid tier) and price a second 'Sportsbet ladder' beside the "
                            "model-only one. AU IP ONLY (Sportsbet geo-blocks elsewhere) -- "
                            "run this on Ben's own machine, never CI.")
    rep_p.add_argument("--sportsbet-urls", type=str, default=None, dest="sportsbet_urls_path",
                       help="JSON list of Sportsbet match URLs for this round. Defaults to "
                            "reports/<year>_r<round>_sportsbet_urls.json.")
    rep_p.add_argument("--outs", type=str, default=None, dest="outs_path",
                       help="JSON {'_outs': {team: [player, ...]}} of players to ALWAYS "
                            "treat as not named, overriding the lineup source (auto or "
                            "manual). Also read from a --lineup file's own '_outs' key.")
    rep_p.add_argument("--greasiness-file", type=str, default=None, dest="greasiness_file",
                       help="JSON {'Home vs Away': 0.75} or {'HomeTeam': 0.75} of per-game "
                            "greasiness overrides (0.0=dry, 1.0=fully wet). Overrides the "
                            "auto forecast for named games; useful when the forecast under-"
                            "calls a slippery ground (e.g. MCG dew/cold night).")

    grade_p = sub.add_parser("grade-round",
                             help="Score a completed round's saved predictions vs actuals.")
    grade_p.add_argument("--year", type=int, required=True)
    grade_p.add_argument("--round", type=int, required=True, dest="round_no")

    multis_p = sub.add_parser("grade-multis",
                              help="Walk-forward backtest of the 3-leg SGM ladder's joint "
                                   "probability vs actual hit rate.")
    multis_p.add_argument("--year", type=str, required=True,
                          help="Year, or comma-separated years (e.g. 2024,2025).")
    multis_p.add_argument("--rounds", type=str, default=None,
                          help="Comma-separated round numbers (default: every completed "
                               "round of each --year).")
    multis_p.add_argument("--n-sims", type=int, default=SIM_ITERATIONS, dest="n_sims")
    multis_p.add_argument("--no-calibration", action="store_true", dest="no_calibration",
                          help="Skip the calibration-ON comparison (raw reliability curve only).")
    multis_p.add_argument("--calibration-source", choices=["proxy", "sim"], default="proxy",
                          dest="calibration_source",
                          help="'proxy' (default) fits calibrators on the shrunk-EWMA NB "
                               "marginal (Phase 2.5). 'sim' fits on the real sim pipeline's "
                               "probabilities instead (Phase 3.1) -- much slower, but the "
                               "acceptance test for whether calibration fidelity flattens the "
                               "curve further than 'proxy' did.")
    multis_p.add_argument("--all-candidates", action="store_true", dest="all_candidates",
                          help="Also grade the FULL un-selected SGM candidate population "
                               "(Phase 3.5) alongside the selected rungs -- the "
                               "optimizer's-curse check. Roughly doubles runtime.")
    multis_p.add_argument("--lcb-z", type=float, default=0.0, dest="lcb_z",
                          help="Phase 3.5 selection haircut: rank candidates by "
                               "joint_prob - lcb_z * MC-standard-error instead of the raw "
                               "point estimate (0 = off, the default).")
    multis_p.add_argument("--price-shrink", type=float, default=0.0, dest="price_shrink",
                          help="Phase 3.5 selection haircut: shrink the selected rung's "
                               "joint_prob toward its target odds' implied probability by "
                               "this factor, 0-1 (0 = off, the default).")
    multis_p.add_argument("--multi-calibration", action="store_true", dest="multi_calibration",
                          help="Phase 3.6: fit a selection-level isotonic calibrator on "
                               "MULTI_CALIBRATION_LOOKBACK prior seasons of selected-rung "
                               "predictions and report a third, MULTI-CALIBRATED reliability "
                               "curve alongside SELECTED/CALIBRATED.")
    multis_p.add_argument("--corr-gain-diagnostic", action="store_true", dest="corr_gain_diag",
                          help="Parked no-odds-needed diagnostic (PHASE-4-CODE-PLAN.md): "
                               "compare the sim's corr_gain to the EMPIRICAL corr_gain (actual "
                               "joint hit-rate minus pooled actual per-leg hit-rates), bucketed "
                               "by predicted joint_prob. Diagnostic only, no fix applied.")
    multis_p.add_argument("--corr-gain-haircut", type=float, default=CORR_GAIN_HAIRCUT,
                          dest="corr_gain_haircut",
                          help="corr_gain-diagnostic follow-up: reprice selected rungs as "
                               "naive_product + haircut*corr_gain instead of the raw sim "
                               "joint_prob (0.0 = naive product only, the live default, "
                               "mirrors round-report; 1.0 = raw/unhaircut sim joint_prob, "
                               "for baseline comparison).")
    multis_p.add_argument("--halflife", type=float, default=PROP_EWMA_HALFLIFE,
                          dest="prop_halflife",
                          help="PROP_EWMA_HALFLIFE override for the proxy-calibration pass "
                               "(calibration-source=proxy only). Use this to verify a halflife "
                               "candidate from sweep-halflife at the multi level. "
                               f"Default: {PROP_EWMA_HALFLIFE} (config default).")

    sweep_p = sub.add_parser("sweep-halflife",
                             help="Diagnostic sweep: compare PROP_EWMA_HALFLIFE candidates "
                                  "[6,8,10,12] on OOS prop log loss / Brier / ECE / "
                                  "high-bucket gap. Fast (no sim — closed-form NB marginal).")
    sweep_p.add_argument("--years", type=str, default="2024,2025",
                         help="Comma-separated eval years (default: 2024,2025).")
    sweep_p.add_argument("--halflives", type=str, default="6,8,10,12",
                         help="Comma-separated halflife candidates (default: 6,8,10,12).")
    sweep_p.add_argument("--cal-lookback", type=int, default=4, dest="cal_lookback",
                         help="Seasons of walk-forward history used to fit calibrators "
                              "for each eval year (default: 4).")

    fit_p = sub.add_parser("fit", help="Re-tune Elo and write a versioned params artifact.")
    fit_p.add_argument("--through", type=int, required=True,
                       help="Fit on completed seasons up to and including this year.")
    fit_p.add_argument("--optuna", action="store_true", dest="use_optuna",
                       help="Use Optuna (optional dep) instead of the dependency-free grid.")
    fit_p.add_argument("--n-trials", type=int, default=150, dest="n_trials")

    fit_corr_p = sub.add_parser("fit-correlations",
                                help="Fit the SGM correlation/dispersion constants from "
                                     "history and write a versioned params artifact.")
    fit_corr_p.add_argument("--through", type=int, required=True,
                            help="Fit on completed seasons up to and including this year.")

    settle_p = sub.add_parser("settle-bets",
                               help="Auto-settle pending bets in reports/bets_ledger.json "
                                    "using completed-round actuals.")
    settle_p.add_argument("--year", type=int, default=None,
                          help="Limit to bets from this year (default: all pending).")
    settle_p.add_argument("--round", type=int, default=None, dest="round_no",
                          help="Limit to bets from this round (default: all pending).")
    settle_p.add_argument("--ledger", type=str, default=None, dest="ledger_path",
                          help="Path to bets ledger JSON (default: reports/bets_ledger.json).")

    cap_p = sub.add_parser("capture-close",
                           help="Snapshot closing reference prices for pending bets "
                                "(run near bounce). Marks CLV unavailable until a "
                                "sharp reference (Betfair / 2nd book) is connected.")
    cap_p.add_argument("--year", type=int, default=None)
    cap_p.add_argument("--round", type=int, default=None, dest="round_no")
    cap_p.add_argument("--ledger", type=str, default=None, dest="ledger_path",
                       help="Path to bets ledger JSON (default: reports/bets_ledger.json).")
    cap_p.add_argument("--sportsbet-urls", type=str, default=None, dest="sportsbet_urls_path",
                       help="JSON list of Sportsbet match URLs (for line-movement tracking). "
                            "Same file as used by round-report --sportsbet.")

    dash_p = sub.add_parser("dashboard",
                             help="Launch the multis dashboard at http://127.0.0.1:8765 .")
    dash_p.add_argument("--port", type=int, default=8765)
    dash_p.add_argument("--no-browser", action="store_true", dest="no_browser",
                        help="Don't auto-open the browser.")

    pcal_p = sub.add_parser("prop-calibration-check",
                            help="OOS prop calibration check: model calibrated prob vs actual "
                                 "hit rate for 2024-2025, overlaid with book implied. "
                                 "Diagnostic only, no model change.")
    pcal_p.add_argument("--years", type=str, default="2024,2025",
                        help="Comma-separated eval years (default: 2024,2025).")
    pcal_p.add_argument("--out", type=str, default=None, dest="out_path",
                        help="Output markdown path (default: reports/prop_calibration_check_2024_2025.md).")
    pcal_p.add_argument("--multis-year", type=int, default=2026, dest="multis_year",
                        help="Year of the multis.json to use for book comparison (default: 2026).")
    pcal_p.add_argument("--multis-round", type=int, default=None, dest="multis_round",
                        help="Round of the multis.json to use for book comparison "
                             "(default: auto-detect latest).")

    evdiag_p = sub.add_parser("ev-diagnostic",
                               help="Per-leg and per-rung EV breakdown for a saved sportsbet "
                                    "ladder: model vs book-implied gap, structural vs "
                                    "leg-disagreement split. Diagnostic only, no model change.")
    evdiag_p.add_argument("--year", type=int, required=True)
    evdiag_p.add_argument("--round", type=int, required=True, dest="round_no")

    args = parser.parse_args(argv)
    if args.command == "run-round":
        run_round(args.year, args.round_no, args.odds, args.n_sims,
                  args.synthetic_props, args.rain_mm, args.bankroll,
                  args.lineup_path, args.allow_synthetic_props)
    elif args.command == "round-report":
        round_report(args.year, args.round_no, args.odds, args.n_sims,
                     args.rain_mm, args.lineup_path, args.use_live, args.multis_only,
                     args.auto_lineup, args.multi_calibration, args.corr_gain_haircut,
                     args.use_sportsbet, args.sportsbet_urls_path, args.outs_path,
                     greasiness_overrides_path=args.greasiness_file)
    elif args.command == "grade-round":
        grade_round(args.year, args.round_no)
    elif args.command == "grade-multis":
        rounds = [int(r) for r in args.rounds.split(",")] if args.rounds else None
        years = [int(y) for y in args.year.split(",")]
        grade_multis(years, rounds, args.n_sims, with_calibration=not args.no_calibration,
                    calibration_source=args.calibration_source, all_candidates=args.all_candidates,
                    lcb_z=args.lcb_z, price_shrink=args.price_shrink,
                    multi_calibration=args.multi_calibration, corr_gain_diag=args.corr_gain_diag,
                    corr_gain_haircut=args.corr_gain_haircut, prop_halflife=args.prop_halflife)
    elif args.command == "sweep-halflife":
        years = [int(y) for y in args.years.split(",")]
        halflives = [float(h) for h in args.halflives.split(",")]
        sweep_halflife_command(years, halflives, args.cal_lookback)
    elif args.command == "fit":
        fit_command(args.through, args.use_optuna, args.n_trials)
    elif args.command == "fit-correlations":
        fit_correlations_command(args.through)
    elif args.command == "settle-bets":
        from afl_bot.dashboard.settle import settle_bets as _settle
        ledger_path = args.ledger_path or str(ROOT_DIR / "reports" / "bets_ledger.json")
        n = _settle(ledger_path, year=args.year, round_no=args.round_no)
        print(f"Settled {n} bet(s). Ledger: {ledger_path}")
    elif args.command == "capture-close":
        from afl_bot.dashboard.capture_close import capture_close as _capture
        ledger_path = args.ledger_path or str(ROOT_DIR / "reports" / "bets_ledger.json")
        sb_urls: list[str] | None = None
        if args.sportsbet_urls_path:
            try:
                sb_urls = json.loads(Path(args.sportsbet_urls_path).read_text())
            except (OSError, json.JSONDecodeError):
                print(f"capture-close: could not read {args.sportsbet_urls_path}", file=sys.stderr)
        # Auto-fetch TAB for consensus CLV reference (fails gracefully -> {})
        tab_odds: dict[str, float] = {}
        try:
            from afl_bot.data.tab_odds import fetch_tab_odds as _fetch_tab
            tab_odds = _fetch_tab()
        except Exception as _exc:
            print(f"capture-close: TAB fetch failed ({_exc}) — CLV consensus unavailable.",
                  file=sys.stderr)
        result = _capture(ledger_path, year=args.year, round_no=args.round_no,
                          sportsbet_urls=sb_urls, tab_odds=tab_odds)
        print(f"capture-close: {result['n_updated']} bet(s) updated "
              f"({result['n_sharp']} sharp reference, "
              f"{result['n_soft_only']} soft-only / unavailable).")
        if result["n_sharp"] == 0 and result["n_updated"] > 0:
            print("  CLV unavailable: ensure Sportsbet + TAB both have prices for "
                  "all legs, or add Betfair for H2H CLV.")
    elif args.command == "dashboard":
        from afl_bot.dashboard.app import run_dashboard
        run_dashboard(port=args.port, open_browser=not args.no_browser)
    elif args.command == "prop-calibration-check":
        years = [int(y) for y in args.years.split(",")]
        prop_calibration_check(years, out_path=args.out_path,
                               multis_year=args.multis_year,
                               multis_round=args.multis_round)
    elif args.command == "ev-diagnostic":
        ev_diagnostic(args.year, args.round_no)


if __name__ == "__main__":
    main()

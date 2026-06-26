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
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    simulate_bankroll,
    simulate_bankroll_joint,
    stake_bets,
)
from afl_bot.config import (
    ANCHOR_MIN_PROB,
    BOOKABLE_TOP_N_BY_STAT,
    CACHE_DIR,
    CORR_GAIN_HAIRCUT,
    DEFAULT_BANKROLL,
    LEG_PROB_MAX,
    LEG_PROB_MIN,
    MC_SE_TARGET,
    MULTI_CALIBRATION_LOOKBACK,
    MULTI_MARKET_SHRINK,
    PLAYER_FORM_WINDOW,
    PROP_CALIBRATION_LOOKBACK,
    PROP_KELLY_MULTIPLIER,
    PROP_LINES,
    PROP_MARKET_BLEND_WEIGHT,
    PROP_RECENT_SEASONS,
    ROOT_DIR,
    SHARE_CONCENTRATION,
    SIM_ITERATIONS,
    TEAM_STAT_DISPERSION,
    TOG_RETURN_DEFAULT,
    WET_THRESHOLD_MM,
)
from afl_bot.data.odds import fetch_historical_odds
from afl_bot.data.lineups import apply_outs, fetch_lineup, load_lineup, load_lineup_tog, load_outs
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


def _rung_to_json(rung: dict, ladder: str, year: int, round_no: int,
                  home: str, away: str,
                  leg_by_name: dict, odds_book: dict) -> dict:
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
        })

    return {
        "id": multi_id,
        "year": year,
        "round": round_no,
        "game": f"{home} vs {away}",
        "ladder": ladder,
        "band": band,
        "legs": legs_json,
        "model_joint": rung["joint_prob"],
        "model_fair": rung["fair_odds"],
        "book_combo": rung.get("book_odds"),
        "edge": rung.get("edge"),
        "value_pick": bool(rung.get("value_pick", False)),
    }


def round_report(year: int, round_no: int | None, odds_path: str | None, n_sims: int,
                 rain_mm: float | None = None, lineup_path: str | None = None,
                 use_live: bool = False, multis_only: bool = False,
                 auto_lineup: bool = False, multi_calibration: bool = False,
                 corr_gain_haircut: float = CORR_GAIN_HAIRCUT,
                 use_sportsbet: bool = False, sportsbet_urls_path: str | None = None,
                 outs_path: str | None = None) -> None:
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
    # Live h2h/totals + player props (The Odds API) and/or real Sportsbet
    # prices, merged with any --odds file; manual overrides everything for
    # hand-fixes (MULTI-CHANGES PART A; FIX-REAL-SPORTSBET-ODDS-AND-LINEUP
    # PART A6). Sportsbet is the richer, real source (props included, not
    # just h2h/totals), so it sits ahead of the (already-live) Odds API feed.
    live = fetch_live_odds(round_no) if use_live else {}
    live_props = fetch_live_props(round_no) if use_live else {}
    if use_sportsbet:
        urls_path = sportsbet_urls_path or str(ROOT_DIR / "reports" /
                                               f"{year}_r{round_no}_sportsbet_urls.json")
        try:
            sb_urls = json.loads(Path(urls_path).read_text())
        except (OSError, json.JSONDecodeError):
            print(f"Sportsbet: no URL file at {urls_path} -- skipping scrape.", file=sys.stderr)
            sb_urls = []
        sb = fetch_sportsbet_odds(sb_urls)
    else:
        sb = {}
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

    sportsbet_note = ""
    if use_sportsbet:
        sportsbet_note = (
            f"_Player-prop odds: live from Sportsbet (scraped, {len(sb)} leg(s) priced)._"
            if sb else
            "_Player-prop odds: Sportsbet scrape unavailable (not in AU / blocked / no URL "
            "file for this round) — fell back to --odds file / model-only._")

    rng = make_rng()
    matches, odds_legs, predictions, multis_records = [], [], [], []
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
        # Totals only become a real SGM-eligible leg once priced (model-upgrade
        # audit Phase 4 STEP 1.2) -- unlike h2h/props it has no model-derived
        # fallback price here, so a no-odds run stays byte-for-byte unchanged.
        if total_leg_name in odds_book:
            total_leg = LegCandidate(total_leg_name, match_id, "total_points", "total",
                                     header["p_total"], odds_book[total_leg_name], mask=total_mask)
            match_legs.append(total_leg)
            odds_legs.append(total_leg)

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
                        if name in odds_book:
                            odds_legs.append(leg)
                        priceable_names.append(name)
                        known_input_keys.update([name, under_name])
                        if devig_prob is not None:
                            priced_legs.append({
                                "name": name, "model_prob": prob, "book_odds": over_odds,
                                "devig_prob": devig_prob, "devig_label": devig_label,
                                "blended_prob": blended_prob,
                                "edge_pct": leg.edge_pct, "classification": leg.classification,
                            })

        for leg in match_legs[:2]:           # H2H legs with book odds feed cross-game multis
            if leg.name in odds_book:
                odds_legs.append(leg)

        sgms = search_match_sgms(match_legs, odds_book=odds_book,
                                 corr_gain_haircut=corr_gain_haircut, multi_calibrator=multi_cal)
        # FIX-REAL-SPORTSBET-ODDS-AND-LINEUP PART C: a second ladder selected
        # and priced on REAL book odds (Sportsbet/--odds) from the same leg
        # pool -- [] when nothing in this match is priced.
        market_sgms = search_market_sgms(match_legs, odds_book=odds_book)
        # Stage 2A: emit machine-readable multis JSON alongside the .md so the
        # dashboard can render these rungs without re-parsing markdown.
        leg_by_name = {l.name: l for l in match_legs}
        for ladder_label, rungs in (("model", sgms), ("sportsbet", market_sgms)):
            for r in rungs:
                multis_records.append(_rung_to_json(
                    r, ladder_label, year, round_no,
                    home_name, away_name, leg_by_name, odds_book))
        matches.append({
            "header": header, "projections": projections,
            "sgms": sgms, "market_sgms": market_sgms, "priced_legs": priced_legs,
            "n_legs": len(match_legs),     # so the report can explain an empty ladder
        })

    # Warn on odds-file keys that never matched a priceable leg (a typo = a
    # leg silently dropped, round-2 §7.4 / model-upgrade audit Phase 4 STEP 1.1).
    unmatched = [k for k in odds_book if not k.startswith("_") and k not in known_input_keys]
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} odds key(s) matched no priceable leg "
              f"(typo? player not in pool/lineup?): {', '.join(unmatched)}", file=sys.stderr)

    multis_section = ""
    if odds_legs:
        promo = build_promo_multi(odds_legs, joint_prob_fn=joint_prob_from_masks)
        lines = ["## Cross-game multis"]
        if promo is not None:
            lines.append(f"- **Promo multi**: {' + '.join(l.name for l in promo.legs)} "
                         f"-> combined {promo.combined_fair_prob:.3f} @ book {promo.combined_market_odds:.2f} "
                         f"(EV {promo.promo['ev_pct'] * 100:+.1f}%)")
        for i, mm in enumerate(build_anchor_multis(odds_legs, joint_prob_fn=joint_prob_from_masks), 1):
            lines.append(f"- Anchor multi {i}: {' + '.join(l.name for l in mm.legs)} "
                         f"-> {mm.combined_fair_prob:.3f}")
        multis_section = "\n".join(lines) + "\n"

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
                         multis_section=multis_section, odds_note=odds_note,
                         sportsbet_note=sportsbet_note, proj_note=proj_note,
                         multis_only=multis_only)
    out_dir = ROOT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{year}_r{round_no}_report.md"
    out_path.write_text(md, encoding="utf-8")
    # Machine-readable predictions sidecar so the round can be graded later (§10.5).
    pred_path = out_dir / f"{year}_r{round_no}_predictions.csv"
    pd.DataFrame(predictions).to_csv(pred_path, index=False)
    # Stage 2A: machine-readable multis JSON for the dashboard.
    multis_path = out_dir / f"{year}_r{round_no}_multis.json"
    multis_path.write_text(json.dumps(multis_records, indent=2), encoding="utf-8")
    # Odds template (model-upgrade audit Phase 4 STEP 1.2): every priceable
    # leg's exact name -> null, copy-paste-able into a fresh --odds file.
    template_path = out_dir / f"{year}_r{round_no}_odds_template.json"
    template_path.write_text(
        json.dumps(build_odds_template(priceable_names), indent=2), encoding="utf-8")
    # Snapshot the filled --odds file (model-upgrade audit Phase 4 STEP 2.4)
    # so the repo accumulates its own prop-odds history over time -- the
    # exact archive STEP 2.2 says doesn't exist yet, the cheapest path to
    # one day fitting PROP_MARKET_BLEND_WEIGHT for real.
    odds_snapshot_path = None
    if manual:
        odds_snapshot_path = out_dir / f"{year}_r{round_no}_odds.json"
        odds_snapshot_path.write_text(json.dumps(manual, indent=2), encoding="utf-8")
    print(md)
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
    combined.to_csv(log_path, index=False)

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
                 corr_gain_haircut: float = CORR_GAIN_HAIRCUT) -> None:
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
                    cache_dir=backtest_cal_cache, force_refresh=True) or None
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

    dash_p = sub.add_parser("dashboard",
                             help="Launch the multis dashboard at http://127.0.0.1:8765 .")
    dash_p.add_argument("--port", type=int, default=8765)
    dash_p.add_argument("--no-browser", action="store_true", dest="no_browser",
                        help="Don't auto-open the browser.")

    args = parser.parse_args(argv)
    if args.command == "run-round":
        run_round(args.year, args.round_no, args.odds, args.n_sims,
                  args.synthetic_props, args.rain_mm, args.bankroll,
                  args.lineup_path, args.allow_synthetic_props)
    elif args.command == "round-report":
        round_report(args.year, args.round_no, args.odds, args.n_sims,
                     args.rain_mm, args.lineup_path, args.use_live, args.multis_only,
                     args.auto_lineup, args.multi_calibration, args.corr_gain_haircut,
                     args.use_sportsbet, args.sportsbet_urls_path, args.outs_path)
    elif args.command == "grade-round":
        grade_round(args.year, args.round_no)
    elif args.command == "grade-multis":
        rounds = [int(r) for r in args.rounds.split(",")] if args.rounds else None
        years = [int(y) for y in args.year.split(",")]
        grade_multis(years, rounds, args.n_sims, with_calibration=not args.no_calibration,
                    calibration_source=args.calibration_source, all_candidates=args.all_candidates,
                    lcb_z=args.lcb_z, price_shrink=args.price_shrink,
                    multi_calibration=args.multi_calibration, corr_gain_diag=args.corr_gain_diag,
                    corr_gain_haircut=args.corr_gain_haircut)
    elif args.command == "fit":
        fit_command(args.through, args.use_optuna, args.n_trials)
    elif args.command == "fit-correlations":
        fit_correlations_command(args.through)
    elif args.command == "settle-bets":
        from afl_bot.dashboard.settle import settle_bets as _settle
        ledger_path = args.ledger_path or str(ROOT_DIR / "reports" / "bets_ledger.json")
        n = _settle(ledger_path, year=args.year, round_no=args.round_no)
        print(f"Settled {n} bet(s). Ledger: {ledger_path}")
    elif args.command == "dashboard":
        from afl_bot.dashboard.app import run_dashboard
        run_dashboard(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

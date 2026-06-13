# AFL Multi Builder

Logic-based Monte Carlo AFL simulator that surfaces betting selections: 2
high-probability "anchor" legs + 1 high-value "edge" leg, sized for promo-style
3-leg multis. Implements the pipeline from `AFL_Multi_Builder_Plan.md`.

## Setup

```
pip install -r requirements.txt
```

## Pipeline (plan §1)

| Stage | Module |
|---|---|
| 1. Data | `afl_bot/data/squiggle.py` (Squiggle API, cached parquet), `afl_bot/data/fryzigg.py` (historical per-player box scores, 2012+), `afl_bot/data/dfs_australia.py` (current-season per-player box scores incl. CBA), `afl_bot/data/player_stats.py` (`load_player_log` combines both, synthetic fallback), `afl_bot/data/odds.py` (historical H2H/totals odds, cached parquet), `afl_bot/data/weather.py` (per-game rainfall/wind via Open-Meteo) + `afl_bot/data/venues.py` (venue metadata/roof), `afl_bot/data/teams.py` (canonical team names), `afl_bot/data/storage.py` (schema-versioned parquet cache + DuckDB views) |
| 2. Features | `afl_bot/models/scoring.py` (points/shots), `afl_bot/models/pace.py` (team volume-stat totals for the pace factor), `afl_bot/models/props.py` (player rates/shares), `afl_bot/models/priors.py` (hierarchical role priors + TOG/CBA adjustments), `afl_bot/models/weather_effects.py` (wet-weather prop multipliers), `afl_bot/models/stoppages.py` (boundary throw-in / OOB market) |
| 3. Ratings | `afl_bot/ratings/elo.py` (margin-based Elo, tunable, "margin"/"mov" update rules) |
| 4. Sim | `afl_bot/sim/engine.py` (Monte Carlo scoring-shots match w/ score copula + pace-driven player props) |
| 5. Pricing | `afl_bot/pricing/edge.py` (fair odds, devig, edge, leg classification) |
| 6/7. Build | `afl_bot/build/multi.py` (promo-aware multi assembly), `afl_bot/build/staking.py` (fractional-Kelly staking + bankroll sims) |
| Backtest | `afl_bot/backtest/walkforward.py` (walk-forward Elo calibration), `afl_bot/backtest/tuning.py` (grid / Optuna Elo tuning), `afl_bot/backtest/ensemble.py` (market-blend + isotonic calibration) |

## Usage

```
python -m afl_bot.cli run-round --year 2026
python -m afl_bot.cli run-round --year 2026 --round 14 --odds sample_odds.json
python -m afl_bot.cli run-round --year 2026 --round 14 --odds odds.json --rain-mm 12 --bankroll 2000
```

Without `--odds`, the CLI prints fair probabilities/odds for every match and a
sample of player props. With `--odds` (a JSON file mapping leg names to market
decimal odds), it classifies legs as ANCHOR/VALUE/SKIP, assembles the 3-leg
promo multi and the "very highly likely" anchor multis, and recommends
fractional-Kelly stakes with a bankroll simulation. `--rain-mm` prices a
wet-weather scenario; `--bankroll` sets the stake-sizing bankroll. `--lineup`
(a `{team: [players]}` JSON) restricts pricing to the confirmed team sheets, and
the player pool is otherwise gated to current-season players so retired/injured
players aren't priced off a career average.

### Weekly round report

```
python -m afl_bot.cli round-report --year 2026 --round 14 [--odds odds.json] [--lineup lineup.json] [--live-odds]
python -m afl_bot.cli grade-round --year 2025 --round 1
```

`round-report` is the weekly deliverable: for every match it prints (and saves
to `reports/<year>_r<N>_report.md`) a header, a **real-player** projection table
per team (projected mean + P(15/20/25 disposals), P(1/2 goals), P(4/6 marks),
P(3/5 tackles)), and a **same-game multi ladder**: a spread of minimum-3-leg
multis bucketed by combined odds (~1.75 → ~5.0), each priced off the **joint**
sim probability (shown with the correlation gain; book odds + edge when odds are
supplied). Every match is guaranteed a full ladder — if a band has no natural
combo it's filled from the closest one, and a match too thin for a 3-leg multi
says *why* (leg count) rather than silently dropping. The lower/mid rungs pick
the safest combo at that price (highest joint prob); the top (~3.5–5.5) rung is
the **VALUE PICK** — chosen by the highest *edge*, not probability. That edge is
**market-shrunk** (`market_anchored_prob`, since per-leg overestimates compound
across a multi) and **capped at 15%**: a larger apparent edge means the model
disagrees with the book because the model is wrong, so it is not flagged. A rung
is only tagged VALUE when a real market price (i.e. prop odds) justifies it.
Cross-game promo / anchor multis (now 3-leg) follow.

Odds can come from a static `--odds` JSON or **live** from
[The Odds API](https://the-odds-api.com) with `--live-odds` (set the
`ODDS_API_KEY` env var; the free tier covers **h2h/totals only**). Live and
manual are merged as `{**live, **manual}` — the `--odds` file overrides live for
hand-fixes and supplies **player-prop** prices the free feed lacks. The report
prints a one-line note saying how many legs have a live market price (legs
without one show no edge); it never pretends props are live. It refuses to run on
a synthetic player log (real names only) and writes a `*_predictions.csv`
sidecar. After the round, `grade-round` scores those saved predictions against
the actual results and appends to `reports/calibration_log.csv` (round +
cumulative log loss / Brier).

## Backtesting

```python
import pandas as pd
from afl_bot.data.squiggle import SquiggleClient
from afl_bot.backtest.walkforward import evaluate_elo, season_by_season_report

client = SquiggleClient()
games = pd.concat([client.get_completed_games(y) for y in range(2018, 2025)], ignore_index=True)
print(season_by_season_report(evaluate_elo(games)))
```

### Market comparison & CLV (plan §4.2)

`afl_bot/data/odds.py` downloads and caches the Australia Sports Betting
historical AFL odds spreadsheet (H2H open/close, totals, back to ~2009). The
walk-forward Elo predictions can then be compared against the closing market
and checked for closing-line value (CLV) -- the standard "fastest honest
signal of real edge":

```python
from afl_bot.data.odds import attach_odds, fetch_historical_odds
from afl_bot.backtest.walkforward import (
    clv_report, clv_summary, evaluate_elo, market_metrics,
)

odds = fetch_historical_odds()
history = evaluate_elo(games)

# How does the model's log loss/Brier compare to the closing market?
print(market_metrics(attach_odds(history, odds)))

# Games where the model disagrees with the closing market by >= VALUE_MIN_EDGE,
# and whether backing the model's side at the *open* would have beaten the close.
flagged = clv_report(history, odds)
print(clv_summary(flagged))
```

`attach_odds` joins on `(year, hteam, ateam)` -- each team plays each opponent
at most twice a season (once per venue), so this triple is a safe join key
without needing exact date/timezone matching.

### Home-ground advantage, venue scoring & the `fit` command (round-2 §6)

`afl_bot/ratings/hga.py` replaces the flat home advantage with a **per-team
venue HGA** (home-minus-away margin swing, shrunk toward the league 10) plus an
**interstate-travel** penalty and a **days-rest** differential, attached as a
`hga_points` column the Elo update/prediction consume. Fitted HGAs match
published analysis (Geelong +11, MCG tenants ~+1–3) and improve the walk-forward
(log loss 0.6266→0.6244). `venue_scoring_factors` feeds a per-venue `venue_factor`
into the total. `margin_calibration` regresses actual on predicted margin
(slope ~1 = calibrated; the real model is ~1.27, i.e. slightly under-confident).
`python -m afl_bot.cli fit --through <year> [--optuna]` re-tunes Elo and writes a
versioned `elo_params.json` (params + held-out metrics + git-sha/date
provenance) that `run-round`/`round-report` pick up — opt-in, so defaults are
used until you deliberately fit.

### Elo tuning (plan §2.3)

`afl_bot/backtest/tuning.py` tunes the Elo hyperparameters (k, season carryover,
home advantage, margin cap, points-per-400, and the "mov" update rule's MOV
correction) against a held-out window of recent seasons, minimising a joint
objective of out-of-sample **log loss + margin MAE**. `EloRatings.fit` is a
tight numpy-array loop (no `iterrows`), so thousands of fits stay cheap
(~15 ms/fit over ~1,750 games).

```python
from afl_bot.backtest.tuning import grid_search_elo, optuna_search_elo, elo_objective

# Dependency-free exhaustive grid:
print(grid_search_elo(games, eval_start_year=2023).head())

# Bayesian search (optional `optuna` dependency), tuning on 2018-2023 and
# scoring only the 2020-2023 window:
train = games[games["year"] <= 2023]
study = optuna_search_elo(train, n_trials=150, eval_start_year=2020)
print(study.best_params)

# Honest check: score the tuned params on seasons tuning never saw.
print(elo_objective(games, eval_start_year=2024, **study.best_params))
```

Tune on early seasons and report on a *further* untouched window — Elo is online
so ratings warm up through the data, but the hyperparameters are global, so the
out-of-sample window is what guards against overfitting. On the current
2018-2025 cache, tuned params beat the defaults on the held-out 2024-2025 window
(log loss ~0.576 vs ~0.603), but the in-sample optimum chases the `k` boundary
on only ~6 seasons of data, so the config defaults are intentionally left
unchanged — re-tune (and widen ranges) as more seasons accumulate before
adopting specific values.

The alternative `update_mode="mov"` uses a FiveThirtyEight margin-of-victory
multiplier (`ln(|margin|+1)` × an autocorrelation correction) on a binary
result instead of the clipped-linear margin squash, so big wins aren't capped
and runaway favourites don't inflate ratings.

### Market-blend ensemble + calibration (plan §3.5)

`afl_bot/backtest/ensemble.py` treats the market as the best single predictor
and blends the model toward it. `assemble_signals` builds three H2H home-win
signals — the Elo model, the devigged closing market, and the Squiggle
crowd-model consensus — and `fit_market_blend` learns an `IsotonicCalibrator`
(PAVA, no sklearn) plus convex blend weights (log-loss-optimal, on the simplex).
`ensemble_report` fits on a training window and scores everything on a held-out
window. On 2018-2025 (train ≤2022, test ≥2023):

| signal | held-out log loss |
|---|---|
| model (Elo) alone | 0.612 |
| Squiggle consensus | 0.576 |
| market (devig close) | 0.5725 |
| **blend** (0.42 model / 0.58 market) | **0.5722** |

So the blend matches/edges the market and crushes the raw model. Calibration's
real job here is *scale alignment*: the raw model gets ~0 blend weight (wrong
scale), but the calibrated model blends in at 0.42 and adds signal.

`MarketBlend` is wired into `run-round`: when an `--odds` file gives both H2H
prices, the leg edge is taken on the **market-anchored blend**, not the raw
model — which kills the runaway false edges a model otherwise throws off. It is
best-effort and degrades to the raw model if odds/tips are unavailable.

### Output honesty (round-2 §8)

Anchor multis are ranked and tagged by **combined edge** (`model prob × book
odds − 1`), not probability — a 0.90 leg at 1.05 stacks to −EV. Before staking,
each multi leg's prob is pulled toward its market price (`MULTI_MARKET_SHRINK`)
so per-leg overestimates don't compound. Every priced market shows its **Monte
Carlo standard error** and `run-round` auto-bumps `n_sims` so the tightest anchor
clears `MC_SE_TARGET` (0.002). An odds file may declare house rules — e.g.
`{"_rules": {"h2h_draw": "refund"}}` conditions the win prob on a non-draw.

### Prop backtest + per-market calibration (round-2 §2)

`afl_bot/backtest/props.py` walk-forward-backtests every player prop:
`walk_forward_prop_predictions` rebuilds each player's rate as-of every
historical round (EWMA shifted to use only prior games), prices the standard
lines with a Negative-Binomial marginal, and records predicted prob vs actual
hit. On 2023-25 (266k predictions) the rate model is already roughly calibrated
(disposals: mean pred 0.286 vs hit 0.305), and an `IsotonicCalibrator` fit
**per market type** improves every stat. `load_or_fit_prop_calibrators` caches
the fitted calibrators (compressed JSON) and they are applied in `run-round` /
`round-report` before legs are classified, staked, and reported. Prop legs are
also staked at half Kelly (`PROP_KELLY_MULTIPLIER`) since they are noisier and
compound across a multi.

### Staking & bankroll (plan §4.4)

`afl_bot/build/staking.py` sizes bets by **capped fractional Kelly**:
`f* = (p·odds − 1)/(odds − 1)`, scaled to `KELLY_FRACTION` (0.25×), capped per
bet (`KELLY_PER_BET_CAP`) and per round (`KELLY_PER_ROUND_CAP`). `run-round`
prints recommended stakes for the +EV legs and the promo multi against
`--bankroll`, then `simulate_bankroll` runs a vectorised Monte Carlo of that
edge profile over a season and `bankroll_report` summarises the terminal and
**max-drawdown** distributions (median end, P(profit), P(bust), P(drawdown
>50%)) — the honest variance picture behind a positive EV.

## Match simulation: scoring shots (plan §2.1, §2.2)

`afl_bot/sim/engine.py` simulates each team's score as scoring shots (goals +
behinds), not a Normal margin/total split:

```
shots  ~ NegativeBinomial(mu_shots, SHOT_DISPERSION)
goals  ~ Binomial(shots, accuracy)        # accuracy ~ team EWMA +/- noise
points = 6 * goals + (shots - goals)
```

`mu_shots` is derived from the existing Elo margin + scoring-profile total
(`afl_bot.models.scoring.points_to_shots`); `accuracy` comes from
`team_shot_accuracy_profiles` (an anti-leakage EWMA goal-conversion rate per
team, default `DEFAULT_SHOT_ACCURACY` for teams with no history). This gives
integer scorelines, real draw probabilities, and -- because Negative Binomial
variance grows with its mean -- heteroscedastic margin/total spread for free
(higher-scoring games carry more variance), without a separate sigma model.
`SHOT_DISPERSION` and `SHOT_ACCURACY_SIGMA` (`afl_bot/config.py`) are
calibrated against 2015+ results so simulated team-points variance matches the
empirical ~633 (std ~25/team).

The two teams' scoring-shot draws are then coupled by a **Gaussian copula**
(`SCORE_SHOT_CORRELATION`, plan §3.3): AFL scores are negatively correlated
(territory is ~zero-sum), so a negative correlation is applied while the NB
marginals are preserved exactly. Calibrated at `-0.32`, this reproduces the
empirical split — margin sigma ~39.3 / total sigma ~31.4 / corr ~-0.22 (vs
empirical 39.4 / 31.3 / -0.224) — instead of the ~36/36 an independent draw
gives. Pass `score_correlation=0.0` to `simulate_match` to recover independence.

## Player props: pace & within-team allocation (plan §2.5, §3.3)

Volume props (disposals / marks / tackles) are simulated as a three-level
*environment → team total → player share* cascade so the correlations books
exploit are priced correctly:

1. **Shared pace** — `draw_pace` draws one mean-1 lognormal multiplier per
   iteration for the whole match. Feeding the *same* array into both teams'
   totals makes disposals correlate *across* teams (a fast, open game lifts
   everyone), not just within a team.
2. **Team total** — `simulate_team_stat_total` draws each team's pace-scaled
   volume total (NB around `team_EWMA_total × pace`), where the per-team EWMA
   comes from `afl_bot/models/pace.py` (`team_stat_total_profiles`,
   anti-leakage) and is adjusted by the opponent's concede multiplier.
3. **Player shares** — `allocate_player_stats` splits that team total among the
   priced players with a `Dirichlet(SHARE_CONCENTRATION × shares)` draw, so
   every iteration the players sum to the team total. Teammates therefore move
   together with the team total (shared pace) but trade share against each
   other (the Dirichlet sum constraint) — the structure independent NB draws
   get wrong for same-game multis. `SHARE_CONCENTRATION` is calibrated (~200)
   to a realistic top-mid disposal CoV (~0.26); per-player dispersion matching
   is left to the hierarchical priors of build-order step 7 (plan §3.1).

Goals stay on the scoreline-correlated NB path (they scale with the iteration's
team goals from the scoring-shots model).

### Hierarchical priors & role adjustments (plan §3.1, §3.2)

`afl_bot/models/priors.py` makes the player rates that feed the allocation less
noisy and more responsive to role:

- **Empirical-Bayes shrinkage (§3.1).** A player's raw EWMA mean/share is shrunk
  toward a **role prior** (the average for his inferred position group) with
  `shrink(raw, n_games, prior, strength) = (n·raw + strength·prior)/(n+strength)`.
  `PROP_PRIOR_STRENGTH` is a pseudo-game count, so a 2-game debutant sits near
  the prior and a long-history player near his own number. NB dispersion is
  likewise pooled by role (`estimate_dispersion_hierarchical`) instead of one
  league fallback.
- **Roles** come from the **real AFL position labels** both sources carry
  (`player_position` / `startingPosition` → `POSITION_TO_ROLE`); a player's modal
  non-bench position sets his role, with box-score inference (`classify_roles`:
  ruck → forward → midfielder → general) only as a fallback for INT/SUB rows or
  a synthetic log. The role mean/share priors are also **era-matched** (last 3
  seasons, `PROP_RECENT_SEASONS`), as is the `opponent_matchup_multiplier`
  league baseline — stat levels drift with rule changes, so an all-history
  baseline biases every multiplier.
- **Minutes & role-change (§3.2).** Expected counts scale by
  `tog_multiplier(projected, historical)` (recent-vs-baseline time-on-ground —
  the best proxy for projected minutes without a confirmed lineup), and a jump
  in a player's centre-bounce attendance (`cba_role_multiplier`, DFS data only)
  lifts his disposals — a wing moved into the centre square. Ruck-vs-ruck hitout
  matchups and opponent tagger flags are left for later (they need data the repo
  doesn't yet carry).

### Weather (plan §1.8, §3.4)

`afl_bot/data/venues.py` is a static venue table (city, coordinates, roof), and
`afl_bot/data/weather.py` pulls per-game daily rainfall + wind from Open-Meteo's
free keyless historical archive (a practical stand-in for BOM), caching to
parquet and degrading to "dry" on any network failure. `attach_weather(games)`
flags each game `is_wet` (rain ≥ `WET_THRESHOLD_MM` and open-air; Marvel/Docklands
is always dry under its roof).

Wet weather is also applied **inside the match sim** (`simulate_match(is_wet=)`):
`mu_total` is cut by `WET_TOTAL_MULTIPLIER` (~0.93, fitted) and shot accuracy by
`WET_ACCURACY_PENALTY`, so totals/margin/H2H move coherently with the wet props
in the same multi (goals then aren't double-discounted in the prop path). `is_wet`
is set automatically from an Open-Meteo **hourly forecast at bounce time**
(`forecast_game_rain`), with `--rain-mm` as a manual override; `fetch_hourly_archive`
supports refitting wet effects on in-game (not daily) rain.

`afl_bot/models/weather_effects.py` then applies per-stat wet multipliers via the
`context_mult` hook: disposals/marks down, tackles up. Price a wet round
with the CLI flag:

```
python -m afl_bot.cli run-round --year 2026 --round 14 --rain-mm 12 --odds odds.json
```

A worked example (Geelong at open-air Kardinia Park, 12mm): 20+ disposals
0.95→0.92, 5+ tackles 0.49→0.59, 6+ marks 0.16→0.07 — while a roofed-venue game
is unaffected.

**Honest caveat (calibration):** the `DEFAULT_RAIN_MULTIPLIERS` are
published-research figures for genuinely *wet play*. Fitting them from
Open-Meteo *daily* totals over 2022-25 games gives far weaker numbers (disposals
~0.99, marks ~0.96) because a daily total is a noisy proxy for conditions at the
bounce — rain can fall all morning and clear by an evening game. Marks are the
one stat with a clear empirical signal through that noise. So the defaults are
kept as the wet-play scenario the user opts into via `--rain-mm`, and
`fit_rain_multipliers` is available to refit when finer (in-game) weather lands.

### Boundary throw-ins / OOB market (plan §1.6c)

`afl_bot/models/stoppages.py` prices the boundary-throw-in (out-of-bounds) total
market. Per iteration the count is drawn NB, coupled *negatively* to the match
total (congestion lowers scoring and raises throw-ins, calibrated to corr ≈
−0.36) and lifted in the wet — so it stays coherent with the rest of the sim.
The CLI prices a "Total boundary throw-ins N+" line (dry P(36.5+) ≈ 0.47 →
12mm-wet ≈ 0.72).

**Data precondition (per the build order, "once team stoppage data flows").**
Boundary-throw-in counts aren't on Squiggle/AFL Tables, and AFL.com.au's
detailed team stats are Champion-Data-token-gated (the `aflapi.afl.com.au`
fixtures endpoint is open, but per-match stat endpoints 404 without a media
token; the full event feed needs a commercial licence). So the model currently
prices off a documented **league prior** (~36/game) — flagged "(prior)" in CLI
output — and `afl_bot/data/stoppages.py` is the validated plug-point
(`load_boundary_throwins`) where real per-game counts drop in (`expected_oob`
then uses their mean) once a feed is wired.

## Data layer

Every cached dataset under `data_cache/` is a parquet file plus a
`<name>.meta.json` sidecar recording a schema version, row count and columns
(`afl_bot/data/storage.py`). `read_parquet(name, expected_schema_version=...)`
warns if a cached file predates the schema a loader expects.

Team names differ across sources (AFL Tables historical names, Footywire
nicknames, 3-letter codes, ...). `afl_bot/data/teams.py` fixes one canonical
name per team (matching the Squiggle API) and provides
`normalize_team_name()` / `normalize_team_column()` so every new loader maps
onto the same names before joining with existing data.

For ad-hoc SQL across the whole cache (optional `duckdb` dependency):

```python
from afl_bot.data.storage import duckdb_connection

con = duckdb_connection()
con.sql("SELECT hteam, ateam, hscore, ascore FROM games_2025 LIMIT 5").show()
```

## Known limitations

- **Player box scores: combined real sources, 2012+.**
  `afl_bot/data/fryzigg.py` pulls full *past*-season per-player box scores
  (2012 onwards: kicks, handballs, behinds, marks, tackles, goals, hitouts,
  ruck contests, frees, TOG%, fantasy/SuperCoach scores) from the Fryzigg
  dataset, while `afl_bot/data/dfs_australia.py` pulls the *current* season
  with the same fields plus centre bounce attendances (CBA), which Fryzigg
  lacks. `load_player_log` (in `afl_bot/data/player_stats.py`) concatenates
  both into one player game log and falls back to a synthetic log if both are
  unavailable (e.g. offline, or `pyreadr` not installed for Fryzigg) — pass
  `--synthetic-props` to the CLI to force the synthetic log.
  Reading Fryzigg's RDS file requires the optional `pyreadr` dependency.
  Pre-2012 history (different team names/eras) is out of scope for now.
- **Live odds: h2h/totals only.** `round-report --live-odds` pulls live h2h +
  totals from The Odds API (`afl_bot/data/live_odds.py`, `ODDS_API_KEY` env
  var). Player-**prop** odds are not on the free tier, so prop prices still come
  from the `--odds` JSON (merged over live). `run-round` still takes odds from a
  JSON file; `afl_bot/data/odds.py` covers *historical* h2h/totals for
  backtesting (CLV). A live prop feed (paid add-on or scrape) is a future step.
- **Lineups not yet wired in.** `LegCandidate.confirmed` defaults to `True`;
  call `client.get_lineup(year, round)` and mark unconfirmed players'
  `confirmed=False` before building multis.

## Tests

```
pytest
```

Beyond unit tests, the suite includes the CI backtest (plan §5.3): loader
schema-contract tests (`tests/test_schemas.py`), Monte Carlo distribution-sanity
checks (margin/total sigma in the calibrated band), and a **golden-file**
walk-forward metric check (`tests/test_ci_backtest.py` vs
`tests/golden/backtest_metrics.json`) so a model change surfaces its accuracy
delta in CI. Regenerate the golden file deliberately when a change is expected.

## Responsible gambling

This is a modelling/analytics project. Even a well-calibrated model loses
regularly — manage variance and EV, it does not remove risk. Only stake what
you can afford to lose. Gambling Help Online: gamblinghelponline.org.au |
1800 858 858 (Australia, 24/7).

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

### Same-game-multi (SGM) walk-forward backtest (model-upgrade audit Phase 1)

Props are calibrated as singles and H2H is calibrated, but the **3-leg joint
probability** — the number actually bet — was never scored against history;
`corr_gain` (`joint_prob_from_masks` vs the naive product) was asserted from
the sim, never tested. `afl_bot/backtest/multis.py` closes that gap:
`walk_forward_multi_predictions` replays, round by round, the exact
match-construction + `search_match_sgms` ladder-selection logic
`round_report` uses live, fed only `games`/`player_log` rows strictly before
the round being predicted (no leakage — same anti-leakage contract as
`EloRatings.fit` / `walk_forward_prop_predictions`), then grades the selected
rungs' joint probability against what actually happened. Run it via:

```
python -m afl_bot.cli grade-multis --year 2025 --rounds 5,10,15,20
```

On 2025 rounds 5/10/15/20 (n=56 rungs): log loss 0.605, mean predicted 0.315
vs actual hit rate 0.268 — directionally overconfident, and the reliability
curve is **not flat**: the 0.4-0.6 predicted bucket lands at only 0.235 actual
(n=17). The sample is still small (a handful of rounds), but it's evidence,
not faith.

Simplifications vs the live path (kept honest, not hidden): no lineup/odds
book (every leg prices off the sim and is "confirmed"), no prop-probability
calibration (this backtest tests the raw sim joint probability calibration
would sit on top of), and no wet-weather flag (reliable historical
per-fixture rainfall isn't available).

### Fitting the correlation/dispersion constants from history (model-upgrade audit Phase 2)

`SCORE_SHOT_CORRELATION` / `PACE_SIGMA` / `SHARE_CONCENTRATION` /
`SHOT_DISPERSION` / `TEAM_STAT_DISPERSION` (`config.py`) drive `corr_gain` but
were hand-set. `afl_bot/backtest/correlations.py` estimates each from history
(closed-form method-of-moments for `SHOT_DISPERSION`; a closed-form joint
solve of `PACE_SIGMA`/`TEAM_STAT_DISPERSION` from a team volume-stat's
empirical mean/variance/home-away correlation; a closed-form
`SHARE_CONCENTRATION` from a top-usage cohort's empirical disposal CoV; a
root-find for `SCORE_SHOT_CORRELATION` since the Gaussian-copula-to-Pearson-
correlation map is nonlinear). Each estimator is unit-tested against data
simulated from the model's own functions at a known ground-truth value and
recovers it. Run via:

```
python -m afl_bot.cli fit-correlations --through 2024
```

**Honest finding: fitting these from history (through 2024) and re-running
the Phase-1 multi backtest (2025 rounds 3/6/9/12/15/18/21, n=106) makes
calibration *worse*, not better** — log loss 0.697 vs 0.617 for the config
defaults, Brier 0.248 vs 0.212. The standout mover is `PACE_SIGMA`: the
fitted value (~0.008) is ~9x smaller than the config default (0.07) because
the empirical home/away team-disposal correlation in the 2018-2024 sample is
essentially zero (~0.005), not the meaningfully-positive "fast open game
lifts both teams" effect the config constant assumes. `SHOT_DISPERSION`
(41.3 vs 42.5) and `SHARE_CONCENTRATION` (220.6 vs 200) land close to the
existing defaults; `SCORE_SHOT_CORRELATION` (-0.284 vs -0.32) is a modest
refinement. Per this instruction's own acceptance rule ("if it regresses,
the fit is wrong — keep the defaults"), **the fitted params are NOT wired
into `round-report`/`run-round`** — `load_fitted_correlation_params()` exists
and is opt-in (same contract as `load_fitted_elo_params`), but nothing calls
it automatically. The estimators are verified correct (synthetic recovery);
the open question is *why* the empirically-near-zero pace correlation
doesn't help the actually-bet multi ladder — plausibly Phase 1's sample
(n=106 graded rungs across 7 rounds) is still too small to distinguish
parameter settings reliably, or correlation isn't actually the dominant
source of the Phase 1 overconfidence (calibration fidelity, Phase 3, may
matter more).

### Acting on the overconfidence finding (model-upgrade audit Phase 2.5)

Phase 1's multi backtest grades the **raw** sim joint probability with leg
calibration off by design; live `round-report` applies per-stat isotonic
calibrators. So the question was whether the overconfidence found in Phase 1
lives in the uncalibrated legs (which calibration should fix) or in the
correlation structure (which Phase 2's fit already showed doesn't help).
`grade-multis` gained a calibration-ON mode: it additionally fits per-stat
prop calibrators on seasons strictly before each eval year (a separate cache
dir, never touching the live `prop_calibrators.json`), applies them to each
leg before rebuilding the joint (`calibrated_joint_prob = clip(calibrated_naive_product
+ corr_gain, 0, 1)` — calibration can't touch `corr_gain` itself, since that
comes from the sim's correlated masks, not `fair_prob`), and reports raw vs
calibrated reliability curves side by side:

```
python -m afl_bot.cli grade-multis --year 2024,2025 --n-sims 3000
```

**Honest finding, full 2024-2025 sample (n=835 rungs, 48 rounds — an order of
magnitude bigger than Phase 1's n=56/n=106):** calibration gives a small,
real improvement (log loss 0.5757 → 0.5680, Brier 0.1948 → 0.1913) but does
**not** flatten the curve. The top bucket (0.4-0.6 predicted) stays
substantially overconfident either way — raw 0.476 predicted vs 0.366 actual,
calibrated 0.458 vs 0.362 — almost the same gap. The low/mid buckets are
already close to flat both raw and calibrated (e.g. mid bucket: raw 0.286 vs
0.275, calibrated 0.264 vs 0.246), so the overall log-loss gain mostly comes
from calibration nudging more rungs out of the high bucket (n=246 raw →
n=235 calibrated) into the mid bucket (n=291 → n=484), not from fixing the
high bucket's miscalibration itself.

**Conclusion: leg calibration is *a* lever, not *the* lever.** It doesn't
fully explain Phase 1's overconfidence — the persistent high-bucket gap
points at calibration *fidelity*, not just calibration's presence: Phase 3's
plan (calibrate against the real sim output per (stat, line), not the
current pooled proxy-marginal calibrators used here) is the next thing worth
trying, not a new correlation re-fit (Phase 2 already ruled that out on a
smaller sample, and this larger run doesn't change that verdict — `corr_gain`
is untouched by either raw or calibrated columns here, by construction).

### Faithful prop calibration (model-upgrade audit Phase 3)

Three changes, in order:

**3.1 One line source of truth.** `PROP_LINES` moved from `cli.py` into
`config.py` — `cli.py`, `build/report.py`'s `projection_rows`, and
`backtest/multis.py` all import the same dict now. `backtest/props.py`'s
separate, narrower, stale `DEFAULT_PROP_LINES` (disposals `[15,20,25]` vs
live `[15,20,25,30,35]`, etc.) is deleted; `walk_forward_prop_predictions`
defaults to the live lines. A guard test
(`test_walk_forward_prop_predictions_defaults_to_live_prop_lines`) asserts
the backtested line set equals the live one so this can't silently drift
apart again.

**3.2 Per-(stat, line) calibration.** `fit_prop_calibrators` now fits one
`IsotonicCalibrator` per `(stat, line)` cell with at least
`PROP_CALIBRATION_MIN_SAMPLES` (200) walk-forward predictions, falling back
to the old pooled-per-stat curve for thinner cells (`apply_prop_calibration`
does the lookup-with-fallback; every call site — `run-round`, `round-report`,
`projection_rows`, the multi backtest — goes through it instead of a raw
`.get(stat)`). The calibrator cache JSON schema changed shape to hold both
curves per stat, with a `_version` field so a pre-Phase-3.2 cache is detected
and silently refit rather than mis-read.

**3.3 Calibrate against the real sim output.** `walk_forward_prop_predictions`
(used for the calibrators `round-report`/`run-round` load by default) is
still the closed-form shrunk-EWMA-NB *proxy* marginal — it doesn't see the
TOG/CBA/matchup multipliers, shared pace draw, Dirichlet share allocation, or
scoreline correlation the live sim actually prices through. New
`afl_bot.backtest.multis.walk_forward_sim_prop_predictions` reuses
`_build_match_legs` (the same per-round, anti-leakage leg construction the
multi backtest already trusts) to emit one row per gated candidate leg with
its **real sim** probability instead — same `fit_prop_calibrators` /
`apply_prop_calibration` consume either source unchanged. `grade-multis`
gained `--calibration-source {proxy,sim}` (default `proxy`, unchanged
behaviour) to compare them:

```
python -m afl_bot.cli grade-multis --year 2024,2025 --n-sims 3000 --calibration-source sim
```

**Honest finding (acceptance test, same n=835 sample as the Phase 2.5 run):
sim-sourced calibration does NOT pass the decision gate.** Log loss 0.5657
vs the proxy source's 0.5680 — a 0.002 difference, within noise for n=835,
not a real improvement. Worse: the sim-calibrated curve produces a **new**
high bucket (0.6-0.8 predicted) that didn't exist under proxy calibration —
predicted 0.684, actual only 0.433 (a +0.251 gap, the worst miscalibration in
either run). The proxy run's own high bucket (0.4-0.6: predicted 0.458 vs
actual 0.362, gap +0.096) is *not* meaningfully fixed by switching sources
either (sim's 0.4-0.6 bucket: 0.480 vs 0.407, gap +0.073 — similar order of
magnitude, smaller sample). A likely cause: `walk_forward_sim_prop_predictions`
is gated to `LEG_PROB_MIN`/`LEG_PROB_MAX` (mirroring live pricing), so its
per-`(stat, line)` sample is far smaller than the *ungated* proxy marginal's
— more cells fall back to the pooled curve, and the pooled curve itself has
less data to work with.

**Per the project's "if it regresses or doesn't help, keep the default"
rule (same precedent as Phase 2's correlation fit), `--calibration-source`
stays `proxy` by default; `sim` is opt-in, not wired into
`round-report`/`run-round`.** This is a real, if disappointing, result: the
persistent multi-ladder overconfidence found in Phase 1 is **not** simply a
calibration-fidelity problem — fitting against the true generative model's
own marginal doesn't flatten the curve more than the cheap proxy does. The
3.1/3.2 changes (single line source of truth, per-line calibration with
fallback) ship anyway since they're correct and free-standing improvements;
what doesn't ship is the assumption that 3.3 would fix the overconfidence.
Open question for whoever picks this up next: is the sim-source sample
(further thinned by the bettable-window gate) just too small, or is the
overconfidence structural — e.g. in `corr_gain` itself, which no calibration
source touches?

### Selection bias / optimizer's curse check (model-upgrade audit Phase 3.5)

Neither Phase 2 (correlation params) nor Phase 3 (calibration fidelity) explained Phase 1's
multi-ladder overconfidence. A third candidate explanation: `search_match_sgms` doesn't grade a
fixed prediction — for every match it builds potentially hundreds of non-conflicting >=3-leg
combos and picks, per target odds, whichever one's **fair odds happens to land closest to the
target** (tie-broken on highest joint prob). Argmax/closest-match selection over many noisy
estimates is a textbook biased estimator (the *optimizer's curse* / winner's curse) even when
every individual estimate is itself unbiased — picking the one that *looks* best (or closest) out
of many is more likely to have gotten there via favourable noise than genuine signal.

**Tooling.** `build/report.py`'s combo-construction loop is now its own function,
`build_sgm_candidates` (returns the full candidate pool, each entry's `n_sims` included for the
haircuts below); `search_match_sgms` calls it and selects from the pool unchanged. New
`afl_bot.backtest.multis.walk_forward_sgm_candidate_predictions` reuses the same per-round,
anti-leakage leg construction as the multi backtest to grade **every** candidate combo per match,
not just the 3 selected ones. `grade-multis` gained `--all-candidates` (prints both reliability
curves side by side) plus two opt-in selection-haircut prototypes: `--price-shrink F` (shrink the
selected rung's joint prob toward its target odds' implied probability by factor `F`) and
`--lcb-z Z` (rank candidates by a lower-confidence-bound estimate, `joint_prob - Z * MC standard
error`, computed in probability space, instead of the raw point estimate).

```
python -m afl_bot.cli grade-multis --year 2024,2025 --n-sims 3000 --no-calibration --all-candidates
```

**Finding 1 (the hypothesis): confirmed.** On the full 2024-2025 sample, the candidate population
is enormous (n=29,593,140 candidate combos vs 835 selected rungs) and its overall calibration looks
very different from the selected population's — but raw `n`/mean-pred aren't comparable (the
ladder deliberately targets specific high-ish probabilities; the candidate pool is dominated by
long-shot combos near the 0.05 floor). The valid comparison is the **matched probability bucket**
present in both (0.4-0.6 predicted):

| | predicted | actual | gap |
|---|---|---|---|
| Selected rungs (n=246) | 0.476 | 0.366 | **+0.110** |
| All candidates (n=87,432) | 0.423 | 0.358 | **+0.065** |

The selected rungs are ~70% more overconfident than the general candidate population at the same
nominal probability level. `search_match_sgms`'s selection mechanism is itself inflating the
joint-probability estimates of the rungs it picks — the optimizer's curse is real here, not just a
theoretical concern.

**Finding 2 (the prototyped fixes): both fail real-data acceptance.**

- `--price-shrink 0.3`: log loss got **worse** (0.5796 vs the unhaircut baseline's 0.5757); the
  0.4-0.6 bucket's gap *widened* to +0.139. Cause: the anchor (`1/target_odds`) isn't a calibrated
  probability estimate, it's a pricing decision — for the bucket that's dominated by the safest
  ($1.75, ~57% implied) rung, the truth (36.6%) sits *below* both the raw estimate (47.6%) and the
  anchor (57.1%), so shrinking toward the anchor pushes the estimate further from truth, not closer.
  The `MULTI_MARKET_SHRINK` mechanic this was modelled on works because it shrinks toward the
  book's own *probability estimate* (devigged market price); a fixed target odds has no such
  guarantee of being closer to truth than the model's own estimate.
- `--lcb-z 1.0`: log loss also got worse (0.5949 vs 0.5757) — but more tellingly, the problematic
  0.4-0.6 bucket was **completely unchanged** (still 0.476 pred / 0.366 actual, n=246 exactly) —
  the haircut didn't even alter which combo got selected there. It changed selection only at the
  lower rungs, and that change made things worse on net. (Unit-tested in isolation the mechanism
  does change selection in a constructed scenario — see `test_search_match_sgms_lcb_z_can_change_the_selected_combo`
  — it's just not doing anything useful at the specific target/candidate-pool geometry that matters here.)

**Conclusion: the diagnosis (selection bias contributes to the overconfidence) is correct, but
neither prototyped haircut fixes it, so per the project's "if it doesn't help, keep the default"
rule, `--price-shrink`/`--lcb-z` stay at 0.0 (off) everywhere, not wired into `round-report`/
`run-round`.** Both ship as opt-in CLI knobs for further experimentation, not as a shipped fix. A
more promising next idea, not implemented here: fit an isotonic recalibration map *on the
all-candidates population* (which is large and much closer to flat) and apply it to the selected
rungs' joint probabilities — a "selection-level" calibrator analogous to Phase 2.5/3's leg-level
calibrators, rather than a hand-picked shrink target or ranking tweak.

### Selection-level isotonic recalibration (model-upgrade audit Phase 3.6) — also fails

Phase 3.5 ended on the idea of fitting an isotonic map on the population actually bet (the selected
rungs), not the all-candidates pool, since the selected-rung track record is what `round-report`
ultimately needs corrected. `fit_multi_calibrator`/`load_or_fit_multi_calibrator` in
`afl_bot/backtest/multis.py` do exactly that: walk-forward backtest `MULTI_CALIBRATION_LOOKBACK`
(5) prior seasons of selected-rung predictions, fit an `IsotonicCalibrator` from predicted
`joint_prob` to `all_hit`, and cache it to disk. `apply_multi_calibration` in `build/report.py`
applies the map to a match's searched SGMs (recomputing `fair_odds`/`edge`/`raw_edge` from the
calibrated probability). Wired in as an opt-in `--multi-calibration` flag on both `round-report`
and `grade-multis`, off by default.

**Real-data acceptance test** (`grade-multis --year 2024,2025 --n-sims 3000 --multi-calibration`,
calibrator fit on 2019-2023 walk-forward selected rungs per eval year):

| | n | log loss | brier | mean pred | actual hit rate |
|---|---|---|---|---|---|
| SELECTED (baseline) | 835 | 0.5757 | 0.1948 | 0.311 | 0.268 |
| MULTI-CALIBRATED | 835 | **0.5836** | **0.1982** | 0.339 | 0.268 |

Both acceptance criteria fail: log loss got worse, not better, and mean predicted probability moved
*further* from the actual hit rate (0.339 vs baseline's 0.311), not closer. The calibrator also
collapsed the bucket structure — the previously separate 0.2-0.4 (n=291) and 0.4-0.6 (n=246) buckets
merged into one n=537 bucket post-calibration, so the original "+0.110 gap" isn't even cleanly
comparable post-fix, but the aggregate trend is unambiguously negative.

**Conclusion: this is the third consecutive failed fix attempt for the multi-ladder overconfidence**
(after Phase 3's sim-based calibration source and Phase 3.5's price-shrink/LCB haircuts). Per the
project's "if it doesn't help, keep the default" rule, `--multi-calibration` stays off everywhere,
not wired into `round-report`/`run-round`'s default path. It ships as a tested, opt-in knob for
further experimentation. The recurring failure across three structurally different fixes (a better
leg-probability source, a selection-mechanism tweak, and a selection-level recalibration) suggests
the overconfidence may not be fixable by post-hoc adjustment of the selected-rung probabilities at
all — it may be non-stationary across seasons (a fit on 2019-2023 doesn't transfer to 2024-2025), or
the true bias may live further upstream, e.g. in the correlation/dispersion model (`corr_gain`)
itself rather than in calibration or selection.

### Manual prop market-blend, STEP 1+2 (model-upgrade audit Phase 4)

`PHASE-4-CODE-PLAN.md` supersedes the API-dependent Phase 4 sketched earlier: there is no free,
stable feed for AFL player-prop odds (The Odds API only carries them on its paid Business tier),
so Phase 4's value — anchoring prop legs to the market — is built on the **manual `--odds` file**
a weekly bettor already fills in by hand, not an automated feed (which stays parked, STEP 4). STEP
1 makes that path clean and first-class; STEP 2 does the actual blend; STEP 3 (acceptance) follows
in a later commit.

- **Odds template.** Every `round-report` run now also writes
  `reports/<year>_r<N>_odds_template.json`: every priceable leg's exact name (both H2H sides, the
  totals leg, every prop line that cleared the `LEG_PROB_MIN`/`LEG_PROB_MAX` gate) mapped to
  `null`, plus a `_rules` stub (`h2h_draw`, the same key `run-round` already consumes, so one
  filled-in file works for either CLI). Fill in the bookie's numbers and pass the file straight
  back via `--odds` — copy-paste, not retyping leg names from scratch, which kills the typo class
  of bug at the source (`build_odds_template` in `build/report.py`).
- **Totals are now a real leg.** `round-report` priced no totals market at all before this step —
  only H2H + props had `LegCandidate`s. There's now a `"Total points {line}+"` leg (same name
  `run-round` already uses), but **only once it has a real price in `--odds`** — unlike H2H/props
  it has no model-derived fallback price, specifically so a no-`--odds` run's leg set (and
  therefore its SGM ladder) stays byte-for-byte unchanged.
- **Single-leg visibility for priced props.** A prop's market price previously only showed up
  inside the SGM ladder table or the cross-game multi section — there was no single-leg view of
  what a book price does to a prop's classification/edge. A new "Priced props (from --odds)" table
  renders per match for any prop with at least one side priced, showing model prob, book price,
  devig prob (see below), edge, and ANCHOR/VALUE/SKIP classification.
- **Devig.** `devig_prop_leg` (`pricing/edge.py`) devigs a prop's price: if **both** "+" (over) and
  the new "-" (under, e.g. `"Tom Mitchell 25- disposals"`) sides are entered, the devig is exact
  (`devig_proportional`, no assumption needed). If only one side is entered (the common case — the
  template only prompts for the "+" side; "-" is optional manual-entry for a cleaner devig), it
  falls back to `implied_prob(odds) / PROP_ASSUMED_OVERROUND` (default 1.06) and is labelled
  **"single-sided (approx)"** everywhere it's shown, so it's never mistaken for a clean two-way
  devig. `PROP_ASSUMED_OVERROUND` is a documented prior (typical AFL prop overround), not fitted —
  there is no historical prop-odds archive to fit it against (the same honesty constraint that
  applies to the market-blend weight in STEP 2).
- **Unmatched-key warning.** `round-report` now warns (same as `run-round`) on any `--odds` key
  that matched no priceable leg — covers the typo case for H2H/totals/prop "+" names and the
  optional prop "-" (under) names.

STEP 1 is mechanical plumbing only — it doesn't change any probability/edge. **STEP 2 does the
actual blend:**

- **Per-leg blend.** Whenever a prop has a book price, its CALIBRATED model probability is pulled
  `PROP_MARKET_BLEND_WEIGHT` (default 0.6) of the way toward its devigged market probability —
  reusing the existing `market_anchored_prob(prob, odds, weight)` pull (same mechanic
  `MULTI_MARKET_SHRINK` already uses at the multi level) by converting the devigged probability
  back to an equivalent "odds" value first (`fair_odds(devig_prob)`), so `market_anchored_prob`'s
  internal `implied_prob()` call recovers it exactly — no new blend math duplicated. The leg is
  then priced/classified on this **blended** probability, not the raw model probability.
- **Honest weight, not a fitted one.** Unlike the H2H ensemble blend (`fit_market_blend`, fitted
  out-of-sample on the historical odds archive), there is **no historical prop-odds archive** to
  fit `PROP_MARKET_BLEND_WEIGHT` against — it is a deliberate prior leaning toward the market
  (props are noisy, the market is sharp), documented as such in `config.py`, not a backtested
  optimum. The STEP 2.4 snapshot below is the cheapest path to one day fitting it for real.
- **Edge on the blend; VALUE still gated on real prices.** A leg's `edge_pct`/classification are
  now computed on the blended probability. Single-leg VALUE was already implicitly gated on having
  a real price (an unpriced leg's `market_odds` falls back to its own `fair_odds`, giving exactly
  zero edge); multi-level VALUE was already gated on `build_sgm_candidates` requiring **every** leg
  in a combo to have a book price before it gets a `book_odds`/`edge` field at all. Both gates
  predate this step but are now covered by a dedicated test
  (`test_build_sgm_candidates_no_edge_unless_every_leg_in_combo_is_priced`) rather than assumed.
  The SGM ladder's own selected-rung joint probabilities (from the correlated sim's per-iteration
  masks) are **untouched** by this per-leg blend — `MULTI_MARKET_SHRINK` still applies on top, at
  the joint level, unchanged.
- **Odds snapshot.** Every run with a filled `--odds` file also writes
  `reports/<year>_r<N>_odds.json` (a copy of exactly what was typed in) — the cheapest path to the
  repo accumulating its own prop-odds history, the archive that doesn't exist yet to fit the blend
  weight against.

**Set expectations accordingly (per PHASE-4-CODE-PLAN.md's own framing):** this mitigates
overconfidence on the legs priced by hand; it does **not** touch the joint/SGM-level overconfidence
that three separate fixes (Phase 3, 3.5, 3.6) already failed to move — that bias lives in the
correlated sim itself (masks), which this per-leg blend never touches.

**STEP 3 — acceptance.** With no historical prop-odds archive, this **cannot** be a backtest/log-loss
claim, and STEP 3 doesn't pretend otherwise — acceptance here is mechanical correctness, checked
both by unit test and by one real `round-report --year 2026 --round 15` run (the live upcoming
round at the time of writing) with a hand-filled `--odds` file:

- **(a) Blended prob lands between model and devigged market; edge computed on it — worked
  example.** `Nick Daicos 1+ goals` priced at book odds 1.40 (over) / 3.40 (under):

  | | value |
  |---|--:|
  | Model (calibrated) | 69.3% |
  | Devig (two-way, both sides entered) | 70.8% |
  | Blended (`PROP_MARKET_BLEND_WEIGHT` 0.6) | 70.2% |
  | Edge on the blend (book 1.40) | −1.7% |
  | Edge on the raw model (no blend, for comparison) | −2.9% |

  The blend sits between 69.3% and 70.8% as required, closer to the market (weight 0.6), and the
  edge moves with it (−1.7% vs the unblended −2.9%) — confirmed live, not just in the unit test
  (`test_market_anchored_prob_via_fair_odds_of_devig_prob_lands_exactly_between`).
- **(b) VALUE only when every leg is priced — confirmed live.** In the same run, with only
  `Total points 165.5+` and `Collingwood to win` priced (not the other two legs in every combo),
  the SGM ladder's rungs that included the priced totals leg still showed `Book`/`Edge` as `-`
  because their *other* legs in the same 3-leg combo had no price — exactly the existing
  `build_sgm_candidates` "all-or-nothing" gate (STEP 2.3), reproduced outside the unit test.
- **(c) No `--odds` -> byte-for-byte unchanged.** Verified by code-path inspection rather than a
  live two-run diff (the live data feed itself drifts minute-to-minute mid-round, which would
  make a raw diff a false negative, not a real regression check): `total_leg_name` is only ever
  appended to `match_legs` `if total_leg_name in odds_book`; `blended_prob` is only ever set away
  from the raw model `prob` `if over_odds or under_odds`; and `priced_legs` (the only new
  `render_markdown` section) is only ever non-empty when at least one such price exists. All three
  conditions are `False` whenever `odds_book` is empty (no `--odds`/`--live-odds`), so every new
  code path is a no-op and the rendered markdown is identical to pre-Phase-4 output — also covered
  directly by `test_render_markdown_no_priced_legs_section_when_empty`.
- **(d) Template round-trips with zero unmatched-key warnings.** The live run's freshly emitted
  `*_odds_template.json` keys were copy-pasted (4 of its 306 priceable legs filled in with prices)
  and passed straight back via `--odds` — `round-report` printed no "unmatched leg" warning.
- **(e) `pytest -q` green** with new unit tests for the devig (`test_devig_prop_leg_*`), the blend
  (`test_market_anchored_prob_via_fair_odds_of_devig_prob_lands_exactly_between`), and the VALUE
  gate (`test_build_sgm_candidates_no_edge_unless_every_leg_in_combo_is_priced`). 288 tests.

H2H's own ensemble blend (`fit_market_blend`) and the rest of the multi pricing path are unchanged
when no prop odds are supplied — this phase only ever adds a probability pull on legs that already
have a real price.

### corr_gain diagnostic (model-upgrade audit Phase 4, parked diagnostic) — confirmed

PHASE-4-CODE-PLAN.md parks a second idea alongside the manual market-blend: a cheap, **no-odds-needed**
test of whether the sim's correlation structure itself (`corr_gain = joint_prob - naive_product`,
from the copula/pace/Dirichlet machinery in `simulate_match`) is the thing systematically inflating
the multi ladder, rather than calibration or selection (both already ruled out, Phases 3/3.5/3.6).
The test: bucket the SELECTED rungs by predicted `joint_prob` (`walk_forward_multi_predictions`'s own
output) and compare the sim's `corr_gain` to the **empirical** corr_gain — actual joint hit-rate minus
the product of *actual* per-leg hit-rates, estimated by pooling every individual leg outcome within a
bucket (`n_legs_hit`/`n_legs`, new columns on the predictions frame) and cubing the pooled rate (every
rung is a 3-leg combo). `corr_gain_diagnostic` in `afl_bot/backtest/multis.py` does this; `grade-multis
--corr-gain-diagnostic` prints it alongside the existing SELECTED reliability curve. Diagnostic only —
no haircut is applied.

```
python -m afl_bot.cli grade-multis --year 2024,2025 --n-sims 3000 --no-calibration --corr-gain-diagnostic
```

**Real-data result (2024-2025, the same n=835 selected-rung sample every prior phase used):**

| bucket | n | sim corr_gain | empirical corr_gain | gap (sim − empirical) |
|---|--:|--:|--:|--:|
| (0, 0.2] | 298 | +0.006 | **−0.044** | **+0.050** |
| (0.2, 0.4] | 291 | +0.009 | +0.005 | +0.003 |
| (0.4, 0.6] | 246 | **+0.022** | **−0.002** | **+0.024** |

**Confirmed: the sim systematically overstates the correlation lift, in every bucket.** The gap is
positive everywhere (the sim's corr_gain is never *smaller* than the empirical one), so this isn't
noise cancelling out across buckets — it's a consistent direction. Two distinct shapes, though:

- **Low bucket (n=298, the biggest sample): the sim has it backwards.** Empirically these legs are
  *negatively* correlated (corr_gain −0.044 — actual joint hit-rate sits well below what independent
  legs with these same actual hit-rates would predict), but the sim says they're mildly *positively*
  correlated (+0.006). The biggest absolute gap in the table, on the largest sample.
- **High bucket (n=246, the bucket every previous phase has flagged as the persistently overconfident
  one — 0.476 predicted vs 0.366 actual): the sim invents lift that isn't there.** Empirical corr_gain
  is ~0 (−0.002, i.e. these legs behave almost exactly like independent draws in reality), but the sim
  adds +0.022 of correlation lift on top of its own (already-uncalibrated) naive product. That +0.022
  is a real, identified slice of this bucket's overconfidence, sitting upstream of every lever the
  previous three phases pulled (leg calibration, selection mechanism, selection-level recalibration).
- The middle bucket (n=291) is the one place sim and empirical roughly agree (gap +0.003, indistinguishable
  from noise at this sample size) — the bias isn't uniform across the probability range.

**Proposed follow-up (not implemented here — diagnostic only): a `corr_gain` haircut.** Shrink the
correlation term itself, before it's added to the naive product, rather than recalibrating the final
joint probability (Phase 3.6's approach, which failed) or fiddling with selection (Phase 3.5's approach,
which also failed) — i.e. target the piece of the pipeline this diagnostic just pinned the bias in.
Concretely: `joint_prob_haircut = naive_product + CORR_GAIN_HAIRCUT * corr_gain`, with
`CORR_GAIN_HAIRCUT` (a new `config.py` tunable, default 1.0 = unhaircut/current behaviour) fit by
walk-forward moment-matching (mirroring Phase 2's correlation-parameter fits, not a gradient search) —
e.g. the ratio of total empirical corr_gain to total sim corr_gain across the selected-rung sample, or
a per-bucket/isotonic map if a single global constant doesn't fit the low and high buckets' very
different gap shapes well simultaneously. **Honest risk going in:** Phase 3.6 already showed that
fitting *any* map on this same thin, multi-season, possibly non-stationary selected-rung sample doesn't
reliably transfer from training years to eval years — a `corr_gain`-specific haircut is a more targeted
intervention (it touches the actual mechanism this diagnostic identified, not just the output), but it
is not guaranteed to fare better than Phase 3.6 did against the same sample-size/non-stationarity risk.
Worth trying before further calibration/selection-layer attempts, given this is the first diagnostic to
actually localise *where* in the pipeline the persistent high-bucket overconfidence is coming from.

### corr_gain haircut (the proposed follow-up, implemented) — CLOSED, now the live default

Implemented exactly as proposed: `search_match_sgms`/`walk_forward_multi_predictions` gained an opt-in
`corr_gain_haircut` param (default 1.0 = unhaircut) that reprices a SELECTED rung as `naive_product +
corr_gain_haircut * corr_gain` instead of the raw sim `joint_prob`, clipped to `[0, 1]`
(`naive_product`/`corr_gain` themselves stay at their pre-haircut, informational values — same
convention as `price_shrink`). New `CORR_GAIN_HAIRCUT` config tunable (1.0, opt-in everywhere). New
`afl_bot.backtest.multis.haircut_joint_prob`/`fit_corr_gain_haircut` compute and grid-search-fit
candidate haircut values **directly off an already-backtested DataFrame's `naive_product`/`corr_gain`
columns** — no second walk-forward sim pass needed per candidate value, since those columns are always
the raw (pre-haircut) sim decomposition regardless of what `corr_gain_haircut` the run itself used.
`grade-multis --corr-gain-diagnostic` also prints the in-sample fitted value; `--corr-gain-haircut F`
re-prices the run with a chosen value, both wired through to `round-report` too.

**Out-of-sample test (the actual acceptance check — fit on one year, evaluate on the other, in both
directions, on the same 2024-2025 sample):**

| | fitted haircut | OOS log loss | vs baseline (1.0) | vs zero-lift (0.0) |
|---|--:|--:|--:|--:|
| Train 2024 (n=459) → test 2025 (n=376) | **+0.200** | fitted: 0.5407 | 0.5454 | **0.5396 (wins)** |
| Train 2025 (n=376) → test 2024 (n=459) | **−0.500** | fitted: 0.6003 | 0.6004 | **0.6000 (wins)** |

**The fitted coefficient does not generalize — and that instability is itself the headline finding.**
Fitting on 2024 gives a modest positive shrink (+0.200); fitting on 2025 gives a *sign-flipped*
overcorrection (−0.500) — the same non-stationarity that sank Phase 3.6's selection-level calibrator
shows up again here. Worse, each fitted value *underperforms the simple zero-lift candidate when
evaluated on the other year* — the in-sample combined-sample fit (−0.500, log loss 0.5719, the
best-looking number in isolation) is the overfit one, exactly the trap flagged as a risk before running
this. **The simple zero-lift variant (`corr_gain_haircut=0.0`, no fitting at all) wins out-of-sample in
*both* directions**, beating both the baseline and the fitted coefficient every time it's evaluated on
a year it wasn't tuned on.

**Combined 2024+2025 sample (n=835) under the winning zero-lift candidate, vs baseline:**

| | log loss | brier | mean pred | hit rate | high bucket (0.4-0.6) pred / actual / gap |
|---|--:|--:|--:|--:|--:|
| baseline (1.0, unhaircut) | 0.5757 | 0.1948 | 0.311 | 0.268 | 0.476 / 0.366 / **+0.110** |
| **zero-lift (0.0)** | **0.5728** | 0.1934 | 0.300 | 0.268 | 0.455 / 0.365 / **+0.090** |

**Acceptance: log loss criterion PASSED, high-bucket-gap criterion PARTIALLY met.** Log loss beats the
0.5757 baseline (0.5728, a real if modest improvement) and the high-bucket gap narrows by ~18% (+0.110
→ +0.090) — but it does not *close* (a meaningful gap remains; pricing purely off the naive/independent
product still isn't enough to fully explain this bucket's actual hit rate). This is, honestly, the
**first attempt across Phase 3/3.5/3.6/this one that shows a genuine, two-directional out-of-sample
improvement** rather than a flat failure — but it's a partial, not complete, fix.

**Stacking with per-leg prop calibration — the closing result.** The `corr_gain_haircut=0.0` backtest
above was run with per-leg prop calibration **OFF** (Phase 2.5/3's lever, separate from this one and
already always-on in `round-report` itself). Since the diagnostic's own numbers hinted some of the
high bucket's overconfidence sits in the per-leg marginals feeding `naive_product`, not just in
`corr_gain`, the two existing levers were tested **stacked** (per-leg calibration ON + `corr_gain_haircut
=0.0`) on the same 2024-2025 selected-rung sample, rather than building a third, dedicated fix first:

| | log loss | high bucket (0.4-0.6) gap |
|---|--:|--:|
| baseline (no calibration, unhaircut) | 0.5757 | +0.110 |
| `corr_gain_haircut=0.0` alone | 0.5728 | +0.090 |
| **calibration ON + `corr_gain_haircut=0.0` (stacked)** | **0.5650** | **+0.051** |

**The two levers are complementary, not redundant** — stacked, they roughly double the improvement
either gives alone (gap narrows ~54% from baseline vs ~18% for the haircut alone), the best result of
the entire Phase 3/3.5/3.6/Phase-4 overconfidence investigation. A third lever was also tried before
settling here: a flat **per-leg marginal haircut** (shrinking the calibrated `naive_product` itself by a
constant factor, `naive_product * (1 - MARGINAL_HAIRCUT)`, mirroring `price_shrink`'s 0=off convention)
on top of the stack, OOS-tested the same way (fit on one year, test on the other, both directions). It
**did not replicate OOS** — it won in one direction and lost in the other, and even the in-sample
combined-sample picture showed the high-bucket gap widening slightly under the best-looking candidate —
so, per the same "don't ship what doesn't survive OOS" standard the corr_gain fit itself was held to, it
was never written into production code.

**Decision: the stack ships as the live default, and the investigation closes here.** Per-leg prop
calibration was already unconditionally on in `round-report`; `CORR_GAIN_HAIRCUT` in `config.py` is now
**0.0** (was 1.0) and is wired as the actual default for `round-report`'s and `grade-multis`'s own
`corr_gain_haircut` parameter (previously the constant existed but nothing referenced it — every caller
had its own independently hardcoded `1.0`). `search_match_sgms`/`walk_forward_multi_predictions`
themselves keep their own bare default at 1.0/unhaircut (these are general-purpose, reused by tests and
diagnostics that need the raw sim decomposition) — the validated `0.0` is a live-default choice made by
the callers that ship it, not a change to the low-level mechanism. Pass `--corr-gain-haircut 1.0` to
either CLI command to recover the original raw/unhaircut baseline for comparison.

The remaining **~+0.05 high-bucket gap is now a known, bounded residual** — not a flaw to keep chasing
with more fitted maps (Phase 3.6, the fitted corr_gain coefficient, and the marginal haircut all tried
that path and failed OOS, the consistent failure mode being non-stationarity across seasons on this thin
selected-rung sample), but an accepted, quantified limit managed downstream by the existing **staking**
(`KELLY_FRACTION`/per-bet/per-round caps already size positions conservatively against estimation error)
and **market-anchoring** (`market_anchored_prob`/`MULTI_MARKET_SHRINK` already pull the priced edge back
toward the book's own number) machinery, rather than something the joint-probability estimate itself
needs to fully resolve before the ladder is usable.

### Bettable-legs book menu (FIX-BETTABLE-LEGS-AND-PRICING)

**Problem.** `round-report` was building a `LegCandidate` for every (player, stat,
line) in `PROP_LINES` that cleared the `LEG_PROB_MIN`/`MAX` gate, with no check
against what a bookmaker would actually post. With no `--odds` entered, the whole
ladder was model-invented legs, including phantoms like "key defender 5+ marks" —
Australian books rarely post a marks/tackles market on a key defender at all.

**Investigated alongside this: was the `corr_gain_haircut` default (closed out
above) actually reaching `round-report`?** A live r16 rung showed `Joint 36%` /
`Corr gain +15.6pp`, which read like `naive_product (~20%) + the full corr_gain
lift` — i.e. the haircut not applying. Tracing it end to end (`config.CORR_GAIN_HAIRCUT`
→ `round_report`'s own default → `search_match_sgms`'s `corr_gain_haircut` param →
the haircut block) and instrumenting the actual call confirmed `joint_prob` *does*
equal `naive_product` with the default unhaircut (`corr_gain_haircut=0.0`) — the
naive product for that combo is genuinely ~36% (three high-probability marks legs),
not ~20%. `corr_gain` in the table is **always the pre-haircut, informational** value
(documented in `search_match_sgms`'s own docstring) so it doesn't move when the
haircut is applied — reading it as "what got added to the displayed joint" was the
misdiagnosis. This was already locked in by
`test_search_match_sgms_corr_gain_haircut_zero_lift_equals_naive_product` and
`test_round_report_and_grade_multis_default_to_the_validated_corr_gain_haircut`
before this fix; no code change was needed here, only the investigation.

**Fix (the real bug): a "book menu" filter for MODEL-ONLY legs.** A leg with a
*real* book price (`--odds`/live odds) is always kept — a posted market is bettable
by definition. A leg with **no** price now only survives if it passes
`is_bookable_model_only_leg` (`afl_bot/build/report.py`):
- the line is on `config.BOOKABLE_PROP_MENU` (drops standalone 35+ disposals etc.),
- the player is in the top `BOOKABLE_TOP_N_BY_STAT[stat]` projected players for that
  stat on their team (`top_n_players_by_stat`) — books price the obvious names, and
- for marks/tackles specifically, the player's inferred role (`classify_roles`) is
  in `BOOKABLE_MARKS_ROLES`/`BOOKABLE_TACKLES_ROLES` — key defenders are excluded.

Applied only to `round_report`'s own live leg construction; `afl_bot/backtest/multis.py`
(the walk-forward SGM backtest) deliberately keeps building legs the old way so its
validated calibration numbers stay comparable — the menu is a live-ladder UX/honesty
fix, not a pricing-model change. The predictions sidecar (`reports/*_predictions.csv`,
used for `grade-round`) still records every leg that clears `LEG_PROB_MIN`/`MAX`
regardless of the menu, so calibration grading is unaffected.

Rungs with no full book price on every leg (`"book_odds" not in combo`) are now
tagged `(model-only — verify market exists)` in the ladder table so a phantom can
never be mistaken for a confirmed, priced bet — `VALUE PICK`/edge were already gated
to fully-priced combos (`build_sgm_candidates`/`search_match_sgms`,
`test_build_sgm_candidates_no_edge_unless_every_leg_in_combo_is_priced`), and
`round-report` itself never calls the staking module, so no stake was ever at risk
of being sized off an unpriced leg.

### Ladder target odds: real combo prices + a band label (FIX-LADDER-TARGET-ODDS)

**Retraction (from the fix above):** the `corr_gain_haircut` claim was wrong — Ben
confirmed commit `2c9028c`'s investigation was correct (the haircut IS applied;
`corr_gain` in the table is always the pre-haircut informational lift). Not touched
again here.

**Problem this fix addresses.** An old report (`2026_r14_report.md`) had every rung
pinned to *exactly* $1.75 / $2.50 / $3.50 — a cosmetic `price_shrink` clamp to the
target, not the combo's real odds. With that clamp off, r15/r16 showed the honest
fair odds (e.g. $2.18 / $3.39 / $4.37), which don't read as clean bet-slip numbers,
and the bottom rung could land *shorter* than its target (e.g. ~$1.50 instead of
~$1.75) when same-game correlation pushed a combo's real joint probability above its
naive product. Ben wants both: the real, honest odds AND a target-band label per
rung, with the bottom rung never landing short.

1. **Targets**: `MULTI_TARGET_ODDS` is now `(1.75, 3.00, 5.00)` (was `3.50` mid).
2. **A real ~$1.75 rung needs near-lock legs.** Three legs each capped at
   `LEG_PROB_MAX` (0.78) cap the naive product at `0.78^3 ≈ 0.475` (fair odds
   ~$2.11) — a genuine $1.75 multi is mathematically unreachable below that without
   admitting higher-probability legs. `SGM_LADDER_LEG_PROB_MAX = 0.95` is a second,
   wider cap used ONLY when building the SGM ladder's own candidate pool (`cli.py
   round_report`'s leg-construction loop gates `match_legs` on
   `LEG_PROB_MIN < prob < SGM_LADDER_LEG_PROB_MAX` instead of `< LEG_PROB_MAX`).
   These near-lock legs (e.g. a top mid's 15+/20+ disposals) feed the multi pool
   ONLY — `LEG_PROB_MAX` (0.78) still gates the predictions CSV (`grade-round`
   calibration tracking) and single-leg ANCHOR/VALUE/SKIP classification unchanged.
   The book-menu filter (above) still applies to these legs the same as any other
   model-only leg.
3. **Selection now lands at-or-above each target, never short.** `search_match_sgms`
   (`afl_bot/build/report.py`) used to pick the combo with the smallest *absolute*
   `|fair_odds - target|`, which could pick a combo whose real joint probability
   (lifted by same-game correlation) made it shorter than the target even though a
   longer, still-close combo existed. `_select_for_target` now prefers, for each
   target, the combo whose odds are **at or longer** than the target (closest from
   above); only falls back to the closest combo below the target when nothing
   reaches it. Scoped to `lcb_z<=0` (round-report's own path) so it doesn't
   short-circuit the Phase 3.5 `lcb_z` selection-haircut diagnostic, which relies on
   pure closest-distance picking to demonstrate its effect
   (`test_search_match_sgms_lcb_z_can_change_the_selected_combo`). Each selected
   rung is tagged with its own `target_odds` (the band it filled).
4. **Display stays honest.** `price_shrink` stays `0.0` (off) — `Joint prob`/`Fair
   odds` are always the combo's real numbers. A new **Band** column shows the target
   each rung filled (`$1.75` / `$3.00` / `$5.00`) right next to them, so the bet-slip
   number and the honest price are both visible.

**Side effect worth knowing:** `grade-multis` calls the same `search_match_sgms`
with its own `lcb_z` default of `0.0`, so the "land at-or-above" selection change
applies there too by default, not just to `round-report` — unlike the book-menu
filter above (deliberately scoped to `round_report` only), this is a fix to
`search_match_sgms`'s own selection mechanism. The historical OOS validation
numbers in the corr_gain-haircut writeup above were captured before this change;
they were not re-run as part of this fix (out of scope here) since neither the
naive product nor the joint probability themselves changed, only which combo gets
selected when an above-target option exists.

### Placeable legs + honest $2.10 floor (FIX-PLACEABLE-LEGS-AND-210-FLOOR) — retires the near-lock anchor idea above

**Two linked problems Ben found in `reports/2026_r16_report.md`.** (A) the ladder
offered legs no book posts a market on — e.g. "Lachie Whitfield 15+ disposals"
(95%) — because `SGM_LADDER_LEG_PROB_MAX = 0.95` admitted near-lock legs into the
multi pool ONLY, exactly the mechanism point 2 above describes; those are phantom
legs, not placeable bets. (B) the multi price could still drift short of its band
(a $1.75-band rung printing **$1.59** Fair odds) because `_select_for_target`'s
"never land shorter" guard (point 3 above) checked the combo's **pre-haircut**
joint probability, while the printed number was the **post-`corr_gain_haircut`**
one — in the r16 GWS case a combo with negative `corr_gain` (-5.6pp) had a raw
joint of 57% (passing the $1.75 guard) that the haircut (the live
`CORR_GAIN_HAIRCUT=0.0` default, pricing off `naive_product` alone) pushed up to
63%, printing $1.59. The root insight: FIX-LADDER-TARGET-ODDS admitted near-lock
legs *on purpose* so a real ~$1.75 multi could exist (`0.78³ ≈ $2.11` otherwise)
— but those near-lock legs are exactly the unplaceable phantoms from (A). You
can't have both; Ben's call: **drop the phantoms, accept the honest floor.**

1. **One leg-probability cap everywhere.** `SGM_LADDER_LEG_PROB_MAX` is retired —
   the SGM ladder pool now uses the same `LEG_PROB_MIN < prob < LEG_PROB_MAX`
   (0.30/0.78) gate as single-leg classification and the predictions CSV
   (`afl_bot/config.py`). No near-lock leg above 0.78 enters the pool unless it
   has a real `--odds` price (the existing bettable-by-definition override).
2. **`BOOKABLE_PROP_MENU`'s "disposals" line drops 15+** (now `[20, 25, 30]`) —
   a near-lock on a gun mid, not a market a book posts standalone.
3. **One best (highest-prob) UNPRICED line per (player, stat)** in the live
   ladder pool — `afl_bot/build/report.py:select_ladder_lines`, called from
   `cli.py round_report`'s leg-construction loop. A book doesn't post both a
   15+ and a 25+ line on the same player; a PRICED line is always exempt from
   this cull (every priced line is a confirmed market). `predictions.csv`
   (grading) is unaffected — it still records every `(player, stat, line)` that
   clears the probability gate, cull or no cull.
4. **Bands move to the honest floor.** `MULTI_TARGET_ODDS = (2.10, 3.00, 5.00)`
   (was `1.75`) — `0.78³ ≈ $2.11` falls out naturally now that the pool has no
   near-lock assist.
5. **The guard now checks the number it prints.** `search_match_sgms` applies
   `corr_gain_haircut` AND an optional `multi_calibrator` to **every candidate
   in the pool before selection** (previously: select on the raw joint, haircut
   only the winners afterwards, calibrate even later in `cli.py`). `_select_for_
   target`'s "never land shorter" test and the top-rung VALUE edge filter both
   now read the same final, priced number that ends up in the report — closing
   exactly the gap that produced the $1.59-under-a-$1.75-band bug. `round_report`
   passes its loaded `multi_cal` straight into `search_match_sgms` now;
   `apply_multi_calibration` (still exported, still tested) is no longer called
   from the live path — it's a no-op-when-`None` convenience for any caller that
   wants to calibrate an already-built rung list directly.

**Net effect:** every ladder leg is a real, placeable market; each player
contributes at most one model-only line per stat; the bottom rung reads
~$2.10–2.40 (never the old ~$1.50/$1.59 drift); and a rung's printed Fair odds
is now guaranteed at or above its Band whenever a qualifying combo exists.

### Real Sportsbet odds + final-22 lineup + model-vs-market ladder (FIX-REAL-SPORTSBET-ODDS-AND-LINEUP)

Ben's next-day review of `2026_r16_report.md` found three more honesty gaps,
the first two only visible once you compare the report against the actual
market: (1) every "Fair odds" in the report was the MODEL's own number — the
Hawthorn/GWS bottom rung priced $2.38, but Sportsbet had the same legs at
$1.53 real. (2) the $2.10 floor is a model construct (`0.78³`); the market
prices genuine near-locks shorter than that, and Ben wants to see BOTH
numbers, not just the model's. (3) Jesse Hogan (not actually playing for GWS
that round) was still appearing in multis — the auto-lineup was parsing
Footywire's EXTENDED SQUAD (26-30 names per team) as if it were the
confirmed 22(+sub).

**PART A — real Sportsbet odds, no paid API.** `afl_bot/data/sportsbet_odds.py`
scrapes Sportsbet's own undocumented JSON API (no key, no login) — confirmed
working from an AU IP only (everyone else gets a non-JSON block page, detected
via `Content-Type` and handled as a clean `{}` fallback, never a retry-storm).
`fetch_sportsbet_odds(event_urls_or_ids)` pulls each event's `SportCard`
(market groupings) then `Markets` for five target groupings only (Top
Markets, Pick Your Own Disposals/Goals, Player Marks, Player Tackles — one
request per grouping, ~2 min cache per event) and reshapes the result into
the exact leg-key format the report already uses:
- `Head to Head` selections → `"<team> to win"` (team names normalised via
  `normalize_team_name`, e.g. "GWS GIANTS" → "Greater Western Sydney").
- `Total Game Points - Over/Under`, Over side → `"Total points <line>+"`.
- Milestone markets (`"20+ Disposals"`, `"1+ Goal"`/`"2+ Goals"`, `"4+
  Marks"`, `"3+ Tackles"` — Sportsbet's own bare-player-name markets, not the
  Over/Under-with-handicap shape) → `"<player> <N>+ <stat>"` directly, one
  regex (`r"^(\d+)\+\s*(Disposals?|Goals?|Marks?|Tackles?)$"`) covering all
  four stats. Player names are taken as Sportsbet's own spelling (no
  roster-based fuzzy matching — the module has no access to `player_log`
  here); in practice these already match the bot's own naming almost always,
  so a join miss is rare, not silent (the per-round leg-matched count prints
  to stderr). New `--sportsbet` flag (`round_report`'s `use_sportsbet`) plus
  `--sportsbet-urls PATH` (default `reports/<year>_r<round>_sportsbet_urls.json`
  — Ben pastes 6-9 match URLs in once per round); merged into `odds_book`
  ahead of the Odds API live feed, behind any manual `--odds` hand-fix.
  **Validated live** against the actual round-16 Sportsbet events from this
  AU-IP environment: 9008 legs scraped across 6 matches in under a minute.

**PART B — final-22 lineup, not the extended squad.** Two independent fixes:
1. *(B2, best-effort)* `data/lineups._parse_footywire_selections` rewritten
   section-aware: the on-field position grid (`<tr class="lightcolor"|
   "darkcolor">` rows) is always confirmed, but the sidebar list's
   `<b>`-headed sections are now read in order and only **Interchange** (the
   bench + medical sub) is kept — **Emergencies**/**Ins**/**Outs** (squad
   cuts and week-to-week deltas, not this week's 22) are excluded. The old
   parser blindly grabbed every `pp-` href on the page regardless of section,
   inflating every team to 26-30 "confirmed" names. `fetch_lineup` now WARNs
   (doesn't fail) when a team still resolves above 24 — some teams' sheets
   hadn't posted Emergencies yet at fetch time, so they stay at their
   extended-squad size; the warning makes that visible instead of silently
   trusting it.
2. *(B1, the dependable complement)* `load_outs`/`apply_outs` — a manual
   override that ALWAYS removes a named player from the resolved lineup
   (auto or manual), via a `"_outs": {team: [player, ...]}` key embedded in
   a `--lineup` file or a dedicated `--outs PATH`. Independent of how good
   B2's HTML parsing is.

   *Verified live*: with B2 alone, Jesse Hogan (the motivating case) already
   has zero Footywire grid/Interchange entries for GWS this round, AND zero
   Sportsbet markets posted on him (PART B3's free cross-check — a player
   not named gets no market) — both confirm independently he's correctly
   excluded without needing B1's override this time.

**PART C — model ladder AND a real-market ladder, side by side.** The
existing $2.10/$3/$5 ladder is unchanged but re-headed **"Model ladder (model
fair odds, no book)"** so it reads as a model number, not a promise. A new
`search_market_sgms` (`afl_bot/build/report.py`) builds a SECOND ladder from
the exact same leg pool, selected and priced on REAL book odds (the per-leg
product, `combined_odds` — Sportsbet's own same-game-multi special prices its
own correlation and isn't scraped, so the report says so explicitly) instead
of the model's joint probability; same "land at-or-above the target, never
short" rule and top-rung VALUE-by-edge promotion as the model ladder, just
keyed on `book_odds`. Rendered as **"Sportsbet ladder (real prices)"** right
under the model ladder, columns `Legs | Book odds (combo) | Model joint % |
Model fair | Edge | Pick` — the model numbers stay attached so the two read
side by side; only printed when at least one combo in that match is fully
priced (silently absent otherwise, same as the existing model-only tags).

*Verified live* on the real round-16 data: Hawthorn/GWS's bottom model rung
priced $2.10 (band) vs the market's actual $2.10-banded real combo at $1.45 —
real markets DO price these legs shorter than the model's 0.78-capped floor,
confirming this is exactly the gap Ben wanted surfaced, not a bug to chase
out of either ladder.

**Why:** Ben's third same-day review in this lineage, and the first to
compare the report against the live market rather than just its own
internals — same investigate-first pattern, this time the "bug" (model vs
market disagreement) is the actual product, not something to fix away.
**How to apply:** `--sportsbet` only works from an Australian IP (this
session's environment happens to be one) — if a future session reports
"Sportsbet scrape unavailable" in the report's own note, check the
environment's egress IP before assuming the scraper broke. `SGM_LADDER_LEG_
PROB_MAX` doesn't exist; `_outs`/`--outs` is the dependable lineup fix,
the section-aware HTML parse is best-effort and can still need it for a team
whose sheet hasn't posted Emergencies yet.

### One player per leg + 6-band ladder (DO-MULTIS-LADDER-FIX-AND-DASHBOARD Stage 1)

**Distinct-subject rule.** `build_sgm_candidates` (the shared combo builder used by both the model
and Sportsbet ladders) now rejects any combo where two legs share the same `subject` (player name).
Previously a combo like "Will Day 20+ disposals + Will Day 4+ marks" could appear. The filter sits
alongside `_no_conflicts` (which already blocks two legs for the same `(match_id, market, subject)`
key), but is broader: it catches same-player combos across different markets.

**6-band ladder.** `MULTI_TARGET_ODDS` widened from `(2.10, 3.00, 5.00)` to
`(2.10, 2.75, 3.50, 5.00, 8.00, 15.00)` — six rungs from safe to longshot. Every match still
guarantees a full ladder; the top rungs (3.50, 5.00, 8.00, 15.00) need a deep enough player pool to
reach them (the Sportsbet ladder's real prices naturally reach further than the model-only floor).

### Multis dashboard + bet tracker (DO-MULTIS-LADDER-FIX-AND-DASHBOARD Stage 2)

**Multis JSON sidecar.** `round-report` now also writes `reports/{year}_r{round}_multis.json`
alongside the `.md` — one record per rung per game, both model and Sportsbet ladders. Each record
has a stable `id` (e.g. `2026-r16-Hawthorn-GWS-model-2.10`) so it can be referenced in the ledger
without re-running the report.

**Bets ledger.** Placed bets are stored in `reports/bets_ledger.json`. Each entry carries a UUID,
the multi's id, a deep snapshot of the legs at placement time, stake, taken odds, status
(`pending`/`won`/`lost`/`void`), and AEST timestamps for placement and settlement.

**Auto-settlement.** Settle pending bets via:

```
python -m afl_bot.cli settle-bets [--year 2026] [--round 16]
```

Settlement reuses the `grade-round` actuals path (Fryzigg/DFS player stats + Squiggle H2H).
Void rule: if a player has no stat entry (DNP), that leg is voided and the multi re-settles on the
remaining legs. A multi where every leg is voided returns the full stake.

**Dashboard.** Launch a local dark-mode dashboard at `http://127.0.0.1:8765` with:

```
python -m afl_bot.cli dashboard [--port 8765]
```

Four panels: **Round View** (all multis from the latest JSON, grouped by game, with inline
Place-a-Bet forms), **Tracker/P&L** (open + settled bets list, cumulative profit chart via
Chart.js), season summary stats, and a **Settle** button that calls `settle-bets` for the current
round. No build step — single Jinja2 template, all state in JSON files.

**Frozen per round.** The dashboard is a read-only view of the frozen `multis.json`. Opening
(or refreshing) the dashboard never re-runs the sim, re-scrapes Sportsbet, or re-selects combos —
it only reads the file. The multis shown are always exactly what `round-report` printed, and they
stay locked until Ben deliberately re-runs `round-report` for that round. Ladder selection uses a
stable tie-break (sorted leg names) so a re-run with identical Sportsbet prices produces
byte-identical output.

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

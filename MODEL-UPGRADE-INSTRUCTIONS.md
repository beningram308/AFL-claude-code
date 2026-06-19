# Instructions: model upgrades from the audit

**From Ben — read fully, then execute in priority order. Work autonomously; only stop if genuinely
blocked. Companion findings doc: `AUDIT-AND-NEXT-STEPS.md`.**

## Ground rules (non-negotiable)
- **No look-ahead.** Every fit/feature uses only games strictly before the round being predicted, same as
  the existing walk-forward (`EloRatings.fit` / `walk_forward_prop_predictions`).
- **Walk-forward score must not regress.** After each change, re-run grading; if out-of-sample log loss
  gets worse, the change is wrong — revert it, don't ship it.
- **Tunables live in `config.py`,** never hard-coded. **Real players only** (keep the synthetic guard).
- `pytest -q` green after every phase. Do phases in order; each ends runnable.

---

## PHASE 1 (HIGHEST VALUE) — validate the multis you actually bet
Right now props are graded as singles and H2H is calibrated, but the **3-leg joint probability** — the
number you bet — is never scored, and `corr_gain` is asserted, never tested.

### 1.1 Build a same-game-multi walk-forward backtest
- New `afl_bot/backtest/multis.py`. For each completed round in the eval window, reconstruct the 3-leg
  combos the model *would* have built (reuse `build/report.py search_match_sgms` +
  `build/multi.py joint_prob_from_masks`), compute the **as-of** joint sim probability (no leakage), and
  record predicted joint prob vs whether all 3 legs actually hit.
- Report **log loss + Brier + a reliability curve for multis** (reuse `walkforward.calibration_curve`):
  bucket predicted multi prob, show predicted vs actual hit rate. A trustworthy ladder has
  `mean_pred ≈ actual_rate` in every bucket.
- Add CLI `grade-multis --year ... --round ...` (or fold into `grade-round`) so it runs like the existing
  grader.

### 1.2 Acceptance
Run it over 2–3 completed 2025/2026 rounds. If the $5.00 (20%) rungs actually land far from 20%, the
multi pricing is miscalibrated — that's the signal to do Phase 2. **Do not trust multi prices until this
reliability curve is flat.**

---

## PHASE 2 (HIGHEST VALUE) — fit the correlation/dispersion params instead of guessing
These constants drive `corr_gain` (the whole reason to price SGMs off the sim) and are currently
hand-set: `SCORE_SHOT_CORRELATION=-0.32`, `PACE_SIGMA=0.07`, `SHARE_CONCENTRATION=200`,
`SHOT_DISPERSION=42.5`, `TEAM_STAT_DISPERSION=150` (`config.py`, used in `sim/engine.py`).

### 2.1 Estimate them from history (anti-leakage, pre-eval seasons only)
- **Teammate disposal correlation** → calibrates `SHARE_CONCENTRATION` (Dirichlet): measure the empirical
  correlation between teammates' disposals within games; fit the concentration that reproduces it.
- **Cross-team total correlation** → `PACE_SIGMA`: measure how the two teams' combined volume totals
  co-move; fit the shared-pace sigma that matches.
- **Two teams' score correlation** → `SCORE_SHOT_CORRELATION`: empirical correlation of the two teams'
  scoring shots per game.
- **NB dispersions** `SHOT_DISPERSION`, `TEAM_STAT_DISPERSION`: method-of-moments / MLE on shots and team
  volume totals (variance vs mean).
- Write the fitted values to a JSON artifact + a `fit-correlations` CLI command; `config.py` keeps the
  current values as fallback defaults.

### 2.2 Acceptance
Re-run Phase 1's multi backtest with the fitted params. Joint-prob calibration must improve (or at worst
not regress). If it regresses, the fit is wrong — keep the defaults.

---

## PHASE 2.5 — act on what Phases 1 & 2 found (do this BEFORE Phase 3)
Findings: Phase 1 graded the multis and the joint-probability reliability curve is **not flat — the model
is moderately overconfident** (worst in the mid bands). Phase 2 fitted the correlation/dispersion params,
but re-grading with them **did not improve** calibration (0.617 → 0.697 log loss on **n=106**).

Read these two facts correctly before continuing:
- **The multi backtest (`afl_bot/backtest/multis.py`) grades the RAW sim joint probability with leg
  calibration OFF** (by design — it states this in the header). Live `round-report` applies the per-stat
  isotonic calibrators, so the multis you actually bet are better-calibrated than this backtest shows.
- Because the backtest legs are uncalibrated, **the overconfidence lives in the legs, not the
  correlation** — correlation only shapes the joint *given* the legs. That's why Phase 2's correlation fit
  couldn't help. And `n=106` is far too small to separate two parameter sets (0.617 vs 0.697 is noise).

### Actions, in order
1. **Commit first.** There is a large amount of uncommitted work across several sessions (lineups,
   live_odds, priors, props, report, cli, config, the new `backtest/multis.py` + `backtest/correlations.py`,
   tests, and the instruction docs). Commit it as a checkpoint **before** any further changes, so nothing
   is lost and changes can be rolled back.
2. **Expand the multi backtest sample.** Grade `grade-multis` over **all of 2024–2025** (every completed
   round), not 7 rounds — get `n` into the thousands so the reliability curve and any parameter comparison
   are statistically meaningful instead of noise. This settles "real bias vs small-sample."
3. **Add a calibration-ON mode to `grade-multis`.** Apply the per-stat calibrators
   (`load_or_fit_prop_calibrators`) to each leg's probability **before** building the joint, and report
   **both** raw and calibrated reliability curves side by side. The calibrated curve is the real-world
   picture (it's what live `round-report` does).
4. **Do NOT auto-wire the fitted correlation params.** Keep them opt-in via `load_fitted_correlation_params()`
   as Claude Code already did. Park the `PACE_SIGMA ≈ 0` finding (teams' disposal totals barely co-move) as
   a note — interesting, not actionable yet.
5. **Then proceed to Phase 3.** After Phase 3 ships, re-run `grade-multis` with calibration ON. If the
   reliability curve flattens, leg calibration was the lever (as expected). That is the acceptance test for
   the whole overconfidence problem.

---

## PHASE 3 (MEDIUM) — make prop calibration match the model you actually price
The isotonic calibrators (`backtest/props.py`) are fit on a **proxy** marginal (plain shrunk-EWMA NB),
missing the TOG/CBA/matchup multipliers, pace, Dirichlet share and scoreline correlation the live sim
uses. They're also pooled per-stat across all lines, and trained on a stale line set.

### 3.1 Calibrate against the real sim output
- Fit the calibrators on predictions generated by the **actual sim pipeline** (or at minimum the full
  multiplier stack used live), not the simplified marginal in `walk_forward_prop_predictions`.

### 3.2 Per-(stat, line) calibration + one line source of truth
- Calibrate per `(stat, line)` (or per probability region), not one curve per stat — the tail lines
  (your $5 legs) are currently least supported.
- Make `backtest/props.py DEFAULT_PROP_LINES` read the **same** `PROP_LINES` the live model prices
  (`config.py`). Backtest exactly the lines you bet.

### 3.3 Acceptance
Prop reliability curve flat across lines (not just pooled); `grade-round` log loss not worse.

---

## PHASE 4 (MEDIUM) — prop market odds + blend props toward the market
The convex market blend (`backtest/ensemble.py`) is **H2H only**. Props get calibration but no blend
toward the market — the single best predictor.

### 4.1 Confirm/finish the per-event prop fetch
- `live_odds.py ODDS_API_URL` is the **bulk** h2h/totals endpoint. Player props need the **per-event**
  endpoint (`/events/{id}/odds?markets=player_disposals_over,...`). Verify it's actually wired (REAL-MULTIS
  Fix A); if only the realistic-lines fallback shipped, finish the real fetch. Needs `ODDS_API_KEY`
  (player-props tier); degrade gracefully without it.

### 4.2 Extend the convex blend to props
- When a prop leg has a real book price, blend the calibrated model prob toward the devigged market prob
  (reuse `fit_blend_weights` / `MarketBlend` machinery), and compute edge on the **blended** prob.
- Only flag VALUE on legs/multis where every leg has a real market price.

---

## PHASE 5 (LOW) — housekeeping & robustness
- Delete the dead `PROP_FORM_GAMES = 20` (`config.py:59`) after confirming nothing reads it (everything
  should use `PLAYER_FORM_WINDOW=40`).
- Wet flag: move from **daily** rainfall (`WET_THRESHOLD_MM`) to the **hourly-at-bounce** Open-Meteo
  reading the code comments already describe; the wet multipliers should be fitted, not just defaults.
- Calibrator/blend **staleness guard**: refit (or warn) when the cached artifact is older than N rounds
  instead of only on `--force-refresh`.

---

## PHASE 6 (OPTIONAL, do LAST) — the one high-value new model
Only after Phases 1–4 (multis validated, calibration faithful, market odds flowing).

### 6.1 Gradient-boosted prop model (XGBoost/LightGBM)
- Train a GBM to predict player stat means/probabilities from rich features (recent form, role/CBA/TOG,
  opponent stat-conceded, venue, weather, home/away, rest). Strict walk-forward; tune out-of-sample.
- **Blend it with the sim, do not replace it** — feed its mean/probability in as another signal the
  existing ensemble/calibration consumes, so correlation structure (Dirichlet/copula) is preserved.
- Ship only if it beats the current prop log loss out-of-sample. New optional dependency, gated like
  `optuna` is today.

### 6.2 (Optional, smaller) attack/defence Elo split
- Split the single Elo into offensive and defensive ratings so a high-scoring/leaky team is modelled
  distinctly from a low-scoring/stingy one; feed both into `expected_total`/`expected_margin`.

---

## Explicitly NOT worth building (audit verdict — don't spend time here)
Poisson (NB is better), Glicko/Glicko-2/TrueSkill (marginal over Elo), Markov models (need play-by-play
you don't have free), survival/hazard (only for time markets), formal causal inference and Bayesian
networks (the sim already encodes the useful structure), SVM, neural nets (GBM is better ROI at this data
size), news/sentiment (noisy for AFL props). Rationale in `AUDIT-AND-NEXT-STEPS.md` PART 4.

## Order of execution
1 → 2 (validate multis, then fix what 1 exposes) → 3 → 4 → 5 → 6. Stop wherever Ben says; each phase is
independently shippable and must leave tests green and walk-forward score no worse.

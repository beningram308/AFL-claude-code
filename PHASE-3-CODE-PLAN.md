# Code plan — close out Phase 2.5, then Phase 3 (faithful prop calibration)

**For Claude Code. Read fully, then execute in order. Work autonomously; only stop if
genuinely blocked.** Companion docs: `MODEL-UPGRADE-INSTRUCTIONS.md` (the phase spec),
`AUDIT-AND-NEXT-STEPS.md` (the findings). This plan supersedes nothing — it makes the
"do Phase 2.5 then Phase 3" instruction concrete against the current code.

## Ground rules (non-negotiable — unchanged from the audit)
- **No look-ahead.** Every fit/feature uses only games strictly before the round being
  predicted, same contract as `EloRatings.fit` / `walk_forward_prop_predictions` /
  `walk_forward_multi_predictions`.
- **Walk-forward score must not regress.** After each change, re-grade; if out-of-sample
  log loss gets worse, the change is wrong — revert it.
- **Tunables live in `config.py`,** never hard-coded. **Real players only** (keep the
  synthetic guard).
- `pytest -q` green after every step. Each step ends runnable and is independently
  shippable. Commit at each ✅ checkpoint.

---

## STEP 0 — Close out Phase 2.5 (the diagnostic already in flight)
The calibration-ON machinery is already built: `grade_multis(..., with_calibration=True)`
emits a `calibrated_joint_prob` column (`afl_bot/backtest/multis.py`), rebasing the naive
product onto calibrated per-leg probs while preserving the sim's `corr_gain`. What's left
is to finish the run and lock it down.

0.1 **Finish the expanded backtest.** Run `grade-multis` over **all completed rounds of
    2024–2025** (≈48 rounds), not a 7-round sample, producing the raw-vs-calibrated
    reliability comparison. Expect ~900 graded rungs (2024 alone ≈443) — large enough that
    the curve is signal, not the n=106 noise.
    ```
    python -m afl_bot.cli grade-multis --year 2024,2025
    ```
0.2 **Cleanup + commit.** Delete the temp `_phase25_check.py`. Commit the expanded
    `grade-multis` run + calibration-ON mode as a checkpoint. Then it's safe to `/clear`.
0.3 **DECISION GATE — read the calibrated curve before touching Phase 3.** This is what
    decides how hard Phase 3 has to work:
    - **Calibrated curve roughly flat** (predicted ≈ actual across bands) → the live
      multis are already trustworthy; the raw "overconfidence" was the uncalibrated legs.
      Phase 3 becomes *polish* — still do 3.2 (line source of truth + per-line), it's cheap
      and correct, but 3.1 is lower urgency.
    - **Calibrated curve still bent** (e.g. says 40%, lands ~30%) → overconfidence survives
      calibration. Phase 3.1 (calibrate against the *real sim*, not the proxy marginal) is
      the actual fix. Do the whole of Phase 3.
    Record which case it is in `AUDIT-AND-NEXT-STEPS.md` (one paragraph) so the reason for
    the Phase 3 scope is on record.

> Do NOT auto-wire the fitted correlation params from Phase 2 — keep them opt-in via
> `load_fitted_correlation_params()`. The `PACE_SIGMA ≈ 0` finding stays a parked note.

---

## STEP 1 (PHASE 3) — make prop calibration match the model you actually price
**Problem (confirmed in code).** `afl_bot/backtest/props.py` fits the isotonic calibrators
on a **proxy** marginal — `walk_forward_prop_predictions` is a shrunk-EWMA mean + role-pooled
NB and nothing else. The live prop price (`cli.py` / `build/report.py` via the sim) layers on
TOG multiplier, CBA/role multiplier, opponent-matchup multiplier, the shared pace draw, the
Dirichlet share allocation and scoreline correlation. So today's calibrator corrects a
*simpler* model than the one that prices your legs. Two more concrete defects:
- `fit_prop_calibrators` (props.py:130) groups by **`stat` only** → one curve for 15+ and
  35+ disposals despite wildly different base rates. The tail lines (your $5 legs) are least
  supported.
- `DEFAULT_PROP_LINES` (props.py:37 — disposals `[15,20,25]`, goals `[1,2]`, marks `[4,6]`,
  tackles `[3,5]`) is **stale and narrower** than live `PROP_LINES` (cli.py:122 — disposals
  `[15,20,25,30,35]`, goals `[1,2,3]`, marks `[4,5,6,7,8]`, tackles `[3,4,5,6,7]`). Live legs
  at lines the calibrator never saw are extrapolated.

### 1.1 One line source of truth (do this first — smallest, unblocks the rest)
- Move `PROP_LINES` out of `cli.py` into `config.py` (single definition) and have
  `cli.py`, `build/report.py`, `backtest/multis.py` and `backtest/props.py` all import it.
- Delete `DEFAULT_PROP_LINES` from `props.py`; `walk_forward_prop_predictions` defaults to
  the config `PROP_LINES`. **Backtest exactly the lines you bet.**
- Tests: assert the line set used by the backtest == the line set priced live (a guard so
  they can't drift again).

### 1.2 Per-(stat, line) calibration
- Change `fit_prop_calibrators` to fit one `IsotonicCalibrator` per **`(stat, line)`** key
  (fall back to the pooled per-stat curve when a `(stat, line)` cell has too few samples —
  set the threshold in `config.py`, e.g. `PROP_CALIBRATION_MIN_SAMPLES`).
- Update the cache schema (`load_or_fit_prop_calibrators`, `CALIBRATOR_CACHE` JSON) to a
  `(stat, line)` keying with a small version bump, and update the apply sites in `run-round`
  and `round-report` (`projection_rows` / leg classification) to look up by `(stat, line)`
  with the per-stat fallback.
- This is the lever that most helps the tail lines that dominate multi value.

### 1.3 Calibrate against the real sim output (the heart of Phase 3.1)
- Fit the calibrators on predictions generated by the **actual sim pipeline** (the full
  multiplier stack used live), not the simplified marginal. Cheapest faithful path: reuse
  `walk_forward_multi_predictions`' as-of construction (it already runs the live
  `_select_players` / `_team_player_samples` / sim per round with no leakage) to emit
  **per-leg** as-of predicted prob vs actual hit, and fit the per-(stat, line) calibrators
  on those. Don't build a second sim path — extend the one the multi backtest already trusts.
- Keep it walk-forward and anti-leakage by construction (same truncation contract).

### 1.4 Acceptance (Phase 3.3)
- Prop reliability curve **flat across lines** (not just pooled per-stat).
- `grade-round` log loss **not worse** than the current cached calibrators.
- Re-run `grade-multis --year 2024,2025` with calibration ON: the calibrated multi
  reliability curve should be **flatter** than at STEP 0.3. **That is the acceptance test
  for the whole overconfidence problem** — if it flattens, leg calibration was the lever, as
  predicted. Commit when green.

---

## STEP 2 (PHASE 4, MEDIUM) — prop market odds + blend props toward the market
Only meaningful once 1.x is in. The convex market blend (`backtest/ensemble.py`) is H2H-only;
props get calibration but no blend toward the market — the single best predictor.
- 4.1 Confirm/finish the **per-event** prop fetch in `data/live_odds.py` (`/events/{id}/odds`
  with `player_disposals_over,...`), not just the bulk h2h/totals endpoint. Needs the
  player-props tier `ODDS_API_KEY`; degrade gracefully without it.
- 4.2 When a prop leg has a real book price, blend the **calibrated** model prob toward the
  devigged market prob (reuse `fit_blend_weights` / `MarketBlend`), and take edge on the
  blended prob. Only flag VALUE where **every** leg has a real market price.

## STEP 3 (PHASE 5, LOW) — housekeeping
- Delete dead `PROP_FORM_GAMES = 20` (config.py:59) after confirming nothing reads it
  (everything should use `PLAYER_FORM_WINDOW = 40`).
- Calibrator/blend **staleness guard**: refit (or warn) when the cached artifact is older
  than N rounds, not only on `--force-refresh`.
- Wet flag: finish moving from **daily** rainfall to the **hourly-at-bounce** reading the
  code comments already describe; refit the wet multipliers once a full hourly backfill
  exists (parked in TODO §4.3 — data-gated, not now).

## STEP 4 (PHASE 6, OPTIONAL — do LAST) — the one high-value new model
Only after multis are validated, calibration is faithful, and market odds flow.
- Gradient-boosted prop model (LightGBM) on rich features (recent form, role/CBA/TOG,
  opponent stat-conceded, venue, weather, home/away, rest). Strict walk-forward; tune
  out-of-sample. **Blend it into the sim as another signal the ensemble/calibration consumes
  — do not replace the sim** (the copula/pace/Dirichlet correlation structure is the reason
  SGMs are priceable). Ship only if it beats current prop log loss out-of-sample. New
  optional dependency, gated like `optuna`.
- (Smaller, optional) attack/defence Elo split feeding `expected_total`/`expected_margin`.

---

## Explicitly NOT worth building (audit verdict — don't spend time here)
Poisson, Glicko/TrueSkill, Markov, survival/hazard, formal causal inference, Bayesian
networks, SVM, neural nets, news/sentiment. Rationale in `AUDIT-AND-NEXT-STEPS.md` PART 4.

## Order of execution
STEP 0 (finish + gate) → STEP 1 / Phase 3 (1.1 → 1.2 → 1.3 → 1.4) → STEP 2 / Phase 4 →
STEP 3 / Phase 5 → STEP 4 / Phase 6. Stop wherever Ben says; each step leaves tests green
and the walk-forward score no worse.

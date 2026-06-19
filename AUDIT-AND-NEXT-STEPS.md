# Audit + next steps (read-only review — no code was changed)

Full read of `afl_bot/` (sim, scoring, pace, ratings, pricing, ensemble, prop backtest, staking,
walk-forward, config). This file is **findings + proposed instructions only** — nothing here has been
applied. Ben decides what to action.

---

## PART 1 — What the model ACTUALLY does today (method inventory)
The system is well-built. Methods currently in use:

- **Monte Carlo** — vectorised numpy match + player sim (`sim/engine.py`). The core.
- **Scoring-shots model** — shots ~ **Negative Binomial**, goals ~ **Binomial(shots, accuracy)**,
  accuracy perturbed per-iteration. Integer scores, real draw probabilities, heteroscedastic variance.
  (Deliberately NOT Normal margin/total, and NOT Poisson — correct for overdispersed counts.)
- **Correlation structure (what makes SGMs worth pricing):**
  - **Gaussian copula** couples the two teams' shot counts (`SCORE_SHOT_CORRELATION = -0.32`).
  - Shared **lognormal pace factor** couples both teams' volume totals (`PACE_SIGMA = 0.07`).
  - **Dirichlet** allocation of team totals across teammates (`SHARE_CONCENTRATION = 200`) → teammates
    negatively correlated (share constraint).
  - **Multinomial** allocation of team goals across players (goal-stack constraint).
  - Player props are **NB**, scaled by that iteration's scoreline → props correlate with the game.
- **Elo ratings** (`ratings/elo.py`) — margin-based, optional 538-style **MOV** update, season carry-over,
  per-game **HGA** (own-venue advantage + interstate travel + days-rest), all empirical-Bayes shrunk.
- **Empirical-Bayes / hierarchical shrinkage** — player rates → role priors; dispersion pooled; venue
  scoring factors; HGA → league mean. (`models/priors.py`, `models/scoring.py`.)
- **EWMA form** — team scoring/accuracy/pace + player rates (now last-40-game window).
- **Isotonic calibration (PAVA)** — per-stat prop calibrators + H2H model calibration (`backtest/`).
- **Convex log-loss-optimal ensemble** — blends model + market (devigged) + Squiggle crowd consensus
  for H2H, weights fit out-of-sample (`backtest/ensemble.py`, SLSQP on the simplex).
- **Market-implied probability + proportional devig** + **market-anchored shrink** of multi legs.
- **CLV (closing-line value)** as the honest edge signal; **walk-forward / time-series CV** everywhere
  (log loss, Brier, margin MAE, reliability curves) — anti-leakage by construction.
- **Bayesian hyperparameter search** — Optuna TPE for Elo tuning (optional).
- **Fractional Kelly** staking (0.25x, per-bet + per-round caps) + bankroll Monte Carlo (incl. a
  *joint* version that resolves correlated bets together).

**Not present:** gradient boosting / neural nets / SVM, Glicko/TrueSkill, Poisson, Markov, survival/
hazard, formal Bayesian networks, formal causal inference, news sentiment. (See PART 4 for verdicts.)

---

## PART 2 — Bugs & gaps found (ranked by impact)

### A. HIGH — the thing you actually bet (multis) is never validated
1. **SGM joint probabilities are unproven.** Props are calibrated as *singles* and H2H is calibrated,
   but the **3-leg joint probability** — the number you bet — is never scored against history. `corr_gain`
   is asserted from the sim, never graded. For a multi-focused bettor this is the #1 gap.
   → Build a walk-forward SGM backtest: for completed rounds, take the 3-leg combos the model would have
   built, compute the sim joint prob as-of, and grade actual hit rate vs predicted (log loss + a
   reliability curve for multis). Until this exists, "fair odds $5.00" on a multi is faith, not evidence.

2. **The correlation/dispersion parameters that drive SGM edge are hand-set, not fitted.**
   `SCORE_SHOT_CORRELATION=-0.32`, `PACE_SIGMA=0.07`, `SHARE_CONCENTRATION=200`, `SHOT_DISPERSION=42.5`,
   `TEAM_STAT_DISPERSION=150` are assumed constants. They determine `corr_gain` — i.e. the whole reason
   to price SGMs off the sim. If they're wrong, every multi price is systematically off.
   → Fit them from history: estimate teammate disposal correlation, cross-team total correlation, and the
   NB dispersions empirically, then check the SGM backtest (item 1) improves. Make them fitted, not faith.

### B. MEDIUM — calibration fidelity
3. **Calibrators are fit against a *proxy* model, not the live sim.** `backtest/props.py` fits the
   isotonic calibrators on a plain shrunk-EWMA NB marginal — it does **not** include the TOG multiplier,
   CBA/role multiplier, matchup multiplier, pace, Dirichlet share allocation or scoreline correlation
   that the live pipeline uses. So the calibrator corrects the bias of a *simpler* model than the one
   that actually prices your legs. → Calibrate against the real sim output (or at least the same
   multiplier stack), so the correction matches the generative model.

4. **One calibrator per stat, pooled across all lines.** `fit_prop_calibrators` groups by `stat` only,
   so 15+ and 35+ disposals share one calibration curve despite very different base rates/biases. The
   tails (the $5 multi legs) are least supported. → Calibrate per (stat, line) or per probability region.

5. **Calibration line set ≠ live line set.** `backtest/props.py DEFAULT_PROP_LINES` is the old
   `{disposals:[15,20,25], goals:[1,2], marks:[4,6], tackles:[3,5]}`, but live `PROP_LINES` was expanded
   (30/35 disposals, 5/6/7/8 marks, etc.). Live legs at lines the calibrator never saw are extrapolated.
   → Single source of truth for lines; backtest the same lines you price.

6. **Props are calibrated but never market-blended.** The ensemble blend (model+market+Squiggle) is
   H2H-only. Props get isotonic calibration but no blend toward the market — the single best predictor —
   because there are no prop odds yet. → Once prop odds exist (Odds API key), extend the convex blend to
   prop legs, not just calibration.

### C. LOW — housekeeping / robustness
7. **Dead knob `PROP_FORM_GAMES = 20`** (config:59) — legacy, superseded by `PLAYER_FORM_WINDOW=40`.
   Confirm nothing reads it, then delete so it can't silently re-enter a code path.
8. **Confirm the live prop-odds path is real.** `live_odds.py ODDS_API_URL` is the **bulk** h2h/totals
   endpoint; player props require the **per-event** endpoint. Verify the per-event prop fetch was
   actually wired (REAL-MULTIS Fix A), not just the realistic-lines fallback.
9. **Wet flag uses *daily* rainfall** (`WET_THRESHOLD_MM=5` daily); the code itself notes daily totals
   are a noisy proxy and hourly-at-bounce was the intent. Low impact, but the wet multipliers are
   research defaults, not reliably fitted.
10. **Calibrator/blend staleness.** Artifacts are cached to JSON and only refit on force — no recency
    guard. A mid-season regime shift runs on stale calibration. Add a refit cadence/age check.

---

## PART 3 — Proposed next instructions (in priority order — NOT yet done)
1. **Build the SGM/multi walk-forward backtest + reliability curve** (addresses A1). This is the highest
   value: it tells you whether your multi prices are real before you bet them.
2. **Fit the correlation + dispersion parameters from history** (A2), then re-run the SGM backtest to
   confirm the joint probs calibrate better.
3. **Calibrate against the real sim output, per (stat, line)** (B3, B4, B5) with one line source of truth.
4. **Wire prop market odds (Odds API per-event) and extend the convex blend to props** (B6, C8) — turns
   model suggestions into measured edges and blends toward the sharpest signal.
5. **Housekeeping** (C7, C9, C10).
6. **Only then** consider a new model class (see PART 4) — a gradient-boosted prop model is the highest-
   value addition, but it should *blend with* the sim, not replace it, and only after 1–4 above.

---

## PART 4 — Verdict on each probability method (incorporate / skip)
| Method | Status | Verdict |
|---|---|---|
| Monte Carlo | In (core) | Keep — it's the right backbone. |
| Frequentist probability | In (sim rel-freq + calibration) | Keep — it's the foundation. |
| Negative Binomial / Binomial / Multinomial / Dirichlet | In | Keep — correct choices for overdispersed, sum-constrained counts. |
| Bayesian updating | Partial (EB shrinkage, isotonic, TPE) | **Deepen** — extend to late-mail/role updates; cheap, high value. |
| Custom ratings (Elo) | In (Elo + MOV + HGA) | Keep; **consider** splitting into attack/defence ratings. |
| Market-implied probability | In for H2H | **Extend to props** (needs odds key). High value. |
| Ensemble (convex blend) | In for H2H | **Extend to props/totals.** High value. |
| Prediction markets (Squiggle crowd) | In | Keep. |
| Gradient boosting (XGBoost/LightGBM) | **Not in** | **Highest-value new model** — but blend with sim, walk-forward hard, do after PART 3. |
| Human expertise / late sharp info | Partial (lineups, injuries) | **Systematise** (Thursday team sheets, late mail). Medium-high. |
| Logistic / linear regression | Implicit (Elo logistic; margin-cal polyfit) | Subsumed by GBM; no separate need. |
| Poisson | Rejected on purpose | **Don't add** — NB is strictly better here. |
| Glicko / Glicko-2 / TrueSkill | Not in | **Skip** — marginal over Elo for an 18-team league. |
| Markov models | Not in | **Skip** — needs play-by-play data you don't have free. |
| Survival / hazard | Not in | **Skip** unless you bet time-to-event markets. |
| Causal models | Heuristic only (TOG/role) | **Skip** formal causal — hard to identify; heuristic multipliers cover the main levers. |
| Bayesian networks | Not in (but sim IS a hand-built PGM) | **Skip** — you already have the useful structure. |
| SVM | Not in | **Skip** — wrong tool for calibrated tabular probs. |
| Neural networks | Not in | **Skip for now** — GBM is better ROI on this data size; revisit only with play-by-play. |
| News/sentiment, weather, trainer comments | Weather in; news not | Weather: keep/improve. News sentiment: **low value/noisy** for AFL props; skip. |

**One-line takeaway:** the *predictions* are strong and partly validated; the *multi-level joint
probabilities and their correlation parameters are not.* Fixing that (PART 3 items 1–2) matters more than
adding any new model. The single best *new* method is gradient boosting — but only after the multis are
validated and prop market odds are flowing.

---

## PART 5 — Phase 2.5 decision gate (2026-06-19)

Per `PHASE-3-CODE-PLAN.md` STEP 0.3, the expanded `grade-multis --year 2024,2025` run (n=835 graded
rungs, 48 rounds — `with_calibration=True`, the default) decides how hard Phase 3 has to work. Result:
calibration gives a small real improvement (log loss 0.5757 → 0.5680, Brier 0.1948 → 0.1913) but the
high-probability bucket (0.4-0.6 predicted) is **still bent** after calibration — predicted 0.458 vs
actual 0.362, almost the same gap as the raw 0.476 vs 0.366. The low/mid buckets are already close to
flat both ways, so the log-loss gain is mostly redistribution (calibration moves rungs out of the high
bucket into the mid bucket, n=246→235 and n=291→484) rather than fixing the high bucket's miscalibration.

**This is the "calibrated curve still bent" case** in STEP 0.3's gate, not the "roughly flat" case — so
the decision is: do the **whole** of Phase 3 (1.1 line source of truth, 1.2 per-(stat,line) calibration,
1.3 calibrate against the real sim output), not just the cheap 1.1 polish. Leg calibration is a real but
partial lever; the persistent high-bucket gap points at calibration *fidelity* (the proxy-marginal
calibrators don't see the same multiplier stack the live sim prices), which is exactly what Phase 3.1
targets.

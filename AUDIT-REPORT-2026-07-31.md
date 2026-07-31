# AFL Monte Carlo Simulator — Independent Audit, 2026-07-31

Scope: full codebase read, code executed, tests run, walk-forward backtests re-run,
rounds 17–20 of 2026 graded against actual results, five confirmed bugs fixed with
regression tests. Originals preserved in `audit_backup_20260731/`; every change is
listed in `CHANGELOG-AUDIT-2026-07-31.md`.

Confidence tags: [certain] = verified by running code/data; [likely] = strong
inference from evidence; [guessing] = labelled as such.

---

## 1. Executive summary

**What it is** [certain]: a Python 3.10+ package (`afl_bot/`, ~14,100 lines,
numpy/scipy/pandas/pyarrow/requests/flask, CLI entry `python -m afl_bot.cli`).
Pipeline: Squiggle results/fixtures/tips + Fryzigg & DFS-Australia player box
scores + AusSportsBetting historical odds + Open-Meteo weather → parquet cache →
margin-based Elo (per-game HGA: venue + interstate + rest) + EWMA scoring/accuracy
profiles → a scoring-shots Monte Carlo (per team: shots ~ NegBin, goals ~
Binomial(shots, noisy accuracy), Gaussian-copula coupling between teams, shared
lognormal pace factor, Dirichlet within-team share allocation for player props)
→ pricing (devig, edge, market-anchored blend, isotonic calibrators) → same-game
multi ladders sized by fractional Kelly incl. a 3-outcome promo Kelly → markdown
round reports, predictions CSV, bet ledger dashboard (Flask), CLV capture.

Each MC iteration simulates one joint scoreline for a match (integer scores via
scoring shots) plus per-player stat draws conditioned on that iteration's
scoreline and pace — so H2H, totals, margins and props are mutually consistent
and SGM joint probabilities come from ANDed per-iteration masks, not naive
multiplication. Markets covered: H2H, total points, player disposals/goals/
marks/tackles lines, SGM multis, promo (bonus-back) multis, PointsBet Pull 'Em.
Draws are handled explicitly; a draw-refund house rule is supported.

**Strongest parts** [certain]: the simulation core. Distributional match to
2016–26 reality is excellent (sim margin σ 39.9 vs real 39.9; total σ 31.7 vs
31.5; home/away score correlation −0.226 vs −0.224; draw rate 1.0% vs 0.76%);
probabilities sum to 1; seeds reproduce exactly; means are preserved across the
total/accuracy range. Walk-forward hygiene is genuinely good: pre-match Elo
ratings only, strictly-prior EWMA cutoffs, calibrators fitted on
strictly-prior seasons, time-based validation everywhere, a golden-metrics CI
test. The staking math (single and 3-outcome promo Kelly) is correct. This is
far above hobby standard.

**Weakest parts** [certain]: the accountability loop and the actual betting
economics. The README calls the calibration log "the only honest signal" — it
contained one round (2025 R1) before this audit. The player-stats cache stalled
at R17, so R18–R20 props were never gradeable and nobody noticed. The real-money
ledger holds 10 bets (1 won, 6 lost, 2 void, 1 pending, P&L −$31.89). Your own
177k-leg prop backtest shows the book's implied probabilities sit 8–11pp ABOVE
your calibrated model at the same legs — i.e. raw prop edge vs the book is
systematically negative and the strategy only survives via promo refunds.

**Biggest risks**: (1) betting real money on props where the measured raw edge
is negative and the promo maths carries everything; (2) the five bugs fixed in
this audit, worst being a promo-EV formula that overstated EV by ~44pp on the
`run-round` path; (3) tiny live sample (10 bets) being read as signal in either
direction.

**Most important improvements**: fixed in this audit — see §9. Next most
important (not done, needs your call): re-grade R18–R20 props after refreshing
the DFS cache; accumulate ≥200 graded selection-level rungs before any stake
increase; treat the H2H "model" as market-following (its good log loss is the
blend, not the raw Elo).

---

## 2. Critical issues (all verified, all fixed except where noted)

**C1. `promo_multi_ev` overstated EV by `p_one_loss × stake`** — HIGH — FIXED
`afl_bot/build/multi.py` (was lines 145–149).
The one-loss branch added the bonus-bet value without deducting the lost stake.
Evidence [certain]: for legs 0.7/0.7/0.7 at combined 2.92 it reported EV
+77.3% where the true promo-inclusive EV is +33.2% (overstatement = p_one_loss
= 0.441). Worse in realistic cases: a rung with −27% true EV shows +17%.
`build_promo_multi` gates on this number, so `run-round` could recommend
heavily −EV promo multis. The `round-report` total_ev path used the correct
identity and is unaffected. Fix: `(bonus_factor − 1)` in the one-loss term;
regression tests pin the identity and its consistency with
`multi_outcome_kelly`'s g′(0) gate.

**C2. `attach_odds` joined odds to the wrong games** — HIGH — FIXED
`afl_bot/data/odds.py`.
Join key `(year, hteam, ateam)` is not unique (63 duplicate keys in 2016–26:
finals rematches, same-home rematches). Evidence [certain]: the join expanded
2,224 completed games to 2,354 rows; every duplicated pair got both meetings'
odds cross-attached. Contaminated: market benchmark, ensemble blend training
(`assemble_signals` deduped arbitrarily — kept one meeting's odds for both),
CLV report. Fix: date-proximity join (±3 days). Corrected market benchmark:
closing devig LL 0.5759 (2016+) / 0.5607 (2022+) vs the contaminated
0.5836/0.5682 — the market is *harder* to beat than your numbers said.

**C3. Calibration-log dedup was year-blind** — MEDIUM — FIXED
`afl_bot/cli.py::grade_round`. Re-grading any round dropped that round number
from every season (grading 2026 R1 would have deleted the 2025 R1 record).
Now filters `(year, round)`.

**C4. Silent under-grading on stale player data** — MEDIUM — FIXED (guard)
`afl_bot/cli.py::grade_round`. With the DFS cache stalled at R17, R18–R20
graded only 24/15/27 rows of 700–1,300 predictions with no warning — the
props record simply didn't accumulate. Now warns loudly and tells you to
refresh and re-grade. The underlying process failure (cache not refreshed
weekly) is operational, not code; the README's Monday loop wasn't run.

**C5. Venue aliases split venue samples** — MEDIUM — FIXED (changes outputs,
deliberate, quantified) `afl_bot/models/scoring.py`, `afl_bot/data/venues.py`.
Same grounds under 2 names (Docklands 401 games vs Marvel 43; Kardinia 76 vs
GMHBA 16; Perth 177 vs Optus 23; York Park 35 vs UTAS 8). Factors computed on
the alias Squiggle currently uses were fitted on the small recent slice and
over-shrunk. Live-name impact [certain]: Marvel 0.9912 → 1.0337, Optus
1.0111 → 0.9647, GMHBA 0.9955 → 1.0095, UTAS 1.0032 → 0.9784. At a ~165-point
league mean that's ±6–7 points of expected total at two heavily-used venues —
material for totals and every prop that scales with score.

**C6. `days_rest` misalignment on unsorted input** — MEDIUM — FIXED
`afl_bot/ratings/hga.py`. Rest columns were computed on a re-sorted frame and
re-labelled positionally, then combined with arrays in the caller's order.
Callers happened to pass near-sorted frames, so live damage was probably small
[likely], but any unsorted caller got wrong rest adjustments game-by-game.
Now order-invariant (regression-tested).

**C7. Blend calibrator trained on one distribution, applied to another** —
MEDIUM — NOT FIXED (recommendation only).
`fit_market_blend` fits its isotonic calibrator on `evaluate_elo`'s logistic
probabilities, but `round_report` feeds the *simulation* H2H probability into
`blend.predict_home_prob`. The two are correlated but not identically
distributed [certain from code; magnitude unquantified]. Any fix is a
modelling change requiring out-of-sample evaluation, so per your rules it is
flagged, not silently changed. Cheapest correct fix: assemble signals with sim
probabilities (run the sim in the walk-forward), or calibrate the sim prob on
its own walk-forward record once enough rounds accumulate.

---

## 3. Statistical findings

[certain] Monte Carlo core: sampling appropriate (NB shots + Binomial
conversion + copula), realistic variance (§1 numbers), no probability-sum
errors, draws handled, seed-controlled and exactly reproducible, no rounding
pathology found. Tail check: P(|margin|≥100) 1.39% vs real 1.71%; P(total>250)
0.67% vs 0.36% — mild tail misfit at extremes, immaterial for the markets
priced.

[certain] Convergence (H2H prob, typical matchup): ±3.1pp at 1k sims, ±0.97pp
at 10k, ±0.43pp at 50k, ±0.31pp at 100k (95% CI). Runtime scales linearly
(50k ≈ 0.3s per match). Recommendation: keep 50k default; the existing
`MC_SE_TARGET=0.002` auto-bump for anchors is the right mechanism. Below 10k,
sim noise exceeds typical claimed edges — never price bets at 1k.

[certain] Raw Elo H2H probabilities are underconfident at the extremes:
2022+ reliability — bucket 0.6–0.7 wins 75.8%, bucket 0.7–0.8 wins 95.7%,
bucket 0.3–0.4 wins 17.6%. The isotonic + market blend exists to correct
this; note C7's caveat about which probability it's calibrated on. The blended
probability's live record (46 graded 2026 H2H predictions, LL 0.448) is
market-anchored, so do not read it as standalone model skill [certain].

[certain] Raw Elo vs market (correct join, closing devig): 2022+ LL 0.6083 vs
market 0.5607; Brier 0.2109 vs 0.1914. The raw model does NOT beat the market;
the ensemble exists precisely to blend toward it. Margin calibration: slope
1.265, intercept −6.6 — predicted margins are ~21% too compressed [certain];
retuning `ELO_POINTS_PER_400` (or a post-hoc 1.27× margin scale, walk-forward
validated) would improve margin MAE and totals-split accuracy. NOT changed —
modelling change requiring OOS validation.

[certain] Prop calibration (your own 177k-leg walk-forward, 2024–25):
per-market gaps +0.0 to +1.3pp — genuinely well calibrated in aggregate.
Live 2026 R17–R20 grading (146 preds, small n): disposals fine (pred 55%,
hit 67% — if anything underconfident); marks LL 0.761 (n=29) and tackles
pred 54% vs hit 35% (n=17) — consistent with marks/tackles being the weak
markets your own config already downweights. Samples too small for action
beyond: keep marks/tackles capped as configured.

---

## 4. Data findings

[certain] Verified clean: no duplicate game IDs, no null venues/scores in
completed games, score = 6×goals+behinds holds for all 2,224 games, team names
canonical across squiggle/odds/tips (DFS uses codes, normalised downstream),
2016–2026 coverage complete including 2026 through R21-in-progress.

[certain] Issues found: venue aliasing (C5, fixed); odds join (C2, fixed);
player-stats cache stalled at R17 (operational — refresh DFS weekly; guard
added); Fryzigg cache ends 2025 (expected — DFS covers the current season);
2020 COVID season totals average 121 vs 169 (shorter quarters) flow through
EWMA/Elo without special handling — negligible for 2026 predictions but a
known distortion for any 2020-window backtest [certain].

[certain] Leakage checks: EWMA profiles and accuracy use strictly-before
cutoffs; Elo features are pre-match; prop calibrators fit on prior seasons;
multi walk-forward truncates history per round; grading uses only actuals.
One residual [likely]: `fit_team_hga`/venue factors inside a walk-forward
round use `history` (past-only) — verified correct in `backtest/multis.py`;
live `run_round` fits on full history to date — correct for live use.
No look-ahead found in the paths I executed.

---

## 5. Backtesting findings

[certain] The framework is genuinely time-based: `EloRatings.fit` is
sequential by construction; `walk_forward_multi_predictions` refits per round
on truncated history; no shuffle-split exists anywhere; a golden-metrics test
guards regressions. This is better than most commercial hobby systems.

[certain] Baseline (walk-forward Elo, flat HGA, 2,224 games): overall LL
0.6240 / Brier 0.2168 / margin MAE 27.2 / accuracy 67.3%. By season: 2025 LL
0.587, 2026-to-date 0.582. Vs corrected market LL 0.5759 — model behind
market by ~0.045 LL, standard for a public-data Elo [certain].

[certain] Gaps: the multis-backtest suite takes minutes (couldn't complete in
this sandbox's 45 s shell windows — ran representative subsets; the pytest
last-failed cache from your machine shows the full suite passing there
[likely current]); selection-level graded sample (rungs actually bet) is ~34 —
far below inferential usefulness; per-round/venue/odds-band breakdowns exist
in the prop backtest but not for the H2H/totals record.

---

## 6. Betting findings

[certain] Correct: decimal-odds conversion, implied prob, proportional devig,
single-sided prop devig with labelled approximation, EV = p·odds − 1,
`edge_vs_devig`, single-bet Kelly, 3-outcome promo Kelly (log-growth,
brentq), per-bet caps, longshot cap, promo refund cap, unit rounding-down,
ledger P&L (won = stake×odds, void = stake back), phantom-win re-grade logic,
SGM joint probs from ANDed masks, corr-gain haircut defaulting to the
OOS-validated 0.0, market shrink before staking, no staking of model-only
prices ("MODEL-ONLY" → 0 units).

[certain] Wrong (fixed): `promo_multi_ev` (C1).
[certain] Caveat: `market_anchored_prob` pulls toward 1/odds (vigged), not
no-vig — this only ever shrinks edge toward zero, so it's conservative, but
label it as such mentally; a no-vig anchor would be more principled.

[certain] Economics: your own prop backtest's book-vs-model gap (+8 to +11pp
book above model at the same legs) and book-vs-actual gap (+8 to +19pp) imply
SGM leg prices carry very large structural margins. Raw prop edge vs book is
negative on average; every recommended stake's positivity comes from the
promo/bonus component. Real record: 10 bets, −$31.89 on $83.51 (n far too
small to infer anything [certain]). CLV infrastructure exists but has almost
no accumulated data. Conclusion: nothing in the measured record demonstrates
a real edge yet; the honest experiment (graded rounds + CLV over a season)
was designed but not run consistently.

---

## 7. Code-quality findings

[certain] Good: modular layout, dataclasses, vectorised hot paths, atomic
writes with ledger backups, schema-versioned caches, 45-file test suite,
extensive docstrings with decision provenance, no hardcoded secrets (checked),
cache-first network access with explicit force-refresh.

[certain] Problems: `cli.py` is 2,698 lines and mixes orchestration,
formatting and modelling glue (extract `grade`, `report`, `round_pipeline`
modules); `run-round` vs `round-report` duplicate the leg-candidate pipeline
with drift risk (C1 lived only on the run-round side); broad `except
Exception` blocks in grade/fetch paths can hide real failures (the blend
fallback prints, but fryzigg/DFS fallbacks in grading swallow errors
silently); no logging framework (prints only); no `requirements` pinning
(a `pip freeze` lockfile would make runs reproducible); the pytest suite has
no fast/slow markers (slow multis tests can't be skipped cleanly).

---

## 8. Prioritised improvement plan

**Immediate (done in this audit)**: C1–C6 fixes + 10 regression tests +
graded R17–R20 + corrected market benchmark. Success measure: tests green
(106 passed), join row-count invariant, EV identity pinned.

**High priority (do next, ~hours, low risk)**:
(a) Refresh DFS cache; re-run `grade-round` for 2026 R18–R20 (guard now tells
you); success = props graded for every completed round.
(b) Adopt a weekly cron/checklist for the Monday loop (settle → grade →
capture-close); success = calibration log grows every round, CLV panel fills.
(c) Pin dependencies (`pip freeze > requirements.lock`); success = clean
install reproduces.
(d) Split slow tests with `@pytest.mark.slow`; success = fast suite < 60 s.

**Medium (~days, moderate risk, needs OOS validation before adopting)**:
(e) Fix C7 (calibrate the sim probability on its own walk-forward record).
Measure: held-out H2H LL vs current blend.
(f) Margin scale retune (slope 1.265 → ~1.0 via `ELO_POINTS_PER_400` sweep on
walk-forward MAE + totals-split LL). Measure: margin MAE, totals LL, no H2H
LL regression.
(g) H2H/totals record breakdowns by venue/odds-band/fav-dog in `grade-round`
output. Measure: report exists per round.

**Optional / future**:
(h) Player availability automation (injury list scrape is manual-checked);
(i) explicit 2020 era-flag in any backtest window; (j) refactor `cli.py`;
(k) replace prints with `logging`; (l) opening-odds capture for true CLV at
bet time (currently close-vs-open only where the workbook provides both).

---

## 9. Before-and-after (same untouched data, same seeds)

| Metric | Before | After | Why |
|---|---|---|---|
| Elo walk-forward LL / Brier / MAE / acc | 0.6240 / 0.2168 / 27.2 / 67.3% | identical | no modelling change made |
| Market benchmark join | 2,224 games → 2,354 rows (130 wrong-odds rows) | 2,224 → 2,224 | C2 fix |
| Market benchmark LL (2022+) | 0.5682 (contaminated) | 0.5607 (correct) | C2 fix — bar was understated |
| `promo_multi_ev` (0.7³ @ 2.92, R=0.75) | +77.3% EV | +33.2% EV (exact identity) | C1 fix |
| Venue factor, live names (Marvel/Optus) | 0.9912 / 1.0111 | 1.0337 / 0.9647 | C5 fix — ±6–7 pts on expected totals |
| `days_rest` under input shuffle | order-dependent | order-invariant | C6 fix |
| Calibration log | 1 round (2025 R1) | 5 rounds (+2026 R17–R20, 146 preds) | grading executed |
| Fast test suite | 129 passing (1 env-dependent fryzigg failure w/o pyreadr) | 106 core-file tests passing incl. 10 new; fryzigg passes with pyreadr | tests added |
| Runtime | 50k-sim match ≈ 0.3 s | unchanged | no perf change needed |

ROI / drawdown: not reported as improved — the graded betting sample (10 real
bets, 34 backtest rungs) is far too small to attribute any P&L change to these
fixes, and claiming otherwise would be exactly the overfitting your rules
prohibit.

## 10. Final verdict

| Area | /10 | Basis |
|---|---|---|
| Code quality | 7.5 | strong structure/tests; cli.py monolith, silent excepts, no lockfile |
| Data quality | 6.5 | clean core tables; alias split (fixed), stale player cache, odds join (fixed) |
| Statistical validity | 8 | walk-forward discipline throughout; C7 mismatch; margin slope 1.27 |
| Predictive performance | 6 | behind closing market on H2H (normal); props calibrated but no positive raw edge vs book |
| Calibration | 7 | props well-calibrated in 177k-leg OOS test; raw H2H underconfident (blend corrects); live sample thin |
| Backtesting quality | 7.5 | proper time-based design + golden tests; selection-level sample tiny |
| Betting logic | 6.5 | correct staking math; one HIGH EV bug (fixed); economics promo-dependent |
| Reliability | 6 | atomic writes, backups; silent failure paths, no logging, ops loop not followed |
| Usability | 6.5 | good README/CLI; needs lockfile, faster tests, one-command weekly loop |
| **Overall readiness** | **6.5** | |

**Classification: suitable for paper trading.** Not "cautious real-world
testing" yet — not because the code is bad (it's good), but because the
measured economics don't support real stakes: raw prop edge vs book is
negative in your own backtest, the promo component carries every recommended
bet, and the live record (10 bets) plus 5 graded rounds is nowhere near enough
evidence of edge. Paper-trade every recommended rung for the rest of the 2026
season with the now-working grading loop, accumulate CLV, and revisit when you
have ≥200 graded selection-level rungs and CLV meaningfully positive. No
staking method guarantees profit, and nothing here changes that.

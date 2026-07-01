# AFL Multi-Bot — Upgrade Plan (distilled from the 5-LLM merged guide)

Date: 2026-06-26 · Author: Cowork (for Ben) · Status: PLAN ONLY — no code changes yet.

This is the filtered, corrected, prioritised version of `AFL_Multi_Bot_Merged_Guide_v3_Opus.docx`
(DeepSeek + GPT + Grok + Copilot + Opus). The raw guide is ~20k words, heavily
repetitive, partly written without knowing what the bot already does, and
contains several **mathematical errors** (corrected below). Hand Claude Code the
PHASES in this doc, one at a time — **not** the raw guide.

> Each phase has a detailed companion instruction file in this folder
> (`FIX-PHASE2-PROMO-ON-LADDER-AND-KELLY.txt`, `FIX-PHASE3-CLV.txt`, …), written
> just before that phase runs so it reflects the audit findings and what the prior
> phase actually shipped. **This doc is the roadmap + dependencies + corrections;
> the FIX-PHASE files are the execution detail.** Phase status: 1 (weather) ✅,
> 2 (promo ladder + Kelly) ✅, 3 (CLV) queued.

Ground rule: the current engine is strong. **No full rebuild.** Everything here is
additive and reversible, validated on history before it goes live, with pytest
green at each step.

---

## 0. What the bot ALREADY does (do NOT rebuild these)

The guide repeatedly says "build X" where X already exists. Skip:

- **Probability calibration** — isotonic, per-(stat,line) prop calibration + a
  selection-level multi calibrator. (Guide's "build a calibration layer" = done.)
- **Game-level correlated Monte Carlo** — the sim produces `corr_gain` (joint −
  naive product); legs are read off one shared simulated match. (Guide's "replace
  independent multiplication with joint simulation" = done.)
- **Walk-forward validation** — `backtest/walkforward.py`, grade-multis. = done.
- **Weather effects, role classification, capped fractional Kelly staking,
  real Sportsbet odds + edge, the dashboard/tracker** — all already shipped.

Grok's section assumes the project is a "monte-carlo-race-sim skill" to be
generalised. **Wrong context — ignore that framing entirely.**

---

## 1. CORRECTIONS — do NOT let these wrong versions get coded

The Opus section caught real errors in the other four contributors. If any phase
below touches these areas, use the corrected form:

1. **Promo objective is Total EV, not "P(exactly one leg loses)."** Do not steer
   legs toward 0.5–0.6 to maximise the refund branch (the cited number is also
   wrong; the peak of 3p²(1−p) is p≈0.667). Maximising the refund probability
   collapses the branch that actually pays. Rank by **Total EV**; let the promo
   term tilt selection only gently.
2. **Stake-back multis need multi-outcome Kelly, not binary Kelly.** The bet has
   three outcomes (all win / exactly one loss → refund / 2+ loss). Binary Kelly
   is mis-specified and returns a *negative* stake for promo-driven multis —
   i.e. it refuses exactly the bets you place. Maximise expected log-growth over
   the three outcomes (solve numerically), then apply 20–25% fractional Kelly.
3. **Free-bet refund value R is a variable, not a constant 0.72.** An SNR bonus
   bet's cash value depends on the odds it's redeployed at (~33% at short odds to
   ~86% at long). Treat R as config + record the realised R each time.
4. **CLV must be measured against a SHARP reference, never the soft book you bet
   into.** Comparing your Sportsbet fill to Sportsbet's own later price mostly
   measures their margin schedule — a phantom signal. Use Betfair exchange where
   liquid, else a de-vigged multi-book consensus.
5. **Correlation sign — the base guide has it backwards; your bot has it right.**
   For a *backed* multi, POSITIVE correlation makes it MORE valuable (true joint
   prob > the book's multiplied price). The guide says the opposite. Your sim
   already does this correctly via `corr_gain` — so trust the engine, and do not
   "fix" correlation toward the guide's wording.

---

## 2. SCOPE decisions (Ben)

- **Promo stake-back multis = IN SCOPE.** Ben places promo-eligible multis ("money
  back if one leg fails"), so Phase 2 (promo-aware EV + staking) matters and the
  corrections in §1.1–1.3 are load-bearing, not academic.
- **Bonus conversion (matched-betting/Betfair hedging) = LIGHT / LATER.** Could be
  useful; not a focus now. Captured as an optional appendix, not a phase.
- **ML ensemble (LightGBM/Bayesian/market model) = DEFERRED.** Large build; the
  single calibrated MC model is performing. Revisit only after CLV data exists.

---

## PHASE 1 — Weather "greasy ball" upgrade  *(do first — fixes the marks issue)*

**Why:** tonight's cold/slippery game produced too many marks legs. Root cause:
`models/weather_effects.py` only suppresses marks when a game is flagged **wet by
rain** (`is_wet`), and leans on the manual `--rain-mm` flag. A cold, dewy,
slippery night with no recorded rain gets **zero** marks suppression. "Slippery"
≠ "wet" in the current logic.

**Build:**
- Replace the binary rain trigger with a continuous **greasiness factor** derived
  from conditions at bounce: temperature, dew point / humidity (dew = slippery
  ball), wind, and rain. Auto-detect from the weather the bot already fetches —
  don't rely on `--rain-mm` alone (keep the flag as a manual override).
- Drive the existing per-stat multipliers from that factor: **marks down, goals
  down, disposal efficiency down, tackles up** (your current 0.85 marks / 1.08
  tackles are the heavy-wet end of a scale, not an on/off switch).
- Make sure it flows all the way into the **marks prop probabilities and leg
  eligibility**, not just the mean — a greasy night should visibly thin the
  marks legs in the ladder.
- Keep all coefficients in config; validate the factor against history (wet/cold
  games vs dry) and keep the safer default if a change regresses grade-multis.

**Acceptance:** re-run a cold/wet round → markedly fewer / shorter marks legs;
dry games unchanged; roofed games neutral.

---

## PHASE 2 — Promo-aware Total EV + correct staking  *(you bet stake-back multis)*

**Why:** every multi Ben places carries the stake-back promo, so the number that
should drive selection and sizing is **Total EV**, not raw model edge — and the
naive formulas in the guide are wrong here (see §1).

**Build:**
- Add **Promo EV**: refund fires on *exactly one* losing leg. The sim already
  produces P(all win) / P(exactly one loss) / P(2+ loss) for each candidate multi
  (count them straight from the simulated outcomes — no analytic formula needed).
  `Total EV = Base EV + P(exactly one loss) × R`.
- **Rank multis by Total EV** (not by refund probability). Promo term tilts, never
  dominates (§1.1).
- **Recommended stake = multi-outcome Kelly** over {all win, one loss→refund, 2+
  loss}, solved numerically, then 20–25% fractional with the existing caps (§1.2).
  Replace binary Kelly **only on promo-eligible multis**; leave singles as-is.
- **R as config** (default conservative, e.g. 0.65–0.72), with a field to log the
  realised refund value per settled bet (§1.3). Feed realised R back over time.
- Surface in the report + dashboard: a **Total-EV** and **suggested-stake** column,
  clearly separate from raw edge.
- Read the promo's **minimum-legs** rule from config (AU stake-backs are usually
  3+). The ladder is already 3-leg, so this mostly just gates eligibility.
- Light "structure" note (not a rebuild): value legs are scarce and clustered; a
  rung that's filler shouldn't be dressed up as value. Keep the honest model-only
  / VALUE tags you already have.

**Acceptance:** promo multis show a Total EV that exceeds their standalone EV; the
suggested stake is positive for genuinely +Total-EV promo multis (binary Kelly
would have refused them); no stake suggested without real odds.

---

## PHASE 3 — Closing Line Value (CLV), done the right way

**Why:** CLV is the lowest-variance way to know if an edge is real — ~100 bets to
confirm via CLV vs ~tens of thousands via raw ROI. But naive prop CLV is a trap
(§1.4 / §1.D).

**Build:**
- Store, per bet: opening odds, **taken** odds, and a **sharp closing reference**.
  Reference priority: Betfair exchange price where the market is liquid (H2H /
  line / popular multis), else a **de-vigged multi-book consensus** for props that
  never trade on the exchange. **Never** Sportsbet's own later price.
- **⚠ PREREQUISITE (this should have been flagged from the start): prop CLV needs a
  SECOND price source.** You currently scrape only Sportsbet, and you can't measure
  a Sportsbet bet against Sportsbet's own close. So Phase 3 ships the CLV
  infrastructure + H2H/line CLV regardless, but **prop CLV stays "n/a" until a
  second AU book (TAB/Ladbrokes, same internal-JSON trick as `sportsbet_odds.py`) is
  scraped.** Build the second-book scraper as the unlock for prop CLV — it is a hard
  dependency, not an optional extra.
- Compute per-bet CLV, rolling CLV (by prop type / player / venue), and a one-sided
  **CLV t-test** vs zero. Display the **minimum detectable edge** for the current
  sample size so a short record honestly says "too soon to tell."
- For props specifically: treat an unattributed line move as **noise**. Only act on
  a move that (a) ties to a concrete catalyst (late out, role/tag change, weather
  shift) and (b) your model independently agrees with.

**Acceptance:** every settled bet carries a CLV vs a non-Sportsbet reference;
dashboard shows rolling CLV + min-detectable-edge; prop CLV is not computed against
the originating book.

---

## PHASE 4 — Statistical honesty guards (anti-phantom-edge)

**Why:** scanning many props guarantees false "value" by chance (~5% of markets
flag at the 5% level with no real edge). Fits your existing honesty ethos.

**Build:**
- **Multiple-testing control**: Benjamini-Hochberg FDR on the value-flag p-values
  per prop type (target ~10% FDR), so the VALUE tag adapts to how many props were
  scanned.
- **Efficiency-scaled edge threshold**: require a larger raw edge on thin,
  low-liquidity props than on liquid lines (`threshold = base + k·(1 −
  liquidity_score)`).
- Report the **minimum detectable edge** alongside any claimed edge.

**Acceptance:** the count of VALUE flags drops to the ones that survive FDR; thin
props need a bigger edge to qualify; nothing is tagged VALUE on a sample too small
to support it.

---

## PHASE 5 — Catalyst edges (the most sustainable prop edge)

**Why:** every contributor agrees durable prop edge lives in late, mechanical
situations the books are slow to fully reprice.

**Build:**
- **Opponent concession rates** per stat (how many disposals/marks/tackles/goals a
  team concedes to each role) as an adjustment into projections.
- **Late team-list / out listener**: when a named player is scratched, redistribute
  their role/opportunity to teammates and recompute affected projections (you
  already have role classification + the lineup/outs plumbing to build on).
- Tag and **log catalyst-driven bets separately** so their edge can be validated on
  its own sample rather than diluted into the general prop population.

**Acceptance:** projections shift sensibly on a late out; opponent-adjustment moves
the right props; catalyst bets are logged as a distinct cohort.

---

## OPTIONAL / LATER (not now)

- **Bonus conversion** (SNR free-bet extraction via Betfair hedging). If pursued:
  deploy the refunded bonus bet near **odds 4.0–6.0** on a *liquid* market (NOT
  line markets at 1.90 — the guide's table is wrong for SNR extraction), measure
  conversion as a fraction of the achievable ceiling, and keep it entirely
  separate from the model. Account-longevity risk applies.
- **ML ensemble** (LightGBM/CatBoost player & role models, Bayesian role updates,
  market-implied component). Defer until Phase 3 has produced CLV data to weight
  against. Big build; current single model is fine.
- **Portfolio optimiser** across a slate (exposure caps per player/game/team).
  Nice-to-have once promo Total-EV staking (Phase 2) is in.

## CUT (don't do)

- Generalising a "race-sim skill" (wrong context).
- Rebuilding calibration / joint simulation / walk-forward (already done).
- Heavy infra (MLflow / Prefect / Streamlit / Postgres). Your file-based setup is
  fine for a personal bot.

---

## Suggested order

Phase 1 (weather) → Phase 2 (promo EV + staking) → Phase 3 (CLV) → Phase 4
(honesty guards) → Phase 5 (catalysts). Each is independently shippable; stop
after any phase. I'll turn each phase into a focused Claude Code instruction when
you're ready — starting with Phase 1.

# AFL Multi-Bot — Full Code Audit (2026-06-26)

Read-only audit of the actual codebase, assessed against the 5-phase upgrade plan.
No files changed. ~9,100 LoC across `afl_bot/`, 39 test files / 353 test functions,
git at `9c5e958`.

**Method & caveat.** Audited the committed code plus the working tree. Claude Code
was **mid-build on Phase 1 (weather)** during the audit — `sim/engine.py`,
`weather.py`, `weather_effects.py`, `stoppages.py` had uncommitted changes, and the
`greasiness` factor is already partly wired into `simulate_match`. So the weather
findings below are "in flight," not stale.

---

## Verdict

**The engine is strong — genuinely above hobby grade, and well ahead of what the
5-LLM guide assumed.** The core simulation already implements the single biggest
thing that guide recommends ("latent game-variable Monte Carlo"), the correlation
sign is correct (the guide got it backwards; your bot does not), calibration and
walk-forward validation are real, and the output is honest. **No rebuild is
warranted.**

The gaps are specific and concentrated, and they cluster exactly where the plan
already points: **promo math isn't wired into the ladder you actually bet from**,
**weather only triggered on rain** (being fixed now), and **there's no live
CLV / multiple-testing discipline yet**. None require touching the engine.

---

## What's genuinely strong (do not touch)

- **Simulation core (`sim/engine.py`).** Scoring-shots model (NB shots → Binomial
  goals, not a Normal margin), a Gaussian copula coupling the two teams' scores, a
  shared lognormal **pace latent factor** drawn once per match, **Dirichlet
  share allocation** (teammates' counts sum-constrained → correctly negatively
  correlated), and Multinomial goal allocation. This *is* the "simulate the game
  environment first, then players conditionally" architecture every contributor
  said to build. It already exists.
- **Correlation handled correctly.** SGM legs price off ANDed per-iteration sim
  masks (`joint_prob_from_masks`), cross-game legs multiply as independent. This is
  the exact split Opus's C6 prescribes, with the **correct sign** — the guide's
  "positive correlation makes a multi less valuable" error is not present.
- **Anti-leakage + era-matching.** Projections use only games strictly before the
  as-of round; opponent-concession and league baselines are restricted to recent
  seasons, not all-history.
- **Calibration is real.** Per-(stat,line) isotonic prop calibration + a
  selection-level multi calibrator, fit walk-forward. The guide's "build a
  calibration layer" is done.
- **Honest pricing.** `market_anchored_prob` haircut, Monte-Carlo standard error,
  model-only tags, no VALUE/stake without real odds, frozen multis.
- **Staking + bankroll risk.** Capped fractional Kelly with per-bet and per-round
  caps, plus a **joint** bankroll Monte Carlo (`simulate_bankroll_joint`) that
  resolves correlated bets together — better than most retail tools.
- **Validation.** Walk-forward with log-loss, Brier, calibration curves,
  market metrics, season-by-season. 353 tests.

---

## Issues found (by severity)

### HIGH

1. **Promo EV is not on the ladder/dashboard you bet from.** `promo_multi_ev` and
   `build_promo_multi` exist and correctly rank by **Total EV** (not the wrong
   "P(one loss)" objective) — but they run as a *separate report section* built from
   `odds_legs`, on the old path. The **6-band ladder** (`search_match_sgms`) that
   feeds `multis.json` and the dashboard carries **no promo term**. Since you bet
   promo multis off that ladder, the number you see understates the bet. *(Plan
   Phase 2 — top priority.)*

2. **Promo "exactly one loss" uses the independence formula even for same-game
   legs.** `promo_multi_ev` computes `p_one_loss` analytically from `p1,p2,p3`. For
   SGM rungs (correlated legs) that's wrong — it should be **counted from the sim
   masks** (the engine already produces them). Over/understates the refund branch on
   correlated multis. *(Phase 2.)*

3. **Staking uses binary Kelly for a 3-outcome bet.** `kelly_fraction` is
   `(p·odds−1)/(odds−1)` — correct for win/lose, mis-specified for a stake-back
   multi (win / one-loss-refund / dead). As Opus shows, binary Kelly can return a
   *negative* stake for a genuinely +Total-EV promo multi. *(Phase 2.)*

4. **Weather only triggered on rain.** `weather_effects.py` applies the marks/goals
   suppression only when `is_wet` (rain) flips, leaning on the manual `--rain-mm`
   flag — so a cold/dewy/slippery dry night gets **zero** marks suppression. This is
   exactly tonight's too-many-marks complaint. **Being fixed now** via the
   continuous `greasiness` factor — confirm it reaches the *marks prop probabilities
   and leg eligibility*, not just team totals. *(Phase 1 — in progress.)*

### MEDIUM

5. **Two parallel multi systems.** `build_promo_multi`/`build_anchor_multis` (old)
   and `search_match_sgms` (new ladder) both run in `round-report`. They can diverge
   (the promo logic and the distinct-player/6-band rules live in different places).
   Worth consolidating so the ladder is the single source and promo EV is computed
   on it.

6. **CLV is H2H-only and backtest-only.** `walkforward.clv_report` measures CLV from
   historical opening/closing **H2H** odds — good, and it respects a real close. But
   there's **no live per-bet CLV, no prop CLV, and no sharp-reference construction**.
   Per Opus C4, prop CLV must use a sharp reference (Betfair/de-vigged consensus),
   never Sportsbet's own later price. *(Phase 3 — partially seeded, not live.)*

7. **No multiple-testing / phantom-edge guard.** `classify_leg` flags VALUE on a
   fixed edge+prob gate. Scanning many props guarantees false positives; there's no
   FDR control or efficiency-scaled threshold, and no minimum-detectable-edge
   reporting. *(Phase 4.)*

8. **Opponent matchup exists; catalyst redistribution does not.** `props.py` already
   has `opponent_matchup_multiplier` (era-matched) — so Phase 5 is *half done*. What's
   missing is the **late-out role/opportunity redistribution** and separate logging
   of catalyst-driven bets. *(Phase 5.)*

9. **Settlement name-matching risk.** `settle.py` matches multi legs to Fryzigg/DFS
   actuals by player name. Sportsbet-normalised names vs stats-source names can
   mismatch → a leg silently voids. Worth a reconciliation/logging check.

### LOW

10. **Scrape fragility (operational, not a bug).** Footywire (extended-squad issue,
    already mitigated with outs + final-22 trim) and the Sportsbet internal JSON
    (AU-only, can change without notice). Fine for personal use; keep the manual
    overrides and fail-soft behaviour.
11. **Minor doc drift.** A few comments still reference the old "1.75 ladder floor"
    after the move to the $2.10 / 6-band ladder.

---

## Gap analysis vs the 5-phase plan

| Plan phase | State today | Gap |
|---|---|---|
| **1 — Weather/greasy ball** | Rain-only multipliers; greasiness factor **being wired now** | Ensure greasiness drives marks *prop probs + leg eligibility*, auto-detected (temp/dew/wind), not just totals |
| **2 — Promo Total EV + staking** | Promo EV exists (old path, Total-EV objective ✓) but **not on the ladder**; independence assumption; binary Kelly | Wire promo onto ladder rungs; count one-loss from masks; multi-outcome Kelly |
| **3 — CLV done right** | H2H CLV **backtest** only | Live per-bet CLV vs a **sharp** reference; prop CLV; min-detectable-edge display |
| **4 — Honesty guards** | None | FDR / efficiency-scaled threshold on VALUE flags |
| **5 — Catalyst edges** | Opponent matchup ✓ | Late-out role redistribution + catalyst bet logging |

Net: roughly **40% of the plan is already latent in the code** (engine, correlation,
calibration, opponent matchup, the promo *formula*, H2H CLV backtest). The work is
mostly **wiring and discipline**, not new modelling.

---

## Recommended order (unchanged from the plan, refined by the audit)

1. **Finish Phase 1 (weather)** — already in flight; verify it thins marks legs.
2. **Phase 2, narrowed:** put **promo Total EV on the ladder rungs** (count one-loss
   from masks) and add **multi-outcome Kelly** as the suggested stake. This is the
   highest-value change because every multi you place is promo-eligible and the
   current ladder number is the wrong one.
3. **Consolidate the two multi paths** while doing Phase 2 (kill the divergence).
4. **Phase 4 (honesty guards)** — cheap, high-integrity, stops phantom VALUE flags.
5. **Phase 3 (live CLV with sharp reference)** — most plumbing; do once you want a
   real "is my edge real" gauge.
6. **Phase 5 (late-out redistribution)** — build on the existing opponent matchup.

**Bottom line:** the foundation is sound and the correlation/calibration/honesty
work is better than the guide assumed. Don't rebuild anything. Spend the effort on
(a) finishing the weather fix and (b) making the ladder you actually bet from
promo-aware with the correct staking — that's where real money is currently being
left on (or taken off) the table.

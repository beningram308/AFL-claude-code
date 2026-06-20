# Code plan — Phase 4 (market-blend props) on MANUAL odds, no paid API

**For Claude Code. Read fully, then execute in order. Work autonomously; only stop if
genuinely blocked.** This **supersedes** the API-dependent Phase 4 sketched in
`PHASE-3-CODE-PLAN.md` STEP 2. Companion docs: `MODEL-UPGRADE-INSTRUCTIONS.md`,
`AUDIT-AND-NEXT-STEPS.md`.

## Why this version
There is **no free, stable feed for AFL player-prop odds** — The Odds API only carries AFL
props on its ~$99/month Business tier, and the other providers are paid too. Ben does not
have a key and isn't buying one right now. But Phase 4's value (anchoring prop legs to the
market, the single sharpest signal) does **not** need an automated feed: the bot already
takes hand-entered prices via the `--odds` JSON, and a weekly bettor only prices a handful
of legs. So we build the **blend on manual odds now** and **park the paid auto-fetch**.

## Context you must hold (read before coding)
Phase 1 found the multi ladder is overconfident; **three structurally different fixes have
since failed** to move it — sim-source calibration (Phase 3), price-shrink/LCB haircuts
(Phase 3.5), and selection-level recalibration (Phase 3.6, which made it *worse*). So the
bias is **not** in the calibration/selection layer; it's upstream (`corr_gain` / the joint
structure) or non-stationary. Market-anchoring is the **pragmatic mitigation**: pulling
each priced leg toward the book's devigged price deflates overconfident legs regardless of
root cause — **but only on legs that actually have a market price.** Set expectations
accordingly: this helps the legs you price by hand, it does not globally fix the joint
overconfidence.

## Ground rules (non-negotiable — unchanged)
- **No look-ahead**; **walk-forward score must not regress**; **tunables in `config.py`**,
  never hard-coded; **real players only** (keep the synthetic guard); `pytest -q` green
  after every step; each step independently shippable; commit at each ✅.

---

## STEP 1 — manual prop odds in, cleanly (replaces the paid 4.1 fetch)
The `--odds` JSON already maps leg name → decimal odds and `search_match_sgms` already
consumes an `odds_book` for multi-level edge. This step makes **single prop legs** first-class
in that path and removes the hand-entry friction.

1.1 **Confirm + document the prop-leg odds path.** Verify a prop price placed in `--odds`
    reaches the leg classifier (`pricing/edge.py classify_leg` / `Leg`) and the report. If a
    prop leg currently only gets a book price at the multi level, wire it at the **single-leg**
    level too. The existing unmatched-key warning (TODO 7.4) stays.
1.2 **Emit an odds template so hand-entry is copy-paste, not guesswork.** Have `round-report`
    also write `reports/<year>_r<N>_odds_template.json` containing **every priced leg's exact
    name as a key with `null` value** (h2h, totals, and each prop line), plus the `_rules`
    stub. Ben fills in numbers off the bookie and passes it back as `--odds`. This kills the
    leg-name typo problem at the source.
1.3 **Devig.** For a prop entered as a complete two-way market (over **and** under), devig the
    pair with the existing `edge.devig_proportional`. If only one side is entered, fall back to
    `edge.implied_prob` with a configurable overround assumption
    (`PROP_ASSUMED_OVERROUND` in `config.py`, e.g. 1.06) — and label that leg's price
    "single-sided (approx)" in the report so it's never mistaken for a clean devig.

## STEP 2 — blend each priced prop leg toward the market (the actual Phase 4.2)
Today H2H gets the full fitted ensemble blend (`fit_market_blend` / `MarketBlend`); props get
isotonic **calibration** but **no** market blend. Close that.

2.1 **Per-leg prop blend.** When a prop leg has a (devigged) book price, pull its **calibrated**
    model prob toward the devigged market prob and price/classify the leg on the **blended**
    prob. Reuse `edge.market_anchored_prob(prob, odds, weight)` — it already does exactly this
    pull — with a new config weight `PROP_MARKET_BLEND_WEIGHT`.
2.2 **HONEST CONSTRAINT — the weight cannot be fitted, so make it a documented prior, not a
    fake learned number.** The H2H blend weight (0.42/0.58) is fitted out-of-sample on the
    historical odds archive. **No historical prop-odds archive exists**, so there is nothing to
    fit a prop blend weight against. Set `PROP_MARKET_BLEND_WEIGHT` as a deliberate prior
    leaning toward the market (start ~0.6 market / 0.4 model — props are noisy and the market
    is sharp) and say so in the README: this is a prior, not a backtested optimum, and should
    be revisited if a prop-odds history is ever collected. **Do not** invent a walk-forward
    number for it.
2.3 **Edge on the blended prob; VALUE gated on real prices.** Compute leg edge against the
    devigged market on the blended prob (`edge.edge` / `edge_vs_devig`). Only tag a leg or a
    multi **VALUE** when **every** leg in it has a real market price — never flag value off a
    fair-odds-only leg. Keep the existing multi-level `market_anchored_prob` shrink
    (`MULTI_MARKET_SHRINK`) on top; the per-leg blend and the joint shrink are complementary.
2.4 **Snapshot the entered odds.** Save the filled `--odds` file alongside the predictions CSV
    (`reports/<year>_r<N>_odds.json`) so that, over time, the repo **accumulates its own
    prop-odds history** — the exact archive Step 2.2 says doesn't exist yet. This is the
    cheapest path to one day fitting the weight for real. (Same snapshot discipline as the DFS
    pull, TODO 7.3.)

## STEP 3 — acceptance (be honest about what can and can't be proven)
- **Mechanical correctness, not a backtest.** With no historical prop odds, you **cannot**
  walk-forward-prove the prop blend improves log loss — and you must not pretend to. Acceptance
  is: (a) a prop leg with a book price shows a blended prob between model and devigged market,
  with edge computed on it; (b) VALUE appears only when all legs are priced; (c) with no
  `--odds`, behaviour is byte-for-byte unchanged (degrade gracefully); (d) the emitted odds
  template round-trips (fill it, pass it back, every key matches a leg — zero "unmatched key"
  warnings); (e) `pytest -q` green with new unit tests for devig, the blend, and the
  VALUE gate.
- **Worked example in the README** (like the wet-weather one): one prop leg, model vs entered
  book price vs blended prob vs edge, so the behaviour is legible.
- H2H ensemble blend and existing multi pricing **unchanged** when no prop odds are supplied.

## STEP 4 — parked (do NOT build now)
- **Paid per-event auto-fetch (the old 4.1).** Leave `data/live_odds.py` as-is; if/when Ben
  gets a player-props-tier `ODDS_API_KEY`, the per-event endpoint
  (`/events/{id}/odds?markets=player_disposals_over,...`) drops in to *populate the same
  `--odds` structure* this plan already consumes — so nothing downstream changes. Gate it
  behind the key and degrade silently without it. Not in scope for this phase.

---

## Ben's weekly workflow (what this buys you)
1. `round-report --year 2026 --round N` → get projections **and** a `*_odds_template.json`.
2. Open Sportsbet/TAB, type the prices for the few legs you care about into that file
   (both over+under where you can, for a clean devig).
3. `round-report --year 2026 --round N --odds that_file.json` → legs now blended toward the
   market, edges computed, VALUE flagged only where you actually have prices.
   No API key, no subscription, ~2 minutes of typing.

## Parked free alternative (if you'd rather not deal with odds yet)
The unresolved overconfidence has one cheap, decisive upstream test that needs **no odds**:
compare the sim's `corr_gain` (joint − naive product) to the **empirical** corr_gain (actual
joint hit-rate − product of actual leg hit-rates) on the selected combos. If the sim's lift
is systematically larger, the copula/pace/Dirichlet structure is overstating co-occurrence —
the root cause pinned in one number, and a `corr_gain` haircut becomes the targeted fix. This
is the better use of a session than Phase 4 if market-anchoring isn't the priority.

## Order of execution
STEP 1 → STEP 2 → STEP 3 (acceptance + README example) → commit. STEP 4 stays parked. Each
step leaves tests green; nothing regresses when no odds file is supplied.

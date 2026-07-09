# Stake-cap backtest — which UNIT_MAX grows the bankroll?

_Diagnostic only. Does not change UNIT_MAX, KELLY_FRACTION, or any other live config — see DO-STAKE-CAP-BACKTEST.txt._

## Scope

Rounds used (real Sportsbet book prices + completed results): **2026 R16, 2026 R18**.
Graded rungs: **34** (model + sportsbet ladders; 88 rung(s) excluded — at least one leg still ungradeable).

Rounds NOT used:
- 2026_r17: no real book_combo prices (model-only run)

> **Honesty note:** this is a 2-round sample (R14/R15 predate the multis.json emitter and R17 was a model-only run with no book prices, so this can run short of the ~5 rounds a full season-in-progress would offer). A handful of multis can dominate the realized P&L at this size. Treat Version A as "what happened", not "what's optimal" — Version B's probabilistic sim is the more meaningful signal here, and even that leans entirely on model calibration (see the cross-check below).

**Not swept:** KELLY_FRACTION. `recommend_units` doesn't parameterise the Kelly fraction (it's a default inside `fractional_kelly_fraction`/`multi_outcome_kelly`), and reimplementing that conversion for this backtest would be exactly the kind of parallel sizing logic the brief says to avoid. **Not backtested:** Pull 'Em — no round in `reports/` currently has a real (priced) Pull 'Em record.

## Version A — realized replay (what literally would have happened)

| UNIT_MAX | n bets | staked | returned | net | ROI% | end bankroll | max DD |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.5u | 7 | $56.25 | $0.00 | $-56.25 | -100.0% | $1,443.75 | 3.8% |
| 2u | 7 | $56.25 | $0.00 | $-56.25 | -100.0% | $1,443.75 | 3.8% |
| 3u | 7 | $56.25 | $0.00 | $-56.25 | -100.0% | $1,443.75 | 3.8% |
| 4u | 7 | $56.25 | $0.00 | $-56.25 | -100.0% | $1,443.75 | 3.8% |

> **All rows above are identical — this is real, not a bug.** Even sized with no cap at all, the strongest bet in this sample only ever wanted **1.25u**, below every candidate cap tested (1.5u, 2u, 3u, 4u). None of them bound in this sample, so this run genuinely cannot answer "is 3u too tight" yet — it can only confirm 3u hasn't cost anything so far. A real answer needs a round where the formula's own uncapped output exceeds 3u.

### Round-cap ON vs OFF (at the live UNIT_MAX=3u)

_The round-level 15u cap (`KELLY_PER_ROUND_CAP`) was removed from the live bot 2026-07-10. This row shows what keeping it would have cost/earned on the same bet set, using a read-only reimplementation of the deleted allocator — the live bot does not have this cap anymore regardless of what this shows._

| Round cap | n bets | staked | returned | net | ROI% | end bankroll | max DD |
|--|--:|--:|--:|--:|--:|--:|--:|
| OFF (live) | 7 | $56.25 | $0.00 | $-56.25 | -100.0% | $1,443.75 | 3.8% |
| ON (15u, removed) | 7 | $56.25 | $0.00 | $-56.25 | -100.0% | $1,443.75 | 3.8% |

## Version B — probabilistic bankroll sim (N=10,000 paths per cap)

### Fixed stake (constant $, matches live behaviour)

| UNIT_MAX | median end | mean | p5 | p95 | median maxDD | P(down) | P(DD>20%) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.5u | $1,500.80 | $1,509.71 | $1,443.75 | $1,613.71 | 2.4% | 48.7% | 0.0% |
| 2u | $1,500.80 | $1,509.71 | $1,443.75 | $1,613.71 | 2.4% | 48.7% | 0.0% |
| 3u | $1,500.80 | $1,509.71 | $1,443.75 | $1,613.71 | 2.4% | 48.7% | 0.0% |
| 4u | $1,500.80 | $1,509.71 | $1,443.75 | $1,613.71 | 2.4% | 48.7% | 0.0% |

### Compounding stake (% of running bankroll — truer Kelly-growth read)

| UNIT_MAX | median end | mean | p5 | p95 | median maxDD | P(down) | P(DD>20%) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.5u | $1,499.81 | $1,509.47 | $1,444.59 | $1,615.46 | 2.5% | 51.2% | 0.0% |
| 2u | $1,499.81 | $1,509.47 | $1,444.59 | $1,615.46 | 2.5% | 51.2% | 0.0% |
| 3u | $1,499.81 | $1,509.47 | $1,444.59 | $1,615.46 | 2.5% | 51.2% | 0.0% |
| 4u | $1,499.81 | $1,509.47 | $1,444.59 | $1,615.46 | 2.5% | 51.2% | 0.0% |

## Cross-check: is the modelled edge real, or just what happened to land?

Modelled hit-rate (mean p_all_win over 34 bets with promo branch probs): **17.9%**. Actual realized hit-rate (fraction of all 34 graded bets that won outright): **11.8%**.

`reports/calibration_log.csv` has no 2026 round-level entries yet (`grade-round` hasn't been run this season, only historical 2025 R1 rows exist) — it can't be used as a third reference point here. This cross-check is Version A vs Version B only.

Gap of -6.1% is small given the sample size — Version B's cap ranking is reasonably credible, for what a 2-round sample is worth.

## Verdict

**No cap comparison is possible this run** — every candidate sized identically because the strongest real bet only ever wanted 1.25u uncapped (see the note under Version A). This round's data says nothing about whether 3u is too tight, too loose, or right — it only confirms it hasn't been the binding constraint yet. Re-run this after a round where the formula's uncapped output for at least one rung exceeds the smallest candidate cap.

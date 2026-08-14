# Edge-floor viability test — staking policy, not modelling

_TASK 1 of `fable tweaks/DO-EDGE-FLOOR-VIABILITY-TEST.txt` — measurement only. Does not touch `afl_bot/build/staking.py`, `afl_bot/build/report.py`, `afl_bot/cli.py`, or `afl_bot/config.py`; every number below comes from re-sizing/re-settling the SAME saved `reports/*_multis.json` rungs through `afl_bot.build.staking`'s own Kelly functions, never a reimplementation._

## Reconciliation gate — Policy A, assumption "none", 2026 R18–R20 only

| Scope | n bets | units staked | P&L (units) |
|---|--:|--:|--:|
| Audit target (`AUDIT-ROUNDS-16-20-BET-LOSS-AUTOPSY.md`) | 119 | 214.75u | -87.33u |
| **Audit-scope** (`grade_total_points=False` — reproduces the audit's own blind spot) | 119 | 214.75u | -87.33u |
| **Full-scope** (`grade_total_points=True` — basis for policies B/C below) | 120 | 216.75u | -89.33u |

**GATE PASSED** — audit-scope lands within 1u of the audit's own numbers on all three metrics.

The full-scope/audit-scope delta is exactly one rung: `2026-r20-Western_Bulldogs-Richmond-model-5.00` (Total points 175.5+ / Rhylee West 1+ goals / Dion Prestia 20+ disposals, 2u PROMO KELLY). Final score Western Bulldogs 105–48 Richmond = 153 total, under the 175.5 line — a clean miss (`data_cache/games_2026.parquet`, round 20, complete). The audit's own `legs_graded.csv` shows this leg's `hit`/`actual` fields empty for that row: its independent re-grading script never sourced match-total actuals at all (it read player box scores only, per its own methodology note), so the whole rung came back "ungradeable" even though the game is complete and the other two legs graded — and matched this harness's leg-level results exactly (both hit). `_settle_leg` reads the real 175.5 threshold out of the leg's `name` string via regex (`"Total points 175.5+"`); the `line` key on every total_points leg in the saved multis.json is a placeholder value of 5, not the real line — both live callers (`dashboard/settle.py`, this module) pass `game` into `_settle_leg` so it can look up the actual score; the audit's separate script apparently didn't. 2.0u of 214.75u — 0.9% — this cannot move the answer the rest of this report exists to give.

## Scope

Rounds: **2026 R16, 2026 R18, 2026 R19, 2026 R20**. 2026 R17 excluded (model-only, no real book prices, 0 gradeable rungs). R21/R22 excluded (not fully gradeable — `AUDIT-ROUNDS-16-20-BET-LOSS-AUTOPSY.md` Finding 0: the stats-cache pipeline stalled again, R21 has 1 of 9 games graded, R22 has none). Full-scope grading (`grade_total_points=True`): **244** gradeable rungs across these 4 rounds, 5 excluded for a still-ungradeable leg (unmatched player name, etc. — never guessed).

## Main grid — one row per (policy × promo assumption)

| Policy | Promo assumption | n bets | units staked | units/round | P&L (units) | P&L ($) | ROI% |
|---|---|--:|--:|--:|--:|--:|--:|
| A — LIVE (PROMO_EV_MIN gate on total_ev, no round cap) | 1. No refunds at all | 127 | 220.50u | 55.12u | -93.08u | $-1,396.25 | -42.2% |
| A — LIVE (PROMO_EV_MIN gate on total_ev, no round cap) | 2. One refund/book/round ($50 cap, 0.75 bonus, largest qualifying one-miss bet only) | 127 | 220.50u | 55.12u | -85.77u | $-1,286.56 | -38.9% |
| A — LIVE (PROMO_EV_MIN gate on total_ev, no round cap) | 3. Bot's current assumption (every one-miss bet refunded @0.75) | 127 | 220.50u | 55.12u | -13.77u | $-206.56 | -6.2% |
| B — EDGE FLOOR (raw edge > 0 required; promo may only up-size) | 1. No refunds at all | 24 ⚠️ | 20.25u | 5.06u | -20.25u | $-303.75 | -100.0% |
| B — EDGE FLOOR (raw edge > 0 required; promo may only up-size) | 2. One refund/book/round ($50 cap, 0.75 bonus, largest qualifying one-miss bet only) | 24 ⚠️ | 20.25u | 5.06u | -18.00u | $-270.00 | -88.9% |
| B — EDGE FLOOR (raw edge > 0 required; promo may only up-size) | 3. Bot's current assumption (every one-miss bet refunded @0.75) | 24 ⚠️ | 20.25u | 5.06u | -14.62u | $-219.38 | -72.2% |
| C — EDGE FLOOR + CAPS (B + 15u round cap + 1 multi/player/round) | 1. No refunds at all | 19 ⚠️ | 16.00u | 4.00u | -16.00u | $-240.00 | -100.0% |
| C — EDGE FLOOR + CAPS (B + 15u round cap + 1 multi/player/round) | 2. One refund/book/round ($50 cap, 0.75 bonus, largest qualifying one-miss bet only) | 19 ⚠️ | 16.00u | 4.00u | -14.12u | $-211.88 | -88.3% |
| C — EDGE FLOOR + CAPS (B + 15u round cap + 1 multi/player/round) | 3. Bot's current assumption (every one-miss bet refunded @0.75) | 19 ⚠️ | 16.00u | 4.00u | -11.12u | $-166.88 | -69.5% |

**⚠️ = n < 30.** Every row so flagged is computed on fewer than 30 graded bets — at that sample size the ROI is NOT evidence of edge in either direction (a single multi can swing it by several points). Report the n; do not call a ⚠️ row "profitable" or "working", and do not call any row profitable off this alone without also looking at its worst round.

> **Why B/C's "no refunds" ROI is exactly -100%:** every raw-edge-positive rung in this 4-round sample sits at band 5, 8, 15 of the ladder (`MULTI_TARGET_ODDS` = 2.10/2.75/3.50/5.00/8.00/15.00), and none of them won outright here (0 of 24 policy-B bets — win probabilities are small by construction at these odds). With no refund, a bet that doesn't win outright loses its full stake, so 0 wins mechanically means -100% ROI on this row. Not a bug; a real property of where this bot's positive raw edge currently lives — see Finding 3 of the autopsy on short-band overconfidence vs long-band small-sample variance.

## Policy C, assumption 2 ("one refund/book/round") — by round

"Eligible" = rungs passing policy C's raw-edge floor + per-player dedup, BEFORE the 15u round cap trims or drops the overflow. "Bets staked" = what's actually staked after the round cap. "Total records" = all gradeable rungs in that round (full scope, model + sportsbet ladders) — the denominator the eligible count is "out of".

| Round | Bets staked | Units staked | Eligible (pre-cap) | Total records |
|--|--:|--:|--:|--:|
| 2026 R16 | 5 | 6.50u | 5 | 34 |
| 2026 R18 | 5 | 3.00u | 5 | 83 |
| 2026 R19 | 4 | 3.25u | 4 | 34 |
| 2026 R20 | 5 | 3.25u | 5 | 93 |

## The question this test exists to answer

**Under Policy C, how many units per round does the bot actually stake?**

Staking (eligibility + sizing) happens before, and independently of, the promo-assumption sweep above — the three promo assumptions only change how an already-placed stake SETTLES, not whether or how much gets staked. So this is one number per round, not one per promo assumption.

Mean: **4.00u/round**, averaged over all 4 scoped rounds (including any at 0u — not dropped).

Per-round list:
- 2026 R16: 6.50u
- 2026 R18: 3.00u
- 2026 R19: 3.25u
- 2026 R20: 3.25u

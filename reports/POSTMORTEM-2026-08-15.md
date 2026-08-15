# Postmortem — 2026-08-15

**The model does not beat closing prices. The betting strategy is retired.**

---

## What was tested

| Test | n | Result |
|---|---|---|
| Prop calibration, walk-forward 2024-25 | 177,480 legs | Calibration itself is fine (gap +0.000 to +0.013 across disposals/goals/marks/tackles, all "WELL CALIBRATED"). The book prices 8-12pp above the model on the same priced legs. That gap is the book's structural margin, not model conservatism — there was never a calibration bug to fix. |
| Single legs, gradeable + book-priced, r16-20 | 590 unique legs (752 leg-occurrences, since a leg reused across multis counts once per bet it rode in) | Flat 1u each: 327 wins vs 355.0 expected — the model is overconfident by ~28 wins against its own probabilities, not just against the book. P&L −86.30u, ROI −14.63% (undeduped: −102.20u, −13.59% on 752). |
| Multis, r16-20 | 232 | 20 of 232 had positive raw edge against the book. 212 didn't. |
| H2H, walk-forward 2013-2026 | 2,786 | Net ROI −9.32% (gross −6.64%). Negative in 12 of 14 seasons. |
| Totals, walk-forward 2013-2026 | 2,578 | Net ROI −5.95% (gross −3.65%). |

Full methodology, per-season and per-bucket breakdowns, and disclosed limitations for the last two rows: `reports/model_vs_close.md`.

## The decisive number

**Mean CLV on the H2H walk-forward: −1.97 percentage points (n=2,786).** Negative in every edge bucket except one (0-2% edge, +0.22pp — noise at that bucket's size), and negative in most individual seasons.

CLV is the number that matters, not ROI. ROI over 2,786 games is still a single, noisy, sample-dependent draw — a real edge can show a losing ROI over any given multi-year stretch, and a fake edge can show a winning one. CLV isn't that: it measures whether the market moves *toward* the model's view between open and close, which happens thousands of times independently of any single game's outcome. If the model had a real edge, the close should drift toward it on average. It drifts away. That's not bad luck across 2,786 games — it's the market consistently taking the other side of the model's disagreement and being right to. Nothing else in this postmortem needed to be true for the strategy to be over; this one number was sufficient on its own.

## The two mistakes that cost the most

**1. Staking on promo EV with negative raw edge.** A 2026-07-10 config change made `PROMO_EV_MIN` the only staking gate, and removed the round-level cap alongside it. 102 of 119 staked bets in rounds 18-20 had zero-or-negative book edge — they were staked anyway because `total_ev` (edge + promo refund value) cleared the floor. That one gate change accounts for −83.1u of the round's −87u loss. It was structurally betting negative-edge multis at size, laundered through promo math the book doesn't actually pay at that scale.

**2. Running two rounds ungraded.** The stats-cache/grading pipeline silently stalled twice: R18-20 in July, then R21-22 in August — R21 had one game graded, R22 had none, a week after R22 recommended 83.2u. Bets were placed and sized with zero live signal on whether the model was working. The first stall should have been the last one; it wasn't, because nothing was made to fail loudly when it happened.

## What stays in place

The edge floor, 15u round cap, and one-multi-per-player cap (`afl_bot/build/staking.py::recommend_units`, `afl_bot/cli.py::_apply_round_stake_caps`) are **not being reverted.** The strategy is retired regardless of whether those guardrails hold — the walk-forward result above is about the model, not the staking policy, and no staking policy fixes a model that doesn't beat the close. They stay because retiring the strategy shouldn't depend on nobody ever running `round-report` again; if it runs, it should not be able to recommend meaningful size.

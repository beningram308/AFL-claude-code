# NEXT-STEPS PLAN — AFL Multi Builder

**For:** Claude Code · **Written:** 2026-06-27 · **Mode:** execute top-to-bottom, commit per step, do NOT stop after one part.

This consolidates everything still outstanding. It supersedes nothing already shipped — verify-don't-redo where a fix is already committed. Work autonomously; only stop to ask if a step needs a real betting/judgement call.

---

## GROUND RULES (unchanged project discipline)

- No look-ahead / no leakage in any backtest. Prior-years-only when fitting calibrators or params.
- Every change: targeted tests added, full suite green before commit.
- Any model/pricing change must NOT regress the walk-forward score (log loss / calibration). If it does, revert and keep the safe default.
- Commit at the end of each part with a clear message. **Push to `origin` when done** (see Part 0).

## ⛔ EXPLICITLY OUT OF SCOPE — the "$1.75 minimum" is retired, leave it that way

Do **not** reintroduce a $1.75 minimum-odds rung, and do **not** add near-lock-assisted legs to force a shorter bottom rung. `FIX-PLACEABLE-LEGS-AND-210-FLOOR` already settled this: with `LEG_PROB_MAX = 0.78`, the honest floor for a 3-leg multi is 0.78³ ≈ **$2.11**, and the ladder is placeable-only. Keep:

```
LEG_PROB_MAX = 0.78
MULTI_TARGET_ODDS = (2.10, 2.75, 3.50, 5.00, 8.00, 15.00)
```

The ladder shows real combo odds (the Band label is the clean bet-slip target, the Fair/Book columns are the truth). Honest odds over round-number cosmetics. No part of this plan should change these values.

---

## PART 0 — REPO HYGIENE & BACKUP (do first, highest risk)

There are ~50 uncommitted modified files in the working tree (config.py, cli.py, sim/engine.py, dashboard/*, reports/*, tests/*, README.md, etc.). The latest work is **not pushed** to `origin`. This is the single biggest risk right now.

1. `git status` and review the diff. Separate real code changes from line-ending (CRLF) churn — set `git config core.autocrlf true` if Windows line-endings are creating noise diffs.
2. Run the full test suite. If green, stage and commit the working tree in logical chunks (don't lump unrelated changes into one commit).
3. If anything is half-finished or fails tests, finish or stash it — do not commit a broken tree.
4. **Push to `origin`** (`https://github.com/beningram308/AFL-claude-code.git`) so the GitHub copy matches local. Confirm the push succeeded and the tree is clean.

Acceptance: `git status` clean, tests green, `git log origin/main..HEAD` empty (local == remote).

---

## PART 1 — VERIFY THE SHIPPED TRACKER FIXES END-TO-END

These are committed (`65dbbfb` settlement + manual bets, `a598602` marks cap + greasiness) but have not all been confirmed on a real regenerated report. Verify, don't rebuild.

1. **Settlement — no phantom wins.** Confirm `dashboard/settle.py` keeps a multi PENDING when any leg is ungradeable, settles WON only when every leg hits, and the re-grade pass reverts previously-wrong won/void bets to pending. Add/keep a test that a "1 hit + 2 no-data" multi stays pending (not WON).
2. **Marks cap on BOTH ladders.** Confirm the cap counts every marks leg (priced + unpriced) so no multi shows 2+ marks legs on the Sportsbet ladder. Regenerate the current round and grep the report to prove ≤1 marks leg per multi.
3. **Greasiness visible + override.** Confirm each game prints its greasiness value and the per-game wet override works (the Collingwood/MCG "looked dry, was slippery" case).
4. **Manual bets.** Confirm the "Add my own bet" path lands in the same ledger, auto-settles when gradeable, and has a manual win/loss/void toggle otherwise.

Acceptance: a freshly regenerated round report + dashboard showing all four behaviours; tests green.

---

## PART 2 — FIX THE BLANK SPORTSBET LADDER

Symptom seen: a report where the Model ladder's Book/Edge columns are all "—" and there's no Sportsbet ladder — i.e. the run had zero book prices.

1. Root cause is one of: `--sportsbet` flag dropped on the regenerate, an empty/missing `reports/<year>_r<N>_sportsbet_urls.json`, or the scrape returning 0 priced events (AU-IP only — Sportsbet geo-blocks non-AU).
2. Make the failure **loud, not silent**: print `Sportsbet: X/N events priced` every run, and if 0 priced, emit a clear warning explaining the likely cause (no URLs file / not AU IP / stale URLs) instead of quietly rendering a model-only ladder that looks broken.
3. Document the one-per-round step in the report header: paste the round's Sportsbet match URLs into the urls JSON, then run `round-report --year <Y> --round <N> --sportsbet`.

Acceptance: running with `--sportsbet` and a populated urls file shows the Sportsbet ladder with real Book/Edge; running without it prints an explicit reason, not a blank.

---

## PART 3 — REAL INJURY / AVAILABILITY FILTER (specced but never built)

The Flanders/Lalor problem: when a team's sheet isn't confirmed at run time, the model falls back to the full current-season pool with **no injury filtering**, so a season-ended or dropped player can appear in a multi. Today there's only a manual outs override and `TOG_RETURN_DEFAULT` — there is no automatic availability check.

1. Add an availability filter that runs on **every** report (not just when a team sheet is posted): fetch the Footywire injury list and drop anyone marked Season / Out / long-term. Lean conservative — if availability is ambiguous, exclude rather than risk.
2. Add a tiny `MANUALLY_UNAVAILABLE` block list in config as an instant insurance net (drop a name, they're gone next run even if the scrape misses them).
3. Add a regression test asserting a known season-out player is excluded even when his team's sheet is NOT fetched.

Acceptance: regenerate a round where a team sheet is unconfirmed and confirm no known-out players appear; test green.

Note: "real players" still means current-season pool until Thursday team sheets drop. This filter reduces the false-positive risk; it does not replace checking picks against confirmed teams before betting.

---

## PART 4 — OPERATIONAL WEEKLY LOOP

The model side is essentially done; the value now comes from running it and letting the record accumulate.

1. **Thursday (after team sheets):** `round-report --year <Y> --round <N> --sportsbet` (+ paste prop odds / urls), check picks against confirmed teams, place bets.
2. **Monday:** `settle-bets` / reload the dashboard and `grade-round` the saved predictions.
3. Let `reports/calibration_log.csv` and the CLV panel build over a real sample of weeks — this is the only thing that proves the edge is real rather than backtested.
4. (Optional, offer to Ben) a scheduled task to auto-run the Thursday-evening report each round.

Acceptance: one full real-round cycle logged (report → settle → grade) with the calibration log appended.

---

## PART 5 — HOUSEKEEPING (Phase 5 from MODEL-UPGRADE-INSTRUCTIONS.md)

Low-risk cleanup, no modelling change:
- Remove the dead `PROP_FORM_GAMES` knob if still unused.
- Hourly (not daily) wet flag where the weather archive supports it.
- Stale-calibrator guard: warn if `prop_calibrators.json` is older than the latest data.

Acceptance: tests green, no walk-forward change.

---

## PART 6 — GRADIENT-BOOSTED PROP MODEL (Phase 6, last, optional, highest-value new model)

The one genuinely valuable method not yet in the system. Build a gradient-boosted (e.g. XGBoost/LightGBM) per-stat prop model that **blends with** the sim — it does not replace it.

1. Features: player form/EWMA, role (CBA/TOG/position), opponent matchup, venue, weather/greasiness, rest/travel.
2. Train strictly walk-forward (prior seasons only), output a probability per line, then blend with the sim's prop probability (start with a documented prior weight, not a fitted one, until there's enough history to fit honestly).
3. Acceptance is mechanical and strict: the blended prop probabilities must **beat the current calibrated sim on out-of-sample log loss / Brier**, or it stays opt-in and off by default. Overfitting here will silently lie — guard hard.

Do NOT build: Poisson (NB already chosen), Glicko/TrueSkill, Markov, neural nets, SVM, causal/Bayesian-net formalism, news sentiment — all judged noise for this data (see AUDIT-AND-NEXT-STEPS.md).

---

## SUGGESTED ORDER

0 → 1 → 2 → 3 → 4 (start the weekly loop as soon as 0–3 are solid) → 5 → 6.

Parts 0–3 are "make the thing you bet off trustworthy and backed up." Part 4 is "use it." Parts 5–6 are improvements, not blockers. Commit each part; push at the end of each work session.

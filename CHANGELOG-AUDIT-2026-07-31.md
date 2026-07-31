# Audit change log — 2026-07-31

Originals of every modified file are preserved in `audit_backup_20260731/`.
No modelling assumption was changed silently; the two changes that move model
outputs (venue pooling) are flagged below and quantified in the audit report.

## Bug fixes

1. **`afl_bot/build/multi.py` — `promo_multi_ev`** (HIGH)
   One-loss branch credited the bonus bet (`+ p_one_loss * stake * bonus_factor`)
   without deducting the lost stake. EV was overstated by exactly
   `p_one_loss * stake` (~+44pp on a typical 3×0.70 multi), so
   `build_promo_multi` (the `run-round` promo path) surfaced deeply −EV multis
   as +EV. Now `+ p_one_loss * stake * (bonus_factor - 1)`, matching the exact
   identity `EV/s = p_all*M − 1 + p_one_loss*R` already used correctly by
   `build/report.py` (total_ev) and `build/staking.py::multi_outcome_kelly`.
   The `round-report` path was already correct and is unchanged.

2. **`afl_bot/data/odds.py` — `attach_odds`** (HIGH)
   The `(year, hteam, ateam)` join key is not unique (finals rematches,
   same-home-team rematches: 63 duplicate keys in 2016–26). The old merge
   expanded 2,224 games → 2,354 rows and attached the wrong game's odds to
   every duplicated pair, contaminating the market benchmark, ensemble blend
   training and CLV. Replaced with a date-proximity join
   (`merge_asof`, nearest date within 3 days, by home/away team). Date-less
   synthetic frames fall back to the legacy key join with the odds side
   deduplicated so it can never expand rows.
   Corrected market benchmark (closing devig): LL 0.5759 (2016+), 0.5607 (2022+)
   — the bar is *higher* than the contaminated numbers previously suggested.

3. **`afl_bot/cli.py` — `grade_round` log dedup** (MEDIUM)
   Re-grading a round dropped that round *number* from every season in
   `calibration_log.csv` (grading 2026 R1 would have wiped the 2025 R1
   record). Now filters on `(year, round)`.

4. **`afl_bot/cli.py` — `grade_round` staleness guard** (MEDIUM)
   When the player-stats cache doesn't cover the graded round, every prop
   prediction was silently skipped (this is how 2026 R18–R20 "graded" only
   24/15/27 of 700–1,300 predictions). Now prints a loud warning telling you
   to refresh the DFS cache and re-grade.

5. **`afl_bot/models/scoring.py` + `afl_bot/data/venues.py` — venue alias
   pooling** (MEDIUM, changes model output, deliberate)
   The games history names the same ground inconsistently (Docklands/Marvel,
   Kardinia Park/GMHBA, Perth Stadium/Optus, York Park/UTAS, Eureka/Mars,
   Jiangwan aliases). `venue_scoring_factors` split one ground's sample across
   aliases and over-shrunk the estimate. Added `VENUE_ALIASES` /
   `canonical_venue` and pooled before grouping; factors are exported under
   canonical name *and* aliases so existing lookups resolve.
   Live-name impact: Marvel 0.9912 → 1.0337, Optus 1.0111 → 0.9647,
   GMHBA 0.9955 → 1.0095, UTAS 1.0032 → 0.9784 (multiplier on expected total).

6. **`afl_bot/ratings/hga.py` — `days_rest` / `game_hga_points`** (MEDIUM)
   `days_rest` returned a unixtime-sorted frame; `game_hga_points` re-labelled
   it positionally with the caller's index and combined it with arrays in the
   caller's order — misaligning every rest adjustment whenever the input wasn't
   already unixtime-sorted. Rest is still computed chronologically but is now
   mapped back to the caller's row order. Output is now order-invariant
   (regression-tested).

## Data / records

7. **`reports/calibration_log.csv`** — graded 2026 R17–R20 appended
   (146 new graded predictions; log previously contained only 2025 R1).
   R21 was mid-round at audit time. R18–R20 props remain ungradeable until the
   DFS player-stats cache is refreshed past R17 — re-run `grade-round` for
   those rounds after refreshing.

## Tests

8. **`tests/test_audit_fixes.py`** — 10 new regression tests covering all of
   the above (EV identity, EV/Kelly gate consistency, join non-expansion,
   date-skew tolerance, year-aware dedup, alias pooling, rest-order
   invariance). Full fast suite: 106 passed.

---

# Phase 2 — validated model improvements (same day)

Both changes below were selected on 2022–24 and validated ONCE on untouched
2025–26. Neither was hand-tuned against the validation window.

9. **`data_cache/elo_params.json` (new artifact) — margin scale fixed via the
   repo's own tuned-params mechanism.** The Elo margin mapping was 21% too
   compressed (walk-forward regression slope actual~predicted = 1.27; the
   default tuning grid never included `points_per_400`, so the repo's own
   tuner could never have found this). Artifact sets `points_per_400: 116.0`;
   both `run-round` and `round-report` already auto-load it
   (`load_fitted_elo_params`). Config defaults untouched — delete the artifact
   to revert. Validation (2025–26, n=388): sim-implied H2H log loss
   0.5479 → 0.5322, margin slope 1.302 → 1.037, margin MAE 26.28 → 26.20.
   Only this one parameter was adopted; the grid's k/carryover moves ranked
   within selection-window noise and were rejected.

10. **`afl_bot/backtest/ensemble.py` + `afl_bot/cli.py` — C7 fixed:
    blend calibrator now trained on the distribution it's applied to.**
    `assemble_signals(sim_style=True)` expresses the model signal as
    `Phi(pred_margin / SIM_MARGIN_SIGMA)` (new config constant, 39.9 — the
    sigma the scoring-shots engine actually produces), and `run-round` fits
    the blend that way. Previously the isotonic was trained on Elo-logistic
    probabilities and applied to Monte-Carlo probabilities. Validation
    (blend, 2025–26 holdout, n=330): LL 0.5300 → 0.5245 (closing market
    alone: 0.5248). Also fixed `assemble_signals`' own copy of the rematch
    odds-join bug (now routes through the date-aware `attach_odds`).

11. **`tests/test_audit_fixes.py`** — 3 more regression tests: sim-style
    signal formula, ensemble join non-expansion, artifact load-and-apply.

12. **`WEEKLY-RUNBOOK.md`** — the operating loop that generates the evidence
    (grading + CLV) this system still lacks.

---

# Phase 3 — ops/hygiene follow-up (same day)

Ops, reporting, and hygiene only. No modelling assumption, constant, or
probability calculation changed.

13. **Grading backlog cleared.** Force-refreshed the DFS Australia player-stats
    cache (now covers 2026 R0–R21). Ran `grade-round` for 2026 R18, R19, R20 —
    no stale-data warning fired (C4's guard from Phase 1 stayed silent, as
    expected once the cache is current). R21 was NOT graded: only 1 of 9 games
    complete at run time. `calibration_log.csv` now holds 2025 R1 (untouched)
    + 2026 R17–R20 (4,339 predictions across 5 rounds).
    Per-market summary, 2026 R17–R20 combined (3,191 predictions):

    | market | n | log loss | Brier | mean pred | hit rate |
    |---|---|---|---|---|---|
    | h2h | 46 | 0.448 | 0.143 | 0.496 | 0.500 |
    | player_disposals | 884 | 0.643 | 0.226 | 0.516 | 0.575 |
    | player_goals | 478 | 0.648 | 0.228 | 0.496 | 0.498 |
    | player_marks | 1204 | 0.681 | 0.244 | 0.497 | 0.538 |
    | player_tackles | 556 | 0.694 | 0.250 | 0.490 | 0.540 |
    | total_points | 23 | 0.693 | 0.250 | 0.434 | 0.391 |
    | **all markets** | **3191** | **0.664** | **0.236** | **0.501** | **0.541** |

    (h2h count above is R17–R20 only, 46 rows; the cumulative log includes
    2025 R1's H2H rows too — see `reports/calibration_summary.md`, added in
    the next item, for the full per-round breakdown.)

14. **`reports/calibration_summary.md` (new) + `grade_round` breakdown report**
    (`afl_bot/grading.py`, see item 19). After the existing summary lines,
    `grade-round` now also prints and appends a per-round section: per-market
    n/log-loss/Brier/mean-pred/hit-rate, an H2H favourite-vs-underdog split,
    and a probability-bucket reliability table (via
    `afl_bot.backtest.walkforward.calibration_curve`). Pure reporting — does
    not alter what's written to `calibration_log.csv`. Re-running `grade-round`
    on an already-graded round replaces that round's section in the summary
    file instead of duplicating it (fixed a blank-line-accumulation
    idempotency bug found during item 19's byte-identical check — repeated
    re-grades of the same round now produce byte-identical file content, not
    just an unchanged section). Tests:
    `test_calibration_summary_written_and_sectioned`,
    `test_calibration_summary_regrade_replaces_not_duplicates`,
    `test_calibration_summary_regrade_is_idempotent` in
    `tests/test_audit_fixes.py`.

15. **`requirements.lock`** (new) — `pip freeze` from the environment that runs
    the bot (50 packages). `README.md` gained a "Reproducible install" note
    pointing at it. `requirements.txt` unchanged.

16. **`pytest.ini`** (new) — registers a `slow` marker.
    `tests/test_multis_backtest.py` and `tests/test_stake_cap_backtest.py`
    gained module-level `pytestmark = pytest.mark.slow`. Fast suite:
    `pytest -m "not slow"` (632 passed, 56 deselected, ~57-67s). Full suite:
    `pytest`. Note: in isolation, `test_stake_cap_backtest.py` actually ran in
    1.8s (not slow by the >30s criterion) but is marked anyway since the task
    named it explicitly; `test_multis_backtest.py` is genuinely slow — timed
    in isolation at **36m05s (2165.5s) for 35 tests**, confirming the marker
    is doing real work, not just following instructions.

17. **Silent exception swallowing removed from grading/fetch/settlement
    paths.** Every bare `except Exception`/`except Exception: pass` in the
    fryzigg-then-DFS-Australia player-stat fallback (the same two-source
    pattern reused in three places) now prints a one-line stderr warning
    naming the source and the exception before falling through — same
    fallback behaviour, no control-flow change:
    - `afl_bot/grading.py::grade_round` (both the fryzigg and DFS branches;
      lived in `afl_bot/cli.py` at the time of this fix, moved by item 19)
    - `afl_bot/backtest/multis.py::_fetch_actual_player_log` (both branches —
      the walk-forward backtest's copy of the same fallback)
    - `afl_bot/dashboard/settle.py::_load_actuals` (both branches — the
      `settle-bets` copy of the same fallback)
    - `afl_bot/data/lineups.py::parse_footywire_injury_list` (BeautifulSoup
      parse failure, previously silently returned `{}`)
    - `afl_bot/dashboard/app.py::_load_multis_files` (one malformed report
      JSON no longer disappears from the round list with zero trace)
    - `afl_bot/dashboard/capture_close.py` (consensus-devig failure inside
      the per-leg CLV loop — kept the existing `break`, just prints why first)

    Left deliberately unchanged (silent-by-design, not grading/fetch paths,
    noted here rather than silently skipped): `backtest/correlations.py` and
    `backtest/tuning.py`'s `git rev-parse` version-tag helpers (cosmetic,
    return `None` on any git-less environment); `dashboard/app.py`'s
    stale-file-timestamp check (cosmetic UI banner); `dashboard/ledger.py`'s
    optional corruption-notice reader (already has explicit not-found
    handling above it); `io_utils.py::atomic_write_text` (re-raises after
    cleanup — not a silent swallow to begin with).

18. **Tests**: 3 new regression tests in `tests/test_audit_fixes.py`
    (`test_calibration_summary_written_and_sectioned`,
    `test_calibration_summary_regrade_replaces_not_duplicates`,
    `test_calibration_summary_regrade_is_idempotent`) covering task 2's
    calibration-summary file, its no-duplicate-section re-grade behaviour,
    and the idempotency fix from item 14. Full fast suite
    (`pytest -m "not slow"`): 632 passed, 56 deselected.

19. **`afl_bot/grading.py` (new, task 6)** — `grade_round`,
    `_format_calibration_section`, and `_write_calibration_summary` moved out
    of `afl_bot/cli.py` (2,698 lines, flagged in §7 of the audit report as a
    monolith mixing orchestration/formatting/modelling glue) verbatim, zero
    behaviour change. `cli.py` now does `from afl_bot.grading import
    grade_round`. Verified: fast suite green before and after (632 passed
    both times); `grade-round --year 2026 --round 18` (an already-graded
    round) produces byte-identical console output before/after (same
    predictions count, log loss, Brier, cumulative numbers) and byte-identical
    `calibration_summary.md` section content — `tests/test_audit_fixes.py`
    imports now target `afl_bot.grading` directly rather than re-exporting
    through `cli`.

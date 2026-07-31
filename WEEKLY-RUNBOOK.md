# Weekly runbook — the loop that decides whether this bot ever earns real stakes

The model is now roughly market-grade on H2H and well-calibrated on props.
That is NOT an edge. The only thing that can prove an edge is this loop run
every week without exception. Skipping it is how the calibration log ended up
with one round in it for 16 months.

## Thursday night (after team sheets drop)

```bash
# 1. Paste this round's Sportsbet match URLs into:
#    reports/2026_r<N>_sportsbet_urls.json
# 2. Run the round report (real lineups + real prices):
python -m afl_bot.cli round-report --year 2026 --round <N> --auto-lineup --sportsbet
```

Rules that are not optional:
- Stake NOTHING tagged MODEL-ONLY, NO BET, or CHECK PRICING.
- Respect the unit caps as printed. Do not round up.
- If a rung's edge looks huge (>40%), assume a data error, not free money —
  the SUSPECT guard exists because that has happened.

## Monday (after all games complete)

```bash
# 1. Refresh player stats FIRST — grading props needs them.
python -c "from afl_bot.data.dfs_australia import fetch_player_stats; fetch_player_stats(force_refresh=True)"

# 2. Settle the ledger, grade the round, snapshot closing prices:
python -m afl_bot.cli settle-bets  --year 2026 --round <N>
python -m afl_bot.cli grade-round  --year 2026 --round <N>
python -m afl_bot.cli capture-close --year 2026 --round <N>
```

`grade-round` now WARNS if props couldn't be graded because the stats cache is
stale. If you see that warning, refresh (step 1) and re-run — the fix in this
audit means re-grading a round safely replaces only that round's rows.

## Backlog to clear once (do this week)

2026 R18–R20 props were never graded (stats cache was stale). After a
force-refresh that reaches those rounds:

```bash
python -m afl_bot.cli grade-round --year 2026 --round 18
python -m afl_bot.cli grade-round --year 2026 --round 19
python -m afl_bot.cli grade-round --year 2026 --round 20
python -m afl_bot.cli grade-round --year 2026 --round 21   # once R21 completes
```

## Decision gates (write these down, hold yourself to them)

- **Until 200+ graded selection-level rungs**: paper-trade or minimum stakes
  only. 34 graded rungs and 10 real bets prove nothing in either direction.
- **Stake increase**: only if, at 200+ rungs, cumulative CLV is positive AND
  realized ROI is above −5% (promo value can carry a small negative raw ROI).
- **Stop condition**: drawdown > 30% of bankroll, or CLV clearly negative at
  100+ rungs → stop betting, keep grading. The model keeps its value as a
  paper record either way.
- **Never** bet a market the grading loop can't grade (if it can't be graded,
  it can't be audited).

## What changed under you this audit (one line each)

- Promo EV formula fixed (was overstating by ~44pp on run-round path).
- Odds joins fixed (market benchmark/CLV/blend no longer see wrong-game odds).
- Venue factors pooled across name aliases (Marvel/Optus totals shifted ~4%).
- Margin scale fixed via `data_cache/elo_params.json` (slope 1.30 → 1.04 OOS).
- H2H blend calibrator now trained on sim-style probabilities (C7).
- Grading is year-safe, warns on stale data; R17–R20 H2H/totals graded.
- 13 regression tests in `tests/test_audit_fixes.py` pin all of it.

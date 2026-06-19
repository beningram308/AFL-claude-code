# Instructions: real, placeable multis (lineups + bettable legs)

**From Ben — read fully, then execute autonomously. Don't pause to ask; only stop if genuinely blocked.**

## The end goal (don't lose sight of this)
Each round, pump out the **highest-probability 3-leg same-game multis from ~1.75 to ~5.0 for every game** —
and every leg must be:
1. on a player who is **actually playing** that round (not injured/dropped/rested), and
2. a **line a bookie actually offers** (not a model-invented threshold).

No hand-entered files. I run one command, I get bettable multis per game.

## What's already done (do NOT redo)
- 40-game form window + TOG scaling (`FORM-WINDOW-INSTRUCTIONS.md`) — keep.
- `--multis-only` flag prints just the per-match multi ladders — keep.
- 3-leg ladder banded ~1.75→5.0 via `search_match_sgms` / `DEFAULT_ODDS_BANDS` — keep.
- `load_lineup` / `load_lineup_tog` (manual lineup JSON) — keep as the manual override path.

Everything below is what's still **wrong** and needs fixing.

---

## PROBLEM 1 — it prices players who aren't playing (e.g. Sam Lalor)
The pool comes from `_select_players(player_log, team, current_season, PLAYERS_PER_TEAM_SAMPLE,
confirmed=lineup.get(team))` (cli.py ~513 and ~778). With no lineup file, `confirmed=None`, so it
prices the top-usage **current-season** players — including anyone currently injured/out. That's why
Sam Lalor still shows up with stats when he's out.

### Fix: automatic lineup fetching
Add `fetch_lineup(year, round_no) -> dict[str, set[str]]` in `afl_bot/data/lineups.py` that pulls the
round's **selected teams** from a free source and returns the same shape `load_lineup` returns.

- **Primary source — Footywire team selections:** `https://www.footywire.com/afl/footy/afl_team_selections`
  (the free source Fable identified; port the parse approach from fitzRoy's `fetch_lineup`,
  `https://github.com/jimmyday12/fitzRoy/blob/main/R/fetch_lineup.R`).
- Normalise every team + player name through `normalize_team_name` and the canonical player-id path
  so fetched names match the projection pool (Fryzigg/DFS spell names differently — §7.1).
- Add a `--auto-lineup` flag to `round-report`, **default ON for the live round**. When set, call
  `fetch_lineup` and feed the result in as `confirmed=` exactly like a manual `--lineup` file, so any
  player not named is dropped from projections **and** every multi.
- **Timing reality:** team sheets only lock ~Thursday night. Before they post, `fetch_lineup` should
  fall back to removing **known unavailable players** via an injury/outs list so long-term outs (like
  Lalor) are still excluded. Print which source supplied the lineup, the round, and **how many players
  were excluded**, so it's visible whether real teams or just the injury list were used.
- A manual `--lineup` file **overrides** the auto-fetch (hand-fix path). Keep the synthetic-data guard
  and real-player guard.

---

## PROBLEM 2 — the legs aren't bettable lines, and "fair odds" is meaningless on them
Leg lines come from `PROP_LINES` (config) and every threshold is emitted if it passes the gate
`if not (0.05 < prob < 0.97): continue` (cli.py:534 and cli.py:798). That 0.97 ceiling lets through
near-locks like **"Jordan Dawson 15+ disposals"** (~97%) — a line **no bookmaker posts** (his lowest
real line is ~20+/25+). The leg odds are then `odds_book.get(name, fair_odds(prob))` — i.e. when there's
no live market, the odds shown are the **model's** price `1/prob`, not a bookie price. So the multi's
"fair odds" is computed from lines you can't even place. That's the core bug.

### Fix A — when live prop odds ARE available (the proper fix)
Build legs **only from the actual posted markets**. `afl_bot/data/live_odds.py` already fetches
h2h/totals from The Odds API (`aussierules_afl`) but **not player props** — props live on the
**per-event** endpoint as an additional/paid market (`player_disposals_over`, `player_goals_scored_over`,
`player_marks_over`, `player_tackles_over`). Wire that per-event prop fetch (it's Part A of
`MULTI-CHANGES-INSTRUCTIONS.md`, specced but never run). Then:
- each leg is a real `{player, stat, line, price}` a book is offering → **placeable by construction**;
- the multi's edge = model joint prob vs the book's price (market-shrunk, as already implemented);
- only build multis from legs that exist in the live market.
- Reads `ODDS_API_KEY` from env (props tier). If unset/empty, fall through to Fix B.

### Fix B — when there's NO odds feed (interim, until the key exists)
Stop inventing silly lines:
- Restrict candidate lines to **bookie-realistic increments** per market. Ensure `PROP_LINES` covers
  the real ladder and goes high enough for ball-magnets: disposals `15/20/25/30/35`, goals `1/2/3`,
  marks `4/5/6/7/8`, tackles `3/4/5/6/7`.
- **Tighten the gate** from `0.05 < prob < 0.97` to a **bettable window ~`0.30 < prob < 0.78`** at both
  cli.py:534 and cli.py:798 (put the bounds in config, e.g. `LEG_PROB_MIN = 0.30`, `LEG_PROB_MAX = 0.78`).
  This automatically drops "Dawson 15+" (too short to be posted) and keeps his 25+/30+ line near his
  projection. Same effect for marks/tackles/goals.
- For each player+stat, this naturally surfaces the line(s) a book would actually hang (straddling the
  projection), instead of every threshold.

### Fix C — label what's real vs modelled (so the odds aren't misleading)
- Mark each leg as **live-market** (real line + book price) or **model-suggested** (model line, no price).
- Show the **"Fair odds" column only as the model price**, clearly labelled as model-implied.
- Show **Book odds + Edge + VALUE PICK only when a real market price exists** for every leg in the multi
  (this is already the rule — keep it). A multi built entirely of model-suggested legs must NOT show a
  book price or be flagged value; it's a model suggestion to look up at the book.

---

## Data sources — use the EXACT sites (do not substitute)
| Purpose | Source | URL |
|---|---|---|
| Round team selections (who's playing) | **Footywire team selections** | `https://www.footywire.com/afl/footy/afl_team_selections` |
| Lineup parse reference | fitzRoy `fetch_lineup` | `https://github.com/jimmyday12/fitzRoy/blob/main/R/fetch_lineup.R` |
| Live odds + player props (real lines/prices) | **The Odds API** (`aussierules_afl`) | `https://the-odds-api.com` |
| Player history / projections | Fryzigg via fitzRoy | `https://jimmyday12.github.io/fitzRoy/articles/using-fryzigg-stats.html` |
| Current-season box scores (TOG/CBA) | DFS Australia | `https://dfsaustralia.com/afl-stats-download/` |

---

## Tests
- `fetch_lineup`: mock the HTTP fetch; assert it returns `{team: set[player]}` with normalised names,
  and that an "out" player is **excluded from the pool and from every multi**.
- `--auto-lineup`: a player not in the fetched lineup never appears in projections or legs.
- Leg gate: no emitted leg has model prob outside `[LEG_PROB_MIN, LEG_PROB_MAX]`; a high-usage player
  (project ~28 disposals) gets a 25+/30+ leg and **never** 15+.
- Live props (mock): legs come only from posted markets; a multi of real legs shows book odds + edge; a
  multi of model-suggested legs shows fair odds only and no VALUE flag.
- Update any existing test that assumed the old `0.05/0.97` gate or 15+ legs for stars.

## Verify
1. `pytest -q` green, `ruff check` clean.
2. `python -m afl_bot.cli round-report --year 2026 --round 15 --multis-only --auto-lineup`
   - every game shows a 3-leg ladder ~1.75→5.0,
   - no out/unnamed players in any leg (spot-check Sam Lalor is gone),
   - no sub-bettable lines like "<star> 15+ disposals",
   - the header states the lineup source + how many players were excluded.
3. If `ODDS_API_KEY` is set: confirm legs carry real book lines/prices and edges compute; if not,
   confirm legs are labelled model-suggested with fair odds only.

## Acceptance (how Ben judges it)
One command → for round 15, the top 3-leg multis 1.75→5.0 per game, built only from players actually
named to play and only from lines a bookie would post. Nothing in a multi that can't be placed.

> Note: the **full** version of this (real lines + real prices + true edge) needs an `ODDS_API_KEY` on
> the player-props tier. Fixes A/B/C make the output honest and placeable either way; the key turns
> "model suggestion" into "priced edge."

---

## ADDENDUM 1 — ladder layout + rung spacing (Ben, after first run)
Two tweaks to how the multi ladder is presented. Modelling/sim logic untouched.

1. **Keep the table layout.** Always render the same-game multi ladder as the markdown table
   (`Legs | Joint prob | Fair odds | Corr gain | Pick`), one per game (the `render_markdown` /
   `--multis-only` output). Do NOT replace it with a prose "standout legs" summary — Ben wants the
   table itself shown.
2. **Make the three rungs target ~$1.75, ~$3.50 and ~$5.00** (currently they land ~1.75 / 2.50 / 3.50 —
   the top rung never reaches $5). Replace the `DEFAULT_ODDS_BANDS` selection in
   `afl_bot/build/report.py` `search_match_sgms` with **target odds**:
   - add `MULTI_TARGET_ODDS = (1.75, 3.50, 5.00)` to config (tunable, not hard-coded);
   - for each target, pick the 3-leg combo whose **fair odds is closest to that target**, using
     highest joint prob as the tie-break;
   - de-dupe so one combo can't fill two rungs;
   - keep the per-game fill so every game still shows all three rungs even when the leg pool is thin.
   When live book odds exist, still compute book odds + edge + the VALUE flag on the chosen rungs as
   today; the targeting is on fair odds, the value flag stays on edge.

**Verify:** `python -m afl_bot.cli round-report --year 2026 --round 15 --multis-only` →
every game shows the table with three rungs at roughly $1.75, $3.50 and $5.00.

---

## ADDENDUM 2 — exclude injured / unavailable players (the injury fallback was never built)
**Bug:** season-ended / injured players are still being priced (e.g. **Sam Flanders** is out for the
season but appears in St Kilda multis; same class of bug as Sam Lalor earlier). Root cause: the pool is
the current-season top-usage players, and `afl_bot/data/lineups.py` only has `fetch_lineup` (Footywire
**team selections**, `FOOTYWIRE_SELECTIONS_URL`). There is **no injury handling at all**. So a team
whose sheet isn't posted yet (St Kilda this run — only 13 of 18 teams were confirmed) falls back to the
full pool with nothing filtering out injured players. A player who played early in the season and is now
out for the year is still "current-season," so he stays in.

This addendum builds the availability filter the original spec called for and never got.

### Fix 1 (primary, automatic) — fetch and apply the injury list every run
- Add `fetch_injury_list() -> dict[str, str]` to `afl_bot/data/lineups.py` (player name → status),
  scraping **Footywire's injury list**: `https://www.footywire.com/afl/footy/injury_list` (same site as
  team selections; reuse the request/parse pattern). Normalise player names through the canonical
  player-id path so they match the projection pool.
- Apply it as a **second exclusion pass that ALWAYS runs**, in both `round-report` and `run-round`,
  **independent of whether the team's sheet was fetched**. Any player whose status marks them
  unavailable is dropped from the pool **and** from every leg/multi:
  - definitely out: `Season`, `Indefinite`, `Out`, `Long-term`, and any multi-week return
    (e.g. "4+ weeks", "3-4 weeks", "TBC").
  - be conservative: if availability is ambiguous (`Test`, `1 week`), **exclude** — better to miss a
    multi than price someone who doesn't play. Make the cutoff a tunable set in config
    (`INJURY_EXCLUDE_STATUSES`).
- This catches injured players on pool-fallback teams (the exact St Kilda/Flanders case).

### Fix 2 (safety net) — manual always-exclude list
- Add `MANUALLY_UNAVAILABLE: set[str]` in config (a tiny editable name list, e.g. `{"Sam Flanders",
  "Sam Lalor"}`). These names are **always** filtered, regardless of scrape result. One-line insurance
  for a known long-term out if the injury-list scrape ever misses or lags. (Not a lineup file — just a
  block list.)

### Hard rule
A player on the injury list (unavailable status) **or** in `MANUALLY_UNAVAILABLE` is **never** priced,
never appears in a leg, never in a multi — even if their team's confirmed sheet isn't posted.

### Reporting
In the lineup/exclusion note, break the counts out: `N excluded by team sheet, M excluded by injury
list, K manual`. So it's visible that the injury filter actually fired.

### Tests
- Mock `fetch_injury_list` returning `{"Sam Flanders": "Season"}`; assert Flanders is absent from the
  pool and from every multi **even when St Kilda's team sheet is NOT fetched** (pool-fallback path).
- `MANUALLY_UNAVAILABLE` name never appears in any leg/multi.
- Ambiguous status ("Test") is excluded under the default `INJURY_EXCLUDE_STATUSES`.

### Verify
`python -m afl_bot.cli round-report --year 2026 --round 15 --multis-only` → confirm **Flanders is gone**
from the St Kilda ladder, and the note shows a non-zero "excluded by injury list" count.

### Data source (exact)
| Purpose | Source | URL |
|---|---|---|
| Player injuries / availability | **Footywire injury list** | `https://www.footywire.com/afl/footy/injury_list` |

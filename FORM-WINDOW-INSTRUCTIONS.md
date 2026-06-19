# Instructions: last-40-games form window + time-on-ground projection

**From Ben — read fully, then execute autonomously. Don't pause to ask; only stop if genuinely blocked.**

## Goal
Stop projecting player stats off *all* past seasons. Project each player off their
**last 40 games**, then scale that to the minutes you expect them to play **this** game —
their time-on-ground (TOG), which depends on the named lineup and whether they're coming
back from injury (a returning/managed player gets capped, e.g. **~75% TOG**).

Two locked decisions (Ben):
1. **Window math:** EWMA recency weighting, but capped to each player's last 40 games,
   **then multiplied by a per-game TOG factor** for the match being priced. TOG depends on
   the lineup / injury status.
2. **Thin history (< 40 games):** use however many games the player has and lean on the
   existing role/position prior shrinkage. **Nobody gets dropped** for a short history.

**Do NOT touch the sim/scoring/odds/multi logic.** This change is confined to the player
projection layer (`models/props.py`, `models/priors.py`), one config block, the lineup
plug-point schema, and the call site in `cli.py`. Plus tests.

---

## PART A — Config tunables (`afl_bot/config.py`)
House rule: tunables live in config, not hard-coded. Add a single form-window block. Keep
the existing `PROP_EWMA_HALFLIFE`, `PROP_RECENT_SEASONS`, `PROP_MIN_DISPERSION`,
`PROP_PRIOR_STRENGTH`, `PROP_DISPERSION_PRIOR_STRENGTH` as-is.

```python
# --- Player form window (last-N games, not all-history) ---
PLAYER_FORM_WINDOW = 40          # project each player off their most recent 40 games
TOG_RETURN_DEFAULT = 0.75        # expected TOG for a player flagged returning-from-injury / managed
# TOG_MULT_BOUNDS already exists — keep it as the clip on the TOG multiplier.
```

`PLAYER_FORM_WINDOW` must be the **only** place the number 40 lives — everything below reads
it from config so it's tunable later.

---

## PART B — Make every player-history average use the last 40 games

The model currently averages a player's own history in several places. Some already window
(20 games), some use **all** seasons (2012+). Point them all at `PLAYER_FORM_WINDOW`.

### B1. `afl_bot/models/props.py` → `player_rate_profile(...)`
This is the main per-stat baseline. It already does `rows.tail(lookback_games)` after
sorting by `["year","round","unixtime"]`, then EWMA. Just change the default:

```python
def player_rate_profile(
    log, player, stat,
    as_of_year=None, as_of_round=None,
    halflife=PROP_EWMA_HALFLIFE, lookback_games=PLAYER_FORM_WINDOW,   # was 20
):
```

Keep the anti-leakage `as_of_year/as_of_round` filter exactly as-is (it runs **before**
`.tail()`, so "last 40 games strictly before this round" stays correct). The EWMA stays —
this gives "EWMA within the last 40", per the locked decision.

### B2. `afl_bot/models/props.py` → `estimate_dispersion(...)`
Currently `log.groupby("player")[stat]` over the **whole** log → variance is computed across
all seasons. Window it to each player's last `PLAYER_FORM_WINDOW` games first:

```python
def estimate_dispersion(log, stat, min_games=6, window=PLAYER_FORM_WINDOW):
    recent = (
        log.sort_values(["year", "round", "unixtime"])
           .groupby("player", group_keys=False)
           .tail(window)
    )
    grouped = recent.groupby("player")[stat].agg(["mean", "var", "count"])
    # ...rest unchanged: pooled fallback for count < min_games, floor at PROP_MIN_DISPERSION
```
The `< min_games` pooled fallback already covers thin histories (decision 2).

### B3. `afl_bot/models/priors.py` → `_player_means_and_shares(...)`
This is the **all-history offender**: `groupby("player")[stat].mean()` over 2012+. It feeds
the role priors and the shrinkage baselines. Window it the same way:

```python
def _player_means_and_shares(log, stat, window=PLAYER_FORM_WINDOW):
    work = log.sort_values(["year", "round", "unixtime"]).copy()
    for col in (...):  # team-total column build stays the same
        work[col] = work.groupby(["year","round","team"])[stat].transform("sum")
    work = work.groupby("player", group_keys=False).tail(window)   # NEW: last-40 per player
    player_mean = work.groupby("player")[stat].mean()
    share = ...                       # unchanged, but now over the windowed rows
    player_share = share.groupby(work["player"]).mean()
    return player_mean, player_share
```

### B4. `afl_bot/models/priors.py` → `estimate_dispersion_hierarchical(...)`
Same fix as B2 — window each player to the last `PLAYER_FORM_WINDOW` games before
`groupby("player")[stat].agg(["mean","var","count"])`. Keep the hierarchical/role prior
shrinkage for players under `min_games`.

### B5. TOG / CBA baselines (`priors.py` `_recent_baseline`, `player_tog`, `player_cba`)
These already use a short recent window (`recent_games=4`) for the "recent" value and an
EWMA for the "baseline". For the **historical TOG baseline** that the TOG multiplier divides
by (Part C), compute the EWMA over the **last `PLAYER_FORM_WINDOW` games** so the denominator
matches the form window:

```python
def player_tog(log, player, recent_games=4, halflife=PROP_EWMA_HALFLIFE,
               window=PLAYER_FORM_WINDOW):
    # _recent_baseline should .tail(window) before the EWMA so 'baseline' = last-40 TOG.
```
Leave `recent_games=4` as the *recent-form* TOG (used only when there's no lineup override).

> **Why all five:** B1 is the headline change, but if B3/B4 keep averaging 2012+ data the
> shrinkage prior silently drags every projection back toward a 14-year mean — defeating the
> point. All player-own-history averages must share the same 40-game window.

---

## PART C — Time-on-ground projection (the "75% if coming off injury" part)

The machinery already exists — wire it into the report. The pieces:
- `priors.player_tog(log, player)` → `(recent_tog, baseline_tog)`.
- `priors.tog_multiplier(projected_tog, historical_tog, bounds=TOG_MULT_BOUNDS)` →
  scales counts by `projected_tog / historical_tog`, clipped, neutral (1.0) if either is missing.
- `props.expected_stat_mean(baseline_mean, share, team_total_mean, matchup_mult, context_mult)`
  → multiplies by `context_mult`. **TOG goes in here as `context_mult`** (compose with the
  existing CBA role multiplier if present: `context_mult = tog_mult * cba_mult`).

### C1. Projected TOG for THIS game (priority order)
For each player being priced, decide `projected_tog` like this:
1. **Lineup override** — if the lineup JSON (Part C2) gives an `expected_tog` for the player,
   use it. If it instead flags `returning_from_injury: true` (or `managed: true`) with no
   explicit number, use `TOG_RETURN_DEFAULT` (0.75).
2. **Else** — use the player's recent-form TOG (`recent_tog` from `player_tog`), i.e. assume
   they play their normal recent minutes.
3. **Historical TOG** (the denominator) is always the player's last-40 EWMA `baseline_tog`.

Then:
```python
recent_tog, baseline_tog = player_tog(log, player)
projected_tog = lineup_tog.get(player)            # override or 0.75 if flagged returning
if projected_tog is None:
    projected_tog = recent_tog
tog_mult = tog_multiplier(projected_tog, baseline_tog)   # e.g. 0.75 / 0.90 ≈ 0.83 → -17%
context_mult = tog_mult * cba_mult                       # cba_mult already computed today
mean = expected_stat_mean(baseline_mean, share, team_total_mean,
                          matchup_mult=matchup_mult, context_mult=context_mult)
```
Net effect: a player who normally plays ~90% but is expected ~75% coming off injury has all
their counting stats (disposals / marks / tackles) scaled by ~0.83 for that match, which
flows straight into the prop probabilities and therefore the SGM ladder. **Goals** are
already team-goal-constrained via the multinomial split — let the TOG-reduced disposal/usage
share feed that as it does now; don't double-apply TOG to goals.

### C2. Extend the lineup JSON plug-point (Section 1's `--lineup`)
**Current state:** `afl_bot/data/lineups.py` `load_lineup(path)` returns `dict[str, set[str]]`
from a plain `{team: [names]}` file; `cli.py` uses it only as a confirmed-set gate
(`lineup.get(team)`, the `--lineup` flag → `lineup_path`). It carries no minutes info.

**Extend it backward-compatibly.** Accept EITHER the existing list-of-names form OR a richer
per-player object form, and return the confirmed set **plus** a `{player: expected_tog}` map:

```json
{
  "Carlton": [
    "Patrick Cripps",
    {"player": "Sam Walsh", "returning_from_injury": true},
    {"player": "Adam Cerra", "expected_tog": 0.70}
  ]
}
```
- A bare string → confirmed, no TOG override (current behaviour, unchanged).
- `{"player": ...}` object → confirmed, and:
  - `expected_tog` (0–1) → used directly as `projected_tog`.
  - `returning_from_injury: true` or `managed: true` with no number → `TOG_RETURN_DEFAULT`.
- Keep `load_lineup` returning the `set[str]` it does today (so every existing call site and
  test still works), and add a companion `load_lineup_tog(path) -> dict[str, float]`
  (or return a `(confirmed, tog)` tuple and update the two call sites). Don't break the plain
  `{team: [names]}` format — old lineup files must still load.
- Thread the `{player: expected_tog}` map into the pricing loop in `cli.py round_report()`
  (the per-player prop pricing around the `f"{player_name} {line}+ {stat}"` construction,
  ~line 769–785).

> `TOG_MULT_BOUNDS = (0.70, 1.15)` already in config — a 0.75/0.90 ≈ 0.83 multiplier sits
> inside it, and even a severe cut floors at 0.70 (never zeroes a player out).

### C3. Report note
In the round-report header, add one line stating the minutes basis, e.g.:
`"Projections = last 40 games (EWMA) × expected TOG. Players flagged returning from injury
capped at 75% TOG; N players had a lineup TOG override this run."`
Count and surface how many players used an override vs recent-form TOG, so it's visible.

---

## PART D — Thin history (< 40 games)
No new code path needed — `.tail(40)` on a 12-game player just returns 12 rows. Confirm:
- `player_rate_profile` returns `n_games = len(rows)` (it does) so the shrinkage in
  `priors.shrink(raw, n_games, prior, strength)` already pulls short histories toward the
  role/position prior. Nobody is dropped.
- Add an assertion/test that a 5-game player still gets a finite projection (shrunk), not NaN.

---

## PART E — Chronology guardrails (don't let "last 40" silently pick the wrong 40)
"Last 40" is only correct if games sort in true chronological order.
- The sort key is `["year","round","unixtime"]`. Fryzigg's `unixtime` was the 1970-bug
  source (already fixed under §7.2 round normalisation) — **keep `year, round` as the primary
  sort and `unixtime` only as a tiebreak**, so a single bad timestamp can't mis-order a
  player's last 40.
- After merging Fryzigg (history) + DFS Australia (current season), de-dupe on the canonical
  player-ID + (year, round) before windowing, or a player who appears in both sources for an
  overlapping round double-counts inside the 40. (Canonical IDs are §7.1 — already done; just
  make sure the de-dupe runs before `.tail(window)`.)

---

## PART F — Data sources: use Fable's EXACT sites, do not substitute
All player-form data must come from the sources already wired/identified. **Do not swap in a
different provider.** If one is unreachable, fall back only to another on this list and say so.

| Purpose | Exact source | URL |
|---|---|---|
| Player game history 2012–2025 (the 40-game window backbone) | **Fryzigg API** via fitzRoy | `http://www.fryziggafl.net/static/fryziggafl.rds` · docs `https://jimmyday12.github.io/fitzRoy/articles/using-fryzigg-stats.html` |
| Current-season per-player box scores (TOG%, CBA — drives Part C) | **DFS Australia** | `https://dfsaustralia.com/afl-stats-download/` · `https://dfsaustralia.com/downloads/` |
| Fixtures / games / tips / ladder | **Squiggle API** | `https://api.squiggle.com.au/` (note: **no** `q=lineup` — that query isn't supported) |
| Confirmed lineups (who's playing → Part C) | **AFL API** `aflapi.afl.com.au` via fitzRoy `fetch_lineup` | port from `https://github.com/jimmyday12/fitzRoy/blob/main/R/fetch_lineup.R` |
| Lineup fallback | **Footywire team selections** | `https://www.footywire.com/afl/footy/afl_team_selections` |
| Weather / wet flags | **Open-Meteo** | forecast `https://api.open-meteo.com/v1/forecast` · hourly archive `https://archive-api.open-meteo.com/v1/archive` |
| Live odds + player props | **The Odds API** (`aussierules_afl`) | `https://the-odds-api.com` |
| Historical closing odds (h2h/totals, CLV) | **aussportsbetting.com** | `https://www.aussportsbetting.com/data/` |
| Ratings benchmark (sanity-check only) | **Wheelo Ratings** | `https://www.wheeloratings.com` |
| Licensed event-level data (optional, not required) | **Champion Data AFL API** | `https://docs.api.afl.championdata.com/` |

Keep snapshotting the DFS pull to a dated parquet after every round (§7.3) — DFS is
current-season-only, so CBA/TOG history can't be re-downloaded later, and the 40-game window
needs that TOG history to survive.

---

## PART G — Tests (`tests/`)
- `player_rate_profile` defaults to a 40-game window: give a player 60 synthetic games with a
  level shift at game 21 from the end; assert the projection reflects ~last 40, not all 60.
- `estimate_dispersion` / `estimate_dispersion_hierarchical` use only the last 40 rows
  per player (construct a player whose early-career variance differs from recent).
- `_player_means_and_shares` ignores games older than the window.
- **TOG:** `tog_multiplier(0.75, 0.90)` ≈ 0.83 within `TOG_MULT_BOUNDS`; a player with
  `expected_tog: 0.75` in the lineup JSON gets lower projected disposals than the same player
  with no override; `returning_from_injury: true` with no number applies `TOG_RETURN_DEFAULT`.
- **Thin history:** a 5-game player returns a finite, shrunk projection (not NaN) and isn't
  dropped from the report.
- **Chronology:** a single bad `unixtime` doesn't reorder a player's last 40 (sort falls back
  to year/round).
- Update any existing test that assumed the 20-game default or all-history means.

## PART H — Verify
1. `pytest -q` (all green) and `ruff check`.
2. Regenerate the current round with a lineup that flags one returning player at 75%:
   `python -m afl_bot.cli round-report --year 2026 --round 14 --lineup lineups_r14.json --live-odds`
3. Open `reports/2026_r14_report.md` and confirm:
   - projections moved vs the all-history version (spot-check a known older player whose
     recent form differs from career),
   - the flagged returning player's disposal/mark/tackle lines are visibly lower,
   - the header note states the TOG basis and the override count,
   - real player names only (synthetic guard still holds), SGM ladders still render.
4. `grade-round` on 2–3 completed rounds — log loss must **not get worse** than the current
   all-history baseline (house rule: walk-forward score never regresses). If it does, the
   window or TOG bounds need tuning, not the sim.

## Acceptance check (how Ben will judge it)
Run one command for the current round → every match shows real players projected off their
**last 40 games**, scaled by expected minutes, with returning-from-injury players sensibly
capped (~75%), and the SGM ladders priced off those adjusted numbers. Grading didn't regress.

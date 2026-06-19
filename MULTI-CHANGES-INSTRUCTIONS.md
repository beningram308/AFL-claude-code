# Instructions: live odds + 3-leg multi ladder (1.75 → 5.0, top = value pick)

Goal of the round report's multi section:
  * Pull **live** market odds (not a hand-typed JSON).
  * Output a **ladder of minimum-3-leg multis** spanning combined odds ~1.75 up to ~5.0.
  * The top (~5.0) rung is a **VALUE PICK** — chosen by positive edge (EV), not by
    probability.
Do NOT touch modelling/sim logic. Only the odds intake + multi assembly/selection
+ the call sites + tests.

================================================================================
## PART A — Live odds access  (do this first; the value pick depends on it)
================================================================================

### Where odds enter today (and why it isn't live)
* `afl_bot/cli.py` `round_report()`:  `odds_book = json.loads(Path(odds_path).read_text())`
  — a STATIC local file passed via `--odds`. One leg-name → odds dict. Not live.
* `afl_bot/data/odds.py` `fetch_historical_odds()` — downloads aussportsbetting.com
  HISTORICAL h2h + totals only (for backtesting). No player props, not live.
* So: nothing currently fetches live prices, and nothing has live PROP prices.

### What to build
Add a new module `afl_bot/data/live_odds.py` exposing:

    def fetch_live_odds(round_no: int | None = None) -> dict[str, float]:
        """Return {leg_name: decimal_odds} in EXACTLY the same key format the
        report already uses, so it drops straight into `odds_book`."""

Key-format requirement (must match how legs are named in cli.py so they join):
    * H2H:    "Brisbane Lions to win"            (see sample_odds.json)
    * Totals: "Total points 170.5+"
    * Props:  "Harry Sheezel 15+ disposals"      (built at cli.py ~line 764:
              name = f"{player_name} {line}+ {stat}")
  If the live source's team/player strings differ, normalise them through
  `afl_bot.data.teams.normalize_team_name` and the same player-id path the report
  uses, BEFORE returning — otherwise legs won't match and edges won't compute.

### Source: The Odds API — AND IT DOES HAVE AFL PLAYER PROPS (corrected)
Earlier note was WRONG. The Odds API (https://the-odds-api.com), sport key
`aussierules_afl`, DOES provide AFL player props from AU books (Sportsbet,
Ladbrokes, TAB, Pointsbet, Betr). Props are the whole point of this — wire them.
Read the API key from env `ODDS_API_KEY`, never hard-code. Cache responses ~2 min.

** Player props are on the per-EVENT endpoint, not the bulk endpoint, and are a
PAID-tier / additional-markets feature (the free tier is featured markets only —
h2h/totals). Confirm the user's plan includes additional markets, else props 402/
return empty and you fall back to the --odds file. **

Two-step fetch:
  1. List events to get eventIds:
       GET https://api.the-odds-api.com/v4/sports/aussierules_afl/events?apiKey=KEY
     (also fetch h2h/totals in bulk here: .../odds?regions=au&markets=h2h,totals
      &oddsFormat=decimal — these ARE on every plan.)
  2. Per event, fetch prop markets (one request per event — costs credits =
     #markets × #regions per event, so request only the markets you use):
       GET https://api.the-odds-api.com/v4/sports/aussierules_afl/events/{eventId}/odds
           ?regions=au&oddsFormat=decimal&apiKey=KEY
           &markets=player_disposals_over,player_goals_scored_over,player_marks_over,player_tackles_over

AFL prop market keys this bot needs (use the "_over" milestone variants — they
list X+ lines as separate priced outcomes, which match the bot's "15+ / 20+ / 4+"
leg format directly):
    player_disposals_over        -> "<player> N+ disposals"
    player_goals_scored_over     -> "<player> N+ goals"
    player_marks_over            -> "<player> N+ marks"
    player_tackles_over          -> "<player> N+ tackles"
  (also available if the bot ever prices them: player_kicks_over,
   player_handballs_over, player_clearances_over, player_afl_fantasy_points.)
  NOTE: `player_disposals` (no _over) is an Over/Under market with a .5 line and
  two outcomes; if you use that instead, "N+" = the OVER outcome at point N-0.5.

Parsing each event response:
    data["bookmakers"][i]["markets"][j]["outcomes"][k] has:
        description = player name      (normalise to the bot's player names)
        name        = "Over" / player  (or the milestone label)
        point       = the line (e.g. 15, 20, 4, or 14.5 for O/U markets)
        price       = decimal odds
  Build the leg name as "<normalised player> <int(point or point+0.5)>+ <stat>"
  EXACTLY matching cli.py's `f"{player_name} {line}+ {stat}"`. Take the BEST
  (highest) price across the AU books for that exact player+line, or pin one book
  — your call, but be consistent. Drop any leg whose name can't be matched to a
  bot player (log how many were dropped).

Fallbacks / merge (do this so the run never silently loses props):
    live_h2h_totals  ⊕  live_props  ⊕  --odds file (file overrides live).
  If props come back empty (free plan / off-season / 402), fall back to the
  --odds JSON for props and SAY which source supplied them in the report note.

### Wiring into the report
In `cli.py` `round_report()`, replace the single static load with a merge:

    live = fetch_live_odds(round_no) if use_live else {}
    manual = json.loads(Path(odds_path).read_text()) if odds_path else {}
    odds_book = {**live, **manual}      # manual file overrides live, for hand-fixes

Add a CLI flag next to `--odds` (around cli.py:962):
    rep_p.add_argument("--live-odds", action="store_true", dest="use_live",
                       help="Fetch live market odds (The Odds API; needs ODDS_API_KEY).")
and thread `use_live` through `round_report(...)` and the `main()` dispatch (cli.py:989).

### Report note — state the actual prop source per run
Props ARE fetched live when the plan supports them. The note must reflect what
actually happened on THIS run, not a blanket disclaimer:
  * props fetched live  -> "Player-prop odds: live from The Odds API (AU books)."
  * props from file     -> "Player-prop odds: from --odds file (live props
                            unavailable on this plan / no key)."
  * no prop odds at all  -> "No prop odds this run; prop rungs show '-' and are
                            not flagged VALUE."
A rung only gets a Book/Edge and the VALUE PICK label when its legs actually have
book odds (live or file). Count and report how many prop legs got a live price vs
how many were dropped for name-matching, so a low hit-rate is visible.

================================================================================
## PART B — Multi ladder 1.75 → 5.0, minimum 3 legs, top rung = VALUE PICK
================================================================================
Replaces the earlier "single 1.75 floor" idea. We want a SPREAD of multis, not just
the safest ones.

### B1. Same-game multis — `afl_bot/build/report.py` `search_match_sgms(...)`
Current: `range(2, max_legs+1)`, ranks all combos by joint_prob desc, top_n=5, no
odds bands. That's why you only see low-odds 2-leggers.

Change to a banded selection. New signature (suggested):

    def search_match_sgms(legs, *, min_legs=3, max_legs=3, odds_book=None,
                          odds_bands=((1.75, 2.50), (2.50, 3.50), (3.50, 5.50)),
                          per_band=1, min_joint_prob=0.05):

Logic:
  1. Generate combos for r in range(min_legs, max_legs+1) (default = exactly 3).
     Keep the `_no_conflicts` check.
  2. For each combo compute `joint` (joint_prob_from_masks), `naive`, `corr_gain`,
     `fair = fair_odds(joint)`, and — when all legs' book odds are present —
     `book_odds = combined_odds(legs_list)`.
     ** EDGE MUST BE SHRUNK (do not use raw joint*book_odds-1). ** Per-leg model
     overestimates compound multiplicatively across legs, which is why a raw 3-leg
     prop edge can read an absurd +30%+. Pull the joint toward the book's implied
     multi probability before taking the edge, using the existing knob
     `MULTI_MARKET_SHRINK` (=0.25) and `market_anchored_prob` from
     `afl_bot.pricing.edge`:
         book_implied = 1.0 / book_odds
         shrunk_joint = market_anchored_prob(joint, book_odds, MULTI_MARKET_SHRINK)
                        # = (1-w)*joint + w*(1/book_odds)
         edge = shrunk_joint * book_odds - 1.0
     (More granular alternative: shrink EACH leg's prob with market_anchored_prob
     against that leg's own book odds, re-combine, then edge — but the joint-level
     shrink above is sufficient and simpler.) Store BOTH the raw and shrunk edge if
     you want, but the value pick and the report's Edge column use the SHRUNK one.
  2b. EDGE SANITY CAP. After shrinking, if `edge > 0.15` (15%), do NOT silently
     present it as the best bet — a 3-leg prop edge that large almost always means
     the model disagrees with the book because the model is wrong, not the book.
     Either (a) skip combos with shrunk edge > a `max_plausible_edge` (default
     0.15) from value-pick selection, or (b) keep them but flag with "⚠ check model"
     instead of "VALUE PICK". Recommend (a): pick the highest edge that is positive
     AND ≤ 0.15.
  3. Decide the odds used for banding: prefer `book_odds` when present, else `fair`.
  4. Bucket combos into `odds_bands`. From each band emit up to `per_band` multis:
       - lower/mid bands: pick the HIGHEST joint_prob in the band (safest at that price).
       - top band (the ~3.5–5.5 "value" rung): pick the highest SHRUNK edge that is
         > 0 and ≤ max_plausible_edge (needs book odds). Tag `{"value_pick": True}`.
         If no book odds in that band, fall back to highest joint_prob and tag
         `value_pick=False` (can't claim value without a market price).
  5. Drop the old hard `min_joint_prob=0.20` floor down to ~0.05 so longer-odds
     (lower-prob) combos survive to fill the 3.5–5.5 band — the band caps now do
     the filtering, not a blanket probability floor.
  6. Return the selected multis ordered safest→longest (band order). Carry an
     `odds` field and the `value_pick` flag so the renderer can label them.

### B2. Renderer — `report.py` `render_markdown(...)`
In the "Same-game multis" block, label the rungs, e.g. a "Pick" column or a marker:
  * rows where `value_pick` is True → append "  **← VALUE PICK**" or a ⭐.
  * Make sure the table still shows Joint prob / Fair odds / Book / Edge as today;
    the Edge column is what justifies the value pick, so keep it visible.

### B3. Cross-game multis — `afl_bot/build/multi.py`
* `build_anchor_multis(...)`: bump default `legs_per_multi=2` → `3`. These are the
  "safe/very likely" builds — their natural combined odds are ~1.2–1.6, BELOW 1.75,
  so they populate the bottom of the ladder. Do NOT force 1.75 here or it returns
  empty (3 anchors can't reach it). Leave them as the low-odds anchor rung, OR drop
  this section if the same-game ladder already covers 1.75.
* For the higher rungs / value pick across games, use a 3-leg version of
  `build_promo_multi` (already 2 anchors + 1 value) OR add a small helper that
  searches 3-leg cross-game combos and filters/banks them into the same
  `odds_bands`, selecting the top band by `combined_edge` (already a property on
  `MultiResult`: `combined_fair_prob*combined_market_odds - 1`). The value pick =
  highest positive `combined_edge` whose `combined_market_odds` lands ~3.5–5.5.

### Call sites (cli.py)
* ~line 781: `search_match_sgms(match_legs, odds_book=odds_book)` — relies on new
  defaults, or pass `odds_bands`/`per_band` explicitly.
* ~line 792: `build_anchor_multis(odds_legs, legs_per_multi=3, joint_prob_fn=...)`.

### B4. EVERY game must get a multi ladder  (required)
Right now the renderer prints "_No qualifying combinations._" for a match when the
band logic finds nothing, and a game can come up empty if all its combos miss the
bands (e.g. every leg is very high prob → all combos sit below 1.75) or it has few
priced legs. That's not acceptable — every game in the round should show a ladder.
Make `search_match_sgms` guarantee output per game:

  1. After banding, if a band is EMPTY, fill that rung from the unbanded combo pool
     by taking the combo whose banding-odds are CLOSEST to that band's midpoint
     (don't leave a blank rung). So each game shows up to `len(odds_bands)` rungs.
  2. If a game still yields nothing (because it has < `min_legs` usable legs after
     `_no_conflicts`), the report should say WHY for that game, e.g.
     "_Only N priced legs for this match — need ≥3 for a multi._", not a bare
     "no qualifying combinations". Surface the count so the user can see it's a
     data gap (missing lineups/odds), not a silent drop.
  3. Make sure enough legs reach the builder so 3-leg combos exist:
       - the leg gate in cli.py (~line 763) is `0.05 < prob < 0.97`; that's fine,
         but confirm each match contributes enough players. If `PLAYERS_PER_TEAM_*`
         sampling is low, a thin match may have too few legs — raise the per-team
         player pool if needed so every match has ≥3 non-conflicting legs.
  4. Keep `min_joint_prob` low (~0.05) so longer-odds combos aren't filtered out
     before banding — the bands + the per-game fill do the shaping.

Goal acceptance: open the regenerated report and confirm EVERY "Same-game multi
ladder" block has at least one 3-leg multi (and ideally the full 1.75/2.5/3.5
ladder), with no game showing a bare "no qualifying combinations".

================================================================================
## PART C — Tests
================================================================================
* `tests/test_report.py` lines ~41/~59 call `search_match_sgms(..., max_legs=2)`.
  With `min_legs=3` default that yields no combos — update to pass ≥3 legs and
  `min_legs=2` (to keep the old 2-leg assertions) or rewrite for the band output.
  Fixtures use ~0.9-prob legs (fair odds ~1.05) so they fall in NO band ≥1.75 —
  either add lower-prob legs or pass an `odds_bands` starting below 1.75 in tests.
* `tests/test_build_multi.py` lines ~42/~53 pass `legs_per_multi=2` explicitly so
  they still run; add a 3-leg case.
* New tests to add:
  - every emitted same-game multi has `len(legs) >= 3` and its banding odds ≥ 1.75.
  - exactly one multi per band (per_band=1), bands non-overlapping.
  - the `value_pick` multi has `edge > 0` when book odds are present.
  - `fetch_live_odds` returns the correct key format (mock the HTTP call; don't hit
    the network in tests — follow how existing data tests stub requests).

================================================================================
## PART D — Verify
================================================================================
1. `pytest -q`
2. Live fetch smoke test:  `ODDS_API_KEY=... python -c "from afl_bot.data.live_odds import fetch_live_odds; print(fetch_live_odds())"`
   — confirm keys look like "Brisbane Lions to win" / "Total points 170.5+".
3. Regenerate:  `python -m afl_bot.cli round-report --year 2026 --round 14 --live-odds`
   (add `--odds prop_odds.json` to top up prop prices the live feed lacks).
   Confirm each match shows a ladder of 3-leg multis from ~1.75 up to ~5.0, and the
   ~5.0 rung is labelled VALUE PICK with a positive edge.

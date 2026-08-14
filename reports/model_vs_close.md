No seasons excluded, no null-closing rows dropped, nothing fit on the evaluation period (see Scope/Walk-forward-discipline below for exactly what that means and what IS disclosed as a genuine limitation).

# Does the model beat closing prices? (H2H and totals, no SGM)

Population: **3467** games, `data_cache/aussportsbetting_afl_odds.parquet`, 2009-2026 closing lines. Anchored to exactly this file's own row count -- enriching with Squiggle round/venue/unixtime never adds or drops a row (asserted in code, not just claimed here).

- Null closing H2H price: **681** rows (kept, not dropped -- these are 2009-2012 seasons with no odds coverage at all, plus a handful elsewhere).
- Null closing totals price: **889** rows (kept, not dropped -- 2009-2013 mostly, totals coverage starts properly in 2014-15).
- **1** game(s) didn't auto-match a Squiggle round/venue (home/away team labels are swapped between the two sources for the 2015 Grand Final, Hawthorn v West Coast, a neutral-ish-venue game): [{'date': '2015-10-03', 'hteam': 'West Coast', 'ateam': 'Hawthorn', 'year': 2015}]. Kept in the population with flat home-ground advantage and a date-derived sort key for that one game -- not excluded.

## Walk-forward discipline

- **Elo (H2H):** ratings update sequentially, game by game, across the full history -- a game's prediction uses only strictly earlier games, including earlier games in the SAME season (finer-grained than a season freeze, not coarser).
- **Per-venue home-ground advantage:** a batch fit (`fit_team_hga`), refit ONCE per season using only strictly prior seasons, held fixed in-season.
- **Totals model** (team off/def EWMA, venue scoring factors): same treatment -- refit once per season on prior seasons only, held fixed in-season.
- **Margin sigma / total-points sigma** (for the Normal-approximation probability): computed walk-forward per season (expanding-window std of prior seasons' actual margins/totals) rather than the fixed config constants.
- **NOT re-fit per season (disclosed):** Elo hyperparameters -- `{'points_per_400': 116.0}` plus K/home_advantage/season_carryover/margin_cap from `afl_bot/config.py` -- are the live, previously-tuned values, used as-is for every season. `points_per_400` specifically was selected on 2022-24 data and validated on 2025-26 (`data_cache/elo_params.json`'s own metadata), so reusing it for 2013-2021 predictions here means an early season's probability is shaped by a scale parameter partly chosen using later seasons' fit quality -- not by any individual future game's result, but a real hyperparameter look-ahead, disclosed rather than silently reused.
- **De-vig method:** proportional (`afl_bot.backtest.walkforward.devig_h2h_probs`, reused verbatim for both markets) -- does not correct for favourite-longshot bias, same limitation the rest of the codebase already accepts.
- **Totals CLV is NOT computable** from this dataset -- it has the opening total LINE (a number) but only CLOSING over/under PRICES, no opening over/under prices. Reporting closing-vs-opening LINE movement in the model's direction instead of a price-based CLV, flagged explicitly in that section rather than fabricated.
- **2020/2021 totals distortion (found during this analysis, not assumed going in):** 2020's COVID-shortened quarters (16 minutes most of the season) genuinely lowered scoring; the walk-forward totals model (EWMA halflife=6 games) picked that up correctly at the time, but had no way to know 2021 reverted to full-length quarters -- there is no quarter-length feature anywhere in `team_scoring_profiles`/`expected_total`. This produces a large, genuine (not a bug) walk-forward miss concentrated in 2021 and late 2020: the model badly under-projects totals, and its "edge" in that window is a blind spot, not skill -- see whether 2020/2021's ROI in the totals section is positive or negative before trusting any of the other seasons' numbers more than this one.
- **Costs:** gross flat-stake ROI, and net of 5% Betfair commission on WINNING bets only (the standard single-bet simplification -- real Betfair commission nets across a whole market, which independent flat bets can't reproduce exactly).

## H2H

Population: **2786** games in the odds file's population; **2786** have a usable closing price (the rest are the null-closing-price rows, kept in the population, never dropped -- see Scope above). Of those, **2786** get a walk-forward model probability (the difference, if any, is the burn-in seasons with no prior data for the walk-forward sigma -- see Scope).

### Overall

| n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|
| 2786 | 0.473 | 0.392 | -6.64% | -9.32% | -1.97pp (n=2786) |

### By season

| season | n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|---|
| 2013 | 207 | 0.432 | 0.335 | -14.95% | -17.63% | -1.12pp (n=207) |
| 2014 | 207 | 0.451 | 0.362 | +15.73% | +12.04% | -0.97pp (n=207) |
| 2015 | 206 | 0.473 | 0.382 | -14.31% | -16.77% | -1.32pp (n=206) |
| 2016 | 207 | 0.435 | 0.350 | -10.47% | -13.24% | -0.90pp (n=207) |
| 2017 | 207 | 0.498 | 0.423 | +3.32% | +0.35% | -1.37pp (n=207) |
| 2018 | 207 | 0.467 | 0.389 | -15.89% | -18.24% | -2.80pp (n=207) |
| 2019 | 207 | 0.493 | 0.424 | +2.71% | -0.20% | -2.89pp (n=207) |
| 2020 | 162 ⚠ | 0.527 | 0.424 | -12.07% | -14.33% | -1.16pp (n=162) |
| 2021 | 207 | 0.483 | 0.401 | -2.06% | -4.95% | -2.43pp (n=207) |
| 2022 | 207 | 0.471 | 0.401 | -12.13% | -14.54% | -2.03pp (n=207) |
| 2023 | 216 | 0.477 | 0.404 | -8.54% | -11.12% | -2.37pp (n=216) |
| 2024 | 216 | 0.498 | 0.428 | -0.87% | -3.51% | -2.26pp (n=216) |
| 2025 | 216 | 0.458 | 0.385 | -16.63% | -18.95% | -2.78pp (n=216) |
| 2026 | 114 ⚠ | 0.467 | 0.384 | -8.58% | -11.00% | -3.71pp (n=114) |

### By model-edge bucket

| edge bucket | n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|---|
| 0-2% | 482 | 0.488 | 0.478 | -12.66% | -14.68% | +0.22pp (n=482) |
| 2-5% | 626 | 0.463 | 0.428 | -10.96% | -13.36% | -1.28pp (n=626) |
| 5-10% | 816 | 0.454 | 0.381 | -13.66% | -16.12% | -2.11pp (n=816) |
| 10%+ | 862 | 0.490 | 0.329 | +6.52% | +3.05% | -3.57pp (n=862) |

### By season × edge bucket

| season | edge bucket | n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|---|---|
| 2013 | 0-2% | 34 ⚠ | 0.487 | 0.478 | -33.83% | -35.22% | +1.09pp (n=34) |
| 2013 | 2-5% | 32 ⚠ | 0.400 | 0.366 | -58.81% | -59.62% | +0.37pp (n=32) |
| 2013 | 5-10% | 57 ⚠ | 0.375 | 0.300 | -14.85% | -17.62% | -1.40pp (n=57) |
| 2013 | 10%+ | 84 ⚠ | 0.460 | 0.290 | +9.34% | +5.48% | -2.39pp (n=84) |
| 2014 | 0-2% | 32 ⚠ | 0.472 | 0.460 | -24.49% | -25.92% | -0.45pp (n=32) |
| 2014 | 2-5% | 41 ⚠ | 0.512 | 0.479 | +41.79% | +37.63% | -0.30pp (n=41) |
| 2014 | 5-10% | 59 ⚠ | 0.384 | 0.310 | +13.39% | +9.50% | -0.71pp (n=59) |
| 2014 | 10%+ | 75 ⚠ | 0.462 | 0.296 | +20.48% | +16.26% | -1.75pp (n=75) |
| 2015 | 0-2% | 31 ⚠ | 0.514 | 0.503 | -41.97% | -42.94% | +0.11pp (n=31) |
| 2015 | 2-5% | 53 ⚠ | 0.471 | 0.437 | +18.93% | +15.34% | -1.05pp (n=53) |
| 2015 | 5-10% | 51 ⚠ | 0.455 | 0.379 | -12.52% | -14.93% | -0.63pp (n=51) |
| 2015 | 10%+ | 71 ⚠ | 0.471 | 0.292 | -28.33% | -30.64% | -2.65pp (n=71) |
| 2016 | 0-2% | 33 ⚠ | 0.499 | 0.488 | -19.99% | -21.57% | -0.91pp (n=33) |
| 2016 | 2-5% | 47 ⚠ | 0.415 | 0.379 | -6.35% | -9.22% | -0.67pp (n=47) |
| 2016 | 5-10% | 56 ⚠ | 0.381 | 0.305 | -32.13% | -34.28% | -0.46pp (n=56) |
| 2016 | 10%+ | 71 ⚠ | 0.461 | 0.301 | +8.30% | +4.58% | -1.41pp (n=71) |
| 2017 | 0-2% | 50 ⚠ | 0.480 | 0.469 | +11.07% | +8.11% | -0.60pp (n=50) |
| 2017 | 2-5% | 40 ⚠ | 0.490 | 0.455 | -7.62% | -10.12% | -1.35pp (n=40) |
| 2017 | 5-10% | 58 ⚠ | 0.488 | 0.412 | -11.37% | -13.82% | -1.30pp (n=58) |
| 2017 | 10%+ | 59 ⚠ | 0.529 | 0.373 | +18.62% | +14.81% | -2.11pp (n=59) |
| 2018 | 0-2% | 39 ⚠ | 0.478 | 0.468 | -22.59% | -24.41% | -0.24pp (n=39) |
| 2018 | 2-5% | 43 ⚠ | 0.495 | 0.461 | -31.12% | -32.70% | -1.11pp (n=43) |
| 2018 | 5-10% | 62 ⚠ | 0.443 | 0.370 | -19.06% | -21.25% | -4.47pp (n=62) |
| 2018 | 10%+ | 63 ⚠ | 0.463 | 0.310 | +1.76% | -1.58% | -3.91pp (n=63) |
| 2019 | 0-2% | 35 ⚠ | 0.504 | 0.493 | -7.03% | -9.11% | +0.44pp (n=35) |
| 2019 | 2-5% | 55 ⚠ | 0.442 | 0.408 | -17.60% | -19.90% | -2.91pp (n=55) |
| 2019 | 5-10% | 60 ⚠ | 0.529 | 0.460 | +3.95% | +1.00% | -3.25pp (n=60) |
| 2019 | 10%+ | 57 ⚠ | 0.498 | 0.361 | +27.00% | +23.02% | -4.52pp (n=57) |
| 2020 | 0-2% | 17 ⚠ | 0.482 | 0.471 | -18.41% | -20.14% | -1.30pp (n=17) |
| 2020 | 2-5% | 27 ⚠ | 0.526 | 0.491 | -31.93% | -33.48% | +0.30pp (n=27) |
| 2020 | 5-10% | 53 ⚠ | 0.519 | 0.447 | -1.00% | -3.31% | -0.73pp (n=53) |
| 2020 | 10%+ | 65 ⚠ | 0.546 | 0.365 | -11.18% | -13.86% | -2.07pp (n=65) |
| 2021 | 0-2% | 37 ⚠ | 0.537 | 0.527 | -13.11% | -14.75% | +0.14pp (n=37) |
| 2021 | 2-5% | 45 ⚠ | 0.473 | 0.438 | +3.73% | +0.55% | -1.64pp (n=45) |
| 2021 | 5-10% | 58 ⚠ | 0.448 | 0.373 | -19.93% | -22.47% | -1.76pp (n=58) |
| 2021 | 10%+ | 67 ⚠ | 0.492 | 0.329 | +15.61% | +11.92% | -4.96pp (n=67) |
| 2022 | 0-2% | 38 ⚠ | 0.456 | 0.445 | -20.68% | -22.41% | +1.00pp (n=38) |
| 2022 | 2-5% | 52 ⚠ | 0.455 | 0.421 | -22.98% | -25.10% | -2.43pp (n=52) |
| 2022 | 5-10% | 61 ⚠ | 0.452 | 0.379 | -20.92% | -23.07% | -1.22pp (n=61) |
| 2022 | 10%+ | 56 ⚠ | 0.517 | 0.375 | +13.32% | +9.89% | -4.59pp (n=56) |
| 2023 | 0-2% | 42 ⚠ | 0.478 | 0.469 | +24.51% | +21.02% | +2.73pp (n=42) |
| 2023 | 2-5% | 45 ⚠ | 0.454 | 0.420 | -16.17% | -18.25% | -0.42pp (n=45) |
| 2023 | 5-10% | 77 ⚠ | 0.471 | 0.398 | -8.63% | -11.38% | -3.82pp (n=77) |
| 2023 | 10%+ | 52 ⚠ | 0.504 | 0.348 | -28.50% | -30.54% | -6.01pp (n=52) |
| 2024 | 0-2% | 33 ⚠ | 0.475 | 0.466 | +1.40% | -1.40% | +0.67pp (n=33) |
| 2024 | 2-5% | 66 ⚠ | 0.485 | 0.450 | -14.02% | -16.04% | -1.38pp (n=66) |
| 2024 | 5-10% | 72 ⚠ | 0.520 | 0.446 | -7.51% | -9.77% | -3.11pp (n=72) |
| 2024 | 10%+ | 45 ⚠ | 0.500 | 0.341 | +27.36% | +23.32% | -4.34pp (n=45) |
| 2025 | 0-2% | 40 ⚠ | 0.505 | 0.494 | -26.77% | -28.43% | -0.06pp (n=40) |
| 2025 | 2-5% | 53 ⚠ | 0.423 | 0.388 | -13.89% | -16.12% | -1.81pp (n=53) |
| 2025 | 5-10% | 64 ⚠ | 0.438 | 0.370 | -41.45% | -42.74% | -2.53pp (n=64) |
| 2025 | 10%+ | 59 ⚠ | 0.479 | 0.325 | +14.69% | +10.74% | -5.76pp (n=59) |
| 2026 | 0-2% | 21 ⚠ | 0.461 | 0.451 | -2.57% | -4.82% | -0.94pp (n=21) |
| 2026 | 2-5% | 27 ⚠ | 0.452 | 0.418 | -18.74% | -20.58% | -1.91pp (n=27) |
| 2026 | 5-10% | 28 ⚠ | 0.421 | 0.349 | -25.21% | -27.35% | -3.43pp (n=28) |
| 2026 | 10%+ | 38 ⚠ | 0.514 | 0.349 | +7.58% | +4.44% | -6.72pp (n=38) |

⚠ = n < 200 for that row. CLV column shows n separately since it can differ from the row's own n (open-price coverage isn't identical to close-price coverage).

## Totals

Population: **2578** games in the odds file's population; **2578** have a usable closing price (the rest are the null-closing-price rows, kept in the population, never dropped -- see Scope above). Of those, **2578** get a walk-forward model probability (the difference, if any, is the burn-in seasons with no prior data for the walk-forward sigma -- see Scope).

### Overall

| n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|
| 2578 | 0.641 | 0.500 | -3.65% | -5.95% | — |

### By season

| season | n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|---|
| 2013 | 3 ⚠ | 0.653 | 0.512 | -36.37% | -37.88% | — |
| 2014 | 203 | 0.610 | 0.499 | -11.95% | -14.06% | — |
| 2015 | 206 | 0.572 | 0.498 | -11.76% | -13.89% | — |
| 2016 | 207 | 0.629 | 0.499 | -2.13% | -4.49% | — |
| 2017 | 207 | 0.618 | 0.499 | +9.90% | +7.26% | — |
| 2018 | 207 | 0.629 | 0.500 | -10.02% | -12.15% | — |
| 2019 | 207 | 0.604 | 0.500 | +3.30% | +0.84% | — |
| 2020 | 162 ⚠ | 0.880 | 0.500 | -10.40% | -12.53% | — |
| 2021 | 207 | 0.847 | 0.500 | -11.41% | -13.52% | — |
| 2022 | 207 | 0.618 | 0.500 | -9.65% | -11.80% | — |
| 2023 | 216 | 0.591 | 0.500 | +0.84% | -1.57% | — |
| 2024 | 216 | 0.568 | 0.500 | +10.44% | +7.81% | — |
| 2025 | 216 | 0.580 | 0.500 | -7.13% | -9.35% | — |
| 2026 | 114 ⚠ | 0.656 | 0.502 | +4.15% | +1.66% | — |

### By model-edge bucket

| edge bucket | n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|---|
| 0-2% | 282 | 0.511 | 0.501 | +1.73% | -0.69% | — |
| 2-5% | 376 | 0.534 | 0.500 | +1.32% | -1.10% | — |
| 5-10% | 555 | 0.574 | 0.500 | -8.76% | -10.94% | — |
| 10%+ | 1365 | 0.725 | 0.499 | -4.06% | -6.35% | — |

### By season × edge bucket

| season | edge bucket | n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |
|---|---|---|---|---|---|---|---|
| 2013 | 5-10% | 1 ⚠ | 0.569 | 0.511 | -100.00% | -100.00% | — |
| 2013 | 10%+ | 2 ⚠ | 0.694 | 0.512 | -4.55% | -6.82% | — |
| 2014 | 0-2% | 33 ⚠ | 0.507 | 0.498 | -7.36% | -9.57% | — |
| 2014 | 2-5% | 29 ⚠ | 0.535 | 0.501 | -15.36% | -17.35% | — |
| 2014 | 5-10% | 50 ⚠ | 0.573 | 0.499 | -18.68% | -20.64% | — |
| 2014 | 10%+ | 91 ⚠ | 0.691 | 0.498 | -8.83% | -11.03% | — |
| 2015 | 0-2% | 36 ⚠ | 0.506 | 0.495 | -8.49% | -10.70% | — |
| 2015 | 2-5% | 55 ⚠ | 0.535 | 0.501 | -8.97% | -11.16% | — |
| 2015 | 5-10% | 63 ⚠ | 0.571 | 0.498 | -7.78% | -10.01% | — |
| 2015 | 10%+ | 52 ⚠ | 0.655 | 0.496 | -21.78% | -23.67% | — |
| 2016 | 0-2% | 19 ⚠ | 0.510 | 0.501 | +12.42% | +9.69% | — |
| 2016 | 2-5% | 23 ⚠ | 0.532 | 0.500 | +0.72% | -1.71% | — |
| 2016 | 5-10% | 46 ⚠ | 0.572 | 0.500 | +8.73% | +6.12% | — |
| 2016 | 10%+ | 119 ⚠ | 0.689 | 0.498 | -9.21% | -11.39% | — |
| 2017 | 0-2% | 16 ⚠ | 0.517 | 0.504 | +7.87% | +5.29% | — |
| 2017 | 2-5% | 34 ⚠ | 0.536 | 0.499 | +14.05% | +11.29% | — |
| 2017 | 5-10% | 45 ⚠ | 0.576 | 0.500 | -1.49% | -3.86% | — |
| 2017 | 10%+ | 112 ⚠ | 0.675 | 0.498 | +13.51% | +10.78% | — |
| 2018 | 0-2% | 21 ⚠ | 0.508 | 0.499 | -9.52% | -11.67% | — |
| 2018 | 2-5% | 24 ⚠ | 0.537 | 0.500 | +3.25% | +0.80% | — |
| 2018 | 5-10% | 47 ⚠ | 0.577 | 0.500 | -27.23% | -28.96% | — |
| 2018 | 10%+ | 115 ⚠ | 0.691 | 0.500 | -5.84% | -8.07% | — |
| 2019 | 0-2% | 21 ⚠ | 0.507 | 0.500 | +27.24% | +24.21% | — |
| 2019 | 2-5% | 33 ⚠ | 0.535 | 0.500 | -13.30% | -15.37% | — |
| 2019 | 5-10% | 56 ⚠ | 0.572 | 0.500 | -21.66% | -23.52% | — |
| 2019 | 10%+ | 97 ⚠ | 0.667 | 0.499 | +18.19% | +15.37% | — |
| 2020 | 10%+ | 162 ⚠ | 0.880 | 0.500 | -10.40% | -12.53% | — |
| 2021 | 5-10% | 1 ⚠ | 0.597 | 0.500 | -100.00% | -100.00% | — |
| 2021 | 10%+ | 206 | 0.848 | 0.500 | -10.98% | -13.10% | — |
| 2022 | 0-2% | 15 ⚠ | 0.510 | 0.503 | -11.40% | -13.50% | — |
| 2022 | 2-5% | 27 ⚠ | 0.534 | 0.500 | +5.89% | +3.37% | — |
| 2022 | 5-10% | 44 ⚠ | 0.576 | 0.500 | -4.32% | -6.60% | — |
| 2022 | 10%+ | 121 ⚠ | 0.666 | 0.500 | -14.83% | -16.86% | — |
| 2023 | 0-2% | 36 ⚠ | 0.510 | 0.501 | -10.11% | -12.24% | — |
| 2023 | 2-5% | 41 ⚠ | 0.535 | 0.500 | +2.41% | -0.02% | — |
| 2023 | 5-10% | 50 ⚠ | 0.574 | 0.500 | -0.55% | -2.92% | — |
| 2023 | 10%+ | 89 ⚠ | 0.659 | 0.500 | +5.31% | +2.80% | — |
| 2024 | 0-2% | 36 ⚠ | 0.509 | 0.500 | +21.86% | +18.96% | — |
| 2024 | 2-5% | 57 ⚠ | 0.534 | 0.500 | +17.08% | +14.30% | — |
| 2024 | 5-10% | 70 ⚠ | 0.573 | 0.500 | +6.27% | +3.75% | — |
| 2024 | 10%+ | 53 ⚠ | 0.640 | 0.500 | +1.02% | -1.39% | — |
| 2025 | 0-2% | 42 ⚠ | 0.512 | 0.501 | -4.50% | -6.78% | — |
| 2025 | 2-5% | 44 ⚠ | 0.531 | 0.499 | +8.75% | +6.16% | — |
| 2025 | 5-10% | 59 ⚠ | 0.574 | 0.500 | -22.31% | -24.16% | — |
| 2025 | 10%+ | 71 ⚠ | 0.654 | 0.500 | -5.93% | -8.17% | — |
| 2026 | 0-2% | 7 ⚠ | 0.562 | 0.549 | +34.29% | +31.14% | — |
| 2026 | 2-5% | 9 ⚠ | 0.531 | 0.500 | -35.11% | -36.69% | — |
| 2026 | 5-10% | 23 ⚠ | 0.575 | 0.496 | +0.70% | -1.73% | — |
| 2026 | 10%+ | 75 ⚠ | 0.705 | 0.500 | +7.11% | +4.55% | — |

⚠ = n < 200 for that row. CLV column shows n separately since it can differ from the row's own n (open-price coverage isn't identical to close-price coverage).

## Verdict

**NO** -- H2H net ROI -9.32% on n=2786; totals net ROI -5.95% on n=2578 (2020/2021 quarter-length distortion included, not excluded); neither clears both n≥200 and positive net-of-commission ROI in a way the other doesn't immediately contradict.

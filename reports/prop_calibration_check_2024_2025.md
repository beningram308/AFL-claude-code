# Prop Calibration Check: Model vs Actual Hit Rate

**Eval years:** [2024, 2025]
**Total legs graded:** 177,480
**Book comparison source:** 2026_r17_multis.json

Calibrators are fitted on data strictly prior to each eval year (PROP_CALIBRATION_LOOKBACK=3), so no leakage.

> **Note on book comparison**: `book_implied` is the average of 1/leg_book_odds for legs appearing in sportsbet SGM multis. These are only the specific high-probability lines the book prices for SGMs - they are NOT comparable to the overall calibration mean (which spans all evaluated lines, including long-shot lines with low hit rates). The `ev_diag_gap` column is the correct comparable: it shows model calibrated prob vs book implied at the SAME specific priced legs (from the ev-diagnostic). Use the per-band table for a fair comparison.

## Per-Market Summary

| Market | n | Cal prob | Hit rate | Cal gap | book_minus_model (priced SGM legs) | Class |
|--------|---|----------|----------|---------|-----------------------------------|-------|
| disposals   | 49,300 | 0.204 | 0.190 | +0.013 |                              +0.078 | WELL CALIBRATED |
| goals       | 29,580 | 0.172 | 0.171 | +0.000 |                              +0.044 | WELL CALIBRATED |
| marks       | 49,300 | 0.298 | 0.289 | +0.009 |                              +0.114 | WELL CALIBRATED |
| tackles     | 49,300 | 0.211 | 0.209 | +0.002 |                              +0.096 | WELL CALIBRATED |

*Cal gap: positive = model over-predicts vs actual; negative = model under-predicts.*
*book_minus_model: positive = book is ABOVE model (book's structural margin); negative = model above book.*

## Per-Market x Probability Band (with book implied per band)

The three-way comparison works correctly within each probability band, because the book's priced legs and the walk-forward sample are in the same prob range.

### Disposals

| Band | n | Cal prob | Hit rate | Gap | Book implied (SGM) | Book vs actual |
|------|---|----------|----------|-----|--------------------|----------------|
| 30%-50% | 4,231 | 0.392 | 0.360 | +0.031 |              0.472 |         +0.112 |
| 50%-70% | 3,342 | 0.584 | 0.563 | +0.021 |              0.684 |         +0.121 |
| 70%-90% | 2,973 | 0.795 | 0.779 | +0.016 |              0.859 |         +0.080 |

### Goals

| Band | n | Cal prob | Hit rate | Gap | Book implied (SGM) | Book vs actual |
|------|---|----------|----------|-----|--------------------|----------------|
| 30%-50% | 3,439 | 0.393 | 0.391 | +0.002 |              0.411 |         +0.020 |
| 50%-70% | 1,460 | 0.586 | 0.592 | -0.006 |              0.700 |         +0.108 |
| 70%-90% | 1,360 | 0.782 | 0.754 | +0.029 |              0.749 |         -0.005 |

### Marks

| Band | n | Cal prob | Hit rate | Gap | Book implied (SGM) | Book vs actual |
|------|---|----------|----------|-----|--------------------|----------------|
| 30%-50% | 10,932 | 0.395 | 0.385 | +0.010 |              0.580 |         +0.194 |
| 50%-70% | 6,774 | 0.592 | 0.581 | +0.011 |              0.746 |         +0.165 |
| 70%-90% | 2,886 | 0.768 | 0.758 | +0.010 |              0.866 |         +0.108 |

### Tackles

| Band | n | Cal prob | Hit rate | Gap | Book implied (SGM) | Book vs actual |
|------|---|----------|----------|-----|--------------------|----------------|
| 30%-50% | 7,671 | 0.388 | 0.386 | +0.002 |              0.484 |         +0.098 |
| 50%-70% | 3,436 | 0.584 | 0.566 | +0.017 |              0.703 |         +0.137 |
| 70%-90% | 2,124 | 0.784 | 0.764 | +0.021 |              0.850 |         +0.086 |

## Cross-check vs calibration_log.csv

calibration_log (1121 prop rows): mean pred = 0.432, actual hit rate = 0.460, gap = -0.028
*(This log uses the model calibrated probs from each live round-report run, not the walk-forward reproduced probs above.)*

## Verdict

**DISPOSALS - WELL CALIBRATED:** Model and actual hit rate are closely aligned for disposals (cal gap +0.013). The book prices its disposals legs ~7.8pp above the model on priced SGM legs (ev_diag_gap=+0.078), which is the book's structural margin, not model error. Accept it; only back legs with clear model edge. No calibration adjustment needed for this market.

**GOALS - WELL CALIBRATED:** Model and actual hit rate are closely aligned for goals (cal gap +0.000). The book prices its goals legs ~4.4pp above the model on priced SGM legs (ev_diag_gap=+0.044), which is the book's structural margin, not model error. Accept it; only back legs with clear model edge. No calibration adjustment needed for this market.

**MARKS - WELL CALIBRATED:** Model and actual hit rate are closely aligned for marks (cal gap +0.009). The book prices its marks legs ~11.4pp above the model on priced SGM legs (ev_diag_gap=+0.114), which is the book's structural margin, not model error. Accept it; only back legs with clear model edge. No calibration adjustment needed for this market.

**TACKLES - WELL CALIBRATED:** Model and actual hit rate are closely aligned for tackles (cal gap +0.002). The book prices its tackles legs ~9.6pp above the model on priced SGM legs (ev_diag_gap=+0.096), which is the book's structural margin, not model error. Accept it; only back legs with clear model edge. No calibration adjustment needed for this market.

## Overall Call

**NO CALIBRATION ACTION NEEDED.** All 4 markets are well-calibrated vs actual hit rates. The observed -EV in SGM multis is attributable to the book's structural margin (8-12pp above model on priced legs), not to model conservatism. No calibration tightening is warranted. Focus on leg selection quality (ev > threshold) to manage the book's edge.

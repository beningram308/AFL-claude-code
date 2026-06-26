"""Closing price snapshot for pending bets — Phase 3 STEP 2.

Run near bounce:
    python -m afl_bot.cli capture-close [--year N] [--round N] [--sportsbet-urls path.json]

Source priority (FIX-PHASE3-CLV.txt):
  1. Betfair exchange (H2H/line) - NOT YET CONNECTED -> clv_available=False, source="no-betfair"
  2. De-vigged consensus across >=2 books (props) - needs 2nd scraper -> source="single-book"
  3. Sportsbet close recorded for line-movement tracking ONLY (Step 4 attribution).
     Never used as the CLV reference (soft-self comparison is meaningless).

The command is idempotent: a bet already bearing close_captured_at is skipped.
Logs n_updated / n_sharp / n_soft_only; currently n_sharp is always 0.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from afl_bot.dashboard.ledger import load_ledger, save_ledger


def _melbourne_now() -> str:
    offset = timedelta(hours=10)
    return datetime.now(tz=timezone(offset)).isoformat()


def _line_move_flag(open_odds: float | None, close_odds: float | None) -> str | None:
    """STEP 4: record price-move direction without catalyst attribution (Phase 5)."""
    if open_odds is None or close_odds is None:
        return None
    ratio = close_odds / open_odds
    if ratio < 0.97:
        return "shortened"
    if ratio > 1.03:
        return "drifted"
    return "stable"


def capture_close(
    ledger_path: str | Path,
    year: int | None = None,
    round_no: int | None = None,
    sportsbet_urls: list[str] | None = None,
) -> dict:
    """Snapshot closing reference prices for pending bets.

    For each pending bet without a prior snapshot:
      - Fetches current Sportsbet prices (line-movement tracking only — not CLV).
      - Sets clv_available=False (no sharp reference available yet).
      - Records close_ref_source: 'no-betfair' for H2H/totals,
        'single-book' for props.

    Returns {"n_updated": int, "n_sharp": int, "n_soft_only": int}.
    """
    bets = load_ledger(ledger_path)
    captured_at = _melbourne_now()

    sb_close: dict[str, float] = {}
    if sportsbet_urls:
        try:
            from afl_bot.data.sportsbet_odds import fetch_sportsbet_odds
            sb_close = fetch_sportsbet_odds(list(sportsbet_urls))
        except Exception as exc:
            print(f"[capture-close] Sportsbet fetch failed: {exc}", file=sys.stderr)

    n_updated = n_sharp = n_soft_only = 0

    for bet in bets:
        if bet.get("close_captured_at") is not None:
            continue  # idempotent
        if bet["status"] != "pending":
            continue
        if year is not None and bet.get("year") != year:
            continue
        if round_no is not None and bet.get("round") != round_no:
            continue

        # Per-leg Sportsbet close prices (line-movement tracking, NOT CLV)
        close_legs: list[dict] = []
        for leg in bet.get("legs", []):
            name = leg.get("name", "")
            open_p = leg.get("book_odds")
            close_p = sb_close.get(name)
            close_legs.append({
                "name": name,
                "open_odds": open_p,
                "close_odds": close_p,
                "line_move_flag": _line_move_flag(open_p, close_p),
            })

        # Source note (why CLV is unavailable)
        markets = {l.get("market", "") for l in bet.get("legs", [])}
        if markets <= {"h2h", "total_points", ""}:
            source = "no-betfair"
        else:
            source = "single-book"

        bet["close_captured_at"] = captured_at
        bet["close_ref_odds"] = None
        bet["close_ref_source"] = source
        bet["close_implied_prob"] = None
        bet["clv_pct"] = None
        bet["clv_available"] = False
        bet["close_legs"] = close_legs
        n_soft_only += 1
        n_updated += 1

    save_ledger(ledger_path, bets)
    return {"n_updated": n_updated, "n_sharp": n_sharp, "n_soft_only": n_soft_only}

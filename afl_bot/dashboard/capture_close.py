"""Closing price snapshot for pending bets — Phase 3 STEP 2.

Run near bounce:
    python -m afl_bot.cli capture-close [--year N] [--round N] [--sportsbet-urls path.json]

Source priority (FIX-PHASE3-CLV.txt / FIX-SECOND-BOOK-FOR-PROP-CLV.txt):
  1. De-vigged consensus across Sportsbet + TAB when BOTH books have a price for
     every leg in the bet:
       clv_available=True, source="consensus:sportsbet+tab"
  2. Only Sportsbet available:
       H2H/totals -> source="no-betfair"   (Betfair is the future upgrade)
       Props      -> source="single-book"  (soft-self comparison is meaningless)
       clv_available=False in both cases.

The command is idempotent: a bet already bearing close_captured_at is skipped.
Logs n_updated / n_sharp / n_soft_only.

tab_odds parameter: pass a pre-fetched {leg_name: decimal_odds} dict from
fetch_tab_odds(), or None (default) to skip TAB.  The CLI fetches TAB
automatically before calling this function.
"""
from __future__ import annotations

import math
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
    tab_odds: dict[str, float] | None = None,
) -> dict:
    """Snapshot closing reference prices for pending bets.

    tab_odds: pre-fetched TAB prices {leg_name: decimal_odds}.  Pass None
    (default) to skip TAB; pass a real dict for consensus CLV.
    Returns {"n_updated": int, "n_sharp": int, "n_soft_only": int}.
    """
    from afl_bot.dashboard.clv import compute_clv, devig_consensus_single_sided

    bets = load_ledger(ledger_path)
    captured_at = _melbourne_now()

    sb_close: dict[str, float] = {}
    if sportsbet_urls:
        try:
            from afl_bot.data.sportsbet_odds import fetch_sportsbet_odds
            sb_close = fetch_sportsbet_odds(list(sportsbet_urls))
        except Exception as exc:
            print(f"[capture-close] Sportsbet fetch failed: {exc}", file=sys.stderr)

    tab_close: dict[str, float] = tab_odds or {}

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

        legs = bet.get("legs", [])

        # Per-leg Sportsbet close prices (line-movement tracking, STEP 4)
        close_legs: list[dict] = []
        for leg in legs:
            name = leg.get("name", "")
            open_p = leg.get("book_odds")
            close_p = sb_close.get(name)
            close_legs.append({
                "name": name,
                "open_odds": open_p,
                "close_odds": close_p,
                "line_move_flag": _line_move_flag(open_p, close_p),
            })

        # Try to build a consensus reference (needs both SB + TAB for every leg)
        got_consensus = False
        if legs and tab_close:
            leg_probs: list[float] = []
            for leg in legs:
                name = leg.get("name", "")
                sb_p = sb_close.get(name)
                tab_p = tab_close.get(name)
                if sb_p is not None and tab_p is not None:
                    try:
                        leg_probs.append(devig_consensus_single_sided([sb_p, tab_p]))
                    except Exception:
                        break
                else:
                    break
            else:
                # For-loop completed without break: all legs have consensus probs
                if len(leg_probs) == len(legs):
                    got_consensus = True

        if got_consensus:
            ref_prob = math.prod(leg_probs)
            close_ref_odds = 1.0 / ref_prob if ref_prob > 0 else None
            open_odds = bet.get("open_odds") or bet.get("taken_odds")
            clv_pct = None
            if close_ref_odds is not None and open_odds is not None:
                clv_pct = compute_clv(open_odds, close_ref_odds)
            bet["close_captured_at"] = captured_at
            bet["close_ref_odds"] = close_ref_odds
            bet["close_ref_source"] = "consensus:sportsbet+tab"
            bet["close_implied_prob"] = (1.0 / close_ref_odds) if close_ref_odds else None
            bet["clv_pct"] = clv_pct
            bet["clv_available"] = True
            bet["close_legs"] = close_legs
            n_sharp += 1
            n_updated += 1
        else:
            # Source note (why CLV is unavailable)
            markets = {l.get("market", "") for l in legs}
            source = "no-betfair" if markets <= {"h2h", "total_points", ""} else "single-book"
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

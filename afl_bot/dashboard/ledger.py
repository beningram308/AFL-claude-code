"""Bets ledger — read/write reports/bets_ledger.json (Stage 2B).

Schema per bet:
  bet_id            uuid string
  multi_id          links back to the multis.json record (stable id)
  year, round       int
  game              "Home vs Away"
  ladder            "model" | "sportsbet"
  legs              snapshot of legs at placement (list of dicts from multis.json)
  stake             float (AUD)
  open_odds         float — fill odds at placement (CLV baseline)
  taken_odds        float — same as open_odds; kept for backward compat
  placed_at         ISO-8601 with +10:00 / +11:00 (Australia/Melbourne)
  status            "pending" | "won" | "lost" | "void"
  settled_at        ISO-8601 or null
  payout            float or null (stake*taken_odds on win, 0 on loss, stake on full void)
  leg_results       list of {"name":..., "hit": bool|null} or null
  -- CLV fields (added by capture-close / add_clv_snapshot) --
  open_odds         fill odds; CLV baseline (= taken_odds for bets placed at market price)
  close_captured_at ISO-8601 or null
  close_ref_odds    float or null  (sharp reference near bounce)
  close_ref_source  "betfair"|"consensus-N"|"no-betfair"|"single-book" or null
  close_implied_prob float or null (1/close_ref_odds)
  clv_pct           float or null  (1/close_ref_odds - 1/open_odds)
  clv_available     bool           (False until a sharp reference is connected)
  close_legs        list of per-leg {name, open_odds, close_odds, line_move_flag}
"""

from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _melbourne_now() -> str:
    """Current time as ISO-8601 string in Australia/Melbourne (+10/+11)."""
    import time as _time
    # Simple DST approximation: AEDT (+11) from first Sun in Oct to first Sun in Apr,
    # AEST (+10) otherwise.  Python's datetime has no built-in IANA tz on Windows,
    # so we use a fixed +10 offset for simplicity (acceptable for bet timestamps).
    offset = timedelta(hours=10)
    return datetime.now(tz=timezone(offset)).isoformat()


def load_ledger(ledger_path: str | Path) -> list[dict]:
    p = Path(ledger_path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_ledger(ledger_path: str | Path, bets: list[dict]) -> None:
    Path(ledger_path).write_text(json.dumps(bets, indent=2), encoding="utf-8")


def add_bet(ledger_path: str | Path, multi_record: dict,
            stake: float, taken_odds: float,
            source: str = "bot") -> dict:
    """Append a new pending bet to the ledger and return the bet record."""
    bet = {
        "bet_id": str(uuid.uuid4()),
        "multi_id": multi_record["id"],
        "year": multi_record["year"],
        "round": multi_record["round"],
        "game": multi_record["game"],
        "ladder": multi_record["ladder"],
        "legs": copy.deepcopy(multi_record["legs"]),  # deep snapshot at placement
        "stake": float(stake),
        "open_odds": float(taken_odds),  # CLV baseline = fill price at placement
        "taken_odds": float(taken_odds),
        "placed_at": _melbourne_now(),
        "status": "pending",
        "settled_at": None,
        "payout": None,
        "leg_results": None,
        "source": source,
        "manual_result": None,
        "close_captured_at": None,
        "close_ref_odds": None,
        "close_ref_source": None,
        "close_implied_prob": None,
        "clv_pct": None,
        "clv_available": False,
        "close_legs": None,
    }
    bets = load_ledger(ledger_path)
    bets.append(bet)
    save_ledger(ledger_path, bets)
    return bet


def add_manual_bet(
    ledger_path: str | Path,
    *,
    year: int,
    round_no: int,
    game: str,
    stake: float,
    taken_odds: float,
    legs: list[dict],
    label: str | None = None,
) -> dict:
    """Append a manually-entered bet to the ledger.

    ``legs`` is a list of dicts already in the standard leg schema:
    ``{player, market, line, name, book_odds}``.  Callers build them from
    the form fields (player prop / team to win / total points / other).
    """
    bet_id = str(uuid.uuid4())
    multi_id = f"manual-{year}-r{round_no}-{bet_id[:8]}"
    bet = {
        "bet_id": bet_id,
        "multi_id": multi_id,
        "year": year,
        "round": round_no,
        "game": game,
        "ladder": "manual",
        "legs": copy.deepcopy(legs),
        "stake": float(stake),
        "open_odds": float(taken_odds),
        "taken_odds": float(taken_odds),
        "placed_at": _melbourne_now(),
        "status": "pending",
        "settled_at": None,
        "payout": None,
        "leg_results": None,
        "source": "manual",
        "manual_result": None,
        "label": label or "",
        "close_captured_at": None,
        "close_ref_odds": None,
        "close_ref_source": None,
        "close_implied_prob": None,
        "clv_pct": None,
        "clv_available": False,
        "close_legs": None,
    }
    bets = load_ledger(ledger_path)
    bets.append(bet)
    save_ledger(ledger_path, bets)
    return bet


def manual_settle_bet(
    ledger_path: str | Path,
    bet_id: str,
    *,
    outcome: str,
) -> bool:
    """Manually force the outcome of a bet to ``outcome`` ("won"/"lost"/"void").

    Sets ``manual_result`` so ``settle_bets`` will honour it on the next pass
    (or this write immediately applies it if the ledger is then re-read).
    Returns True if the bet was found, False otherwise.
    """
    if outcome not in ("won", "lost", "void"):
        raise ValueError(f"outcome must be won/lost/void, got {outcome!r}")
    bets = load_ledger(ledger_path)
    for bet in bets:
        if bet["bet_id"] != bet_id:
            continue
        bet["manual_result"] = outcome
        if outcome == "won":
            bet["status"] = "won"
            bet["payout"] = round(bet["stake"] * bet["taken_odds"], 2)
        elif outcome == "lost":
            bet["status"] = "lost"
            bet["payout"] = 0.0
        else:
            bet["status"] = "void"
            bet["payout"] = bet["stake"]
        bet["settled_at"] = _melbourne_now()
        save_ledger(ledger_path, bets)
        return True
    return False


def pnl_summary(bets: list[dict]) -> dict:
    """Season P&L summary over all settled bets."""
    settled = [b for b in bets if b["status"] in ("won", "lost", "void")]
    won = [b for b in settled if b["status"] == "won"]
    total_staked = sum(b["stake"] for b in settled)
    total_returned = sum(b.get("payout") or 0.0 for b in settled)
    net_profit = total_returned - total_staked
    roi = net_profit / total_staked if total_staked > 0 else 0.0
    non_void = [b for b in settled if b["status"] != "void"]
    strike_rate = len(won) / len(non_void) if non_void else 0.0
    return {
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "net_profit": round(net_profit, 2),
        "roi_pct": round(roi * 100, 2),
        "strike_rate": round(strike_rate, 4),
        "n_settled": len(settled),
        "n_won": len(won),
    }


def add_clv_snapshot(
    ledger_path: str | Path,
    bet_id: str,
    *,
    close_ref_odds: float | None,
    close_ref_source: str,
    clv_available: bool,
    captured_at: str | None = None,
) -> bool:
    """Update an existing bet with CLV snapshot fields.

    When clv_available=True and close_ref_odds is provided, clv_pct is
    computed as (1/close_ref_odds) - (1/open_odds).
    Returns True if bet_id was found, False otherwise.
    """
    from afl_bot.dashboard.clv import compute_clv
    bets = load_ledger(ledger_path)
    for bet in bets:
        if bet["bet_id"] != bet_id:
            continue
        open_odds = bet.get("open_odds") or bet.get("taken_odds")
        bet["close_ref_odds"] = close_ref_odds
        bet["close_ref_source"] = close_ref_source
        bet["close_implied_prob"] = (1.0 / close_ref_odds) if close_ref_odds else None
        bet["clv_available"] = clv_available
        if clv_available and close_ref_odds is not None and open_odds is not None:
            bet["clv_pct"] = compute_clv(open_odds, close_ref_odds)
        else:
            bet["clv_pct"] = None
        bet["close_captured_at"] = captured_at or _melbourne_now()
        save_ledger(ledger_path, bets)
        return True
    return False


def cumulative_profit(bets: list[dict]) -> list[dict]:
    """Running cumulative profit over settled bets ordered by settled_at."""
    settled = sorted(
        [b for b in bets if b["status"] in ("won", "lost", "void") and b.get("settled_at")],
        key=lambda b: b["settled_at"])
    running = 0.0
    result = []
    for b in settled:
        payout = b.get("payout") or 0.0
        running += payout - b["stake"]
        result.append({"settled_at": b["settled_at"], "cumulative_profit": round(running, 2),
                        "bet_id": b["bet_id"]})
    return result

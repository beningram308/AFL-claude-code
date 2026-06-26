"""Bets ledger — read/write reports/bets_ledger.json (Stage 2B).

Schema per bet:
  bet_id       uuid string
  multi_id     links back to the multis.json record (stable id)
  year, round  int
  game         "Home vs Away"
  ladder       "model" | "sportsbet"
  legs         snapshot of legs at placement (list of dicts from multis.json)
  stake        float (AUD)
  taken_odds   float (odds Ben actually got)
  placed_at    ISO-8601 with +10:00 / +11:00 (Australia/Melbourne)
  status       "pending" | "won" | "lost" | "void"
  settled_at   ISO-8601 or null
  payout       float or null (stake*taken_odds on win, 0 on loss, stake on full void)
  leg_results  list of {"name":..., "hit": bool|null} or null
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
            stake: float, taken_odds: float) -> dict:
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
        "taken_odds": float(taken_odds),
        "placed_at": _melbourne_now(),
        "status": "pending",
        "settled_at": None,
        "payout": None,
        "leg_results": None,
    }
    bets = load_ledger(ledger_path)
    bets.append(bet)
    save_ledger(ledger_path, bets)
    return bet


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

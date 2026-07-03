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
import shutil
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from afl_bot.io_utils import atomic_write_text

# Rolling backup: keep at most this many daily snapshots.
_BACKUP_KEEP = 14
# Sidecar written on corruption so the dashboard can show a red banner.
_CORRUPTION_NOTICE_NAME = "ledger-corruption-notice.json"


def _melbourne_now() -> str:
    """Current time as ISO-8601 string in Australia/Melbourne (+10/+11)."""
    offset = timedelta(hours=10)
    return datetime.now(tz=timezone(offset)).isoformat()


def _write_backup(ledger_path: Path, bets: list[dict]) -> None:
    """Write a dated rolling backup; prune to the most recent _BACKUP_KEEP."""
    backup_dir = ledger_path.parent / "ledger_backups"
    backup_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    backup_path = backup_dir / f"bets_ledger.{date_str}.json"
    atomic_write_text(backup_path, json.dumps(bets, indent=2))
    # Prune old backups
    backups = sorted(backup_dir.glob("bets_ledger.*.json"))
    for old in backups[:-_BACKUP_KEEP]:
        try:
            old.unlink()
        except OSError:
            pass


def load_ledger(ledger_path: str | Path) -> list[dict]:
    p = Path(ledger_path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _recover_ledger(p, text)


def _recover_ledger(p: Path, text: str) -> list[dict]:
    """Corruption recovery: save a backup of the corrupt file, attempt to
    parse all complete JSON objects out of the partial content, then
    atomically overwrite the ledger with only the recovered records.

    Writes a sidecar file so the dashboard can display a red banner.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    corrupt_backup = p.with_name(f"bets_ledger.corrupt-{ts}.json")
    try:
        shutil.copy2(p, corrupt_backup)
    except OSError as e:
        print(f"[LEDGER] WARNING: could not copy corrupt file: {e}", file=sys.stderr)

    # Recover complete JSON objects using raw_decode
    recovered: list[dict] = []
    decoder = json.JSONDecoder()
    raw = text.strip().lstrip("[")
    pos = 0
    while pos < len(raw):
        chunk = raw[pos:].lstrip(" \t\n\r,")
        if not chunk or chunk.startswith("]"):
            break
        skip = len(raw[pos:]) - len(chunk)
        try:
            obj, end = decoder.raw_decode(chunk)
            if isinstance(obj, dict):
                recovered.append(obj)
            pos = pos + skip + end
        except json.JSONDecodeError:
            break

    n_expected = text.count('"bet_id"')
    n_lost = max(0, n_expected - len(recovered))
    msg = (
        f"CORRUPTION DETECTED in {p.name} — "
        f"{len(recovered)} bets recovered, {n_lost} lost. "
        f"Corrupt backup: {corrupt_backup.name}"
    )
    print(f"[LEDGER] {msg}", file=sys.stderr)

    # Atomically overwrite with recovered data
    atomic_write_text(p, json.dumps(recovered, indent=2))

    # Write sidecar so dashboard can show red banner
    notice = {
        "timestamp": ts,
        "recovered": len(recovered),
        "lost": n_lost,
        "corrupt_backup": str(corrupt_backup.name),
        "message": msg,
    }
    notice_path = p.parent / _CORRUPTION_NOTICE_NAME
    atomic_write_text(notice_path, json.dumps(notice, indent=2))

    return recovered


def load_corruption_notice(ledger_path: str | Path) -> dict | None:
    """Return the corruption notice dict if one exists, else None."""
    p = Path(ledger_path).parent / _CORRUPTION_NOTICE_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def dismiss_corruption_notice(ledger_path: str | Path) -> None:
    """Delete the corruption notice file (user has acknowledged it)."""
    p = Path(ledger_path).parent / _CORRUPTION_NOTICE_NAME
    try:
        p.unlink()
    except OSError:
        pass


def save_ledger(ledger_path: str | Path, bets: list[dict]) -> None:
    p = Path(ledger_path)
    _write_backup(p, bets)
    atomic_write_text(p, json.dumps(bets, indent=2))


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
        "legs": copy.deepcopy(multi_record["legs"]),
        "stake": float(stake),
        "open_odds": float(taken_odds),
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
    book: str = "other",
) -> dict:
    """Append a manually-entered bet to the ledger."""
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
        "book": book,
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
    """Manually force the outcome of a bet to *outcome* ("won"/"lost"/"void").

    Sets ``manual_result`` so ``settle_bets`` will honour it on the next pass.
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


def reopen_bet(ledger_path: str | Path, bet_id: str) -> bool:
    """Clear ``manual_result`` and reset the bet to pending so the next
    settle pass re-grades it from real stats.  Returns True if found."""
    bets = load_ledger(ledger_path)
    for bet in bets:
        if bet["bet_id"] != bet_id:
            continue
        bet["manual_result"] = None
        bet["status"] = "pending"
        bet["payout"] = None
        bet["settled_at"] = None
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

"""Tests for ledger safety: corruption recovery, rolling backups, reopen_bet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afl_bot.dashboard.ledger import (
    add_manual_bet,
    load_ledger,
    manual_settle_bet,
    reopen_bet,
    save_ledger,
    load_corruption_notice,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_bet(bet_id: str = "aaa-111") -> dict:
    return {
        "bet_id": bet_id,
        "multi_id": "manual-2026-r17-aaa",
        "year": 2026,
        "round": 17,
        "game": "Sydney vs West Coast",
        "ladder": "manual",
        "legs": [{"name": "Papley 2+ goals", "market": "player_goals", "line": 2}],
        "stake": 25.0,
        "open_odds": 6.50,
        "taken_odds": 6.50,
        "placed_at": "2026-07-01T12:00:00+10:00",
        "status": "pending",
        "settled_at": None,
        "payout": None,
        "leg_results": None,
        "source": "manual",
        "book": "sportsbet",
        "manual_result": None,
        "label": "test",
        "close_captured_at": None,
        "close_ref_odds": None,
        "close_ref_source": None,
        "close_implied_prob": None,
        "clv_pct": None,
        "clv_available": False,
        "close_legs": None,
    }


# ── corruption recovery ──────────────────────────────────────────────────────

def test_corruption_recovery_complete_records(tmp_path):
    """Truncated JSON array: complete bets before truncation are recovered."""
    bet1 = _make_bet("bet-001")
    bet2 = _make_bet("bet-002")
    # Build a valid JSON array, then truncate mid-second-record
    full = json.dumps([bet1, bet2], indent=2)
    # Truncate: keep full first record + partial second
    truncated = full[:full.index('"bet_id": "bet-002"') + 5]
    p = tmp_path / "bets_ledger.json"
    p.write_text(truncated, encoding="utf-8")

    recovered = load_ledger(p)
    assert len(recovered) == 1
    assert recovered[0]["bet_id"] == "bet-001"


def test_corruption_recovery_backup_file_created(tmp_path):
    """Corrupt file is backed up before being overwritten."""
    p = tmp_path / "bets_ledger.json"
    p.write_text("[{bad json", encoding="utf-8")

    load_ledger(p)

    corrupt_backups = list(tmp_path.glob("bets_ledger.corrupt-*.json"))
    assert len(corrupt_backups) == 1
    assert corrupt_backups[0].read_text(encoding="utf-8") == "[{bad json"


def test_corruption_recovery_no_exception(tmp_path):
    """load_ledger must never raise even on fully corrupt content."""
    p = tmp_path / "bets_ledger.json"
    p.write_text("this is not json at all !!!!", encoding="utf-8")
    result = load_ledger(p)
    assert isinstance(result, list)


def test_corruption_recovery_writes_sidecar(tmp_path):
    """A ledger-corruption-notice.json sidecar is written so the dashboard can warn."""
    p = tmp_path / "bets_ledger.json"
    p.write_text("[{bad json", encoding="utf-8")
    load_ledger(p)
    notice = load_corruption_notice(p)
    assert notice is not None
    assert "recovered" in notice
    assert "corrupt_backup" in notice


def test_corruption_recovery_recovers_all_complete_records(tmp_path):
    """Three bets in array, truncation after second — recover exactly two."""
    bets = [_make_bet(f"bet-{i:03d}") for i in range(3)]
    full = json.dumps(bets, indent=2)
    # Find start of third bet and truncate there
    idx = full.index('"bet-002"') + 5   # just past start of 3rd bet_id value
    p = tmp_path / "bets_ledger.json"
    p.write_text(full[:idx], encoding="utf-8")
    recovered = load_ledger(p)
    assert len(recovered) == 2
    assert {b["bet_id"] for b in recovered} == {"bet-000", "bet-001"}


# ── rolling backup ────────────────────────────────────────────────────────────

def test_save_ledger_creates_backup(tmp_path):
    p = tmp_path / "bets_ledger.json"
    bets = [_make_bet("bet-111")]
    save_ledger(p, bets)
    backups = list((tmp_path / "ledger_backups").glob("bets_ledger.*.json"))
    assert len(backups) == 1


def test_save_ledger_backup_content_matches(tmp_path):
    p = tmp_path / "bets_ledger.json"
    bets = [_make_bet("bet-222")]
    save_ledger(p, bets)
    backup = sorted((tmp_path / "ledger_backups").glob("bets_ledger.*.json"))[0]
    assert json.loads(backup.read_text(encoding="utf-8")) == bets


# ── reopen_bet ────────────────────────────────────────────────────────────────

def test_reopen_bet_clears_manual_result(tmp_path):
    p = tmp_path / "bets_ledger.json"
    bet = _make_bet("reopen-001")
    save_ledger(p, [bet])
    manual_settle_bet(p, "reopen-001", outcome="void")
    reopen_bet(p, "reopen-001")
    loaded = load_ledger(p)
    b = next(b for b in loaded if b["bet_id"] == "reopen-001")
    assert b["manual_result"] is None
    assert b["status"] == "pending"
    assert b["payout"] is None
    assert b["settled_at"] is None


def test_reopen_bet_returns_false_for_unknown_id(tmp_path):
    p = tmp_path / "bets_ledger.json"
    save_ledger(p, [_make_bet("known-id")])
    result = reopen_bet(p, "nonexistent-id")
    assert result is False


def test_reopen_bet_returns_true_when_found(tmp_path):
    p = tmp_path / "bets_ledger.json"
    bet = _make_bet("found-001")
    bet["manual_result"] = "lost"
    bet["status"] = "lost"
    save_ledger(p, [bet])
    result = reopen_bet(p, "found-001")
    assert result is True


# ── add-manual-bet failure path ───────────────────────────────────────────────

def test_add_manual_bet_failure_raises(tmp_path, monkeypatch):
    """Monkeypatch save_ledger to raise; add_manual_bet must propagate the error."""
    import afl_bot.dashboard.ledger as _ledger

    def bad_save(path, bets):
        raise OSError("disk full")

    monkeypatch.setattr(_ledger, "save_ledger", bad_save)
    p = tmp_path / "bets_ledger.json"
    with pytest.raises(OSError, match="disk full"):
        add_manual_bet(p, year=2026, round_no=17, game="Test vs Test",
                       stake=25.0, taken_odds=3.50,
                       legs=[{"name": "leg1", "market": "player_disposals", "line": 25}])


def test_add_manual_bet_returns_bet_dict(tmp_path):
    p = tmp_path / "bets_ledger.json"
    bet = add_manual_bet(p, year=2026, round_no=17, game="Sydney vs WC",
                         stake=20.0, taken_odds=4.0,
                         legs=[{"name": "X 25+ disposals", "market": "player_disposals", "line": 25}],
                         label="test", book="sportsbet")
    assert "bet_id" in bet
    assert bet["status"] == "pending"
    loaded = load_ledger(p)
    assert any(b["bet_id"] == bet["bet_id"] for b in loaded)

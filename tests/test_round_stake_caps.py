"""DO-EDGE-FLOOR-VIABILITY-TEST (2026-08-14) TASK 2 -- afl_bot.cli._apply_round_
stake_caps, the live round-level allocator (per-player cap + restored 15u
round cap) ported from the validated afl_bot.backtest.stake_cap policy C.
Required cases (b)/(c)/(d) from the port brief; (b)/(c) use the real saved
r21/r22 multis.json files as regression fixtures (91.0u and 83.2u today, per
the port brief) so this pins actual production data, not just synthetic
shapes."""

from __future__ import annotations

import json
from pathlib import Path

from afl_bot.cli import _apply_round_stake_caps
from afl_bot.config import ROOT_DIR, UNIT_SIZE

REPORTS_DIR = ROOT_DIR / "reports"


def _staked(records: list[dict]) -> list[dict]:
    return [r for r in records
            if r.get("ladder") in ("model", "sportsbet")
            and not r.get("no_bet")
            and (r.get("units") or 0.0) > 0.0]


def _leg(player: str, market: str = "disposals", line: float = 20.0) -> dict:
    return {"player": player, "market": market, "line": line, "name": f"{player} {line:g}+ {market}"}


def _rec(*, ladder="model", units=1.0, total_ev=0.20, joint=0.40, book=3.0,
        p_win=0.35, p_one_loss=0.35, players=("Alice",), band=3.0,
        game="Home vs Away", no_bet=False) -> dict:
    return {
        "ladder": ladder, "no_bet": no_bet, "units": units, "units_tag": f"{units:g}u",
        "model_joint": joint, "book_combo": book, "total_ev": total_ev,
        "promo_ev": total_ev, "p_all_win": p_win, "p_one_loss": p_one_loss,
        "band": band, "game": game,
        "legs": [_leg(p) for p in players],
    }


# ── (b) / (c): real r21/r22 fixtures trim to the 15u round cap ─────────────

def test_r21_saved_multis_trims_to_round_cap():
    path = REPORTS_DIR / "2026_r21_multis.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    staked_before = _staked(records)
    total_before = sum(r["units"] for r in staked_before)
    assert total_before == 91.0, (
        f"fixture drifted: expected r21's known 91.0u, got {total_before}u -- "
        "re-read the port brief's numbers before trusting this test"
    )

    _apply_round_stake_caps(records, 2026, 21)

    staked_after = _staked(records)
    total_after = sum(r["units"] for r in staked_after)
    assert total_after <= 15.0 + 1e-9, f"round cap must trim to <=15u, got {total_after}u"
    assert total_after < total_before, "the cap must have actually done something on a 91u round"


def test_r22_saved_multis_trims_to_round_cap():
    path = REPORTS_DIR / "2026_r22_multis.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    staked_before = _staked(records)
    total_before = sum(r["units"] for r in staked_before)
    assert total_before == 83.25, (
        f"fixture drifted: expected r22's known ~83.2u, got {total_before}u -- "
        "re-read the port brief's numbers before trusting this test"
    )

    _apply_round_stake_caps(records, 2026, 22)

    staked_after = _staked(records)
    total_after = sum(r["units"] for r in staked_after)
    assert total_after <= 15.0 + 1e-9, f"round cap must trim to <=15u, got {total_after}u"
    assert total_after < total_before, "the cap must have actually done something on an 83u round"


def test_round_cap_trim_updates_units_tag_leading_number():
    # Whatever survives the round cap at a REDUCED size must have its
    # units_tag's leading number kept in sync (the .md/JSON-agreement
    # invariant _rung_to_json's docstring states) -- not a stale "3u" tag on
    # a rung that got trimmed to 1u.
    path = REPORTS_DIR / "2026_r21_multis.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    _apply_round_stake_caps(records, 2026, 21)
    for r in _staked(records):
        import re
        m = re.match(r"^([\d.]+)u", r["units_tag"])
        assert m, f"units_tag must start with the units number, got {r['units_tag']!r}"
        assert float(m.group(1)) == r["units"], (
            f"units_tag {r['units_tag']!r} disagrees with units={r['units']}"
        )


# ── (d): same player cannot appear in two staked multis in a round ─────────

def test_same_player_cannot_appear_in_two_staked_multis_same_round():
    # Directly the shape Ben's own backtest found in R16: 7 edge-floor-eligible
    # rungs sharing players, only 5 survive the per-player cap. Two rungs here
    # share "Alice" with different raw edge (strong=0.40*3.0-1=+0.20, weak=
    # 0.34*3.0-1=+0.02); a third rung on a different player must be unaffected.
    strong = _rec(units=2.0, joint=0.40, book=3.0, total_ev=0.30, players=("Alice",))
    weak = _rec(units=1.0, joint=0.34, book=3.0, total_ev=0.05, players=("Alice", "Bob"))
    unrelated = _rec(units=1.0, joint=0.40, book=3.0, total_ev=0.30, players=("Carol",))
    records = [strong, weak, unrelated]

    _apply_round_stake_caps(records, 2026, 16)

    assert strong["units"] == 2.0, "higher-raw-edge rung on Alice must survive"
    assert weak["units"] == 0.0 and weak["units_tag"] == "NO BET (player cap)", (
        "lower-raw-edge rung sharing Alice must be dropped, not just co-staked"
    )
    assert unrelated["units"] == 1.0, "a rung on an unrelated player must be untouched"
    # No player appears in more than one STAKED rung's leg list.
    from collections import Counter
    counts = Counter(leg["player"] for r in _staked(records) for leg in r["legs"])
    assert all(n == 1 for n in counts.values()), f"a player is staked twice: {counts}"


def test_same_player_cap_is_per_round_not_global():
    r16 = _rec(units=1.0, joint=0.40, book=3.0, total_ev=0.30, players=("Alice",), band=3.0)
    r17_shape = dict(r16)  # same player, but this call is scoped to a DIFFERENT round
    records_r16 = [r16]
    records_r17 = [dict(r17_shape)]

    _apply_round_stake_caps(records_r16, 2026, 16)
    _apply_round_stake_caps(records_r17, 2026, 17)

    assert records_r16[0]["units"] == 1.0
    assert records_r17[0]["units"] == 1.0, "the same player in a DIFFERENT round must not be capped"


def test_no_bet_and_pull_em_records_are_untouched():
    no_bet = _rec(units=0.0, no_bet=True, players=("Dave",))
    no_bet["units_tag"] = "NO BET"
    pull_em = {"ladder": "pull_em", "units": 3.0, "units_tag": "3u PROMO KELLY",
              "leg_names": ["Eve 20+ disposals"]}
    records = [no_bet, pull_em]
    _apply_round_stake_caps(records, 2026, 16)
    assert no_bet["units"] == 0.0 and no_bet["units_tag"] == "NO BET"
    assert pull_em["units"] == 3.0 and pull_em["units_tag"] == "3u PROMO KELLY"


def test_empty_records_is_a_noop():
    records: list[dict] = []
    _apply_round_stake_caps(records, 2026, 16)  # must not raise
    assert records == []

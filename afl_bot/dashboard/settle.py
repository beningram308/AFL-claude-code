"""Auto-settlement of pending bets (Stage 2C).

Reuses the same actuals path as grade-round: player stats from Fryzigg/DFS
and H2H/totals from Squiggle.  Called by `python -m afl_bot.cli settle-bets`
and automatically when the dashboard loads / the "Settle now" button is pressed.

Settlement rules (FIX-SETTLEMENT-NO-PHANTOM-WINS):
  WON    = every leg HIT (True) — no exceptions.
  LOST   = at least one leg a definite MISS (False); can settle immediately
           even if other legs are still ungradeable (can never win).
  PENDING = any leg is ungradeable (None — no data yet, player not found,
           name mismatch, "other" market).  Never auto-settle over an
           unresolved leg.  The ungradeable leg names are logged in
           `ungradeable_legs` for diagnostics.

  Void (full stake return) is NOT auto-generated.  Genuine did-not-play or
  fully-cancelled multis are handled via the manual-settle control.

1C re-grade: every call also scans existing won/void bets for any leg_result
with hit=null and reverts them to PENDING (clears payout/settled_at) so
previously-phantom wins disappear and re-settle correctly when data arrives.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

from afl_bot.dashboard.ledger import load_ledger, save_ledger


def _melbourne_now() -> str:
    offset = timedelta(hours=10)
    return datetime.now(tz=timezone(offset)).isoformat()


def _load_actuals(year: int, round_no: int) -> tuple[dict, dict, dict]:
    """Return (h2h_actual, total_actual, player_stat) for the given round.

    h2h_actual  : {team_name: 1|0}
    total_actual: {match_id: total_pts}
    player_stat : {(player, stat): value}
    """
    from afl_bot.data.squiggle import SquiggleClient

    client = SquiggleClient()
    games = client.get_completed_games(year)
    games = games[games["round"] == round_no]
    if games.empty:
        return {}, {}, {}

    h2h_actual: dict[str, int] = {}
    total_actual: dict[str, float] = {}
    for _, g in games.iterrows():
        h2h_actual[g["hteam"]] = int(g["hscore"] > g["ascore"])
        h2h_actual[g["ateam"]] = int(g["ascore"] > g["hscore"])
        mid = f"{year}_r{round_no}_{g['hteam']}_v_{g['ateam']}"
        total_actual[mid] = g["hscore"] + g["ascore"]

    player_stat: dict[tuple[str, str], float] = {}
    try:
        from afl_bot.data.fryzigg import fetch_fryzigg_player_stats
        import pandas as pd
        raw = fetch_fryzigg_player_stats()
        raw = raw.assign(
            _year=pd.to_datetime(raw["match_date"]).dt.year,
            _player=(raw["player_first_name"].str.strip() + " " +
                     raw["player_last_name"].str.strip()))
        rnd = raw[(raw["_year"] == year) & (raw["match_round"].astype(str) == str(round_no))]
        if not rnd.empty:
            for _, row in rnd.iterrows():
                for stat in ("disposals", "goals", "marks", "tackles"):
                    if stat in row:
                        player_stat[(row["_player"], stat)] = float(row[stat])
    except Exception:
        pass

    if not player_stat:
        try:
            from afl_bot.data.dfs_australia import fetch_player_stats, to_player_log as _dfs_to_log
            dfs_raw = fetch_player_stats()
            # Force-refresh if the requested round isn't in the cached data.
            if dfs_raw.empty or round_no not in dfs_raw["round"].unique():
                dfs_raw = fetch_player_stats(force_refresh=True)
            dfs = _dfs_to_log(dfs_raw, games)
            dfs_round = dfs[dfs["round"] == round_no]
            for _, row in dfs_round.iterrows():
                for stat in ("disposals", "goals", "marks", "tackles"):
                    if stat in row:
                        player_stat[(row["player"], stat)] = float(row[stat])
        except Exception:
            pass

    return h2h_actual, total_actual, player_stat


def _settle_leg(leg: dict, h2h_actual: dict, total_actual: dict,
                player_stat: dict, year: int, round_no: int) -> tuple[bool | None, str | None]:
    """Return (hit, reason).

    hit    = True (hit) / False (miss) / None (ungradeable).
    reason = None when the leg is gradeable; a short description string when
             hit is None so the dashboard can show WHY a leg is ungradeable
             ("no game result yet" vs "player not found in stats data", etc.).
    """
    market = leg.get("market", "")
    player = leg.get("player", "")
    line = leg.get("line")
    name = leg.get("name", "")

    if market == "other":
        return None, "manual market — requires explicit result"

    if market == "h2h":
        val = h2h_actual.get(player)
        return (bool(val), None) if val is not None else (None, "no game result yet")

    elif market == "total_points" or name.startswith("Total points"):
        import re
        m = re.search(r"([\d.]+)\+", name)
        if m is None:
            return None, "could not parse line from leg name"
        threshold = float(m.group(1))
        game = leg.get("game", "")
        for mid, tot in total_actual.items():
            if game:
                home, _, away = game.partition(" vs ")
                if home in mid and away in mid:
                    return tot >= threshold, None
        return None, "no game total yet"

    else:
        # Player prop: market is the stat name (e.g. "disposals", "player_disposals")
        stat = market.replace("player_", "") if market.startswith("player_") else market
        if line is None or not player:
            return None, "missing player or line data"
        val = player_stat.get((player, stat))
        if val is None:
            if not player_stat:
                return None, "no player stats for this round yet"
            return None, "player not found in stats data"
        return val >= line, None


def settle_bets(ledger_path: str | Path, year: int | None = None,
                round_no: int | None = None) -> int:
    """Settle PENDING bets for the given round (or all pending rounds if None).

    FIX-SETTLEMENT-NO-PHANTOM-WINS: A bet is settled WON only when every single
    leg is a definite hit.  Any unresolved leg (None) keeps the whole bet
    PENDING — we log the ungradeable leg names.  A definite miss on any leg
    settles LOST immediately even if others are still ungradeable (can never win).

    1C re-grade: also scans existing won/void bets; any that have a leg_result
    with hit=null are reverted to PENDING so previous phantom wins disappear.

    Returns the number of bets newly settled this call.
    """
    bets = load_ledger(ledger_path)

    # ── 1C: re-grade phantom wins / voids that slipped through old logic ──────
    for bet in bets:
        if bet.get("manual_result"):   # never revert manually-settled bets
            continue
        if bet.get("status") in ("won", "void") and bet.get("leg_results"):
            if any(lr.get("hit") is None for lr in bet["leg_results"]):
                bet["status"] = "pending"
                bet["payout"] = None
                bet["settled_at"] = None
                # keep leg_results for diagnostic display

    pending = [b for b in bets if b["status"] == "pending"]
    if not pending:
        save_ledger(ledger_path, bets)
        return 0

    actuals_cache: dict[tuple[int, int], tuple] = {}
    n_settled = 0

    for bet in bets:
        if bet["status"] != "pending":
            continue
        y, r = bet.get("year"), bet.get("round")
        if not y or not r:
            continue
        if (year is not None and y != year) or (round_no is not None and r != round_no):
            continue

        # Manual result override (Part 3C): if Ben has already forced win/loss/void,
        # honour it and skip auto-grading entirely.
        manual = bet.get("manual_result")
        if manual in ("won", "lost", "void"):
            if manual == "won":
                bet["status"] = "won"
                bet["payout"] = round(bet["stake"] * bet["taken_odds"], 2)
            elif manual == "lost":
                bet["status"] = "lost"
                bet["payout"] = 0.0
            else:
                bet["status"] = "void"
                bet["payout"] = bet["stake"]
            bet["settled_at"] = _melbourne_now()
            n_settled += 1
            continue

        key = (y, r)
        if key not in actuals_cache:
            actuals_cache[key] = _load_actuals(y, r)
        h2h_actual, total_actual, player_stat = actuals_cache[key]

        if not h2h_actual and not total_actual and not player_stat:
            continue   # round not complete / data unavailable

        leg_results = []
        game = bet.get("game", "")
        for leg in bet["legs"]:
            leg_with_game = {**leg, "game": game}
            hit, reason = _settle_leg(leg_with_game, h2h_actual, total_actual, player_stat, y, r)
            leg_results.append({"name": leg.get("name", ""), "hit": hit, "reason": reason})

        # Never settle over an unresolved leg.
        has_miss = any(lr["hit"] is False for lr in leg_results)
        has_ungradeable = any(lr["hit"] is None for lr in leg_results)

        if has_miss:
            # Definite loss — at least one leg failed; can never win regardless
            # of the ungradeable ones.
            bet["status"] = "lost"
            bet["payout"] = 0.0
            bet["settled_at"] = _melbourne_now()
            bet["leg_results"] = leg_results
            n_settled += 1
        elif has_ungradeable:
            # Cannot confirm outcome yet — store per-leg reason for the dashboard.
            bet["ungradeable_legs"] = [
                {"name": lr["name"], "reason": lr.get("reason") or "ungradeable"}
                for lr in leg_results if lr["hit"] is None
            ]
            bet["leg_results"] = leg_results
            # Stay PENDING — no settled_at, no payout.
        else:
            # All legs are definite hits.
            bet["status"] = "won"
            bet["payout"] = round(bet["stake"] * bet["taken_odds"], 2)
            bet["settled_at"] = _melbourne_now()
            bet["leg_results"] = leg_results
            n_settled += 1

    save_ledger(ledger_path, bets)
    return n_settled

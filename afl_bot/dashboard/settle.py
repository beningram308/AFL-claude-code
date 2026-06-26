"""Auto-settlement of pending bets (Stage 2C).

Reuses the same actuals path as grade-round: player stats from Fryzigg/DFS
and H2H/totals from Squiggle.  Called by `python -m afl_bot.cli settle-bets`
and automatically when the dashboard loads / the "Settle now" button is pressed.

Void rule: if a player did not play (no stat entry), void that leg and re-settle
the remaining legs.  Won only if EVERY non-void leg hit.  A fully-voided multi
returns the stake.
"""

from __future__ import annotations

import sys
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
    from afl_bot.cli import _history_years

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
        raw = fetch_fryzigg_player_stats()
        import pandas as pd
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
            dfs = _dfs_to_log(fetch_player_stats(), games)
            dfs_round = dfs[dfs["round"] == round_no]
            for _, row in dfs_round.iterrows():
                for stat in ("disposals", "goals", "marks", "tackles"):
                    if stat in row:
                        player_stat[(row["player"], stat)] = float(row[stat])
        except Exception:
            pass

    return h2h_actual, total_actual, player_stat


def _settle_leg(leg: dict, h2h_actual: dict, total_actual: dict,
                player_stat: dict, year: int, round_no: int) -> bool | None:
    """Return True (hit), False (miss), or None (void — player didn't play / no data)."""
    market = leg.get("market", "")
    player = leg.get("player", "")
    line = leg.get("line")
    name = leg.get("name", "")

    if market == "h2h":
        val = h2h_actual.get(player)
        return bool(val) if val is not None else None
    elif market == "total_points" or name.startswith("Total points"):
        import re
        m = re.search(r"([\d.]+)\+", name)
        if m is None:
            return None
        threshold = float(m.group(1))
        # match_id reconstruction: we don't have it in the leg, so scan total_actual
        for mid, tot in total_actual.items():
            # Check if the game is this match (game string is "Home vs Away")
            game = leg.get("game", "")
            if game:
                home, _, away = game.partition(" vs ")
                if home in mid and away in mid:
                    return tot >= threshold
        return None
    else:
        # Player prop: market is the stat name (e.g. "disposals")
        if line is None or not player:
            return None
        val = player_stat.get((player, market))
        if val is None:
            return None   # void — player didn't play / no data
        return val >= line


def settle_bets(ledger_path: str | Path, year: int | None = None,
                round_no: int | None = None) -> int:
    """Settle PENDING bets for the given round (or all pending rounds if None).
    Returns the number of bets settled."""
    bets = load_ledger(ledger_path)
    pending = [b for b in bets if b["status"] == "pending"]
    if not pending:
        return 0

    # Group by (year, round) so we load actuals once per round
    rounds_to_check: set[tuple[int, int]] = set()
    for b in pending:
        y, r = b.get("year"), b.get("round")
        if y and r:
            if (year is None or y == year) and (round_no is None or r == round_no):
                rounds_to_check.add((y, r))

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
            hit = _settle_leg(leg_with_game, h2h_actual, total_actual, player_stat, y, r)
            leg_results.append({"name": leg.get("name", ""), "hit": hit})

        non_void = [lr for lr in leg_results if lr["hit"] is not None]
        all_void = len(non_void) == 0

        if all_void:
            bet["status"] = "void"
            bet["payout"] = bet["stake"]
        elif all(lr["hit"] for lr in non_void):
            bet["status"] = "won"
            bet["payout"] = round(bet["stake"] * bet["taken_odds"], 2)
        else:
            bet["status"] = "lost"
            bet["payout"] = 0.0

        bet["settled_at"] = _melbourne_now()
        bet["leg_results"] = leg_results
        n_settled += 1

    save_ledger(ledger_path, bets)
    return n_settled

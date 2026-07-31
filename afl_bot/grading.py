"""Round grading (§10.5): score saved predictions against actual results and
report calibration. Extracted from `afl_bot/cli.py` (audit Phase 3, task 6) --
`grade_round` and its report-writing helpers, zero behaviour change."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

from afl_bot.config import ROOT_DIR
from afl_bot.data.squiggle import SquiggleClient
from afl_bot.io_utils import atomic_write_text


def _format_calibration_section(year: int, round_no: int, graded_df: pd.DataFrame) -> str:
    """Render one round's markdown section for reports/calibration_summary.md
    (audit Phase 3): per-market breakdown, H2H favourite/underdog split, and a
    probability-bucket reliability table. Pure formatting -- does not touch
    calibration_log.csv."""
    from afl_bot.backtest.walkforward import brier_score, calibration_curve, log_loss

    marker_start = f"<!-- calibration-summary:{year}:{round_no}:start -->"
    marker_end = f"<!-- calibration-summary:{year}:{round_no}:end -->"
    lines = [marker_start, f"## {year} Round {round_no}", "",
             "| market | n | log loss | Brier | mean pred | hit rate |",
             "|---|---|---|---|---|---|"]
    for market, g in graded_df.groupby("market"):
        p, a = g["prob"].to_numpy(), g["actual"].to_numpy(dtype=float)
        lines.append(f"| {market} | {len(g)} | {log_loss(p, a):.4f} | {brier_score(p, a):.4f} "
                     f"| {p.mean():.3f} | {a.mean():.3f} |")
    p_all = graded_df["prob"].to_numpy()
    a_all = graded_df["actual"].to_numpy(dtype=float)
    lines.append(f"| **all markets** | **{len(graded_df)}** | **{log_loss(p_all, a_all):.4f}** "
                 f"| **{brier_score(p_all, a_all):.4f}** | **{p_all.mean():.3f}** | **{a_all.mean():.3f}** |")
    lines.append("")

    h2h = graded_df[graded_df["market"] == "h2h"]
    if not h2h.empty:
        lines.append("H2H favourite (prob >= 0.5) vs underdog:")
        lines.append("")
        lines.append("| split | n | log loss | Brier | mean pred | hit rate |")
        lines.append("|---|---|---|---|---|---|")
        for label, g in (("favourite", h2h[h2h["prob"] >= 0.5]), ("underdog", h2h[h2h["prob"] < 0.5])):
            if g.empty:
                continue
            p, a = g["prob"].to_numpy(), g["actual"].to_numpy(dtype=float)
            lines.append(f"| {label} | {len(g)} | {log_loss(p, a):.4f} | {brier_score(p, a):.4f} "
                         f"| {p.mean():.3f} | {a.mean():.3f} |")
        lines.append("")

    curve = calibration_curve(p_all, a_all)
    lines.append("Probability-bucket reliability (all markets):")
    lines.append("")
    lines.append("| bucket | n | mean pred | actual rate |")
    lines.append("|---|---|---|---|")
    for _, row in curve.iterrows():
        lines.append(f"| {row['bucket']} | {int(row['n'])} | {row['mean_pred']:.3f} | {row['actual_rate']:.3f} |")
    lines.append("")
    lines.append(marker_end)
    return "\n".join(lines) + "\n"


def _write_calibration_summary(out_dir: Path, year: int, round_no: int, graded_df: pd.DataFrame) -> Path:
    """Write/replace this round's section in reports/calibration_summary.md
    (audit Phase 3, task 2). Re-grading a round regenerates its own section
    in place rather than appending a duplicate."""
    summary_path = out_dir / "calibration_summary.md"
    section = _format_calibration_section(year, round_no, graded_df)
    marker_start = f"<!-- calibration-summary:{year}:{round_no}:start -->"
    marker_end = f"<!-- calibration-summary:{year}:{round_no}:end -->"
    if summary_path.exists():
        existing = summary_path.read_text(encoding="utf-8")
    else:
        existing = ("# Calibration summary\n\nPer-round grading breakdown from `grade-round` "
                    "(audit Phase 3 follow-up). Re-grading a round replaces its section in "
                    "place -- never duplicated.\n")
    if marker_start in existing:
        start_idx = existing.index(marker_start)
        end_idx = existing.index(marker_end) + len(marker_end)
        tail = existing[end_idx:].lstrip("\n")
        existing = existing[:start_idx] + section + ("\n" + tail if tail else "")
    else:
        existing = existing.rstrip("\n") + "\n\n" + section
    atomic_write_text(summary_path, existing)
    return summary_path


def grade_round(year: int, round_no: int) -> None:
    """Grade a completed round (§10.5): score every saved prediction against
    what actually happened, append to reports/calibration_log.csv, and print the
    round + cumulative calibration. Feeds Section 2's calibration work."""
    out_dir = ROOT_DIR / "reports"
    pred_path = out_dir / f"{year}_r{round_no}_predictions.csv"
    if not pred_path.exists():
        print(f"No predictions file {pred_path}; run round-report for {year} r{round_no} first.",
              file=sys.stderr)
        return
    preds = pd.read_csv(pred_path)

    client = SquiggleClient()
    games = client.get_completed_games(year)
    games = games[games["round"] == round_no]
    if games.empty:
        print(f"{year} round {round_no} is not completed yet — nothing to grade.", file=sys.stderr)
        return
    # Actual player stats for the round, matched by the REAL round number.
    # Past seasons: Fryzigg raw `match_round` (its to_player_log round is a
    # chronological ordinal, §7.2, and its unixtime is unreliable). Current
    # season: DFS, which carries the real round via the Squiggle join.
    player_round = pd.DataFrame(columns=["player", "disposals", "goals", "marks", "tackles"])
    try:
        from afl_bot.data.fryzigg import fetch_fryzigg_player_stats
        raw = fetch_fryzigg_player_stats()
        raw = raw.assign(_year=pd.to_datetime(raw["match_date"]).dt.year,
                         _player=(raw["player_first_name"].str.strip() + " "
                                  + raw["player_last_name"].str.strip()))
        rnd = raw[(raw["_year"] == year) & (raw["match_round"].astype(str) == str(round_no))]
        if not rnd.empty:
            player_round = rnd.rename(columns={"_player": "player"})
    except Exception as exc:  # noqa: BLE001 - fryzigg is one of two fallback sources, must not abort grading
        print(f"WARNING: grade_round fryzigg fetch failed ({exc!r}); falling back to DFS Australia.",
              file=sys.stderr)
    if player_round.empty:
        try:
            from afl_bot.data.dfs_australia import fetch_player_stats
            from afl_bot.data.dfs_australia import to_player_log as _dfs_to_log
            dfs = _dfs_to_log(fetch_player_stats(), games)
            player_round = dfs[dfs["round"] == round_no]
        except Exception as exc:  # noqa: BLE001 - both player-stat sources exhausted, grade H2H/totals only
            print(f"WARNING: grade_round DFS Australia fetch failed ({exc!r}); "
                  f"player-prop predictions for {year} round {round_no} cannot be graded.",
                  file=sys.stderr)

    # actual H2H/total per match (totals keyed by match_id)
    h2h_actual, total_actual = {}, {}
    for _, g in games.iterrows():
        h2h_actual[g["hteam"]] = int(g["hscore"] > g["ascore"])
        h2h_actual[g["ateam"]] = int(g["ascore"] > g["hscore"])
        total_actual[f"{year}_r{round_no}_{g['hteam']}_v_{g['ateam']}"] = g["hscore"] + g["ascore"]
    player_stat = {  # (player, stat) -> actual value that round
        (r["player"], stat): r[stat]
        for _, r in player_round.iterrows() for stat in ("disposals", "goals", "marks", "tackles")
    }

    graded = []
    for _, p in preds.iterrows():
        market, subject, line = p["market"], p["subject"], p["line"]
        if market == "h2h":
            if subject not in h2h_actual:
                continue
            actual = h2h_actual[subject]
        elif market == "total_points":
            tot = total_actual.get(p["match_id"])
            actual = int(tot >= float(line)) if tot is not None else None
        elif market.startswith("player_"):
            stat = market.split("_", 1)[1]
            val = player_stat.get((subject, stat))
            actual = int(val >= float(line)) if val is not None else None
        else:
            actual = None
        if actual is None:
            continue
        graded.append({"year": year, "round": round_no, "market": market, "subject": subject,
                       "line": line, "prob": float(p["prob"]), "actual": actual})

    if not graded:
        print("No predictions could be matched to actuals (player names/rounds).", file=sys.stderr)
        return
    graded_df = pd.DataFrame(graded)

    # AUDIT FIX 2026-07-31: data-staleness guard. When the player-stats cache
    # doesn't cover the round being graded, every player-prop prediction is
    # silently skipped and the round "grades" on H2H/totals alone (this is how
    # 2026 R18-R20 ended up with 24/15/27 graded rows out of 700-1300
    # predictions). Say so loudly instead of silently under-grading.
    n_props_pred = int(preds["market"].astype(str).str.startswith("player_").sum())
    n_props_graded = int(graded_df["market"].astype(str).str.startswith("player_").sum())
    if n_props_pred > 0 and n_props_graded == 0:
        print(f"WARNING: {n_props_pred} player-prop predictions could not be graded — "
              f"the player-stats cache has no round-{round_no} data. Refresh it "
              f"(e.g. force-refresh DFS Australia player stats) and re-run grade-round.",
              file=sys.stderr)

    log_path = out_dir / "calibration_log.csv"
    if log_path.exists():
        prev = pd.read_csv(log_path)
        # AUDIT FIX 2026-07-31: dedupe on (year, round), not round alone — the
        # old filter dropped OTHER seasons' same-numbered rounds from the log
        # (grading 2026 R1 would have silently wiped the 2025 R1 record).
        if "round" in prev.columns and "year" in prev.columns:
            prev = prev[~((prev["year"] == year) & (prev["round"] == round_no))]
        combined = pd.concat([prev, graded_df], ignore_index=True)
    else:
        combined = graded_df
    _cal_buf = io.StringIO()
    combined.to_csv(_cal_buf, index=False)
    atomic_write_text(log_path, _cal_buf.getvalue())

    from afl_bot.backtest.walkforward import brier_score, log_loss
    probs = graded_df["prob"].to_numpy()
    actuals = graded_df["actual"].to_numpy(dtype=float)
    print(f"=== Graded {year} Round {round_no}: {len(graded_df)} predictions ===")
    print(f"  log loss {log_loss(probs, actuals):.4f} | brier {brier_score(probs, actuals):.4f} "
          f"| mean pred {probs.mean():.3f} | hit rate {actuals.mean():.3f}")
    cum_probs = combined["prob"].to_numpy()
    cum_act = combined["actual"].to_numpy(dtype=float)
    print(f"  cumulative ({len(combined)} preds across {combined['round'].nunique()} rounds): "
          f"log loss {log_loss(cum_probs, cum_act):.4f} | brier {brier_score(cum_probs, cum_act):.4f}")
    print(f"  [appended to {log_path}]")

    # AUDIT PHASE 3 (task 2): per-market/H2H-split/reliability breakdown,
    # reporting only -- does not touch calibration_log.csv above.
    summary_path = _write_calibration_summary(out_dir, year, round_no, graded_df)
    print(f"  [calibration breakdown written to {summary_path}]")

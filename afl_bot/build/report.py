"""
Weekly round report (round-2 §10) - the actual product: per match, real-player
projection tables and same-game multis ranked by the *joint* sim probability,
saved to a file.

Pure rendering/search helpers here; the simulation orchestration lives in
``afl_bot.cli.round_report`` (it reuses the same per-team sample engine as
run-round). SGM joint probabilities come from the per-iteration leg masks
(``afl_bot.build.multi.joint_prob_from_masks``), so combos are priced off the
correlated sim, not the naive product.
"""

from __future__ import annotations

from itertools import combinations

from afl_bot.build.multi import LegCandidate, _no_conflicts, combined_odds, joint_prob_from_masks
from afl_bot.config import MULTI_MARKET_SHRINK
from afl_bot.pricing.edge import fair_odds, market_anchored_prob


def projection_rows(player_samples: dict[str, dict], lines: dict[str, list],
                    calibrators: dict | None = None) -> list[dict]:
    """One row per player: projected mean + P(line) for every stat/line, sorted
    by projected disposals (descending). ``calibrators`` (per stat) apply the
    §2.3 per-market calibration to the printed probabilities."""
    calibrators = calibrators or {}
    rows = []
    for player, samples in player_samples.items():
        row: dict = {"player": player}
        for stat, stat_lines in lines.items():
            arr = samples.get(stat)
            if arr is None:
                continue
            row[f"{stat}_mean"] = float(arr.mean())
            cal = calibrators.get(stat)
            for line in stat_lines:
                p = float((arr >= line).mean())
                if cal is not None:
                    p = float(cal.predict([p])[0])
                row[f"{stat}_{line}+"] = p
        rows.append(row)
    rows.sort(key=lambda r: r.get("disposals_mean", 0.0), reverse=True)
    return rows


# Combined-odds bands for the multi ladder (~1.75 -> ~5.0). The top band is the
# "value" rung (MULTI-CHANGES PART B).
DEFAULT_ODDS_BANDS = ((1.75, 2.50), (2.50, 3.50), (3.50, 5.50))


def search_match_sgms(legs: list[LegCandidate], *, min_legs: int = 3, max_legs: int = 3,
                      odds_book: dict | None = None, odds_bands=DEFAULT_ODDS_BANDS,
                      per_band: int = 1, min_joint_prob: float = 0.05,
                      max_plausible_edge: float = 0.15) -> list[dict]:
    """Build a *ladder* of same-game multis spanning the combined-odds bands
    (MULTI-CHANGES PART B). For each ``min_legs``..``max_legs``-leg combo it
    computes the JOINT sim prob (from masks), naive product, correlation gain,
    fair odds, and — when every leg has a book price — book odds + a SHRUNK edge.

    Edge is *not* the raw ``joint*book-1``: per-leg model overestimates compound
    multiplicatively across a multi, so the joint is first pulled toward the
    book's implied multi prob via ``market_anchored_prob`` (``MULTI_MARKET_SHRINK``,
    round-2 §8.2) before the edge. A shrunk edge above ``max_plausible_edge``
    (default 15%) is treated as "model is wrong, not the book" and is NOT eligible
    to be the value pick.

    Combos are bucketed by their banding odds (book odds if priced, else fair
    odds); each band emits ``per_band`` multis:
      * lower/mid bands -> highest joint probability (safest at that price);
      * the TOP band -> highest shrunk *edge* in (0, max_plausible_edge], tagged
        ``value_pick``; with no qualifying priced combo it falls back to highest
        joint prob and stays ``value_pick=False`` (no value claim without a market).

    Every band is guaranteed a rung: an empty band is filled from the remaining
    pool with the combo whose banding odds are closest to the band midpoint
    (MULTI-CHANGES PART B4), so each game shows a full ladder. Returned
    safest -> longest (by odds)."""
    odds_book = odds_book or {}
    combos: list[dict] = []
    for r in range(min_legs, max_legs + 1):
        for combo in combinations(legs, r):
            legs_list = list(combo)
            if not _no_conflicts(legs_list):
                continue
            joint = joint_prob_from_masks(legs_list)
            if joint < min_joint_prob:
                continue
            naive = 1.0
            for leg in legs_list:
                naive *= leg.fair_prob

            entry = {
                "legs": [leg.name for leg in legs_list],
                "joint_prob": joint,
                "naive_product": naive,
                "corr_gain": joint - naive,
                "fair_odds": fair_odds(joint),
                "value_pick": False,
            }
            if all(odds_book.get(leg.name) is not None for leg in legs_list):
                book = combined_odds(legs_list)
                shrunk = market_anchored_prob(joint, book, MULTI_MARKET_SHRINK)
                entry["book_odds"] = book
                entry["raw_edge"] = joint * book - 1.0           # un-shrunk, for reference
                entry["edge"] = shrunk * book - 1.0              # shrunk; used everywhere
            entry["odds"] = entry.get("book_odds", entry["fair_odds"])  # banding odds
            combos.append(entry)

    if not combos:
        return []

    selected: list[dict] = []
    chosen: set[int] = set()
    top_band = odds_bands[-1]
    for band in odds_bands:
        lo, hi = band
        in_band = [c for c in combos if id(c) not in chosen and lo <= c["odds"] < hi]
        if in_band:
            if band == top_band:
                valued = [c for c in in_band if c.get("edge") is not None
                          and 0.0 < c["edge"] <= max_plausible_edge]
                if valued:
                    valued.sort(key=lambda c: c["edge"], reverse=True)
                    pick = valued[:per_band]
                    for c in pick:
                        c["value_pick"] = True
                else:
                    in_band.sort(key=lambda c: c["joint_prob"], reverse=True)
                    pick = in_band[:per_band]   # no plausible market edge -> not value
            else:
                in_band.sort(key=lambda c: c["joint_prob"], reverse=True)
                pick = in_band[:per_band]
        else:
            # B4: fill the empty band from the leftover pool, closest to midpoint.
            mid = (lo + hi) / 2.0
            remaining = [c for c in combos if id(c) not in chosen]
            pick = [min(remaining, key=lambda c: abs(c["odds"] - mid))] if remaining else []
        for c in pick:
            chosen.add(id(c))
        selected.extend(pick)

    selected.sort(key=lambda c: c["odds"])     # safest -> longest by odds
    return selected


def _fmt_pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def render_markdown(year: int, round_no: int, matches: list[dict], *,
                    has_odds: bool, multis_section: str = "", odds_note: str = "") -> str:
    """Render the round report to markdown. ``matches`` is a list of dicts from
    ``afl_bot.cli.round_report`` (header, team projection rows, sgms)."""
    out: list[str] = [f"# AFL Round Report - {year} Round {round_no}", ""]
    out.append("_Real-player projections + same-game multis priced off the "
               "correlated Monte Carlo sim (joint probability, not naive product)._")
    if odds_note:
        out.append("")
        out.append(odds_note)
    out.append("")

    for m in matches:
        h = m["header"]
        wet = " |**WET**" if h["is_wet"] else (" |roofed" if h["roofed"] else "")
        out.append(f"## {h['home']} vs {h['away']} - {h['venue']}{wet}")
        out.append(f"- Margin (home): **{h['mu_margin']:+.1f}** |Total: **{h['mu_total']:.0f}**")
        out.append(f"- P({h['home']}) = **{_fmt_pct(h['p_home'])}** |"
                   f"P({h['away']}) = **{_fmt_pct(h['p_away'])}**"
                   + (f" |P(draw) {_fmt_pct(h['p_draw'])}" if h["p_draw"] > 0 else ""))
        out.append(f"- {h['total_line_name']} = **{_fmt_pct(h['p_total'])}**")
        out.append("")

        for team, rows in m["projections"]:
            if not rows:
                continue
            out.append(f"### {team} - player projections")
            out.append("| Player | Disp | 15+ | 20+ | 25+ | Goals | 1+ | 2+ | Marks | 4+ | Tackles | 3+ |")
            out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
            for row in rows:
                out.append(
                    f"| {row['player']} "
                    f"| {row.get('disposals_mean', 0):.1f} | {_fmt_pct(row.get('disposals_15+', 0))} "
                    f"| {_fmt_pct(row.get('disposals_20+', 0))} | {_fmt_pct(row.get('disposals_25+', 0))} "
                    f"| {row.get('goals_mean', 0):.1f} | {_fmt_pct(row.get('goals_1+', 0))} "
                    f"| {_fmt_pct(row.get('goals_2+', 0))} "
                    f"| {row.get('marks_mean', 0):.1f} | {_fmt_pct(row.get('marks_4+', 0))} "
                    f"| {row.get('tackles_mean', 0):.1f} | {_fmt_pct(row.get('tackles_3+', 0))} |"
                )
            out.append("")

        out.append("### Same-game multi ladder (3-leg, ~1.75 -> ~5.0; top rung = value pick)")
        if not m["sgms"]:
            n_legs = m.get("n_legs")
            if n_legs is not None and n_legs < 3:
                out.append(f"_Only {n_legs} candidate legs for this match - need >=3 "
                           "non-conflicting for a multi (likely missing lineup/odds, not a drop)._")
            else:
                out.append("_No 3-leg combination cleared the probability floor for this match._")
        else:
            header = "| Legs | Joint prob | Fair odds | Corr gain |"
            sep = "|---|--:|--:|--:|"
            if has_odds:
                header += " Book | Edge |"
                sep += "--:|--:|"
            header += " Pick |"
            sep += "---|"
            out.append(header)
            out.append(sep)
            for s in m["sgms"]:
                line = (f"| {' + '.join(s['legs'])} | {_fmt_pct(s['joint_prob'])} "
                        f"| {s['fair_odds']:.2f} | {s['corr_gain'] * 100:+.1f}pp |")
                if has_odds:
                    if "book_odds" in s:
                        line += f" {s['book_odds']:.2f} | {s['edge'] * 100:+.1f}% |"
                    else:
                        line += " - | - |"
                line += " **VALUE PICK** |" if s.get("value_pick") else "  |"
                out.append(line)
        out.append("")

    if multis_section:
        out.append(multis_section)
    out.append("\n_Modelling tool only - even a well-calibrated model loses regularly. "
               "Gambling Help Online 1800 858 858._")
    return "\n".join(out)

"""
Cross-source player identity (round-2 §7.1).

Player rows are keyed by display name, but Fryzigg builds "First Last" and DFS
ships its own string — if a name renders differently across sources ("Bailey
Smith" vs "B. Smith", hyphens, suffixes) that player's history silently splits
in two. Fryzigg carries a stable ``player_id`` (now passed through
``afl_bot.data.fryzigg.to_player_log``), so it's the natural primary key; DFS
names are mapped onto it via a normalised-name lookup.

``validate_source_overlap`` is the A1.2 check: for any season present in BOTH
sources, assert per-player-game disposals/goals/marks/tackles agree. There is no
real overlap yet (the cached Fryzigg ends at 2025, DFS serves 2026), so it runs
once Fryzigg updates or a DFS season is snapshotted into the past — until then
it reports ``n_overlap == 0``. AFL Tables (afltables.com) is the tie-break for
names that don't map.
"""

from __future__ import annotations

import re

import pandas as pd

_SUFFIXES = (" jr", " jnr", " snr", " sr", " ii", " iii")
STATS = ("disposals", "goals", "marks", "tackles")


def normalise_player_name(name: str) -> str:
    """Canonical name key for cross-source matching: lowercased, punctuation
    (apostrophes/periods/hyphens) flattened, whitespace collapsed, common
    generational suffixes stripped."""
    s = str(name).lower().strip()
    s = s.replace("'", "").replace(".", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return s


def build_player_id_map(fryzigg_log: pd.DataFrame) -> dict[str, int]:
    """``{normalised_name: player_id}`` from a Fryzigg player log (its
    ``player_id`` is the canonical key). Empty if the column is absent."""
    if fryzigg_log.empty or "player_id" not in fryzigg_log.columns:
        return {}
    seen = fryzigg_log.dropna(subset=["player_id"]).drop_duplicates("player")
    return {normalise_player_name(p): int(pid)
            for p, pid in zip(seen["player"], seen["player_id"])}


def attach_player_id(log: pd.DataFrame, id_map: dict[str, int]) -> pd.DataFrame:
    """Attach a canonical ``player_id`` (from ``build_player_id_map``) to any
    player log by normalised name. Unmapped players get ``<NA>``."""
    out = log.copy()
    out["player_id"] = out["player"].map(lambda p: id_map.get(normalise_player_name(p)))
    return out


def validate_source_overlap(fryzigg_log: pd.DataFrame, dfs_log: pd.DataFrame,
                            stats=STATS, tol: float = 0.0) -> dict:
    """For seasons present in BOTH logs, join player-games on
    (year, round, normalised name) and check the counting stats agree within
    ``tol`` (A1.2). Returns the overlap count and a DataFrame of any mismatches."""
    if fryzigg_log.empty or dfs_log.empty:
        return {"n_overlap": 0, "n_mismatch": 0, "mismatches": pd.DataFrame()}

    common = set(fryzigg_log["year"]) & set(dfs_log["year"])
    if not common:
        return {"n_overlap": 0, "n_mismatch": 0, "mismatches": pd.DataFrame()}

    def _key(df):
        d = df[df["year"].isin(common)].copy()
        d["_name"] = d["player"].map(normalise_player_name)
        return d

    fz, dfs = _key(fryzigg_log), _key(dfs_log)
    merged = fz.merge(dfs, on=["year", "round", "_name"], suffixes=("_fz", "_dfs"))
    if merged.empty:
        return {"n_overlap": 0, "n_mismatch": 0, "mismatches": pd.DataFrame()}

    disagree = pd.Series(False, index=merged.index)
    for stat in stats:
        if f"{stat}_fz" in merged and f"{stat}_dfs" in merged:
            disagree |= (merged[f"{stat}_fz"] - merged[f"{stat}_dfs"]).abs() > tol
    mismatches = merged.loc[disagree, ["year", "round", "_name", *[f"{s}_fz" for s in stats],
                                       *[f"{s}_dfs" for s in stats]]]
    return {"n_overlap": len(merged), "n_mismatch": int(disagree.sum()), "mismatches": mismatches}

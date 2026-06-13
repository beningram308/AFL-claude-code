"""
Confirmed team lineups (Fable round-2 §1.2).

A priced player who isn't actually named in the team for a game makes every prop
on him worthless and crowds a real player out of the pool. The official AFL
lineup feed (aflapi.afl.com.au, what fitzRoy's ``fetch_lineup`` wraps) is
media-token-gated — the same wall the detailed stoppage stats hit — so the
reliable free path is the team sheets that drop ~Thursday evening, supplied as a
small JSON (the same hand-entered interchange pattern as ``--odds``):

    {"Geelong": ["Patrick Dangerfield", "Bailey Smith", ...],
     "Carlton": ["Patrick Cripps", ...]}

``load_lineup`` reads that file (keys starting with ``_`` are ignored, e.g. a
future ``_rules`` block) and normalises team names. ``run_round`` then marks any
priced player not named ``confirmed=False`` — the multi builder already excludes
those (``afl_bot.build.multi.usable_legs``). With no file, every player stays
confirmed (unchanged behaviour).
"""

from __future__ import annotations

import json
from pathlib import Path

from afl_bot.data.teams import normalize_team_name


def load_lineup(path: str | None) -> dict[str, set[str]]:
    """Load ``{team: {player, ...}}`` from a confirmed-lineup JSON file, or
    return ``{}`` when no path is given. Unknown team names are skipped rather
    than raising, so a partial/typo'd sheet still gates the teams it does name."""
    if not path:
        return {}

    data = json.loads(Path(path).read_text())
    lineup: dict[str, set[str]] = {}
    for team, players in data.items():
        if team.startswith("_"):
            continue
        try:
            canonical = normalize_team_name(team)
        except KeyError:
            continue
        lineup[canonical] = {str(p).strip() for p in players}
    return lineup

"""
Boundary throw-in (out-of-bounds) counts per game — the data plug-point for the
OOB market model (plan §1.6).

Boundary-throw-in counts are NOT on Squiggle or AFL Tables, and the AFL.com.au
match-centre detailed team stats (the Champion-Data-fed feed that carries
stoppage breakdowns) are token-gated — ``aflapi.afl.com.au/afl/v2/matches``
returns fixtures, but the per-match team-stats endpoints 404 without a media
token, and the full event/chain feed needs a commercial Champion Data licence
(https://docs.api.afl.championdata.com/). So this is the explicit precondition
the build order names ("once team stoppage data flows"): the *model*
(``afl_bot.models.stoppages``) is ready and prices off a league prior, and this
module is where real per-game counts plug in once a feed is wired.

Data contract: a DataFrame with ``[year, round, hteam, ateam, boundary_throwins]``
(``boundary_throwins`` is the whole-game count — a throw-in is a neutral match
event, not a team stat). ``load_boundary_throwins`` accepts such a table (from a
future AFL/Champion Data ingestion or a test), caches it, and otherwise returns
an empty frame so callers fall back to the prior.
"""

from __future__ import annotations

import pandas as pd

from afl_bot.config import CACHE_DIR
from afl_bot.data.storage import read_parquet, write_parquet
from afl_bot.data.teams import normalize_team_name

BOUNDARY_THROWIN_COL = "boundary_throwins"
CACHE_NAME = "boundary_throwins"
SCHEMA_VERSION = 1
_REQUIRED = ["year", "round", "hteam", "ateam", BOUNDARY_THROWIN_COL]


def load_boundary_throwins(source_df: pd.DataFrame | None = None, *,
                           cache_dir=CACHE_DIR) -> pd.DataFrame:
    """Per-game boundary-throw-in counts.

    Pass ``source_df`` (the data contract columns) to ingest a freshly-fetched
    table — it is validated, team names normalised, and cached. With no
    ``source_df`` it returns the cache if present, else an empty frame (signal
    to the model layer to use the league prior).
    """
    if source_df is not None:
        missing = [c for c in _REQUIRED if c not in source_df.columns]
        if missing:
            raise ValueError(f"boundary throw-in table missing columns: {missing}")
        df = source_df[_REQUIRED].copy()
        df["hteam"] = df["hteam"].map(normalize_team_name)
        df["ateam"] = df["ateam"].map(normalize_team_name)
        df = df.sort_values(["year", "round"]).reset_index(drop=True)
        write_parquet(df, CACHE_NAME, schema_version=SCHEMA_VERSION, cache_dir=cache_dir)
        return df

    return read_parquet(CACHE_NAME, expected_schema_version=SCHEMA_VERSION, cache_dir=cache_dir)

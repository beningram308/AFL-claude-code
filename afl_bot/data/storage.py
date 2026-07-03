"""
Schema-versioned parquet data layer (plan §5.1, build-order step 1).

Every dataset (Squiggle games/tips, player box scores, odds, weather, ...) is
cached under ``data_cache/<name>.parquet`` with a small sidecar JSON file
recording a schema version and row count. Loaders bump the version when the
columns they write change shape; ``read_parquet`` warns if a cached file was
written by an older schema version so stale caches don't silently feed
mismatched columns into the pipeline.

A thin DuckDB helper (``duckdb_connection``) registers every cached parquet
file as a view for ad-hoc SQL queries across the cache, e.g.:

    con = duckdb_connection()
    con.sql("SELECT hteam, ateam, hscore, ascore FROM games_2025 LIMIT 5")
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from afl_bot.config import CACHE_DIR


def _meta_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.meta.json"


def _parquet_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def write_parquet(
    df: pd.DataFrame,
    name: str,
    schema_version: int = 1,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """Write ``df`` to ``<cache_dir>/<name>.parquet`` plus a sidecar
    ``<name>.meta.json`` recording the schema version, row count and columns."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(cache_dir, name)
    df.to_parquet(path, index=False)
    from afl_bot.io_utils import atomic_write_text
    atomic_write_text(_meta_path(cache_dir, name), json.dumps({
        "schema_version": schema_version,
        "rows": len(df),
        "columns": list(df.columns),
    }, indent=2))
    return path


def read_parquet(
    name: str,
    expected_schema_version: int | None = None,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Read ``<cache_dir>/<name>.parquet``. If ``expected_schema_version`` is
    given and the sidecar metadata records an older (or missing) version, emit
    a warning so callers know the cache may need refreshing."""
    path = _parquet_path(cache_dir, name)
    if not path.exists():
        return pd.DataFrame()

    if expected_schema_version is not None:
        meta_path = _meta_path(cache_dir, name)
        version = None
        if meta_path.exists():
            try:
                version = json.loads(meta_path.read_text()).get("schema_version")
            except (json.JSONDecodeError, OSError):
                version = None
        if version != expected_schema_version:
            warnings.warn(
                f"Cached '{name}' has schema_version={version!r}, "
                f"expected {expected_schema_version}. Consider refreshing the cache.",
                stacklevel=2,
            )

    return pd.read_parquet(path)


def cached_dataset_names(cache_dir: Path = CACHE_DIR) -> list[str]:
    """Names of all parquet datasets currently in the cache (without extension)."""
    if not cache_dir.exists():
        return []
    return sorted(p.stem for p in cache_dir.glob("*.parquet"))


def duckdb_connection(cache_dir: Path = CACHE_DIR):
    """Return an in-memory DuckDB connection with one view per cached parquet
    file (view name = file stem, e.g. ``games_2025``), for ad-hoc SQL queries
    across the cache.

    Raises ``ImportError`` with a helpful message if ``duckdb`` isn't
    installed -- it's an optional dependency for ad-hoc analysis only, nothing
    in the core pipeline requires it.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "duckdb is required for duckdb_connection(); install it with "
            "`pip install duckdb`."
        ) from exc

    con = duckdb.connect(database=":memory:")
    for name in cached_dataset_names(cache_dir):
        path = _parquet_path(cache_dir, name)
        con.execute(
            f'CREATE VIEW "{name}" AS SELECT * FROM read_parquet(?)', [str(path)]
        )
    return con

import json

import pandas as pd
import pytest

from afl_bot.data.storage import (
    cached_dataset_names,
    read_parquet,
    write_parquet,
)


def test_write_then_read_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    write_parquet(df, "thing", schema_version=3, cache_dir=tmp_path)

    out = read_parquet("thing", expected_schema_version=3, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(out, df)

    meta = json.loads((tmp_path / "thing.meta.json").read_text())
    assert meta["schema_version"] == 3
    assert meta["rows"] == 2
    assert meta["columns"] == ["a", "b"]


def test_read_missing_returns_empty_dataframe(tmp_path):
    out = read_parquet("missing", cache_dir=tmp_path)
    assert out.empty


def test_read_warns_on_schema_mismatch(tmp_path):
    df = pd.DataFrame({"a": [1]})
    write_parquet(df, "thing", schema_version=1, cache_dir=tmp_path)

    with pytest.warns(UserWarning, match="schema_version"):
        read_parquet("thing", expected_schema_version=2, cache_dir=tmp_path)


def test_cached_dataset_names(tmp_path):
    write_parquet(pd.DataFrame({"a": [1]}), "alpha", cache_dir=tmp_path)
    write_parquet(pd.DataFrame({"a": [1]}), "beta", cache_dir=tmp_path)
    assert cached_dataset_names(tmp_path) == ["alpha", "beta"]


def test_cached_dataset_names_missing_dir(tmp_path):
    assert cached_dataset_names(tmp_path / "nope") == []

"""Tests for atomic_write_text (io_utils.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from afl_bot.io_utils import atomic_write_text


def test_atomic_write_creates_file(tmp_path):
    p = tmp_path / "out.txt"
    atomic_write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"


def test_atomic_write_overwrites_existing(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("old", encoding="utf-8")
    atomic_write_text(p, "new")
    assert p.read_text(encoding="utf-8") == "new"


def test_atomic_write_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "out.txt"
    atomic_write_text(p, "nested")
    assert p.read_text(encoding="utf-8") == "nested"


def test_atomic_write_no_tmp_file_left_on_success(tmp_path):
    p = tmp_path / "out.txt"
    atomic_write_text(p, "data")
    tmp_files = list(tmp_path.glob("*.tmp-*"))
    assert tmp_files == [], f"Leftover tmp files: {tmp_files}"


def test_atomic_write_original_untouched_when_write_interrupted(tmp_path, monkeypatch):
    """Simulate a process that writes the tmp file but never calls os.replace.
    The original file must be intact (old content) after the simulated kill."""
    p = tmp_path / "ledger.json"
    p.write_text("original", encoding="utf-8")

    # Monkeypatch os.replace to raise SIGKILL-equivalent: just raise
    import afl_bot.io_utils as _mod
    original_replace = os.replace

    def fake_replace(src, dst):
        # Simulate: tmp was written but process died before replace
        raise OSError("simulated kill")

    monkeypatch.setattr(_mod.os, "replace", fake_replace)

    with pytest.raises(OSError, match="simulated kill"):
        atomic_write_text(p, "new content")

    # Original must still be intact
    assert p.read_text(encoding="utf-8") == "original"
    # Tmp file should have been cleaned up in the except block
    tmp_files = list(tmp_path.glob("*.tmp-*"))
    assert tmp_files == [], f"Leftover tmp after failed replace: {tmp_files}"


def test_atomic_write_utf8_encoding(tmp_path):
    p = tmp_path / "unicode.txt"
    text = "em—dash and café"
    atomic_write_text(p, text)
    assert p.read_bytes() == text.encode("utf-8")
    assert p.read_text(encoding="utf-8") == text


def test_atomic_write_json_roundtrip(tmp_path):
    import json
    p = tmp_path / "data.json"
    payload = {"bet_id": "abc-123", "stake": 25.0, "legs": [1, 2, 3]}
    atomic_write_text(p, json.dumps(payload, indent=2))
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded == payload

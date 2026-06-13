"""Cross-source player identity + overlap validation (round-2 §7.1)."""

import pandas as pd

from afl_bot.data.identity import (
    attach_player_id,
    build_player_id_map,
    normalise_player_name,
    validate_source_overlap,
)


def test_normalise_player_name_handles_punctuation_and_suffix():
    assert normalise_player_name("Bailey Smith") == "bailey smith"
    assert normalise_player_name("  O'Brien-Jones  ") == "obrien jones"
    assert normalise_player_name("Nic Naitanui Jr") == "nic naitanui"
    assert normalise_player_name("J. Kennedy") == "j kennedy"


def test_build_and_attach_player_id():
    fz = pd.DataFrame({"player": ["Bailey Smith", "Max Holmes"], "player_id": [101, 202]})
    id_map = build_player_id_map(fz)
    assert id_map == {"bailey smith": 101, "max holmes": 202}

    dfs = pd.DataFrame({"player": ["BAILEY SMITH", "Unknown Guy"]})
    out = attach_player_id(dfs, id_map)
    assert out.loc[0, "player_id"] == 101          # matched despite case
    assert pd.isna(out.loc[1, "player_id"])         # unmapped -> NA


def test_validate_source_overlap_detects_mismatch():
    fz = pd.DataFrame([
        {"year": 2026, "round": 1, "player": "Bailey Smith", "disposals": 30, "goals": 1, "marks": 5, "tackles": 4},
        {"year": 2026, "round": 1, "player": "Max Holmes", "disposals": 25, "goals": 0, "marks": 4, "tackles": 3},
    ])
    dfs = fz.copy()
    dfs.loc[1, "disposals"] = 19                      # introduce a disagreement
    rep = validate_source_overlap(fz, dfs)
    assert rep["n_overlap"] == 2
    assert rep["n_mismatch"] == 1
    assert rep["mismatches"].iloc[0]["_name"] == "max holmes"


def test_validate_source_overlap_no_common_season():
    fz = pd.DataFrame([{"year": 2025, "round": 1, "player": "X", "disposals": 20,
                        "goals": 0, "marks": 0, "tackles": 0}])
    dfs = pd.DataFrame([{"year": 2026, "round": 1, "player": "X", "disposals": 20,
                         "goals": 0, "marks": 0, "tackles": 0}])
    assert validate_source_overlap(fz, dfs)["n_overlap"] == 0

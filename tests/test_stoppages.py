import numpy as np
import pandas as pd
import pytest

from afl_bot.config import LEAGUE_OOB_PER_GAME
from afl_bot.data.stoppages import BOUNDARY_THROWIN_COL, load_boundary_throwins
from afl_bot.models.stoppages import expected_oob, simulate_boundary_throwins

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Data plug-point
# --------------------------------------------------------------------------- #
def test_load_boundary_throwins_ingests_validates_and_caches(tmp_path):
    src = pd.DataFrame([
        {"year": 2024, "round": 1, "hteam": "Geelong", "ateam": "Carlton", "boundary_throwins": 33},
        {"year": 2024, "round": 1, "hteam": "Sydney", "ateam": "Essendon", "boundary_throwins": 41},
    ])
    out = load_boundary_throwins(src, cache_dir=tmp_path)
    assert list(out.columns) == ["year", "round", "hteam", "ateam", BOUNDARY_THROWIN_COL]
    assert len(out) == 2

    # cached: a no-source call returns the stored table
    cached = load_boundary_throwins(cache_dir=tmp_path)
    assert len(cached) == 2


def test_load_boundary_throwins_missing_columns_raises(tmp_path):
    with pytest.raises(ValueError):
        load_boundary_throwins(pd.DataFrame([{"year": 2024, "round": 1}]), cache_dir=tmp_path)


def test_load_boundary_throwins_empty_when_no_feed(tmp_path):
    assert load_boundary_throwins(cache_dir=tmp_path).empty


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_expected_oob_uses_prior_without_data_and_mean_with_data():
    assert expected_oob(None) == LEAGUE_OOB_PER_GAME
    assert expected_oob(pd.DataFrame()) == LEAGUE_OOB_PER_GAME
    log = pd.DataFrame({BOUNDARY_THROWIN_COL: [30, 40, 50]})
    assert expected_oob(log) == 40.0


def test_simulate_boundary_throwins_mean_near_prior_and_negative_total_corr():
    total = rng.normal(162, 31, 200_000).clip(60)
    oob = simulate_boundary_throwins(36.0, total, rng)
    assert abs(oob.mean() - 36.0) < 2.0                       # marginal mean ~ prior
    assert np.corrcoef(oob, total)[0, 1] < -0.2               # congestion: OOB up when total down
    assert (oob >= 0).all()


def test_simulate_boundary_throwins_greasy_lifts_count():
    total = rng.normal(162, 31, 200_000).clip(60)
    dry = simulate_boundary_throwins(36.0, total, rng, greasiness=0.0)
    wet = simulate_boundary_throwins(36.0, total, rng, greasiness=1.0)
    assert wet.mean() > dry.mean() * 1.1                      # rain raises OOB

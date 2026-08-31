from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

import RamanPhaseID_0p99beta as app
import raman_core as rc


def _synthetic_spectrum(x: np.ndarray) -> np.ndarray:
    baseline = 0.15 + 0.000002 * (x - 280.0) ** 2
    peak_a = 1.2 * np.exp(-0.5 * ((x - 170.0) / 9.0) ** 2)
    peak_b = 0.7 * np.exp(-0.5 * ((x - 335.0) / 14.0) ** 2)
    return baseline + peak_a + peak_b


def test_measurement_preprocessing_is_stable_across_native_spacing() -> None:
    dense_x = np.arange(80.0, 501.0, 0.5)
    sparse_x = np.arange(80.0, 501.0, 2.0)
    target_x = np.arange(80.0, 501.0, 1.0)
    baseline_cfg = app._default_baseline_cfg()
    smoothing_cfg = {"enabled": True, "window": 11, "poly": 3}

    dense = app._process_measurement(
        dense_x,
        _synthetic_spectrum(dense_x),
        apply_baseline=True,
        baseline_cfg=baseline_cfg,
        smoothing_cfg=smoothing_cfg,
        target_x=target_x,
    )
    sparse = app._process_measurement(
        sparse_x,
        _synthetic_spectrum(sparse_x),
        apply_baseline=True,
        baseline_cfg=baseline_cfg,
        smoothing_cfg=smoothing_cfg,
        target_x=target_x,
    )

    # Remaining differences are interpolation error in the sparse observation,
    # not a change in lambda or in the physical Savitzky-Golay span.
    assert np.corrcoef(dense, sparse)[0, 1] > 0.9999
    assert float(np.max(np.abs(dense - sparse))) < 0.005


def test_database_vectors_are_not_savgol_smoothed() -> None:
    grid = np.arange(100.0, 131.0, 1.0)
    impulse = np.zeros_like(grid)
    impulse[15] = 1.0
    baseline_cfg = app._fixed_db_baseline_cfg()

    raw = app._prepare_db_signal_on_target_grid(
        grid,
        impulse,
        grid,
        apply_baseline_db=False,
        baseline_cfg=baseline_cfg,
    )
    normalized = app._process_db_on_target_grid(
        grid,
        impulse,
        grid,
        apply_baseline_db=False,
        baseline_cfg=baseline_cfg,
    )

    np.testing.assert_array_equal(raw, impulse)
    np.testing.assert_array_equal(normalized, impulse)


def test_database_baseline_is_stable_across_native_spacing() -> None:
    dense_x = np.arange(80.0, 501.0, 0.5)
    sparse_x = np.arange(80.0, 501.0, 2.0)
    target_x = np.arange(80.0, 501.0, 1.0)
    cfg = app._fixed_db_baseline_cfg()

    dense = app._process_db_on_target_grid(
        dense_x,
        _synthetic_spectrum(dense_x),
        target_x,
        apply_baseline_db=True,
        baseline_cfg=cfg,
    )
    sparse = app._process_db_on_target_grid(
        sparse_x,
        _synthetic_spectrum(sparse_x),
        target_x,
        apply_baseline_db=True,
        baseline_cfg=cfg,
    )

    assert np.corrcoef(dense, sparse)[0, 1] > 0.9999
    assert float(np.max(np.abs(dense - sparse))) < 0.007


def test_legacy_database_worker_never_calls_smoothing() -> None:
    grid = np.arange(100.0, 131.0, 1.0)
    impulse = np.zeros_like(grid)
    impulse[15] = 1.0
    entry = {"path": Path("reference.txt")}

    with (
        patch.object(rc, "_parse_rruff", return_value=(grid, impulse)),
        patch.object(rc, "_smooth", side_effect=AssertionError("DB smoothing called")),
    ):
        similarity, returned_entry = rc._similarity_worker(entry, grid, impulse)

    assert similarity == 1.0
    assert returned_entry is entry


def test_query_range_is_only_a_post_preprocessing_mask() -> None:
    x = np.arange(80.0, 501.0, 0.5)
    y = _synthetic_spectrum(x)
    grid = np.arange(60.0, 551.0, 1.0)
    cfg = app._default_baseline_cfg()
    smoothing = {"enabled": True, "window": 11, "poly": 3}

    wide, _wide_l2, wide_mask = app._prepare_query_vector(
        x,
        y,
        120,
        420,
        grid,
        apply_baseline=True,
        baseline_cfg=cfg,
        smoothing_cfg=smoothing,
    )
    narrow, _narrow_l2, narrow_mask = app._prepare_query_vector(
        x,
        y,
        200,
        300,
        grid,
        apply_baseline=True,
        baseline_cfg=cfg,
        smoothing_cfg=smoothing,
    )

    np.testing.assert_allclose(wide[narrow_mask], narrow[narrow_mask], rtol=0.0, atol=0.0)
    assert np.all(narrow[~narrow_mask] == 0.0)
    assert np.count_nonzero(wide_mask) > np.count_nonzero(narrow_mask)

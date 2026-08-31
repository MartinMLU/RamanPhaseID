from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from scipy.signal import savgol_filter

import RamanPhaseID_0p99beta as app
import raman_ai_denoiser as ai
import raman_preprocessing as preprocessing


def test_deeper_identity_runner_preserves_a_linear_spectrum() -> None:
    x = np.arange(400.0, 1901.0, 1.0)
    y = 2.0 + 0.003 * x
    seen: list[np.ndarray] = []

    def identity_runner(values: np.ndarray) -> np.ndarray:
        seen.append(values.copy())
        return values

    result = ai.denoise(x, y, runner=identity_runner)

    np.testing.assert_allclose(result, y, rtol=0.0, atol=1e-12)
    assert len(seen) == 2
    for window in seen:
        assert window.shape == (ai.MODEL_POINTS,)
        assert float(np.min(window)) == pytest.approx(0.0)
        assert float(np.max(window)) == pytest.approx(1.0)


def test_guarded_deeper_uses_points_below_500_and_above_1800() -> None:
    x = np.arange(60.0, 2501.0, 1.0)
    y = 1.0 + (0.0002 * x) + (0.08 * np.where(np.arange(x.size) % 2, -1.0, 1.0))

    result = ai.denoise(x, y, runner=lambda values: np.zeros_like(values))

    assert np.any(result[x < 500.0] != y[x < 500.0])
    assert np.any(result[x > 1800.0] != y[x > 1800.0])
    assert np.any(result[(x >= 500.0) & (x <= 1800.0)] != y[(x >= 500.0) & (x <= 1800.0)])


def test_guarded_deeper_accepts_a_short_arbitrary_range() -> None:
    x = np.arange(100.0, 901.0, 1.0)
    y = np.sin(x / 100.0) + (0.02 * np.sin(x * 2.0))

    result = ai.denoise(x, y, runner=lambda values: values)

    assert result.shape == y.shape
    assert np.all(np.isfinite(result))
    cap = min(
        ai.DEFAULT_MAX_CHANGE_SIGMA * ai.estimate_noise_sigma(x, y),
        ai.GUARD_MAX_DYNAMIC_RANGE_FRACTION * float(np.ptp(y)),
    )
    assert float(np.max(np.abs(result - y))) <= cap + 1e-12


def test_hallucinated_background_and_missing_peaks_are_bounded() -> None:
    x = np.arange(60.0, 2501.0, 1.0)
    peaks = (
        8.0 * np.exp(-0.5 * ((x - 410.0) / 5.0) ** 2)
        + 5.0 * np.exp(-0.5 * ((x - 1210.0) / 8.0) ** 2)
        + 6.5 * np.exp(-0.5 * ((x - 2200.0) / 6.0) ** 2)
    )
    y = peaks + (0.05 * np.sin(x * 2.1))

    def hallucinating_runner(_values: np.ndarray) -> np.ndarray:
        channels = np.linspace(0.0, 1.0, ai.MODEL_POINTS)
        return 0.35 + (0.55 * np.exp(-0.5 * ((channels - 0.72) / 0.13) ** 2))

    result = ai.denoise(x, y, runner=hallucinating_runner, max_change_sigma=3.0)
    change = result - y
    sigma = ai.estimate_noise_sigma(x, y)
    expected_cap = min(
        3.0 * sigma,
        ai.GUARD_MAX_DYNAMIC_RANGE_FRACTION * float(np.ptp(y)),
    )

    assert float(np.max(np.abs(change))) <= expected_cap + 1e-12
    # A biological-looking broad model template cannot become a new baseline.
    broad_change = savgol_filter(change, 101, 3)
    assert float(np.max(np.abs(broad_change))) < 0.2 * expected_cap
    # Even when the raw model contains none of the three real peaks, their
    # centres can move only by the conservative pointwise noise cap.
    for centre in (410.0, 1210.0, 2200.0):
        index = int(np.flatnonzero(x == centre)[0])
        assert abs(float(result[index] - y[index])) <= expected_cap + 1e-12


def test_deeper_constant_signal_does_not_load_or_run_model() -> None:
    x = np.arange(400.0, 1901.0, 1.0)
    y = np.full_like(x, 7.5)

    def should_not_run(_values: np.ndarray) -> np.ndarray:
        raise AssertionError("constant input should bypass inference")

    np.testing.assert_array_equal(ai.denoise(x, y, runner=should_not_run), y)


def test_smoothing_payload_separates_all_three_methods_and_legacy_configs() -> None:
    legacy_sg = app._smoothing_cfg_payload({"enabled": True, "window": 11, "poly": 3})
    none = app._smoothing_cfg_payload({"method": "none", "window": 99, "poly": 8})
    deeper = app._smoothing_cfg_payload({"method": "deeper_ai"})

    assert legacy_sg["method"] == "savgol"
    assert legacy_sg["window"] == 11
    assert none == {
        "v": 5,
        "grid_step_cm1": app.PREPROCESS_GRID_STEP_CM1,
        "method": "none",
    }
    assert deeper["method"] == "deeper_ai"
    assert deeper["model_sha256"] == ai.MODEL_SHA256
    assert deeper["model_training_range_cm1"] == [500.0, 1800.0]
    assert deeper["full_range_adapter_v"] == ai.FULL_RANGE_ADAPTER_VERSION
    assert deeper["window_span_cm1"] == 1300.0
    assert deeper["max_change_sigma"] == ai.DEFAULT_MAX_CHANGE_SIGMA
    assert len(
        {
            app._smoothing_cfg_token({"method": "none"}),
            app._smoothing_cfg_token({"method": "savgol", "window": 11, "poly": 3}),
            app._smoothing_cfg_token({"method": "deeper_ai"}),
        }
    ) == 3
    assert app._smoothing_cfg_token(
        {"method": "deeper_ai", "max_change_sigma": 0.5}
    ) != app._smoothing_cfg_token(
        {"method": "deeper_ai", "max_change_sigma": 3.0}
    )


def test_main_preview_labels_follow_the_selected_preprocessing_method() -> None:
    none = app._smoothing_preview_ui({"method": "none"})
    savgol = app._smoothing_preview_ui({"method": "savgol"})
    deeper = app._smoothing_preview_ui({"method": "deeper_ai"})

    assert none["title"] == "Unchanged measurement preview"
    assert none["spectrum_label"] == "Unchanged spectrum"
    assert savgol["title"] == "Savitzky–Golay smoothing preview"
    assert savgol["preview_label"] == "Smoothing preview"
    assert savgol["curve_label"] == (
        "smoothed · Savitzky-Golay (window = 5, poly = 3)"
    )
    assert deeper["title"] == "Guarded DeepeR denoising preview"
    assert deeper["curve_label"] == "guarded AI denoising (experimental)"
    assert (
        preprocessing.smoothing_input_curve_label("BC")
        == "baseline-corrected measurement"
    )
    assert preprocessing.processing_difference_curve_label(10.0) == (
        "difference curve; x10"
    )


def test_main_preprocessing_dispatches_ai_method() -> None:
    x = np.arange(500.0, 1801.0, 1.0)
    y = np.linspace(0.0, 1.0, x.size)
    expected = y + 2.0

    with patch.object(app.ai_denoiser, "denoise", return_value=expected) as mocked:
        result = app._apply_smoothing(
            x,
            y,
            {"method": "deeper_ai", "max_change_sigma": 1.75},
        )

    np.testing.assert_array_equal(result, expected)
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["max_change_sigma"] == 1.75


def test_database_preprocessing_never_invokes_ai_denoiser() -> None:
    x = np.arange(500.0, 551.0, 1.0)
    y = np.zeros_like(x)
    y[25] = 1.0

    with patch.object(
        app.ai_denoiser,
        "denoise",
        side_effect=AssertionError("database AI denoising called"),
    ):
        result = app._prepare_db_signal_on_target_grid(
            x,
            y,
            x,
            apply_baseline_db=False,
            baseline_cfg=app._fixed_db_baseline_cfg(),
        )

    np.testing.assert_array_equal(result, y)

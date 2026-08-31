from __future__ import annotations

import numpy as np

import raman_preprocessing as preprocessing


def test_measurement_defaults_change_without_altering_database_baseline() -> None:
    measurement_baseline = preprocessing.default_baseline_settings()
    database_baseline = preprocessing.fixed_database_baseline_settings()
    smoothing = preprocessing.default_smoothing_settings()

    assert measurement_baseline.lam_exp == 5
    assert measurement_baseline.lam == 1.0e5
    assert smoothing.window == 5
    assert smoothing.poly == 3
    assert database_baseline.lam_exp == 4
    assert database_baseline.lam == 1.0e4
    assert preprocessing.BaselineSettings.from_mapping({"lam_exp": 4}).lam == 1.0e4


def test_parsed_measurement_artifact_is_immutable_and_includes_qc() -> None:
    parsed = preprocessing.parse_measurement_text("100 1\n101 2\n102 3\n")

    np.testing.assert_array_equal(parsed.axis_cm1, [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(parsed.intensity, [1.0, 2.0, 3.0])
    assert parsed.quality.finite_point_count == 3
    assert not parsed.axis_cm1.flags.writeable
    assert not parsed.intensity.flags.writeable


def test_banded_arpls_matches_sparse_reference() -> None:
    if not preprocessing.HAVE_SCIPY:
        return
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.special import expit

    rng = np.random.default_rng(17)
    axis = np.linspace(0.0, 1.0, 400)
    signal = 0.2 + 0.6 * axis + np.exp(-0.5 * ((axis - 0.43) / 0.025) ** 2)
    signal += rng.normal(0.0, 0.02, axis.size)

    length = signal.size
    lam = 1.0e4
    difference = sp.diags(
        [1, -2, 1], [0, -1, -2], shape=(length, length - 2), dtype=float
    ).T
    penalty = (lam * (difference.T @ difference)).tocsc()
    weights = np.ones(length)
    sparse_result = signal.copy()
    for _ in range(50):
        sparse_result = spla.spsolve(
            (sp.diags(weights, 0) + penalty).tocsc(), weights * signal
        )
        residual = signal - sparse_result
        negative = residual[residual < 0]
        if negative.size == 0:
            break
        mean = float(negative.mean())
        std = max(float(negative.std()), 1e-12)
        updated = expit(-2.0 * (residual - (2.0 * std - mean)) / std)
        relative = np.linalg.norm(weights - updated) / (np.linalg.norm(weights) + 1e-12)
        weights = updated
        if relative < 1e-3:
            break

    banded_result = preprocessing.baseline_arpls(signal, lam=lam)
    np.testing.assert_allclose(banded_result, sparse_result, rtol=1e-9, atol=1e-9)


def test_large_detector_gap_is_not_interpolated_or_marked_valid() -> None:
    left_x = np.arange(100.0, 201.0)
    right_x = np.arange(400.0, 501.0)
    axis = np.concatenate([left_x, right_x])
    intensity = np.concatenate([np.ones(left_x.size), np.full(right_x.size, 2.0)])

    result = preprocessing.preprocess_spectrum(
        axis,
        intensity,
        apply_baseline=False,
        baseline_settings=preprocessing.BaselineSettings(method="RAW"),
        smoothing_settings=preprocessing.SmoothingSettings(method="none"),
    )
    target = np.arange(100.0, 501.0)
    projected, valid = result.project(target)

    assert len(result.segments) == 2
    assert not result.segments[0].axis_cm1.flags.writeable
    assert not result.segments[0].processed.flags.writeable
    assert not np.any(valid[(target > 200.0) & (target < 400.0)])
    assert np.all(projected[(target > 200.0) & (target < 400.0)] == 0.0)
    assert result.quality.gap_intervals_cm1 == ((200.0, 400.0),)


def test_reference_alignment_reports_partial_overlap() -> None:
    target = np.arange(100.0, 301.0)
    reference_axis = np.arange(150.0, 251.0)
    reference = preprocessing.align_reference_to_target(
        target, reference_axis, np.ones(reference_axis.size)
    )

    assert np.all(reference.values[~reference.overlap_mask] == 0.0)
    assert np.count_nonzero(reference.overlap_mask) == reference_axis.size
    assert 0.49 < reference.overlap_fraction < 0.51
    assert not reference.values.flags.writeable
    assert not reference.overlap_mask.flags.writeable


def test_quality_flags_saturation_and_possible_spike() -> None:
    axis = np.arange(200.0)
    intensity = 10.0 + 0.01 * axis
    intensity[70] = 80.0
    intensity[150:] = 100.0
    quality = preprocessing.assess_axis_quality(axis, intensity)

    assert quality.spike_indices
    assert quality.saturation_fraction >= 0.01
    assert any("saturation" in warning.lower() for warning in quality.warnings)
    assert not any("spike" in warning.lower() for warning in quality.warnings)


def test_negative_baseline_diagnostic_ignores_noise_scale_zero_crossings() -> None:
    rng = np.random.default_rng(20260830)
    axis = np.arange(5000, dtype=float)
    centred_noise = rng.normal(0.0, 1.0, axis.size)

    result = preprocessing.preprocess_spectrum(
        axis,
        centred_noise,
        apply_baseline=False,
        baseline_settings=preprocessing.BaselineSettings(method="RAW"),
        smoothing_settings=preprocessing.SmoothingSettings(method="none"),
    )

    assert 0.45 < result.diagnostics["negative_fraction"] < 0.55
    assert result.diagnostics["material_negative_fraction"] < 0.01
    assert result.diagnostics["material_negative_threshold"] >= (
        3.0 * result.diagnostics["noise_sigma"]
    )


def test_negative_baseline_diagnostic_still_flags_material_offset() -> None:
    rng = np.random.default_rng(17)
    axis = np.arange(2000, dtype=float)
    materially_negative = -10.0 + rng.normal(0.0, 0.2, axis.size)

    result = preprocessing.preprocess_spectrum(
        axis,
        materially_negative,
        apply_baseline=False,
        baseline_settings=preprocessing.BaselineSettings(method="RAW"),
        smoothing_settings=preprocessing.SmoothingSettings(method="none"),
    )

    assert result.diagnostics["material_negative_fraction"] > 0.99

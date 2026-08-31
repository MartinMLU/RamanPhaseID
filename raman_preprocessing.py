"""Typed, testable spectral preprocessing for RamanPhaseID.

The public API is independent of Streamlit.  It keeps processing on a physical
1 cm⁻¹ grid, never extrapolates beyond measured support, and treats large axis
gaps as separate detector segments instead of interpolating through them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

import raman_ai_denoiser as ai_denoiser
import raman_core as rc

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.linalg import solveh_banded
    from scipy.signal import find_peaks, medfilt
    from scipy.special import expit

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only in reduced installations
    HAVE_SCIPY = False


PREPROCESS_GRID_STEP_CM1 = 1.0
PREPROCESS_PIPELINE_VERSION = 4


@dataclass(frozen=True)
class BaselineSettings:
    method: str = "arPLS"
    lam: float = 1.0e5
    lam_exp: int = 5
    itermax: int = 50
    tol: float = 1.0e-3
    p: float = 0.010
    niter: int = 20
    lam1: float = 1.0e2
    lam1_exp: int = 2
    autoscale: bool = True
    db_strength: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "BaselineSettings") -> "BaselineSettings":
        if isinstance(value, cls):
            return value
        lam_exp = int(value.get("lam_exp", 5))
        lam1_exp = int(value.get("lam1_exp", 2))
        return cls(
            method=str(value.get("method", "arPLS")),
            lam=float(value.get("lam", 10.0**lam_exp)),
            lam_exp=lam_exp,
            itermax=int(value.get("itermax", 50)),
            tol=float(value.get("tol", 1.0e-3)),
            p=float(value.get("p", 0.010)),
            niter=int(value.get("niter", 20)),
            lam1=float(value.get("lam1", 10.0**lam1_exp)),
            lam1_exp=lam1_exp,
            autoscale=bool(value.get("autoscale", True)),
            db_strength=float(np.clip(value.get("db_strength", 1.0), 0.0, 1.0)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "lam": self.lam,
            "lam_exp": self.lam_exp,
            "itermax": self.itermax,
            "tol": self.tol,
            "p": self.p,
            "niter": self.niter,
            "lam1": self.lam1,
            "lam1_exp": self.lam1_exp,
            "autoscale": self.autoscale,
            "db_strength": self.db_strength,
        }


@dataclass(frozen=True)
class SmoothingSettings:
    method: str = "savgol"
    window: int = 5
    poly: int = 3
    max_change_sigma: float = ai_denoiser.DEFAULT_MAX_CHANGE_SIGMA

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "SmoothingSettings" | None
    ) -> "SmoothingSettings":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls(method="none")
        raw_method = value.get("method")
        if raw_method is None:
            raw_method = "savgol" if bool(value.get("enabled", True)) else "none"
        aliases = {
            "none": "none",
            "off": "none",
            "savgol": "savgol",
            "savitzky-golay": "savgol",
            "ai": "deeper_ai",
            "deeper": "deeper_ai",
            "deeper_ai": "deeper_ai",
        }
        method = aliases.get(str(raw_method).strip().lower())
        if method is None:
            raise ValueError(f"Unknown denoising method: {raw_method!r}")
        return cls(
            method=method,
            window=int(value.get("window", 5)),
            poly=int(value.get("poly", 3)),
            max_change_sigma=float(
                value.get("max_change_sigma", ai_denoiser.DEFAULT_MAX_CHANGE_SIGMA)
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "enabled": self.method != "none",
            "window": self.window,
            "poly": self.poly,
            "max_change_sigma": self.max_change_sigma,
        }


@dataclass(frozen=True)
class AxisQuality:
    point_count: int
    finite_point_count: int
    minimum_cm1: float | None
    maximum_cm1: float | None
    median_spacing_cm1: float | None
    spacing_cv: float | None
    duplicate_count: int
    gap_intervals_cm1: tuple[tuple[float, float], ...]
    saturation_fraction: float
    spike_indices: tuple[int, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeasurementSpectrum:
    """One validated measurement input plus its immutable QC summary."""

    axis_cm1: np.ndarray
    intensity: np.ndarray
    quality: AxisQuality

    def __post_init__(self) -> None:
        axis = np.array(self.axis_cm1, dtype=float, copy=True).reshape(-1)
        intensity = np.array(self.intensity, dtype=float, copy=True).reshape(-1)
        if axis.size != intensity.size:
            raise ValueError("measurement axis and intensity must have equal length")
        if axis.size == 0:
            raise ValueError("measurement contains no usable points")
        axis.setflags(write=False)
        intensity.setflags(write=False)
        object.__setattr__(self, "axis_cm1", axis)
        object.__setattr__(self, "intensity", intensity)


def parse_measurement_text(text: str) -> MeasurementSpectrum:
    """Parse and assess one measurement without importing the Streamlit UI."""

    axis, intensity = rc.parse_measurement(str(text))
    if axis.size == 0 or intensity.size == 0:
        raise ValueError("measurement contains no usable points")
    return MeasurementSpectrum(
        axis_cm1=axis,
        intensity=intensity,
        quality=assess_axis_quality(axis, intensity),
    )


@dataclass(frozen=True, slots=True)
class ProcessedSegment:
    axis_cm1: np.ndarray
    raw: np.ndarray
    baseline: np.ndarray
    processed: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.array(item, dtype=float, copy=True).reshape(-1) for item in (
            self.axis_cm1, self.raw, self.baseline, self.processed
        )]
        if len({item.size for item in arrays}) != 1:
            raise ValueError("processed segment arrays must have equal length")
        for name, value in zip(("axis_cm1", "raw", "baseline", "processed"), arrays):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ProcessedSpectrum:
    segments: tuple[ProcessedSegment, ...]
    quality: AxisQuality
    baseline_settings: BaselineSettings
    smoothing_settings: SmoothingSettings
    apply_baseline: bool
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def axis_cm1(self) -> np.ndarray:
        if not self.segments:
            return np.array([], dtype=float)
        return np.concatenate([segment.axis_cm1 for segment in self.segments])

    @property
    def processed(self) -> np.ndarray:
        if not self.segments:
            return np.array([], dtype=float)
        return np.concatenate([segment.processed for segment in self.segments])

    @property
    def baseline(self) -> np.ndarray:
        if not self.segments:
            return np.array([], dtype=float)
        return np.concatenate([segment.baseline for segment in self.segments])

    def project(self, target_axis_cm1: np.ndarray, *, field_name: str = "processed") -> tuple[np.ndarray, np.ndarray]:
        """Project a field without extrapolating or bridging detector gaps."""

        target = np.asarray(target_axis_cm1, dtype=float).reshape(-1)
        values = np.zeros_like(target, dtype=float)
        valid = np.zeros_like(target, dtype=bool)
        for segment in self.segments:
            axis = segment.axis_cm1
            if axis.size == 0:
                continue
            segment_values = np.asarray(getattr(segment, field_name), dtype=float)
            mask = np.isfinite(target) & (target >= axis[0]) & (target <= axis[-1])
            if not np.any(mask):
                continue
            if axis.size == 1:
                values[mask] = float(segment_values[0])
            else:
                values[mask] = np.interp(target[mask], axis, segment_values)
            valid[mask] = True
        return values, valid


def project_processed_spectrum(
    spectrum: ProcessedSpectrum,
    target_axis_cm1: np.ndarray,
    *,
    axis_shift_cm1: float = 0.0,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Project approved intensities after applying an axis-only calibration.

    Baseline and denoising are intentionally evaluated on the native calibrated
    acquisition spacing first.  The shift changes only the x coordinates used
    for projection and can therefore never change fitted intensities.
    """

    target = np.asarray(target_axis_cm1, dtype=float).reshape(-1)
    values = np.zeros_like(target, dtype=float)
    valid = np.zeros_like(target, dtype=bool)
    if not spectrum.segments:
        return values, valid
    concatenated = np.concatenate([segment.processed for segment in spectrum.segments])
    if normalize:
        concatenated = rc._normalize(concatenated)
    offset = 0
    shift = float(axis_shift_cm1)
    for segment in spectrum.segments:
        length = segment.processed.size
        segment_values = concatenated[offset : offset + length]
        offset += length
        shifted_axis = segment.axis_cm1 + shift
        mask = np.isfinite(target) & (target >= shifted_axis[0]) & (target <= shifted_axis[-1])
        if not np.any(mask):
            continue
        if shifted_axis.size == 1:
            values[mask] = float(segment_values[0])
        else:
            values[mask] = np.interp(target[mask], shifted_axis, segment_values)
        valid[mask] = True
    return values, valid


@dataclass(frozen=True, slots=True)
class ReferenceAlignment:
    values: np.ndarray
    overlap_mask: np.ndarray
    overlap_fraction: float

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True).reshape(-1)
        overlap = np.array(self.overlap_mask, dtype=bool, copy=True).reshape(-1)
        if values.size != overlap.size:
            raise ValueError("reference alignment values and mask must have equal length")
        fraction = float(self.overlap_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("reference overlap fraction must be between zero and one")
        values.setflags(write=False)
        overlap.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "overlap_mask", overlap)
        object.__setattr__(self, "overlap_fraction", fraction)


def default_baseline_settings() -> BaselineSettings:
    return BaselineSettings()


def fixed_database_baseline_settings() -> BaselineSettings:
    # Keep reference preprocessing stable when the user-facing measurement
    # default changes; otherwise a UI-default adjustment would force a full
    # database-cache rebuild and alter every baseline-corrected reference.
    return BaselineSettings(lam=1.0e4, lam_exp=4)


def default_smoothing_settings() -> SmoothingSettings:
    return SmoothingSettings()


def baseline_payload(value: Mapping[str, Any] | BaselineSettings) -> dict[str, Any]:
    cfg = BaselineSettings.from_mapping(value)
    payload: dict[str, Any] = {
        "v": 5,
        "grid_step_cm1": PREPROCESS_GRID_STEP_CM1,
        "method": cfg.method,
        "lam_exp": cfg.lam_exp,
        "lam": cfg.lam,
        "autoscale": cfg.autoscale,
        "db_strength": round(cfg.db_strength, 4),
        "have_scipy": HAVE_SCIPY,
        "banded_solver": HAVE_SCIPY,
    }
    if cfg.method == "arPLS":
        payload.update(itermax=cfg.itermax, tol=cfg.tol)
    elif cfg.method == "ALS":
        payload.update(p=cfg.p, niter=cfg.niter, lam1_exp=cfg.lam1_exp, lam1=cfg.lam1)
    return payload


def smoothing_payload(value: Mapping[str, Any] | SmoothingSettings | None) -> dict[str, Any]:
    cfg = SmoothingSettings.from_mapping(value)
    payload: dict[str, Any] = {
        "v": 5,
        "grid_step_cm1": PREPROCESS_GRID_STEP_CM1,
        "method": cfg.method,
    }
    if cfg.method == "savgol":
        payload.update(window=cfg.window, poly=cfg.poly)
    elif cfg.method == "deeper_ai":
        payload.update(
            model_id=ai_denoiser.MODEL_ID,
            model_sha256=ai_denoiser.MODEL_SHA256,
            model_points=ai_denoiser.MODEL_POINTS,
            model_training_range_cm1=list(ai_denoiser.MODEL_RANGE_CM1),
            full_range_adapter_v=ai_denoiser.FULL_RANGE_ADAPTER_VERSION,
            window_span_cm1=ai_denoiser.MODEL_WINDOW_SPAN_CM1,
            window_overlap_cm1=ai_denoiser.MODEL_WINDOW_OVERLAP_CM1,
            guard_guide_span_cm1=ai_denoiser.GUARD_GUIDE_SPAN_CM1,
            guard_trend_span_cm1=ai_denoiser.GUARD_TREND_SPAN_CM1,
            guard_max_range_fraction=ai_denoiser.GUARD_MAX_DYNAMIC_RANGE_FRACTION,
            max_change_sigma=round(cfg.max_change_sigma, 3),
        )
    return payload


def config_token(payload: Mapping[str, Any], *, length: int = 12) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:length]


def assess_axis_quality(x: np.ndarray, y: np.ndarray) -> AxisQuality:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must contain the same number of values")
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    xf = x_arr[finite]
    yf = y_arr[finite]
    warnings: list[str] = []
    if xf.size == 0:
        return AxisQuality(x_arr.size, 0, None, None, None, None, 0, (), 0.0, (), ("No finite spectral points.",))

    order = np.argsort(xf, kind="mergesort")
    xf = xf[order]
    yf = yf[order]
    duplicate_count = int(xf.size - np.unique(xf).size)
    unique_x = np.unique(xf)
    diffs = np.diff(unique_x)
    positive = diffs[diffs > 0]
    median_spacing = float(np.median(positive)) if positive.size else None
    spacing_cv = None
    gap_intervals: list[tuple[float, float]] = []
    if positive.size and median_spacing and median_spacing > 0:
        spacing_cv = float(np.std(positive) / median_spacing)
        gap_threshold = max(5.0 * median_spacing, median_spacing + 10.0)
        for index in np.where(diffs > gap_threshold)[0]:
            gap_intervals.append((float(unique_x[index]), float(unique_x[index + 1])))
        if spacing_cv > 0.25:
            warnings.append("Raman-shift spacing is strongly irregular.")
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate Raman-shift values will be averaged.")
    if gap_intervals:
        warnings.append(f"Detected {len(gap_intervals)} large axis gap(s); they will not be interpolated across.")

    saturation_fraction = 0.0
    spike_indices: tuple[int, ...] = ()
    if yf.size:
        ymax = float(np.max(yf))
        yrange = float(np.ptp(yf))
        atol = max(1e-12, yrange * 1e-8)
        saturation_fraction = float(np.mean(np.isclose(yf, ymax, rtol=0.0, atol=atol)))
        if saturation_fraction >= 0.01 and yf.size >= 50:
            warnings.append("A repeated intensity ceiling suggests detector saturation or clipping.")
        if HAVE_SCIPY and yf.size >= 7 and yrange > 0:
            local = medfilt(yf, kernel_size=5)
            residual = yf - local
            mad = float(np.median(np.abs(residual - np.median(residual))))
            sigma = max(1.4826 * mad, yrange * 1e-8)
            indices = np.where(residual > max(8.0 * sigma, 0.04 * yrange))[0]
            spike_indices = tuple(int(item) for item in indices[:100])
    if xf.size < 30:
        warnings.append("Fewer than 30 finite points is generally insufficient for robust library matching.")

    return AxisQuality(
        point_count=int(x_arr.size),
        finite_point_count=int(xf.size),
        minimum_cm1=float(xf[0]),
        maximum_cm1=float(xf[-1]),
        median_spacing_cm1=median_spacing,
        spacing_cv=spacing_cv,
        duplicate_count=duplicate_count,
        gap_intervals_cm1=tuple(gap_intervals),
        saturation_fraction=saturation_fraction,
        spike_indices=spike_indices,
        warnings=tuple(warnings),
    )


def clean_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must contain the same number of values")
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(x_arr, kind="mergesort")
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    unique_x, inverse = np.unique(x_arr, return_inverse=True)
    if unique_x.size == x_arr.size:
        return x_arr, y_arr
    sums = np.bincount(inverse, weights=y_arr)
    counts = np.bincount(inverse)
    return unique_x, sums / np.maximum(counts, 1)


def support_slices(x: np.ndarray) -> tuple[slice, ...]:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return ()
    if arr.size == 1:
        return (slice(0, 1),)
    diffs = np.diff(arr)
    positive = diffs[diffs > 0]
    if not positive.size:
        return (slice(0, arr.size),)
    median = float(np.median(positive))
    threshold = max(5.0 * median, median + 10.0)
    breaks = np.where(diffs > threshold)[0]
    starts = [0, *[int(index + 1) for index in breaks]]
    stops = [*[int(index + 1) for index in breaks], arr.size]
    return tuple(slice(start, stop) for start, stop in zip(starts, stops) if stop > start)


def canonical_grid(x: np.ndarray, *, step_cm1: float = PREPROCESS_GRID_STEP_CM1) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < 2:
        return arr.copy()
    step = float(step_cm1)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("canonical preprocessing step must be positive")
    first = math.ceil((float(arr[0]) / step) - 1e-9)
    last = math.floor((float(arr[-1]) / step) + 1e-9)
    if last >= first:
        grid = np.arange(first, last + 1, dtype=float) * step
        if grid.size:
            return grid
    return np.array([float(arr[0]), float(arr[-1])], dtype=float)


def _second_difference_bands(length: int, lam: float) -> np.ndarray:
    bands = np.zeros((3, length), dtype=float)
    if length < 3:
        return bands
    main = np.full(length, 6.0, dtype=float)
    main[0] = main[-1] = 1.0
    if length > 2:
        main[1] = main[-2] = 5.0
    first = np.full(length - 1, -4.0, dtype=float)
    first[0] = first[-1] = -2.0
    bands[2] = lam * main
    bands[1, 1:] = lam * first
    bands[0, 2:] = lam
    return bands


def _solve_banded_or_sparse(
    rhs: np.ndarray,
    weights: np.ndarray,
    second_bands: np.ndarray,
    *,
    first_penalty_weights: np.ndarray | None = None,
    lam1: float = 0.0,
) -> np.ndarray:
    length = rhs.size
    if HAVE_SCIPY:
        bands = second_bands.copy()
        bands[2] += weights
        if first_penalty_weights is not None and length > 1 and lam1:
            q = np.asarray(first_penalty_weights, dtype=float).reshape(-1)
            if q.size != length - 1:
                raise ValueError("first-penalty weights have wrong length")
            bands[2, 0] += lam1 * q[0]
            bands[2, -1] += lam1 * q[-1]
            if length > 2:
                bands[2, 1:-1] += lam1 * (q[:-1] + q[1:])
            bands[1, 1:] += -lam1 * q
        try:
            return solveh_banded(
                bands,
                rhs,
                lower=False,
                overwrite_ab=True,
                overwrite_b=False,
                check_finite=False,
            )
        except Exception:
            pass

    if not HAVE_SCIPY:
        return rc._baseline_als(rhs)
    d2 = sp.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2), dtype=float).T
    matrix = sp.diags(weights, 0) + second_bands[0, 2] * (d2.T @ d2)
    if first_penalty_weights is not None and length > 1 and lam1:
        d1 = sp.diags([-1, 1], [0, 1], shape=(length - 1, length), dtype=float)
        matrix = matrix + lam1 * (d1.T @ sp.diags(first_penalty_weights, 0) @ d1)
    return np.asarray(spla.spsolve(matrix.tocsc(), rhs), dtype=float)


def baseline_arpls(y: np.ndarray, *, lam: float = 1.0e4, itermax: int = 50, tol: float = 1.0e-3) -> np.ndarray:
    signal = np.asarray(y, dtype=float).reshape(-1)
    length = signal.size
    if length < 5:
        return np.zeros_like(signal)
    if not HAVE_SCIPY:
        return rc._baseline_als(signal)
    second = _second_difference_bands(length, float(lam))
    weights = np.ones(length, dtype=float)
    baseline = signal.copy()
    for _ in range(max(1, int(itermax))):
        baseline = _solve_banded_or_sparse(weights * signal, weights, second)
        residual = signal - baseline
        negative = residual[residual < 0]
        if negative.size == 0:
            break
        mean = float(np.mean(negative))
        std = max(float(np.std(negative)), 1e-12)
        updated = expit(-2.0 * (residual - (2.0 * std - mean)) / std)
        relative_change = np.linalg.norm(weights - updated) / (np.linalg.norm(weights) + 1e-12)
        weights = updated
        if relative_change < float(tol):
            break
    return np.asarray(baseline, dtype=float)


def baseline_iasls(
    y: np.ndarray,
    *,
    lam: float = 1.0e4,
    p: float = 0.01,
    niter: int = 20,
    lam1: float = 1.0e2,
) -> np.ndarray:
    signal = np.asarray(y, dtype=float).reshape(-1)
    length = signal.size
    if length < 5:
        return np.zeros_like(signal)
    if not HAVE_SCIPY:
        return rc._baseline_als(signal)
    second = _second_difference_bands(length, float(lam))
    weights = np.ones(length, dtype=float)
    baseline = signal.copy()
    for _ in range(max(1, int(niter))):
        residual = signal - baseline
        roughness = np.abs(np.diff(residual))
        roughness[~np.isfinite(roughness)] = 0.0
        maximum = float(np.max(roughness)) if roughness.size else 0.0
        if maximum > 0:
            roughness = roughness / maximum
        roughness += 1e-6
        baseline = _solve_banded_or_sparse(
            weights * signal,
            weights,
            second,
            first_penalty_weights=roughness,
            lam1=float(lam1),
        )
        weights = np.where(signal > baseline, float(p), 1.0 - float(p))
    return np.asarray(baseline, dtype=float)


def compute_baseline(y: np.ndarray, settings: Mapping[str, Any] | BaselineSettings) -> np.ndarray:
    signal = np.asarray(y, dtype=float).reshape(-1)
    if signal.size < 5:
        return np.zeros_like(signal)
    cfg = BaselineSettings.from_mapping(settings)
    if cfg.method not in {"arPLS", "ALS"}:
        return np.zeros_like(signal)
    if cfg.autoscale:
        finite = np.isfinite(signal)
        offset = float(np.min(signal[finite])) if np.any(finite) else 0.0
        scale = max(float(np.ptp(signal[finite])) if np.any(finite) else 0.0, 1e-12)
        working = (signal - offset) / scale
    else:
        offset, scale, working = 0.0, 1.0, signal
    if cfg.method == "arPLS":
        result = baseline_arpls(working, lam=cfg.lam, itermax=cfg.itermax, tol=cfg.tol)
    else:
        result = baseline_iasls(working, lam=cfg.lam, p=cfg.p, niter=cfg.niter, lam1=cfg.lam1)
    return result * scale + offset


def sanitize_savgol_params(length: int, window: int, poly: int) -> tuple[int, int] | None:
    if length < 3:
        return None
    actual_window = max(3, int(window))
    if actual_window % 2 == 0:
        actual_window += 1
    maximum = length if length % 2 else length - 1
    if maximum < 3:
        return None
    actual_window = min(actual_window, maximum)
    actual_poly = min(max(0, int(poly)), actual_window - 1)
    return actual_window, actual_poly


def apply_smoothing(x: np.ndarray, y: np.ndarray, settings: Mapping[str, Any] | SmoothingSettings | None) -> np.ndarray:
    axis = np.asarray(x, dtype=float).reshape(-1)
    signal = np.asarray(y, dtype=float).reshape(-1)
    cfg = SmoothingSettings.from_mapping(settings)
    if signal.size == 0 or cfg.method == "none":
        return signal.copy()
    if cfg.method == "deeper_ai":
        return ai_denoiser.denoise(axis, signal, max_change_sigma=cfg.max_change_sigma)
    params = sanitize_savgol_params(signal.size, cfg.window, cfg.poly)
    if params is None:
        return signal.copy()
    try:
        return rc._smooth(signal, window=params[0], poly=params[1])
    except Exception:
        return signal.copy()


def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    apply_baseline: bool,
    baseline_settings: Mapping[str, Any] | BaselineSettings,
    smoothing_settings: Mapping[str, Any] | SmoothingSettings | None = None,
    baseline_strength: float = 1.0,
) -> ProcessedSpectrum:
    quality = assess_axis_quality(x, y)
    clean_x_arr, clean_y_arr = clean_xy(x, y)
    baseline_cfg = BaselineSettings.from_mapping(baseline_settings)
    smoothing_cfg = SmoothingSettings.from_mapping(smoothing_settings)
    segments: list[ProcessedSegment] = []
    all_raw: list[np.ndarray] = []
    all_corrected: list[np.ndarray] = []
    all_processed: list[np.ndarray] = []
    for support_slice in support_slices(clean_x_arr):
        native_x = clean_x_arr[support_slice]
        native_y = clean_y_arr[support_slice]
        grid = canonical_grid(native_x)
        if grid.size == 0:
            continue
        raw = np.interp(grid, native_x, native_y) if native_x.size > 1 else np.full(grid.size, native_y[0])
        if apply_baseline:
            baseline = compute_baseline(raw, baseline_cfg)
            corrected = raw - float(np.clip(baseline_strength, 0.0, 1.0)) * baseline
        else:
            baseline = np.zeros_like(raw)
            corrected = raw.copy()
        processed = apply_smoothing(grid, corrected, smoothing_cfg)
        segments.append(ProcessedSegment(grid, raw, baseline, processed))
        all_raw.append(raw)
        all_corrected.append(corrected)
        all_processed.append(processed)

    diagnostics: dict[str, float] = {}
    if all_processed:
        raw_all = np.concatenate(all_raw)
        corrected_all = np.concatenate(all_corrected)
        processed_all = np.concatenate(all_processed)
        dynamic_range = max(float(np.ptp(raw_all)), 1e-12)
        within_segment_differences = [
            np.diff(values) for values in all_corrected if values.size > 1
        ]
        if within_segment_differences:
            differences = np.concatenate(within_segment_differences)
            difference_median = float(np.median(differences))
            noise_sigma = (
                1.4826
                * float(np.median(np.abs(differences - difference_median)))
                / math.sqrt(2.0)
            )
        else:
            noise_sigma = 0.0
        if not math.isfinite(noise_sigma):
            noise_sigma = 0.0
        # A baseline-corrected noisy signal is expected to cross zero often.
        # Treat only excursions beyond both three robust noise sigma and 0.5%
        # of the input range as materially negative; the 0.5% value is a
        # diagnostic tolerance, not an alteration of the fitted baseline.
        material_negative_threshold = max(
            3.0 * noise_sigma,
            0.005 * dynamic_range,
            1e-12,
        )
        diagnostics = {
            "negative_fraction": float(np.mean(corrected_all < 0.0)),
            "material_negative_fraction": float(
                np.mean(corrected_all < -material_negative_threshold)
            ),
            "material_negative_threshold": float(material_negative_threshold),
            "noise_sigma": float(noise_sigma),
            "change_rms_fraction": float(np.sqrt(np.mean((processed_all - raw_all) ** 2)) / dynamic_range),
            "peak_change_fraction": float((np.max(processed_all) - np.max(raw_all)) / dynamic_range),
        }
    return ProcessedSpectrum(
        segments=tuple(segments),
        quality=quality,
        baseline_settings=baseline_cfg,
        smoothing_settings=smoothing_cfg,
        apply_baseline=bool(apply_baseline),
        diagnostics=diagnostics,
    )


def align_reference_to_target(target_x: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> ReferenceAlignment:
    target = np.asarray(target_x, dtype=float).reshape(-1)
    source_x, source_y = clean_xy(ref_x, ref_y)
    values = np.zeros_like(target, dtype=float)
    overlap = np.zeros_like(target, dtype=bool)
    if target.size == 0 or source_x.size < 2:
        return ReferenceAlignment(values, overlap, 0.0)
    for support_slice in support_slices(source_x):
        segment_x = source_x[support_slice]
        segment_y = source_y[support_slice]
        if segment_x.size < 2:
            continue
        mask = np.isfinite(target) & (target >= segment_x[0]) & (target <= segment_x[-1])
        if np.any(mask):
            values[mask] = np.interp(target[mask], segment_x, segment_y)
            overlap[mask] = True
    fraction = float(np.count_nonzero(overlap) / max(1, target.size))
    return ReferenceAlignment(values, overlap, fraction)


def _align_reference_to_target(
    target_x: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray
) -> np.ndarray:
    """Legacy ndarray-only view of :func:`align_reference_to_target`."""

    return align_reference_to_target(target_x, ref_x, ref_y).values


def normalize_to_peak(y: np.ndarray) -> np.ndarray:
    signal = np.asarray(y, dtype=float)
    output = np.zeros_like(signal, dtype=float)
    finite = np.isfinite(signal)
    if not np.any(finite):
        return output
    peak = float(np.max(signal[finite]))
    if peak <= 1e-12:
        peak = float(np.max(np.abs(signal[finite])))
    if not np.isfinite(peak) or peak <= 1e-12:
        return output
    output[finite] = signal[finite] / peak
    return output


def prepare_measurement_signal(
    x: np.ndarray,
    y: np.ndarray,
    *,
    apply_baseline: bool,
    baseline_cfg: Mapping[str, Any] | BaselineSettings,
    smoothing_cfg: Mapping[str, Any] | SmoothingSettings | None = None,
    target_x: np.ndarray | None = None,
) -> np.ndarray:
    result = preprocess_spectrum(
        x,
        y,
        apply_baseline=apply_baseline,
        baseline_settings=baseline_cfg,
        smoothing_settings=smoothing_cfg,
    )
    target = np.asarray(x if target_x is None else target_x, dtype=float)
    return result.project(target)[0]


def prepare_db_signal_on_target_grid(
    db_x: np.ndarray,
    db_y: np.ndarray,
    target_x: np.ndarray,
    *,
    apply_baseline_db: bool,
    baseline_cfg: Mapping[str, Any] | BaselineSettings,
) -> np.ndarray:
    cfg = BaselineSettings.from_mapping(baseline_cfg)
    result = preprocess_spectrum(
        db_x,
        db_y,
        apply_baseline=apply_baseline_db,
        baseline_settings=cfg,
        baseline_strength=cfg.db_strength,
        smoothing_settings=SmoothingSettings(method="none"),
    )
    return result.project(np.asarray(target_x, dtype=float))[0]


def process_measurement(
    x: np.ndarray,
    y: np.ndarray,
    *,
    apply_baseline: bool,
    baseline_cfg: Mapping[str, Any] | BaselineSettings,
    smoothing_cfg: Mapping[str, Any] | SmoothingSettings | None = None,
    target_x: np.ndarray | None = None,
) -> np.ndarray:
    result = preprocess_spectrum(
        x,
        y,
        apply_baseline=apply_baseline,
        baseline_settings=baseline_cfg,
        smoothing_settings=smoothing_cfg,
    )
    normalized_segments = tuple(
        ProcessedSegment(segment.axis_cm1, segment.raw, segment.baseline, rc._normalize(segment.processed))
        for segment in result.segments
    )
    normalized = ProcessedSpectrum(
        normalized_segments,
        result.quality,
        result.baseline_settings,
        result.smoothing_settings,
        result.apply_baseline,
        result.diagnostics,
    )
    target = np.asarray(x if target_x is None else target_x, dtype=float)
    return normalized.project(target)[0]


def process_db_on_target_grid(
    db_x: np.ndarray,
    db_y: np.ndarray,
    target_x: np.ndarray,
    *,
    apply_baseline_db: bool,
    baseline_cfg: Mapping[str, Any] | BaselineSettings,
) -> np.ndarray:
    signal = prepare_db_signal_on_target_grid(
        db_x,
        db_y,
        target_x,
        apply_baseline_db=apply_baseline_db,
        baseline_cfg=baseline_cfg,
    )
    return rc._normalize(signal) if signal.size else signal


def compute_baseline_on_axis(
    x: np.ndarray,
    y: np.ndarray,
    baseline_cfg: Mapping[str, Any] | BaselineSettings,
    *,
    target_x: np.ndarray | None = None,
) -> np.ndarray:
    result = preprocess_spectrum(
        x,
        y,
        apply_baseline=True,
        baseline_settings=baseline_cfg,
        smoothing_settings=SmoothingSettings(method="none"),
    )
    target = np.asarray(x if target_x is None else target_x, dtype=float)
    return result.project(target, field_name="baseline")[0]


# Compatibility helpers retained while callers migrate to typed settings.
def _default_baseline_cfg() -> dict[str, Any]:
    return default_baseline_settings().to_mapping()


def _fixed_db_baseline_cfg() -> dict[str, Any]:
    return fixed_database_baseline_settings().to_mapping()


def _baseline_cfg_payload(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return baseline_payload(cfg)


def _baseline_cfg_token(cfg: Mapping[str, Any]) -> str:
    return config_token(baseline_payload(cfg))


def _baseline_label(cfg: Mapping[str, Any]) -> str:
    value = BaselineSettings.from_mapping(cfg)
    if value.method == "arPLS":
        return f"arPLS (λ=1e{int(round(np.log10(max(value.lam, 1e-12))))}, iter≤{value.itermax}, tol={value.tol})"
    if value.method == "ALS":
        return f"IAsLS (λ=1e{int(round(np.log10(max(value.lam, 1e-12))))}, λ1=1e{int(round(np.log10(max(value.lam1, 1e-12))))}, p={value.p:.3f}, iters={value.niter})"
    return "none (RAW)"


def _default_smoothing_cfg() -> dict[str, Any]:
    return default_smoothing_settings().to_mapping()


def _smoothing_method(cfg: Mapping[str, Any] | None) -> str:
    return SmoothingSettings.from_mapping(cfg).method


def _smoothing_cfg_payload(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return smoothing_payload(cfg)


def _smoothing_cfg_token(cfg: Mapping[str, Any]) -> str:
    return config_token(smoothing_payload(cfg))


def _smoothing_label(cfg: Mapping[str, Any]) -> str:
    value = SmoothingSettings.from_mapping(cfg)
    if value.method == "none":
        return "none"
    if value.method == "deeper_ai":
        return "AI-assisted · guarded DeepeR (full range, experimental)"
    return f"Savitzky-Golay (window={value.window} points at {PREPROCESS_GRID_STEP_CM1:g} cm⁻¹ spacing, poly={value.poly})"


def _smoothing_preview_ui(cfg: Mapping[str, Any]) -> dict[str, str]:
    value = SmoothingSettings.from_mapping(cfg)
    if value.method == "none":
        return {
            "title": "Unchanged measurement preview",
            "curve_label": "",
            "preview_label": "Measurement preview",
            "preview_file_tag": "measurement_unchanged_preview",
            "spectrum_label": "Unchanged spectrum",
            "spectrum_file_tag": "unchanged",
        }
    if value.method == "deeper_ai":
        return {
            "title": "Guarded DeepeR denoising preview",
            "curve_label": "guarded AI denoising (experimental)",
            "preview_label": "Denoising preview",
            "preview_file_tag": "denoising_preview",
            "spectrum_label": "Guarded denoised spectrum",
            "spectrum_file_tag": "deeper_ai_guarded_full_range",
        }
    return {
        "title": "Savitzky–Golay smoothing preview",
        "curve_label": (
            "smoothed · Savitzky-Golay "
            f"(window = {value.window}, poly = {value.poly})"
        ),
        "preview_label": "Smoothing preview",
        "preview_file_tag": "smoothing_preview",
        "spectrum_label": "Smoothed spectrum",
        "spectrum_file_tag": "savgol",
    }


def smoothing_input_curve_label(
    measurement_mode: str,
    *,
    white_reference_applied: bool = False,
) -> str:
    """Return the compact input-curve label used by smoothing previews."""

    if str(measurement_mode).strip().upper() == "BC":
        return "baseline-corrected measurement"
    if white_reference_applied:
        return "white-reference-corrected measurement"
    return "raw measurement"


def processing_difference_curve_label(magnification: float) -> str:
    """Return the compact, dynamic label for the offset difference trace."""

    value = float(magnification)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("difference-curve magnification must be finite and positive")
    return f"difference curve; x{value:g}"


_baseline_arpls = baseline_arpls
_baseline_iasls = baseline_iasls
_compute_baseline = compute_baseline
_sanitize_savgol_params = sanitize_savgol_params
_apply_smoothing = apply_smoothing
_clean_xy_for_preprocessing = clean_xy
_canonical_grid_for_support = canonical_grid
_compute_baseline_on_axis = compute_baseline_on_axis
_normalize_to_peak = normalize_to_peak
_prepare_measurement_signal = prepare_measurement_signal
_prepare_db_signal_on_target_grid = prepare_db_signal_on_target_grid
_process_measurement = process_measurement
_process_db_on_target_grid = process_db_on_target_grid


__all__ = [
    "AxisQuality",
    "BaselineSettings",
    "MeasurementSpectrum",
    "ProcessedSegment",
    "ProcessedSpectrum",
    "ReferenceAlignment",
    "SmoothingSettings",
    "align_reference_to_target",
    "assess_axis_quality",
    "baseline_arpls",
    "baseline_iasls",
    "compute_baseline",
    "compute_baseline_on_axis",
    "normalize_to_peak",
    "parse_measurement_text",
    "processing_difference_curve_label",
    "prepare_db_signal_on_target_grid",
    "prepare_measurement_signal",
    "preprocess_spectrum",
    "project_processed_spectrum",
    "process_db_on_target_grid",
    "process_measurement",
    "smoothing_input_curve_label",
]

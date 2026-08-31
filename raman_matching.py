"""Pure, typed Raman matching primitives and evidence summaries.

This module is intentionally independent of Streamlit, filesystem state, and
the database cache format.  It provides the numerical alignment operation used
by RamanPhaseID together with explicit evidence objects that make the scored
candidate, comparison support, fitted shift, and rank composition auditable.

The evidence-status helper implements conservative *operational guardrails*.
Its output is not a calibrated probability, confidence interval, or confirmed
phase identification.  Calibration requires an external, specimen-grouped
validation set representative of the intended instruments and samples.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
except Exception:  # pragma: no cover - optional SciPy fallback
    _scipy_find_peaks = None


DEFAULT_GRADIENT_WEIGHT = 0.20
DEFAULT_FINAL_SIMILARITY_WEIGHT = 0.88
# Scores closer than this absolute tolerance are treated as numerically tied
# during discrete shift selection.  The remaining tie-breaks prefer evidence
# supported by more measured points and avoid gratuitous fitted shifts.
ALIGNMENT_SCORE_TIE_TOLERANCE = 1e-12


NumericArray = NDArray[Any]
BoolArray = NDArray[np.bool_]
SupportRuns = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class MatchingParameters:
    """Validated numerical/search policy for one matching run."""

    gradient_weight: float = DEFAULT_GRADIENT_WEIGHT
    spectral_similarity_weight: float = DEFAULT_FINAL_SIMILARITY_WEIGHT
    peak_f1_weight: float = 0.75
    remove_query_local_offset: bool = True
    minimum_candidate_peak_consistency: float = 0.0
    peak_tolerance_points: int = 5
    maximum_shift_points: int = 5
    minimum_coverage_fraction: float = 0.70
    screen_chunk_rows: int = 1024
    raw_minimum_shortlist: int = 3600
    baseline_minimum_shortlist: int = 1800
    raw_top_n_factor: int = 12
    baseline_top_n_factor: int = 8
    per_phase_cap: int = 5
    peak_phase_slot_cap: int = 12
    peak_phase_minimum_score: float = 0.52
    peak_phase_minimum_similarity: float = 0.50
    alignment_score_tie_tolerance: float = ALIGNMENT_SCORE_TIE_TOLERANCE
    peak_detection_max_peaks: int = 80
    peak_detection_minimum_run_points: int = 5
    peak_detection_minimum_signal: float = 1e-9
    peak_detection_minimum_prominence_absolute: float = 1e-6
    peak_detection_minimum_prominence_fraction: float = 0.03
    peak_detection_minimum_distance_points: int = 3
    peak_consistency_minimum_support_points: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.remove_query_local_offset, (bool, np.bool_)):
            raise TypeError("remove_query_local_offset must be boolean")
        for name, value in (
            ("gradient_weight", self.gradient_weight),
            ("spectral_similarity_weight", self.spectral_similarity_weight),
            ("peak_f1_weight", self.peak_f1_weight),
            (
                "minimum_candidate_peak_consistency",
                self.minimum_candidate_peak_consistency,
            ),
            ("minimum_coverage_fraction", self.minimum_coverage_fraction),
            ("peak_phase_minimum_score", self.peak_phase_minimum_score),
            ("peak_phase_minimum_similarity", self.peak_phase_minimum_similarity),
            (
                "peak_detection_minimum_prominence_fraction",
                self.peak_detection_minimum_prominence_fraction,
            ),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
        for name, value in (
            ("peak_tolerance_points", self.peak_tolerance_points),
            ("maximum_shift_points", self.maximum_shift_points),
            ("screen_chunk_rows", self.screen_chunk_rows),
            ("raw_minimum_shortlist", self.raw_minimum_shortlist),
            ("baseline_minimum_shortlist", self.baseline_minimum_shortlist),
            ("raw_top_n_factor", self.raw_top_n_factor),
            ("baseline_top_n_factor", self.baseline_top_n_factor),
            ("per_phase_cap", self.per_phase_cap),
            ("peak_phase_slot_cap", self.peak_phase_slot_cap),
            ("peak_detection_max_peaks", self.peak_detection_max_peaks),
            ("peak_detection_minimum_run_points", self.peak_detection_minimum_run_points),
            (
                "peak_detection_minimum_distance_points",
                self.peak_detection_minimum_distance_points,
            ),
            (
                "peak_consistency_minimum_support_points",
                self.peak_consistency_minimum_support_points,
            ),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.screen_chunk_rows) < 1 or int(self.per_phase_cap) < 1:
            raise ValueError("screen_chunk_rows and per_phase_cap must be positive")
        for name, value in (
            ("alignment_score_tie_tolerance", self.alignment_score_tie_tolerance),
            ("peak_detection_minimum_signal", self.peak_detection_minimum_signal),
            (
                "peak_detection_minimum_prominence_absolute",
                self.peak_detection_minimum_prominence_absolute,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            int(self.peak_detection_max_peaks) < 1
            or int(self.peak_detection_minimum_run_points) < 3
            or int(self.peak_detection_minimum_distance_points) < 1
            or int(self.peak_consistency_minimum_support_points) < 2
        ):
            raise ValueError("peak-detection limits must be positive and scientifically usable")

    def payload(self) -> dict[str, float | int | bool]:
        """Return every numerical choice that can change a matching result."""

        return {
            "v": 4,
            "gradient_weight": float(self.gradient_weight),
            "spectral_similarity_weight": float(self.spectral_similarity_weight),
            "peak_f1_weight": float(self.peak_f1_weight),
            "remove_query_local_offset": bool(self.remove_query_local_offset),
            "minimum_candidate_peak_consistency": float(
                self.minimum_candidate_peak_consistency
            ),
            "peak_tolerance_points": int(self.peak_tolerance_points),
            "maximum_shift_points": int(self.maximum_shift_points),
            "minimum_coverage_fraction": float(self.minimum_coverage_fraction),
            "screen_chunk_rows": int(self.screen_chunk_rows),
            "raw_minimum_shortlist": int(self.raw_minimum_shortlist),
            "baseline_minimum_shortlist": int(self.baseline_minimum_shortlist),
            "raw_top_n_factor": int(self.raw_top_n_factor),
            "baseline_top_n_factor": int(self.baseline_top_n_factor),
            "per_phase_cap": int(self.per_phase_cap),
            "peak_phase_slot_cap": int(self.peak_phase_slot_cap),
            "peak_phase_minimum_score": float(self.peak_phase_minimum_score),
            "peak_phase_minimum_similarity": float(
                self.peak_phase_minimum_similarity
            ),
            "alignment_score_tie_tolerance": float(
                self.alignment_score_tie_tolerance
            ),
            "peak_detection_max_peaks": int(self.peak_detection_max_peaks),
            "peak_detection_minimum_run_points": int(
                self.peak_detection_minimum_run_points
            ),
            "peak_detection_minimum_signal": float(
                self.peak_detection_minimum_signal
            ),
            "peak_detection_minimum_prominence_absolute": float(
                self.peak_detection_minimum_prominence_absolute
            ),
            "peak_detection_minimum_prominence_fraction": float(
                self.peak_detection_minimum_prominence_fraction
            ),
            "peak_detection_minimum_distance_points": int(
                self.peak_detection_minimum_distance_points
            ),
            "peak_consistency_minimum_support_points": int(
                self.peak_consistency_minimum_support_points
            ),
        }


@dataclass(frozen=True, slots=True)
class ResidualSearchPolicy:
    """Signed numerical guardrails for exploratory residual rematching."""

    minimum_common_points: int = 20
    minimum_fit_improvement_fraction: float = 0.02
    support_edge_guard_points: int = 3

    def __post_init__(self) -> None:
        if int(self.minimum_common_points) < 2:
            raise ValueError("minimum_common_points must be at least 2")
        improvement = float(self.minimum_fit_improvement_fraction)
        if not math.isfinite(improvement) or not 0.0 <= improvement <= 1.0:
            raise ValueError(
                "minimum_fit_improvement_fraction must be finite and between zero and one"
            )
        if int(self.support_edge_guard_points) < 0:
            raise ValueError("support_edge_guard_points must be non-negative")

    def payload(self) -> dict[str, float | int]:
        return {
            "v": 2,
            "minimum_common_points": int(self.minimum_common_points),
            "minimum_fit_improvement_fraction": float(
                self.minimum_fit_improvement_fraction
            ),
            "support_edge_guard_points": int(self.support_edge_guard_points),
        }


@dataclass(frozen=True, slots=True)
class AlignmentEvidence:
    """Auditable support and shift evidence for one aligned comparison."""

    fitted_shift_points: int
    fitted_shift_cm1: float
    maximum_shift_points: int
    shift_search_boundary_hit: bool
    reference_support_clipped_at_grid_boundary: bool
    requested_point_count: int
    reference_support_point_count: int
    shifted_reference_support_point_count: int
    common_point_count: int
    query_coverage_fraction: float
    reference_overlap_fraction: float


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Best spectral alignment plus the exact vector and masks that were scored."""

    shape_similarity: float
    gradient_similarity: float
    spectral_similarity: float
    gradient_weight: float
    evidence: AlignmentEvidence
    aligned_candidate: NumericArray
    comparison_mask: BoolArray
    gradient_comparison_mask: BoolArray


@dataclass(frozen=True, slots=True)
class RankComponents:
    """All numerical components and contributions to the final rank score."""

    shape_similarity: float
    gradient_similarity: float
    peak_consistency: float
    spectral_similarity: float
    final_rank_score: float
    gradient_weight_within_similarity: float
    spectral_similarity_weight: float
    peak_consistency_weight: float
    shape_contribution: float
    gradient_contribution: float
    peak_consistency_contribution: float


@dataclass(frozen=True, slots=True)
class ReferenceMatchEvidence:
    """Evidence for one library trace.

    ``independence_group`` identifies the specimen, for example an RRUFF
    accession. ``acquisition_group`` identifies one spectrum acquisition on
    that specimen, with raw/processed representations sharing a value. This
    distinction prevents duplicate exports from inflating a score without
    averaging away real orientation or excitation-wavelength variants. If
    either value is blank, unresolved traces of the same phase share one
    conservative group; distinct file paths are not presumed independent.
    """

    phase_name: str
    reference_id: str
    independence_group: str
    rank: RankComponents
    alignment: AlignmentEvidence
    acquisition_group: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        phase_name = str(self.phase_name).strip()
        reference_id = str(self.reference_id).strip()
        if not phase_name:
            raise ValueError("phase_name must not be empty")
        if not reference_id:
            raise ValueError("reference_id must not be empty")
        group = str(self.independence_group).strip() or (
            f"unresolved:{phase_key(phase_name)}"
        )
        acquisition = str(self.acquisition_group).strip() or group
        object.__setattr__(self, "phase_name", phase_name)
        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "independence_group", group)
        object.__setattr__(self, "acquisition_group", acquisition)
        object.__setattr__(self, "source", str(self.source).strip())


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    """Duplicate-safe summary across acquisitions and specimens for one phase.

    Raw/processed representations are averaged within an acquisition. For
    each specimen, the best compatible acquisition is retained because Raman
    orientation and excitation can legitimately change relative intensity.
    The phase score is the best resulting specimen score. Evidence support and
    coverage remain separate diagnostics, so a large and diverse library does
    not lower a common phase merely by containing more measured variants.
    """

    phase_name: str
    phase_key: str
    aggregate_score: float
    median_group_score: float
    best_reference_score: float
    mean_coverage_fraction: float
    minimum_group_coverage_fraction: float
    independent_reference_count: int
    reference_variant_count: int
    shift_boundary_group_count: int
    grid_boundary_clipped_group_count: int
    best_match: ReferenceMatchEvidence
    group_scores: tuple[float, ...]
    supporting_matches: tuple[ReferenceMatchEvidence, ...]
    minimum_group_common_point_count: int = 0


class EvidenceStatus(str, Enum):
    """Non-probabilistic evidence states for a guided matching workflow."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNKNOWN_OR_OUT_OF_LIBRARY = "unknown_or_out_of_library"
    AMBIGUOUS = "ambiguous"
    SUPPORTED_CANDIDATE = "supported_candidate"


@dataclass(frozen=True, slots=True)
class EvidenceDecisionPolicy:
    """Conservative, uncalibrated guardrails for :func:`decide_evidence_status`."""

    minimum_phase_score: float = 0.80
    minimum_score_margin: float = 0.05
    minimum_coverage_fraction: float = 0.80
    minimum_common_points: int = 50
    minimum_independent_references: int = 2
    reject_shift_search_boundary: bool = True
    reject_grid_boundary_clipping: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_phase_score", self.minimum_phase_score),
            ("minimum_score_margin", self.minimum_score_margin),
            ("minimum_coverage_fraction", self.minimum_coverage_fraction),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if int(self.minimum_common_points) < 2:
            raise ValueError("minimum_common_points must be at least 2")
        if int(self.minimum_independent_references) < 1:
            raise ValueError("minimum_independent_references must be at least 1")

    def payload(self) -> dict[str, float | int | bool]:
        """Return all uncalibrated evidence guardrails for result identity."""

        return {
            "v": 1,
            "minimum_phase_score": float(self.minimum_phase_score),
            "minimum_score_margin": float(self.minimum_score_margin),
            "minimum_coverage_fraction": float(self.minimum_coverage_fraction),
            "minimum_common_points": int(self.minimum_common_points),
            "minimum_independent_references": int(
                self.minimum_independent_references
            ),
            "reject_shift_search_boundary": bool(self.reject_shift_search_boundary),
            "reject_grid_boundary_clipping": bool(
                self.reject_grid_boundary_clipping
            ),
        }


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    """A transparent guardrail decision, explicitly not calibrated confidence."""

    status: EvidenceStatus
    best_phase: str | None
    best_score: float | None
    runner_up_score: float | None
    score_margin: float | None
    reasons: tuple[str, ...]
    is_calibrated_confidence: bool = False
    runner_up_phase: str | None = None


@dataclass(frozen=True, slots=True)
class ResidualProjection:
    """Auditable one-component least-squares subtraction.

    ``scale_factor`` is a fitted spectral scale on normalized traces; it is
    not an abundance estimate.  The signed residual is retained so negative
    over-subtraction is visible instead of being silently clipped away.
    """

    matching_vector: NumericArray
    signed_residual: NumericArray
    aligned_candidate: NumericArray
    comparison_mask: BoolArray
    residual_mask: BoolArray
    scale_factor: float
    fit_improvement_fraction: float
    negative_point_fraction: float
    negative_energy_fraction: float
    common_point_count: int


def _one_dimensional_numeric(values: ArrayLike, *, name: str) -> NumericArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    return array


def _integer(value: int, *, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _readonly_copy(values: NumericArray | BoolArray) -> NumericArray | BoolArray:
    copied = np.array(values, copy=True)
    copied.setflags(write=False)
    return copied


def _normalise_support_runs(
    size: int,
    start_idx: int,
    end_idx: int,
    support_runs: Iterable[Iterable[int]] | None,
) -> SupportRuns:
    """Validate and merge inclusive native-grid support runs.

    ``None`` is the legacy-cache sentinel and falls back to the inclusive
    ``start_idx:end_idx`` interval.  An explicitly empty iterable represents
    an explicitly empty reference support.  Overlapping or adjacent runs are
    merged so counts always describe the union rather than double-counting.
    """

    length = _integer(size, name="size")
    if length < 0:
        raise ValueError("size must be non-negative")
    if support_runs is None:
        start = _integer(start_idx, name="start_idx")
        end = _integer(end_idx, name="end_idx")
        return ((start, end),) if 0 <= start <= end < length else ()
    if isinstance(support_runs, (str, bytes)):
        raise TypeError("support_runs must be an iterable of inclusive index pairs")

    validated: list[tuple[int, int]] = []
    for run_index, raw_run in enumerate(support_runs):
        if isinstance(raw_run, (str, bytes)):
            raise TypeError(f"support_runs[{run_index}] must be an index pair")
        try:
            pair = tuple(raw_run)
        except TypeError as exc:
            raise TypeError(
                f"support_runs[{run_index}] must be an iterable index pair"
            ) from exc
        if len(pair) != 2:
            raise ValueError(
                f"support_runs[{run_index}] must contain exactly two indices"
            )
        run_start = _integer(pair[0], name=f"support_runs[{run_index}][0]")
        run_end = _integer(pair[1], name=f"support_runs[{run_index}][1]")
        if not 0 <= run_start <= run_end < length:
            raise ValueError(
                f"support_runs[{run_index}] is outside the grid or reversed"
            )
        validated.append((run_start, run_end))

    if not validated:
        return ()
    validated.sort()
    merged: list[tuple[int, int]] = [validated[0]]
    for run_start, run_end in validated[1:]:
        previous_start, previous_end = merged[-1]
        if run_start <= previous_end + 1:
            merged[-1] = (previous_start, max(previous_end, run_end))
        else:
            merged.append((run_start, run_end))
    return tuple(merged)


def _support_runs_point_count(support_runs: Iterable[Iterable[int]]) -> int:
    """Count an inclusive run union when no surrounding grid is available."""

    materialized = tuple(tuple(run) for run in support_runs)
    if not materialized:
        return 0
    maximum = max(
        _integer(run[1], name="support_runs end")
        if len(run) == 2
        else -1
        for run in materialized
    )
    if maximum < 0:
        raise ValueError("support_runs must contain valid non-negative index pairs")
    normalized = _normalise_support_runs(maximum + 1, 0, -1, materialized)
    return int(sum(end - start + 1 for start, end in normalized))


def shift_candidate(candidate: ArrayLike, shift_points: int) -> NumericArray:
    """Shift a candidate without wraparound, filling uncovered points with zero.

    Positive shifts move values toward larger array indices, matching the
    convention in the current RamanPhaseID refinement code.
    """

    candidate_array = _one_dimensional_numeric(candidate, name="candidate")
    shift = _integer(shift_points, name="shift_points")
    size = int(candidate_array.size)
    shifted = np.zeros_like(candidate_array)
    if size == 0 or abs(shift) >= size:
        return shifted
    if shift >= 0:
        shifted[shift:] = candidate_array[: size - shift]
    else:
        shifted[: size + shift] = candidate_array[-shift:]
    return shifted


def build_residual_projection(
    query: ArrayLike,
    candidate: ArrayLike,
    query_mask: ArrayLike,
    start_idx: int,
    end_idx: int,
    fitted_shift_points: int,
    *,
    minimum_common_points: int = 20,
    support_edge_guard_points: int = 3,
    support_runs: Iterable[Iterable[int]] | None = None,
) -> ResidualProjection:
    """Fit and subtract one aligned trace while preserving signed residuals.

    The non-negative least-squares coefficient is solved on exact common
    support.  No upper coefficient clamp and no zero clipping are applied.
    The returned matching vector is normalized by maximum absolute residual,
    so its sign remains intact.
    """

    query_array = _one_dimensional_numeric(query, name="query").astype(float, copy=False)
    candidate_array = _one_dimensional_numeric(candidate, name="candidate").astype(
        float, copy=False
    )
    if query_array.size != candidate_array.size:
        raise ValueError("query and candidate must have equal length")
    mask_array = np.asarray(query_mask)
    if mask_array.ndim != 1 or mask_array.size != query_array.size:
        raise ValueError("query_mask must be one-dimensional and match query length")
    if not np.issubdtype(mask_array.dtype, np.bool_):
        raise TypeError("query_mask must contain boolean values")
    if not np.all(np.isfinite(query_array)) or not np.all(np.isfinite(candidate_array)):
        raise ValueError("query and candidate must contain only finite values")

    minimum_points = _integer(minimum_common_points, name="minimum_common_points")
    if minimum_points < 2:
        raise ValueError("minimum_common_points must be at least 2")
    edge_guard = _integer(
        support_edge_guard_points,
        name="support_edge_guard_points",
    )
    if edge_guard < 0:
        raise ValueError("support_edge_guard_points must be non-negative")
    shift = _integer(fitted_shift_points, name="fitted_shift_points")
    aligned = shift_candidate(candidate_array, shift)
    common = aligned_support_mask(
        mask_array,
        start_idx,
        end_idx,
        shift,
        support_runs=support_runs,
    )
    common_count = int(np.count_nonzero(common))
    if common_count < minimum_points:
        raise ValueError(
            f"residual projection needs at least {minimum_points} common points"
        )

    # Subtracting a truncated reference with a non-zero edge would otherwise
    # switch its contribution on/off in one grid sample.  Keep the measured
    # regions on both sides available, but separate them with a short masked
    # guard band so the artificial jump cannot be plotted, differentiated, or
    # interpreted as a peak by residual rematching.
    residual_mask = np.array(mask_array, dtype=bool, copy=True)
    if edge_guard:
        boundaries = np.flatnonzero(common[1:] != common[:-1]) + 1
        for boundary in boundaries:
            if not (mask_array[boundary - 1] and mask_array[boundary]):
                continue
            start = max(0, int(boundary) - edge_guard)
            stop = min(residual_mask.size, int(boundary) + edge_guard)
            residual_mask[start:stop] = False

    denominator = float(np.dot(aligned[common], aligned[common]))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("candidate has no finite energy on common support")
    coefficient = float(np.dot(query_array[common], aligned[common]) / denominator)
    scale_factor = max(0.0, coefficient)

    residual = np.array(query_array, dtype=float, copy=True)
    residual[common] -= scale_factor * aligned[common]
    residual[~mask_array] = 0.0

    before_sse = float(np.dot(query_array[common], query_array[common]))
    after_sse = float(np.dot(residual[common], residual[common]))
    improvement = (
        max(0.0, min(1.0, (before_sse - after_sse) / before_sse))
        if before_sse > 1e-12
        else 0.0
    )

    common_residual = residual[common]
    amplitude = max(1e-12, float(np.max(np.abs(query_array[common]))))
    negative = common_residual < (-1e-6 * amplitude)
    negative_point_fraction = float(np.count_nonzero(negative) / common_count)
    squared = common_residual * common_residual
    residual_energy = float(np.sum(squared))
    negative_energy_fraction = (
        float(np.sum(squared[negative]) / residual_energy)
        if residual_energy > 1e-12
        else 0.0
    )

    valid_residual = residual[residual_mask]
    normalizer = float(np.max(np.abs(valid_residual))) if valid_residual.size else 0.0
    matching_vector = residual / normalizer if normalizer > 1e-12 else np.zeros_like(residual)
    matching_vector[~residual_mask] = 0.0

    return ResidualProjection(
        matching_vector=_readonly_copy(matching_vector.astype(np.float32)),
        signed_residual=_readonly_copy(residual),
        aligned_candidate=_readonly_copy(aligned),
        comparison_mask=_readonly_copy(common.astype(bool, copy=False)),
        residual_mask=_readonly_copy(residual_mask),
        scale_factor=scale_factor,
        fit_improvement_fraction=improvement,
        negative_point_fraction=negative_point_fraction,
        negative_energy_fraction=negative_energy_fraction,
        common_point_count=common_count,
    )


def _rankdata_average(values: ArrayLike) -> NumericArray:
    array = np.asarray(values, dtype=float)
    ranks = np.empty(array.size, dtype=float)
    order = np.argsort(array, kind="mergesort")
    index = 0
    while index < array.size:
        end = index
        while end + 1 < array.size and array[order[end + 1]] == array[order[index]]:
            end += 1
        ranks[order[index : end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    return ranks


def _spearman_correlation(first: NumericArray, second: NumericArray) -> float:
    if first.size < 3 or second.size < 3:
        return 0.0
    first_ranks = _rankdata_average(first)
    second_ranks = _rankdata_average(second)
    if float(np.std(first_ranks)) <= 1e-12 or float(np.std(second_ranks)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(first_ranks, second_ranks)[0, 1])
    return float(np.clip(value, -1.0, 1.0)) if math.isfinite(value) else 0.0


def _normalise_positive_peak(values: NumericArray) -> NumericArray:
    array = np.asarray(values, dtype=float)
    result = np.zeros_like(array)
    finite = np.isfinite(array)
    if not np.any(finite):
        return result
    peak = float(np.max(array[finite]))
    if not math.isfinite(peak) or peak <= 1e-12:
        peak = float(np.max(np.abs(array[finite])))
    if math.isfinite(peak) and peak > 1e-12:
        result[finite] = array[finite] / peak
    return result


def _weighted_peaks(
    values: NumericArray,
    *,
    max_peaks: int = 80,
    minimum_run_points: int = 5,
    minimum_signal: float = 1e-9,
    minimum_prominence_absolute: float = 1e-6,
    minimum_prominence_fraction: float = 0.03,
    minimum_distance_points: int = 3,
) -> tuple[NDArray[np.int_], NumericArray]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if array.size < int(minimum_run_points) or not np.any(finite):
        return np.array([], dtype=int), np.array([], dtype=float)
    array = array.copy()
    array[~finite] = 0.0
    maximum = float(np.max(array))
    if not math.isfinite(maximum) or maximum <= float(minimum_signal):
        return np.array([], dtype=int), np.array([], dtype=float)
    minimum_prominence = max(
        float(minimum_prominence_absolute),
        float(minimum_prominence_fraction) * maximum,
    )
    if _scipy_find_peaks is not None:
        peaks, properties = _scipy_find_peaks(
            array,
            prominence=minimum_prominence,
            distance=int(minimum_distance_points),
        )
        if peaks.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        weights = np.asarray(properties.get("prominences", array[peaks]), dtype=float)
    else:
        candidates = np.where(
            (array[1:-1] > array[:-2]) & (array[1:-1] >= array[2:])
        )[0] + 1
        if candidates.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        weights = array[candidates] - np.maximum(
            array[candidates - 1], array[candidates + 1]
        )
        retained = weights >= minimum_prominence
        peaks = candidates[retained]
        weights = weights[retained]
        if peaks.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
    if peaks.size > int(max_peaks):
        retained = np.argsort(-weights)[: int(max_peaks)]
        peaks = peaks[retained]
        weights = weights[retained]
    order = np.argsort(peaks)
    return peaks[order].astype(int), weights[order].astype(float)


def _contiguous_true_runs(mask: BoolArray) -> tuple[tuple[int, int], ...]:
    """Return half-open contiguous runs from a one-dimensional boolean mask."""

    true_indices = np.flatnonzero(mask)
    if true_indices.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(true_indices) > 1)
    starts = np.concatenate((true_indices[:1], true_indices[breaks + 1]))
    ends = np.concatenate((true_indices[breaks] + 1, true_indices[-1:] + 1))
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _weighted_peaks_in_runs(
    values: NumericArray,
    mask: BoolArray,
    *,
    max_peaks: int = 80,
    minimum_run_points: int = 5,
    minimum_signal: float = 1e-9,
    minimum_prominence_absolute: float = 1e-6,
    minimum_prominence_fraction: float = 0.03,
    minimum_distance_points: int = 3,
) -> tuple[NDArray[np.int_], NumericArray, NDArray[np.int_]]:
    """Find peaks per valid run while retaining original grid coordinates."""

    all_indices: list[NDArray[np.int_]] = []
    all_weights: list[NumericArray] = []
    all_run_ids: list[NDArray[np.int_]] = []
    for run_id, (start, end) in enumerate(_contiguous_true_runs(mask)):
        local_indices, local_weights = _weighted_peaks(
            np.asarray(values[start:end], dtype=float),
            max_peaks=max_peaks,
            minimum_run_points=minimum_run_points,
            minimum_signal=minimum_signal,
            minimum_prominence_absolute=minimum_prominence_absolute,
            minimum_prominence_fraction=minimum_prominence_fraction,
            minimum_distance_points=minimum_distance_points,
        )
        if local_indices.size == 0:
            continue
        all_indices.append((local_indices + start).astype(int, copy=False))
        all_weights.append(local_weights)
        all_run_ids.append(np.full(local_indices.size, run_id, dtype=int))
    if not all_indices:
        empty_indices = np.array([], dtype=int)
        return empty_indices, np.array([], dtype=float), empty_indices.copy()

    indices = np.concatenate(all_indices)
    weights = np.concatenate(all_weights)
    run_ids = np.concatenate(all_run_ids)
    if indices.size > int(max_peaks):
        retained = np.argsort(-weights, kind="stable")[: int(max_peaks)]
        indices = indices[retained]
        weights = weights[retained]
        run_ids = run_ids[retained]
    order = np.argsort(indices, kind="stable")
    return (
        indices[order].astype(int, copy=False),
        weights[order].astype(float, copy=False),
        run_ids[order].astype(int, copy=False),
    )


def _match_weighted_peaks(
    query_indices: NDArray[np.int_],
    query_weights: NumericArray,
    candidate_indices: NDArray[np.int_],
    candidate_weights: NumericArray,
    *,
    tolerance_points: int,
) -> tuple[float, list[tuple[int, int]]]:
    if query_indices.size == 0 or candidate_indices.size == 0:
        return 0.0, []
    used_candidate = np.zeros(candidate_indices.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    matched_weight = 0.0
    for query_position in np.argsort(-query_weights):
        distance = np.abs(candidate_indices - query_indices[query_position])
        choices = np.where(
            (~used_candidate) & (distance <= int(tolerance_points))
        )[0]
        if choices.size == 0:
            continue
        best = max(choices, key=lambda item: (candidate_weights[item], -distance[item]))
        used_candidate[best] = True
        pairs.append((int(query_position), int(best)))
        matched_weight += 0.5 * float(
            query_weights[query_position] + candidate_weights[best]
        )
    return matched_weight, pairs


def peak_consistency_score(
    query: ArrayLike,
    aligned_candidate: ArrayLike,
    comparison_mask: ArrayLike,
    *,
    tolerance_points: int = 5,
    f1_weight: float = 0.75,
    max_peaks: int = 80,
    minimum_run_points: int = 5,
    minimum_signal: float = 1e-9,
    minimum_prominence_absolute: float = 1e-6,
    minimum_prominence_fraction: float = 0.03,
    minimum_distance_points: int = 3,
    minimum_support_points: int = 20,
) -> tuple[float, float, float]:
    """Return peak-location F1, relative-height agreement, and their blend.

    Peak finding and pairing are performed independently in each contiguous
    valid-mask run, using original grid coordinates.  Missing reference ranges
    are therefore never compressed away or bridged by the peak tolerance.
    """

    query_array = _one_dimensional_numeric(query, name="query").astype(float, copy=False)
    candidate_array = _one_dimensional_numeric(
        aligned_candidate, name="aligned_candidate"
    ).astype(float, copy=False)
    mask = np.asarray(comparison_mask, dtype=bool)
    if query_array.size != candidate_array.size or mask.shape != query_array.shape:
        raise ValueError("query, aligned_candidate, and comparison_mask must share one grid")
    weight = float(f1_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("f1_weight must be between zero and one")
    if int(tolerance_points) < 0:
        raise ValueError("tolerance_points must be non-negative")
    if int(np.count_nonzero(mask)) < int(minimum_support_points):
        return 0.0, 0.0, 0.0
    query_local = np.asarray(query_array[mask], dtype=float)
    candidate_local = np.asarray(candidate_array[mask], dtype=float)
    query_normalized = np.zeros_like(query_array, dtype=float)
    candidate_normalized = np.zeros_like(candidate_array, dtype=float)
    query_normalized[mask] = _normalise_positive_peak(
        query_local - np.min(query_local)
    )
    candidate_normalized[mask] = _normalise_positive_peak(
        candidate_local - np.min(candidate_local)
    )
    query_indices, query_weights, query_run_ids = _weighted_peaks_in_runs(
        query_normalized,
        mask,
        max_peaks=max_peaks,
        minimum_run_points=minimum_run_points,
        minimum_signal=minimum_signal,
        minimum_prominence_absolute=minimum_prominence_absolute,
        minimum_prominence_fraction=minimum_prominence_fraction,
        minimum_distance_points=minimum_distance_points,
    )
    candidate_indices, candidate_weights, candidate_run_ids = (
        _weighted_peaks_in_runs(
            candidate_normalized,
            mask,
            max_peaks=max_peaks,
            minimum_run_points=minimum_run_points,
            minimum_signal=minimum_signal,
            minimum_prominence_absolute=minimum_prominence_absolute,
            minimum_prominence_fraction=minimum_prominence_fraction,
            minimum_distance_points=minimum_distance_points,
        )
    )
    if query_indices.size == 0 or candidate_indices.size == 0:
        return 0.0, 0.0, 0.0
    matched_weight = 0.0
    pairs: list[tuple[int, int]] = []
    for run_id in np.intersect1d(query_run_ids, candidate_run_ids):
        query_positions = np.flatnonzero(query_run_ids == run_id)
        candidate_positions = np.flatnonzero(candidate_run_ids == run_id)
        run_weight, run_pairs = _match_weighted_peaks(
            query_indices[query_positions],
            query_weights[query_positions],
            candidate_indices[candidate_positions],
            candidate_weights[candidate_positions],
            tolerance_points=int(tolerance_points),
        )
        matched_weight += run_weight
        pairs.extend(
            (
                int(query_positions[query_position]),
                int(candidate_positions[candidate_position]),
            )
            for query_position, candidate_position in run_pairs
        )
    total_weight = float(np.sum(query_weights) + np.sum(candidate_weights))
    if total_weight <= 1e-12 or matched_weight <= 0.0 or not pairs:
        return 0.0, 0.0, 0.0
    peak_f1 = float(np.clip((2.0 * matched_weight) / total_weight, 0.0, 1.0))
    query_heights = np.array(
        [query_normalized[query_indices[first]] for first, _second in pairs],
        dtype=float,
    )
    candidate_heights = np.array(
        [candidate_normalized[candidate_indices[second]] for _first, second in pairs],
        dtype=float,
    )
    height_rho = _spearman_correlation(query_heights, candidate_heights)
    score = float(
        np.clip(weight * peak_f1 + (1.0 - weight) * max(0.0, height_rho), 0.0, 1.0)
    )
    return score, peak_f1, height_rho


def topk_cosine_subset(
    query: ArrayLike,
    matrix: NumericArray,
    metadata: Iterable[Mapping[str, Any]],
    subset_ids: ArrayLike,
    topk: int,
    *,
    support_mask: ArrayLike,
    chunk_rows: int = 1024,
    remove_query_local_offset: bool = True,
) -> list[int]:
    """Screen a subset exactly on each reference's common range support.

    Library rows remain locally offset-invariant. A signed residual query can
    disable its own offset removal so negative over-subtraction lowers, rather
    than spuriously raises, similarity to a positive reference spectrum.
    """

    rows_metadata = tuple(metadata)
    ids = np.asarray(subset_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or int(topk) <= 0:
        return []
    if np.ndim(matrix) != 2:
        raise ValueError("matrix must be two-dimensional")
    query_array = np.asarray(query, dtype=np.float32).copy()
    support = np.asarray(support_mask, dtype=bool)
    if (
        query_array.ndim != 1
        or support.shape != query_array.shape
        or matrix.shape[1] != query_array.size
    ):
        raise ValueError("query, support_mask, and database vectors must share one grid")
    if (
        np.any(ids < 0)
        or np.any(ids >= matrix.shape[0])
        or np.any(ids >= len(rows_metadata))
    ):
        raise IndexError("subset_ids contains an out-of-range database row")
    query_array[~np.isfinite(query_array)] = 0.0
    support_indices = np.flatnonzero(support)
    if support_indices.size == 0 or not np.any(query_array[support_indices]):
        return []
    query_support = query_array[support_indices]
    similarities = np.full(ids.size, -np.inf, dtype=np.float64)
    block_size = max(1, int(chunk_rows))
    for offset in range(0, ids.size, block_size):
        stop = min(ids.size, offset + block_size)
        row_ids = ids[offset:stop]
        starts = np.array(
            [rows_metadata[int(index)].get("start_idx", 0) for index in row_ids],
            dtype=np.int64,
        )
        ends = np.array(
            [rows_metadata[int(index)].get("end_idx", -1) for index in row_ids],
            dtype=np.int64,
        )
        valid_coverage = (starts >= 0) & (ends >= starts) & (ends < query_array.size)
        block = np.array(
            matrix[np.ix_(row_ids, support_indices)],
            dtype=np.float32,
            copy=True,
        )
        common = (
            valid_coverage[:, None]
            & (support_indices[None, :] >= starts[:, None])
            & (support_indices[None, :] <= ends[:, None])
        )
        # Preserve the vectorized contiguous fast path.  New caches also store
        # a one-run ``support_runs`` value, so only truly gapped rows need a
        # Python-level override.
        for block_row, row_id in enumerate(row_ids):
            row = rows_metadata[int(row_id)]
            raw_runs = row.get("support_runs")
            if raw_runs is None:
                continue
            exact_runs = _normalise_support_runs(
                query_array.size,
                int(row.get("start_idx", 0)),
                int(row.get("end_idx", -1)),
                raw_runs,
            )
            legacy_run = (
                ((int(starts[block_row]), int(ends[block_row])),)
                if valid_coverage[block_row]
                else ()
            )
            if exact_runs == legacy_run:
                continue
            exact_common = np.zeros(support_indices.size, dtype=bool)
            for run_start, run_end in exact_runs:
                exact_common |= (support_indices >= run_start) & (
                    support_indices <= run_end
                )
            common[block_row, :] = exact_common
        block[~common] = 0.0
        block[~np.isfinite(block)] = 0.0
        has_common = np.any(common, axis=1)
        if remove_query_local_offset:
            query_minimum = np.min(
                np.where(common, query_support[None, :], np.inf), axis=1
            )
        else:
            query_minimum = np.zeros(row_ids.size, dtype=np.float32)
        database_minimum = np.min(np.where(common, block, np.inf), axis=1)
        query_minimum[~has_common] = 0.0
        database_minimum[~has_common] = 0.0
        block -= database_minimum[:, None]
        block[~common] = 0.0
        query_sum = np.einsum(
            "ij,j->i", common, query_support, dtype=np.float64, optimize=True
        )
        query_raw_square = np.einsum(
            "ij,j->i",
            common,
            query_support * query_support,
            dtype=np.float64,
            optimize=True,
        )
        common_count = np.count_nonzero(common, axis=1)
        query_square = (
            query_raw_square
            - (2.0 * query_minimum * query_sum)
            + (common_count * query_minimum * query_minimum)
        )
        query_square = np.maximum(query_square, 0.0)
        dots = np.einsum(
            "ij,j->i", block, query_support, dtype=np.float64, optimize=True
        )
        dots -= query_minimum * np.sum(block, axis=1, dtype=np.float64)
        database_square = np.einsum(
            "ij,ij->i", block, block, dtype=np.float64, optimize=True
        )
        denominator = np.sqrt(database_square * query_square)
        valid = (
            has_common
            & (denominator > 0.0)
            & np.isfinite(dots)
            & np.isfinite(denominator)
        )
        chunk_similarities = similarities[offset:stop]
        chunk_similarities[valid] = np.clip(
            dots[valid] / denominator[valid], -1.0, 1.0
        )
    valid_positions = np.flatnonzero(np.isfinite(similarities))
    if valid_positions.size == 0:
        return []
    take = min(int(topk), int(valid_positions.size))
    valid_similarities = similarities[valid_positions]
    if take >= valid_positions.size:
        order = np.argsort(-valid_similarities, kind="stable")
    else:
        partition = np.argpartition(-valid_similarities, take - 1)[:take]
        order = partition[
            np.argsort(-valid_similarities[partition], kind="stable")
        ]
    return ids[valid_positions[order]].tolist()


_RRUFF_ACQUISITION_SUFFIX_RE = re.compile(
    r"__Raman_Data_(?:raw|processed)(?:__[0-9a-f]+)?$",
    flags=re.IGNORECASE,
)


def _reference_acquisition_group(
    metadata: Mapping[str, Any],
    provenance: Mapping[str, Any],
    independence_group: str,
) -> str:
    """Return a duplicate-safe acquisition identity for one reference row.

    RRUFF raw and processed exports encode the same acquisition before their
    ``Raman_Data_*`` suffix. Different orientations, wavelengths, or scans
    retain distinct prefixes. Other databases remain conservatively grouped
    by their explicit acquisition metadata or specimen identity.
    """

    explicit = _first_nonempty_text(
        metadata,
        ("acquisition_group", "acquisition_id", "spectrum_id"),
    ) or _first_nonempty_text(
        provenance,
        ("acquisition_group", "acquisition_id", "spectrum_id"),
    )
    if explicit:
        return explicit

    filename = _first_nonempty_text(
        metadata,
        ("orig_filename", "filename", "path"),
    )
    stem = Path(filename).stem
    prefix = _RRUFF_ACQUISITION_SUFFIX_RE.sub("", stem)
    database = _first_nonempty_text(provenance, ("database",)).casefold()
    source_root = str(metadata.get("source_root", "")).strip().casefold()
    if prefix != stem and (database == "rruff" or source_root == "rruff"):
        return f"rruff-acquisition:{prefix.casefold()}"
    return str(independence_group).strip()


def refine_and_rank(
    query: ArrayLike,
    query_mask: ArrayLike,
    candidate_indices: Iterable[int],
    pack: Mapping[str, Any],
    top_n: int,
    *,
    parameters: MatchingParameters | None = None,
) -> list[dict[str, Any]]:
    """Align shortlisted rows and expose every component of the earned score."""

    active = parameters or MatchingParameters()
    query_array = np.asarray(query)
    mask_array = np.asarray(query_mask, dtype=bool)
    matrix = pack["X"]
    metadata = pack["meta"]
    grid_step = float(pack.get("grid_info", {}).get("step", 1.0))
    query_gradient = np.gradient(query_array)
    scored: list[dict[str, Any]] = []
    for raw_index in candidate_indices:
        index = int(raw_index)
        row = metadata[index]
        if row.get("l2", 0.0) <= 0.0:
            continue
        start_idx = int(row.get("start_idx", 0))
        end_idx = int(row.get("end_idx", -1))
        normalized_support_runs = _normalise_support_runs(
            query_array.size,
            start_idx,
            end_idx,
            row.get("support_runs"),
        )
        if not normalized_support_runs:
            continue
        alignment = best_aligned_score(
            query_array,
            matrix[index, :],
            mask_array,
            start_idx,
            end_idx,
            max_shift=int(active.maximum_shift_points),
            gradient_weight=float(active.gradient_weight),
            grid_step_cm1=grid_step,
            support_runs=normalized_support_runs,
            query_gradient=query_gradient,
            score_tie_tolerance=float(active.alignment_score_tie_tolerance),
            remove_query_local_offset=bool(active.remove_query_local_offset),
        )
        if alignment.spectral_similarity < 0.0:
            continue
        peak_score, peak_f1, peak_rho = peak_consistency_score(
            query_array,
            alignment.aligned_candidate,
            alignment.comparison_mask,
            tolerance_points=int(active.peak_tolerance_points),
            f1_weight=float(active.peak_f1_weight),
            max_peaks=int(active.peak_detection_max_peaks),
            minimum_run_points=int(active.peak_detection_minimum_run_points),
            minimum_signal=float(active.peak_detection_minimum_signal),
            minimum_prominence_absolute=float(
                active.peak_detection_minimum_prominence_absolute
            ),
            minimum_prominence_fraction=float(
                active.peak_detection_minimum_prominence_fraction
            ),
            minimum_distance_points=int(
                active.peak_detection_minimum_distance_points
            ),
            minimum_support_points=int(
                active.peak_consistency_minimum_support_points
            ),
        )
        rank = compose_rank_components(
            alignment,
            peak_score,
            spectral_similarity_weight=float(active.spectral_similarity_weight),
        )
        source_path = Path(str(row.get("path", "")))
        provenance_value = row.get("provenance", {})
        provenance: Mapping[str, Any] = (
            provenance_value if isinstance(provenance_value, Mapping) else {}
        )
        accession = _first_nonempty_text(
            row,
            ("accession", "rruff_id", "rod_id", "source_accession"),
        )
        if not accession:
            accession = _first_nonempty_text(
                provenance,
                ("accession", "rruff_id", "rod_id", "source_accession"),
            )
        independence_group = _first_nonempty_text(
            row,
            ("independence_group", "specimen_id"),
        )
        if not independence_group:
            independence_group = _first_nonempty_text(
                provenance,
                ("independence_group", "specimen_id"),
            )
        if not independence_group:
            provenance_source = _first_nonempty_text(
                provenance,
                ("source_accession", "source"),
            )
            independence_group = (
                accession
                or (f"provenance-source:{provenance_source}" if provenance_source else "")
                or f"unresolved:{phase_key(row['name'])}"
            )
        acquisition_group = _reference_acquisition_group(
            row,
            provenance,
            independence_group,
        )
        evidence = alignment.evidence
        scored.append(
            {
                "name": row["name"],
                "formula": row["formula"],
                "flag": row.get("flag", ""),
                "similarity": alignment.spectral_similarity,
                "shape_similarity": alignment.shape_similarity,
                "gradient_similarity": alignment.gradient_similarity,
                "pcs": peak_score,
                "pcs_f1": peak_f1,
                "pcs_rho": peak_rho,
                "filename": row.get("filename", ""),
                "orig_filename": row.get("orig_filename", ""),
                "path": source_path,
                "reference_id": str(source_path),
                "source": row.get("source", ""),
                "provenance": dict(provenance),
                "accession": accession,
                "independence_group": independence_group,
                "acquisition_group": acquisition_group,
                "db_idx": index,
                "shift": evidence.fitted_shift_points,
                "shift_cm1": evidence.fitted_shift_cm1,
                "shift_boundary_hit": evidence.shift_search_boundary_hit,
                "grid_boundary_clipped": evidence.reference_support_clipped_at_grid_boundary,
                "common_point_count": evidence.common_point_count,
                "requested_point_count": evidence.requested_point_count,
                "reference_support_point_count": evidence.reference_support_point_count,
                "shifted_reference_support_point_count": evidence.shifted_reference_support_point_count,
                "maximum_shift_points": evidence.maximum_shift_points,
                "coverage_fraction": evidence.query_coverage_fraction,
                "reference_overlap_fraction": evidence.reference_overlap_fraction,
                "rank_score": rank.final_rank_score,
                "gradient_weight_within_similarity": rank.gradient_weight_within_similarity,
                "spectral_similarity_weight": rank.spectral_similarity_weight,
                "peak_consistency_weight": rank.peak_consistency_weight,
                "rank_components": {
                    "shape": rank.shape_similarity,
                    "gradient": rank.gradient_similarity,
                    "pcs": rank.peak_consistency,
                    "shape_contribution": rank.shape_contribution,
                    "gradient_contribution": rank.gradient_contribution,
                    "pcs_contribution": rank.peak_consistency_contribution,
                },
                "start_idx": start_idx,
                "end_idx": end_idx,
                "support_runs": normalized_support_runs,
                "db_baseline": bool(row.get("db_baseline", False)),
            }
        )
    scored.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return scored[: max(0, int(top_n))]


def reference_support_mask(
    size: int,
    start_idx: int,
    end_idx: int,
    *,
    support_runs: Iterable[Iterable[int]] | None = None,
) -> BoolArray:
    """Return exact native reference support on the common grid.

    ``support_runs`` contains inclusive grid-index pairs and takes precedence
    when supplied.  ``None`` retains compatibility with legacy caches by using
    the inclusive ``start_idx:end_idx`` interval.
    """

    length = _integer(size, name="size")
    runs = _normalise_support_runs(length, start_idx, end_idx, support_runs)
    support = np.zeros(length, dtype=bool)
    for start, end in runs:
        support[start : end + 1] = True
    return support


def shift_support_mask(support_mask: ArrayLike, shift_points: int) -> BoolArray:
    """Shift a support mask exactly as :func:`shift_candidate` shifts values."""

    support = np.asarray(support_mask, dtype=bool)
    if support.ndim != 1:
        raise ValueError("support_mask must be one-dimensional")
    shift = _integer(shift_points, name="shift_points")
    size = int(support.size)
    shifted = np.zeros(size, dtype=bool)
    if size == 0 or abs(shift) >= size:
        return shifted
    if shift >= 0:
        shifted[shift:] = support[: size - shift]
    else:
        shifted[: size + shift] = support[-shift:]
    return shifted


def aligned_support_mask(
    query_mask: ArrayLike,
    start_idx: int,
    end_idx: int,
    shift_points: int,
    *,
    support_runs: Iterable[Iterable[int]] | None = None,
) -> BoolArray:
    """Intersect query support with the *shifted* native reference support.

    Moving candidate values must move their validity mask as well.  Keeping the
    unshifted ``start_idx:end_idx`` interval would score zero-filled values on
    one edge and discard valid aligned values on the other.
    """

    query_support = np.asarray(query_mask, dtype=bool)
    if query_support.ndim != 1:
        raise ValueError("query_mask must be one-dimensional")
    native_support = reference_support_mask(
        query_support.size,
        start_idx,
        end_idx,
        support_runs=support_runs,
    )
    shifted_support = shift_support_mask(native_support, shift_points)
    return query_support & shifted_support


def gradient_support_mask(comparison_mask: ArrayLike) -> BoolArray:
    """Erode contiguous support by one point for central-difference gradients."""

    mask = np.asarray(comparison_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("comparison_mask must be one-dimensional")
    gradient_mask = np.zeros_like(mask)
    if mask.size > 2:
        gradient_mask[1:-1] = mask[1:-1] & mask[:-2] & mask[2:]
    return gradient_mask


def masked_cosine(
    first: ArrayLike,
    second: ArrayLike,
    mask: ArrayLike,
    *,
    remove_local_offset: bool = False,
    remove_first_offset: bool | None = None,
    remove_second_offset: bool | None = None,
) -> float:
    """Cosine similarity on explicit support with independently chosen offsets.

    ``remove_local_offset`` remains the backwards-compatible shorthand for
    offsetting both inputs. Residual matching deliberately keeps the signed
    query centred on zero while still removing an arbitrary local offset from
    the non-negative library trace. Otherwise a deep negative over-subtraction
    trough is translated upward and can masquerade as broad positive phase
    evidence.
    """

    first_array = _one_dimensional_numeric(first, name="first")
    second_array = _one_dimensional_numeric(second, name="second")
    support = np.asarray(mask, dtype=bool)
    if second_array.shape != first_array.shape or support.shape != first_array.shape:
        raise ValueError("first, second, and mask must share one shape")
    if int(support.sum()) < 2:
        return -1.0
    aa = first_array[support]
    bb = second_array[support]
    offset_first = (
        bool(remove_local_offset)
        if remove_first_offset is None
        else bool(remove_first_offset)
    )
    offset_second = (
        bool(remove_local_offset)
        if remove_second_offset is None
        else bool(remove_second_offset)
    )
    if offset_first:
        aa = aa - np.min(aa)
    if offset_second:
        bb = bb - np.min(bb)
    first_norm = np.linalg.norm(aa)
    second_norm = np.linalg.norm(bb)
    if first_norm == 0.0 or second_norm == 0.0:
        return -1.0
    return float(np.dot(aa, bb) / (first_norm * second_norm))


def best_aligned_score(
    query: ArrayLike,
    candidate: ArrayLike,
    query_mask: ArrayLike,
    start_idx: int,
    end_idx: int,
    *,
    max_shift: int = 5,
    gradient_weight: float = DEFAULT_GRADIENT_WEIGHT,
    grid_step_cm1: float = 1.0,
    support_runs: Iterable[Iterable[int]] | None = None,
    query_gradient: ArrayLike | None = None,
    score_tie_tolerance: float = ALIGNMENT_SCORE_TIE_TOLERANCE,
    remove_query_local_offset: bool = True,
) -> AlignmentResult:
    """Return the best alignment and the exact candidate/support used to score it.

    Integer shifts from ``-max_shift`` through ``+max_shift`` are compared on
    exact shifted reference support.  Scores within ``score_tie_tolerance``
    (default :data:`ALIGNMENT_SCORE_TIE_TOLERANCE`) are tied, then resolved by greater
    common support, smallest absolute shift, and finally smaller signed shift.
    This avoids reporting a gratuitous boundary shift for numerically identical
    fits.  At shift zero, numerical operations remain legacy-compatible.

    A precomputed ``query_gradient`` can be supplied when refining many
    references against one query; omitting it preserves the standalone API.
    """

    query_array = _one_dimensional_numeric(query, name="query")
    candidate_array = _one_dimensional_numeric(candidate, name="candidate")
    support = np.asarray(query_mask, dtype=bool)
    if query_array.shape != candidate_array.shape or support.shape != query_array.shape:
        raise ValueError("query, candidate, and query_mask must share one shape")
    if query_array.size < 2:
        raise ValueError("alignment requires at least two grid points")

    maximum_shift = _integer(max_shift, name="max_shift")
    if maximum_shift < 0:
        raise ValueError("max_shift must be non-negative")
    weight = float(gradient_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("gradient_weight must be finite and between 0 and 1")
    step = float(grid_step_cm1)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("grid_step_cm1 must be finite and positive")
    tie_tolerance = float(score_tie_tolerance)
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("score_tie_tolerance must be finite and non-negative")

    if query_gradient is None:
        query_gradient_array = np.gradient(query_array)
    else:
        query_gradient_array = _one_dimensional_numeric(
            query_gradient,
            name="query_gradient",
        )
        if query_gradient_array.shape != query_array.shape:
            raise ValueError("query_gradient must share the query shape")
    candidate_gradient = np.gradient(candidate_array)
    native_reference_support = reference_support_mask(
        query_array.size,
        start_idx,
        end_idx,
        support_runs=support_runs,
    )

    # These shift-zero values are the deterministic fallback if every tested
    # comparison is invalid (-1 or NaN), matching the old best_k=0 fallback.
    best_shift = 0
    best_candidate = shift_candidate(candidate_array, 0)
    best_mask = support & native_reference_support
    best_gradient_mask = gradient_support_mask(best_mask)
    best_shape = masked_cosine(
        query_array,
        best_candidate,
        best_mask,
        remove_first_offset=bool(remove_query_local_offset),
        remove_second_offset=True,
    )
    best_gradient = (
        masked_cosine(
            query_gradient_array,
            shift_candidate(candidate_gradient, 0),
            best_gradient_mask,
        )
        if weight > 0.0
        else 0.0
    )
    best_similarity = -1.0
    best_common_points = int(np.count_nonzero(best_mask))

    shifts = (0,) + tuple(
        shift
        for shift in range(-maximum_shift, maximum_shift + 1)
        if shift != 0
    )
    for shift in shifts:
        aligned_candidate = shift_candidate(candidate_array, shift)
        shifted_support = shift_support_mask(native_reference_support, shift)
        comparison_mask = support & shifted_support
        shape_similarity = masked_cosine(
            query_array,
            aligned_candidate,
            comparison_mask,
            remove_first_offset=bool(remove_query_local_offset),
            remove_second_offset=True,
        )
        gradient_mask = gradient_support_mask(comparison_mask)
        if weight > 0.0:
            # The gradient support is eroded by one point, so shifting the
            # precomputed gradient is exactly equivalent on every scored point
            # to differentiating each zero-padded shifted allocation.
            aligned_candidate_gradient = shift_candidate(candidate_gradient, shift)
            gradient_similarity = masked_cosine(
                query_gradient_array,
                aligned_candidate_gradient,
                gradient_mask,
            )
            similarity = ((1.0 - weight) * shape_similarity) + (
                weight * gradient_similarity
            )
        else:
            gradient_similarity = 0.0
            similarity = shape_similarity

        if not math.isfinite(float(similarity)) or float(similarity) <= -1.0:
            continue
        common_points = int(np.count_nonzero(comparison_mask))
        score_delta = float(similarity) - best_similarity
        score_is_tied = abs(score_delta) <= tie_tolerance
        better_tie_break = score_is_tied and (
            common_points > best_common_points
            or (
                common_points == best_common_points
                and (
                    abs(shift) < abs(best_shift)
                    or (
                        abs(shift) == abs(best_shift)
                        and shift < best_shift
                    )
                )
            )
        )
        if score_delta > tie_tolerance or better_tie_break:
            best_similarity = float(similarity)
            best_shift = shift
            best_common_points = common_points
            best_shape = float(shape_similarity)
            best_gradient = float(gradient_similarity)
            best_candidate = aligned_candidate
            best_mask = comparison_mask
            best_gradient_mask = gradient_mask

    shifted_reference_support = shift_support_mask(native_reference_support, best_shift)
    requested_points = int(np.count_nonzero(support))
    native_reference_points = int(np.count_nonzero(native_reference_support))
    shifted_reference_points = int(np.count_nonzero(shifted_reference_support))
    common_points = int(np.count_nonzero(best_mask))
    query_coverage = common_points / requested_points if requested_points else 0.0
    reference_overlap = (
        common_points / shifted_reference_points if shifted_reference_points else 0.0
    )

    evidence = AlignmentEvidence(
        fitted_shift_points=int(best_shift),
        fitted_shift_cm1=float(best_shift * step),
        maximum_shift_points=int(maximum_shift),
        shift_search_boundary_hit=bool(
            maximum_shift > 0 and abs(best_shift) == maximum_shift
        ),
        reference_support_clipped_at_grid_boundary=bool(
            shifted_reference_points < native_reference_points
        ),
        requested_point_count=requested_points,
        reference_support_point_count=native_reference_points,
        shifted_reference_support_point_count=shifted_reference_points,
        common_point_count=common_points,
        query_coverage_fraction=float(query_coverage),
        reference_overlap_fraction=float(reference_overlap),
    )
    return AlignmentResult(
        shape_similarity=float(best_shape),
        gradient_similarity=float(best_gradient),
        spectral_similarity=float(best_similarity),
        gradient_weight=weight,
        evidence=evidence,
        aligned_candidate=_readonly_copy(best_candidate),
        comparison_mask=_readonly_copy(best_mask),
        gradient_comparison_mask=_readonly_copy(best_gradient_mask),
    )


def compose_rank_components(
    alignment: AlignmentResult,
    peak_consistency: float,
    *,
    spectral_similarity_weight: float = DEFAULT_FINAL_SIMILARITY_WEIGHT,
) -> RankComponents:
    """Compose the final rank while exposing each weighted contribution."""

    peak_score = float(peak_consistency)
    if not math.isfinite(peak_score) or not 0.0 <= peak_score <= 1.0:
        raise ValueError("peak_consistency must be finite and between 0 and 1")
    spectral_weight = float(spectral_similarity_weight)
    if not math.isfinite(spectral_weight) or not 0.0 <= spectral_weight <= 1.0:
        raise ValueError(
            "spectral_similarity_weight must be finite and between 0 and 1"
        )
    peak_weight = 1.0 - spectral_weight
    shape_weight = 1.0 - alignment.gradient_weight

    shape_contribution = (
        spectral_weight * shape_weight * alignment.shape_similarity
    )
    gradient_contribution = (
        spectral_weight
        * alignment.gradient_weight
        * alignment.gradient_similarity
    )
    peak_contribution = peak_weight * peak_score
    final_score = (
        spectral_weight * alignment.spectral_similarity
    ) + peak_contribution

    return RankComponents(
        shape_similarity=float(alignment.shape_similarity),
        gradient_similarity=float(alignment.gradient_similarity),
        peak_consistency=peak_score,
        spectral_similarity=float(alignment.spectral_similarity),
        final_rank_score=float(final_score),
        gradient_weight_within_similarity=float(alignment.gradient_weight),
        spectral_similarity_weight=spectral_weight,
        peak_consistency_weight=float(peak_weight),
        shape_contribution=float(shape_contribution),
        gradient_contribution=float(gradient_contribution),
        peak_consistency_contribution=float(peak_contribution),
    )


def _mapping_value(
    values: Mapping[str, Any],
    primary_key: str,
    *alternate_keys: str,
) -> Any:
    for key in (primary_key, *alternate_keys):
        if key in values:
            return values[key]
    alternatives = ", ".join(repr(key) for key in alternate_keys)
    suffix = f" (or {alternatives})" if alternatives else ""
    raise KeyError(f"missing required field {primary_key!r}{suffix}")


def _finite_mapping_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _mapping_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _mapping_count(value: Any, *, name: str) -> int:
    count = _integer(value, name=name)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _count_from_fraction(
    numerator: int,
    fraction: float,
    *,
    count_name: str,
    fraction_name: str,
) -> int:
    """Recover an integer denominator only when the ratio determines it."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{fraction_name} must be between 0 and 1")
    if numerator == 0:
        raise ValueError(
            f"cannot infer {count_name} from zero {fraction_name}; "
            f"include {count_name!r} explicitly"
        )
    if fraction == 0.0:
        raise ValueError(f"positive numerator is inconsistent with zero {fraction_name}")
    estimate = numerator / fraction
    count = int(round(estimate))
    if count < numerator or not math.isclose(
        numerator / count,
        fraction,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{fraction_name} does not determine a consistent integer {count_name}"
        )
    return count


def _assert_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"{name} is inconsistent with its component scores "
            f"({actual!r} != {expected!r})"
        )


def rank_components_from_mapping(
    result: Mapping[str, Any],
    *,
    gradient_weight: float = DEFAULT_GRADIENT_WEIGHT,
    spectral_similarity_weight: float = DEFAULT_FINAL_SIMILARITY_WEIGHT,
) -> RankComponents:
    """Convert a current matcher result mapping into checked rank evidence.

    The bridge deliberately verifies the stated similarity, final rank, and
    optional contribution fields.  A stale or partially mutated result record
    therefore fails clearly rather than becoming apparently precise evidence.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    nested_value = result.get("rank_components", {})
    if nested_value is None:
        nested_value = {}
    if not isinstance(nested_value, Mapping):
        raise TypeError("rank_components must be a mapping when provided")
    nested: Mapping[str, Any] = nested_value

    def component(top_key: str, nested_key: str) -> float:
        if top_key in result:
            return _finite_mapping_float(result[top_key], name=top_key)
        return _finite_mapping_float(
            _mapping_value(nested, nested_key),
            name=f"rank_components.{nested_key}",
        )

    shape = component("shape_similarity", "shape")
    gradient = component("gradient_similarity", "gradient")
    peak = component("pcs", "pcs")
    for key, expected in (
        ("shape", shape),
        ("gradient", gradient),
        ("pcs", peak),
    ):
        if key in nested:
            stated = _finite_mapping_float(
                nested[key],
                name=f"rank_components.{key}",
            )
            _assert_close(stated, expected, name=f"rank_components.{key}")
    spectral = _finite_mapping_float(
        _mapping_value(result, "similarity", "spectral_similarity"),
        name="similarity",
    )
    final_score = _finite_mapping_float(
        _mapping_value(result, "rank_score", "final_rank_score"),
        name="rank_score",
    )

    nested_gradient_weight = nested.get(
        "gradient_weight_within_similarity",
        nested.get("gradient_weight", gradient_weight),
    )
    active_gradient_weight = _finite_mapping_float(
        result.get("gradient_weight_within_similarity", nested_gradient_weight),
        name="gradient_weight_within_similarity",
    )
    nested_spectral_weight = nested.get(
        "spectral_similarity_weight",
        spectral_similarity_weight,
    )
    active_spectral_weight = _finite_mapping_float(
        result.get("spectral_similarity_weight", nested_spectral_weight),
        name="spectral_similarity_weight",
    )
    for name, weight in (
        ("gradient_weight_within_similarity", active_gradient_weight),
        ("spectral_similarity_weight", active_spectral_weight),
    ):
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 <= peak <= 1.0:
        raise ValueError("pcs must be between 0 and 1")

    peak_weight = 1.0 - active_spectral_weight
    if "peak_consistency_weight" in result:
        stated_peak_weight = _finite_mapping_float(
            result["peak_consistency_weight"],
            name="peak_consistency_weight",
        )
        _assert_close(
            stated_peak_weight,
            peak_weight,
            name="peak_consistency_weight",
        )

    expected_spectral = (
        (1.0 - active_gradient_weight) * shape
        + active_gradient_weight * gradient
    )
    _assert_close(spectral, expected_spectral, name="similarity")

    shape_contribution = (
        active_spectral_weight * (1.0 - active_gradient_weight) * shape
    )
    gradient_contribution = (
        active_spectral_weight * active_gradient_weight * gradient
    )
    peak_contribution = peak_weight * peak
    expected_final = shape_contribution + gradient_contribution + peak_contribution
    _assert_close(final_score, expected_final, name="rank_score")

    for key, expected in (
        ("shape_contribution", shape_contribution),
        ("gradient_contribution", gradient_contribution),
        ("pcs_contribution", peak_contribution),
        ("peak_consistency_contribution", peak_contribution),
    ):
        if key in nested:
            stated = _finite_mapping_float(
                nested[key],
                name=f"rank_components.{key}",
            )
            _assert_close(stated, expected, name=f"rank_components.{key}")

    return RankComponents(
        shape_similarity=shape,
        gradient_similarity=gradient,
        peak_consistency=peak,
        spectral_similarity=spectral,
        final_rank_score=final_score,
        gradient_weight_within_similarity=active_gradient_weight,
        spectral_similarity_weight=active_spectral_weight,
        peak_consistency_weight=peak_weight,
        shape_contribution=shape_contribution,
        gradient_contribution=gradient_contribution,
        peak_consistency_contribution=peak_contribution,
    )


def alignment_evidence_from_mapping(
    result: Mapping[str, Any],
    *,
    maximum_shift_points: int = 5,
    grid_step_cm1: float = 1.0,
) -> AlignmentEvidence:
    """Convert the app's result fields into auditable alignment evidence.

    Older result dictionaries omit three raw support counts.  For non-empty
    comparisons they can be reconstructed exactly from ``support_runs`` when
    present, otherwise from legacy inclusive native bounds and the two reported
    fractions.  Ambiguous zero-overlap records must provide counts explicitly;
    this function never guesses.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    fitted_shift = _integer(
        _mapping_value(result, "shift", "fitted_shift_points"),
        name="shift",
    )
    maximum_shift = _mapping_count(
        result.get("maximum_shift_points", maximum_shift_points),
        name="maximum_shift_points",
    )
    if abs(fitted_shift) > maximum_shift:
        raise ValueError("fitted shift exceeds maximum_shift_points")
    step = _finite_mapping_float(grid_step_cm1, name="grid_step_cm1")
    if step <= 0.0:
        raise ValueError("grid_step_cm1 must be positive")
    shift_cm1 = _finite_mapping_float(
        result.get("shift_cm1", fitted_shift * step),
        name="shift_cm1",
    )
    _assert_close(
        shift_cm1,
        fitted_shift * step,
        name="shift_cm1",
    )

    common = _mapping_count(
        _mapping_value(result, "common_point_count"),
        name="common_point_count",
    )
    coverage = _finite_mapping_float(
        _mapping_value(result, "coverage_fraction", "query_coverage_fraction"),
        name="coverage_fraction",
    )
    reference_overlap = _finite_mapping_float(
        _mapping_value(result, "reference_overlap_fraction"),
        name="reference_overlap_fraction",
    )
    for name, fraction in (
        ("coverage_fraction", coverage),
        ("reference_overlap_fraction", reference_overlap),
    ):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    if "requested_point_count" in result:
        requested = _mapping_count(
            result["requested_point_count"],
            name="requested_point_count",
        )
    else:
        requested = _count_from_fraction(
            common,
            coverage,
            count_name="requested_point_count",
            fraction_name="coverage_fraction",
        )

    if "reference_support_point_count" in result:
        native_reference = _mapping_count(
            result["reference_support_point_count"],
            name="reference_support_point_count",
        )
    else:
        if result.get("support_runs") is not None:
            native_reference = _support_runs_point_count(result["support_runs"])
        else:
            start = _integer(_mapping_value(result, "start_idx"), name="start_idx")
            end = _integer(_mapping_value(result, "end_idx"), name="end_idx")
            if start < 0 or end < start:
                raise ValueError("start_idx and end_idx do not define native support")
            native_reference = end - start + 1

    if "shifted_reference_support_point_count" in result:
        shifted_reference = _mapping_count(
            result["shifted_reference_support_point_count"],
            name="shifted_reference_support_point_count",
        )
    else:
        shifted_reference = _count_from_fraction(
            common,
            reference_overlap,
            count_name="shifted_reference_support_point_count",
            fraction_name="reference_overlap_fraction",
        )

    if common > requested:
        raise ValueError("common_point_count exceeds requested_point_count")
    if common > shifted_reference:
        raise ValueError(
            "common_point_count exceeds shifted_reference_support_point_count"
        )
    if shifted_reference > native_reference:
        raise ValueError(
            "shifted_reference_support_point_count exceeds native reference support"
        )
    expected_coverage = common / requested if requested else 0.0
    expected_reference_overlap = (
        common / shifted_reference if shifted_reference else 0.0
    )
    _assert_close(coverage, expected_coverage, name="coverage_fraction")
    _assert_close(
        reference_overlap,
        expected_reference_overlap,
        name="reference_overlap_fraction",
    )

    expected_shift_boundary = bool(
        maximum_shift > 0 and abs(fitted_shift) == maximum_shift
    )
    shift_boundary = _mapping_bool(
        result.get("shift_boundary_hit", expected_shift_boundary),
        name="shift_boundary_hit",
    )
    if shift_boundary != expected_shift_boundary:
        raise ValueError(
            "shift_boundary_hit is inconsistent with shift and maximum_shift_points"
        )
    expected_grid_clipped = shifted_reference < native_reference
    grid_clipped = _mapping_bool(
        result.get("grid_boundary_clipped", expected_grid_clipped),
        name="grid_boundary_clipped",
    )
    if grid_clipped != expected_grid_clipped:
        raise ValueError(
            "grid_boundary_clipped is inconsistent with reference support counts"
        )

    return AlignmentEvidence(
        fitted_shift_points=fitted_shift,
        fitted_shift_cm1=shift_cm1,
        maximum_shift_points=maximum_shift,
        shift_search_boundary_hit=shift_boundary,
        reference_support_clipped_at_grid_boundary=grid_clipped,
        requested_point_count=requested,
        reference_support_point_count=native_reference,
        shifted_reference_support_point_count=shifted_reference,
        common_point_count=common,
        query_coverage_fraction=coverage,
        reference_overlap_fraction=reference_overlap,
    )


def _first_nonempty_text(values: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key not in values or values[key] is None:
            continue
        text = str(values[key]).strip()
        if text:
            return text
    return ""


def reference_match_evidence_from_mapping(
    result: Mapping[str, Any],
    *,
    reference_id: str | None = None,
    independence_group: str | None = None,
    acquisition_group: str | None = None,
    source: str | None = None,
    maximum_shift_points: int = 5,
    grid_step_cm1: float = 1.0,
    gradient_weight: float = DEFAULT_GRADIENT_WEIGHT,
    spectral_similarity_weight: float = DEFAULT_FINAL_SIMILARITY_WEIGHT,
) -> ReferenceMatchEvidence:
    """Bridge one current result dictionary to typed reference evidence.

    Identity precedence is explicit trace ID, then path, accession, filename,
    and finally database row. Independence grouping uses only explicit
    specimen/accession provenance. Acquisition grouping uses an explicit value
    produced during refinement and otherwise falls back to the specimen group.
    Thus legacy/unresolved file paths never establish independence by accident.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    phase_name = _first_nonempty_text(result, ("name", "phase_name"))
    provenance_value = result.get("provenance", {})
    provenance: Mapping[str, Any] = (
        provenance_value if isinstance(provenance_value, Mapping) else {}
    )
    resolved_reference_id = str(reference_id).strip() if reference_id is not None else ""
    if not resolved_reference_id:
        resolved_reference_id = _first_nonempty_text(
            result,
            (
                "reference_id",
                "path",
                "accession",
                "orig_filename",
                "filename",
                "db_idx",
            ),
        )

    resolved_group = (
        str(independence_group).strip() if independence_group is not None else ""
    )
    if not resolved_group:
        resolved_group = _first_nonempty_text(
            result,
            (
                "independence_group",
                "specimen_id",
                "accession",
                "rruff_id",
                "rod_id",
                "source_accession",
            ),
        )
    if not resolved_group:
        resolved_group = _first_nonempty_text(
            provenance,
            (
                "independence_group",
                "specimen_id",
                "accession",
                "rruff_id",
                "rod_id",
                "source_accession",
            ),
        )
    if not resolved_group:
        provenance_source = _first_nonempty_text(provenance, ("source",))
        if provenance_source:
            resolved_group = f"provenance-source:{provenance_source}"
    if not resolved_group:
        resolved_group = f"unresolved:{phase_key(phase_name)}"

    resolved_acquisition = (
        str(acquisition_group).strip() if acquisition_group is not None else ""
    )
    if not resolved_acquisition:
        resolved_acquisition = _first_nonempty_text(
            result,
            ("acquisition_group", "acquisition_id", "spectrum_id"),
        )
    if not resolved_acquisition:
        resolved_acquisition = _first_nonempty_text(
            provenance,
            ("acquisition_group", "acquisition_id", "spectrum_id"),
        )
    if not resolved_acquisition:
        resolved_acquisition = resolved_group

    resolved_source = str(source).strip() if source is not None else ""
    if not resolved_source:
        resolved_source = _first_nonempty_text(result, ("source", "database_source"))

    return ReferenceMatchEvidence(
        phase_name=phase_name,
        reference_id=resolved_reference_id,
        independence_group=resolved_group,
        rank=rank_components_from_mapping(
            result,
            gradient_weight=gradient_weight,
            spectral_similarity_weight=spectral_similarity_weight,
        ),
        alignment=alignment_evidence_from_mapping(
            result,
            maximum_shift_points=maximum_shift_points,
            grid_step_cm1=grid_step_cm1,
        ),
        acquisition_group=resolved_acquisition,
        source=resolved_source,
    )


def phase_evidence_from_mappings(
    results: Iterable[Mapping[str, Any]],
    *,
    limit: int | None = None,
    maximum_shift_points: int = 5,
    grid_step_cm1: float = 1.0,
    gradient_weight: float = DEFAULT_GRADIENT_WEIGHT,
    spectral_similarity_weight: float = DEFAULT_FINAL_SIMILARITY_WEIGHT,
) -> tuple[PhaseEvidence, ...]:
    """Bridge current result dictionaries and return duplicate-aware phases."""

    references = (
        reference_match_evidence_from_mapping(
            result,
            maximum_shift_points=maximum_shift_points,
            grid_step_cm1=grid_step_cm1,
            gradient_weight=gradient_weight,
            spectral_similarity_weight=spectral_similarity_weight,
        )
        for result in results
    )
    return rank_phases(references, limit=limit)


def phase_key(phase_name: str) -> str:
    """Return a stable case-insensitive phase grouping key."""

    normalised = re.sub(r"\s+", " ", str(phase_name).strip()).casefold()
    return normalised or "?"


def group_matches_by_phase(
    matches: Iterable[ReferenceMatchEvidence],
) -> dict[str, tuple[ReferenceMatchEvidence, ...]]:
    """Group reference evidence by normalised phase name."""

    grouped: dict[str, list[ReferenceMatchEvidence]] = {}
    for match in matches:
        grouped.setdefault(phase_key(match.phase_name), []).append(match)
    return {key: tuple(values) for key, values in grouped.items()}


def rank_phases(
    matches: Iterable[ReferenceMatchEvidence],
    *,
    limit: int | None = None,
) -> tuple[PhaseEvidence, ...]:
    """Rank phases without duplicate inflation or library-size dilution.

    Representations of one acquisition (for example paired RRUFF raw and
    processed exports) are averaged first. The best acquisition is then kept
    for each specimen because orientation and excitation variants are valid
    alternatives, not repeated votes. Phase ordering uses the best specimen
    score. Coverage and boundary evidence are evaluated on at most the two
    strongest independent specimens, so unrelated weak library entries cannot
    poison an otherwise well-supported candidate.
    """

    if limit is not None and int(limit) < 0:
        raise ValueError("limit must be non-negative or None")

    ranked: list[PhaseEvidence] = []
    for key, phase_matches_tuple in group_matches_by_phase(matches).items():
        phase_matches = list(phase_matches_tuple)
        for match in phase_matches:
            if not math.isfinite(float(match.rank.final_rank_score)):
                raise ValueError("reference rank scores must be finite")

        independence_groups: dict[str, list[ReferenceMatchEvidence]] = {}
        for match in phase_matches:
            group = phase_key(match.independence_group)
            independence_groups.setdefault(group, []).append(match)

        # score, best trace, all representations of the winning acquisition
        specimen_evidence: list[
            tuple[
                float,
                ReferenceMatchEvidence,
                tuple[ReferenceMatchEvidence, ...],
            ]
        ] = []
        for group_matches in independence_groups.values():
            acquisitions: dict[str, list[ReferenceMatchEvidence]] = {}
            for match in group_matches:
                acquisition = phase_key(match.acquisition_group)
                acquisitions.setdefault(acquisition, []).append(match)
            acquisition_evidence: list[
                tuple[
                    float,
                    ReferenceMatchEvidence,
                    tuple[ReferenceMatchEvidence, ...],
                ]
            ] = []
            for acquisition_matches in acquisitions.values():
                ordered_acquisition = tuple(
                    sorted(
                        acquisition_matches,
                        key=lambda item: (
                            -float(item.rank.final_rank_score),
                            item.reference_id.casefold(),
                        ),
                    )
                )
                acquisition_evidence.append(
                    (
                        float(
                            np.mean(
                                [
                                    match.rank.final_rank_score
                                    for match in ordered_acquisition
                                ]
                            )
                        ),
                        ordered_acquisition[0],
                        ordered_acquisition,
                    )
                )
            specimen_evidence.append(
                max(
                    acquisition_evidence,
                    key=lambda item: (
                        item[0],
                        float(item[1].rank.final_rank_score),
                        item[1].reference_id.casefold(),
                    ),
                )
            )

        specimen_evidence.sort(
            key=lambda item: (
                -item[0],
                -float(item[1].rank.final_rank_score),
                item[1].reference_id.casefold(),
            )
        )
        group_scores = [item[0] for item in specimen_evidence]
        strongest_support = specimen_evidence[:2]
        group_coverages = [
            float(
                np.mean(
                    [
                        match.alignment.query_coverage_fraction
                        for match in acquisition_matches
                    ]
                )
            )
            for _score, _best, acquisition_matches in strongest_support
        ]
        group_minimum_coverages = [
            float(
                min(
                    match.alignment.query_coverage_fraction
                    for match in acquisition_matches
                )
            )
            for _score, _best, acquisition_matches in strongest_support
        ]
        group_common_point_counts = [
            min(
                match.alignment.common_point_count
                for match in acquisition_matches
            )
            for _score, _best, acquisition_matches in strongest_support
        ]
        boundary_group_count = sum(
            any(
                match.alignment.shift_search_boundary_hit
                for match in acquisition_matches
            )
            for _score, _best, acquisition_matches in strongest_support
        )
        clipped_group_count = sum(
            any(
                match.alignment.reference_support_clipped_at_grid_boundary
                for match in acquisition_matches
            )
            for _score, _best, acquisition_matches in strongest_support
        )

        supporting = tuple(
            sorted(
                phase_matches,
                key=lambda item: (
                    -float(item.rank.final_rank_score),
                    item.reference_id.casefold(),
                ),
            )
        )
        # Retain the strongest individual trace for audit/overlay metadata.
        # It does not define the duplicate-safe phase score above.
        best_match = supporting[0]
        ranked.append(
            PhaseEvidence(
                phase_name=phase_matches[0].phase_name,
                phase_key=key,
                aggregate_score=float(group_scores[0]),
                median_group_score=float(np.median(group_scores)),
                best_reference_score=float(best_match.rank.final_rank_score),
                mean_coverage_fraction=float(np.mean(group_coverages)),
                minimum_group_coverage_fraction=float(
                    np.min(group_minimum_coverages)
                ),
                independent_reference_count=len(independence_groups),
                reference_variant_count=len(phase_matches),
                shift_boundary_group_count=boundary_group_count,
                grid_boundary_clipped_group_count=clipped_group_count,
                best_match=best_match,
                group_scores=tuple(group_scores),
                supporting_matches=supporting,
                minimum_group_common_point_count=min(group_common_point_counts),
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.aggregate_score,
            -item.median_group_score,
            -item.best_reference_score,
            item.phase_key,
        )
    )
    if limit is not None:
        ranked = ranked[: int(limit)]
    return tuple(ranked)


def decide_evidence_status(
    ranked_phases: Iterable[PhaseEvidence],
    *,
    policy: EvidenceDecisionPolicy | None = None,
) -> EvidenceDecision:
    """Apply transparent, conservative evidence guardrails.

    The decision is deliberately categorical and uncalibrated.  In particular,
    ``SUPPORTED_CANDIDATE`` means only that the supplied evidence passes these
    policy checks; it is not a probability or a confirmed identification.
    """

    active_policy = policy or EvidenceDecisionPolicy()
    phases = sorted(
        tuple(ranked_phases),
        key=lambda item: (
            -item.aggregate_score,
            -item.median_group_score,
            -item.best_reference_score,
            item.phase_key,
        ),
    )
    if not phases:
        return EvidenceDecision(
            status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
            best_phase=None,
            best_score=None,
            runner_up_score=None,
            score_margin=None,
            reasons=("no_phase_candidates",),
        )

    best = phases[0]
    runner_up = phases[1] if len(phases) > 1 else None
    runner_up_phase = runner_up.phase_name if runner_up is not None else None
    runner_up_score = (
        float(runner_up.aggregate_score) if runner_up is not None else None
    )
    score_margin = (
        float(best.aggregate_score - runner_up_score)
        if runner_up_score is not None
        else None
    )
    insufficiency_reasons: list[str] = []
    if best.minimum_group_common_point_count < active_policy.minimum_common_points:
        insufficiency_reasons.append("too_few_common_points")
    if best.minimum_group_coverage_fraction < active_policy.minimum_coverage_fraction:
        insufficiency_reasons.append("insufficient_minimum_group_coverage")
    if insufficiency_reasons:
        return EvidenceDecision(
            status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
            best_phase=best.phase_name,
            best_score=float(best.aggregate_score),
            runner_up_score=runner_up_score,
            score_margin=score_margin,
            reasons=tuple(insufficiency_reasons),
            runner_up_phase=runner_up_phase,
        )

    if best.aggregate_score < active_policy.minimum_phase_score:
        return EvidenceDecision(
            status=EvidenceStatus.UNKNOWN_OR_OUT_OF_LIBRARY,
            best_phase=best.phase_name,
            best_score=float(best.aggregate_score),
            runner_up_score=runner_up_score,
            score_margin=score_margin,
            reasons=("phase_score_below_uncalibrated_guardrail",),
            runner_up_phase=runner_up_phase,
        )

    ambiguity_reasons: list[str] = []
    runner_up_has_comparable_support = bool(
        runner_up is not None
        and runner_up.minimum_group_common_point_count
        >= active_policy.minimum_common_points
        and runner_up.minimum_group_coverage_fraction
        >= active_policy.minimum_coverage_fraction
    )
    if not runner_up_has_comparable_support:
        ambiguity_reasons.append("phase_separation_not_assessed")
    elif (
        score_margin is not None
        and score_margin < active_policy.minimum_score_margin
    ):
        ambiguity_reasons.append("leading_phase_margin_too_small")
    if best.independent_reference_count < active_policy.minimum_independent_references:
        ambiguity_reasons.append("too_few_independent_references")
    if (
        active_policy.reject_shift_search_boundary
        and best.shift_boundary_group_count > 0
    ):
        ambiguity_reasons.append("leading_phase_has_shift_search_boundary_evidence")
    if (
        active_policy.reject_grid_boundary_clipping
        and best.grid_boundary_clipped_group_count > 0
    ):
        ambiguity_reasons.append("leading_phase_has_grid_clipped_reference_support")

    if ambiguity_reasons:
        return EvidenceDecision(
            status=EvidenceStatus.AMBIGUOUS,
            best_phase=best.phase_name,
            best_score=float(best.aggregate_score),
            runner_up_score=runner_up_score,
            score_margin=score_margin,
            reasons=tuple(ambiguity_reasons),
            runner_up_phase=runner_up_phase,
        )

    return EvidenceDecision(
        status=EvidenceStatus.SUPPORTED_CANDIDATE,
        best_phase=best.phase_name,
        best_score=float(best.aggregate_score),
        runner_up_score=runner_up_score,
        score_margin=score_margin,
        reasons=("uncalibrated_evidence_guardrails_passed",),
        runner_up_phase=runner_up_phase,
    )


def annotate_phase_evidence(
    candidates: Iterable[Mapping[str, Any]],
    *,
    grid_step_cm1: float,
    parameters: MatchingParameters | None = None,
    policy: EvidenceDecisionPolicy | None = None,
) -> list[dict[str, Any]]:
    """Attach full-pool, duplicate-aware phase and decision evidence."""

    active = parameters or MatchingParameters()
    materialized = [dict(candidate) for candidate in candidates]
    if not materialized:
        return []
    phases = phase_evidence_from_mappings(
        materialized,
        grid_step_cm1=float(grid_step_cm1),
        maximum_shift_points=int(active.maximum_shift_points),
        gradient_weight=float(active.gradient_weight),
        spectral_similarity_weight=float(active.spectral_similarity_weight),
    )
    decision = decide_evidence_status(phases, policy=policy)
    phase_by_key = {phase.phase_key: phase for phase in phases}
    phase_ranks = {phase.phase_key: index + 1 for index, phase in enumerate(phases)}
    decision_payload = {
        "evidence_status": decision.status.value,
        "evidence_best_phase": decision.best_phase,
        "evidence_runner_up_phase": decision.runner_up_phase,
        "evidence_best_score": decision.best_score,
        "evidence_runner_up_score": decision.runner_up_score,
        "evidence_score_margin": decision.score_margin,
        "evidence_reasons": list(decision.reasons),
        "evidence_is_calibrated_confidence": decision.is_calibrated_confidence,
    }
    annotated: list[dict[str, Any]] = []
    for result in materialized:
        key = phase_key(str(result.get("name", "")))
        phase = phase_by_key[key]
        result.update(
            {
                "phase_rank": phase_ranks[key],
                "phase_score": phase.aggregate_score,
                "phase_median_group_score": phase.median_group_score,
                "phase_best_reference_score": phase.best_reference_score,
                "phase_mean_coverage_fraction": phase.mean_coverage_fraction,
                "phase_minimum_group_coverage_fraction": phase.minimum_group_coverage_fraction,
                "phase_minimum_group_common_point_count": phase.minimum_group_common_point_count,
                "phase_independent_reference_count": phase.independent_reference_count,
                "phase_reference_variant_count": phase.reference_variant_count,
                "phase_shift_boundary_group_count": phase.shift_boundary_group_count,
                "phase_grid_boundary_clipped_group_count": phase.grid_boundary_clipped_group_count,
                **decision_payload,
            }
        )
        annotated.append(result)
    return annotated


def final_rank_score(
    result: Mapping[str, Any],
    *,
    spectral_similarity_weight: float = DEFAULT_FINAL_SIMILARITY_WEIGHT,
) -> float:
    if "rank_score" in result:
        return float(result["rank_score"])
    weight = float(spectral_similarity_weight)
    return weight * float(result.get("similarity", 0.0)) + (1.0 - weight) * float(
        result.get("pcs", 0.0)
    )


def _presentation_identity(result: Mapping[str, Any]) -> str:
    path = str(result.get("path", "")).strip()
    if path and path != "__unresolved_source__":
        return f"path:{path}"
    reference = str(result.get("reference_id", "")).strip()
    if reference and reference != "__unresolved_source__":
        return f"reference:{reference}"
    return (
        f"row:{result.get('db_variant', '')}:{result.get('db_idx', '')}:"
        f"{result.get('orig_filename', '')}"
    )


def select_diverse_top(
    candidates: Iterable[Mapping[str, Any]],
    top_n: int,
    *,
    parameters: MatchingParameters | None = None,
) -> list[dict[str, Any]]:
    """Select auditable trace representatives after phase consensus ranking."""

    requested = int(top_n)
    if requested <= 0:
        return []
    active = parameters or MatchingParameters()
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        result = dict(candidate)
        result["rank_score"] = final_rank_score(
            result,
            spectral_similarity_weight=float(active.spectral_similarity_weight),
        )
        scored.append(result)
    scored.sort(
        key=lambda result: (
            float(result.get("phase_score", result["rank_score"])),
            float(result["rank_score"]),
            float(result.get("similarity", 0.0)),
            float(result.get("pcs", 0.0)),
        ),
        reverse=True,
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in scored:
        identity = _presentation_identity(result)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(result)
    phase_cap = max(
        2,
        min(int(active.per_phase_cap), int(math.ceil(requested / 8))),
    )
    selected: list[dict[str, Any]] = []
    phase_counts: dict[str, int] = {}
    deferred: list[dict[str, Any]] = []
    for result in unique:
        key = phase_key(str(result.get("name", "")))
        count = phase_counts.get(key, 0)
        if count < phase_cap:
            selected.append(result)
            phase_counts[key] = count + 1
            if len(selected) >= requested:
                break
        else:
            deferred.append(result)
    if len(selected) < requested:
        selected_ids = {_presentation_identity(result) for result in selected}
        for result in deferred:
            if _presentation_identity(result) in selected_ids:
                continue
            selected.append(result)
            if len(selected) >= requested:
                break

    peak_slots = max(
        2,
        min(int(active.peak_phase_slot_cap), requested // 5),
    )
    peak_best_by_phase: dict[str, dict[str, Any]] = {}
    for result in unique:
        key = phase_key(str(result.get("name", "")))
        current = peak_best_by_phase.get(key)
        if current is None or (
            float(result.get("pcs", 0.0)),
            float(result.get("similarity", 0.0)),
        ) > (
            float(current.get("pcs", 0.0)),
            float(current.get("similarity", 0.0)),
        ):
            peak_best_by_phase[key] = result
    peak_pool = sorted(
        peak_best_by_phase.values(),
        key=lambda result: (
            float(result.get("pcs", 0.0)),
            float(result.get("similarity", 0.0)),
        ),
        reverse=True,
    )
    selected_ids = {_presentation_identity(result) for result in selected}
    selected_phases = {phase_key(str(result.get("name", ""))) for result in selected}
    peak_additions: list[dict[str, Any]] = []
    for result in peak_pool:
        key = phase_key(str(result.get("name", "")))
        identity = _presentation_identity(result)
        if identity in selected_ids or key in selected_phases:
            continue
        if float(result.get("pcs", 0.0)) < float(active.peak_phase_minimum_score):
            continue
        if float(result.get("similarity", 0.0)) < float(
            active.peak_phase_minimum_similarity
        ):
            continue
        peak_additions.append(result)
        selected_ids.add(identity)
        selected_phases.add(key)
        if len(peak_additions) >= peak_slots:
            break
    if peak_additions:
        selected = selected[: max(0, requested - len(peak_additions))] + peak_additions
    selected.sort(
        key=lambda result: (
            float(result.get("phase_score", result["rank_score"])),
            float(result["rank_score"]),
            float(result.get("similarity", 0.0)),
            float(result.get("pcs", 0.0)),
        ),
        reverse=True,
    )
    return selected[:requested]


def _coverage_eligible_ids(
    pack: Mapping[str, Any],
    allowed_ids: ArrayLike,
    range_low: int,
    range_high: int,
    minimum_fraction: float,
    *,
    query_mask: ArrayLike | None = None,
) -> NDArray[np.int32]:
    ids = np.asarray(allowed_ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return ids
    grid_info = pack["grid_info"]
    grid_minimum = int(grid_info["min"])
    grid_step = int(grid_info["step"])
    grid_length = int(grid_info["len"])
    query_start = max(0, int((int(range_low) - grid_minimum) // grid_step))
    query_end = min(
        grid_length - 1,
        int((int(range_high) - grid_minimum) // grid_step),
    )
    requested_mask = np.zeros(grid_length, dtype=bool)
    if query_end >= query_start:
        requested_mask[query_start : query_end + 1] = True
    if query_mask is not None:
        exact_query_mask = np.asarray(query_mask, dtype=bool)
        if exact_query_mask.ndim != 1 or exact_query_mask.size != grid_length:
            raise ValueError("query_mask must match the database grid length")
        requested_mask &= exact_query_mask
    requested = int(np.count_nonzero(requested_mask))
    if requested == 0:
        return np.array([], dtype=np.int32)
    requested_prefix = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(requested_mask, dtype=np.int64))
    )
    metadata = pack["meta"]
    starts = np.fromiter(
        (int(metadata[int(index)].get("start_idx", 0)) for index in ids),
        dtype=np.int64,
        count=ids.size,
    )
    ends = np.fromiter(
        (int(metadata[int(index)].get("end_idx", -1)) for index in ids),
        dtype=np.int64,
        count=ids.size,
    )
    overlap = np.zeros(ids.size, dtype=np.int64)
    valid_contiguous = (
        (starts >= 0)
        & (ends >= starts)
        & (ends < grid_length)
    )
    overlap[valid_contiguous] = (
        requested_prefix[ends[valid_contiguous] + 1]
        - requested_prefix[starts[valid_contiguous]]
    )
    for position, index in enumerate(ids):
        row = metadata[int(index)]
        raw_runs = row.get("support_runs")
        if raw_runs is None:
            continue
        runs = _normalise_support_runs(
            grid_length,
            int(starts[position]),
            int(ends[position]),
            raw_runs,
        )
        legacy_run = (
            ((int(starts[position]), int(ends[position])),)
            if valid_contiguous[position]
            else ()
        )
        if runs == legacy_run:
            continue
        overlap[position] = sum(
            int(requested_prefix[end + 1] - requested_prefix[start])
            for start, end in runs
        )
    return ids[(overlap / requested) >= float(minimum_fraction)]


def match_query_vector(
    query: ArrayLike,
    query_mask: ArrayLike,
    range_low: int,
    range_high: int,
    raw_pack: Mapping[str, Any],
    baseline_pack: Mapping[str, Any],
    allowed_raw_ids: ArrayLike,
    allowed_baseline_ids: ArrayLike,
    measurement_variant: str,
    *,
    top_n: int = 60,
    parameters: MatchingParameters | None = None,
    evidence_policy: EvidenceDecisionPolicy | None = None,
    excluded_phase_keys: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Run range-local screening, exact alignment, and phase consensus.

    ``excluded_phase_keys`` is applied before phase aggregation.  It supports
    residual searches that must not allow the already-subtracted primary phase
    (including another duplicate reference) to define either the leading or
    runner-up evidence.
    """

    active = parameters or MatchingParameters()
    raw_ids = _coverage_eligible_ids(
        raw_pack,
        allowed_raw_ids,
        range_low,
        range_high,
        active.minimum_coverage_fraction,
        query_mask=query_mask,
    )
    baseline_ids = _coverage_eligible_ids(
        baseline_pack,
        allowed_baseline_ids,
        range_low,
        range_high,
        active.minimum_coverage_fraction,
        query_mask=query_mask,
    )
    requested = max(1, int(top_n))
    raw_shortlist_size = min(
        int(raw_ids.size),
        max(300, requested * int(active.raw_top_n_factor), int(active.raw_minimum_shortlist)),
    )
    baseline_shortlist_size = min(
        int(baseline_ids.size),
        max(
            300,
            requested * int(active.baseline_top_n_factor),
            int(active.baseline_minimum_shortlist),
        ),
    )

    def shortlist(pack: Mapping[str, Any], ids: NDArray[np.int32], size: int) -> list[int]:
        if ids.size == 0:
            return []
        return topk_cosine_subset(
            query,
            pack["X"],
            pack["meta"],
            ids,
            size,
            support_mask=query_mask,
            chunk_rows=int(active.screen_chunk_rows),
            remove_query_local_offset=bool(active.remove_query_local_offset),
        )

    raw_refined = refine_and_rank(
        query,
        query_mask,
        shortlist(raw_pack, raw_ids, raw_shortlist_size),
        raw_pack,
        max(requested * 4, raw_shortlist_size),
        parameters=active,
    )
    for result in raw_refined:
        result.update(
            {"meas_variant": str(measurement_variant), "db_variant": "DB-RAW"}
        )
    baseline_refined = refine_and_rank(
        query,
        query_mask,
        shortlist(baseline_pack, baseline_ids, baseline_shortlist_size),
        baseline_pack,
        max(requested * 4, baseline_shortlist_size),
        parameters=active,
    )
    for result in baseline_refined:
        result.update(
            {"meas_variant": str(measurement_variant), "db_variant": "DB-BC"}
        )
    excluded = {
        phase_key(value)
        for value in excluded_phase_keys
        if str(value).strip()
    }
    evidence_pool = [
        result
        for result in raw_refined + baseline_refined
        if phase_key(str(result.get("name", ""))) not in excluded
        and float(result.get("pcs", 0.0))
        >= float(active.minimum_candidate_peak_consistency)
    ]
    annotated = annotate_phase_evidence(
        evidence_pool,
        grid_step_cm1=float(raw_pack.get("grid_info", {}).get("step", 1.0)),
        parameters=active,
        policy=evidence_policy,
    )
    return select_diverse_top(annotated, requested, parameters=active)


__all__ = [
    "ALIGNMENT_SCORE_TIE_TOLERANCE",
    "AlignmentEvidence",
    "AlignmentResult",
    "DEFAULT_FINAL_SIMILARITY_WEIGHT",
    "DEFAULT_GRADIENT_WEIGHT",
    "EvidenceDecision",
    "EvidenceDecisionPolicy",
    "EvidenceStatus",
    "MatchingParameters",
    "PhaseEvidence",
    "RankComponents",
    "ReferenceMatchEvidence",
    "ResidualProjection",
    "ResidualSearchPolicy",
    "SupportRuns",
    "alignment_evidence_from_mapping",
    "aligned_support_mask",
    "annotate_phase_evidence",
    "best_aligned_score",
    "build_residual_projection",
    "compose_rank_components",
    "decide_evidence_status",
    "final_rank_score",
    "gradient_support_mask",
    "group_matches_by_phase",
    "masked_cosine",
    "match_query_vector",
    "peak_consistency_score",
    "phase_evidence_from_mappings",
    "phase_key",
    "rank_components_from_mapping",
    "rank_phases",
    "refine_and_rank",
    "reference_match_evidence_from_mapping",
    "reference_support_mask",
    "shift_candidate",
    "shift_support_mask",
    "select_diverse_top",
    "topk_cosine_subset",
]

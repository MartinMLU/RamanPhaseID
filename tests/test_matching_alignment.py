from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

import raman_matching as matching


def test_matching_and_evidence_policy_payloads_are_complete() -> None:
    parameters = matching.MatchingParameters()
    policy = matching.EvidenceDecisionPolicy()
    residual_policy = matching.ResidualSearchPolicy()

    assert {field.name for field in fields(parameters)} <= set(parameters.payload())
    assert parameters.payload()["spectral_similarity_weight"] == pytest.approx(
        parameters.spectral_similarity_weight
    )
    assert parameters.payload()["maximum_shift_points"] == parameters.maximum_shift_points
    assert parameters.payload()["alignment_score_tie_tolerance"] == pytest.approx(
        parameters.alignment_score_tie_tolerance
    )
    assert parameters.payload()["peak_detection_max_peaks"] == 80
    assert parameters.payload()["peak_consistency_minimum_support_points"] == 20
    assert parameters.payload()["minimum_candidate_peak_consistency"] == 0.0
    assert parameters.payload()["remove_query_local_offset"] is True
    assert policy.payload()["minimum_common_points"] == policy.minimum_common_points
    assert (
        policy.payload()["reject_grid_boundary_clipping"]
        is policy.reject_grid_boundary_clipping
    )
    assert {field.name for field in fields(residual_policy)} <= set(
        residual_policy.payload()
    )
    assert residual_policy.payload()["minimum_fit_improvement_fraction"] == pytest.approx(
        0.02
    )
    assert residual_policy.payload()["support_edge_guard_points"] == 3


def _legacy_masked_cosine(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    *,
    remove_local_offset: bool = False,
) -> float:
    if mask.sum() < 2:
        return -1.0
    aa = first[mask]
    bb = second[mask]
    if remove_local_offset:
        aa = aa - np.min(aa)
        bb = bb - np.min(bb)
    first_norm = np.linalg.norm(aa)
    second_norm = np.linalg.norm(bb)
    if first_norm == 0.0 or second_norm == 0.0:
        return -1.0
    return float(np.dot(aa, bb) / (first_norm * second_norm))


def _legacy_shift_zero_score(
    query: np.ndarray,
    candidate: np.ndarray,
    query_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    gradient_weight: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    coverage = np.zeros(query.size, dtype=bool)
    coverage[start_idx : end_idx + 1] = True
    comparison_mask = query_mask & coverage
    shape = _legacy_masked_cosine(
        query,
        candidate,
        comparison_mask,
        remove_local_offset=True,
    )
    gradient_mask = np.zeros_like(comparison_mask)
    gradient_mask[1:-1] = (
        comparison_mask[1:-1]
        & comparison_mask[:-2]
        & comparison_mask[2:]
    )
    gradient = _legacy_masked_cosine(
        np.gradient(query),
        np.gradient(candidate),
        gradient_mask,
    )
    combined = ((1.0 - gradient_weight) * shape) + (gradient_weight * gradient)
    return combined, shape, gradient, comparison_mask, gradient_mask


def test_shift_zero_preserves_legacy_numerical_behavior_and_exposes_rank_parts() -> None:
    query = np.array([5.0, 4.0, 7.0, 10.0, 8.0, 3.0, 6.0, 9.0, 4.0, 2.0])
    candidate = np.array([2.0, 3.0, 5.0, 9.0, 7.0, 4.0, 5.5, 8.0, 3.0, 1.0])
    query_mask = np.zeros(query.size, dtype=bool)
    query_mask[2:9] = True
    expected, shape, gradient, mask, gradient_mask = _legacy_shift_zero_score(
        query,
        candidate,
        query_mask,
        1,
        8,
        matching.DEFAULT_GRADIENT_WEIGHT,
    )

    result = matching.best_aligned_score(
        query,
        candidate,
        query_mask,
        1,
        8,
        max_shift=0,
    )

    assert result.spectral_similarity == expected
    assert result.shape_similarity == shape
    assert result.gradient_similarity == gradient
    assert result.evidence.fitted_shift_points == 0
    assert result.evidence.fitted_shift_cm1 == 0.0
    assert not result.evidence.shift_search_boundary_hit
    np.testing.assert_array_equal(result.aligned_candidate, candidate)
    np.testing.assert_array_equal(result.comparison_mask, mask)
    np.testing.assert_array_equal(result.gradient_comparison_mask, gradient_mask)
    assert not result.aligned_candidate.flags.writeable
    assert not result.comparison_mask.flags.writeable

    components = matching.compose_rank_components(result, peak_consistency=0.60)
    assert components.final_rank_score == pytest.approx(
        (0.88 * expected) + (0.12 * 0.60)
    )
    assert (
        components.shape_contribution
        + components.gradient_contribution
        + components.peak_consistency_contribution
    ) == pytest.approx(components.final_rank_score)
    assert components.gradient_weight_within_similarity == 0.20
    assert components.spectral_similarity_weight == 0.88
    assert components.peak_consistency_weight == pytest.approx(0.12)


def test_positive_shift_moves_reference_support_and_reports_partial_coverage() -> None:
    candidate = np.zeros(12, dtype=float)
    candidate[2:8] = [1.0, 4.0, 2.0, 7.0, 3.0, 5.0]
    expected_aligned = matching.shift_candidate(candidate, 2)
    query = expected_aligned.copy()
    query_mask = np.zeros(query.size, dtype=bool)
    query_mask[:8] = True

    result = matching.best_aligned_score(
        query,
        candidate,
        query_mask,
        2,
        7,
        max_shift=2,
        gradient_weight=0.0,
    )

    expected_mask = np.zeros(query.size, dtype=bool)
    # Native support 2:7 moves to 4:9; query support ends at index 7.
    expected_mask[4:8] = True
    old_unshifted_mask = np.zeros(query.size, dtype=bool)
    old_unshifted_mask[2:8] = True

    assert result.evidence.fitted_shift_points == 2
    assert result.evidence.fitted_shift_cm1 == 2.0
    assert result.evidence.shift_search_boundary_hit
    assert not result.evidence.reference_support_clipped_at_grid_boundary
    assert result.evidence.requested_point_count == 8
    assert result.evidence.reference_support_point_count == 6
    assert result.evidence.shifted_reference_support_point_count == 6
    assert result.evidence.common_point_count == 4
    assert result.evidence.query_coverage_fraction == pytest.approx(0.5)
    assert result.evidence.reference_overlap_fraction == pytest.approx(4.0 / 6.0)
    assert result.spectral_similarity == pytest.approx(1.0)
    np.testing.assert_array_equal(result.aligned_candidate, expected_aligned)
    np.testing.assert_array_equal(result.comparison_mask, expected_mask)
    assert not np.array_equal(result.comparison_mask, query_mask & old_unshifted_mask)


def test_negative_shift_clips_grid_edge_without_scoring_zero_filled_support() -> None:
    candidate = np.zeros(10, dtype=float)
    candidate[1:6] = [8.0, 2.0, 7.0, 3.0, 5.0]
    expected_aligned = matching.shift_candidate(candidate, -2)
    query = expected_aligned.copy()
    query_mask = np.ones(query.size, dtype=bool)

    result = matching.best_aligned_score(
        query,
        candidate,
        query_mask,
        1,
        5,
        max_shift=2,
        gradient_weight=0.0,
    )

    expected_mask = np.zeros(query.size, dtype=bool)
    # Index 1 is shifted beyond the left grid boundary; indices 2:5 land at 0:3.
    expected_mask[:4] = True
    assert result.evidence.fitted_shift_points == -2
    assert result.evidence.shift_search_boundary_hit
    assert result.evidence.reference_support_clipped_at_grid_boundary
    assert result.evidence.reference_support_point_count == 5
    assert result.evidence.shifted_reference_support_point_count == 4
    assert result.evidence.common_point_count == 4
    assert result.evidence.reference_overlap_fraction == 1.0
    assert result.spectral_similarity == pytest.approx(1.0)
    np.testing.assert_array_equal(result.aligned_candidate, expected_aligned)
    np.testing.assert_array_equal(result.comparison_mask, expected_mask)


def test_residual_projection_preserves_negative_oversubtraction_for_audit() -> None:
    query = np.array([1.0, 1.0, 4.0, 1.0, 1.0])
    candidate = np.ones(5, dtype=float)

    projection = matching.build_residual_projection(
        query,
        candidate,
        np.ones(5, dtype=bool),
        0,
        4,
        0,
        minimum_common_points=2,
    )

    assert projection.scale_factor == pytest.approx(1.6)
    np.testing.assert_allclose(projection.signed_residual, [-0.6, -0.6, 2.4, -0.6, -0.6])
    assert np.any(projection.matching_vector < 0.0)
    assert projection.negative_point_fraction == pytest.approx(0.8)
    assert projection.negative_energy_fraction > 0.0
    assert projection.fit_improvement_fraction > 0.0
    assert not projection.signed_residual.flags.writeable
    assert not projection.matching_vector.flags.writeable


def test_residual_projection_does_not_cap_fitted_scale_at_legacy_limit() -> None:
    candidate = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    projection = matching.build_residual_projection(
        2.0 * candidate,
        candidate,
        np.ones(5, dtype=bool),
        0,
        4,
        0,
        minimum_common_points=2,
    )

    assert projection.scale_factor == pytest.approx(2.0)
    assert projection.fit_improvement_fraction == pytest.approx(1.0)
    np.testing.assert_allclose(projection.signed_residual, 0.0)


def test_residual_projection_masks_artificial_reference_support_steps() -> None:
    grid = np.arange(60.0, 1901.0)
    start = int(100.0 - grid[0])
    end = int(1600.0 - grid[0])
    candidate = np.zeros(grid.size, dtype=float)
    candidate[start : end + 1] = (
        0.20
        + np.exp(-0.5 * ((grid[start : end + 1] - 500.0) / 18.0) ** 2)
    )
    query = candidate + 0.05
    query_mask = np.ones(grid.size, dtype=bool)

    unguarded = matching.build_residual_projection(
        query,
        candidate,
        query_mask,
        start,
        end,
        0,
        minimum_common_points=20,
        support_edge_guard_points=0,
    )
    guarded = matching.build_residual_projection(
        query,
        candidate,
        query_mask,
        start,
        end,
        0,
        minimum_common_points=20,
        support_edge_guard_points=3,
    )

    # The signed audit data remains exact and exposes the old one-sample jump.
    np.testing.assert_allclose(guarded.signed_residual, unguarded.signed_residual)
    assert abs(guarded.signed_residual[start] - guarded.signed_residual[start - 1]) > 0.02
    assert abs(guarded.signed_residual[end + 1] - guarded.signed_residual[end]) > 0.02

    # A short NaN/invalid separator at each support transition stops plots and
    # derivative/peak matching from turning those bookkeeping edges into peaks.
    expected_invalid = np.zeros(grid.size, dtype=bool)
    expected_invalid[start - 3 : start + 3] = True
    expected_invalid[end - 2 : end + 4] = True
    np.testing.assert_array_equal(~guarded.residual_mask, expected_invalid)
    assert guarded.residual_mask[start - 4]
    assert guarded.residual_mask[start + 3]
    assert guarded.residual_mask[end - 3]
    assert guarded.residual_mask[end + 4]
    np.testing.assert_allclose(guarded.matching_vector[expected_invalid], 0.0)
    assert np.any(guarded.residual_mask[: start - 3])
    assert np.any(guarded.residual_mask[end + 4 :])

    transitions = np.flatnonzero(
        guarded.comparison_mask[1:] != guarded.comparison_mask[:-1]
    ) + 1
    assert transitions.tolist() == [start, end + 1]
    assert all(
        not (
            guarded.residual_mask[boundary - 1]
            and guarded.residual_mask[boundary]
        )
        for boundary in transitions
    )
    assert not guarded.residual_mask.flags.writeable


def _alignment_evidence(
    *,
    coverage: float = 0.95,
    common_points: int = 100,
    shift_boundary: bool = False,
    grid_clipped: bool = False,
) -> matching.AlignmentEvidence:
    requested = max(common_points, int(round(common_points / coverage)))
    return matching.AlignmentEvidence(
        fitted_shift_points=5 if shift_boundary else 0,
        fitted_shift_cm1=5.0 if shift_boundary else 0.0,
        maximum_shift_points=5,
        shift_search_boundary_hit=shift_boundary,
        reference_support_clipped_at_grid_boundary=grid_clipped,
        requested_point_count=requested,
        reference_support_point_count=requested,
        shifted_reference_support_point_count=requested,
        common_point_count=common_points,
        query_coverage_fraction=coverage,
        reference_overlap_fraction=coverage,
    )


def _rank_components(score: float) -> matching.RankComponents:
    return matching.RankComponents(
        shape_similarity=score,
        gradient_similarity=score,
        peak_consistency=score,
        spectral_similarity=score,
        final_rank_score=score,
        gradient_weight_within_similarity=0.20,
        spectral_similarity_weight=0.88,
        peak_consistency_weight=0.12,
        shape_contribution=0.704 * score,
        gradient_contribution=0.176 * score,
        peak_consistency_contribution=0.12 * score,
    )


def _reference(
    phase: str,
    reference_id: str,
    group: str,
    score: float,
    *,
    acquisition_group: str = "",
    coverage: float = 0.95,
    common_points: int = 100,
    shift_boundary: bool = False,
    grid_clipped: bool = False,
) -> matching.ReferenceMatchEvidence:
    return matching.ReferenceMatchEvidence(
        phase_name=phase,
        reference_id=reference_id,
        independence_group=group,
        rank=_rank_components(score),
        alignment=_alignment_evidence(
            coverage=coverage,
            common_points=common_points,
            shift_boundary=shift_boundary,
            grid_clipped=grid_clipped,
        ),
        acquisition_group=acquisition_group,
        source="test-library",
    )


def test_phase_ranking_uses_best_specimen_without_duplicate_raw_maximum() -> None:
    evidence = [
        # One very high variant cannot define phase A by itself.  Its paired
        # non-independent variant shares specimen-a1 and is averaged with it.
        _reference("Phase A", "a-raw", "specimen-a1", 0.99),
        _reference("Phase A", "a-processed", "specimen-a1", 0.01),
        _reference("Phase A", "a-second", "specimen-a2", 0.55),
        _reference("Phase B", "b-first", "specimen-b1", 0.76),
        _reference("Phase B", "b-second", "specimen-b2", 0.74),
    ]

    phases = matching.rank_phases(evidence)

    assert [phase.phase_name for phase in phases] == ["Phase B", "Phase A"]
    phase_b, phase_a = phases
    assert phase_b.aggregate_score == pytest.approx(0.76)
    assert phase_a.aggregate_score == pytest.approx(0.55)
    assert phase_a.best_reference_score == 0.99
    assert phase_a.independent_reference_count == 2
    assert phase_a.reference_variant_count == 3
    assert sorted(phase_a.group_scores) == pytest.approx([0.50, 0.55])


def test_phase_ranking_averages_representation_pairs_but_keeps_acquisitions_distinct() -> None:
    evidence = [
        _reference(
            "Oriented phase",
            "orientation-a-raw",
            "specimen-a",
            0.92,
            acquisition_group="orientation-a",
        ),
        _reference(
            "Oriented phase",
            "orientation-a-processed",
            "specimen-a",
            0.88,
            acquisition_group="orientation-a",
        ),
        _reference(
            "Oriented phase",
            "orientation-b",
            "specimen-a",
            0.20,
            acquisition_group="orientation-b",
        ),
        _reference("Alternative", "alternative", "specimen-b", 0.85),
    ]

    phases = matching.rank_phases(evidence)

    assert [phase.phase_name for phase in phases] == [
        "Oriented phase",
        "Alternative",
    ]
    assert phases[0].aggregate_score == pytest.approx(0.90)
    assert phases[0].independent_reference_count == 1
    assert phases[0].reference_variant_count == 3


def test_evidence_decision_is_conservative_explicit_and_uncalibrated() -> None:
    supported_phases = matching.rank_phases(
        [
            _reference("Supported", "s1", "s-group-1", 0.91),
            _reference("Supported", "s2", "s-group-2", 0.89),
            _reference("Runner up", "r1", "r-group-1", 0.70),
            _reference("Runner up", "r2", "r-group-2", 0.68),
        ]
    )
    supported = matching.decide_evidence_status(supported_phases)
    assert supported.status is matching.EvidenceStatus.SUPPORTED_CANDIDATE
    assert supported.best_phase == "Supported"
    assert supported.score_margin == pytest.approx(0.21)
    assert not supported.is_calibrated_confidence
    assert supported.reasons == ("uncalibrated_evidence_guardrails_passed",)

    low_score = matching.decide_evidence_status(
        matching.rank_phases(
            [
                _reference("Weak", "w1", "w-group-1", 0.65),
                _reference("Weak", "w2", "w-group-2", 0.63),
            ]
        )
    )
    assert low_score.status is matching.EvidenceStatus.UNKNOWN_OR_OUT_OF_LIBRARY
    assert low_score.reasons == ("phase_score_below_uncalibrated_guardrail",)
    assert not low_score.is_calibrated_confidence

    boundary_limited = matching.decide_evidence_status(
        matching.rank_phases(
            [
                _reference(
                    "Boundary phase",
                    "edge-1",
                    "edge-group-1",
                    0.92,
                    shift_boundary=True,
                ),
                _reference("Boundary phase", "edge-2", "edge-group-2", 0.90),
            ]
        )
    )
    assert boundary_limited.status is matching.EvidenceStatus.AMBIGUOUS
    assert "leading_phase_has_shift_search_boundary_evidence" in boundary_limited.reasons

    insufficient_support = matching.decide_evidence_status(
        matching.rank_phases(
            [
                _reference(
                    "Partial",
                    "partial-1",
                    "partial-group-1",
                    0.95,
                    coverage=0.60,
                ),
                _reference(
                    "Partial",
                    "partial-2",
                    "partial-group-2",
                    0.93,
                    coverage=0.60,
                ),
            ]
        )
    )
    assert insufficient_support.status is matching.EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert insufficient_support.reasons == (
        "insufficient_minimum_group_coverage",
    )


def _current_result_mapping(
    phase: str,
    path: str,
    score: float,
    *,
    accession: str = "",
    db_variant: str = "DB-RAW",
) -> dict[str, object]:
    """Build the dictionary schema currently emitted by _refine_and_rank."""

    result: dict[str, object] = {
        "name": phase,
        "formula": "?",
        "filename": path.rsplit("/", 1)[-1],
        "orig_filename": path.rsplit("/", 1)[-1],
        "path": path,
        "db_idx": 4,
        "shift": 0,
        "shift_cm1": 0.0,
        "shift_boundary_hit": False,
        "grid_boundary_clipped": False,
        "common_point_count": 80,
        "coverage_fraction": 0.8,
        "reference_overlap_fraction": 0.8,
        "shape_similarity": score,
        "gradient_similarity": score,
        "similarity": score,
        "pcs": score,
        "rank_score": score,
        "rank_components": {
            "shape": score,
            "gradient": score,
            "pcs": score,
            "shape_contribution": 0.704 * score,
            "gradient_contribution": 0.176 * score,
            "pcs_contribution": 0.12 * score,
        },
        "start_idx": 10,
        "end_idx": 109,
        "db_baseline": db_variant == "DB-BC",
        "db_variant": db_variant,
        "meas_variant": "BC",
    }
    if accession:
        # The evidence bridge accepts this provenance field as soon as callers
        # propagate it from the database layer; the current app does not yet.
        result["accession"] = accession
    return result


def test_current_result_mapping_bridge_recovers_exact_typed_evidence() -> None:
    result = _current_result_mapping(
        "Quartz",
        "/library/RRUFF/R040031.txt",
        0.90,
    )
    result.update(
        {
            "shift": 2,
            "shift_cm1": 2.0,
            "shape_similarity": 0.91,
            "gradient_similarity": 0.81,
            "similarity": 0.89,
            "pcs": 0.70,
            "rank_score": 0.8672,
            "rank_components": {
                "shape": 0.91,
                "gradient": 0.81,
                "pcs": 0.70,
                "shape_contribution": 0.64064,
                "gradient_contribution": 0.14256,
                "pcs_contribution": 0.084,
            },
        }
    )

    evidence = matching.reference_match_evidence_from_mapping(result)

    assert evidence.phase_name == "Quartz"
    assert evidence.reference_id == "/library/RRUFF/R040031.txt"
    assert evidence.independence_group == "unresolved:quartz"
    assert evidence.source == ""
    assert evidence.rank.spectral_similarity == pytest.approx(0.89)
    assert evidence.rank.final_rank_score == pytest.approx(0.8672)
    assert evidence.rank.shape_contribution == pytest.approx(0.64064)
    assert evidence.alignment.fitted_shift_points == 2
    assert evidence.alignment.requested_point_count == 100
    assert evidence.alignment.reference_support_point_count == 100
    assert evidence.alignment.shifted_reference_support_point_count == 100
    assert evidence.alignment.common_point_count == 80

    clipped = _current_result_mapping("Edge phase", "/library/edge.txt", 0.80)
    clipped.update(
        {
            "shift": -2,
            "shift_cm1": -2.0,
            "shift_boundary_hit": True,
            "grid_boundary_clipped": True,
            "common_point_count": 4,
            "coverage_fraction": 0.4,
            "reference_overlap_fraction": 1.0,
            "start_idx": 1,
            "end_idx": 5,
        }
    )
    clipped_evidence = matching.alignment_evidence_from_mapping(
        clipped,
        maximum_shift_points=2,
    )
    assert clipped_evidence.requested_point_count == 10
    assert clipped_evidence.reference_support_point_count == 5
    assert clipped_evidence.shifted_reference_support_point_count == 4
    assert clipped_evidence.reference_support_clipped_at_grid_boundary


def test_mapping_phase_bridge_groups_variants_by_accession_before_ranking() -> None:
    results = [
        _current_result_mapping(
            "Phase A",
            "/library/raw/a.txt",
            0.99,
            accession="R-A1",
            db_variant="DB-RAW",
        ),
        _current_result_mapping(
            "Phase A",
            "/library/processed/a.txt",
            0.01,
            accession="R-A1",
            db_variant="DB-BC",
        ),
        _current_result_mapping(
            "Phase A",
            "/library/raw/a2.txt",
            0.55,
            accession="R-A2",
        ),
        _current_result_mapping(
            "Phase B",
            "/library/raw/b1.txt",
            0.76,
            accession="R-B1",
        ),
        _current_result_mapping(
            "Phase B",
            "/library/raw/b2.txt",
            0.74,
            accession="R-B2",
        ),
    ]

    phases = matching.phase_evidence_from_mappings(results)

    assert [phase.phase_name for phase in phases] == ["Phase B", "Phase A"]
    phase_b, phase_a = phases
    assert phase_b.aggregate_score == pytest.approx(0.76)
    assert phase_a.aggregate_score == pytest.approx(0.55)
    assert phase_a.independent_reference_count == 2
    assert phase_a.reference_variant_count == 3
    assert sorted(phase_a.group_scores) == pytest.approx([0.50, 0.55])


def test_mapping_bridge_rejects_stale_rank_and_ambiguous_zero_support() -> None:
    valid = _current_result_mapping("Quartz", "/library/quartz.txt", 0.90)
    stale_rank = {**valid, "rank_score": 0.70}
    with pytest.raises(ValueError, match="rank_score is inconsistent"):
        matching.reference_match_evidence_from_mapping(stale_rank)

    stale_components = {
        **valid,
        "rank_components": {
            **dict(valid["rank_components"]),
            "shape": 0.40,
        },
    }
    with pytest.raises(ValueError, match=r"rank_components\.shape is inconsistent"):
        matching.reference_match_evidence_from_mapping(stale_components)

    zero_support = {
        **valid,
        "common_point_count": 0,
        "coverage_fraction": 0.0,
        "reference_overlap_fraction": 0.0,
    }
    with pytest.raises(ValueError, match="cannot infer requested_point_count"):
        matching.reference_match_evidence_from_mapping(zero_support)


def test_peak_consistency_never_bridges_a_gap_in_the_valid_mask() -> None:
    query = np.zeros(100, dtype=float)
    candidate = np.zeros(100, dtype=float)
    mask = np.zeros(100, dtype=bool)
    mask[0:15] = True
    mask[70:85] = True

    # Compressing mask=True values would place these peaks five positions apart,
    # despite their belonging to different support runs 60 grid points apart.
    query[11:14] = [0.2, 1.0, 0.2]
    candidate[71:74] = [0.2, 1.0, 0.2]

    score, peak_f1, height_rho = matching.peak_consistency_score(
        query,
        candidate,
        mask,
        tolerance_points=5,
    )

    assert score == 0.0
    assert peak_f1 == 0.0
    assert height_rho == 0.0


def test_gapped_reference_support_is_exact_through_alignment_and_residuals() -> None:
    rng = np.random.default_rng(20260830)
    candidate = rng.uniform(0.1, 2.0, size=32)
    query = matching.shift_candidate(candidate, 1)
    support_runs = ((2, 9), (20, 27))

    result = matching.best_aligned_score(
        query,
        candidate,
        np.ones(candidate.size, dtype=bool),
        2,
        27,
        max_shift=1,
        gradient_weight=0.0,
        support_runs=support_runs,
    )

    expected_mask = np.zeros(candidate.size, dtype=bool)
    expected_mask[3:11] = True
    expected_mask[21:29] = True
    assert result.evidence.fitted_shift_points == 1
    assert result.evidence.reference_support_point_count == 16
    assert result.evidence.shifted_reference_support_point_count == 16
    assert result.evidence.common_point_count == 16
    assert result.evidence.query_coverage_fraction == pytest.approx(0.5)
    assert result.spectral_similarity == pytest.approx(1.0)
    np.testing.assert_array_equal(result.comparison_mask, expected_mask)
    assert not np.any(result.comparison_mask[11:21])

    projection = matching.build_residual_projection(
        query,
        candidate,
        np.ones(candidate.size, dtype=bool),
        2,
        27,
        1,
        support_runs=support_runs,
        minimum_common_points=10,
    )
    np.testing.assert_array_equal(projection.comparison_mask, expected_mask)
    assert projection.common_point_count == 16


def test_alignment_ties_prefer_zero_shift_and_accept_precomputed_query_gradient() -> None:
    candidate = np.linspace(1.0, 100.0, 100)
    query = candidate.copy()
    query_mask = np.ones(candidate.size, dtype=bool)

    ordinary = matching.best_aligned_score(
        query,
        candidate,
        query_mask,
        10,
        89,
        max_shift=5,
    )
    precomputed = matching.best_aligned_score(
        query,
        candidate,
        query_mask,
        10,
        89,
        max_shift=5,
        query_gradient=np.gradient(query),
    )

    assert ordinary.evidence.fitted_shift_points == 0
    assert not ordinary.evidence.shift_search_boundary_hit
    assert precomputed.evidence == ordinary.evidence
    assert precomputed.spectral_similarity == ordinary.spectral_similarity

    # With locally offset-invariant linear traces all shifts score identically;
    # a shifted native interval can nevertheless cover more measured points.
    support_wins = matching.best_aligned_score(
        query,
        candidate,
        np.arange(candidate.size) >= 10,
        0,
        89,
        max_shift=5,
    )
    assert support_wins.evidence.fitted_shift_points == 5
    assert support_wins.evidence.common_point_count == 85


def test_coverage_filter_counts_support_runs_and_query_holes_exactly() -> None:
    pack = {
        "grid_info": {"min": 0, "step": 1, "len": 20},
        "meta": [
            {"start_idx": 0, "end_idx": 19, "support_runs": ((0, 4), (15, 19))},
            {"start_idx": 0, "end_idx": 19},
        ],
    }
    allowed = np.array([0, 1], dtype=np.int32)

    full_range = matching._coverage_eligible_ids(
        pack,
        allowed,
        0,
        19,
        0.75,
    )
    np.testing.assert_array_equal(full_range, [1])

    query_mask = np.zeros(20, dtype=bool)
    query_mask[0:5] = True
    query_mask[15:20] = True
    exact_query = matching._coverage_eligible_ids(
        pack,
        allowed,
        0,
        19,
        1.0,
        query_mask=query_mask,
    )
    np.testing.assert_array_equal(exact_query, [0, 1])


def test_shortlist_screening_excludes_values_inside_reference_support_holes() -> None:
    query = np.sin(np.linspace(0.0, 8.0, 30)) + 1.5
    gapped_runs = ((0, 9), (20, 29))
    invalid_on_support = np.ones(30, dtype=np.float32)
    invalid_on_support[10:20] = query[10:20]
    exact_match = np.full(30, 1000.0, dtype=np.float32)
    exact_match[0:10] = query[0:10]
    exact_match[20:30] = query[20:30]
    matrix = np.vstack((invalid_on_support, exact_match)).astype(np.float32)
    metadata = [
        {"start_idx": 0, "end_idx": 29, "support_runs": gapped_runs},
        {"start_idx": 0, "end_idx": 29, "support_runs": gapped_runs},
    ]

    shortlisted = matching.topk_cosine_subset(
        query,
        matrix,
        metadata,
        np.array([0, 1], dtype=np.int32),
        2,
        support_mask=np.ones(30, dtype=bool),
    )

    # Row zero agrees only inside the unsupported hole and has zero variance on
    # true support, so it is not a valid cosine candidate.
    assert shortlisted == [1]


def test_evidence_requires_a_runner_up_and_minimum_support_from_every_group() -> None:
    no_runner_up = matching.decide_evidence_status(
        matching.rank_phases(
            [
                _reference("Only phase", "only-1", "specimen-1", 0.92),
                _reference("Only phase", "only-2", "specimen-2", 0.90),
            ]
        )
    )
    assert no_runner_up.status is matching.EvidenceStatus.AMBIGUOUS
    assert no_runner_up.runner_up_phase is None
    assert no_runner_up.reasons == ("phase_separation_not_assessed",)

    weak_group = matching.decide_evidence_status(
        matching.rank_phases(
            [
                _reference("Leading", "lead-1", "lead-group-1", 0.95),
                _reference(
                    "Leading",
                    "lead-2",
                    "lead-group-2",
                    0.93,
                    common_points=10,
                ),
                _reference("Runner", "run-1", "run-group-1", 0.60),
                _reference("Runner", "run-2", "run-group-2", 0.58),
            ]
        )
    )
    assert weak_group.status is matching.EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert weak_group.reasons == ("too_few_common_points",)

    weak_runner_up = matching.decide_evidence_status(
        matching.rank_phases(
            [
                _reference("Leading", "lead-1", "lead-group-1", 0.95),
                _reference("Leading", "lead-2", "lead-group-2", 0.93),
                _reference(
                    "Poorly covered alternative",
                    "weak-runner",
                    "weak-runner-group",
                    0.70,
                    common_points=10,
                ),
            ]
        )
    )
    assert weak_runner_up.status is matching.EvidenceStatus.AMBIGUOUS
    assert "phase_separation_not_assessed" in weak_runner_up.reasons


def test_unrelated_weak_library_specimen_does_not_poison_leading_evidence() -> None:
    decision = matching.decide_evidence_status(
        matching.rank_phases(
            [
                _reference("Leading", "lead-1", "lead-group-1", 0.95),
                _reference("Leading", "lead-2", "lead-group-2", 0.93),
                _reference(
                    "Leading",
                    "unrelated-weak-variant",
                    "lead-group-3",
                    0.10,
                    coverage=0.10,
                    common_points=10,
                ),
                _reference("Runner", "run-1", "run-group-1", 0.62),
                _reference("Runner", "run-2", "run-group-2", 0.60),
            ]
        )
    )

    assert decision.status is matching.EvidenceStatus.SUPPORTED_CANDIDATE
    assert decision.best_phase == "Leading"


def test_unprovenanced_files_do_not_count_as_independent_specimens() -> None:
    mapped = matching.phase_evidence_from_mappings(
        [
            _current_result_mapping("Quartz", "/library/copy-one.txt", 0.90),
            _current_result_mapping("Quartz", "/library/copy-two.txt", 0.88),
        ]
    )
    assert mapped[0].independent_reference_count == 1
    assert {
        match.independence_group for match in mapped[0].supporting_matches
    } == {"unresolved:quartz"}

    query = np.sin(np.linspace(0.0, 6.0, 60)) + 1.2
    matrix = np.vstack((query, query * 0.99)).astype(np.float32)
    pack = {
        "X": matrix,
        "grid_info": {"min": 0, "step": 1, "len": 60},
        "meta": [
            {
                "name": "Quartz",
                "formula": "SiO2",
                "path": "/library/copy-one.txt",
                "start_idx": 0,
                "end_idx": 59,
                "l2": 1.0,
            },
            {
                "name": "Quartz",
                "formula": "SiO2",
                "path": "/library/copy-two.txt",
                "start_idx": 0,
                "end_idx": 59,
                "l2": 1.0,
            },
        ],
    }
    refined = matching.refine_and_rank(
        query,
        np.ones(60, dtype=bool),
        [0, 1],
        pack,
        2,
        parameters=matching.MatchingParameters(maximum_shift_points=0),
    )
    assert {result["independence_group"] for result in refined} == {
        "unresolved:quartz"
    }

    for metadata_row in pack["meta"]:
        metadata_row["provenance"] = {"accession": "R040031", "source": "RRUFF"}
    provenance_refined = matching.refine_and_rank(
        query,
        np.ones(60, dtype=bool),
        [0, 1],
        pack,
        2,
        parameters=matching.MatchingParameters(maximum_shift_points=0),
    )
    assert {result["independence_group"] for result in provenance_refined} == {
        "R040031"
    }
    assert {result["accession"] for result in provenance_refined} == {"R040031"}


def test_rruff_raw_processed_pair_shares_acquisition_but_orientation_does_not() -> None:
    query = np.sin(np.linspace(0.0, 5.0, 60)) + 1.2
    matrix = np.vstack((query, query * 0.99, query[::-1])).astype(np.float32)
    filenames = [
        "Quartz__R040031-3__Raman__514__0-000____Raman_Data_RAW__aaaa.txt",
        "Quartz__R040031-3__Raman__514__0-000____Raman_Data_Processed__bbbb.txt",
        "Quartz__R040031-3__Raman__514__90-000__ccw__Raman_Data_RAW__cccc.txt",
    ]
    pack = {
        "X": matrix,
        "grid_info": {"min": 0, "step": 1, "len": 60},
        "meta": [
            {
                "name": "Quartz",
                "formula": "SiO2",
                "path": f"/library/{filename}",
                "orig_filename": filename,
                "start_idx": 0,
                "end_idx": 59,
                "l2": 1.0,
                "source_root": "RRUFF",
                "provenance": {
                    "database": "RRUFF",
                    "accession": "R040031",
                },
            }
            for filename in filenames
        ],
    }

    refined = matching.refine_and_rank(
        query,
        np.ones(60, dtype=bool),
        [0, 1, 2],
        pack,
        3,
        parameters=matching.MatchingParameters(maximum_shift_points=0),
    )
    groups = {
        result["orig_filename"]: result["acquisition_group"]
        for result in refined
    }

    assert groups[filenames[0]] == groups[filenames[1]]
    assert groups[filenames[0]] != groups[filenames[2]]
    assert {result["independence_group"] for result in refined} == {"R040031"}


def test_match_query_excludes_primary_phase_before_evidence_aggregation() -> None:
    grid = np.arange(80, dtype=float)
    primary = np.exp(-((grid - 25.0) / 5.0) ** 2) + 0.7 * np.exp(
        -((grid - 55.0) / 7.0) ** 2
    )
    primary_variant = primary * 0.98 + 0.01
    phase_b = np.exp(-((grid - 31.0) / 6.0) ** 2) + 0.5 * np.exp(
        -((grid - 60.0) / 8.0) ** 2
    )
    phase_c = np.exp(-((grid - 18.0) / 7.0) ** 2) + 0.4 * np.exp(
        -((grid - 48.0) / 6.0) ** 2
    )
    spectra = np.vstack((primary, primary_variant, phase_b, phase_c)).astype(np.float32)

    def row(name: str, accession: str, index: int) -> dict[str, object]:
        return {
            "name": name,
            "formula": "?",
            "path": f"/library/{index}.txt",
            "accession": accession,
            "start_idx": 0,
            "end_idx": 79,
            "support_runs": ((0, 79),),
            "l2": 1.0,
        }

    raw_pack = {
        "X": spectra,
        "grid_info": {"min": 0, "step": 1, "len": 80},
        "meta": [
            row("Primary", "P-1", 0),
            row("Primary", "P-2", 1),
            row("Phase B", "B-1", 2),
            row("Phase C", "C-1", 3),
        ],
    }
    baseline_pack = {
        "X": np.empty((0, 80), dtype=np.float32),
        "grid_info": {"min": 0, "step": 1, "len": 80},
        "meta": [],
    }
    results = matching.match_query_vector(
        primary,
        np.ones(80, dtype=bool),
        0,
        79,
        raw_pack,
        baseline_pack,
        np.arange(4, dtype=np.int32),
        np.array([], dtype=np.int32),
        "BC",
        top_n=4,
        parameters=matching.MatchingParameters(maximum_shift_points=0),
        excluded_phase_keys={matching.phase_key("Primary")},
    )

    assert {matching.phase_key(result["name"]) for result in results} == {
        matching.phase_key("Phase B"),
        matching.phase_key("Phase C"),
    }
    for result in results:
        assert matching.phase_key(result["evidence_best_phase"]) != "primary"
        assert matching.phase_key(result["evidence_runner_up_phase"]) != "primary"


def test_residual_peak_gate_rejects_background_only_similarity() -> None:
    query = np.linspace(0.1, 1.0, 80, dtype=np.float32)
    raw_pack = {
        "X": query.reshape(1, -1),
        "grid_info": {"min": 0, "step": 1, "len": query.size},
        "meta": [
            {
                "name": "Background slope",
                "formula": "?",
                "path": "/library/background.txt",
                "start_idx": 0,
                "end_idx": query.size - 1,
                "support_runs": ((0, query.size - 1),),
                "l2": float(np.linalg.norm(query)),
            }
        ],
    }
    empty_pack = {
        "X": np.empty((0, query.size), dtype=np.float32),
        "grid_info": {"min": 0, "step": 1, "len": query.size},
        "meta": [],
    }
    ordinary = matching.match_query_vector(
        query,
        np.ones(query.size, dtype=bool),
        0,
        query.size - 1,
        raw_pack,
        empty_pack,
        np.array([0], dtype=np.int32),
        np.array([], dtype=np.int32),
        "BC",
        top_n=1,
        parameters=matching.MatchingParameters(maximum_shift_points=0),
    )
    peak_supported_only = matching.match_query_vector(
        query,
        np.ones(query.size, dtype=bool),
        0,
        query.size - 1,
        raw_pack,
        empty_pack,
        np.array([0], dtype=np.int32),
        np.array([], dtype=np.int32),
        "BC",
        top_n=1,
        parameters=matching.MatchingParameters(
            maximum_shift_points=0,
            minimum_candidate_peak_consistency=1.0e-6,
        ),
    )

    assert len(ordinary) == 1
    assert ordinary[0]["pcs"] == 0.0
    assert peak_supported_only == []


def test_signed_residual_similarity_does_not_lift_negative_trough_to_zero() -> None:
    axis = np.arange(101, dtype=float)
    candidate = np.exp(-0.5 * ((axis - 50.0) / 8.0) ** 2).astype(np.float32)
    # This is the same positive shape translated deeply below zero. It is an
    # over-subtraction pattern, not evidence for adding another positive phase.
    signed_residual = (candidate - 0.8).astype(np.float32)
    row = {
        "name": "Offset artefact",
        "formula": "?",
        "path": "/library/offset-artefact.txt",
        "start_idx": 0,
        "end_idx": candidate.size - 1,
        "support_runs": ((0, candidate.size - 1),),
        "l2": float(np.linalg.norm(candidate)),
    }
    pack = {
        "X": candidate.reshape(1, -1),
        "grid_info": {"min": 0, "step": 1, "len": candidate.size},
        "meta": [row],
    }
    empty_pack = {
        "X": np.empty((0, candidate.size), dtype=np.float32),
        "grid_info": {"min": 0, "step": 1, "len": candidate.size},
        "meta": [],
    }

    offset_invariant = matching.match_query_vector(
        signed_residual,
        np.ones(candidate.size, dtype=bool),
        0,
        candidate.size - 1,
        pack,
        empty_pack,
        np.array([0], dtype=np.int32),
        np.array([], dtype=np.int32),
        "BC",
        top_n=1,
        parameters=matching.MatchingParameters(
            maximum_shift_points=0,
            minimum_candidate_peak_consistency=0.01,
            remove_query_local_offset=True,
        ),
    )
    signed = matching.match_query_vector(
        signed_residual,
        np.ones(candidate.size, dtype=bool),
        0,
        candidate.size - 1,
        pack,
        empty_pack,
        np.array([0], dtype=np.int32),
        np.array([], dtype=np.int32),
        "BC",
        top_n=1,
        parameters=matching.MatchingParameters(
            maximum_shift_points=0,
            minimum_candidate_peak_consistency=0.01,
            remove_query_local_offset=False,
        ),
    )

    assert offset_invariant[0]["shape_similarity"] > 0.99
    assert signed[0]["shape_similarity"] < 0.20
    assert signed[0]["rank_score"] < 0.30

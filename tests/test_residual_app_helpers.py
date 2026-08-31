from __future__ import annotations

import numpy as np

import RamanPhaseID_0p99beta as app


def test_result_variant_selects_exact_scored_cache_even_when_not_newly_corrected() -> None:
    raw_candidate = np.linspace(0.2, 1.0, 40, dtype=np.float32)
    baseline_candidate = np.exp(
        -0.5 * ((np.arange(40, dtype=float) - 20.0) / 4.0) ** 2
    ).astype(np.float32)
    raw_pack = {"X": raw_candidate.reshape(1, -1), "meta": [{}]}
    baseline_pack = {"X": baseline_candidate.reshape(1, -1), "meta": [{}]}
    selected = {
        "name": "Processed reference",
        "db_idx": 0,
        "db_variant": "DB-BC",
        # This records whether a new baseline was applied, not the matrix that
        # earned the score, and is legitimately false for processed sources.
        "db_baseline": False,
        "start_idx": 0,
        "end_idx": 39,
        "support_runs": ((0, 39),),
        "shift": 0,
    }

    projection = app._build_residual_query_vector(
        1.5 * baseline_candidate,
        np.ones(40, dtype=bool),
        selected,
        raw_pack,
        baseline_pack,
    )

    assert projection is not None
    assert app._result_uses_baseline_pack(selected)
    np.testing.assert_allclose(projection.aligned_candidate, baseline_candidate)
    np.testing.assert_allclose(projection.signed_residual, 0.0, atol=1e-7)


def test_residual_reference_selection_excludes_raw_source_backgrounds() -> None:
    raw_metadata = [
        {"provenance": {"processing": "raw"}},
        {"provenance": {"processing": "processed"}},
        {"provenance": {"processing": "raw"}},
        {"provenance": {"processing": "processed"}},
    ]
    baseline_metadata = [
        {"db_baseline": True},
        {"db_baseline": False},
        {"db_baseline": True},
        {"db_baseline": False},
    ]

    raw_ids, baseline_ids = app._background_neutral_residual_reference_ids(
        np.arange(4, dtype=np.int32),
        np.arange(4, dtype=np.int32),
        raw_metadata,
        baseline_metadata,
    )

    np.testing.assert_array_equal(raw_ids, [1, 3])
    np.testing.assert_array_equal(baseline_ids, [0, 2])
    assert not raw_ids.flags.writeable
    assert not baseline_ids.flags.writeable


def test_fixed_full_grid_uses_2000_cm1_initial_matching_limit() -> None:
    assert app.DATABASE_GRID == {"min": 60, "max": 4000, "step": 1}
    assert app._initial_matching_range(60, 4000) == (60, 2000)
    assert app._initial_matching_range(100, 1800) == (100, 1800)
    assert app._initial_matching_range(2200, 4000) == (2200, 4000)


def test_residual_display_gate_rejects_unknown_ranking_but_keeps_ambiguity() -> None:
    assert not app._residual_candidates_are_actionable([])
    assert not app._residual_candidates_are_actionable(
        [{"evidence_status": "unknown_or_out_of_library"}]
    )
    assert not app._residual_candidates_are_actionable(
        [{"evidence_status": "insufficient_evidence"}]
    )
    assert app._residual_candidates_are_actionable(
        [{"evidence_status": "ambiguous"}]
    )
    assert app._residual_candidates_are_actionable(
        [{"evidence_status": "supported_candidate"}]
    )

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest

from raman_workflow import (
    BaselineConfig,
    CalibrationConfig,
    MatchingConfig,
    PrimaryResultSnapshot,
    ResidualReferenceIdentity,
    ResidualResultIdentity,
    ResidualResultSnapshot,
    ResultIdentity,
    SmoothingConfig,
    SpectralRange,
    UploadIdentity,
    WhiteReferenceConfig,
    WorkflowOrderError,
    WorkflowState,
    WorkflowValidationError,
    canonical_json,
    payload_signature,
    residual_query_content_sha256,
)


def _upload(name: str = "measurement.txt", content: bytes = b"100 1\n101 2\n") -> UploadIdentity:
    return UploadIdentity.from_bytes(name, content)


def _matching(low: float = 100.0, high: float = 1800.0) -> MatchingConfig:
    return MatchingConfig(
        range_cm1=SpectralRange(low, high),
        database_folders=("databases/OWN", "databases/ROD", "databases/RRUFF"),
        raw_database_signature="raw-signature",
        baseline_database_signature="bc-signature",
    )


def _complete_state() -> WorkflowState:
    state = WorkflowState().set_measurement(_upload())
    state = state.apply_input().apply_baseline().apply_smoothing()
    return state.with_matching(_matching()).apply_matching().record_result()


def _config_token(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:12]


def test_payload_serialization_is_deterministic_and_config_token_compatible() -> None:
    left = {"z": 1, "a": {"second": 2, "first": 1}}
    right = {"a": {"first": 1, "second": 2}, "z": 1}

    assert canonical_json(left) == canonical_json(right)
    assert payload_signature(left) == payload_signature(right)

    baseline = BaselineConfig()
    assert baseline.lam_exp == 5
    assert baseline.lam == 1e5
    assert SmoothingConfig().window == 5
    smoothing = SmoothingConfig.from_mapping(
        {"enabled": True, "window": 11, "poly": 3}
    )
    white_reference = WhiteReferenceConfig()
    assert baseline.token == _config_token(baseline.payload())
    assert smoothing.token == _config_token(smoothing.payload())
    assert white_reference.token == _config_token(white_reference.payload())


def test_upload_identity_tracks_filename_hash_and_size_without_bytes() -> None:
    upload = UploadIdentity.from_bytes("quartz.txt", b"spectrum")

    assert upload.filename == "quartz.txt"
    assert upload.sha1 == hashlib.sha1(b"spectrum").hexdigest()
    assert upload.size_bytes == 8
    assert upload.payload() == {
        "v": 1,
        "filename": "quartz.txt",
        "sha1": upload.sha1,
        "size_bytes": 8,
    }

    legacy = UploadIdentity.from_mapping(
        {
            "ref_filename": "lamp.csv",
            "ref_sha1": "A" * 40,
            "ref_size_bytes": 123,
        }
    )
    assert legacy == UploadIdentity("lamp.csv", "a" * 40, 123)


def test_legacy_configuration_mappings_are_normalized() -> None:
    raw = BaselineConfig.from_mapping(
        {"baseline_mode": "RAW (no baseline)", "lam_exp": 5}
    )
    als = BaselineConfig.from_mapping(
        {"method": "ALS (IAsLS)", "lam_exp": 6, "lam1_exp": 3, "p": 0.02}
    )
    legacy_savgol = SmoothingConfig.from_mapping(
        {"enabled": True, "window": 15, "poly": 2}
    )
    disabled = SmoothingConfig.from_mapping(
        {"enabled": False, "window": 99, "poly": 8}
    )
    ai = SmoothingConfig.from_mapping(
        {"method": "AI-assisted · guarded DeepeR (full range)"}
    )

    assert raw.method == "NONE"
    assert raw.measurement_mode == "RAW"
    assert als.method == "ALS"
    assert als.payload()["lam1"] == 1e3
    assert legacy_savgol.method == "savgol"
    assert legacy_savgol.payload()["window"] == 15
    assert disabled.method == "none"
    assert disabled.payload() == {
        "v": 5,
        "grid_step_cm1": 1.0,
        "method": "none",
    }
    assert ai.method == "deeper_ai"
    assert ai.payload()["model_id"] == "DeepeR-ResUNet-500"

    ref = WhiteReferenceConfig.from_mapping(
        {
            "enabled": "true",
            "scale": 1.25,
            "ref_sha1": "b" * 40,
            "ref_filename": "white.txt",
        }
    )
    assert ref.is_ready
    assert ref.payload()["ref_sha1"] == "b" * 40
    assert ref.to_mapping()["ref_filename"] == "white.txt"


def test_matching_mapping_accepts_current_app_shape_and_canonicalizes_elements() -> None:
    config = MatchingConfig.from_mapping(
        {
            "range_low": 180,
            "range_high": 1750,
            "ultra": True,
            "include": "si, O, Si, invalid3",
            "exclude": "Fe; C",
            "mode": "Exactly this set",
            "allow": False,
            "reference_scope": "All experimental references",
            "folders": ["OWN", "ROD"],
            "sig_raw": "raw",
            "sig_bcb": "bcb",
            "match_selection_v": 7,
        }
    )

    assert config.range_cm1 == SpectralRange(180, 1750)
    assert config.include_elements == ("O", "Si")
    assert config.exclude_elements == ("C", "Fe")
    assert config.element_mode == "exact_set"
    assert config.allow_missing_formula is False
    assert config.selection_version == 7
    assert config.to_mapping()["mode"] == "Exactly this set"
    assert "ultra" not in config.to_mapping()
    assert "full_range" not in config.payload()
    assert "reference_scope" not in config.payload()


def test_workflow_requires_ordered_explicit_approvals() -> None:
    state = WorkflowState().set_measurement(_upload())

    with pytest.raises(WorkflowOrderError):
        state.apply_baseline()

    state = state.apply_input()
    assert state.next_required_stage == "baseline"
    with pytest.raises(WorkflowOrderError):
        state.apply_smoothing()

    state = state.apply_baseline().apply_smoothing()
    assert state.next_required_stage == "matching"
    with pytest.raises(WorkflowValidationError):
        state.apply_matching()

    state = state.with_matching(_matching()).apply_matching()
    assert state.next_required_stage == "matching"
    assert not state.has_current_result
    assert state.expected_result_identity is not None

    state = state.record_result()
    assert state.next_required_stage == "complete"
    assert state.has_current_result
    assert state.result_identity.matching_signature == state.matching_approval.signature
    assert state.active_result_signature == state.result_identity.signature


def test_draft_edits_mark_results_stale_but_do_not_destroy_applied_snapshot() -> None:
    complete = _complete_state()
    old_baseline = complete.baseline_approval
    old_smoothing = complete.smoothing_approval
    old_matching = complete.matching_approval

    draft = complete.with_baseline(replace(complete.baseline_draft, lam_exp=6, lam=1e6))

    assert draft.input_dirty is False
    assert draft.baseline_dirty is True
    assert draft.smoothing_dirty is True
    assert draft.matching_dirty is True
    assert draft.baseline_approval is old_baseline
    assert draft.smoothing_approval is old_smoothing
    assert draft.matching_approval is old_matching
    assert draft.has_current_result is False
    assert draft.active_result_signature is None

    applied = draft.apply_baseline()
    assert applied.input_approval == complete.input_approval
    assert applied.baseline_approval != old_baseline
    assert applied.smoothing_approval is None
    assert applied.matching_approval is None


def test_idempotent_apply_preserves_downstream_approvals() -> None:
    complete = _complete_state()

    assert complete.apply_input() is complete
    assert complete.apply_baseline() is complete
    assert complete.apply_smoothing() is complete
    assert complete.apply_matching() is complete


def test_inactive_method_fields_do_not_invalidate_semantically_equal_approvals() -> None:
    complete = _complete_state()
    changed_unused_fields = complete.with_baseline(
        replace(complete.baseline_draft, p=0.123, niter=77)
    ).with_smoothing(
        replace(complete.smoothing_draft, max_change_sigma=2.75)
    )

    assert changed_unused_fields.baseline_dirty is False
    assert changed_unused_fields.smoothing_dirty is False
    assert changed_unused_fields.matching_dirty is False
    assert changed_unused_fields.apply_baseline() is changed_unused_fields
    assert changed_unused_fields.apply_smoothing() is changed_unused_fields
    assert changed_unused_fields.matching_approval is complete.matching_approval


def test_calibration_is_chained_through_every_approval_signature() -> None:
    original = _complete_state()
    original_signatures = (
        original.input_approval.signature,
        original.baseline_approval.signature,
        original.smoothing_approval.signature,
        original.matching_approval.signature,
    )

    draft = original.with_calibration(CalibrationConfig(shift_cm1=1.2))
    assert draft.input_dirty
    assert draft.input_approval == original.input_approval
    with pytest.raises(WorkflowOrderError):
        draft.apply_baseline()

    shifted = draft.apply_input()
    assert shifted.baseline_approval is None
    shifted = shifted.apply_baseline().apply_smoothing().apply_matching()
    shifted_signatures = (
        shifted.input_approval.signature,
        shifted.baseline_approval.signature,
        shifted.smoothing_approval.signature,
        shifted.matching_approval.signature,
    )
    assert all(before != after for before, after in zip(original_signatures, shifted_signatures))


def test_acquisition_metadata_is_auditable_and_unknown_axis_cannot_be_applied() -> None:
    calibration = CalibrationConfig(
        shift_cm1=0.4,
        axis_unit="cm^-1",
        calibrant="silicon 520.5 cm⁻¹",
        residual_cm1=0.2,
        excitation_wavelength_nm=532.0,
        spectral_resolution_cm1=2.0,
        instrument="Lab Raman A",
    )
    payload = calibration.payload()
    assert payload["calibrant"] == "silicon 520.5 cm⁻¹"
    assert payload["excitation_wavelength_nm"] == 532.0

    unknown_axis = (
        WorkflowState()
        .set_measurement(_upload())
        .with_calibration(CalibrationConfig(axis_unit="unknown"))
    )
    with pytest.raises(WorkflowValidationError, match="Raman shift"):
        unknown_axis.apply_input()


def test_display_viewport_is_separate_from_applied_matching_range() -> None:
    complete = _complete_state()
    result_signature = complete.active_result_signature
    match_range = complete.matching_approval.config.range_cm1

    zoomed = complete.with_display_viewport((440.0, 520.0))

    assert zoomed.display_viewport == SpectralRange(440.0, 520.0)
    assert zoomed.matching_approval.config.range_cm1 == match_range
    assert zoomed.active_result_signature == result_signature
    assert zoomed.matching_dirty is False


def test_matching_range_is_a_draft_until_explicitly_applied() -> None:
    complete = _complete_state()
    old_approval = complete.matching_approval

    draft = complete.with_matching(complete.matching_draft.with_range((250.0, 1400.0)))
    assert draft.matching_dirty
    assert draft.matching_approval is old_approval

    applied = draft.apply_matching()
    assert applied.matching_approval != old_approval
    assert applied.matching_approval.config.range_cm1 == SpectralRange(250.0, 1400.0)
    assert not applied.has_current_result
    assert applied.result_is_stale

    completed = applied.record_result()
    assert completed.has_current_result
    assert completed.result_identity != complete.result_identity


def test_retired_grid_and_scope_mapping_keys_cannot_change_matching_identity() -> None:
    common = {
        "range": (100.0, 1800.0),
        "folders": ("OWN", "ROD", "RRUFF"),
        "sig_raw": "raw-signature",
        "sig_bcb": "bc-signature",
    }
    former_default = MatchingConfig.from_mapping(
        {
            **common,
            "ultra": False,
            "reference_scope": "High-quality experimental references (recommended)",
        }
    )
    former_broadest = MatchingConfig.from_mapping(
        {
            **common,
            "ultra": True,
            "reference_scope": "All references including theoretical / non-target materials",
        }
    )

    assert former_default == former_broadest
    assert former_default.token == former_broadest.token


def test_complete_matching_policy_signature_is_part_of_result_identity() -> None:
    first = _matching()
    second = replace(first, policy_signature="policy-v2")

    assert first.token != second.token
    assert first.payload()["policy_signature"] == ""
    assert second.payload()["policy_signature"] == "policy-v2"


def test_measurement_replacement_invalidates_all_applied_state() -> None:
    complete = _complete_state().with_display_viewport((300.0, 900.0))
    replacement = _upload("replacement.txt", b"100 9\n101 8\n")

    changed = complete.set_measurement(replacement)

    assert changed.measurement == replacement
    assert changed.display_viewport is None
    assert changed.input_approval is None
    assert changed.baseline_approval is None
    assert changed.smoothing_approval is None
    assert changed.matching_approval is None
    assert changed.result_identity is None
    assert changed.next_required_stage == "input"


def test_enabled_white_reference_requires_an_active_reference_upload() -> None:
    state = (
        WorkflowState()
        .set_measurement(_upload())
        .with_white_reference(WhiteReferenceConfig(enabled=True, scale=1.1))
    )
    with pytest.raises(WorkflowValidationError):
        state.apply_input()

    reference = UploadIdentity.from_bytes("white.txt", b"100 0.1\n101 0.2\n")
    applied = state.with_white_reference_upload(reference).apply_input()
    assert applied.input_approval.white_reference.reference == reference
    assert applied.input_approval.payload()["white_reference_upload"]["filename"] == "white.txt"


def test_explicit_invalidation_has_clear_stage_boundaries() -> None:
    complete = _complete_state()

    after_baseline = complete.invalidate_downstream("baseline")
    assert after_baseline.input_approval is not None
    assert after_baseline.baseline_approval is not None
    assert after_baseline.smoothing_approval is None
    assert after_baseline.matching_approval is None
    assert after_baseline.result_is_stale

    from_baseline = complete.invalidate_from("baseline")
    assert from_baseline.input_approval is not None
    assert from_baseline.baseline_approval is None
    assert from_baseline.smoothing_approval is None
    assert from_baseline.matching_approval is None
    assert from_baseline.result_is_stale


def test_primary_result_snapshot_is_immutable_and_records_empty_success() -> None:
    approved = (
        WorkflowState()
        .set_measurement(_upload())
        .apply_input()
        .apply_baseline()
        .apply_smoothing()
        .with_matching(_matching())
        .apply_matching()
    )
    query = np.array([0.0, 0.5, 1.0], dtype=float)
    mask = np.array([False, True, True])
    snapshot = PrimaryResultSnapshot.from_workflow(
        approved,
        [],
        query,
        mask,
    )

    query[1] = 99.0
    mask[1] = False
    assert snapshot.is_empty
    np.testing.assert_allclose(snapshot.query_vector, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(snapshot.query_mask, [False, True, True])
    assert not snapshot.query_vector.flags.writeable
    assert not snapshot.query_mask.flags.writeable

    completed = approved.record_result(snapshot)
    assert completed.has_current_result
    assert completed.next_required_stage == "complete"


def test_result_snapshot_freezes_nested_result_mappings() -> None:
    approved = (
        WorkflowState()
        .set_measurement(_upload())
        .apply_input()
        .apply_baseline()
        .apply_smoothing()
        .with_matching(_matching())
        .apply_matching()
    )
    source = {"name": "Quartz", "rank": {"score": 0.9}, "runs": [[1, 2]]}
    snapshot = PrimaryResultSnapshot.from_workflow(
        approved,
        [source],
        np.array([0.0, 1.0]),
        np.array([True, True]),
    )
    source["name"] = "Changed"
    source["rank"]["score"] = 0.1

    assert snapshot.results[0]["name"] == "Quartz"
    assert snapshot.results[0]["rank"]["score"] == 0.9
    with pytest.raises(TypeError):
        snapshot.results[0]["name"] = "Changed"
    detached = snapshot.result_mappings()
    detached[0]["rank"]["score"] = 0.2
    assert snapshot.results[0]["rank"]["score"] == 0.9


def test_record_result_rejects_dirty_or_mismatched_approval() -> None:
    approved = (
        WorkflowState()
        .set_measurement(_upload())
        .apply_input()
        .apply_baseline()
        .apply_smoothing()
        .with_matching(_matching())
        .apply_matching()
    )
    mismatched = ResultIdentity("a" * 40)
    with pytest.raises(WorkflowValidationError, match="does not match"):
        approved.record_result(mismatched)

    dirty = approved.with_matching(_matching(200.0, 1700.0))
    with pytest.raises(WorkflowOrderError, match="apply all current"):
        dirty.record_result()


def test_public_state_and_config_objects_are_frozen() -> None:
    state = _complete_state()

    with pytest.raises(FrozenInstanceError):
        state.display_viewport = SpectralRange(100, 200)
    with pytest.raises(FrozenInstanceError):
        state.baseline_draft.lam = 1e8


def test_residual_result_snapshot_binds_exact_subtraction_and_numerical_content() -> None:
    approved = (
        WorkflowState()
        .set_measurement(_upload())
        .apply_input()
        .apply_baseline()
        .apply_smoothing()
        .with_matching(_matching())
        .apply_matching()
    )
    primary = PrimaryResultSnapshot.from_workflow(
        approved,
        [{"name": "Quartz"}],
        np.array([0.0, 0.4, 1.0, 0.2], dtype=np.float32),
        np.array([False, True, True, True]),
    )
    reference = ResidualReferenceIdentity(
        phase_name="Quartz",
        database_variant="DB-RAW",
        database_signature="raw-signature",
        database_index=17,
        reference_id="rruff:R050125",
        path="databases/RRUFF/R050125.txt",
        accession="R050125",
        filename="R050125.txt",
        fitted_shift_points=-2,
        fitted_shift_cm1=-2.0,
        start_idx=1,
        end_idx=3,
        support_runs=((1, 1), (3, 3)),
    )
    query = np.array([0.0, -0.2, 1.0, 0.3], dtype=np.float32)
    mask = np.array([False, True, True, True])
    signed = np.array([0.0, -0.4, 0.8, 0.2], dtype=float)
    policy_signature = payload_signature(
        {"minimum_common_points": 20, "minimum_fit_improvement_fraction": 0.02}
    )
    source_result = {"name": "Calcite", "rank": {"score": 0.82}}

    snapshot = ResidualResultSnapshot.from_primary(
        primary,
        reference,
        0.75,
        [source_result],
        query,
        mask,
        signed,
        policy_signature,
        {"fit_improvement_fraction": 0.25},
    )

    query[2] = 99.0
    mask[2] = False
    signed[2] = 99.0
    source_result["rank"]["score"] = 0.1
    assert snapshot.primary_identity == primary.identity
    assert snapshot.identity.subtracted_reference == reference
    assert snapshot.identity.scale_factor == pytest.approx(0.75)
    assert snapshot.identity.residual_query_mask_sha256 == residual_query_content_sha256(
        snapshot.query_vector,
        snapshot.query_mask,
    )
    assert snapshot.results[0]["rank"]["score"] == pytest.approx(0.82)
    assert not snapshot.query_vector.flags.writeable
    assert not snapshot.query_mask.flags.writeable
    assert not snapshot.signed_residual.flags.writeable
    assert snapshot.identity.payload()["subtracted_reference"]["database_index"] == 17
    assert snapshot.identity.payload()["subtracted_reference"]["support_runs"] == [
        [1, 1],
        [3, 3],
    ]


def test_residual_identity_changes_for_hit_scale_query_mask_or_policy() -> None:
    approved = (
        WorkflowState()
        .set_measurement(_upload())
        .apply_input()
        .apply_baseline()
        .apply_smoothing()
        .with_matching(_matching())
        .apply_matching()
    )
    primary = PrimaryResultSnapshot.from_workflow(
        approved,
        [],
        np.array([0.0, 1.0, 0.5]),
        np.array([True, True, True]),
    )
    reference = ResidualReferenceIdentity(
        phase_name="Quartz",
        database_variant="DB-BC",
        database_signature="bc-signature",
        database_index=3,
        reference_id="ref-3",
        path="db/q.txt",
        accession="Q3",
        filename="q.txt",
        fitted_shift_points=1,
        fitted_shift_cm1=1.0,
        start_idx=0,
        end_idx=2,
        support_runs=((0, 2),),
    )
    query = np.array([0.0, 0.8, -0.1], dtype=np.float32)
    mask = np.array([True, True, True])
    signed = np.array([0.0, 0.4, -0.05])
    policy = payload_signature({"policy": 1})

    def identity(**changes: object) -> ResidualResultIdentity:
        return ResidualResultIdentity.from_components(
            primary.identity,
            changes.get("reference", reference),
            changes.get("scale", 0.5),
            changes.get("query", query),
            changes.get("mask", mask),
            signed,
            changes.get("policy", policy),
        )

    baseline = identity()
    variants = (
        identity(reference=replace(reference, database_index=4)),
        identity(reference=replace(reference, fitted_shift_points=2)),
        identity(reference=replace(reference, support_runs=((0, 1),))),
        identity(scale=0.6),
        identity(query=np.array([0.0, 0.7, -0.1], dtype=np.float32)),
        identity(mask=np.array([True, True, False])),
        identity(policy=payload_signature({"policy": 2})),
    )
    assert all(candidate.signature != baseline.signature for candidate in variants)

    mismatched = identity(query=np.array([0.0, 0.7, -0.1], dtype=np.float32))
    with pytest.raises(WorkflowValidationError, match="numerical content"):
        ResidualResultSnapshot(
            identity=mismatched,
            primary_identity=primary.identity,
            results=(),
            query_vector=query,
            query_mask=mask,
            signed_residual=signed,
            diagnostics={},
        )

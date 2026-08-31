from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import raman_database as db


def _write_reference(path: Path, text: str = "100 1\n101 2\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_reference_catalog_summary_is_unique_sorted_and_compact() -> None:
    entries = [
        {"name": "Quartz", "path": "/unused/a.txt"},
        {"name": "quartz", "path": "/unused/b.txt"},
        {"name": "Calcite", "path": "/unused/c.txt"},
        {"name": "", "path": "/unused/d.txt"},
    ]
    summary = db.ReferenceCatalogSummary.from_entries(entries, skipped_count=3)

    assert summary.unique_names == ("Calcite", "Quartz")
    assert summary.reference_count == 4
    assert summary.skipped_count == 3


def test_reference_eligibility_request_is_a_complete_normalized_cache_key() -> None:
    request = db.ReferenceEligibilityRequest(
        raw_signature="raw-signature",
        baseline_signature="baseline-signature",
        library_variant="DB-RAW",
        include_elements=("Si", "O", "Si"),
        exclude_elements=("Fe", ""),
        element_mode="Must include all",
        allow_missing_formula=False,
        filtering_policy_version=3,
    )

    assert request.library_variant == "raw"
    assert request.library_signature == "raw-signature"
    assert request.include_elements == ("O", "Si")
    assert request.exclude_elements == ("Fe",)
    assert request.element_mode == "must_include_all"
    assert replace(request, library_variant="DB-BC").library_signature == "baseline-signature"
    assert replace(request, filtering_policy_version=4) != request


def test_reference_eligibility_keeps_all_reference_classes_and_filters_each_variant_once() -> None:
    rows = [
        {
            "name": "excellent raw silica",
            "elements": ("O", "Si"),
            "has_formula": True,
            "provenance": {
                "database": "RRUFF",
                "quality": "excellent",
                "determination": "experimental",
                "processing": "raw",
            },
        },
        {
            "name": "poor raw silica",
            "elements": ("O", "Si"),
            "has_formula": True,
            "provenance": {
                "database": "RRUFF",
                "quality": "poor",
                "determination": "experimental",
                "processing": "raw",
            },
        },
        {
            "name": "excellent iron oxide",
            "elements": ("Fe", "O"),
            "has_formula": True,
            "provenance": {
                "database": "RRUFF",
                "quality": "excellent",
                "determination": "experimental",
                "processing": "raw",
            },
        },
        {
            "name": "unknown formula",
            "elements": (),
            "has_formula": False,
            "provenance": {"database": "OWN", "processing": "raw"},
        },
        {
            "name": "excellent processed silica",
            "elements": ("O", "Si"),
            "has_formula": True,
            "provenance": {
                "database": "RRUFF",
                "quality": "excellent",
                "determination": "experimental",
                "processing": "processed",
            },
        },
        {
            "name": "theoretical silica",
            "elements": ("O", "Si"),
            "has_formula": True,
            "provenance": {
                "database": "ROD",
                "quality": "excellent",
                "determination": "theoretical",
                "processing": "raw",
            },
        },
    ]

    def is_processed(row) -> bool:
        return row.get("provenance", {}).get("processing") == "processed"

    raw_request = db.ReferenceEligibilityRequest(
        raw_signature="raw-signature",
        baseline_signature="baseline-signature",
        library_variant="raw",
        include_elements=("Si",),
        exclude_elements=("Fe",),
        element_mode="Must include all",
        allow_missing_formula=False,
        filtering_policy_version=1,
    )
    raw = db.compute_reference_eligibility(
        rows,
        raw_request,
        is_already_processed=is_processed,
    )
    baseline = db.compute_reference_eligibility(
        rows,
        replace(raw_request, library_variant="baseline_corrected"),
        is_already_processed=is_processed,
    )

    # Poor/unrated and theoretical sources are not silently removed. Explicit
    # chemistry still excludes the iron-bearing and unknown-formula rows.
    np.testing.assert_array_equal(raw.row_ids, [0, 1, 4, 5])
    np.testing.assert_array_equal(baseline.row_ids, [0, 1, 5])
    assert raw.scanned_count == len(rows)
    assert raw.eligible_count == 4
    assert not raw.row_ids.flags.writeable


def _write_legacy_pack(
    directory: Path,
    *,
    source_paths: list[str],
    baseline: bool,
    grid: db.GridSpec | None = None,
    trailing_vector_bytes: bytes = b"",
    metadata_row_delta: int = 0,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    grid = grid or db.GridSpec(60.0, 63.0, 1.0, 4)
    (directory / db.DEFAULT_GRID_FILE).write_text(
        json.dumps(grid.as_dict()), encoding="utf-8"
    )

    rows = []
    for index, source_path in enumerate(source_paths):
        rows.append(
            {
                "name": f"phase-{index}",
                "formula": "SiO2",
                "flag": "s",
                "filename": Path(source_path).name,
                "orig_filename": Path(source_path).name,
                "path": source_path,
                "start_idx": 0,
                "end_idx": grid.length - 1,
                "l2": float(index + 1),
                "elements": ["O", "Si"],
                "has_formula": True,
                "parser_version": 1,
                "db_baseline": baseline,
            }
        )
    if metadata_row_delta > 0:
        rows.extend(rows[:metadata_row_delta])
    elif metadata_row_delta < 0:
        rows = rows[:metadata_row_delta]
    (directory / db.DEFAULT_METADATA_FILE).write_text(
        json.dumps(rows), encoding="utf-8"
    )

    vectors = np.arange(len(source_paths) * grid.length, dtype=np.float32)
    with (directory / db.DEFAULT_VECTOR_FILE).open("wb") as handle:
        handle.write(vectors.tobytes())
        handle.write(trailing_vector_bytes)


def test_inventory_signature_is_portable_and_uses_mtime_ns(tmp_path: Path) -> None:
    first_root = tmp_path / "copy-a" / "databases" / "OWN"
    second_root = tmp_path / "copy-b" / "databases" / "OWN"
    first_file = first_root / "nested" / "phase.txt"
    second_file = second_root / "nested" / "phase.txt"
    _write_reference(first_file)
    _write_reference(second_file)
    timestamp_ns = 1_812_345_678_901_234_567
    os.utime(first_file, ns=(timestamp_ns, timestamp_ns))
    os.utime(second_file, ns=(timestamp_ns, timestamp_ns))

    first = db.scan_database_inventory({"OWN": first_root})
    second = db.scan_database_inventory({"OWN": second_root})

    assert first.signature == second.signature
    assert first.files[0].mtime_ns == timestamp_ns
    assert first.files[0].relative_path == "nested/phase.txt"

    os.utime(second_file, ns=(timestamp_ns + 1, timestamp_ns + 1))
    changed = db.scan_database_inventory({"OWN": second_root})
    assert changed.signature != first.signature

    grid = db.GridSpec(60.0, 63.0, 1.0, 4)
    raw_signature = db.derive_precompute_signature(
        first,
        grid,
        variant="DB-RAW",
        preprocessing={"pipeline": 2},
    )
    moved_signature = db.derive_precompute_signature(
        second,
        grid,
        variant="DB-RAW",
        preprocessing={"pipeline": 2},
    )
    corrected_signature = db.derive_precompute_signature(
        second,
        grid,
        variant="DB-BC",
        preprocessing={"pipeline": 2},
    )
    assert raw_signature == moved_signature
    assert corrected_signature != raw_signature


def test_inventory_manager_changes_only_on_explicit_refresh(tmp_path: Path) -> None:
    root = tmp_path / "OWN"
    source = root / "phase.txt"
    _write_reference(source)
    manager = db.DatabaseInventoryManager({"OWN": root})

    initial = manager.snapshot()
    _write_reference(source, "100 9\n101 8\n102 7\n")
    unchanged_snapshot = manager.snapshot()
    refreshed = manager.refresh()

    assert unchanged_snapshot is initial
    assert refreshed.generation == initial.generation + 1
    assert refreshed.refresh_token != initial.refresh_token
    assert refreshed.signature != initial.signature


def test_pair_load_validates_and_shares_metadata_after_workspace_move(
    tmp_path: Path,
) -> None:
    current_root = tmp_path / "new-workspace" / "databases" / "OWN"
    first_source = current_root / "phase-a.txt"
    second_source = current_root / "nested" / "phase-b.txt"
    _write_reference(first_source)
    _write_reference(second_source)
    inventory = db.scan_database_inventory({"OWN": current_root})

    old_prefix = Path("/retired/workspace/databases/OWN")
    legacy_paths = [
        str(old_prefix / "phase-a.txt"),
        str(old_prefix / "nested" / "phase-b.txt"),
    ]
    raw_dir = tmp_path / "cache" / "raw-signature"
    baseline_dir = tmp_path / "cache" / "bc-signature"
    _write_legacy_pack(raw_dir, source_paths=legacy_paths, baseline=False)
    moved_again_paths = [
        path.replace("/retired/workspace/", "/another/retired-copy/")
        for path in legacy_paths
    ]
    _write_legacy_pack(
        baseline_dir,
        source_paths=moved_again_paths,
        baseline=True,
    )
    # A junk ANN file must remain untouched under the default exact-range loader.
    (raw_dir / db.DEFAULT_HNSW_FILE).write_bytes(b"not an index")

    pair = db.load_precompute_pair(
        raw_dir,
        baseline_dir,
        roots={"OWN": current_root},
        inventory=inventory,
        expected_parser_version=1,
        strict_sources=True,
    )

    assert pair.raw.metadata is pair.baseline_corrected.metadata
    assert pair.raw.hnsw_index is None
    assert pair.raw.matrix.shape == (2, 4)
    assert not pair.raw.matrix.flags.writeable
    assert pair.raw.resolve_source(0) == first_source.resolve()
    assert pair.raw.resolve_source(1) == second_source.resolve()
    assert pair.metadata.by_element["Si"] == (0, 1)

    legacy_view = pair.raw.matcher_view()
    assert legacy_view["X"] is pair.raw.matrix
    assert legacy_view["meta"][1]["path"] == second_source.resolve()
    assert legacy_view["ann"] is None


def test_pack_rejects_vector_byte_remainder(tmp_path: Path) -> None:
    root = tmp_path / "OWN"
    source = root / "phase.txt"
    _write_reference(source)
    pack_dir = tmp_path / "bad-byte-cache"
    _write_legacy_pack(
        pack_dir,
        source_paths=[str(source)],
        baseline=False,
        trailing_vector_bytes=b"x",
    )

    with pytest.raises(db.CacheValidationError, match="not divisible"):
        db.load_precompute_pack(pack_dir, roots={"OWN": root})


def test_pack_rejects_vector_metadata_row_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "OWN"
    source_a = root / "a.txt"
    source_b = root / "b.txt"
    _write_reference(source_a)
    _write_reference(source_b)
    pack_dir = tmp_path / "bad-row-cache"
    _write_legacy_pack(
        pack_dir,
        source_paths=[str(source_a), str(source_b)],
        baseline=False,
        metadata_row_delta=-1,
    )

    with pytest.raises(db.CacheValidationError, match="do not match"):
        db.load_precompute_pack(pack_dir, roots={"OWN": root})


def test_manifest_and_directory_promotion_are_atomic_and_non_replacing(
    tmp_path: Path,
) -> None:
    final = tmp_path / "precomputed" / "signature-a"
    grid = db.GridSpec(60.0, 63.0, 1.0, 4)
    manifest = db.CacheManifest.create(
        signature="signature-a",
        inventory_signature="inventory-a",
        variant="DB-RAW",
        grid=grid,
        metadata_rows=1,
        vector_bytes=grid.length * np.dtype("float32").itemsize,
    )

    with db.atomic_cache_directory(final) as staging:
        (staging / db.DEFAULT_VECTOR_FILE).write_bytes(
            np.arange(grid.length, dtype=np.float32).tobytes()
        )
        (staging / db.DEFAULT_METADATA_FILE).write_text("[{}]", encoding="utf-8")
        (staging / db.DEFAULT_GRID_FILE).write_text(
            json.dumps(grid.as_dict()), encoding="utf-8"
        )
        db.write_cache_manifest(staging, manifest)
        assert not final.exists()

    assert final.is_dir()
    assert db.read_cache_manifest(final, required=True) == manifest
    loaded = db.load_precompute_pack(
        final,
        roots={"OWN": tmp_path / "OWN"},
        expected_signature="signature-a",
        require_manifest=True,
    )
    assert loaded.matrix.shape == (1, grid.length)
    assert loaded.manifest == manifest

    staging = db.create_cache_staging_directory(final)
    db.write_cache_manifest(staging, manifest)
    try:
        with pytest.raises(FileExistsError, match="Refusing to replace"):
            db.promote_cache_directory(staging, final)
        assert final.is_dir()
    finally:
        # This is test-owned staging, not an application cleanup action.
        if staging.exists():
            (staging / db.DEFAULT_MANIFEST_FILE).unlink()
            staging.rmdir()

    failed_final = tmp_path / "precomputed" / "failed-signature"
    failed_staging: Path | None = None
    with pytest.raises(RuntimeError, match="simulated build failure"):
        with db.atomic_cache_directory(failed_final) as failed_staging:
            (failed_staging / "partial.bin").write_bytes(b"recoverable")
            raise RuntimeError("simulated build failure")
    assert failed_staging is not None
    assert failed_staging.is_dir()
    assert (failed_staging / "partial.bin").read_bytes() == b"recoverable"
    assert not failed_final.exists()


def test_cleanup_planning_is_read_only_and_protects_active_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "precomputed"
    active = cache_root / "active"
    incomplete = cache_root / "incomplete"
    explicit = cache_root / "explicit"
    for directory in (active, incomplete, explicit):
        directory.mkdir(parents=True)
        (directory / "payload.bin").write_bytes(b"1234")

    plan = db.plan_cache_cleanup(
        cache_root,
        active_signatures=["active"],
        candidate_signatures=["explicit"],
        include_incomplete=True,
    )

    candidate_names = {candidate.entry.path.name for candidate in plan.candidates}
    assert candidate_names == {"explicit", "incomplete"}
    assert {entry.path.name for entry in plan.keep} == {"active"}
    assert plan.reclaimable_bytes == 8
    assert all(directory.exists() for directory in (active, incomplete, explicit))


def test_cleanup_candidates_are_moved_to_recoverable_quarantine(tmp_path: Path) -> None:
    cache_root = tmp_path / "precomputed"
    keep = cache_root / "complete"
    candidate = cache_root / "incomplete"
    for directory in (keep, candidate):
        directory.mkdir(parents=True)
        (directory / "payload.bin").write_bytes(b"recoverable")

    plan = db.plan_cache_cleanup(
        cache_root,
        active_signatures=["complete"],
        include_incomplete=True,
    )
    moved = db.quarantine_cleanup_candidates(plan)

    assert len(moved) == 1
    assert moved[0].original_path == candidate
    assert not candidate.exists()
    assert (moved[0].quarantine_path / "payload.bin").read_bytes() == b"recoverable"
    assert keep.is_dir()


def test_rruff_provenance_normalizes_folder_filename_and_header(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "excellent_oriented"
        / "Quartz__R123456__Raman__514-5__90-000__ccw__Raman_Data_Processed__hash.txt"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "##NAMES=Quartz",
                "##RRUFFID=R123456",
                "##IDEAL CHEMISTRY=SiO_2_",
                "##MEASURED CHEMISTRY=SiO_2_",
                "##SOURCE=University collection",
                "##OWNER=RRUFF",
                "##STATUS=Confirmed by X-ray diffraction",
                "##URL=https://example.invalid/R123456",
                "##FILETYPE=Raman Processed",
                "##ORIENTATION=Laser parallel to c",
                "# Fluorescence background corrected by the contributor.",
                "100 1",
            ]
        ),
        encoding="utf-8",
    )

    provenance = db.parse_spectrum_provenance(path)

    assert provenance.database == "RRUFF"
    assert provenance.accession == "R123456"
    assert provenance.quality == "excellent"
    assert provenance.processing == "processed"
    assert provenance.determination == "experimental"
    assert provenance.orientation == "oriented"
    assert provenance.orientation_detail == "Laser parallel to c"
    assert provenance.excitation_wavelength_nm == pytest.approx(514.5)
    assert provenance.chemistry.ideal == "SiO_2_"
    assert provenance.corrections[0].applied is True
    assert "background corrected" in provenance.notes[0].casefold()


def test_rod_provenance_reads_scalar_and_multiline_correction_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1000123.rod"
    path.write_text(
        "\n".join(
            [
                "data_1000123",
                "_chemical_compound_source 'synthetic crystal'",
                "_chemical_formula_structural 'Al2 O3'",
                "_chemical_formula_sum 'Al2 O3'",
                "_raman_determination_method experimental",
                "_raman_measurement_device.excitation_laser_wavelength 532",
                "_raman_measurement_device.resolution 4",
                "_raman_measurement_device.direction_polarization Z(XX)Z",
                "_raman_measurement.background_subtraction yes",
                "_raman_measurement.background_subtraction_details",
                ";",
                " A dark spectrum was subtracted.",
                ";",
                "_raman_measurement.baseline_correction no",
                "_raman_measurement.baseline_correction_details",
                ";",
                " No baseline correction was applied.",
                ";",
                "_rod_data_source.file R123456.rod",
                "_rod_data_source.block 3500000",
                "_rod_database.code 1000123",
                "loop_",
                "_raman_spectrum.raman_shift",
                "_raman_spectrum.raw_intensity",
                "_raman_spectrum.intensity",
                "100 10 8",
            ]
        ),
        encoding="utf-8",
    )

    provenance = db.parse_rod_provenance(path)

    assert provenance.database == "ROD"
    assert provenance.accession == "1000123"
    assert provenance.source == "R123456.rod"
    assert provenance.source_accession == "3500000"
    assert provenance.processing == "processed"
    assert provenance.raw_intensity_available is True
    assert provenance.processed_intensity_available is True
    assert provenance.determination == "experimental"
    assert provenance.orientation == "oriented"
    assert provenance.excitation_wavelength_nm == pytest.approx(532.0)
    assert provenance.resolution_cm1 == pytest.approx(4.0)
    assert provenance.chemistry.ideal == "Al2 O3"
    assert provenance.chemistry.sample_source == "synthetic crystal"
    assert provenance.corrections[0].applied is True
    assert provenance.corrections[0].details == "A dark spectrum was subtracted."
    assert provenance.corrections[1].applied is False
    assert "not applied" in provenance.correction_history[1]

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import threading
import time

import numpy as np
import pytest

import raman_database as database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "raman_phase_id_pair_builder", PROJECT_ROOT / "RamanPhaseID_0p99beta.py"
)
APP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(APP)


def _write_rruff(path: Path, *, processed: bool) -> None:
    state = "Processed" if processed else "RAW"
    lines = [
        "##NAMES=Quartz",
        "##IDEAL CHEMISTRY=Si_O_2",
        f"##FILETYPE=Raman {state}",
    ]
    for shift in range(60, 101):
        baseline = 0.02 * (shift - 60)
        peak = 5.0 * np.exp(-0.5 * ((shift - 80.0) / 2.0) ** 2)
        lines.append(f"{shift} {1.0 + baseline + peak:.8f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_paired_builder_parses_once_and_promotes_valid_portable_packs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    _write_rruff(
        source_root
        / "Quartz__R000001__Raman__532______Raman_Data_RAW__raw.txt",
        processed=False,
    )
    _write_rruff(
        source_root
        / "Quartz__R000002__Raman__532______Raman_Data_Processed__processed.txt",
        processed=True,
    )
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()
    parse_calls = 0
    original_parser = APP.rc._parse_rruff

    def counted_parser(path: Path):
        nonlocal parse_calls
        parse_calls += 1
        return original_parser(path)

    monkeypatch.setattr(APP.rc, "_parse_rruff", counted_parser)
    progress_events: list[database.PairBuildProgress] = []
    progress_thread_ids: list[int] = []

    def capture_progress(event: database.PairBuildProgress) -> None:
        progress_events.append(event)
        progress_thread_ids.append(threading.get_ident())

    coordinator_thread_id = threading.get_ident()
    report = APP._build_precompute_pair_core(
        "raw-signature",
        "bc-signature",
        (source_root,),
        60,
        100,
        1,
        baseline_cfg=APP._fixed_db_baseline_cfg(),
        workers=2,
        progress_callback=capture_progress,
    )
    assert isinstance(report, database.PairBuildReport)
    assert report.reused_existing is False
    assert set(progress_thread_ids) == {coordinator_thread_id}
    processing_events = [
        event for event in progress_events if event.stage == "processing"
    ]
    assert [
        (event.completed_sources, event.total_sources)
        for event in processing_events
    ] == [(0, 2), (1, 2), (2, 2)]
    assert processing_events[-1].valid_rows == 2
    assert processing_events[-1].failed_rows == 0
    assert progress_events[-1] == database.PairBuildProgress(
        stage="complete",
        completed_sources=2,
        total_sources=2,
        valid_rows=2,
        failed_rows=0,
    )

    roots = (database.DatabaseRoot("RRUFF", source_root),)
    inventory = database.scan_database_inventory(roots)
    pair = database.load_precompute_pair(
        APP.PRECOMP_ROOT / "raw-signature",
        APP.PRECOMP_ROOT / "bc-signature",
        roots=roots,
        inventory=inventory,
        expected_raw_signature="raw-signature",
        expected_baseline_signature="bc-signature",
        expected_parser_version=APP.ELEMENT_PARSER_VERSION,
        pair_commit_root=APP.PRECOMP_ROOT,
        require_pair_commit=True,
    )

    assert parse_calls == 2
    assert pair.raw.matrix.shape == pair.baseline_corrected.matrix.shape == (2, 41)
    assert pair.raw.manifest is not None and pair.raw.manifest.complete
    assert pair.baseline_corrected.manifest is not None
    assert pair.raw.resolve_source(0).is_file()
    commit = database.read_pair_commit(
        APP.PRECOMP_ROOT,
        "raw-signature",
        "bc-signature",
        required=True,
    )
    assert commit is not None
    assert commit.valid_rows == 2
    assert commit.failed_rows == 0
    assert pair.raw.rows[0].support_runs == ((0, 40),)
    raw_view = pair.raw.matcher_view()["meta"]
    baseline_view = pair.baseline_corrected.matcher_view()["meta"]
    assert isinstance(raw_view, database.MatcherMetadataSequence)
    assert isinstance(raw_view[0], database.MatcherMetadataRow)
    assert raw_view[0]["provenance"] is baseline_view[0]["provenance"]
    assert raw_view[0]["provenance"]["database"] == "RRUFF"
    assert raw_view[0]["provenance"]["processing"] == "raw"
    processed_rows = [
        row
        for identity, row in zip(
            pair.baseline_corrected.metadata.entries,
            pair.baseline_corrected.rows,
            strict=True,
        )
        if "processed" in identity.orig_filename.casefold()
    ]
    assert len(processed_rows) == 1
    assert processed_rows[0].db_baseline is False


def test_paired_builder_persists_gap_aware_support_runs(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    path = source_root / "Gapped__R000010__Raman__532______Raman_Data_RAW__gap.txt"
    lines = [
        "##NAMES=Gapped",
        "##IDEAL CHEMISTRY=Si_O_2",
        "##FILETYPE=Raman RAW",
    ]
    for shift in (*range(60, 71), *range(90, 101)):
        lines.append(f"{shift} {1 + (shift == 65) + (shift == 95)}")
    path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()

    APP._build_precompute_pair_core(
        "raw-gap",
        "bc-gap",
        (source_root,),
        60,
        100,
        1,
        baseline_cfg=APP._fixed_db_baseline_cfg(),
        workers=1,
    )
    pair = database.load_precompute_pair(
        APP.PRECOMP_ROOT / "raw-gap",
        APP.PRECOMP_ROOT / "bc-gap",
        roots=(database.DatabaseRoot("RRUFF", source_root),),
        inventory=database.scan_database_inventory({"RRUFF": source_root}),
        expected_raw_signature="raw-gap",
        expected_baseline_signature="bc-gap",
        expected_parser_version=APP.ELEMENT_PARSER_VERSION,
        pair_commit_root=APP.PRECOMP_ROOT,
        require_pair_commit=True,
    )

    assert pair.raw.rows[0].support_runs == ((0, 10), (30, 40))
    assert pair.raw.rows[0].start_idx == 0
    assert pair.raw.rows[0].end_idx == 40


def test_paired_builder_refuses_all_failed_rows(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    for index in range(2):
        (source_root / f"Broken{index}.txt").write_text(
            "##NAMES=Broken\n##IDEAL CHEMISTRY=Si_O_2\nnot spectral data",
            encoding="utf-8",
        )
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()

    with pytest.raises(database.PairBuildError, match="no valid reference rows"):
        APP._build_precompute_pair_core(
            "raw-failed",
            "bc-failed",
            (source_root,),
            60,
            100,
            1,
            baseline_cfg=APP._fixed_db_baseline_cfg(),
            workers=1,
        )

    assert not (APP.PRECOMP_ROOT / "raw-failed").exists()
    assert not (APP.PRECOMP_ROOT / "bc-failed").exists()
    assert database.read_pair_commit(
        APP.PRECOMP_ROOT, "raw-failed", "bc-failed"
    ) is None
    assert not list(APP.PRECOMP_ROOT.glob(".*.staging-*"))
    quarantine = APP.PRECOMP_ROOT.parent / ".precomputed-quarantine"
    recovered_stages = [
        path
        for path in quarantine.iterdir()
        if path.is_dir() and ".staging-" in path.name
    ]
    assert len(recovered_stages) == 2
    assert all((path / database.DEFAULT_VECTOR_FILE).is_file() for path in recovered_stages)


def test_paired_builder_reports_sparse_parse_failures(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    _write_rruff(source_root / "Good.txt", processed=False)
    (source_root / "Broken.txt").write_text(
        "##NAMES=Broken\n##IDEAL CHEMISTRY=Si_O_2\nnot spectral data",
        encoding="utf-8",
    )
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()

    APP._build_precompute_pair_core(
        "raw-sparse",
        "bc-sparse",
        (source_root,),
        60,
        100,
        1,
        baseline_cfg=APP._fixed_db_baseline_cfg(),
        workers=1,
    )
    commit = database.read_pair_commit(
        APP.PRECOMP_ROOT, "raw-sparse", "bc-sparse", required=True
    )
    assert commit is not None
    assert commit.valid_rows == 1
    assert commit.failed_rows == 1
    assert dict(commit.failure_counts) == {"parse:ValueError": 1}


def test_paired_builder_refuses_dominant_systemic_preprocess_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    for index in range(10):
        _write_rruff(source_root / f"Phase{index}.txt", processed=False)
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()
    original_parser = APP.rc._parse_rruff
    original_prepare = APP.rprep.process_db_on_target_grid

    def marked_parser(path: Path):
        x, y = original_parser(path)
        if path.stem != "Phase0":
            y[0] = -999.0
        return x, y

    def mostly_broken_prepare(db_x, db_y, *args, **kwargs):
        if float(db_y[0]) == -999.0:
            raise RuntimeError("simulated shared preprocessing defect")
        return original_prepare(db_x, db_y, *args, **kwargs)

    monkeypatch.setattr(APP.rc, "_parse_rruff", marked_parser)
    monkeypatch.setattr(APP.rprep, "process_db_on_target_grid", mostly_broken_prepare)

    with pytest.raises(database.PairBuildError, match="systemic preprocessing failure"):
        APP._build_precompute_pair_core(
            "raw-systemic",
            "bc-systemic",
            (source_root,),
            60,
            100,
            1,
            baseline_cfg=APP._fixed_db_baseline_cfg(),
            workers=1,
        )
    assert database.read_pair_commit(
        APP.PRECOMP_ROOT, "raw-systemic", "bc-systemic"
    ) is None


def test_pair_build_lock_prevents_duplicate_concurrent_builds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    _write_rruff(source_root / "Concurrent.txt", processed=False)
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()
    original_parser = APP.rc._parse_rruff
    parse_calls = 0
    calls_lock = threading.Lock()

    def slow_parser(path: Path):
        nonlocal parse_calls
        with calls_lock:
            parse_calls += 1
        time.sleep(0.03)
        return original_parser(path)

    monkeypatch.setattr(APP.rc, "_parse_rruff", slow_parser)

    def build() -> None:
        APP._build_precompute_pair_core(
            "raw-concurrent",
            "bc-concurrent",
            (source_root,),
            60,
            100,
            1,
            baseline_cfg=APP._fixed_db_baseline_cfg(),
            workers=1,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(build) for _ in range(2)]
        for future in futures:
            future.result()

    assert parse_calls == 1
    assert database.committed_pair_available(
        APP.PRECOMP_ROOT, "raw-concurrent", "bc-concurrent"
    )


def test_partial_pair_is_quarantined_then_rebuilt_as_one_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    _write_rruff(source_root / "Recovery.txt", processed=False)
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()
    partial_raw = APP.PRECOMP_ROOT / "raw-recovery"
    partial_raw.mkdir(parents=True)
    (partial_raw / "partial.bin").write_bytes(b"recover me")

    APP._build_precompute_pair_core(
        "raw-recovery",
        "bc-recovery",
        (source_root,),
        60,
        100,
        1,
        baseline_cfg=APP._fixed_db_baseline_cfg(),
        workers=1,
    )

    assert database.committed_pair_available(
        APP.PRECOMP_ROOT, "raw-recovery", "bc-recovery"
    )
    quarantine = APP.PRECOMP_ROOT.parent / ".precomputed-quarantine"
    recovered = list(quarantine.rglob("partial.bin"))
    assert len(recovered) == 1
    assert recovered[0].read_bytes() == b"recover me"


def test_dominant_parser_failure_is_treated_as_systemic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    for index in range(10):
        _write_rruff(source_root / f"Phase{index}.txt", processed=False)
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()
    original_parser = APP.rc._parse_rruff

    def mostly_broken_parser(path: Path):
        if path.stem != "Phase0":
            raise ValueError("simulated parser regression")
        return original_parser(path)

    monkeypatch.setattr(APP.rc, "_parse_rruff", mostly_broken_parser)
    with pytest.raises(database.PairBuildError, match="systemic reference-build failure"):
        APP._build_precompute_pair_core(
            "raw-parser-systemic",
            "bc-parser-systemic",
            (source_root,),
            60,
            100,
            1,
            baseline_cfg=APP._fixed_db_baseline_cfg(),
            workers=1,
        )
    assert database.read_pair_commit(
        APP.PRECOMP_ROOT,
        "raw-parser-systemic",
        "bc-parser-systemic",
    ) is None


def test_valid_own_txt_keeps_own_database_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "OWN"
    source_root.mkdir()
    _write_rruff(source_root / "LaboratoryQuartz.txt", processed=False)
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()

    APP._build_precompute_pair_core(
        "raw-own",
        "bc-own",
        (source_root,),
        60,
        100,
        1,
        baseline_cfg=APP._fixed_db_baseline_cfg(),
        workers=1,
    )
    roots = (database.DatabaseRoot("OWN", source_root),)
    pair = database.load_precompute_pair(
        APP.PRECOMP_ROOT / "raw-own",
        APP.PRECOMP_ROOT / "bc-own",
        roots=roots,
        inventory=database.scan_database_inventory(roots),
        expected_raw_signature="raw-own",
        expected_baseline_signature="bc-own",
        expected_parser_version=APP.ELEMENT_PARSER_VERSION,
        pair_commit_root=APP.PRECOMP_ROOT,
        require_pair_commit=True,
    )
    row = pair.raw.matcher_view()["meta"][0]
    assert pair.raw.metadata.entries[0].source.root_alias == "OWN"
    assert row["source_root"] == "OWN"
    assert row["provenance"]["database"] == "OWN"


def test_manifest_write_failure_quarantines_both_staging_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "RRUFF"
    source_root.mkdir()
    _write_rruff(source_root / "Quartz.txt", processed=False)
    monkeypatch.setattr(APP, "PRECOMP_ROOT", tmp_path / "precomputed")
    APP.rc.load_reference_folders.cache_clear()

    def fail_manifest(*_args, **_kwargs):
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(database, "write_cache_manifest", fail_manifest)
    with pytest.raises(OSError, match="manifest write failure"):
        APP._build_precompute_pair_core(
            "raw-write-failed",
            "bc-write-failed",
            (source_root,),
            60,
            100,
            1,
            baseline_cfg=APP._fixed_db_baseline_cfg(),
            workers=1,
        )

    assert not (APP.PRECOMP_ROOT / "raw-write-failed").exists()
    assert not (APP.PRECOMP_ROOT / "bc-write-failed").exists()
    assert not list(APP.PRECOMP_ROOT.glob(".*.staging-*"))
    quarantine = APP.PRECOMP_ROOT.parent / ".precomputed-quarantine"
    staged = [path for path in quarantine.iterdir() if ".staging-" in path.name]
    assert len(staged) == 2

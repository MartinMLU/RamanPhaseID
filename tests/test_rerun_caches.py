from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import RamanPhaseID_0p99beta as app
import raman_database as database


def test_reference_eligibility_cache_skips_unchanged_metadata_scan(monkeypatch) -> None:
    rows = [
        {
            "name": "Quartz",
            "elements": ("O", "Si"),
            "has_formula": True,
            "provenance": {"processing": "raw"},
        },
        {
            "name": "Processed quartz",
            "elements": ("O", "Si"),
            "has_formula": True,
            "provenance": {"processing": "processed"},
        },
    ]
    request = database.ReferenceEligibilityRequest(
        raw_signature="rerun-cache-test-raw",
        baseline_signature="rerun-cache-test-baseline",
        library_variant="baseline_corrected",
        filtering_policy_version=app.REFERENCE_FILTER_POLICY_VERSION,
    )
    calls = {"processed": 0}

    def is_processed(row) -> bool:
        calls["processed"] += 1
        return row["provenance"]["processing"] == "processed"

    monkeypatch.setattr(app, "_reference_is_already_processed", is_processed)
    app._cached_reference_eligibility.clear()
    try:
        first = app._cached_reference_eligibility(request, rows)
        second = app._cached_reference_eligibility(request, rows)
    finally:
        app._cached_reference_eligibility.clear()

    np.testing.assert_array_equal(first.row_ids, [0])
    np.testing.assert_array_equal(second.row_ids, [0])
    assert calls == {"processed": 2}


def test_figure_bundle_cache_does_not_call_factory_on_unchanged_render() -> None:
    render_signature = hashlib.sha256(
        b"ramanphaseid-rerun-render-cache-test"
    ).hexdigest()
    factory_calls = 0

    def render_once():
        nonlocal factory_calls
        factory_calls += 1
        figure, axis = plt.subplots()
        axis.plot([100.0, 101.0], [0.0, 1.0])
        return figure

    def must_not_render_again():
        raise AssertionError("unchanged semantic signature should be a cache hit")

    app._cached_figure_render_bundle.clear()
    try:
        first = app._cached_figure_render_bundle(render_signature, render_once)
        second = app._cached_figure_render_bundle(
            render_signature,
            must_not_render_again,
        )
    finally:
        app._cached_figure_render_bundle.clear()

    assert factory_calls == 1
    assert first == second
    assert first.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"<svg" in first.svg[:1024].lower()

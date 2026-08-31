from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from streamlit.testing.v1 import AppTest

import raman_workflow as workflow
import raman_plotting as plotting


APP_SOURCE = r'''
from pathlib import Path
from types import SimpleNamespace

import hashlib
import json
import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import RamanPhaseID_0p99beta as app

# Exercise workflow/session behavior without invoking Streamlit's native
# Matplotlib serialization, which is unrelated to these state assertions and
# can make a large in-process AppTest suite abort during figure GC.
def capture_pyplot(figure, *args, **kwargs):
    if figure.axes and any(
        "matching query" in line.get_label()
        for line in figure.axes[0].lines
    ):
        st.session_state["qa_residual_audit_y"] = np.asarray(
            figure.axes[0].lines[0].get_ydata(), dtype=float
        )
        st.session_state["qa_residual_audit_colors"] = tuple(
            line.get_color() for line in figure.axes[0].lines
        )
        st.session_state["qa_residual_audit_widths"] = tuple(
            line.get_linewidth() for line in figure.axes[0].lines
        )


app.st.pyplot = capture_pyplot
app.st.image = lambda *args, **kwargs: None


def capture_figure_bundle(_signature, figure_factory):
    figure = figure_factory()
    try:
        if figure.axes:
            plotted_lines = tuple(figure.axes[0].lines)
            plotted_labels = tuple(line.get_label() for line in plotted_lines)
            plotted_colors = tuple(line.get_color() for line in plotted_lines)
            plotted_widths = tuple(line.get_linewidth() for line in plotted_lines)
            if "measurement (raw)" in plotted_labels:
                st.session_state["qa_raw_plot_colors"] = plotted_colors
                st.session_state["qa_raw_plot_widths"] = plotted_widths
                st.session_state["qa_raw_plot_ylabel"] = figure.axes[0].get_ylabel()
            if any(label.startswith("difference curve; x") for label in plotted_labels):
                st.session_state["qa_smoothing_plot_colors"] = plotted_colors
                st.session_state["qa_smoothing_plot_widths"] = plotted_widths
                st.session_state["qa_smoothing_plot_labels"] = plotted_labels
                st.session_state["qa_smoothing_plot_ylabel"] = (
                    figure.axes[0].get_ylabel()
                )
                st.session_state["qa_smoothing_plot_y"] = tuple(
                    np.asarray(line.get_ydata(), dtype=float)
                    for line in plotted_lines
                )
            lines_by_label = {
                line.get_label(): line
                for line in plotted_lines
            }
            baseline_label = next(
                (
                    label
                    for label in lines_by_label
                    if label.startswith("baseline · ")
                ),
                None,
            )
            if (
                baseline_label is not None
                and "corrected = input − baseline" in lines_by_label
            ):
                input_label = next(
                    label
                    for label in lines_by_label
                    if label not in {
                        baseline_label,
                        "corrected = input − baseline",
                    }
                )
                baseline_lines = (
                    lines_by_label[input_label],
                    lines_by_label[baseline_label],
                    lines_by_label["corrected = input − baseline"],
                )
                st.session_state["qa_baseline_preview_colors"] = tuple(
                    line.get_color() for line in baseline_lines
                )
                st.session_state["qa_baseline_preview_widths"] = tuple(
                    line.get_linewidth() for line in baseline_lines
                )
                st.session_state["qa_baseline_preview_ylabel"] = (
                    figure.axes[0].get_ylabel()
                )
                legend = figure.axes[0].get_legend()
                legend_handles = getattr(
                    legend,
                    "legend_handles",
                    getattr(legend, "legendHandles", ()),
                )
                st.session_state["qa_baseline_preview_legend_colors"] = tuple(
                    handle.get_color() for handle in legend_handles
                )
        if figure.axes and any(
            "score-aligned" in line.get_label()
            for line in figure.axes[0].lines
        ):
            st.session_state["qa_overlay_plot_colors"] = tuple(
                line.get_color() for line in figure.axes[0].lines
            )
            st.session_state["qa_overlay_plot_widths"] = tuple(
                line.get_linewidth() for line in figure.axes[0].lines
            )
            st.session_state["qa_overlay_measurement_y"] = np.asarray(
                figure.axes[0].lines[0].get_ydata(), dtype=float
            )
            st.session_state["qa_overlay_provided_y"] = np.asarray(
                figure.axes[0].lines[1].get_ydata(), dtype=float
            )
            st.session_state["qa_overlay_aligned_y"] = np.asarray(
                figure.axes[0].lines[2].get_ydata(), dtype=float
            )
            st.session_state["qa_overlay_ylim"] = tuple(
                float(value) for value in figure.axes[0].get_ylim()
            )
        return SimpleNamespace(
            png=b"workflow-test-png",
            svg=b"workflow-test-svg",
        )
    finally:
        plt.close(figure)


app._cached_figure_render_bundle = capture_figure_bundle
_original_cached_processed_spectrum = app._cached_processed_spectrum


def _tracked_cached_processed_spectrum(
    measurement_sha256,
    axis_cm1,
    intensity,
    apply_baseline,
    baseline_payload_json,
    smoothing_payload_json,
):
    artifact = _original_cached_processed_spectrum(
        measurement_sha256,
        axis_cm1,
        intensity,
        apply_baseline,
        baseline_payload_json,
        smoothing_payload_json,
    )
    if json.loads(smoothing_payload_json).get("method") == "none":
        baseline_values = np.concatenate(
            [segment.baseline for segment in artifact.segments]
        )
        st.session_state["qa_baseline_preview_payload"] = baseline_payload_json
        st.session_state["qa_baseline_preview_sha256"] = hashlib.sha256(
            np.asarray(baseline_values, dtype=np.float64).tobytes()
        ).hexdigest()
    return artifact


app._cached_processed_spectrum = _tracked_cached_processed_spectrum

RETURN_EMPTY = __RETURN_EMPTY__
EVIDENCE_STATUS = __EVIDENCE_STATUS__
x_measurement = np.arange(100.0, 501.0, 2.0)
y_measurement = (
    0.05
    + np.exp(-0.5 * ((x_measurement - 220.0) / 9.0) ** 2)
    + 0.65 * np.exp(-0.5 * ((x_measurement - 410.0) / 12.0) ** 2)
)
MEASUREMENT_BYTES = (
    "\n".join(
        f"{x_value:.1f}\t{y_value:.8f}"
        for x_value, y_value in zip(x_measurement, y_measurement)
    )
    + "\n"
).encode()


class DummyUpload:
    name = "workflow-state-test.txt"

    def getvalue(self):
        return MEASUREMENT_BYTES


app.st.file_uploader = lambda *args, **kwargs: DummyUpload()
st.session_state["measurement_file_0"] = True
app.PRECOMP_ROOT = Path("/tmp/ramanphaseid-workflow-test-cache-does-not-exist")
app._inventory_snapshot = lambda *args, **kwargs: SimpleNamespace(
    signature="workflow-test-inventory",
    files=(),
    refresh_token="workflow-test-inventory:g0",
)
app._runtime_export_metadata = lambda stamps: (
    "workflow-test-commit",
    {"numpy": np.__version__},
    {name: "workflow-test-hash" for name, _size, _mtime in stamps},
)

grid = np.arange(60.0, 1901.0, dtype=np.float32)
trace = (
    np.exp(-0.5 * ((grid - 220.0) / 9.0) ** 2)
    + 0.65 * np.exp(-0.5 * ((grid - 410.0) / 12.0) ** 2)
).astype(np.float32)
trace /= max(float(np.max(trace)), 1e-9)
support_runs = ((40, 440),)
provenance = {
    "database": "QA",
    "source": "in-memory test reference",
    "status": "experimental",
    "quality": "excellent",
    "quality_folder": "excellent_unoriented",
    "processing": "raw",
    "determination": "experimental",
    "orientation": "unoriented",
    "orientation_detail": "",
    "excitation_wavelength_nm": 532.0,
    "resolution_cm1": 2.0,
    "measured_chemistry": "",
    "correction_history": [],
    "accession": "QA-QUARTZ-1",
}
metadata = {
    "name": "Quartz",
    "formula": "SiO2",
    "elements": ("Si", "O"),
    "has_formula": True,
    "flag": "s",
    "path": "/tmp/OWN/quartz.txt",
    "filename": "quartz.txt",
    "orig_filename": "quartz.txt",
    "start_idx": 40,
    "end_idx": 440,
    "support_runs": support_runs,
    "l2": float(np.linalg.norm(trace)),
    "db_baseline": False,
    "provenance": provenance,
}
pack = {
    "grid": grid,
    "X": trace.reshape(1, -1),
    "meta": [metadata],
    "grid_info": {"min": 60, "max": 1900, "step": 1},
}
baseline_metadata = {
    **metadata,
    "db_baseline": True,
}
baseline_pack = {
    **pack,
    "X": (0.5 * trace).reshape(1, -1),
    "meta": [baseline_metadata],
}
def fake_ensure_precompute_pair(**kwargs):
    st.session_state["qa_precompute_grid"] = (
        int(kwargs["grid_min"]),
        int(kwargs["grid_max"]),
        int(kwargs["grid_step"]),
    )
    return pack, baseline_pack


app._ensure_precompute_pair = fake_ensure_precompute_pair


def fake_match(
    query,
    query_mask,
    range_low,
    range_high,
    pack_raw,
    pack_bcb,
    allowed_raw,
    allowed_bcb,
    measurement_mode,
    *,
    top_n=60,
    excluded_phase_keys=(),
    matching_parameters=None,
):
    st.session_state["qa_match_calls"] = int(
        st.session_state.get("qa_match_calls", 0)
    ) + 1
    if RETURN_EMPTY:
        return []
    if excluded_phase_keys:
        st.session_state["qa_residual_allowed_raw"] = tuple(
            int(value) for value in allowed_raw
        )
        st.session_state["qa_residual_allowed_bcb"] = tuple(
            int(value) for value in allowed_bcb
        )
        st.session_state["qa_residual_minimum_pcs"] = float(
            matching_parameters.minimum_candidate_peak_consistency
        )
        st.session_state["qa_residual_remove_query_offset"] = bool(
            matching_parameters.remove_query_local_offset
        )
    common = int(np.count_nonzero(query_mask))
    residual_search = bool(excluded_phase_keys)
    return [
        {
            "name": "Second phase" if residual_search else "Quartz",
            "formula": "SiO2",
            "flag": "s",
            "filename": "quartz.txt",
            "orig_filename": "quartz.txt",
            "path": "/tmp/OWN/quartz.txt",
            "db_idx": 0,
            "db_baseline": residual_search,
            "db_variant": "DB-BC" if residual_search else "DB-RAW",
            "meas_variant": measurement_mode,
            "shift": 0,
            "shift_cm1": 0.0,
            "start_idx": 40,
            "end_idx": 440,
            "support_runs": support_runs,
            "shift_boundary_hit": False,
            "grid_boundary_clipped": False,
            "similarity": 0.91,
            "shape_similarity": 0.91,
            "gradient_similarity": 0.86,
            "pcs": 0.88,
            "rank_score": 0.90,
            "rank_components": {"shape": 0.91, "gradient": 0.86},
            "common_point_count": common,
            "coverage_fraction": 1.0,
            "reference_overlap_fraction": 1.0,
            "requested_point_count": common,
            "reference_support_point_count": common,
            "phase_rank": 1,
            "phase_score": 0.90,
            "phase_independent_reference_count": 2,
            "phase_reference_variant_count": 2,
            "evidence_status": EVIDENCE_STATUS,
            "evidence_best_phase": "Quartz",
            "evidence_best_score": 0.90,
            "evidence_runner_up_score": 0.70,
            "evidence_score_margin": 0.20,
            "evidence_reasons": ["uncalibrated_evidence_guardrails_passed"],
            "database_source": "QA",
            "source": "in-memory test reference",
            "accession": "QA-QUARTZ-1",
            "quality": "excellent",
            "quality_folder": "excellent_unoriented",
            "reference_processing": "raw",
            "determination": "experimental",
            "orientation": "unoriented",
            "orientation_detail": "",
            "excitation_wavelength_nm": 532.0,
            "resolution_cm1": 2.0,
            "measured_chemistry": "",
            "correction_history": [],
        }
    ]


app._compute_matches_from_query_vector = fake_match
app._run_streamlit()
'''


def _app(
    *,
    empty: bool = False,
    evidence_status: str = "supported_candidate",
) -> AppTest:
    return AppTest.from_string(
        APP_SOURCE.replace("__RETURN_EMPTY__", repr(bool(empty))).replace(
            "__EVIDENCE_STATUS__",
            repr(str(evidence_status)),
        ),
        default_timeout=60,
    ).run()


def _session_value(app: AppTest, key: str, default=None):
    try:
        return app.session_state[key]
    except KeyError:
        return default


def _click_key(app: AppTest, key: str) -> AppTest:
    button = next(button for button in app.button if button.key == key)
    return button.click().run(timeout=60)


def _approve_through_smoothing(app: AppTest) -> AppTest:
    for key in (
        "approve_white_ref_btn",
        "approve_baseline_btn",
        "approve_smoothing_btn",
    ):
        app = _click_key(app, key)
        assert not list(app.exception)
    return app


def _update_matching(app: AppTest) -> AppTest:
    update = next(
        button
        for button in app.button
        if button.key == "update_database_matching_btn"
    )
    return update.click().run(timeout=60)


def test_intensity_toggle_updates_first_three_plot_axis_titles() -> None:
    app = _app()
    toggle = next(
        button
        for button in app.sidebar.button
        if button.key == "toggle_preview_intensity_numbers_btn"
    )
    assert toggle.label == "Hide intensity numbers"
    assert _session_value(app, "qa_raw_plot_ylabel") == (
        plotting.RAMAN_INTENSITY_LABEL
    )

    app = toggle.click().run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "show_preview_intensity_numbers") is False
    assert _session_value(app, "qa_raw_plot_ylabel") == (
        plotting.RAMAN_INTENSITY_LABEL_NO_UNITS
    )
    toggle = next(
        button
        for button in app.sidebar.button
        if button.key == "toggle_preview_intensity_numbers_btn"
    )
    assert toggle.label == "Show intensity numbers"

    app = _click_key(app, "approve_white_ref_btn")
    assert _session_value(app, "qa_baseline_preview_ylabel") == (
        plotting.RAMAN_INTENSITY_LABEL_NO_UNITS
    )
    app = _click_key(app, "approve_baseline_btn")
    assert _session_value(app, "qa_smoothing_plot_ylabel") == (
        plotting.RAMAN_INTENSITY_LABEL_NO_UNITS
    )

    toggle = next(
        button
        for button in app.sidebar.button
        if button.key == "toggle_preview_intensity_numbers_btn"
    )
    assert toggle.label == "Show intensity numbers"
    app = toggle.click().run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "show_preview_intensity_numbers") is True
    assert _session_value(app, "qa_raw_plot_ylabel") == (
        plotting.RAMAN_INTENSITY_LABEL
    )
    assert _session_value(app, "qa_baseline_preview_ylabel") == (
        plotting.RAMAN_INTENSITY_LABEL
    )
    assert _session_value(app, "qa_smoothing_plot_ylabel") == (
        plotting.RAMAN_INTENSITY_LABEL
    )
    toggle = next(
        button
        for button in app.sidebar.button
        if button.key == "toggle_preview_intensity_numbers_btn"
    )
    assert toggle.label == "Hide intensity numbers"


def test_appearance_group_controls_global_plot_color_scheme() -> None:
    app = _app()
    assert any(
        expander.label == "Appearance"
        for expander in app.sidebar.expander
    )
    for button_key in (
        "toggle_app_theme_btn",
        "toggle_plot_theme_btn",
        "toggle_preview_intensity_numbers_btn",
    ):
        assert any(button.key == button_key for button in app.sidebar.button)

    scheme_selector = next(
        selectbox
        for selectbox in app.sidebar.selectbox
        if selectbox.key == "plot_color_scheme"
    )
    assert scheme_selector.label == "Plot line colors"
    assert scheme_selector.value == "standard"
    assert _session_value(app, "qa_raw_plot_colors") == (
        plotting.baseline_preview_colors("standard", "dark").input_signal,
    )
    assert _session_value(app, "qa_raw_plot_widths") == (
        plotting.PLOT_LINEWIDTH,
    )

    app = scheme_selector.select("colorblind").run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "plot_color_scheme") == "colorblind"
    assert _session_value(app, "qa_raw_plot_colors") == (
        plotting.plot_curve_colors("colorblind", "dark")[0],
    )
    app = _click_key(app, "approve_white_ref_btn")
    expected = plotting.baseline_preview_colors("colorblind", "dark").as_tuple()
    assert _session_value(app, "qa_baseline_preview_colors") == expected
    assert _session_value(app, "qa_baseline_preview_legend_colors") == expected
    assert _session_value(app, "qa_baseline_preview_widths") == (
        plotting.PLOT_LINEWIDTH,
    ) * 3

    app = _click_key(app, "approve_baseline_btn")
    expected_cycle = plotting.plot_curve_colors("colorblind", "dark")
    assert _session_value(app, "qa_smoothing_plot_colors") == expected_cycle[:3]
    assert _session_value(app, "qa_smoothing_plot_widths") == (
        plotting.PLOT_LINEWIDTH,
    ) * 3
    smoothing_labels = _session_value(app, "qa_smoothing_plot_labels")
    assert smoothing_labels == (
        "baseline-corrected measurement",
        "smoothed · Savitzky-Golay (window = 5, poly = 3)",
        "difference curve; x2",
    )
    smoothing_y = tuple(
        np.asarray(values, dtype=float)
        for values in _session_value(app, "qa_smoothing_plot_y")
    )
    assert float(np.nanmax(smoothing_y[2])) < min(
        float(np.nanmin(smoothing_y[0])),
        float(np.nanmin(smoothing_y[1])),
    )
    difference_span_2x = float(
        np.nanmax(smoothing_y[2]) - np.nanmin(smoothing_y[2])
    )
    smoothing_draft_before_display_change = _session_value(
        app,
        "workflow_state",
    ).smoothing_draft
    magnification_slider = next(
        slider
        for slider in app.sidebar.slider
        if slider.key == "processing_difference_magnification"
    )
    assert magnification_slider.value == 2.0
    app = magnification_slider.set_value(4.0).run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls", 0) == 0
    assert (
        _session_value(app, "workflow_state").smoothing_draft
        == smoothing_draft_before_display_change
    )
    smoothing_labels = _session_value(app, "qa_smoothing_plot_labels")
    assert smoothing_labels[2] == "difference curve; x4"
    magnified_y = np.asarray(
        _session_value(app, "qa_smoothing_plot_y")[2],
        dtype=float,
    )
    difference_span_4x = float(np.nanmax(magnified_y) - np.nanmin(magnified_y))
    np.testing.assert_allclose(
        difference_span_4x,
        2.0 * difference_span_2x,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    app = _click_key(app, "approve_smoothing_btn")
    assert _session_value(app, "qa_match_calls") == 1
    assert _session_value(app, "qa_overlay_plot_colors") == expected_cycle[:3]
    assert _session_value(app, "qa_overlay_plot_widths") == (
        plotting.PLOT_LINEWIDTH,
    ) * 3
    completed_result_identity = _session_value(app, "workflow_state").result_identity
    magnification_slider = next(
        slider
        for slider in app.sidebar.slider
        if slider.key == "processing_difference_magnification"
    )
    app = magnification_slider.set_value(3.0).run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1
    assert _session_value(app, "workflow_state").has_current_result
    assert (
        _session_value(app, "workflow_state").result_identity
        == completed_result_identity
    )

    scheme_selector = next(
        selectbox
        for selectbox in app.sidebar.selectbox
        if selectbox.key == "plot_color_scheme"
    )
    app = scheme_selector.select("grayscale").run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1
    expected = plotting.baseline_preview_colors("grayscale", "dark").as_tuple()
    assert _session_value(app, "qa_baseline_preview_colors") == expected
    assert _session_value(app, "qa_baseline_preview_legend_colors") == expected
    expected_cycle = plotting.plot_curve_colors("grayscale", "dark")
    assert _session_value(app, "qa_raw_plot_colors") == expected_cycle[:1]
    assert _session_value(app, "qa_smoothing_plot_colors") == expected_cycle[:3]
    assert _session_value(app, "qa_overlay_plot_colors") == expected_cycle[:3]

    app = _click_key(app, "run_residual_phase_search_btn")
    assert not list(app.exception)
    assert _session_value(app, "qa_residual_audit_colors") == expected_cycle[:2]
    assert _session_value(app, "qa_residual_audit_widths") == (
        plotting.PLOT_LINEWIDTH,
    ) * 2
    assert _session_value(app, "qa_overlay_plot_colors") == expected_cycle[:3]


def test_matching_starts_after_smoothing_and_matching_edits_require_update() -> None:
    app = _app()
    app = _click_key(app, "approve_white_ref_btn")
    assert _session_value(app, "qa_match_calls", 0) == 0
    app = _click_key(app, "approve_baseline_btn")
    assert _session_value(app, "qa_match_calls", 0) == 0
    app = _click_key(app, "approve_smoothing_btn")
    assert _session_value(app, "qa_match_calls", 0) == 1
    assert any(
        "materially negative" in str(caption.value)
        for caption in app.caption
    )
    assert not any(
        "More than 10% of baseline-corrected points are negative" in str(
            warning.value
        )
        for warning in app.warning
    )
    assert any(
        expander.label == "Matching parameters and controls"
        for expander in app.sidebar.expander
    )
    assert any(
        slider.label == "Applied matching range (cm⁻¹)"
        for slider in app.sidebar.slider
    )
    assert not any(
        selectbox.label == "Reference-library scope"
        for selectbox in app.sidebar.selectbox
    )
    assert not any(
        checkbox.key == "match_ultra_draft"
        for checkbox in app.checkbox
    )
    assert not any(
        button.key == "clear_measurement_btn"
        for button in app.button
    )
    assert not any(
        expander.label == "Input quality and calibration"
        for expander in app.expander
    )
    assert any(
        "201 finite points" in str(caption.value)
        and "100.00–500.00 cm⁻¹" in str(caption.value)
        and "median spacing 2 cm⁻¹" in str(caption.value)
        for caption in app.main.caption
    )
    assert any(
        button.label == "Update database matching"
        for button in app.sidebar.button
    )
    assert not any(
        slider.label == "Applied matching range (cm⁻¹)"
        for slider in app.main.slider
    )
    assert not list(app.exception)
    assert not list(app.error)
    state = _session_value(app, "workflow_state")
    snapshot = _session_value(app, "primary_result_snapshot")
    assert isinstance(snapshot, workflow.PrimaryResultSnapshot)
    assert state.has_current_result
    assert snapshot.identity == state.result_identity
    assert snapshot.matching_approval.config.policy_signature
    assert state.next_required_stage == "complete"
    assert _session_value(app, "qa_precompute_grid") == (60, 4000, 1)
    assert any(
        "Reproducibility manifest" in element.label
        for element in app.get("download_button")
    )
    update_button = next(
        button
        for button in app.sidebar.button
        if button.key == "update_database_matching_btn"
    )
    assert update_button.proto.type == "secondary"
    caption_text = "\n".join(str(caption.value) for caption in app.main.caption)
    sidebar_caption_text = "\n".join(
        str(caption.value) for caption in app.sidebar.caption
    )
    info_text = "\n".join(str(info.value) for info in app.info)
    for removed_message in (
        "Active measurement:",
        "Ordinary zero crossings are expected",
        "Export (baseline-app compatible)",
        "Peak-preservation diagnostic:",
        "Selected matching parameters",
        "All configured spectra enter the reference pool",
        "Optional exploratory mixture aid:",
    ):
        assert removed_message not in caption_text
    assert "All spectra in OWN, ROD, and RRUFF are searched" not in (
        sidebar_caption_text
    )
    assert (
        "Highlighted when selected parameters differ from the current match."
        in sidebar_caption_text
    )
    assert "grey when they are already applied" not in sidebar_caption_text
    assert "Uncalibrated evidence state:" not in info_text

    include_elements = next(
        text_input
        for text_input in app.sidebar.text_input
        if text_input.key == "match_include_draft"
    )
    app = include_elements.set_value("Si").run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1
    assert _session_value(app, "workflow_state").result_is_stale
    update_button = next(
        button
        for button in app.sidebar.button
        if button.key == "update_database_matching_btn"
    )
    assert update_button.proto.type == "primary"
    assert not any(
        "Selected matching parameters" in str(caption.value)
        for caption in app.main.caption
    )

    app = _update_matching(app)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 2
    assert _session_value(app, "workflow_state").has_current_result
    update_button = next(
        button
        for button in app.sidebar.button
        if button.key == "update_database_matching_btn"
    )
    assert update_button.proto.type == "secondary"

    matching_range = next(
        slider
        for slider in app.sidebar.slider
        if slider.key == "matching_range_draft"
    )
    app = matching_range.set_value((100, 450)).run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 2
    assert _session_value(app, "workflow_state").result_is_stale
    assert any(
        "retained below as **stale**" in str(warning.value)
        for warning in app.warning
    )
    update_button = next(
        button
        for button in app.sidebar.button
        if button.key == "update_database_matching_btn"
    )
    assert update_button.proto.type == "primary"
    assert not any(
        "Selected matching parameters" in str(caption.value)
        for caption in app.main.caption
    )

    app = _update_matching(app)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 3
    assert _session_value(app, "workflow_state").has_current_result
    update_button = next(
        button
        for button in app.sidebar.button
        if button.key == "update_database_matching_btn"
    )
    assert update_button.proto.type == "secondary"

    reload_button = next(button for button in app.button if button.label == "Reload DB")
    app = reload_button.click().run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 3
    assert _session_value(app, "workflow_state").result_is_stale
    update_button = next(
        button
        for button in app.sidebar.button
        if button.key == "update_database_matching_btn"
    )
    assert update_button.proto.type == "primary"

    for legacy_key in (
        "white_ref_ready_sig",
        "baseline_ready_sig",
        "smoothing_ready_sig",
        "applied_matching_settings",
        "results_sig",
        "top_combined",
    ):
        assert _session_value(app, legacy_key) is None


def test_empty_successful_matching_is_recorded_and_not_repeated() -> None:
    app = _approve_through_smoothing(_app(empty=True))
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1
    state = _session_value(app, "workflow_state")
    snapshot = _session_value(app, "primary_result_snapshot")
    assert state.has_current_result
    assert state.next_required_stage == "complete"
    assert snapshot.is_empty
    assert any("No matches found" in str(info.value) for info in app.info)

    app = app.run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1
    assert _session_value(app, "workflow_state").has_current_result


def test_baseline_slider_recalculates_preview_after_completed_matching() -> None:
    app = _approve_through_smoothing(_app())
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1
    previous_payload = _session_value(app, "qa_baseline_preview_payload")
    previous_preview = _session_value(app, "qa_baseline_preview_sha256")

    lambda_slider = next(
        slider for slider in app.slider if slider.label == "λ (10^x)"
    )
    app = lambda_slider.set_value(0).run(timeout=60)

    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1
    assert _session_value(app, "qa_baseline_preview_payload") != previous_payload
    assert _session_value(app, "qa_baseline_preview_sha256") != previous_preview
    state = _session_value(app, "workflow_state")
    assert state.baseline_draft.lam_exp == 0
    assert state.baseline_approval.config.lam_exp == 5
    assert state.baseline_dirty
    assert state.result_is_stale
    assert state.next_required_stage == "baseline"
    assert any(
        "preview above has been recalculated" in str(info.value)
        for info in app.info
    )


def test_approving_changed_baseline_clears_derived_residual_results() -> None:
    app = _approve_through_smoothing(_app())
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 1

    app = _click_key(app, "run_residual_phase_search_btn")
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 2
    assert isinstance(
        _session_value(app, "residual_result_snapshot"),
        workflow.ResidualResultSnapshot,
    )
    assert _session_value(app, "residual_mode_active") is True
    assert _session_value(app, "qa_residual_allowed_raw") == ()
    assert _session_value(app, "qa_residual_allowed_bcb") == (0,)
    assert _session_value(app, "qa_residual_minimum_pcs") == 0.15
    assert _session_value(app, "qa_residual_remove_query_offset") is False
    audit_y = np.asarray(_session_value(app, "qa_residual_audit_y"), dtype=float)
    overlay_y = np.asarray(
        _session_value(app, "qa_overlay_measurement_y"), dtype=float
    )
    np.testing.assert_allclose(audit_y, overlay_y, equal_nan=True)
    provided_y = np.asarray(_session_value(app, "qa_overlay_provided_y"), dtype=float)
    aligned_y = np.asarray(_session_value(app, "qa_overlay_aligned_y"), dtype=float)
    plotted_reference = np.isfinite(provided_y) & np.isfinite(aligned_y)
    np.testing.assert_allclose(
        aligned_y[plotted_reference],
        0.5 * provided_y[plotted_reference],
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    overlay_ylim = _session_value(app, "qa_overlay_ylim")
    assert float(overlay_ylim[0]) <= float(np.nanmin(overlay_y))

    lambda_slider = next(
        slider for slider in app.slider if slider.label == "λ (10^x)"
    )
    app = lambda_slider.set_value(0).run(timeout=60)
    assert not list(app.exception)
    assert _session_value(app, "workflow_state").baseline_dirty
    assert _session_value(app, "residual_result_snapshot") is not None

    app = _click_key(app, "approve_baseline_btn")
    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 2
    assert _session_value(app, "residual_result_snapshot") is None
    assert _session_value(app, "top_combined_residual") is None
    assert _session_value(app, "residual_mode_active") is None
    assert _session_value(app, "residual_search_info") is None
    assert _session_value(app, "residual_parent_identity") is None
    state = _session_value(app, "workflow_state")
    assert state.baseline_approval.config.lam_exp == 0
    assert state.next_required_stage == "smoothing"


def test_unsupported_residual_ranking_remains_inspectable_as_hypothesis() -> None:
    app = _approve_through_smoothing(
        _app(evidence_status="unknown_or_out_of_library")
    )
    assert not list(app.exception)

    app = _click_key(app, "run_residual_phase_search_btn")

    assert not list(app.exception)
    assert _session_value(app, "qa_match_calls") == 2
    snapshot = _session_value(app, "residual_result_snapshot")
    assert snapshot is not None
    assert _session_value(app, "residual_mode_active") is True
    assert snapshot.diagnostics_mapping()["evidence_gate_cleared"] is False
    assert any(
        "exploratory only; evidence guardrails not cleared" in str(warning.value)
        for warning in app.warning
    )

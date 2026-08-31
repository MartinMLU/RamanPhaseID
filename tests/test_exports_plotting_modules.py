from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import AutoLocator, MultipleLocator
import numpy as np

import raman_exports as exports
import raman_plotting as plotting


def test_plot_color_schemes_are_distinct_and_theme_aware() -> None:
    standard = plotting.baseline_preview_colors("standard", "dark")
    colorblind_light = plotting.baseline_preview_colors("colorblind", "light")
    colorblind_dark = plotting.baseline_preview_colors("colorblind", "dark")
    grayscale_light = plotting.baseline_preview_colors("grayscale", "light")
    grayscale_dark = plotting.baseline_preview_colors("grayscale", "dark")

    assert standard.as_tuple() == ("tab:blue", "tab:orange", "tab:green")
    for colors in (
        standard,
        colorblind_light,
        colorblind_dark,
        grayscale_light,
        grayscale_dark,
    ):
        assert len(set(colors.as_tuple())) == 3
    assert colorblind_light != colorblind_dark
    for colors in (grayscale_light, grayscale_dark):
        for color in colors.as_tuple():
            red, green, blue = to_rgb(color)
            assert red == green == blue
    for theme in ("light", "dark"):
        for scheme in ("standard", "colorblind", "grayscale"):
            assert plotting.plot_curve_colors(scheme, theme)[:3] == (
                plotting.baseline_preview_colors(scheme, theme).as_tuple()
            )
    assert plotting.normalize_plot_color_scheme("unknown") == "standard"


def test_plot_style_applies_selected_color_scheme_to_lines_and_legend() -> None:
    for scheme, theme in (("colorblind", "dark"), ("grayscale", "light")):
        fig, ax = plt.subplots()
        for index in range(3):
            ax.plot([0.0, 1.0], [float(index), float(index + 1)], label=str(index))
        legend = ax.legend()

        plotting.apply_plot_style(
            fig,
            ax,
            theme,
            color_scheme=scheme,
        )

        expected = plotting.plot_curve_colors(scheme, theme)[:3]
        assert tuple(line.get_color() for line in ax.lines) == expected
        handles = getattr(
            legend,
            "legend_handles",
            getattr(legend, "legendHandles", ()),
        )
        assert tuple(handle.get_color() for handle in handles) == expected
        plt.close(fig)


def test_raman_axes_use_shared_labels_and_fingerprint_tick_hierarchy() -> None:
    fig, ax = plt.subplots()
    ax.plot([60.0, 2000.0], [0.0, 1.0])

    plotting.apply_plot_style(fig, ax, "dark")

    assert ax.get_xlabel() == "Raman wavenumber / cm⁻¹"
    assert ax.get_ylabel() == "Raman intensity / Arbitr. Units"
    assert isinstance(ax.xaxis.get_major_locator(), MultipleLocator)
    visible_major_ticks = ax.get_xticks()
    visible_major_ticks = visible_major_ticks[
        (visible_major_ticks >= 0.0) & (visible_major_ticks <= 2000.0)
    ]
    np.testing.assert_allclose(
        visible_major_ticks,
        np.arange(0.0, 2000.0 + 1.0, 200.0),
    )
    visible_minor_ticks = ax.get_xticks(minor=True)
    visible_minor_ticks = visible_minor_ticks[
        (visible_minor_ticks >= 0.0) & (visible_minor_ticks <= 2000.0)
    ]
    expected_minor_ticks = np.array(
        [
            value
            for value in np.arange(0.0, 2000.0 + 1.0, 50.0)
            if value % 200.0 != 0.0
        ]
    )
    np.testing.assert_allclose(visible_minor_ticks, expected_minor_ticks)
    assert 50.0 in visible_minor_ticks
    assert 100.0 in visible_minor_ticks
    assert all(label.get_text() == "" for label in ax.get_xticklabels(minor=True))
    plt.close(fig)


def test_intensity_number_visibility_updates_axis_title_and_ticks() -> None:
    fig, ax = plt.subplots()
    ax.plot([60.0, 2000.0], [0.0, 1.0])
    plotting.apply_plot_style(fig, ax, "dark")

    plotting.set_intensity_number_visibility(ax, False)
    assert ax.get_ylabel() == "Raman intensity"
    assert not any(label.get_visible() for label in ax.get_yticklabels())
    assert not ax.yaxis.get_offset_text().get_visible()

    plotting.set_intensity_number_visibility(ax, True)
    assert ax.get_ylabel() == "Raman intensity / Arbitr. Units"
    assert any(label.get_visible() for label in ax.get_yticklabels())
    assert ax.yaxis.get_offset_text().get_visible()
    plt.close(fig)


def test_long_range_raman_axis_keeps_automatic_major_labels() -> None:
    fig, ax = plt.subplots()
    ax.plot([60.0, 4000.0], [0.0, 1.0])

    plotting.apply_plot_style(fig, ax, "dark")

    assert isinstance(ax.xaxis.get_major_locator(), AutoLocator)
    assert isinstance(ax.xaxis.get_minor_locator(), MultipleLocator)
    visible_minor_ticks = ax.get_xticks(minor=True)
    visible_minor_ticks = visible_minor_ticks[
        (visible_minor_ticks >= 0.0) & (visible_minor_ticks <= 4000.0)
    ]
    assert 100.0 in visible_minor_ticks
    assert 200.0 in visible_minor_ticks
    assert 50.0 not in visible_minor_ticks
    assert 150.0 not in visible_minor_ticks
    assert all(label.get_text() == "" for label in ax.get_xticklabels(minor=True))
    plt.close(fig)


def test_run_manifest_serializes_numpy_and_paths_safely(tmp_path) -> None:
    manifest = exports.RunManifest(
        app_version="test",
        app_commit="abc123",
        measurement_name="sample.txt",
        measurement_sha256="deadbeef",
        database_signature="dbsig",
        settings={"range": np.array([100, 900]), "path": tmp_path},
        results=({"score": np.float32(0.75), "bad": float("nan")},),
    )
    decoded = json.loads(exports.manifest_json_bytes(manifest))
    assert decoded["settings"]["range"] == [100, 900]
    assert decoded["results"][0]["score"] == 0.75
    assert decoded["results"][0]["bad"] is None


def test_resolve_git_commit_reads_symbolic_head_without_running_git(tmp_path) -> None:
    git_dir = tmp_path / ".git"
    branch_file = git_dir / "refs" / "heads" / "main"
    branch_file.parent.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    commit = "a" * 40
    branch_file.write_text(commit + "\n", encoding="utf-8")

    assert exports.resolve_git_commit(tmp_path) == commit


def test_installed_package_versions_marks_missing_distribution() -> None:
    versions = exports.installed_package_versions(
        {"definitely_missing": "ramanphaseid-package-that-does-not-exist"}
    )
    assert versions == {"definitely_missing": "not-installed"}


def test_spectrum_text_layout_is_typed_and_immutable() -> None:
    layout = exports.inspect_spectrum_text("# source\n100;1\n101;2\n")

    assert layout.header_lines == ("# source",)
    assert layout.delimiter_hint == ";"
    assert layout.exact_body_available
    np.testing.assert_array_equal(layout.axis, [100.0, 101.0])
    assert not layout.axis.flags.writeable
    assert not layout.intensity.flags.writeable


def test_alignment_overlay_plots_source_and_scored_trace() -> None:
    axis = np.arange(10.0)
    provided_mask = np.zeros(axis.size, dtype=bool)
    provided_mask[1:8] = True
    aligned_mask = np.zeros(axis.size, dtype=bool)
    aligned_mask[3:10] = True
    aligned_mask[5:7] = False
    measurement_mask = np.zeros(axis.size, dtype=bool)
    measurement_mask[2:9] = True
    measurement_mask[4] = False
    overlay = plotting.AlignmentOverlay(
        axis_cm1=axis,
        measurement=np.linspace(0.0, 1.0, axis.size),
        library_as_provided=np.linspace(0.0, 0.8, axis.size),
        library_aligned=np.linspace(0.0, 0.9, axis.size),
        valid_mask=aligned_mask,
        label="Quartz",
        shift_cm1=2.0,
        library_as_provided_mask=provided_mask,
        measurement_mask=measurement_mask,
        score=0.88,
        coverage_fraction=0.95,
        measurement_label="normalised signed residual (matching query)",
        peak_consistency=0.73,
    )
    fig, ax = plt.subplots()
    plotting.plot_alignment_evidence(ax, overlay)
    plotting.apply_plot_style(fig, ax, "dark")
    assert len(ax.lines) == 3
    assert ax.lines[0].get_label() == "normalised signed residual (matching query)"
    assert "score-aligned" in ax.lines[2].get_label()
    assert any("peak agreement=0.730" in text.get_text() for text in ax.texts)
    assert not overlay.measurement.flags.writeable
    assert not overlay.valid_mask.flags.writeable
    assert not overlay.measurement_mask.flags.writeable
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), axis)
    np.testing.assert_array_equal(ax.lines[2].get_xdata(), axis)
    assert np.isnan(ax.lines[0].get_ydata()[4])
    assert np.isnan(ax.lines[1].get_ydata()[~provided_mask]).all()
    assert np.isnan(ax.lines[2].get_ydata()[5:7]).all()
    assert ax.lines[1].get_alpha() == 1.0
    assert ax.lines[1].get_zorder() > ax.lines[2].get_zorder()
    assert getattr(ax.lines[1], "_unscaled_dash_pattern") == (
        plotting.BASELINE_DOTTED_LINESTYLE
    )
    assert plotting.figure_to_bytes(fig, "png").startswith(b"\x89PNG")
    plt.close(fig)


def test_segmented_line_data_inserts_read_only_nan_gap() -> None:
    axis = np.array([100.0, 101.0, 102.0, 250.0, 251.0])
    signal = np.arange(axis.size, dtype=float)

    plotted_axis, plotted_signal = plotting.segmented_line_data(
        axis,
        signal,
        (slice(0, 3), slice(3, 5)),
    )

    np.testing.assert_allclose(plotted_axis[:3], axis[:3])
    np.testing.assert_allclose(plotted_axis[4:], axis[3:])
    assert np.isnan(plotted_axis[3])
    assert np.isnan(plotted_signal[3])
    assert not plotted_axis.flags.writeable
    assert not plotted_signal.flags.writeable


def test_unsupported_projection_padding_is_hidden_from_line_plot() -> None:
    projected = np.array([0.0, 4.0, 5.0, 0.0])
    valid = np.array([False, True, True, False])

    plotted = plotting.mask_unsupported_line_values(projected, valid)

    assert np.isnan(plotted[[0, 3]]).all()
    np.testing.assert_allclose(plotted[1:3], [4.0, 5.0])
    assert not plotted.flags.writeable


def test_processing_difference_can_be_offset_below_visible_curves() -> None:
    difference = np.array([-0.4, 0.2, 0.5, np.nan, 20.0])
    input_trace = np.array([10.0, 12.0, 11.0, 13.0, -100.0])
    output_trace = np.array([9.5, 12.2, 11.5, 13.1, -90.0])
    visible = np.array([True, True, True, True, False])

    shifted, offset = plotting.offset_trace_below(
        difference,
        (input_trace, output_trace),
        visible,
    )

    finite_visible = visible & np.isfinite(difference)
    np.testing.assert_allclose(
        shifted[finite_visible] - difference[finite_visible],
        offset,
    )
    assert np.max(shifted[finite_visible]) < min(
        np.min(input_trace[visible]),
        np.min(output_trace[visible]),
    )
    assert np.isnan(shifted[3])
    assert not shifted.flags.writeable


def test_plot_style_uses_long_dotted_pattern_for_curves_and_legend() -> None:
    fig, ax = plt.subplots()
    explicit_dotted, = ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle=":",
        linewidth=0.6,
        label="explicit dotted",
    )
    automatic_second_dotted, = ax.plot(
        [0.0, 1.0],
        [1.0, 2.0],
        label="second",
    )
    ax.plot([0.0, 1.0], [2.0, 3.0], label="third")
    automatic_dotted, = ax.plot(
        [0.0, 1.0],
        [3.0, 4.0],
        linewidth=2.0,
        label="fourth",
    )
    legend = ax.legend()

    plotting.apply_plot_style(fig, ax, "dark")

    handles = getattr(
        legend,
        "legend_handles",
        getattr(legend, "legendHandles", ()),
    )
    for line, handle in (
        (explicit_dotted, handles[0]),
        (automatic_second_dotted, handles[1]),
        (automatic_dotted, handles[3]),
    ):
        assert line.get_linewidth() == plotting.PLOT_LINEWIDTH
        assert handle.get_linewidth() == plotting.PLOT_LINEWIDTH
        assert getattr(line, "_unscaled_dash_pattern") == (
            plotting.BASELINE_DOTTED_LINESTYLE
        )
        assert getattr(handle, "_unscaled_dash_pattern") == (
            plotting.BASELINE_DOTTED_LINESTYLE
        )
        np.testing.assert_allclose(
            getattr(line, "_dash_pattern")[1],
            plotting.BASELINE_DOTTED_LINESTYLE[1],
        )
        np.testing.assert_allclose(
            getattr(handle, "_dash_pattern")[1],
            plotting.BASELINE_DOTTED_LINESTYLE[1],
        )
    plt.close(fig)


def test_plot_style_preserves_semantic_colours_and_styles_in_legend() -> None:
    fig, ax = plt.subplots()
    baseline_line, = ax.plot(
        [100.0, 101.0],
        [2.0, 2.1],
        color="tab:orange",
        linestyle=plotting.BASELINE_DOTTED_LINESTYLE,
        label="baseline",
    )
    corrected_line, = ax.plot(
        [100.0, 101.0],
        [0.1, 0.2],
        color="tab:green",
        linestyle="-",
        label="corrected",
    )
    legend = ax.legend()

    plotting.apply_plot_style(
        fig,
        ax,
        "dark",
        preserve_line_appearance=True,
    )

    handles = getattr(
        legend,
        "legend_handles",
        getattr(legend, "legendHandles", ()),
    )
    assert baseline_line.get_color() == "tab:orange"
    assert corrected_line.get_color() == "tab:green"
    assert getattr(baseline_line, "_unscaled_dash_pattern") == (
        plotting.BASELINE_DOTTED_LINESTYLE
    )
    assert corrected_line.get_linestyle() == "-"
    assert baseline_line.get_linewidth() == plotting.PLOT_LINEWIDTH
    assert corrected_line.get_linewidth() == plotting.PLOT_LINEWIDTH
    assert [handle.get_color() for handle in handles] == [
        baseline_line.get_color(),
        corrected_line.get_color(),
    ]
    assert getattr(handles[0], "_unscaled_dash_pattern") == (
        plotting.BASELINE_DOTTED_LINESTYLE
    )
    assert handles[1].get_linestyle() == "-"
    assert all(
        handle.get_linewidth() == plotting.PLOT_LINEWIDTH
        for handle in handles
    )
    plt.close(fig)


def test_figure_render_bundle_contains_display_png_and_download_svg() -> None:
    fig, ax = plt.subplots()
    ax.plot([100.0, 101.0], [0.0, 1.0], label="spectrum")
    plotting.apply_plot_style(fig, ax, "dark")

    bundle = plotting.render_figure_bundle(fig)

    assert isinstance(bundle, plotting.FigureRenderBundle)
    assert bundle.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"<svg" in bundle.svg[:1024].lower()
    plt.close(fig)

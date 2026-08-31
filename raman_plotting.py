"""Plot styling, rendering, and evidence-overlay helpers for RamanPhaseID."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Literal, Sequence

import numpy as np
from matplotlib.ticker import AutoLocator, MultipleLocator, NullFormatter


PlotTheme = Literal["dark", "light"]
PlotColorScheme = Literal["standard", "colorblind", "grayscale"]
# Compatibility name retained for callers from the baseline-only selector.
BaselineColorScheme = PlotColorScheme

# Matplotlib's standard dotted pattern is 1.0 points on, 1.65 points off.
# Doubling both values keeps the same visual rhythm while making each mark and
# intervening gap easier to distinguish. This is the shared comparison-curve
# pattern: notably the fitted baseline, smoothed signal, and source-library
# trace in the database-match overlay.
BASELINE_DOTTED_LINESTYLE = (0.0, (2.0, 3.3))
PLOT_LINEWIDTH = 1.0
RAMAN_WAVENUMBER_LABEL = "Raman wavenumber / cm⁻¹"
RAMAN_INTENSITY_LABEL = "Raman intensity / Arbitr. Units"
RAMAN_INTENSITY_LABEL_NO_UNITS = "Raman intensity"
FINGERPRINT_MAX_CM1 = 2000.0
FINGERPRINT_MAJOR_TICK_CM1 = 200.0
RAMAN_MINOR_TICK_CM1 = 50.0
LONG_RANGE_TICK_THRESHOLD_CM1 = 2200.0
LONG_RANGE_MINOR_TICK_CM1 = 100.0


@dataclass(frozen=True, slots=True)
class PlotPalette:
    background: str
    panel: str
    text: str
    spine: str
    grid: str
    curves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FigureRenderBundle:
    """Immutable browser-display PNG plus vector-quality SVG download."""

    png: bytes
    svg: bytes

    def __post_init__(self) -> None:
        png = bytes(self.png)
        svg = bytes(self.svg)
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("figure bundle PNG payload has an invalid signature")
        if b"<svg" not in svg[:1024].lower():
            raise ValueError("figure bundle SVG payload has no SVG root element")
        object.__setattr__(self, "png", png)
        object.__setattr__(self, "svg", svg)


@dataclass(frozen=True, slots=True)
class BaselinePreviewColors:
    """Semantic line colours for the baseline-inspection figure."""

    input_signal: str
    fitted_baseline: str
    corrected_signal: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (
            self.input_signal,
            self.fitted_baseline,
            self.corrected_signal,
        )


@dataclass(frozen=True, slots=True)
class AlignmentOverlay:
    """All curves required to audit the evidence used for one match."""

    axis_cm1: np.ndarray
    measurement: np.ndarray
    library_as_provided: np.ndarray
    library_aligned: np.ndarray
    valid_mask: np.ndarray
    label: str
    shift_cm1: float
    library_as_provided_mask: np.ndarray | None = None
    measurement_mask: np.ndarray | None = None
    score: float | None = None
    coverage_fraction: float | None = None
    shift_at_boundary: bool = False
    measurement_label: str = "measurement"
    peak_consistency: float | None = None
    aligned_treatment: str = ""

    def __post_init__(self) -> None:
        axis = np.array(self.axis_cm1, dtype=float, copy=True).reshape(-1)
        measurement = np.array(self.measurement, dtype=float, copy=True).reshape(-1)
        provided = np.array(self.library_as_provided, dtype=float, copy=True).reshape(-1)
        aligned = np.array(self.library_aligned, dtype=float, copy=True).reshape(-1)
        valid = np.array(self.valid_mask, dtype=bool, copy=True).reshape(-1)
        provided_valid = (
            valid.copy()
            if self.library_as_provided_mask is None
            else np.array(
                self.library_as_provided_mask,
                dtype=bool,
                copy=True,
            ).reshape(-1)
        )
        measurement_valid = (
            np.isfinite(measurement)
            if self.measurement_mask is None
            else np.array(
                self.measurement_mask,
                dtype=bool,
                copy=True,
            ).reshape(-1)
        )
        lengths = {
            axis.size,
            measurement.size,
            provided.size,
            aligned.size,
            valid.size,
            provided_valid.size,
            measurement_valid.size,
        }
        if len(lengths) != 1:
            raise ValueError("all alignment-overlay arrays must have equal length")
        for value in (
            axis,
            measurement,
            provided,
            aligned,
            valid,
            provided_valid,
            measurement_valid,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "axis_cm1", axis)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "library_as_provided", provided)
        object.__setattr__(self, "library_aligned", aligned)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "library_as_provided_mask", provided_valid)
        object.__setattr__(self, "measurement_mask", measurement_valid)
        object.__setattr__(self, "measurement_label", str(self.measurement_label))
        object.__setattr__(self, "aligned_treatment", str(self.aligned_treatment))


_STANDARD_CURVE_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)
_COLORBLIND_LIGHT_CURVE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#009E73",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
_COLORBLIND_DARK_CURVE_COLORS = (
    "#56B4E9",
    "#E69F00",
    "#CC79A7",
    "#F0E442",
    "#009E73",
    "#D55E00",
    "#F5F5F5",
)
_GRAYSCALE_LIGHT_CURVE_COLORS = (
    "#222222",
    "#666666",
    "#999999",
    "#444444",
    "#777777",
    "#888888",
    "#111111",
)
_GRAYSCALE_DARK_CURVE_COLORS = (
    "#F2F2F2",
    "#B8B8B8",
    "#7A7A7A",
    "#DADADA",
    "#A0A0A0",
    "#8B8B8B",
    "#C8C8C8",
)


_PALETTES: dict[PlotTheme, PlotPalette] = {
    "light": PlotPalette(
        background="#FFFFFF",
        panel="#FFFFFF",
        text="#1F2933",
        spine="#BFC7D0",
        grid="#E6EBF1",
        curves=_STANDARD_CURVE_COLORS,
    ),
    "dark": PlotPalette(
        background="#0E1117",
        panel="#161B22",
        text="#E6EDF3",
        spine="#3D4854",
        grid="#2B3440",
        curves=_STANDARD_CURVE_COLORS,
    ),
}


_BASELINE_PREVIEW_COLORS: dict[
    PlotTheme,
    dict[PlotColorScheme, BaselinePreviewColors],
] = {
    "light": {
        "standard": BaselinePreviewColors(*_STANDARD_CURVE_COLORS[:3]),
        # Okabe-Ito-derived hues selected to remain separable under common
        # red-green colour-vision deficiencies on a light plot background.
        "colorblind": BaselinePreviewColors(
            *_COLORBLIND_LIGHT_CURVE_COLORS[:3]
        ),
        "grayscale": BaselinePreviewColors(
            *_GRAYSCALE_LIGHT_CURVE_COLORS[:3]
        ),
    },
    "dark": {
        "standard": BaselinePreviewColors(*_STANDARD_CURVE_COLORS[:3]),
        # The lighter Okabe-Ito blue/orange variants retain contrast against
        # the dark panel while preserving the same three semantic hues.
        "colorblind": BaselinePreviewColors(
            *_COLORBLIND_DARK_CURVE_COLORS[:3]
        ),
        "grayscale": BaselinePreviewColors(
            *_GRAYSCALE_DARK_CURVE_COLORS[:3]
        ),
    },
}


_PLOT_CURVE_COLORS: dict[
    PlotTheme,
    dict[PlotColorScheme, tuple[str, ...]],
] = {
    "light": {
        "standard": _PALETTES["light"].curves,
        "colorblind": _COLORBLIND_LIGHT_CURVE_COLORS,
        "grayscale": _GRAYSCALE_LIGHT_CURVE_COLORS,
    },
    "dark": {
        "standard": _PALETTES["dark"].curves,
        "colorblind": _COLORBLIND_DARK_CURVE_COLORS,
        "grayscale": _GRAYSCALE_DARK_CURVE_COLORS,
    },
}


def normalize_plot_theme(theme: str | None) -> PlotTheme:
    return "light" if str(theme).strip().lower() == "light" else "dark"


def normalize_plot_color_scheme(
    scheme: str | None,
) -> PlotColorScheme:
    normalized = str(scheme).strip().lower()
    if normalized in {"colorblind", "grayscale"}:
        return normalized
    return "standard"


# Compatibility alias for sessions/tests created while this choice applied
# only to the baseline preview.
normalize_baseline_color_scheme = normalize_plot_color_scheme


def plot_curve_colors(
    scheme: str | None,
    theme: str | None = "dark",
) -> tuple[str, ...]:
    """Return the repeatable line-color cycle for any RamanPhaseID plot."""

    return _PLOT_CURVE_COLORS[normalize_plot_theme(theme)][
        normalize_plot_color_scheme(scheme)
    ]


def baseline_preview_colors(
    scheme: str | None,
    theme: str | None = "dark",
) -> BaselinePreviewColors:
    """Return a contrast-aware semantic palette for the baseline preview."""

    return _BASELINE_PREVIEW_COLORS[normalize_plot_theme(theme)][
        normalize_plot_color_scheme(scheme)
    ]


def _axis_is_within_fingerprint_region(ax) -> bool:
    """Return whether either the displayed or autoscaled data range is 0–2000 cm⁻¹."""

    tolerance = 1.0e-9
    candidate_bounds = [ax.get_xlim()]
    data_bounds = np.asarray(ax.dataLim.intervalx, dtype=float).reshape(-1)
    if data_bounds.size == 2 and np.all(np.isfinite(data_bounds)):
        candidate_bounds.append((float(data_bounds[0]), float(data_bounds[1])))
    for first, second in candidate_bounds:
        lower, upper = sorted((float(first), float(second)))
        if (
            np.isfinite(lower)
            and np.isfinite(upper)
            and lower >= -tolerance
            and upper <= FINGERPRINT_MAX_CM1 + tolerance
        ):
            return True
    return False


def _axis_is_positive_long_range(ax) -> bool:
    """Return whether a positive Raman range extends beyond 2200 cm⁻¹."""

    tolerance = 1.0e-9
    candidate_bounds = [ax.get_xlim()]
    data_bounds = np.asarray(ax.dataLim.intervalx, dtype=float).reshape(-1)
    if data_bounds.size == 2 and np.all(np.isfinite(data_bounds)):
        candidate_bounds.append((float(data_bounds[0]), float(data_bounds[1])))
    for first, second in candidate_bounds:
        lower, upper = sorted((float(first), float(second)))
        if (
            np.isfinite(lower)
            and np.isfinite(upper)
            and lower >= -tolerance
            and upper > LONG_RANGE_TICK_THRESHOLD_CM1 + tolerance
        ):
            return True
    return False


def configure_raman_axes(ax) -> None:
    """Apply shared spectral labels and hierarchical wavenumber ticks.

    Fingerprint-region plots label every 200 cm⁻¹. Unlabelled minor marks
    occur every 50 cm⁻¹, which includes both the requested 50 and 100 cm⁻¹
    positions. Positive ranges extending beyond 2200 cm⁻¹ retain Matplotlib's
    less crowded automatic major labels and reduce the unlabelled minor marks
    to every 100 cm⁻¹.
    """

    ax.set_xlabel(RAMAN_WAVENUMBER_LABEL)
    ax.set_ylabel(RAMAN_INTENSITY_LABEL)
    if _axis_is_within_fingerprint_region(ax):
        ax.xaxis.set_major_locator(MultipleLocator(FINGERPRINT_MAJOR_TICK_CM1))
    else:
        # Make repeated calls deterministic if a caller changes from a
        # fingerprint viewport to a longer range on the same axes.
        ax.xaxis.set_major_locator(AutoLocator())
    minor_tick_spacing = (
        LONG_RANGE_MINOR_TICK_CM1
        if _axis_is_positive_long_range(ax)
        else RAMAN_MINOR_TICK_CM1
    )
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", length=6.0)
    ax.tick_params(axis="x", which="minor", length=3.0)


def apply_plot_style(
    fig,
    ax,
    theme: str | None = "dark",
    *,
    color_scheme: str | None = "standard",
    preserve_line_appearance: bool = False,
) -> None:
    """Apply the RamanPhaseID frame style and keep legends synchronized.

    Most scientific overlays use the selected theme-aware color scheme plus
    alternating line styles.  Every curve uses the baseline preview's
    one-point width so dash lengths are physically consistent.  A caller with
    semantically assigned colours/styles (for example the
    raw/baseline/corrected preview) can preserve those assignments explicitly.
    """

    normalized_theme = normalize_plot_theme(theme)
    palette = _PALETTES[normalized_theme]
    curve_colors = plot_curve_colors(color_scheme, normalized_theme)
    # The second curve is the orange comparison trace throughout the guided
    # workflow. Give it the same relatively open pattern as the fitted
    # baseline so a nearly coincident solid curve remains visible in the gaps.
    line_styles = (
        "-",
        BASELINE_DOTTED_LINESTYLE,
        "-.",
        BASELINE_DOTTED_LINESTYLE,
    )
    configure_raman_axes(ax)
    fig.patch.set_facecolor(palette.background)
    ax.set_facecolor(palette.panel)
    for index, line in enumerate(ax.get_lines()):
        line.set_linewidth(PLOT_LINEWIDTH)
        if not preserve_line_appearance:
            line.set_color(curve_colors[index % len(curve_colors)])
            if line.get_linestyle() in (None, "-"):
                line.set_linestyle(line_styles[index % len(line_styles)])
            elif line.get_linestyle() == ":":
                line.set_linestyle(BASELINE_DOTTED_LINESTYLE)
        elif line.get_linestyle() == ":":
            line.set_linestyle(BASELINE_DOTTED_LINESTYLE)
    for spine in ax.spines.values():
        spine.set_color(palette.spine)
    ax.tick_params(axis="both", which="both", colors=palette.text, direction="in", top=True, right=True)
    ax.xaxis.label.set_color(palette.text)
    ax.yaxis.label.set_color(palette.text)
    ax.title.set_color(palette.text)
    ax.grid(alpha=0.25, color=palette.grid)
    for item in ax.texts:
        item.set_color(palette.text)
    legend = ax.get_legend()
    if legend is not None:
        legend_handles = getattr(
            legend,
            "legend_handles",
            getattr(legend, "legendHandles", ()),
        )
        for line, handle in zip(ax.get_lines(), legend_handles):
            handle.set_color(line.get_color())
            # get_linestyle() collapses a custom dash sequence to Matplotlib's
            # generic "--" alias.  Preserve the exact unscaled dash pattern so
            # legend samples use the same marks and gaps as their curves.
            dash_pattern = getattr(line, "_unscaled_dash_pattern", None)
            if (
                isinstance(dash_pattern, tuple)
                and len(dash_pattern) == 2
                and dash_pattern[1] is not None
            ):
                handle.set_linestyle(dash_pattern)
            else:
                handle.set_linestyle(line.get_linestyle())
            handle.set_linewidth(line.get_linewidth())
            handle.set_alpha(line.get_alpha())
        frame = legend.get_frame()
        frame.set_facecolor(palette.panel)
        frame.set_edgecolor(palette.spine)
        frame.set_alpha(0.85)
        for item in legend.get_texts():
            item.set_color(palette.text)


def mask_unsupported_line_values(
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Replace zero-filled projection padding with NaN plot separators."""

    plotted = np.array(values, dtype=float, copy=True).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if plotted.size != valid.size:
        raise ValueError("line values and validity mask must have equal length")
    plotted[~valid] = np.nan
    plotted.setflags(write=False)
    return plotted


def offset_trace_below(
    values: np.ndarray,
    references: Sequence[np.ndarray],
    visible_mask: np.ndarray | None = None,
    *,
    gap_fraction: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Translate one trace below visible references without changing its shape."""

    trace = np.array(values, dtype=float, copy=True).reshape(-1)
    if not np.isfinite(gap_fraction) or gap_fraction < 0.0:
        raise ValueError("gap_fraction must be a finite non-negative value")
    if visible_mask is None:
        visible = np.ones(trace.size, dtype=bool)
    else:
        visible = np.asarray(visible_mask, dtype=bool).reshape(-1)
        if visible.size != trace.size:
            raise ValueError("visible mask and trace must have equal length")

    reference_min = np.inf
    reference_max = -np.inf
    for reference in references:
        reference_array = np.asarray(reference, dtype=float).reshape(-1)
        if reference_array.size != trace.size:
            raise ValueError("reference traces and translated trace must have equal length")
        finite_reference = visible & np.isfinite(reference_array)
        if np.any(finite_reference):
            reference_min = min(
                reference_min,
                float(np.min(reference_array[finite_reference])),
            )
            reference_max = max(
                reference_max,
                float(np.max(reference_array[finite_reference])),
            )

    finite_trace = visible & np.isfinite(trace)
    if not np.any(finite_trace) or not np.isfinite(reference_min):
        trace.setflags(write=False)
        return trace, 0.0

    visible_trace = trace[finite_trace]
    reference_scale = max(abs(reference_min), abs(reference_max), 1.0)
    display_scale = max(
        reference_max - reference_min,
        float(np.ptp(visible_trace)),
        0.01 * reference_scale,
        1.0e-12,
    )
    gap = float(gap_fraction) * display_scale
    offset = reference_min - gap - float(np.max(visible_trace))
    trace[np.isfinite(trace)] += offset
    trace.setflags(write=False)
    return trace, float(offset)


def set_intensity_number_visibility(ax, show_numbers: bool) -> None:
    show_numbers = bool(show_numbers)
    ax.set_ylabel(
        RAMAN_INTENSITY_LABEL
        if show_numbers
        else RAMAN_INTENSITY_LABEL_NO_UNITS
    )
    ax.tick_params(axis="y", labelleft=show_numbers)
    ax.yaxis.get_offset_text().set_visible(show_numbers)


def figure_to_bytes(fig, fmt: str) -> bytes:
    normalized = str(fmt).strip().lower()
    if normalized not in {"png", "svg", "pdf"}:
        raise ValueError(f"unsupported figure export format: {fmt!r}")
    buffer = io.BytesIO()
    fig.savefig(buffer, format=normalized, bbox_inches="tight")
    return buffer.getvalue()


def render_figure_bundle(fig) -> FigureRenderBundle:
    """Render one Matplotlib figure for cached display and download reuse."""

    return FigureRenderBundle(
        png=figure_to_bytes(fig, "png"),
        svg=figure_to_bytes(fig, "svg"),
    )


def segmented_line_data(
    axis_cm1: np.ndarray,
    values: np.ndarray,
    segments: Sequence[slice],
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN separators so Matplotlib cannot bridge acquisition gaps."""

    axis = np.asarray(axis_cm1, dtype=float).reshape(-1)
    signal = np.asarray(values, dtype=float).reshape(-1)
    if axis.size != signal.size:
        raise ValueError("axis_cm1 and values must have equal length")
    axis_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    for segment in segments:
        if not isinstance(segment, slice):
            raise TypeError("segments must contain slice objects")
        segment_axis = axis[segment]
        segment_signal = signal[segment]
        if segment_axis.size == 0:
            continue
        if axis_parts:
            axis_parts.append(np.array([np.nan], dtype=float))
            signal_parts.append(np.array([np.nan], dtype=float))
        axis_parts.append(segment_axis)
        signal_parts.append(segment_signal)
    if axis_parts:
        plotted_axis = np.concatenate(axis_parts)
        plotted_signal = np.concatenate(signal_parts)
    else:
        plotted_axis = np.array([], dtype=float)
        plotted_signal = np.array([], dtype=float)
    plotted_axis.setflags(write=False)
    plotted_signal.setflags(write=False)
    return plotted_axis, plotted_signal


def plot_alignment_evidence(ax, overlay: AlignmentOverlay) -> None:
    """Plot both the source trace and the exact shifted trace used for scoring."""

    aligned_mask = overlay.valid_mask
    provided_mask = overlay.library_as_provided_mask
    measurement_mask = overlay.measurement_mask
    assert provided_mask is not None
    assert measurement_mask is not None
    axis = overlay.axis_cm1
    # Retain the full x axis and mask y with NaN. Boolean-indexed disjoint runs
    # would otherwise be joined by Matplotlib across unmeasured spectral gaps.
    ax.plot(
        axis,
        np.where(measurement_mask, overlay.measurement, np.nan),
        label=overlay.measurement_label,
        linewidth=PLOT_LINEWIDTH,
        zorder=1.0,
    )
    ax.plot(
        axis,
        np.where(provided_mask, overlay.library_as_provided, np.nan),
        label=f"{overlay.label} · library as provided",
        linewidth=PLOT_LINEWIDTH,
        alpha=1.0,
        linestyle=BASELINE_DOTTED_LINESTYLE,
        zorder=3.0,
    )
    treatment = f"{overlay.aligned_treatment} · " if overlay.aligned_treatment else ""
    score_label = (
        f"{overlay.label} · {treatment}score-aligned "
        f"(Δ={overlay.shift_cm1:+g} cm⁻¹)"
    )
    ax.plot(
        axis,
        np.where(aligned_mask, overlay.library_aligned, np.nan),
        label=score_label,
        linewidth=PLOT_LINEWIDTH,
        alpha=0.9,
        linestyle="--",
        zorder=2.0,
    )
    details: list[str] = []
    if overlay.score is not None:
        details.append(f"rank evidence={overlay.score:.3f}")
    if overlay.coverage_fraction is not None:
        details.append(f"common support={100.0 * overlay.coverage_fraction:.1f}%")
    if overlay.peak_consistency is not None:
        details.append(f"peak agreement={overlay.peak_consistency:.3f}")
    if overlay.shift_at_boundary:
        details.append("alignment reached search boundary")
    if details:
        ax.text(0.01, 0.98, " · ".join(details), transform=ax.transAxes, va="top", fontsize="small")
    configure_raman_axes(ax)
    ax.legend()


# Compatibility aliases retained while the Streamlit script is migrated.
_fig_to_bytes = figure_to_bytes
_normalize_plot_theme = normalize_plot_theme
_apply_plot_style = apply_plot_style
_set_intensity_number_visibility = set_intensity_number_visibility


__all__ = [
    "AlignmentOverlay",
    "BASELINE_DOTTED_LINESTYLE",
    "BaselineColorScheme",
    "BaselinePreviewColors",
    "FINGERPRINT_MAJOR_TICK_CM1",
    "FINGERPRINT_MAX_CM1",
    "FigureRenderBundle",
    "LONG_RANGE_MINOR_TICK_CM1",
    "LONG_RANGE_TICK_THRESHOLD_CM1",
    "PlotPalette",
    "PlotColorScheme",
    "PLOT_LINEWIDTH",
    "RAMAN_INTENSITY_LABEL",
    "RAMAN_INTENSITY_LABEL_NO_UNITS",
    "RAMAN_MINOR_TICK_CM1",
    "RAMAN_WAVENUMBER_LABEL",
    "PlotTheme",
    "apply_plot_style",
    "baseline_preview_colors",
    "configure_raman_axes",
    "figure_to_bytes",
    "mask_unsupported_line_values",
    "normalize_baseline_color_scheme",
    "normalize_plot_color_scheme",
    "normalize_plot_theme",
    "offset_trace_below",
    "plot_curve_colors",
    "plot_alignment_evidence",
    "render_figure_bundle",
    "segmented_line_data",
    "set_intensity_number_visibility",
]

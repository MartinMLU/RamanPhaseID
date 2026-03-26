from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter


st.set_page_config(page_title="Raman Subtraction", layout="wide")
st.title("Raman Spectra: Subtraction with Scaling")
st.caption(
    "Select two spectra. The second spectrum is scaled and subtracted from the first."
)


@st.cache_data
def load_spectrum(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["shift", "intensity"],
        engine="python",
        encoding="latin-1",
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("shift").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No spectral data found in file: {file_path}")
    return df


@st.cache_data
def load_header(file_path: str) -> str:
    header_lines = []
    with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
            else:
                break
    return "".join(header_lines)


def align_to_reference(x_ref: np.ndarray, x_other: np.ndarray, y_other: np.ndarray) -> np.ndarray:
    x_match = len(x_ref) == len(x_other) and np.allclose(x_ref, x_other, rtol=1e-7, atol=1e-9)
    if x_match:
        return y_other
    return np.interp(x_ref, x_other, y_other)


def smooth_spectrum_savgol(
    y: np.ndarray, window_length: int = 21, polyorder: int = 3
) -> tuple[np.ndarray, int, int]:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 5:
        return y, 0, 0

    win = int(window_length)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1

    max_odd = n if n % 2 == 1 else n - 1
    if max_odd < 3:
        return y, 0, 0
    win = min(win, max_odd)

    poly = int(polyorder)
    if poly < 1:
        poly = 1
    poly = min(poly, win - 1)

    y_smooth = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    return y_smooth, win, poly


def arpls_baseline(
    y: np.ndarray, lam: float = 1e5, ratio: float = 1e-6, max_iter: int = 50
) -> tuple[np.ndarray, int]:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return np.zeros_like(y), 0

    d2 = np.diff(np.eye(n), 2, axis=0)
    h = lam * (d2.T @ d2)
    w = np.ones(n, dtype=float)
    z = np.zeros_like(y)

    for i in range(max_iter):
        w_prev = w.copy()
        w_mat = np.diag(w)
        z = np.linalg.solve(w_mat + h, w * y)

        residual = y - z
        neg = residual[residual < 0]
        if neg.size < 2:
            return z, i + 1

        mu = float(np.mean(neg))
        sigma = float(np.std(neg))
        if sigma <= np.finfo(float).eps:
            return z, i + 1

        exponent = np.clip(2.0 * (residual - (2.0 * sigma - mu)) / sigma, -60.0, 60.0)
        w = 1.0 / (1.0 + np.exp(exponent))

        rel_change = np.linalg.norm(w - w_prev) / (np.linalg.norm(w_prev) + 1e-12)
        if rel_change < ratio:
            return z, i + 1

    return z, max_iter


txt_files = sorted(Path(".").glob("*.txt"))

if not txt_files:
    st.error("No .txt files found in the current folder.")
    st.stop()

file_names = [f.name for f in txt_files]
default_second = 1 if len(file_names) > 1 else 0

col1, col2, col3 = st.columns([1, 1, 1.2])
with col1:
    ref_name = st.selectbox("Spectrum 1 (measurement)", file_names, index=0)
with col2:
    sub_name = st.selectbox("Spectrum 2 (correction, subtracted)", file_names, index=default_second)
with col3:
    scale_factor = st.slider(
        "Scaling for Spectrum 2 before subtraction",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.001,
    )

with st.expander("Smoothing for Spectrum 2 (Savitzky-Golay)", expanded=False):
    use_savgol = st.checkbox("Enable Savitzky-Golay", value=True)
    sg_window = st.slider(
        "Window length (points, odd)",
        min_value=5,
        max_value=201,
        value=21,
        step=2,
    )
    sg_poly = st.slider("Polynomial order", min_value=1, max_value=7, value=3, step=1)

with st.expander("Baseline correction on difference spectrum (arPLS)", expanded=False):
    use_arpls = st.checkbox("Enable arPLS", value=False)
    lam_exp = st.slider("Smoothing lambda as 10^x", min_value=2, max_value=9, value=5, step=1)
    ratio_exp = st.slider(
        "Convergence threshold as 10^(-x)", min_value=2, max_value=10, value=6, step=1
    )
    max_iter = st.slider("Maximum arPLS iterations", min_value=10, max_value=200, value=50, step=5)

df_ref = load_spectrum(ref_name)
df_sub = load_spectrum(sub_name)

x_ref = df_ref["shift"].to_numpy()
y_ref = df_ref["intensity"].to_numpy()

x_sub = df_sub["shift"].to_numpy()
y_sub = df_sub["intensity"].to_numpy()
y_sub_aligned = align_to_reference(x_ref, x_sub, y_sub)
y_sub_prepared = y_sub_aligned
sg_window_eff = 0
sg_poly_eff = 0
if use_savgol:
    y_sub_prepared, sg_window_eff, sg_poly_eff = smooth_spectrum_savgol(
        y_sub_aligned, window_length=sg_window, polyorder=sg_poly
    )

y_sub_scaled = scale_factor * y_sub_prepared
y_diff = y_ref - y_sub_scaled

baseline = None
y_diff_corrected = None
arpls_iterations = 0
if use_arpls:
    lam = float(10**lam_exp)
    ratio = float(10 ** (-ratio_exp))
    baseline, arpls_iterations = arpls_baseline(y_diff, lam=lam, ratio=ratio, max_iter=max_iter)
    y_diff_corrected = y_diff - baseline

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_ref, y_ref, label="Measurement", linewidth=1.6)
if use_savgol:
    ax.plot(
        x_ref,
        y_sub_scaled,
        label="Correction",
        linewidth=1.6,
    )
else:
    ax.plot(
        x_ref,
        y_sub_scaled,
        label="Correction",
        linewidth=1.6,
    )
ax.plot(x_ref, y_diff, label="Difference", linewidth=2.0)
if use_savgol:
    st.caption(f"Savitzky-Golay enabled: window={sg_window_eff}, polynomial order={sg_poly_eff}")
if use_arpls and baseline is not None and y_diff_corrected is not None:
    ax.plot(x_ref, baseline, label="arPLS baseline (on difference)", linestyle="--", linewidth=1.4)
    ax.plot(
        x_ref,
        y_diff_corrected,
        label="Difference after arPLS baseline correction",
        linewidth=2.0,
    )
    st.caption(f"arPLS enabled: lambda=1e{lam_exp}, ratio=1e-{ratio_exp}, iterations={arpls_iterations}")

ax.set_xlabel("Raman shift (cm^-1)")
ax.set_ylabel("Intensity (a.u.)")
ax.tick_params(axis="both", direction="in", top=True, right=True)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
st.pyplot(fig, use_container_width=True)

png_buffer = BytesIO()
fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
png_buffer.seek(0)
svg_buffer = BytesIO()
fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
svg_buffer.seek(0)

dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    st.download_button(
        "Download first plot as PNG",
        data=png_buffer.getvalue(),
        file_name=f"{Path(ref_name).stem}_plot.png",
        mime="image/png",
    )
with dl_col2:
    st.download_button(
        "Download first plot as SVG",
        data=svg_buffer.getvalue(),
        file_name=f"{Path(ref_name).stem}_plot.svg",
        mime="image/svg+xml",
    )

if use_arpls and y_diff_corrected is not None:
    st.subheader("Difference after arPLS baseline correction (interactive)")
    fig_diff = go.Figure()
    fig_diff.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_diff_corrected,
            mode="lines+markers",
            name="Difference after arPLS",
            marker={"size": 4},
            line={"width": 2},
            hovertemplate=(
                "Raman shift: %{x:.2f} cm^-1"
                "<br>Intensity: %{y:.2f} a.u."
                "<extra></extra>"
            ),
        )
    )
    fig_diff.update_layout(
        xaxis_title="Raman shift (cm^-1)",
        yaxis_title="Intensity (a.u.)",
        dragmode="zoom",
        hovermode="closest",
    )
    st.plotly_chart(fig_diff, use_container_width=True)
else:
    st.info("Enable arPLS to show the separate interactive difference plot.")

out_df = pd.DataFrame(
    {
        "shift_cm-1": x_ref,
        "spectrum_1": y_ref,
        "spectrum_2_aligned_raw": y_sub_aligned,
        "spectrum_2_after_savgol": y_sub_prepared,
        "spectrum_2_scaled": y_sub_scaled,
        "difference": y_diff,
        "difference_baseline_arpls": baseline if baseline is not None else np.nan,
        "difference_minus_baseline_arpls": (
            y_diff_corrected if y_diff_corrected is not None else np.nan
        ),
    }
)

st.download_button(
    "Download difference as CSV",
    data=out_df.to_csv(index=False).encode("utf-8"),
    file_name="raman_difference.csv",
    mime="text/csv",
)

ref_header = load_header(ref_name)
if use_arpls and y_diff_corrected is not None:
    lines = []
    for x_val, y_val in zip(x_ref, y_diff_corrected):
        x_str = np.format_float_positional(float(x_val), trim="-")
        y_str = np.format_float_positional(float(y_val), trim="-")
        lines.append(f"{x_str}\t{y_str}\n")

    txt_payload = ref_header
    if txt_payload and not txt_payload.endswith("\n"):
        txt_payload += "\n"
    txt_payload += "".join(lines)

    st.download_button(
        "Download final corrected spectrum as TXT (header from Spectrum 1)",
        data=txt_payload.encode("latin-1", errors="replace"),
        file_name=f"{Path(ref_name).stem}_difference_arpls.txt",
        mime="text/plain",
    )
else:
    st.download_button(
        "Download final corrected spectrum as TXT (header from Spectrum 1)",
        data=b"",
        file_name=f"{Path(ref_name).stem}_difference_arpls.txt",
        mime="text/plain",
        disabled=True,
        help="Enable arPLS to export the final corrected spectrum.",
    )

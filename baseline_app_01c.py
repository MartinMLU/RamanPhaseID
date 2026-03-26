# baseline_app.py
from __future__ import annotations
import io
import re
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# SciPy: Sparse-Löser + stabile Sigmoid-Funktion
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.special import expit  # numerisch stabile Sigmoid-Funktion
except Exception as exc:
    raise RuntimeError(
        "SciPy wird benötigt (scipy.sparse & scipy.special). "
        "Bitte installieren: conda install scipy oder pip install scipy"
    ) from exc


# ─────────────────────────── Helpers: Parsing & Header ───────────────────────────

_NUM_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)(?:[,\t; ]+)([+-]?\d+(?:[.,]\d+)?)\s*$")

def _split_header_data(text: str):
    """
    Zerlegt eine ASCII-Datei in (header_lines, data_lines, delimiter_hint).
    Erkennt erste Zeilen mit 2 numerischen Spalten als Daten, den Rest als Header.
    """
    lines = text.splitlines()
    header, data = [], []
    delim_hint = None
    in_data = False
    for ln in lines:
        m = _NUM_RE.match(ln)
        if m:
            if not in_data:
                # Trennzeichen heuristisch ermitteln
                if "," in ln and "." not in ln:
                    delim_hint = re.findall(r"[,\t; ]", ln)[0]
                else:
                    parts = re.split(r"([,\t; ]+)", ln.strip())
                    delim_hint = parts[1] if len(parts) > 2 else "\t"
                in_data = True
            data.append(ln)
        else:
            header.append(ln)
    if delim_hint is None:
        delim_hint = "\t"
    return header, data, delim_hint


def _parse_xy_from_data_lines(data_lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for ln in data_lines:
        m = _NUM_RE.match(ln)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        xs.append(float(a.replace(",", ".")))
        ys.append(float(b.replace(",", ".")))
    return np.array(xs, float), np.array(ys, float)


def _format_number(val: float, decimals: int) -> str:
    return f"{val:.{decimals}f}"


def _rebuild_file_bytes(
    header_lines: list[str],
    x: np.ndarray,
    y_new: np.ndarray,
    *,
    decimals: int = 6,
    delimiter: str = "\t",
    keep_header_exact: bool = True,
    extra_note: str | None = None,
) -> bytes:
    """
    Baut eine Ausgabedatei mit identischem Header (optional plus Notiz) und 2 Spalten (x, y_new).
    """
    out = io.StringIO()
    if keep_header_exact:
        for ln in header_lines:
            out.write(ln + "\n")
    else:
        for ln in header_lines:
            out.write(ln + "\n")
        if extra_note:
            out.write(f"# {extra_note}\n")
    for xi, yi in zip(x, y_new):
        out.write(_format_number(xi, decimals) + delimiter + _format_number(yi, decimals) + "\n")
    return out.getvalue().encode("utf-8")


# ─────────────────────────── Baselines: IAsLS & arPLS ───────────────────────────

def baseline_iasls(
    y: np.ndarray,
    lam: float = 1e4,
    p: float = 0.01,
    niter: int = 20,
    lam1: float = 1e2,
) -> np.ndarray:
    """
    IAsLS (improved AsLS): AsLS mit zusätzlicher 1.-Ableitungs-Strafe.
    lam: 2.-Ableitungs-Strafe (Glätte)
    lam1: 1.-Ableitungs-Strafe (zusätzliche Baseline-Stabilisierung)
    p: Asymmetrie (klein -> Peaks stärker unterdrückt), typ. 0.001–0.05
    niter: 10–30 üblich
    """
    y = np.asarray(y, float)
    L = y.size
    if L < 5:
        return np.zeros_like(y)
    d2 = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2)).T
    h2 = lam * (d2.T @ d2)
    d1 = sp.diags([-1, 1], [0, 1], shape=(L - 1, L))
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        r = y - z
        q = np.abs(np.diff(r))
        if q.size == 0:
            q = np.ones(max(1, L - 1), dtype=float)
        q = np.asarray(q, dtype=float)
        q[~np.isfinite(q)] = 0.0
        qmax = float(np.max(q)) if q.size else 0.0
        if qmax > 0.0:
            q = q / qmax
        q += 1e-6
        h1 = lam1 * (d1.T @ sp.diags(q, 0) @ d1)
        W = sp.diags(w, 0)
        z = spla.spsolve(W + h2 + h1, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def baseline_arpls(y: np.ndarray, lam: float = 1e4, itermax: int = 50, tol: float = 1e-3) -> np.ndarray:
    """
    arPLS nach Baek et al., Analyst, 2015.
    lam: Glättung (größer → glatter)
    itermax: max. Iterationen
    tol: relative Gewichtsänderung als Abbruchkriterium
    (numerisch stabil dank scipy.special.expit)
    """
    y = np.asarray(y, float)
    L = y.size
    if L < 5:
        return np.zeros_like(y)
    D = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2)).T
    H = lam * (D.T @ D)

    w = np.ones(L)
    for _ in range(itermax):
        W = sp.diags(w, 0)
        z = spla.spsolve(W + H, w * y)
        d = y - z

        dn = d[d < 0]
        if dn.size == 0:
            break

        m = float(dn.mean())
        s = float(dn.std())
        if not np.isfinite(s) or s < 1e-12:
            s = 1e-12

        # Stabil: Sigmoid via expit; äquivalent zu 1/(1+exp(...)), aber ohne Overflow
        # arPLS weighting (Baek et al., Analyst 2015): use (2*s - m), not (m + 2*s)
        zlog = 2.0 * (d - (2.0 * s - m)) / s
        w_new = expit(-zlog)

        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < tol:
            w = w_new
            break
        w = w_new
    return z


# ─────────────────────────── Interne Auto-Skalierung (unsichtbar) ───────────────────────────

def _autoscale_prepare(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Interne Stabilisierung: y_norm = (y - offset) / scale.
    Offset & scale werden später rückgerechnet → Anzeige/Export bleiben in Original-Einheiten.
    """
    y = np.asarray(y, float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return y, 0.0, 1.0
    ymin = float(np.nanmin(y[finite]))
    ymax = float(np.nanmax(y[finite]))
    scale = max(ymax - ymin, 1e-12)
    offset = ymin
    y_norm = (y - offset) / scale
    return y_norm, offset, scale


# ─────────────────────────── Streamlit UI ───────────────────────────

st.set_page_config(page_title="Baseline subtractor (arPLS / IAsLS)", layout="wide")
st.title("🧹 Baseline subtraction for Raman spectra (arPLS / IAsLS)")

with st.sidebar:
    st.header("1) Datei laden")
    up = st.file_uploader("Messspektrum (.txt / .csv)", type=["txt", "csv"])

    st.header("2) Methode & Parameter")
    method = st.radio(
        "Baseline-Methode",
        ["arPLS", "IAsLS"],
        index=0,
        help="arPLS ist oft robuster; IAsLS ist die verbesserte AsLS-Variante."
    )
    lam_exp = st.slider("λ (10^x)", min_value=0, max_value=9, value=4, step=1,
                        help="Baseline-Steifigkeit (10^x). Niedrige Werte folgen gekrümmtem Untergrund besser; hohe Werte werden schnell fast linear.")
    lam = 10.0 ** lam_exp

    if method == "arPLS":
        itermax = st.slider("max. Iterationen", 5, 200, 50, step=1)
        tol_choice = st.select_slider(
            "Toleranz (Abbruch)", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3
        )
        params = {"lam": lam, "itermax": int(itermax), "tol": float(tol_choice)}
    else:
        p = st.slider("Asymmetrie p", 0.000, 0.200, 0.010, step=0.001,
                      help="Kleinere Werte unterdrücken Peaks stärker (typ. 0.001–0.05).")
        niter = st.slider("Iterationen", 1, 80, 20, step=1)
        lam1_exp = st.slider(
            "λ1 (IAsLS, 10^x)", min_value=-2, max_value=6, value=2, step=1,
            help="Zusätzliche 1.-Ableitungs-Strafe. Höher = stärkere Baseline-Glättung."
        )
        params = {"lam": lam, "p": float(p), "niter": int(niter), "lam1": float(10.0 ** lam1_exp)}

    st.header("3) Anzeige")
    show_raw = st.checkbox("Rohsignal anzeigen", value=True)
    show_baseline = st.checkbox("Baseline anzeigen", value=True)
    show_corrected = st.checkbox("Korrigiertes Signal anzeigen", value=True)
    decimals = st.slider("Dezimalstellen (Export)", 0, 10, 6, step=1)

    st.header("4) Export-Optionen")
    keep_header = st.checkbox("Header exakt beibehalten (empfohlen)", value=True)
    add_note = st.checkbox("Zusätzliche Notiz in Header schreiben", value=False, disabled=keep_header)
    note_text = st.text_input(
        "Notiz-Text (optional)", value="Baseline subtracted by arPLS/IAsLS",
        disabled=(keep_header or not add_note)
    )

# Keine Datei geladen?
if not up:
    st.info("Bitte eine Datei hochladen.")
    st.stop()

# Datei lesen & parsen
content = up.read().decode("utf-8", errors="ignore")
header_lines, data_lines, delimiter_hint = _split_header_data(content)
if len(data_lines) == 0:
    st.error("Keine Datenzeilen erkannt (zwei Spalten mit Zahlen erwartet). Prüfe Datei/Trennzeichen.")
    st.stop()

x, y = _parse_xy_from_data_lines(data_lines)
if x.size < 5:
    st.error("Zu wenige Datenpunkte (<5).")
    st.stop()

# Anzeige-Range (nur Zoom)
xmin, xmax = float(np.min(x)), float(np.max(x))
step_guess = float((xmax - xmin) / 1000.0) if xmax > xmin else 1.0
st.write("")  # kleiner Puffer
rng = st.slider(
    "Anzeigebereich (cm⁻¹)",
    min_value=float(xmin),
    max_value=float(xmax),
    value=(float(xmin), float(xmax)),
    step=step_guess
)

# ── Interne Auto-Skalierung (immer aktiv, keine UI)
use_autoscale = True
if use_autoscale:
    y_work, offset, scale = _autoscale_prepare(y)
else:
    y_work, offset, scale = y, 0.0, 1.0  # (hier nie genutzt)

# ── Baseline auf intern skaliertem Signal berechnen
if method == "arPLS":
    z_work = baseline_arpls(y_work, **params)
    meth_label = f"arPLS (λ=1e{int(np.log10(params['lam']))}, iter≤{params['itermax']}, tol={params['tol']})"
else:
    z_work = baseline_iasls(y_work, **params)
    meth_label = f"IAsLS (λ=1e{int(np.log10(params['lam']))}, λ1=1e{int(np.log10(params['lam1']))}, p={params['p']:.3f}, iters={params['niter']})"

# ── Baseline in Original-Einheiten rückskalieren; Korrektur bilden
z = z_work * scale + offset
y_corr = y - z

# Plot
mask = (x >= rng[0]) & (x <= rng[1])
fig, ax = plt.subplots(figsize=(11, 4.6))
if show_raw:
    ax.plot(x[mask], y[mask], label="raw", linewidth=1.0)
if show_baseline:
    ax.plot(x[mask], z[mask], label=f"baseline · {meth_label}", linewidth=1.0)
if show_corrected:
    ax.plot(x[mask], y_corr[mask], label="corrected = raw − baseline", linewidth=1.0)
ax.set_xlabel("Raman shift (cm⁻¹)")
ax.set_ylabel("Intensity (a.u.)")
ax.legend(loc="best")
ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)

# Downloads
st.subheader("💾 Export")
export_note = (note_text if (not keep_header and add_note and note_text.strip()) else None)

col_a, col_b, _ = st.columns([1, 1, 2])
with col_a:
    st.download_button(
        "⬇️ Spektrum (baseline-korrigiert)",
        data=_rebuild_file_bytes(
            header_lines, x, y_corr,
            decimals=int(decimals), delimiter=delimiter_hint,
            keep_header_exact=bool(keep_header), extra_note=export_note
        ),
        file_name=(up.name.rsplit(".", 1)[0] + "_baseline_corrected.txt"),
        mime="text/plain",
        use_container_width=True,
    )
with col_b:
    st.download_button(
        "⬇️ Baseline (nur Untergrund)",
        data=_rebuild_file_bytes(
            header_lines, x, z,
            decimals=int(decimals), delimiter=delimiter_hint,
            keep_header_exact=bool(keep_header), extra_note=export_note
        ),
        file_name=(up.name.rsplit(".", 1)[0] + "_baseline_only.txt"),
        mime="text/plain",
        use_container_width=True,
    )

st.caption("Hinweis: Header bleibt unverändert, sofern gewählt. Trennzeichen wird aus der Datei übernommen.")

"""
Raman-Matcher – Streamlit-GUI + CLI
Dual-DB mit wählbarer Messungs-Preprocessing-Variante: vergleicht Messung
(**BC oder RAW**) gegen **DB-RAW & DB-BC**.
Scoring nutzt standardmäßig **20% Gradient-Similarity** (baseline-invariant) + 80% Form.
Integriert Baseline-Parameter aus der separaten Baseline-App (IAsLS/arPLS).
"""

from __future__ import annotations
import argparse, os, sys, json, math, shutil, hashlib, io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re  # Regex + Unicode-Maps

import matplotlib
if os.getenv("DISPLAY", "") == "" and os.getenv("MPLBACKEND", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading, time  # (für Shutdown)

import numpy as np
import raman_core as rc
import streamlit as st

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.special import expit
    from scipy.signal import find_peaks
    HAVE_SCIPY_BASELINE = True
except Exception:
    HAVE_SCIPY_BASELINE = False


# ────────────────────────────────────────────────────────────────────────────
# Unicode-Maps & Formel-Helfer
SUPERSCRIPT_MAP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
SUBSCRIPT_MAP   = str.maketrans("0123456789",   "₀₁₂₃₄₅₆₇₈₉")

def _format_formula(raw: str) -> str:
    s = raw.replace("&#183;", "·").replace("&middot;", "·")
    s = s.replace("_", "")
    s = re.sub(r"\^(.*?)\^", lambda m: m.group(1).translate(SUPERSCRIPT_MAP), s)
    out, after_dot = [], False
    for ch in s:
        if ch == "·":
            after_dot = True
            out.append(ch)
        elif ch.isdigit():
            out.append(ch if after_dot else ch.translate(SUBSCRIPT_MAP))
        else:
            out.append(ch)
            if not ch.isdigit():
                after_dot = False
    return "".join(out)


# ────────────────────────────────────────────────────────────────────────────
# Pfade & Parameter
BASE_DIR               = Path(__file__).resolve().parent
DB_ROOT                = BASE_DIR / "databases"

def _pick_existing_dir(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

DEFAULT_OWN_DB_DIR   = _pick_existing_dir(
    BASE_DIR / "CleanDB",          # legacy layout
    DB_ROOT / "CleanDB",           # optional nested clean DB
    DB_ROOT / "OWN",               # current workspace layout
)
DEFAULT_ROD_DIR        = _pick_existing_dir(BASE_DIR / "ROD",      DB_ROOT / "ROD")
DEFAULT_RRUFF_DIR      = _pick_existing_dir(
    BASE_DIR / "RRUFF",            # legacy flat location
    DB_ROOT / "RRUFF",             # current workspace layout with subfolders
)

MATCH_FOLDERS_STANDARD = (
    DEFAULT_OWN_DB_DIR,
    DEFAULT_ROD_DIR,
    DEFAULT_RRUFF_DIR,
)
MATCH_FOLDERS_ULTRA = MATCH_FOLDERS_STANDARD

# Precompute-Ziel
PRECOMP_ROOT = BASE_DIR / "precomputed"
PRECOMP_MAX_WORKERS_DEFAULT = 3
PRECOMP_PAIR_WORKER_BUDGET_DEFAULT = 6

# Grid-Profile
CLEAN_GRID = {"min": 60, "max": 1900, "step": 1}
ULTRA_GRID = {"min": 60, "max": 4000, "step": 1}

DEFAULT_TOP_N = 60
MAX_OVERLAY_MINERALS = 12   # max. gleichzeitig eingegebene Mineralnamen
MATCH_SELECTION_VERSION = 4
TOP_PER_MINERAL_CAP = 5
PCS_MINERAL_SLOT_CAP = 12
PCS_MINERAL_MIN_PCS = 0.52
PCS_MINERAL_MIN_SIM = 0.50

# Scoring
GRAD_WEIGHT = 0.20  # 20% Gradient-Similarity
PCS_F1_WEIGHT = 0.75
PCS_PEAK_TOL = 5
FINAL_SIM_WEIGHT = 0.88

# Formel-Parser
ELEMENT_PARSER_VERSION = 1
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


def _resolve_precompute_workers() -> int:
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, min(PRECOMP_MAX_WORKERS_DEFAULT, cpu_count))
    raw = os.getenv("RAMAN_PRECOMP_WORKERS", "").strip()
    if not raw:
        return default_workers
    try:
        requested = int(raw)
    except Exception:
        return default_workers
    return max(1, min(requested, cpu_count))


def _resolve_precompute_pair_workers() -> int:
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, min(PRECOMP_PAIR_WORKER_BUDGET_DEFAULT, cpu_count))
    raw = os.getenv("RAMAN_PRECOMP_PAIR_WORKERS", "").strip()
    if not raw:
        return default_workers
    try:
        requested = int(raw)
    except Exception:
        return default_workers
    return max(1, min(requested, cpu_count))


def _extract_formula_elements(formula: str) -> list[str]:
    if not formula or formula.startswith("?"):
        return []
    els = {tok for tok in _ELEMENT_RE.findall(formula)}
    return sorted(els)


# ────────────────────────────────────────────────────────────────────────────
# Baseline-Helper (aus Baseline-App integriert)

_NUM_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)(?:[,\t; ]+)([+-]?\d+(?:[.,]\d+)?)\s*$")

def _split_header_data(text: str):
    lines = text.splitlines()
    header, data = [], []
    delim_hint = None
    in_data = False
    for ln in lines:
        m = _NUM_RE.match(ln)
        if m:
            if not in_data:
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
        xs.append(float(m.group(1).replace(",", ".")))
        ys.append(float(m.group(2).replace(",", ".")))
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


def _autoscale_prepare(y: np.ndarray) -> tuple[np.ndarray, float, float]:
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


def _baseline_iasls(
    y: np.ndarray,
    lam: float = 1e4,
    p: float = 0.01,
    niter: int = 20,
    lam1: float = 1e2,
) -> np.ndarray:
    y = np.asarray(y, float)
    L = y.size
    if L < 5 or not HAVE_SCIPY_BASELINE:
        return rc._baseline_als(y)
    d2 = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2)).T
    h2 = (lam * (d2.T @ d2)).tocsc()
    d1 = sp.diags([-1, 1], [0, 1], shape=(L - 1, L)).tocsc()
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        # IAsLS: additional first-derivative penalty weighted by current residual roughness
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
        h1 = (lam1 * (d1.T @ sp.diags(q, 0) @ d1)).tocsc()
        W = sp.diags(w, 0)
        z = spla.spsolve((W + h2 + h1).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def _baseline_arpls(y: np.ndarray, lam: float = 1e4, itermax: int = 50, tol: float = 1e-3) -> np.ndarray:
    y = np.asarray(y, float)
    L = y.size
    if L < 5:
        return np.zeros_like(y)
    if not HAVE_SCIPY_BASELINE:
        return rc._baseline_als(y)

    D = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2)).T
    H = (lam * (D.T @ D)).tocsc()
    w = np.ones(L)

    for _ in range(itermax):
        W = sp.diags(w, 0)
        z = spla.spsolve((W + H).tocsc(), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0:
            break
        m = float(dn.mean())
        s = float(dn.std())
        if not np.isfinite(s) or s < 1e-12:
            s = 1e-12
        # arPLS weighting (Baek et al., Analyst 2015): use (2*s - m), not (m + 2*s)
        zlog = 2.0 * (d - (2.0 * s - m)) / s
        w_new = expit(-zlog)
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < tol:
            w = w_new
            break
        w = w_new
    return z


def _default_baseline_cfg() -> dict:
    lam_exp = 4
    lam1_exp = 2
    return {
        "method": "arPLS",
        "lam_exp": lam_exp,
        "lam": 10.0 ** lam_exp,
        "itermax": 50,
        "tol": 1e-3,
        "p": 0.010,
        "niter": 20,
        "lam1_exp": lam1_exp,
        "lam1": 10.0 ** lam1_exp,
        "autoscale": True,
        "db_strength": 1.00,
    }


def _fixed_db_baseline_cfg() -> dict:
    """Fixed DB baseline settings used for DB-BC cache/preprocessing."""
    return {
        "method": "arPLS",
        "lam_exp": 4,
        "lam": 1e4,
        "itermax": 50,
        "tol": 1e-3,
        "p": 0.010,
        "niter": 20,
        "lam1_exp": 2,
        "lam1": 1e2,
        "autoscale": True,
        "db_strength": 1.00,
    }


def _baseline_cfg_payload(cfg: dict) -> dict:
    method = str(cfg.get("method", "arPLS"))
    payload = {
        "v": 3,
        "method": method,
        "lam_exp": int(cfg.get("lam_exp", 4)),
        "lam": float(cfg.get("lam", 1e4)),
        "autoscale": bool(cfg.get("autoscale", True)),
        "db_strength": round(float(np.clip(cfg.get("db_strength", 1.00), 0.0, 1.0)), 4),
        "have_scipy": bool(HAVE_SCIPY_BASELINE),
    }
    if method == "arPLS":
        payload["itermax"] = int(cfg.get("itermax", 50))
        payload["tol"] = float(cfg.get("tol", 1e-3))
    elif method == "ALS":
        payload["p"] = float(cfg.get("p", 0.010))
        payload["niter"] = int(cfg.get("niter", 20))
        payload["lam1_exp"] = int(cfg.get("lam1_exp", 2))
        payload["lam1"] = float(cfg.get("lam1", 1e2))
    return payload


def _baseline_cfg_token(cfg: dict) -> str:
    payload = _baseline_cfg_payload(cfg)
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _baseline_label(cfg: dict) -> str:
    method = str(cfg.get("method", "arPLS"))
    lam = float(cfg.get("lam", 1e4))
    if method == "arPLS":
        return f"arPLS (λ=1e{int(round(np.log10(max(lam, 1e-12))))}, iter≤{int(cfg.get('itermax', 50))}, tol={float(cfg.get('tol', 1e-3))})"
    if method == "ALS":
        lam1 = float(cfg.get("lam1", 1e2))
        return (
            f"IAsLS (λ=1e{int(round(np.log10(max(lam, 1e-12))))}, "
            f"λ1=1e{int(round(np.log10(max(lam1, 1e-12))))}, "
            f"p={float(cfg.get('p', 0.010)):.3f}, iters={int(cfg.get('niter', 20))})"
        )
    return "none (RAW)"


def _compute_baseline(y: np.ndarray, baseline_cfg: dict) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return np.zeros_like(y)

    method = str(baseline_cfg.get("method", "arPLS"))
    if method not in ("arPLS", "ALS"):
        return np.zeros_like(y)

    use_autoscale = bool(baseline_cfg.get("autoscale", True))
    if use_autoscale:
        y_work, offset, scale = _autoscale_prepare(y)
    else:
        y_work, offset, scale = y, 0.0, 1.0

    lam = float(baseline_cfg.get("lam", 1e4))
    if method == "arPLS":
        z_work = _baseline_arpls(
            y_work,
            lam=lam,
            itermax=int(baseline_cfg.get("itermax", 50)),
            tol=float(baseline_cfg.get("tol", 1e-3)),
        )
    else:
        z_work = _baseline_iasls(
            y_work,
            lam=lam,
            p=float(baseline_cfg.get("p", 0.010)),
            niter=int(baseline_cfg.get("niter", 20)),
            lam1=float(baseline_cfg.get("lam1", 1e2)),
        )
    return z_work * scale + offset


def _apply_db_baseline(y: np.ndarray, baseline_cfg: dict) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    method = str(baseline_cfg.get("method", "arPLS"))
    if method not in ("arPLS", "ALS"):
        return y_arr.copy()
    strength = float(np.clip(float(baseline_cfg.get("db_strength", 1.00)), 0.0, 1.0))
    if strength <= 0.0:
        return y_arr.copy()
    baseline = _compute_baseline(y_arr, baseline_cfg)
    if strength >= 1.0:
        return y_arr - baseline
    # Blend RAW and BC for DB spectra to avoid over-subtraction artifacts near sharp peaks.
    return y_arr - (strength * baseline)


def _default_smoothing_cfg() -> dict:
    return {
        "enabled": True,
        "window": 11,
        "poly": 3,
    }


def _smoothing_cfg_payload(cfg: dict) -> dict:
    return {
        "v": 1,
        "enabled": bool(cfg.get("enabled", True)),
        "window": int(cfg.get("window", 11)),
        "poly": int(cfg.get("poly", 3)),
    }


def _smoothing_cfg_token(cfg: dict) -> str:
    payload = _smoothing_cfg_payload(cfg)
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _smoothing_label(cfg: dict) -> str:
    if not bool(cfg.get("enabled", True)):
        return "off"
    return f"Savitzky-Golay (window={int(cfg.get('window', 11))}, poly={int(cfg.get('poly', 3))})"


def _default_white_ref_cfg() -> dict:
    return {
        "enabled": False,
        "scale": 1.0,
        "ref_sha1": "",
    }


def _white_ref_cfg_payload(cfg: dict) -> dict:
    return {
        "v": 1,
        "enabled": bool(cfg.get("enabled", False)),
        "scale": round(float(cfg.get("scale", 1.0)), 6),
        "ref_sha1": str(cfg.get("ref_sha1", "")),
    }


def _white_ref_cfg_token(cfg: dict) -> str:
    payload = _white_ref_cfg_payload(cfg)
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _white_ref_label(cfg: dict) -> str:
    if not bool(cfg.get("enabled", False)):
        return "off"
    return f"on (scale={float(cfg.get('scale', 1.0)):.3f})"


def _align_reference_to_target(
    target_x: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> np.ndarray:
    tgt = np.asarray(target_x, dtype=float)
    src_x = np.asarray(ref_x, dtype=float)
    src_y = np.asarray(ref_y, dtype=float)
    if tgt.size == 0:
        return np.array([], dtype=float)

    finite = np.isfinite(src_x) & np.isfinite(src_y)
    src_x = src_x[finite]
    src_y = src_y[finite]
    if src_x.size < 2:
        return np.zeros_like(tgt, dtype=float)

    order = np.argsort(src_x)
    src_x = src_x[order]
    src_y = src_y[order]

    uniq_x, uniq_idx = np.unique(src_x, return_index=True)
    uniq_y = src_y[uniq_idx]
    if uniq_x.size < 2:
        return np.zeros_like(tgt, dtype=float)

    return np.interp(tgt, uniq_x, uniq_y, left=0.0, right=0.0)


def _sanitize_savgol_params(n: int, window: int, poly: int) -> tuple[int, int] | None:
    if n < 3:
        return None
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    max_w = n if (n % 2 == 1) else (n - 1)
    if max_w < 3:
        return None
    if w > max_w:
        w = max_w
    p = max(0, int(poly))
    if p >= w:
        p = w - 1
    return w, p


def _apply_smoothing(y: np.ndarray, smoothing_cfg: dict | None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    if smoothing_cfg is None:
        return rc._smooth(y)
    if not bool(smoothing_cfg.get("enabled", True)):
        return y.copy()
    params = _sanitize_savgol_params(
        y.size,
        int(smoothing_cfg.get("window", 11)),
        int(smoothing_cfg.get("poly", 3)),
    )
    if params is None:
        return y.copy()
    window, poly = params
    try:
        return rc._smooth(y, window=window, poly=poly)
    except Exception:
        return y.copy()


# ────────────────────────────────────────────────────────────────────────────
# Signatur der DB-Inhalte (zur Invalidierung)

def _compute_db_signature(folders: list[Path]) -> str:
    h = hashlib.sha1()
    for folder in folders:
        if not folder.exists():
            h.update(f"!MISS|{folder}".encode())
            continue
        candidates = list(folder.rglob("*.rod")) + list(folder.rglob("*.txt"))
        for p in sorted(candidates):
            try:
                stt = p.stat()
                rel = p.relative_to(folder)
                h.update(f"{folder.name}/{rel}|{stt.st_size}|{int(stt.st_mtime)}".encode())
            except Exception:
                h.update(f"!ERR|{p}".encode())
    return h.hexdigest()


def _compute_signature_with_grid(folders: list[Path], grid_min: int, grid_max: int, grid_step: int) -> str:
    import hashlib
    base = _compute_db_signature(folders)
    h = hashlib.sha1()
    h.update(f"{base}|g:{grid_min}-{grid_max}-{grid_step}|p:{ELEMENT_PARSER_VERSION}".encode())
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Processing-Helfer

def _normalize_to_peak(y: np.ndarray) -> np.ndarray:
    """Scale by positive peak (keeps negative artifacts visible)."""
    arr = np.asarray(y, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    peak = float(np.nanmax(arr[finite]))
    if not np.isfinite(peak) or peak <= 1e-12:
        peak = float(np.nanmax(np.abs(arr[finite])))
    if not np.isfinite(peak) or peak <= 1e-12:
        peak = 1.0

    out[finite] = arr[finite] / peak
    return out


def _prepare_measurement_signal(
    y: np.ndarray,
    *,
    apply_baseline: bool,
    baseline_cfg: dict,
    smoothing_cfg: dict | None = None,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    y_proc = y_arr - _compute_baseline(y_arr, baseline_cfg) if apply_baseline else y_arr
    return _apply_smoothing(y_proc, smoothing_cfg)


def _prepare_db_signal_on_target_grid(
    db_x: np.ndarray,
    db_y: np.ndarray,
    target_x: np.ndarray,
    *,
    apply_baseline_db: bool,
    baseline_cfg: dict,
    smoothing_cfg_db: dict | None = None,
) -> np.ndarray:
    y_in = _apply_db_baseline(db_y, baseline_cfg) if apply_baseline_db else np.asarray(db_y, dtype=float)
    db_res = rc._resample(db_x, y_in, target_x)
    return _apply_smoothing(db_res, smoothing_cfg_db)


def _process_measurement(
    y: np.ndarray,
    *,
    apply_baseline: bool,
    baseline_cfg: dict,
    smoothing_cfg: dict | None = None,
) -> np.ndarray:
    y_proc = _prepare_measurement_signal(
        y,
        apply_baseline=apply_baseline,
        baseline_cfg=baseline_cfg,
        smoothing_cfg=smoothing_cfg,
    )
    return rc._normalize(y_proc)


def _process_db(y: np.ndarray, *, apply_baseline_db: bool, baseline_cfg: dict) -> np.ndarray:
    y_proc = _apply_db_baseline(y, baseline_cfg) if apply_baseline_db else np.asarray(y, dtype=float)
    y_proc = _apply_smoothing(y_proc, None)
    return rc._normalize(y_proc)


def _process_db_on_target_grid(
    db_x: np.ndarray,
    db_y: np.ndarray,
    target_x: np.ndarray,
    *,
    apply_baseline_db: bool,
    baseline_cfg: dict,
    smoothing_cfg_db: dict | None = None,
) -> np.ndarray:
    y_proc = _prepare_db_signal_on_target_grid(
        db_x,
        db_y,
        target_x,
        apply_baseline_db=apply_baseline_db,
        baseline_cfg=baseline_cfg,
        smoothing_cfg_db=smoothing_cfg_db,
    )
    return rc._normalize(y_proc)


# ────────────────────────────────────────────────────────────────────────────
# Overlay-Helfer (ohne Extrapolation)

def _plot_overlay(
    meas_x: np.ndarray,
    meas_y: np.ndarray,
    db_x: np.ndarray,
    db_y: np.ndarray,
    label: str,
    baseline_cfg: dict,
    *,
    baseline_cfg_db: dict | None = None,
    ax=None,
    alpha=0.7,
    lw=1.2,
    apply_baseline_meas: bool = True,
    apply_baseline_db: bool = False,
    smoothing_cfg_meas: dict | None = None,
):
    """Overlay OHNE Extrapolation; DB nur in eigenem x-Bereich, Treatments wie beim Matching."""
    meas_sig = _prepare_measurement_signal(
        meas_y,
        apply_baseline=apply_baseline_meas,
        baseline_cfg=baseline_cfg,
        smoothing_cfg=smoothing_cfg_meas,
    )
    meas_proc = _normalize_to_peak(meas_sig)
    ax = ax or plt.gca()
    meas_tag = "meas (BC)" if apply_baseline_meas else "meas (RAW)"
    ax.plot(meas_x, meas_proc, label=meas_tag, lw=lw)

    lo, hi = float(np.min(db_x)), float(np.max(db_x))
    mask = (meas_x >= lo) & (meas_x <= hi)
    if mask.sum() < 2:
        ax.text(0.02, 0.94, f"{label}: no overlap", transform=ax.transAxes, fontsize="small", va="top")
        return

    x_clip = meas_x[mask]
    db_cfg = baseline_cfg if baseline_cfg_db is None else baseline_cfg_db
    db_sig = _prepare_db_signal_on_target_grid(
        db_x,
        db_y,
        x_clip,
        apply_baseline_db=apply_baseline_db,
        baseline_cfg=db_cfg,
        smoothing_cfg_db={"enabled": False},
    )
    db_proc = _normalize_to_peak(db_sig)
    db_tag = "DB (BC)" if apply_baseline_db else "DB (RAW)"
    ax.plot(x_clip, db_proc, label=f"{label} · {db_tag}", lw=lw, alpha=alpha)
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("normalised intensity")
    ax.legend()


def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _delay_then_rerun(delay_seconds: float = 2.0):
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    _safe_rerun()


def _fig_to_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    return buf.getvalue()


def _normalize_plot_theme(theme: str | None) -> str:
    return "light" if str(theme).strip().lower() == "light" else "dark"


def _normalize_app_theme(theme: str | None) -> str:
    return "light" if str(theme).strip().lower() == "light" else "dark"


def _apply_app_theme(theme: str = "light") -> None:
    theme = _normalize_app_theme(theme)
    if theme == "light":
        app_bg = "#F4F7FB"
        sidebar_bg = "#E8EDF4"
        text = "#1F2933"
        button_bg = "#FFFFFF"
        button_text = "#1F2933"
        input_bg = "#FFFFFF"
        border = "#BAC5D3"
        accent = "#0B5FA5"
    else:
        app_bg = "#0E1117"
        sidebar_bg = "#161B22"
        text = "#E6EDF3"
        button_bg = "#1D2530"
        button_text = "#E6EDF3"
        input_bg = "#11161D"
        border = "#3B4652"
        accent = "#7DCBFF"

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {app_bg};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {sidebar_bg};
        }}
        [data-testid="stAppViewContainer"] * {{
            color: {text};
        }}
        :root {{
            --rm-primary: var(--primary-color, var(--st-primary-color, #ff4841));
            --rm-primary-pressed: #1F6A42;
            --rm-primary-pressed-border: #175136;
        }}
        .stButton > button {{
            background-color: {button_bg};
            color: {button_text};
            border: 1px solid {border};
        }}
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primary"] {{
            background-color: var(--rm-primary) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--rm-primary) !important;
            transition: background-color 0.2s ease, border-color 0.2s ease, filter 0.2s ease;
        }}
        .stDownloadButton > button {{
            background-color: {button_bg};
            color: {button_text};
            border: 1px solid {border};
        }}
        .stButton > button:hover {{
            border-color: {accent};
            color: {accent};
        }}
        .stDownloadButton > button:hover {{
            border-color: {accent};
            color: {accent};
        }}
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {{
            background-color: var(--rm-primary) !important;
            color: #FFFFFF !important;
            border-color: var(--rm-primary) !important;
            filter: brightness(1.03);
        }}
        .stButton > button[kind="primary"]:active,
        .stButton > button[data-testid="stBaseButton-primary"]:active,
        button[data-testid="stBaseButton-primary"]:active {{
            background-color: var(--rm-primary-pressed) !important;
            border-color: var(--rm-primary-pressed-border) !important;
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"]:focus,
        .stButton > button[data-testid="stBaseButton-primary"]:focus,
        button[data-testid="stBaseButton-primary"]:focus {{
            animation: primary_click_feedback 2s ease;
        }}
        @keyframes primary_click_feedback {{
            0%, 85% {{
                background-color: var(--rm-primary-pressed) !important;
                border-color: var(--rm-primary-pressed-border) !important;
                color: #FFFFFF !important;
            }}
            100% {{
                background-color: var(--rm-primary) !important;
                border-color: var(--rm-primary) !important;
                color: #FFFFFF !important;
            }}
        }}
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {{
            background-color: {input_bg};
            color: {text};
        }}
        [data-testid="stSidebar"] hr {{
            border-color: {border};
        }}
        [data-testid="stExpander"] details {{
            background-color: {button_bg} !important;
            border: 1px solid {border} !important;
            border-radius: 0.5rem;
        }}
        [data-testid="stExpander"] details summary {{
            background-color: {button_bg} !important;
            color: {text} !important;
        }}
        [data-testid="stExpander"] details summary * {{
            color: {text} !important;
        }}
        [data-testid="stExpanderDetails"] {{
            background-color: {button_bg} !important;
            color: {text} !important;
        }}
        [data-testid="stFileUploader"] {{
            background-color: transparent;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {button_bg} !important;
            border: 1px dashed {border} !important;
            color: {text} !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {text} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: 1px solid {border} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            border-color: {accent} !important;
            color: {accent} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_plot_style(fig, ax, theme: str = "dark") -> None:
    theme = _normalize_plot_theme(theme)
    if theme == "light":
        bg = "#FFFFFF"
        panel = "#FFFFFF"
        text = "#1F2933"
        spine = "#BFC7D0"
        grid = "#E6EBF1"
        palette = ["#0B5FA5", "#C84C09", "#2E7D32", "#7B1FA2", "#9A3E00", "#00838F", "#5D4037"]
    else:
        bg = "#0E1117"
        panel = "#161B22"
        text = "#E6EDF3"
        spine = "#3D4854"
        grid = "#2B3440"
        palette = ["#7DCBFF", "#FFB86C", "#8BE9A8", "#FF79C6", "#F1FA8C", "#50FAE3", "#FF6B6B"]

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(panel)
    for i, line in enumerate(ax.get_lines()):
        line.set_color(palette[i % len(palette)])
    for spn in ax.spines.values():
        spn.set_color(spine)
    ax.tick_params(
        axis="both",
        which="both",
        colors=text,
        direction="in",
        top=True,
        right=True,
    )
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)
    ax.grid(alpha=0.25, color=grid)
    for txt in ax.texts:
        txt.set_color(text)
    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor(panel)
        frame.set_edgecolor(spine)
        frame.set_alpha(0.85)
        for t in leg.get_texts():
            t.set_color(text)


def _set_intensity_number_visibility(ax, show_numbers: bool) -> None:
    ax.tick_params(axis="y", labelleft=bool(show_numbers))
    ax.yaxis.get_offset_text().set_visible(bool(show_numbers))


# ────────────────────────────────────────────────────────────────────────────
# Precompute: fixes Gitter + ANN (generisch für Clean & Ultra)

def _fixed_grid(grid_min: int, grid_max: int, grid_step: int) -> np.ndarray:
    return np.arange(grid_min, grid_max + grid_step, grid_step, dtype=np.float32)


def _precomp_dir(signature: str) -> Path:
    return PRECOMP_ROOT / signature


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _try_import_hnswlib():
    try:
        import hnswlib  # type: ignore
        return hnswlib
    except Exception:
        return None


def _prepare_db_vector_on_grid(
    db_x: np.ndarray,
    db_y: np.ndarray,
    grid: np.ndarray,
    grid_min: int,
    grid_step: int,
    grid_len: int,
    *,
    apply_baseline_db: bool = False,
    baseline_cfg: dict | None = None,
) -> tuple[np.ndarray, int, int, float]:
    """DB-Spektrum → fixes Gitter; optional Baseline-Correction."""
    cfg = baseline_cfg or _default_baseline_cfg()
    res = _process_db_on_target_grid(
        db_x,
        db_y,
        grid,
        apply_baseline_db=apply_baseline_db,
        baseline_cfg=cfg,
    )
    lo = float(np.min(db_x)); hi = float(np.max(db_x))
    start_idx = int(max(0, math.ceil((lo - grid_min) / grid_step)))
    end_idx   = int(min(grid_len - 1, math.floor((hi - grid_min) / grid_step)))
    l2 = float(np.linalg.norm(res))
    return res.astype(np.float32), start_idx, end_idx, l2


def _build_precompute_core(
    signature: str,
    folders: tuple[Path, ...],
    grid_min: int,
    grid_max: int,
    grid_step: int,
    *,
    apply_baseline_db: bool,
    baseline_cfg: dict,
    progress=None,
    workers: int | None = None,
):
    grid = _fixed_grid(grid_min, grid_max, grid_step)
    grid_len = grid.shape[0]

    out_dir = _precomp_dir(signature)
    _ensure_dir(out_dir)

    entries, _skipped = rc.load_reference_folders(folders)
    if not entries:
        raise RuntimeError("Precompute: no references found")

    worker_count = _resolve_precompute_workers() if workers is None else max(1, int(workers))

    N = len(entries)
    X_path = out_dir / "X.float32.npy"
    meta_path = out_dir / "meta.json"
    grid_path = out_dir / "grid.json"
    ann_path  = out_dir / "ann_hnsw.bin"

    X = np.memmap(X_path, mode="w+", dtype=np.float32, shape=(N, grid_len))

    def _process_one_entry(e: dict) -> tuple[np.ndarray, dict]:
        p = Path(e["path"])
        try:
            if p.suffix.lower() == ".txt":
                db_x, db_y = rc._parse_rruff(p)
            else:
                db_x, db_y = rc._parse_rod(p)

            v, start_idx, end_idx, l2 = _prepare_db_vector_on_grid(
                db_x, db_y, grid, grid_min, grid_step, grid_len,
                apply_baseline_db=apply_baseline_db, baseline_cfg=baseline_cfg
            )

            elements = _extract_formula_elements(e.get("formula", ""))
            row_meta = {
                "name": e["name"],
                "formula": e["formula"],
                "flag": e.get("flag", ""),
                "filename": e.get("filename", ""),
                "orig_filename": e.get("orig_filename", p.name),
                "path": str(p),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "l2": l2,
                "elements": elements,
                "has_formula": bool(elements),
                "parser_version": ELEMENT_PARSER_VERSION,
                "source": e.get("source", ""),
                "db_baseline": bool(apply_baseline_db),
            }
            return v, row_meta
        except Exception as exc:
            row_meta = {
                "name": "SKIPPED",
                "formula": "?",
                "flag": "",
                "filename": "",
                "orig_filename": e.get("orig_filename", p.name),
                "path": str(p),
                "start_idx": 0,
                "end_idx": -1,
                "l2": 0.0,
                "elements": [],
                "has_formula": False,
                "parser_version": ELEMENT_PARSER_VERSION,
                "error": str(exc),
                "db_baseline": bool(apply_baseline_db),
            }
            return np.zeros(grid_len, dtype=np.float32), row_meta

    meta: list[dict | None] = [None] * N

    if worker_count == 1 or N == 1:
        for i, e in enumerate(entries):
            v, row_meta = _process_one_entry(e)
            X[i, :] = v
            meta[i] = row_meta
            if progress:
                progress.progress((i + 1) / N, text=f"Precomputing… {i + 1}/{N}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_process_one_entry, e): i for i, e in enumerate(entries)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                v, row_meta = fut.result()
                X[i, :] = v
                meta[i] = row_meta
                done += 1
                if progress:
                    progress.progress(done / N, text=f"Precomputing… {done}/{N}")

    if any(m is None for m in meta):
        raise RuntimeError("Precompute: incomplete worker results")
    meta = [m for m in meta if m is not None]

    del X

    with grid_path.open("w", encoding="utf-8") as fh:
        json.dump({"min": grid_min, "max": grid_max, "step": grid_step, "len": grid_len}, fh)

    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)

    hnswlib = _try_import_hnswlib()
    if hnswlib is not None:
        X_m = np.memmap(X_path, mode="r", dtype=np.float32, shape=(len(meta), grid_len))
        valid_idx = [i for i, m in enumerate(meta) if m.get("l2", 0.0) > 0.0 and m.get("end_idx", -1) >= m.get("start_idx", 0)]
        if valid_idx:
            data = X_m[valid_idx, :]
            dim = data.shape[1]
            p = hnswlib.Index(space='cosine', dim=dim)
            p.init_index(max_elements=len(valid_idx), ef_construction=300, M=32)
            p.add_items(data, ids=np.array(valid_idx, dtype=np.int32))
            p.set_ef(200)
            p.save_index(str(ann_path))


def _build_precompute(
    signature: str,
    folders: tuple[Path, ...],
    grid_min: int,
    grid_max: int,
    grid_step: int,
    *,
    apply_baseline_db: bool,
    baseline_cfg: dict,
    progress=None,
    workers: int | None = None,
):
    return _build_precompute_core(
        signature,
        folders,
        grid_min,
        grid_max,
        grid_step,
        apply_baseline_db=apply_baseline_db,
        baseline_cfg=baseline_cfg,
        progress=progress,
        workers=workers,
    )


def _load_precompute(signature: str):
    out_dir = _precomp_dir(signature)
    X_path = out_dir / "X.float32.npy"
    meta_path = out_dir / "meta.json"
    grid_path = out_dir / "grid.json"
    ann_path  = out_dir / "ann_hnsw.bin"

    if not (X_path.exists() and meta_path.exists() and grid_path.exists()):
        return None

    with grid_path.open("r", encoding="utf-8") as fh:
        g = json.load(fh)
    grid_min, grid_max, grid_step, grid_len = g["min"], g["max"], g["step"], g["len"]

    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    if not meta or meta[0].get("parser_version") != ELEMENT_PARSER_VERSION:
        return None

    file_size = X_path.stat().st_size
    n_rows = file_size // (grid_len * 4)
    X = np.memmap(X_path, mode="r", dtype=np.float32, shape=(n_rows, grid_len))

    hnsw_index = None
    hnswlib = _try_import_hnswlib()
    if hnswlib is not None and ann_path.exists():
        dim = grid_len
        hnsw_index = hnswlib.Index(space='cosine', dim=dim)
        hnsw_index.load_index(str(ann_path))
        hnsw_index.set_ef(200)

    grid = _fixed_grid(grid_min, grid_max, grid_step)

    inv_idx = {}
    for i, m in enumerate(meta):
        for el in m.get("elements", []):
            inv_idx.setdefault(el, []).append(i)
    for k, arr in inv_idx.items():
        inv_idx[k] = np.array(arr, dtype=np.int32)

    return {
        "X": X,
        "meta": meta,
        "grid": grid,
        "grid_info": {"min": grid_min, "max": grid_max, "step": grid_step, "len": grid_len},
        "ann": hnsw_index,
        "dir": out_dir,
        "inv_elements": inv_idx,
    }


def _ensure_precompute(
    signature: str,
    folders: tuple[Path, ...],
    grid_min: int,
    grid_max: int,
    grid_step: int,
    *,
    apply_baseline_db: bool,
    baseline_cfg: dict,
    label_for_spinner: str,
    precompute_workers: int | None = None,
):
    pack = _load_precompute(signature)
    if pack is not None:
        return pack
    with st.spinner(label_for_spinner):
        prog = st.progress(0.0)
        _ensure_dir(_precomp_dir(signature))
        _build_precompute_core(
            signature, folders, grid_min, grid_max, grid_step,
            apply_baseline_db=apply_baseline_db,
            baseline_cfg=baseline_cfg,
            progress=prog,
            workers=precompute_workers,
        )
    pack = _load_precompute(signature)
    if pack is None:
        raise RuntimeError(f"Precompute could not be loaded after build: {signature}")
    return pack


def _ensure_precompute_pair(
    *,
    signature_raw: str,
    signature_bcb: str,
    folders: tuple[Path, ...],
    grid_min: int,
    grid_max: int,
    grid_step: int,
    baseline_cfg: dict,
) -> tuple[dict, dict]:
    pack_raw = _load_precompute(signature_raw)
    pack_bcb = _load_precompute(signature_bcb)

    need_raw = pack_raw is None
    need_bcb = pack_bcb is None
    if not need_raw and not need_bcb:
        return pack_raw, pack_bcb

    worker_budget = _resolve_precompute_pair_workers() if (need_raw and need_bcb) else _resolve_precompute_workers()
    if need_raw and need_bcb and worker_budget >= 2:
        workers_per_build = max(1, worker_budget // 2)
        with st.spinner("Building fixed-grid caches in parallel (DB RAW + DB baseline-corrected)…"):
            _ensure_dir(_precomp_dir(signature_raw))
            _ensure_dir(_precomp_dir(signature_bcb))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        _build_precompute_core,
                        signature_raw, folders, grid_min, grid_max, grid_step,
                        apply_baseline_db=False, baseline_cfg=baseline_cfg,
                        progress=None, workers=workers_per_build,
                    ),
                    pool.submit(
                        _build_precompute_core,
                        signature_bcb, folders, grid_min, grid_max, grid_step,
                        apply_baseline_db=True, baseline_cfg=baseline_cfg,
                        progress=None, workers=workers_per_build,
                    ),
                ]
                for fut in as_completed(futures):
                    fut.result()
        pack_raw = _load_precompute(signature_raw)
        pack_bcb = _load_precompute(signature_bcb)
    else:
        if need_raw:
            pack_raw = _ensure_precompute(
                signature_raw, folders, grid_min, grid_max, grid_step,
                apply_baseline_db=False,
                baseline_cfg=baseline_cfg,
                label_for_spinner="Building fixed-grid cache (DB RAW)…",
                precompute_workers=worker_budget,
            )
        if need_bcb:
            pack_bcb = _ensure_precompute(
                signature_bcb, folders, grid_min, grid_max, grid_step,
                apply_baseline_db=True,
                baseline_cfg=baseline_cfg,
                label_for_spinner="Building fixed-grid cache (DB baseline-corrected)…",
                precompute_workers=worker_budget,
            )

    if pack_raw is None or pack_bcb is None:
        raise RuntimeError("Precompute caches could not be loaded after build.")
    return pack_raw, pack_bcb


# ────────────────────────────────────────────────────────────────────────────
# Query-Vorbereitung (Messung → fixes Gitter)

def _prepare_query_vector(
    meas_x: np.ndarray,
    meas_y: np.ndarray,
    range_low: int,
    range_high: int,
    grid: np.ndarray,
    *,
    apply_baseline: bool = True,
    baseline_cfg: dict | None = None,
    smoothing_cfg: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    cfg = baseline_cfg or _default_baseline_cfg()
    meas_proc = _process_measurement(
        meas_y,
        apply_baseline=apply_baseline,
        baseline_cfg=cfg,
        smoothing_cfg=smoothing_cfg,
    )
    q = rc._resample(meas_x, meas_proc, grid).astype(np.float32)
    q[~np.isfinite(q)] = 0.0
    q_mask = (grid >= range_low) & (grid <= range_high)
    q[~q_mask] = 0.0
    q_l2 = float(np.linalg.norm(q))
    return q, q_l2, q_mask


# ────────────────────────────────────────────────────────────────────────────
# TopK Cosinus über Teilmenge (Element-Filter)

def _topk_cosine_subset(query: np.ndarray, X: np.memmap, meta: list[dict], subset_ids: np.ndarray, topk: int) -> list[int]:
    if subset_ids.size == 0:
        return []

    # Query säubern
    q = np.asarray(query, dtype=np.float32).copy()
    q[~np.isfinite(q)] = 0.0
    q_l2 = float(np.linalg.norm(q))

    # DB-L2 und Gültigkeitsmaske
    db_l2 = np.array([meta[i].get("l2", 0.0) for i in subset_ids], dtype=np.float32)
    valid = db_l2 > 0

    # Punktprodukte nur für gültige Zeilen
    dots = np.zeros_like(db_l2, dtype=np.float32)
    if np.any(valid):
        d = X[subset_ids[valid], :].dot(q)
        d[~np.isfinite(d)] = 0.0
        dots[valid] = d

    # Nenner und maskierte Division (kein np.where mit Division!)
    denom = db_l2 * (q_l2 if q_l2 > 0 else 1.0)
    sims = np.full_like(db_l2, -1.0, dtype=np.float32)
    mask = valid & (denom > 0) & np.isfinite(dots) & np.isfinite(denom)
    sims[mask] = dots[mask] / denom[mask]

    # Auswahl TopK
    if topk >= sims.size:
        order = np.argsort(-sims)
        return subset_ids[order].tolist()
    idx_part = np.argpartition(-sims, topk)[:topk]
    return subset_ids[idx_part[np.argsort(-sims[idx_part])]].tolist()


# ────────────────────────────────────────────────────────────────────────────
# Auto-Align-Score (mit Gradient-Weight)

def _masked_cosine(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 2:
        return -1.0
    aa = a[mask]
    bb = b[mask]
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(aa, bb) / (na * nb))


def _shift_candidate(cand: np.ndarray, k: int) -> np.ndarray:
    M = len(cand)
    cand_view = np.zeros_like(cand)
    if k >= 0:
        cand_view[k:] = cand[:M - k]
    else:
        cand_view[:M + k] = cand[-k:]
    return cand_view


def _aligned_mask(q_mask: np.ndarray, start_idx: int, end_idx: int, k: int, M: int) -> np.ndarray:
    cov = np.zeros(M, dtype=bool)
    if 0 <= start_idx <= end_idx < M:
        cov[start_idx:end_idx + 1] = True
    seg = np.zeros(M, dtype=bool)
    if k >= 0:
        seg[k:] = True
    else:
        seg[:M + k] = True
    return q_mask & cov & seg


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    n = arr.size
    ranks = np.empty(n, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = rank
        i = j + 1
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    sa = float(np.std(ra))
    sb = float(np.std(rb))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    rho = float(np.corrcoef(ra, rb)[0, 1])
    if not np.isfinite(rho):
        return 0.0
    return float(np.clip(rho, -1.0, 1.0))


def _find_weighted_peaks(y: np.ndarray, *, max_peaks: int = 80) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(y, dtype=float)
    finite = np.isfinite(arr)
    if arr.size < 5 or not np.any(finite):
        return np.array([], dtype=int), np.array([], dtype=float)
    arr = arr.copy()
    arr[~finite] = 0.0
    ymax = float(np.max(arr))
    if not np.isfinite(ymax) or ymax <= 1e-9:
        return np.array([], dtype=int), np.array([], dtype=float)
    prom_min = max(1e-6, 0.03 * ymax)

    if HAVE_SCIPY_BASELINE:
        pk, props = find_peaks(arr, prominence=prom_min, distance=3)
        if pk.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        w = np.asarray(props.get("prominences", arr[pk]), dtype=float)
    else:
        cand = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] >= arr[2:]))[0] + 1
        if cand.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        w = arr[cand] - np.maximum(arr[cand - 1], arr[cand + 1])
        keep = w >= prom_min
        pk = cand[keep]
        w = w[keep]
        if pk.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

    if pk.size > max_peaks:
        keep = np.argsort(-w)[:max_peaks]
        pk = pk[keep]
        w = w[keep]

    order = np.argsort(pk)
    return pk[order].astype(int), w[order].astype(float)


def _match_weighted_peaks(
    q_idx: np.ndarray,
    q_w: np.ndarray,
    c_idx: np.ndarray,
    c_w: np.ndarray,
    *,
    tol: int,
) -> tuple[float, list[tuple[int, int]]]:
    if q_idx.size == 0 or c_idx.size == 0:
        return 0.0, []

    used_c = np.zeros(c_idx.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    w_match = 0.0

    for qi in np.argsort(-q_w):
        dist = np.abs(c_idx - q_idx[qi])
        candidates = np.where((~used_c) & (dist <= tol))[0]
        if candidates.size == 0:
            continue
        best = max(candidates, key=lambda j: (c_w[j], -dist[j]))
        used_c[best] = True
        pairs.append((int(qi), int(best)))
        w_match += 0.5 * float(q_w[qi] + c_w[best])

    return w_match, pairs


def _peak_consistency_score(
    query: np.ndarray,
    cand_view: np.ndarray,
    mask: np.ndarray,
    *,
    tol: int = PCS_PEAK_TOL,
    f1_weight: float = PCS_F1_WEIGHT,
) -> tuple[float, float, float]:
    if int(np.count_nonzero(mask)) < 20:
        return 0.0, 0.0, 0.0

    q = np.asarray(query[mask], dtype=float)
    c = np.asarray(cand_view[mask], dtype=float)
    q_idx, q_w = _find_weighted_peaks(q)
    c_idx, c_w = _find_weighted_peaks(c)
    if q_idx.size == 0 or c_idx.size == 0:
        return 0.0, 0.0, 0.0

    w_match, pairs = _match_weighted_peaks(q_idx, q_w, c_idx, c_w, tol=tol)
    w_total = float(np.sum(q_w) + np.sum(c_w))
    if w_total <= 1e-12 or w_match <= 0.0 or not pairs:
        return 0.0, 0.0, 0.0

    f1_peak = float(np.clip((2.0 * w_match) / w_total, 0.0, 1.0))
    q_h = np.array([q[q_idx[i]] for i, _ in pairs], dtype=float)
    c_h = np.array([c[c_idx[j]] for _, j in pairs], dtype=float)
    rho = _spearman_corr(q_h, c_h)
    pcs = float(np.clip(f1_weight * f1_peak + (1.0 - f1_weight) * max(0.0, rho), 0.0, 1.0))
    return pcs, f1_peak, rho


def _best_aligned_score(query: np.ndarray, cand: np.ndarray, q_mask: np.ndarray, start_idx: int, end_idx: int, *, max_shift: int = 5, grad_weight: float = GRAD_WEIGHT) -> tuple[float, int]:
    M = len(query)
    best_s, best_k = -1.0, 0

    # Precompute gradient of query once
    qg = np.gradient(query)

    for k in range(-max_shift, max_shift + 1):
        mask = _aligned_mask(q_mask, start_idx, end_idx, k, M)
        cand_view = _shift_candidate(cand, k)

        s_shape = _masked_cosine(query, cand_view, mask)
        if grad_weight > 0.0:
            cg = np.gradient(cand_view)
            mg = mask.copy()
            if mg.size > 2:
                mg[0] = False; mg[-1] = False  # stabiler für Gradientenränder
            s_grad = _masked_cosine(qg, cg, mg)
            s = (1.0 - grad_weight) * s_shape + grad_weight * s_grad
        else:
            s = s_shape
        if s > best_s:
            best_s, best_k = s, k
    return best_s, best_k


def _refine_and_rank(q: np.ndarray, q_mask: np.ndarray, cand_idx: list[int], pack: dict, top_n: int) -> list[dict]:
    X: np.memmap = pack["X"]
    meta: list[dict] = pack["meta"]
    align_max_shift = 5

    scored = []
    for i in cand_idx:
        m = meta[i]
        if m.get("l2", 0.0) <= 0.0 or m.get("end_idx", -1) < m.get("start_idx", 0):
            continue
        v = X[i, :]
        s, k = _best_aligned_score(q, v, q_mask, m["start_idx"], m["end_idx"], max_shift=align_max_shift, grad_weight=GRAD_WEIGHT)
        if s < 0:
            continue
        cand_view = _shift_candidate(v, k)
        mask = _aligned_mask(q_mask, m["start_idx"], m["end_idx"], k, len(q))
        pcs, pcs_f1, pcs_rho = _peak_consistency_score(q, cand_view, mask)
        scored.append({
            "name": m["name"],
            "formula": m["formula"],
            "flag": m.get("flag",""),
            "similarity": s,
            "pcs": pcs,
            "pcs_f1": pcs_f1,
            "pcs_rho": pcs_rho,
            "filename": m.get("filename",""),
            "orig_filename": m.get("orig_filename",""),
            "path": Path(m["path"]),
            "db_idx": int(i),
            "shift": k,
            "start_idx": m["start_idx"],
            "end_idx": m["end_idx"],
            "db_baseline": bool(m.get("db_baseline", False)),
        })

    scored.sort(key=lambda d: d["similarity"], reverse=True)
    return scored[:top_n]


# ────────────────────────────────────────────────────────────────────────────
# Element-Filter Hilfsfunktionen

def _parse_element_list(text: str) -> set[str]:
    if not text.strip():
        return set()
    tokens = re.split(r"[,\s;]+", text.strip())
    clean = {tok.capitalize() for tok in tokens if re.fullmatch(r"[A-Za-z]{1,2}", tok)}
    return clean


def _make_element_filter_fn(include_set: set[str], exclude_set: set[str], mode: str, allow_no_formula: bool):
    def _allowed(m: dict) -> bool:
        els = set(m.get("elements", []))
        has_formula = m.get("has_formula", False)
        if not has_formula:
            return allow_no_formula
        if exclude_set and els & exclude_set:
            return False
        if include_set:
            if mode == "Must include all":
                if not include_set.issubset(els):
                    return False
            elif mode == "Only from this list":
                if not els <= include_set:
                    return False
            elif mode == "Exactly this set":
                if els != include_set:
                    return False
        return True
    return _allowed


# ────────────────────────────────────────────────────────────────────────────
# Input-Signatur (für Result-Caching)

def _inputs_signature(meas_bytes: bytes, range_low: int, range_high: int, ultra: bool,
                      include_str: str, exclude_str: str, mode: str, allow_no_formula: bool,
                      folders: tuple[Path, ...], sig_raw: str, sig_bcb: str, meas_mode: str,
                      baseline_cfg: dict, smoothing_cfg: dict, white_ref_cfg: dict,
                      meas_shift_cm1: float) -> str:
    payload = {
        "meas_sha1": hashlib.sha1(meas_bytes).hexdigest(),
        "range": (int(range_low), int(range_high)),
        "ultra": bool(ultra),
        "include": include_str.strip(),
        "exclude": exclude_str.strip(),
        "mode": mode,
        "allow": bool(allow_no_formula),
        "folders": [str(p) for p in folders],
        "sig_raw": sig_raw,
        "sig_bcb": sig_bcb,
        "meas_mode": meas_mode,
        "grad_w": float(GRAD_WEIGHT),
        "pcs": {"f1_w": float(PCS_F1_WEIGHT), "tol": int(PCS_PEAK_TOL), "v": 1},
        "white_ref": _white_ref_cfg_payload(white_ref_cfg),
        "baseline": _baseline_cfg_payload(baseline_cfg),
        "smoothing": _smoothing_cfg_payload(smoothing_cfg),
        "meas_shift_cm1": float(meas_shift_cm1),
        "match_selection_v": int(MATCH_SELECTION_VERSION),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Ergebnis-Berechnung (teuer) – ausgelagert & nur bei Signaturwechsel

def _mineral_key(name: str) -> str:
    txt = re.sub(r"\s+", " ", str(name or "").strip().lower())
    return txt if txt else "?"


def _final_rank_score(item: dict) -> float:
    sim = float(item.get("similarity", 0.0))
    pcs = float(item.get("pcs", 0.0))
    return FINAL_SIM_WEIGHT * sim + (1.0 - FINAL_SIM_WEIGHT) * pcs


def _select_diverse_top(candidates: list[dict], top_n: int) -> list[dict]:
    if top_n <= 0 or not candidates:
        return []

    scored = []
    for d in candidates:
        c = dict(d)
        c["rank_score"] = _final_rank_score(c)
        scored.append(c)

    scored.sort(
        key=lambda d: (float(d["rank_score"]), float(d.get("similarity", 0.0)), float(d.get("pcs", 0.0))),
        reverse=True,
    )

    unique_by_path: list[dict] = []
    seen_paths: set[str] = set()
    for d in scored:
        path_key = str(d.get("path", ""))
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        unique_by_path.append(d)

    per_mineral_cap = max(2, min(TOP_PER_MINERAL_CAP, int(math.ceil(top_n / 8))))

    selected: list[dict] = []
    per_mineral_count: dict[str, int] = {}
    deferred: list[dict] = []
    for d in unique_by_path:
        mk = _mineral_key(d.get("name", ""))
        cnt = per_mineral_count.get(mk, 0)
        if cnt < per_mineral_cap:
            selected.append(d)
            per_mineral_count[mk] = cnt + 1
            if len(selected) >= top_n:
                break
        else:
            deferred.append(d)

    if len(selected) < top_n:
        selected_paths = {str(d.get("path", "")) for d in selected}
        for d in deferred:
            if str(d.get("path", "")) in selected_paths:
                continue
            selected.append(d)
            if len(selected) >= top_n:
                break

    # Add a small PCS-driven mineral channel for mixed-phase recall.
    # This can surface secondary components that are weaker in full-shape cosine ranking.
    pcs_slots = max(2, min(PCS_MINERAL_SLOT_CAP, top_n // 5))
    pcs_best_by_mineral: dict[str, dict] = {}
    for d in unique_by_path:
        mk = _mineral_key(d.get("name", ""))
        cur = pcs_best_by_mineral.get(mk)
        if cur is None:
            pcs_best_by_mineral[mk] = d
            continue
        if (
            float(d.get("pcs", 0.0)) > float(cur.get("pcs", 0.0))
            or (
                float(d.get("pcs", 0.0)) == float(cur.get("pcs", 0.0))
                and float(d.get("similarity", 0.0)) > float(cur.get("similarity", 0.0))
            )
        ):
            pcs_best_by_mineral[mk] = d

    pcs_pool = sorted(
        pcs_best_by_mineral.values(),
        key=lambda d: (float(d.get("pcs", 0.0)), float(d.get("similarity", 0.0))),
        reverse=True,
    )
    selected_paths = {str(d.get("path", "")) for d in selected}
    selected_minerals = {_mineral_key(d.get("name", "")) for d in selected}
    pcs_additions: list[dict] = []
    for d in pcs_pool:
        mk = _mineral_key(d.get("name", ""))
        p = str(d.get("path", ""))
        if p in selected_paths or mk in selected_minerals:
            continue
        if float(d.get("pcs", 0.0)) < PCS_MINERAL_MIN_PCS:
            continue
        if float(d.get("similarity", 0.0)) < PCS_MINERAL_MIN_SIM:
            continue
        pcs_additions.append(d)
        selected_paths.add(p)
        selected_minerals.add(mk)
        if len(pcs_additions) >= pcs_slots:
            break
    if pcs_additions:
        keep_n = max(0, top_n - len(pcs_additions))
        selected = selected[:keep_n] + pcs_additions

    selected.sort(
        key=lambda d: (float(d["rank_score"]), float(d.get("similarity", 0.0)), float(d.get("pcs", 0.0))),
        reverse=True,
    )
    return selected[:top_n]

def _compute_matches_from_query_vector(
    q_sel: np.ndarray,
    q_mask: np.ndarray,
    range_low: int,
    range_high: int,
    pack_raw: dict,
    pack_bcb: dict,
    allowed_ids_raw: np.ndarray,
    allowed_ids_bcb: np.ndarray,
    meas_mode: str,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    # Kandidaten je DB-Variante (2x)
    topk_raw = min(int(allowed_ids_raw.size), max(300, top_n * 12, 3600))
    topk_bcb = min(int(allowed_ids_bcb.size), max(300, top_n * 8, 1800))

    def _cands(q_vec, pack, allowed_ids, topk):
        if allowed_ids.size == 0:
            return []
        return _topk_cosine_subset(q_vec, pack["X"], pack["meta"], allowed_ids, topk)

    cand_sel_raw = _cands(q_sel, pack_raw, allowed_ids_raw, topk_raw)
    cand_sel_bcb = _cands(q_sel, pack_bcb, allowed_ids_bcb, topk_bcb)

    # Coverage-Filter ≥70 % (Pack-Grid identisch → reicht einmal)
    gi = pack_raw["grid_info"]
    GRID_MIN, GRID_STEP, GRID_LEN = gi["min"], gi["step"], gi["len"]
    def _overlap_frac(start_idx, end_idx):
        a0 = max(0, int((range_low  - GRID_MIN) // GRID_STEP))
        a1 = min(GRID_LEN - 1, int((range_high - GRID_MIN) // GRID_STEP))
        b0, b1 = start_idx, end_idx
        left = max(a0, b0)
        right = min(a1, b1)
        return max(0, right - left + 1) / max(1, (a1 - a0 + 1))

    def _filter_cov(cands, meta):
        return [i for i in cands if _overlap_frac(meta[i]["start_idx"], meta[i]["end_idx"]) >= 0.7]

    cand_sel_raw = _filter_cov(cand_sel_raw, pack_raw["meta"])
    cand_sel_bcb = _filter_cov(cand_sel_bcb, pack_bcb["meta"])

    # Feinranking + Label
    pool_top_n_raw = max(top_n * 4, topk_raw)
    pool_top_n_bcb = max(top_n * 4, topk_bcb)

    top_sel_raw = _refine_and_rank(q_sel, q_mask, cand_sel_raw, pack_raw, pool_top_n_raw)
    for d in top_sel_raw:
        d.update({"meas_variant": meas_mode, "db_variant": "DB-RAW"})

    top_sel_bcb = _refine_and_rank(q_sel, q_mask, cand_sel_bcb, pack_bcb, pool_top_n_bcb)
    for d in top_sel_bcb:
        d.update({"meas_variant": meas_mode, "db_variant": "DB-BC"})

    top_combined = _select_diverse_top(top_sel_raw + top_sel_bcb, top_n)
    return top_combined


def _build_residual_query_vector(
    q_sel: np.ndarray,
    q_mask: np.ndarray,
    selected_match: dict,
    pack_raw: dict,
    pack_bcb: dict,
) -> tuple[np.ndarray, float] | None:
    db_idx = int(selected_match.get("db_idx", -1))
    if db_idx < 0:
        return None

    use_bcb = bool(selected_match.get("db_baseline", selected_match.get("db_variant") == "DB-BC"))
    pack_sel = pack_bcb if use_bcb else pack_raw
    if db_idx >= len(pack_sel["meta"]):
        return None

    cand = np.asarray(pack_sel["X"][db_idx, :], dtype=float)
    shift = int(selected_match.get("shift", 0))
    start_idx = int(selected_match.get("start_idx", 0))
    end_idx = int(selected_match.get("end_idx", -1))
    cand_view = _shift_candidate(cand, shift)
    mask = _aligned_mask(q_mask, start_idx, end_idx, shift, len(q_sel))
    if int(np.count_nonzero(mask)) < 20:
        return None

    denom = float(np.dot(cand_view[mask], cand_view[mask]))
    if denom <= 1e-12 or not np.isfinite(denom):
        return None
    alpha = float(np.dot(q_sel[mask], cand_view[mask]) / denom)
    alpha = float(np.clip(alpha, 0.0, 1.5))

    residual = np.asarray(q_sel, dtype=float).copy()
    residual[mask] = residual[mask] - alpha * cand_view[mask]
    residual[~q_mask] = 0.0
    residual = np.maximum(residual, 0.0)
    residual = _normalize_to_peak(residual)
    return residual.astype(np.float32), alpha


def _compute_matches(meas_x: np.ndarray, meas_y: np.ndarray,
                     range_low: int, range_high: int,
                     pack_raw: dict, pack_bcb: dict,
                     allowed_ids_raw: np.ndarray, allowed_ids_bcb: np.ndarray,
                     baseline_cfg: dict, smoothing_cfg: dict, meas_mode: str,
                     top_n: int = DEFAULT_TOP_N) -> list[dict]:

    # Query-Vektor nur für die gewählte Messungsvariante
    use_bc = (meas_mode == "BC")
    q_sel, _q_l2_sel, q_mask = _prepare_query_vector(
        meas_x, meas_y, range_low, range_high, pack_raw["grid"],
        apply_baseline=use_bc, baseline_cfg=baseline_cfg, smoothing_cfg=smoothing_cfg
    )
    return _compute_matches_from_query_vector(
        q_sel, q_mask,
        range_low, range_high,
        pack_raw, pack_bcb,
        allowed_ids_raw, allowed_ids_bcb,
        meas_mode,
        top_n=top_n,
    )


# ────────────────────────────────────────────────────────────────────────────
# Streamlit-GUI

def _run_streamlit() -> None:
    import pandas as pd

    st.set_page_config(page_title="RamanPhaseID", layout="wide")
    app_theme_key = "app_theme"
    if app_theme_key not in st.session_state:
        st.session_state[app_theme_key] = "dark"
    app_theme = _normalize_app_theme(st.session_state.get(app_theme_key))
    st.session_state[app_theme_key] = app_theme
    _apply_app_theme(app_theme)
    st.title("RamanPhaseID - 0.98beta")

    plot_theme_key = "plot_theme"
    if plot_theme_key not in st.session_state:
        st.session_state[plot_theme_key] = app_theme
    plot_theme = _normalize_plot_theme(st.session_state.get(plot_theme_key))
    st.session_state[plot_theme_key] = plot_theme
    intensity_numbers_key = "show_preview_intensity_numbers"
    if intensity_numbers_key not in st.session_state:
        st.session_state[intensity_numbers_key] = True
    show_preview_intensity_numbers = bool(st.session_state.get(intensity_numbers_key, True))

    @st.cache_data(show_spinner=False, persist="disk")
    def _get_db_entries_cached(folders_as_str: tuple[str, ...], signature: str):
        return rc.load_reference_folders(tuple(Path(s) for s in folders_as_str))

    @st.cache_data(show_spinner=False)
    def _parse_db_cached(path_str: str):
        p = Path(path_str)
        if p.suffix.lower() == ".txt":
            return rc._parse_rruff(p)
        return rc._parse_rod(p)

    # Quit-Button
    def _shutdown():
        def _kill():
            time.sleep(0.3)
            os._exit(0)
        threading.Thread(target=_kill, daemon=True).start()

    # Modi
    ultra = st.sidebar.checkbox(
        "Full-range matching (60–4000 cm⁻¹)",
        help="Uses the extended matching grid above 1900 cm⁻¹, up to 4000 cm⁻¹ (slower, larger cache needed).",
    )

    white_defaults = _default_white_ref_cfg()
    with st.sidebar.expander("White-light subtraction", expanded=False):
        white_ref_enabled = st.checkbox(
            "Subtract white-light reference before baseline",
            value=bool(white_defaults["enabled"]),
            help="Adds an explicit preprocessing step before baseline correction.",
        )
        white_ref_scale = st.slider(
            "White-light scaling factor",
            min_value=0.0,
            max_value=6.0,
            value=float(white_defaults["scale"]),
            step=0.001,
            disabled=not white_ref_enabled,
        )
        st.caption("Step 1 is confirmed explicitly before baseline correction.")

    # Baseline-Settings (integriert aus baseline_app_01c.py)
    with st.sidebar.expander("Baseline mode (arPLS/IAsLS/RAW)", expanded=True):
        baseline_mode = st.radio(
            "Baseline method",
            ["arPLS", "ALS (IAsLS)", "RAW (no baseline)"],
            index=0,
            help="Choose arPLS or improved ALS (IAsLS) for baseline-corrected matching, or RAW to skip baseline correction.",
        )
        if baseline_mode == "RAW (no baseline)":
            method = "NONE"
            lam_exp = 5
            lam = 10.0 ** lam_exp
            itermax = 50
            tol_choice = 1e-3
            p = 0.010
            niter = 20
            lam1_exp = 2
            lam1 = 10.0 ** lam1_exp
            use_autoscale = False
            st.caption("No baseline correction: matching will use RAW measurement in step 4.")
        else:
            method = "arPLS" if baseline_mode == "arPLS" else "ALS"
            lam_exp = st.slider(
                "λ (10^x)", min_value=0, max_value=9, value=4, step=1,
                help="Baseline stiffness (10^x). Lower values follow curved backgrounds better; high values can become almost linear.",
            )
            lam = 10.0 ** lam_exp

            if method == "arPLS":
                itermax = st.slider("max iterations", 5, 200, 50, step=1)
                tol_choice = st.select_slider(
                    "Tolerance (stop criterion)",
                    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                    value=1e-3,
                )
                p = 0.010
                niter = 20
                lam1_exp = 2
                lam1 = 10.0 ** lam1_exp
            else:
                p = st.slider(
                    "Asymmetry p", 0.000, 0.200, 0.010, step=0.001,
                    help="Lower values suppress peaks more strongly (typ. 0.001–0.05).",
                )
                niter = st.slider("Iterations", 1, 80, 20, step=1)
                lam1_exp = st.slider(
                    "λ1 (IAsLS, 10^x)", min_value=-2, max_value=6, value=2, step=1,
                    help="Additional first-derivative penalty in IAsLS. Higher values enforce stronger baseline smoothness.",
                )
                lam1 = 10.0 ** lam1_exp
                itermax = 50
                tol_choice = 1e-3

            use_autoscale = st.checkbox(
                "Internal autoscaling",
                value=True,
                help="Stabilizes IAsLS/arPLS internally; display/export remain in original units.",
            )
            if not HAVE_SCIPY_BASELINE:
                st.warning("SciPy not found: using raman_core ALS as fallback.")
        st.caption("DB baseline for DB-BC cache/overlay is fixed: arPLS, λ=10^4, iter≤50, tol=1e-3, autoscaling on.")

        st.caption("Display/export options from baseline app")
        show_raw = st.checkbox("Show raw signal", value=True)
        show_baseline = st.checkbox("Show baseline", value=True)
        show_corrected = st.checkbox("Show corrected signal", value=True)
        decimals = st.slider("Decimal places (export)", 0, 10, 6, step=1)
        keep_header = st.checkbox("Keep header exactly (recommended)", value=True)
        add_note = st.checkbox("Add note to header", value=False, disabled=keep_header)
        note_text = st.text_input(
            "Note text (optional)",
            value="Baseline subtracted by arPLS/IAsLS",
            disabled=(keep_header or not add_note),
        )

    baseline_cfg = {
        "method": method,
        "lam_exp": int(lam_exp),
        "lam": float(lam),
        "itermax": int(itermax),
        "tol": float(tol_choice),
        "p": float(p),
        "niter": int(niter),
        "lam1_exp": int(lam1_exp),
        "lam1": float(lam1),
        "autoscale": bool(use_autoscale),
        "db_strength": 1.00,
    }
    db_baseline_cfg = _fixed_db_baseline_cfg()
    db_baseline_token = _baseline_cfg_token(db_baseline_cfg)
    meas_mode = "RAW" if method == "NONE" else "BC"

    smooth_defaults = _default_smoothing_cfg()
    with st.sidebar.expander("Smoothing before matching", expanded=True):
        smoothing_enabled = st.checkbox(
            "Enable Savitzky-Golay smoothing",
            value=bool(smooth_defaults["enabled"]),
            help="Applied in step 3 to the selected measurement variant before matching.",
        )
        smoothing_window = st.slider(
            "Window length (odd)",
            min_value=3,
            max_value=101,
            value=int(smooth_defaults["window"]),
            step=2,
            disabled=not smoothing_enabled,
        )
        smooth_poly_max = max(0, min(9, int(smoothing_window) - 1))
        smoothing_poly = st.slider(
            "Polynomial order",
            min_value=0,
            max_value=smooth_poly_max,
            value=min(int(smooth_defaults["poly"]), smooth_poly_max),
            step=1,
            disabled=not smoothing_enabled,
        )
        st.caption("Step 3 is confirmed explicitly before matching starts.")

    smoothing_cfg = {
        "enabled": bool(smoothing_enabled),
        "window": int(smoothing_window),
        "poly": int(smoothing_poly),
    }
    smoothing_token = _smoothing_cfg_token(smoothing_cfg)

    # Reload-Caches
    if st.sidebar.button("Reload DB"):
        rc.load_reference_folders.cache_clear()
        st.cache_data.clear()
        try:
            sig_standard = _compute_signature_with_grid(
                list(MATCH_FOLDERS_STANDARD),
                CLEAN_GRID["min"], CLEAN_GRID["max"], CLEAN_GRID["step"]
            )
            sig_ultra = _compute_signature_with_grid(
                list(MATCH_FOLDERS_ULTRA),
                ULTRA_GRID["min"], ULTRA_GRID["max"], ULTRA_GRID["step"]
            )
            for pc_sig in (
                f"{sig_standard}-dbbc0",
                f"{sig_standard}-b{db_baseline_token}-dbbc1",
                f"{sig_ultra}-dbbc0",
                f"{sig_ultra}-b{db_baseline_token}-dbbc1",
            ):
                pc_dir = _precomp_dir(pc_sig)
                if pc_dir.exists():
                    shutil.rmtree(pc_dir, ignore_errors=True)
        except Exception:
            pass

    active_folders = MATCH_FOLDERS_ULTRA if ultra else MATCH_FOLDERS_STANDARD
    active_grid_cfg = ULTRA_GRID if ultra else CLEAN_GRID
    active_entries_sig = hashlib.sha1(
        "|".join(sorted(str(Path(p).resolve()) for p in active_folders)).encode()
    ).hexdigest()
    try:
        active_entries, active_skipped = _get_db_entries_cached(
            tuple(str(p) for p in active_folders),
            active_entries_sig,
        )
    except Exception as exc:
        active_entries, active_skipped = [], []
        st.sidebar.warning(f"Could not read active DB cache: {exc}")

    unique_name_map: dict[str, str] = {}
    for e in active_entries:
        nm = str(e.get("name", "")).strip()
        if not nm:
            continue
        key = nm.casefold()
        if key not in unique_name_map:
            unique_name_map[key] = nm
    unique_names = sorted(unique_name_map.values(), key=str.casefold)

    with st.sidebar.expander("Overlay", expanded=True):
        meas_shift_cm1 = st.slider(
            "Measurement Raman-shift offset (cm⁻¹)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            key="measurement_shift_cm1",
            help="Shifts the measurement x-axis before matching and overlays. Positive values move peaks to higher Raman-shift values.",
        )
        st.caption(f"Current offset: {float(meas_shift_cm1):+.1f} cm⁻¹")

    with st.sidebar.expander("Current cached DB overview", expanded=False):
        st.caption(f"Active folders: {', '.join(Path(p).name for p in active_folders)}")
        st.markdown(f"- Cached entries: **{len(active_entries)}**")
        st.markdown(f"- Different mineral/phase names: **{len(unique_names)}**")
        if active_skipped:
            st.caption(f"Skipped files while loading headers: {len(active_skipped)}")

        sidebar_name_query = st.text_input(
            "Search mineral/phase in current cache",
            value="",
            placeholder="e.g. quartz",
            key="sidebar_db_name_query",
        ).strip()
        if sidebar_name_query:
            q = sidebar_name_query.casefold()
            exact = unique_name_map.get(q)
            hits = [nm for nm in unique_names if q in nm.casefold()]
            if exact is not None:
                st.success(f"Exact name found: {exact}")
            if hits:
                shown = 30
                st.caption(f"{len(hits)} matching names:")
                st.write(", ".join(hits[:shown]))
                if len(hits) > shown:
                    st.caption(f"... and {len(hits) - shown} more")
            else:
                st.info("No matching mineral/phase found in the current cached DB.")

    with st.sidebar.expander("Element filter (optional)", expanded=False):
        include_str = st.text_input("Include elements (comma-separated)", key="el_inc")
        mode = st.radio(
            "Mode",
            ["Must include all", "Only from this list", "Exactly this set"],
            index=0,
            key="el_mode"
        )
        exclude_str = st.text_input("Exclude elements", key="el_exc")
        allow_no_formula = st.checkbox(
            "Include entries without formula",
            value=not (include_str.strip() or exclude_str.strip()),
            key="el_allow_noform"
        )

    st.sidebar.divider()
    next_app_theme = "light" if app_theme == "dark" else "dark"
    st.sidebar.caption(f"App theme: {app_theme}")
    if st.sidebar.button(
        f"Switch app to {next_app_theme} mode",
        key="toggle_app_theme_btn",
        width="stretch",
    ):
        st.session_state[app_theme_key] = next_app_theme
        st.session_state[plot_theme_key] = next_app_theme
        _safe_rerun()

    next_plot_theme = "light" if plot_theme == "dark" else "dark"
    st.sidebar.caption(f"Plot theme: {plot_theme}")
    if st.sidebar.button(
        f"Switch plots to {next_plot_theme} theme",
        key="toggle_plot_theme_btn",
        width="stretch",
    ):
        st.session_state[plot_theme_key] = next_plot_theme
        plot_theme = next_plot_theme

    next_intensity_state = not show_preview_intensity_numbers
    intensity_status = "shown" if show_preview_intensity_numbers else "hidden"
    intensity_target = "show" if next_intensity_state else "hide"
    st.sidebar.caption(f"Preview intensity numbers: {intensity_status}")
    if st.sidebar.button(
        f"{intensity_target.capitalize()} intensity numbers (first 3 plots)",
        key="toggle_preview_intensity_numbers_btn",
        width="stretch",
    ):
        st.session_state[intensity_numbers_key] = next_intensity_state
        show_preview_intensity_numbers = next_intensity_state

    st.sidebar.button("❌ Quit application", on_click=_shutdown, width="stretch")

    # Messspektrum
    meas_file = st.file_uploader("Measurement spectrum (.txt / .csv)", key="measurement_file")
    meas_bytes_key = "measurement_file_bytes_cached"
    meas_name_key = "measurement_file_name_cached"
    meas_file_name = ""
    if meas_file is not None:
        meas_bytes = meas_file.getvalue()
        meas_file_name = str(getattr(meas_file, "name", "") or "measurement.txt")
        st.session_state[meas_bytes_key] = meas_bytes
        st.session_state[meas_name_key] = meas_file_name
    else:
        meas_bytes = bytes(st.session_state.get(meas_bytes_key, b""))
        meas_file_name = str(st.session_state.get(meas_name_key, "") or "").strip()
    if not meas_bytes:
        st.stop()
    if not meas_file_name:
        meas_file_name = "measurement.txt"
    meas_x_full, meas_y_raw_full = rc.parse_measurement(meas_bytes.decode(errors="ignore"))

    # --- NEU: Guard gegen leere Arrays/NaNs, damit _normalize nicht crasht ---
    if meas_x_full.size == 0 or meas_y_raw_full.size == 0:
        st.error("The measurement file contains no usable data points (x or y empty). Please check the file.")
        st.stop()

    white_ref_file = None
    if white_ref_enabled:
        white_ref_file = st.file_uploader(
            "White-light reference spectrum (.txt / .csv)",
            key="white_ref_file",
        )

    white_ref_bytes_key = "white_ref_file_bytes_cached"
    white_ref_bytes = b""
    if white_ref_enabled and white_ref_file is not None:
        white_ref_bytes = white_ref_file.getvalue()
        st.session_state[white_ref_bytes_key] = white_ref_bytes
    elif white_ref_enabled:
        white_ref_bytes = bytes(st.session_state.get(white_ref_bytes_key, b""))

    white_ref_x = np.array([], dtype=float)
    white_ref_y = np.array([], dtype=float)
    white_ref_error = ""
    if white_ref_enabled and white_ref_bytes:
        try:
            white_ref_x, white_ref_y = rc.parse_measurement(white_ref_bytes.decode(errors="ignore"))
            finite_ref = np.isfinite(white_ref_x) & np.isfinite(white_ref_y)
            unique_x = np.unique(np.asarray(white_ref_x, dtype=float)[finite_ref]).size
            if white_ref_x.size < 2 or white_ref_y.size < 2 or unique_x < 2:
                white_ref_error = "White-light reference must contain at least 2 valid points with distinct Raman shifts."
        except Exception as exc:
            white_ref_error = f"Could not parse white-light reference: {exc}"

    white_ref_applied = bool(white_ref_enabled and white_ref_bytes and not white_ref_error)
    if white_ref_applied:
        white_ref_aligned_full = _align_reference_to_target(meas_x_full, white_ref_x, white_ref_y)
    else:
        white_ref_aligned_full = np.zeros_like(meas_y_raw_full, dtype=float)
    white_ref_scaled_full = float(white_ref_scale) * white_ref_aligned_full
    meas_y_full = meas_y_raw_full - white_ref_scaled_full if white_ref_applied else meas_y_raw_full.copy()
    meas_x_shifted_full = np.asarray(meas_x_full, dtype=float) + float(meas_shift_cm1)

    white_ref_sha1 = hashlib.sha1(white_ref_bytes).hexdigest() if white_ref_bytes else ""
    white_ref_cfg = {
        "enabled": bool(white_ref_enabled),
        "scale": float(white_ref_scale),
        "ref_sha1": white_ref_sha1 if white_ref_enabled else "",
    }
    white_ref_token = _white_ref_cfg_token(white_ref_cfg)

    base_name = Path(meas_file_name).stem
    meas_text = meas_bytes.decode("utf-8", errors="ignore")
    header_lines, data_lines, delimiter_hint = _split_header_data(meas_text)
    export_x = meas_x_full.copy()
    export_y_raw = meas_y_raw_full.copy()
    export_header_exact_ok = False
    if data_lines:
        x_data, y_data = _parse_xy_from_data_lines(data_lines)
        if x_data.size > 0 and x_data.size == y_data.size:
            export_x, export_y_raw = x_data, y_data
            export_header_exact_ok = True

    if white_ref_applied:
        white_ref_aligned_export = _align_reference_to_target(export_x, white_ref_x, white_ref_y)
        export_y = export_y_raw - float(white_ref_scale) * white_ref_aligned_export
    else:
        white_ref_aligned_export = np.zeros_like(export_y_raw, dtype=float)
        export_y = export_y_raw.copy()

    white_ref_ready_key = "white_ref_ready_sig"
    baseline_ready_key = "baseline_ready_sig"
    smoothing_ready_key = "smoothing_ready_sig"

    st.subheader("Measurement spectrum (raw)")
    fig_white, ax_white = plt.subplots(figsize=(11, 4.6))
    try:
        ax_white.plot(meas_x_shifted_full, meas_y_raw_full, label="measurement (raw)", linewidth=1.0)
        if white_ref_applied:
            ax_white.plot(
                meas_x_shifted_full,
                white_ref_scaled_full,
                label=f"white-light reference × {float(white_ref_scale):.3f}",
                linewidth=1.0,
            )
            ax_white.plot(
                meas_x_shifted_full,
                meas_y_full,
                label="measurement − white-light reference",
                linewidth=1.1,
            )
        ax_white.set_xlabel("Raman shift (cm⁻¹)")
        ax_white.set_ylabel("Intensity (a.u.)")
        ax_white.legend(loc="best")
        _apply_plot_style(fig_white, ax_white, theme=plot_theme)
        _set_intensity_number_visibility(ax_white, show_preview_intensity_numbers)
        st.pyplot(fig_white)
        fig_white_png = _fig_to_bytes(fig_white, "png")
        fig_white_svg = _fig_to_bytes(fig_white, "svg")
        if white_ref_applied:
            col_white_png, col_white_svg, col_white_txt, _ = st.columns([1, 1, 1, 1])
        else:
            col_white_png, col_white_svg, _ = st.columns([1, 1, 2])
        with col_white_png:
            st.download_button(
                "⬇️ Measurement figure (PNG)",
                data=fig_white_png,
                file_name=f"{base_name}_measurement_raw.png",
                mime="image/png",
                width="stretch",
            )
        with col_white_svg:
            st.download_button(
                "⬇️ Measurement figure (SVG)",
                data=fig_white_svg,
                file_name=f"{base_name}_measurement_raw.svg",
                mime="image/svg+xml",
                width="stretch",
            )
        if white_ref_applied:
            with col_white_txt:
                st.download_button(
                    "⬇️ Measurement − white-light reference",
                    data=_rebuild_file_bytes(
                        header_lines if export_header_exact_ok else [],
                        export_x,
                        export_y,
                        decimals=int(decimals),
                        delimiter=delimiter_hint,
                        keep_header_exact=bool(keep_header and export_header_exact_ok),
                        extra_note=(
                            f"White-light reference subtracted (scale={float(white_ref_scale):.3f})."
                            if not keep_header
                            else None
                        ),
                    ),
                    file_name=f"{base_name}_measurement_minus_white_light_reference.txt",
                    mime="text/plain",
                    width="stretch",
                )
    finally:
        plt.close(fig_white)

    if white_ref_applied and keep_header and not export_header_exact_ok:
        st.caption("Exact header export for the white-light-subtracted spectrum was not available; data export uses a rebuilt plain-text body.")

    if white_ref_enabled and not white_ref_bytes:
        st.info("Upload a white-light reference spectrum to continue.")
    if white_ref_error:
        st.error(white_ref_error)
    if not white_ref_enabled:
        st.caption("White-light subtraction disabled. Baseline preview uses the raw measurement.")
    elif white_ref_applied:
        st.caption(
            f"White-light subtraction enabled: measurement − ({float(white_ref_scale):.3f} × reference)."
        )

    white_ref_ready_payload = {
        "meas_sha1": hashlib.sha1(meas_bytes).hexdigest(),
        "white_ref": _white_ref_cfg_payload(white_ref_cfg),
    }
    white_ref_ready_sig = hashlib.sha1(json.dumps(white_ref_ready_payload, sort_keys=True).encode()).hexdigest()
    can_confirm_white_ref = (not white_ref_enabled) or white_ref_applied

    if st.session_state.get(white_ref_ready_key) != white_ref_ready_sig:
        st.info("Step 1/4: Adjust white-light subtraction and confirm to unlock baseline correction.")
        if st.button(
            "Apply white-light subtraction and/or continue to baseline",
            type="primary",
            width="stretch",
            key="approve_white_ref_btn",
            disabled=not can_confirm_white_ref,
        ):
            st.session_state[white_ref_ready_key] = white_ref_ready_sig
            st.session_state.pop(baseline_ready_key, None)
            st.session_state.pop(smoothing_ready_key, None)
            st.session_state.pop("results_sig", None)
            st.session_state.pop("top_combined", None)
            _delay_then_rerun(2.0)
        st.stop()
    else:
        st.success("Step 1/4 complete: white-light subtraction settings applied.")
        if st.button(
            "Adjust white-light subtraction settings again",
            width="stretch",
            key="edit_white_ref_again_btn",
        ):
            st.session_state.pop(white_ref_ready_key, None)
            st.session_state.pop(baseline_ready_key, None)
            st.session_state.pop(smoothing_ready_key, None)
            st.session_state.pop("results_sig", None)
            st.session_state.pop("top_combined", None)
            _safe_rerun()

    # Baseline-Preview + Export (wie baseline_app_01c.py)
    baseline_full = _compute_baseline(meas_y_full, baseline_cfg)
    meas_corr_full = meas_y_full - baseline_full
    baseline_export = _compute_baseline(export_y, baseline_cfg)
    corr_export = export_y - baseline_export

    st.subheader("Baseline preview")
    x_min_f = float(np.min(meas_x_shifted_full))
    x_max_f = float(np.max(meas_x_shifted_full))
    range_key = "baseline_preview_range"
    prev_shift_key = "baseline_preview_shift_cm1_prev"
    prev_shift_cm1 = float(st.session_state.get(prev_shift_key, meas_shift_cm1))
    if range_key in st.session_state:
        prev_rng = st.session_state.get(range_key)
        if isinstance(prev_rng, (tuple, list)) and len(prev_rng) == 2:
            d_shift = float(meas_shift_cm1) - prev_shift_cm1
            lo_prev = float(prev_rng[0]) + d_shift
            hi_prev = float(prev_rng[1]) + d_shift
            lo_prev = float(np.clip(lo_prev, x_min_f, x_max_f))
            hi_prev = float(np.clip(hi_prev, x_min_f, x_max_f))
            if hi_prev < lo_prev:
                lo_prev, hi_prev = x_min_f, x_max_f
            st.session_state[range_key] = (lo_prev, hi_prev)
    st.session_state[prev_shift_key] = float(meas_shift_cm1)
    step_guess = float((x_max_f - x_min_f) / 1000.0) if x_max_f > x_min_f else 1.0
    rng_preview = st.slider(
        "Matching range (cm⁻¹)",
        min_value=x_min_f,
        max_value=x_max_f,
        value=(x_min_f, x_max_f),
        step=step_guess,
        key=range_key,
    )
    mask_prev = (meas_x_shifted_full >= rng_preview[0]) & (meas_x_shifted_full <= rng_preview[1])
    baseline_input_label = "measurement − white-light reference" if white_ref_applied else "raw measurement"
    fig_prev, ax_prev = plt.subplots(figsize=(11, 4.6))
    try:
        if show_raw:
            ax_prev.plot(meas_x_shifted_full[mask_prev], meas_y_full[mask_prev], label=baseline_input_label, linewidth=1.0)
        if show_baseline:
            ax_prev.plot(
                meas_x_shifted_full[mask_prev],
                baseline_full[mask_prev],
                label=f"baseline · {_baseline_label(baseline_cfg)}",
                linewidth=1.0,
            )
        if show_corrected:
            ax_prev.plot(
                meas_x_shifted_full[mask_prev],
                meas_corr_full[mask_prev],
                label="corrected = input − baseline",
                linewidth=1.0,
            )
        ax_prev.set_xlabel("Raman shift (cm⁻¹)")
        ax_prev.set_ylabel("Intensity (a.u.)")
        ax_prev.legend(loc="best")
        _apply_plot_style(fig_prev, ax_prev, theme=plot_theme)
        _set_intensity_number_visibility(ax_prev, show_preview_intensity_numbers)
        st.pyplot(fig_prev)
        base_name = Path(meas_file_name).stem
        fig_prev_png = _fig_to_bytes(fig_prev, "png")
        fig_prev_svg = _fig_to_bytes(fig_prev, "svg")
        col_prev_png, col_prev_svg, _ = st.columns([1, 1, 2])
        with col_prev_png:
            st.download_button(
                "⬇️ Baseline preview (PNG)",
                data=fig_prev_png,
                file_name=f"{base_name}_baseline_preview.png",
                mime="image/png",
                width="stretch",
            )
        with col_prev_svg:
            st.download_button(
                "⬇️ Baseline preview (SVG)",
                data=fig_prev_svg,
                file_name=f"{base_name}_baseline_preview.svg",
                mime="image/svg+xml",
                width="stretch",
            )
    finally:
        plt.close(fig_prev)

    st.caption("Export (baseline-app compatible): corrected spectrum and baseline only.")
    if keep_header and not export_header_exact_ok:
        st.info("Exact header export could not be determined; exporting without unchanged original header.")
    export_note = note_text if ((not keep_header) and add_note and note_text.strip()) else None
    keep_header_effective = bool(keep_header and export_header_exact_ok)
    header_lines_export = header_lines if export_header_exact_ok else []
    col_exp_a, col_exp_b, _ = st.columns([1, 1, 2])
    with col_exp_a:
        st.download_button(
            "⬇️ Spectrum (baseline corrected)",
            data=_rebuild_file_bytes(
                header_lines_export, export_x, corr_export,
                decimals=int(decimals),
                delimiter=delimiter_hint,
                keep_header_exact=keep_header_effective,
                extra_note=export_note,
            ),
            file_name=(Path(meas_file_name).stem + "_baseline_corrected.txt"),
            mime="text/plain",
            width="stretch",
        )
    with col_exp_b:
        st.download_button(
            "⬇️ Baseline (background only)",
            data=_rebuild_file_bytes(
                header_lines_export, export_x, baseline_export,
                decimals=int(decimals),
                delimiter=delimiter_hint,
                keep_header_exact=keep_header_effective,
                extra_note=export_note,
            ),
            file_name=(Path(meas_file_name).stem + "_baseline_only.txt"),
            mime="text/plain",
            width="stretch",
        )

    baseline_ready_payload = {
        "meas_sha1": hashlib.sha1(meas_bytes).hexdigest(),
        "white_ref_ready_sig": white_ref_ready_sig,
        "white_ref": _white_ref_cfg_payload(white_ref_cfg),
        "baseline": _baseline_cfg_payload(baseline_cfg),
        "meas_mode": meas_mode,
    }
    baseline_ready_sig = hashlib.sha1(json.dumps(baseline_ready_payload, sort_keys=True).encode()).hexdigest()

    if st.session_state.get(baseline_ready_key) != baseline_ready_sig:
        st.info("Step 2/4: Adjust baseline mode (arPLS/IAsLS/RAW) and confirm to unlock smoothing.")
        if st.button(
            "Apply baseline settings and continue to smoothing",
            type="primary",
            width="stretch",
            key="approve_baseline_btn",
        ):
            st.session_state[baseline_ready_key] = baseline_ready_sig
            st.session_state.pop(smoothing_ready_key, None)
            st.session_state.pop("results_sig", None)
            st.session_state.pop("top_combined", None)
            _delay_then_rerun(2.0)
        st.stop()
    else:
        st.success("Step 2/4 complete: baseline settings applied.")
        if st.button(
            "Adjust baseline settings again",
            width="stretch",
            key="edit_baseline_again_btn",
        ):
            st.session_state.pop(baseline_ready_key, None)
            st.session_state.pop(smoothing_ready_key, None)
            st.session_state.pop("results_sig", None)
            st.session_state.pop("top_combined", None)
            _safe_rerun()

    smooth_input_full = meas_corr_full if meas_mode == "BC" else meas_y_full
    if meas_mode == "BC":
        smooth_input_label = "baseline-corrected measurement (BC)"
    elif white_ref_applied:
        smooth_input_label = "white-ref corrected measurement (RAW)"
    else:
        smooth_input_label = "raw measurement (RAW)"
    smooth_output_full = _apply_smoothing(smooth_input_full, smoothing_cfg)
    smooth_export_input = corr_export if meas_mode == "BC" else export_y
    smooth_export_full = _apply_smoothing(smooth_export_input, smoothing_cfg)
    smooth_params = _sanitize_savgol_params(
        smooth_input_full.size,
        int(smoothing_cfg.get("window", 11)),
        int(smoothing_cfg.get("poly", 3)),
    ) if smoothing_cfg.get("enabled", True) else None

    st.subheader("Smoothing preview")
    fig_smooth, ax_smooth = plt.subplots(figsize=(11, 4.6))
    try:
        ax_smooth.plot(
            meas_x_shifted_full[mask_prev],
            smooth_input_full[mask_prev],
            label=f"input · {smooth_input_label}",
            linewidth=1.0,
        )
        if bool(smoothing_cfg.get("enabled", True)):
            ax_smooth.plot(
                meas_x_shifted_full[mask_prev],
                smooth_output_full[mask_prev],
                label=f"smoothed · {_smoothing_label(smoothing_cfg)}",
                linewidth=1.2,
            )
        ax_smooth.set_xlabel("Raman shift (cm⁻¹)")
        ax_smooth.set_ylabel("Intensity (a.u.)")
        ax_smooth.legend(loc="best")
        _apply_plot_style(fig_smooth, ax_smooth, theme=plot_theme)
        _set_intensity_number_visibility(ax_smooth, show_preview_intensity_numbers)
        st.pyplot(fig_smooth)
        base_name = Path(meas_file_name).stem
        fig_smooth_png = _fig_to_bytes(fig_smooth, "png")
        fig_smooth_svg = _fig_to_bytes(fig_smooth, "svg")
        smooth_txt = _rebuild_file_bytes(
            header_lines_export,
            export_x,
            smooth_export_full,
            decimals=int(decimals),
            delimiter=delimiter_hint,
            keep_header_exact=keep_header_effective,
            extra_note=export_note,
        )
        smooth_suffix = "bc" if meas_mode == "BC" else "raw"
        col_smooth_png, col_smooth_svg, col_smooth_txt, _ = st.columns([1, 1, 1, 1])
        with col_smooth_png:
            st.download_button(
                "⬇️ Smoothing preview (PNG)",
                data=fig_smooth_png,
                file_name=f"{base_name}_smoothing_preview.png",
                mime="image/png",
                width="stretch",
            )
        with col_smooth_svg:
            st.download_button(
                "⬇️ Smoothing preview (SVG)",
                data=fig_smooth_svg,
                file_name=f"{base_name}_smoothing_preview.svg",
                mime="image/svg+xml",
                width="stretch",
            )
        with col_smooth_txt:
            st.download_button(
                "⬇️ Smoothed spectrum (TXT)",
                data=smooth_txt,
                file_name=f"{base_name}_smoothed_{smooth_suffix}.txt",
                mime="text/plain",
                width="stretch",
            )
    finally:
        plt.close(fig_smooth)

    if bool(smoothing_cfg.get("enabled", True)) and smooth_params is None:
        st.warning("Selected smoothing settings cannot be applied to this signal length. Matching will use the unsmoothed signal.")

    smoothing_ready_payload = {
        "meas_sha1": hashlib.sha1(meas_bytes).hexdigest(),
        "white_ref_ready_sig": white_ref_ready_sig,
        "baseline_ready_sig": baseline_ready_sig,
        "meas_mode": meas_mode,
        "smoothing": _smoothing_cfg_payload(smoothing_cfg),
    }
    smoothing_ready_sig = hashlib.sha1(json.dumps(smoothing_ready_payload, sort_keys=True).encode()).hexdigest()

    if st.session_state.get(smoothing_ready_key) != smoothing_ready_sig:
        st.info("Step 3/4: Adjust smoothing and confirm to unlock database matching.")
        if st.button(
            "Apply smoothing settings and continue to database matching",
            type="primary",
            width="stretch",
            key="approve_smoothing_btn",
        ):
            st.session_state[smoothing_ready_key] = smoothing_ready_sig
            st.session_state.pop("results_sig", None)
            st.session_state.pop("top_combined", None)
            _delay_then_rerun(2.0)
        st.stop()
    else:
        st.success("Step 3/4 complete: smoothing settings applied.")
        if st.button(
            "Adjust smoothing settings again",
            width="stretch",
            key="edit_smoothing_again_btn",
        ):
            st.session_state.pop(smoothing_ready_key, None)
            st.session_state.pop("results_sig", None)
            st.session_state.pop("top_combined", None)
            _safe_rerun()

    # Matching-Range kommt direkt vom oberen Slider
    range_low = int(round(rng_preview[0]))
    range_high = int(round(rng_preview[1]))

    mask = (meas_x_shifted_full >= range_low) & (meas_x_shifted_full <= range_high)
    if not np.any(mask):
        st.error("Selected range contains no points. Please adjust the range.")
        st.stop()
    meas_x = meas_x_shifted_full[mask]
    meas_y = meas_y_full[mask]
    if meas_x.size < 3:
        st.error("Selected range is too small (fewer than 3 points).")
        st.stop()
    elif meas_x.size < 30:
        st.warning(f"Warning: only {meas_x.size} points in the selected range; results may be unstable.")

    # Referenz-Ordner/Grids
    folders = active_folders
    grid_cfg = active_grid_cfg
    sig_base = _compute_signature_with_grid(list(folders), grid_cfg["min"], grid_cfg["max"], grid_cfg["step"])
    # Zwei Caches: DB-RAW und DB-BC (gleiche Grid-Config, anderer Signature-Suffix)
    sig_raw = f"{sig_base}-dbbc0"
    sig_bcb = f"{sig_base}-b{db_baseline_token}-dbbc1"

    sig_in = _inputs_signature(meas_bytes, range_low, range_high, ultra,
                               include_str, exclude_str, mode, allow_no_formula,
                               folders, sig_raw, sig_bcb, meas_mode, baseline_cfg, smoothing_cfg, white_ref_cfg,
                               float(meas_shift_cm1))
    # Suche startet erst nach bestätigter White-Light-Korrektur, Untergrundkorrektur UND Glättung.
    pack_raw, pack_bcb = _ensure_precompute_pair(
        signature_raw=sig_raw,
        signature_bcb=sig_bcb,
        folders=folders,
        grid_min=grid_cfg["min"],
        grid_max=grid_cfg["max"],
        grid_step=grid_cfg["step"],
        baseline_cfg=db_baseline_cfg,
    )

    include_set = _parse_element_list(include_str)
    exclude_set = _parse_element_list(exclude_str)
    filter_active = bool(include_set or exclude_set)
    allowed_fn = _make_element_filter_fn(include_set, exclude_set, mode, allow_no_formula)

    def _allowed_ids(meta):
        return np.array([i for i, m in enumerate(meta) if allowed_fn(m)], dtype=np.int32)

    allowed_ids_raw = _allowed_ids(pack_raw["meta"])
    allowed_ids_bcb = _allowed_ids(pack_bcb["meta"])

    if filter_active:
        st.write(
            f"🔎 **Filter active – eligible references:** RAW {allowed_ids_raw.size}/{len(pack_raw['meta'])} · BC {allowed_ids_bcb.size}/{len(pack_bcb['meta'])}"
        )

    if allowed_ids_raw.size == 0 and allowed_ids_bcb.size == 0:
        st.error("No reference spectrum matches the element filter.")
        st.stop()

    residual_matches_key = "top_combined_residual"
    residual_mode_key = "residual_mode_active"
    residual_info_key = "residual_search_info"
    residual_parent_sig_key = "residual_parent_sig"
    overlay_reset_key = "overlay_idx_reset_pending"

    if st.session_state.get("results_sig") != sig_in:
        top_primary = _compute_matches(
            meas_x, meas_y, range_low, range_high,
            pack_raw, pack_bcb,
            allowed_ids_raw, allowed_ids_bcb, baseline_cfg, smoothing_cfg, meas_mode,
            top_n=DEFAULT_TOP_N,
        )
        st.session_state["results_sig"] = sig_in
        st.session_state["top_combined"] = top_primary
        st.session_state["overlay_idx"] = 0
        st.session_state.pop(residual_matches_key, None)
        st.session_state.pop(residual_mode_key, None)
        st.session_state.pop(residual_info_key, None)
        st.session_state.pop(residual_parent_sig_key, None)
    else:
        top_primary = st.session_state.get("top_combined", [])
        if st.session_state.get(residual_parent_sig_key) != sig_in:
            st.session_state.pop(residual_matches_key, None)
            st.session_state.pop(residual_mode_key, None)
            st.session_state.pop(residual_info_key, None)
            st.session_state.pop(residual_parent_sig_key, None)

    residual_matches = st.session_state.get(residual_matches_key, [])
    residual_mode_active = bool(st.session_state.get(residual_mode_key, False)) and bool(residual_matches)
    top_combined = residual_matches if residual_mode_active else top_primary

    # Statuszeile
    db_names = ", ".join(Path(p).name for p in folders)
    if residual_mode_active:
        residual_info = st.session_state.get(residual_info_key, {})
        base_name = str(residual_info.get("base_name", "selected match"))
        base_file = str(residual_info.get("base_file", "")).strip()
        alpha = float(residual_info.get("alpha", 1.0))
        src_txt = f" ({base_file})" if base_file else ""
        st.success(
            f"Residual second-phase search active (subtracted: {base_name}{src_txt}, scale {alpha:.2f})  |  Top {DEFAULT_TOP_N} residual matches"
        )
    else:
        st.success(
            f"Top {DEFAULT_TOP_N} matches (meas: {meas_mode} only  ·  DB: RAW+BC (fixed: arPLS λ=1e4)  ·  grad {int(GRAD_WEIGHT*100)}%)  |  DBs: {db_names}  |  White ref: {_white_ref_label(white_ref_cfg)} [{white_ref_token}]  |  Baseline: {_baseline_label(baseline_cfg)}  |  Smoothing: {_smoothing_label(smoothing_cfg)} [{smoothing_token}]  |  Shift: {float(meas_shift_cm1):+.1f} cm⁻¹  |  Range: {range_low}–{range_high} cm⁻¹"
        )

    if not top_combined:
        st.info("No matches found. Check range, filter, or input file.")
        st.stop()

    # ——— Overlay figure first (same footprint as baseline/smoothing previews) ———
    if "overlay_idx" not in st.session_state:
        st.session_state.overlay_idx = 0
    if st.session_state.pop(overlay_reset_key, False):
        st.session_state["overlay_idx"] = 0

    def _fmt_opt(i: int) -> str:
        d = top_combined[i]
        formula = _format_formula(d.get("formula", "") or "—")
        # Nummeriert (ab 01), Formel statt Variant, "sim" -> "S:"
        return f"{i+1:02d} · {d['name']} · {formula} · S:{d['similarity']:.3f} · PCS:{float(d.get('pcs', 0.0)):.3f}"

    if st.session_state.overlay_idx >= len(top_combined):
        st.session_state.overlay_idx = 0

    # Eine Figur erzeugen, für Anzeige und Export verwenden
    sel = top_combined[st.session_state.overlay_idx]
    svg_data, png_data, filename_svg, filename_png = b"", b"", "", ""
    db_overlay_txt_data, filename_db_overlay_txt = b"", ""
    fig = None
    try:
        db_x, db_y = _parse_db_cached(str(sel["path"]))
        apply_baseline_db = bool(sel.get("db_baseline", sel.get("db_variant") == "DB-BC"))
        fig, ax = plt.subplots(figsize=(11, 4.6))
        _plot_overlay(
            meas_x, meas_y, db_x, db_y,
            f"{sel['name']}", baseline_cfg,
            ax=ax,
            baseline_cfg_db=db_baseline_cfg,
            apply_baseline_meas=(sel.get('meas_variant') == 'BC'),
            apply_baseline_db=apply_baseline_db,
            smoothing_cfg_meas=smoothing_cfg,
        )
        _apply_plot_style(fig, ax, theme=plot_theme)
        y_min, _y_max = ax.get_ylim()
        if np.isfinite(y_min):
            ax.set_ylim(bottom=max(-0.25, float(y_min)))
        _set_intensity_number_visibility(ax, show_preview_intensity_numbers)

        # Exportdaten vorbereiten
        svg_data = _fig_to_bytes(fig, "svg")
        png_data = _fig_to_bytes(fig, "png")

        # Anzeige in voller Breite (wie Baseline-/Smoothing-Preview)
        st.pyplot(fig)

        meas_name    = Path(meas_file_name).stem
        mineral_name = sel["name"].replace(" ", "_").replace("/", "_")
        var_tag      = f"meas-{sel.get('meas_variant','?')}_db-{'BC' if sel.get('db_baseline', False) else 'RAW'}"
        filename_svg = f"{meas_name}_fit_{mineral_name}_{var_tag}_{range_low}-{range_high}cm-1.svg"
        filename_png = f"{meas_name}_fit_{mineral_name}_{var_tag}_{range_low}-{range_high}cm-1.png"

        # Export der tatsächlich überlagerten DB-Kurve (entsprechend DB-RAW/DB-BC)
        lo, hi = float(np.min(db_x)), float(np.max(db_x))
        mask_db_overlay = (meas_x >= lo) & (meas_x <= hi)
        if int(np.count_nonzero(mask_db_overlay)) >= 2:
            x_clip = meas_x[mask_db_overlay]
            db_sig = _prepare_db_signal_on_target_grid(
                db_x,
                db_y,
                x_clip,
                apply_baseline_db=apply_baseline_db,
                baseline_cfg=db_baseline_cfg,
                smoothing_cfg_db={"enabled": False},
            )
            db_proc = _normalize_to_peak(db_sig)
            db_note = (
                f"Overlay DB trace ({'BC' if apply_baseline_db else 'RAW'}) "
                f"from {sel.get('orig_filename', '')}"
            )
            db_overlay_txt_data = _rebuild_file_bytes(
                [],
                x_clip,
                db_proc,
                decimals=int(decimals),
                delimiter="\t",
                keep_header_exact=False,
                extra_note=db_note,
            )
            filename_db_overlay_txt = (
                f"{meas_name}_overlay_db_{mineral_name}_"
                f"{'BC' if apply_baseline_db else 'RAW'}_{range_low}-{range_high}cm-1.txt"
            )

    except Exception as e:
        st.error(f"Error plotting overlay: {e}")
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            pass

    st.markdown("**Overlay selection**")
    st.selectbox(
        "Select a match to overlay",
        options=list(range(len(top_combined))),
        index=min(st.session_state.overlay_idx, len(top_combined)-1),
        format_func=_fmt_opt,
        key="overlay_idx",
    )
    if residual_mode_active:
        if st.button(
            "Back to primary match list",
            key="residual_back_to_primary_btn",
            width="stretch",
        ):
            st.session_state[residual_mode_key] = False
            st.session_state[overlay_reset_key] = True
            _safe_rerun()
    else:
        if st.button(
            "Search further phase from selected match (subtract and rematch)",
            key="run_residual_phase_search_btn",
            width="stretch",
        ):
            sel_for_residual = top_combined[int(st.session_state.overlay_idx)]
            use_bc = (meas_mode == "BC")
            q_sel, _q_l2_sel, q_mask = _prepare_query_vector(
                meas_x,
                meas_y,
                range_low,
                range_high,
                pack_raw["grid"],
                apply_baseline=use_bc,
                baseline_cfg=baseline_cfg,
                smoothing_cfg=smoothing_cfg,
            )
            residual_payload = _build_residual_query_vector(q_sel, q_mask, sel_for_residual, pack_raw, pack_bcb)
            if residual_payload is None:
                st.warning("Residual search could not be built from the selected match.")
            else:
                residual_q, alpha = residual_payload
                with st.spinner("Searching for additional phase on residual spectrum…"):
                    residual_top = _compute_matches_from_query_vector(
                        residual_q,
                        q_mask,
                        range_low,
                        range_high,
                        pack_raw,
                        pack_bcb,
                        allowed_ids_raw,
                        allowed_ids_bcb,
                        meas_mode,
                        top_n=DEFAULT_TOP_N,
                    )
                selected_mineral = _mineral_key(sel_for_residual.get("name", ""))
                residual_top = [d for d in residual_top if _mineral_key(d.get("name", "")) != selected_mineral]
                for d in residual_top:
                    d["residual_phase"] = True
                if not residual_top:
                    st.warning("No additional phase candidates found in residual search.")
                else:
                    st.session_state[residual_matches_key] = residual_top[:DEFAULT_TOP_N]
                    st.session_state[residual_mode_key] = True
                    st.session_state[residual_parent_sig_key] = sig_in
                    st.session_state[residual_info_key] = {
                        "base_name": sel_for_residual.get("name", ""),
                        "base_file": sel_for_residual.get("orig_filename", ""),
                        "alpha": float(alpha),
                    }
                    st.session_state[overlay_reset_key] = True
                    _safe_rerun()
    if svg_data and png_data and filename_svg and filename_png:
        col_ov_svg, col_ov_png, col_ov_txt, _ = st.columns([1, 1, 1, 1])
        with col_ov_svg:
            st.download_button(
                label="📊 Export as SVG",
                data=svg_data,
                file_name=filename_svg,
                mime="image/svg+xml",
                width="stretch",
            )
        with col_ov_png:
            st.download_button(
                label="📊 Export as PNG",
                data=png_data,
                file_name=filename_png,
                mime="image/png",
                width="stretch",
            )
        with col_ov_txt:
            if db_overlay_txt_data and filename_db_overlay_txt:
                st.download_button(
                    label="📄 Overlay DB (TXT)",
                    data=db_overlay_txt_data,
                    file_name=filename_db_overlay_txt,
                    mime="text/plain",
                    width="stretch",
                )

    with st.expander("Details table", expanded=False):
        import pandas as pd
        df_full = pd.DataFrame(
            {
                "ID":         list(range(1, len(top_combined) + 1)),  # passt zur Dropdown-Nummer
                "Mineral":    [d["name"]                              for d in top_combined],
                "Formula":    [_format_formula(d.get("formula",""))   for d in top_combined],
                "Flag":       [d.get("flag","")                       for d in top_combined],
                "Variant":    [f"meas:{d['meas_variant']} · {('DB-BC' if d.get('db_baseline', False) else 'DB-RAW')}" for d in top_combined],
                "Similarity": [d["similarity"]                        for d in top_combined],
                "PCS":        [float(d.get("pcs", 0.0))               for d in top_combined],
                "File":       [d.get("orig_filename","")              for d in top_combined],
            }
        )
        st.dataframe(df_full, hide_index=True, width="stretch", height=420)


    # –– Overlay by mineral names (selected databases) — same layout/handling as above
    st.subheader("Overlay by mineral names")

    help_txt = (
        "Separate mineral names with commas, semicolons, or new lines "
        "(e.g. 'quartz, calcite'). Exactly one entry per mineral "
        "can be selected from the active databases and overlaid. "
        f"Active matching range: {range_low}–{range_high} cm⁻¹."
    )

    names_in = st.text_area(
        "Minerals",
        height=90,
        help=help_txt,
        placeholder="quartz, calcite",
        key="names_in_clean",
    )

    chosen_entries = []
    missing_names = []
    overlay_specs = []

    if names_in:
        raw_names = re.split(r"[,\n;]+", names_in)
        requested = [n.strip() for n in raw_names if n.strip()]
        requested_unique = []
        seen = set()
        for n in requested:
            k = n.casefold()
            if k not in seen:
                seen.add(k)
                requested_unique.append(n)

        # Ausgewählte Datenbanken laden (gecacht)
        clean_entries = active_entries

        from collections import defaultdict
        name_index = defaultdict(list)
        for e in clean_entries:
            name_index[e["name"].casefold()].append(e)

        for n in requested_unique[:MAX_OVERLAY_MINERALS]:
            lst = name_index.get(n.casefold(), [])
            if not lst:
                missing_names.append(n)
                continue

            lst_sorted = sorted(
                lst,
                key=lambda e: ({"s": 0, "": 1, "x": 2}.get(e.get("flag", ""), 3), e.get("orig_filename", "")),
            )

            key_base = f"minsel_{n.casefold()}"
            sel_key = key_base + "_sel"
            ed_key = key_base + "_editor"

            if sel_key not in st.session_state:
                st.session_state[sel_key] = 0
            st.session_state[sel_key] = int(np.clip(st.session_state[sel_key], 0, len(lst_sorted) - 1))

            overlay_specs.append((n, lst_sorted, sel_key, ed_key))
            chosen_entries.append(lst_sorted[st.session_state[sel_key]])

    # Baseline-Schalter analog oben (für DB-Kurven im Namen-Overlay)
    apply_db_baseline_for_names = False
    if names_in:
        apply_db_baseline_for_names = st.checkbox(
            "Apply baseline correction to DB traces (name overlay)",
            value=False,
            key="names_db_baseline",
            help="Applies fixed DB baseline settings (arPLS, λ=10^4, standard defaults) to selected DB traces before resampling.",
        )

    # Plot in voller Breite, Auswahlfelder darunter
    fig2 = None
    svg_data2, filename2 = b"", ""
    if names_in and chosen_entries:
        try:
            fig2, ax2 = plt.subplots(figsize=(11, 4.6))

            # Messung gemäß gewähltem Matching-Modus plotten
            meas_sig = _prepare_measurement_signal(
                meas_y,
                apply_baseline=(meas_mode == "BC"),
                baseline_cfg=baseline_cfg,
                smoothing_cfg=smoothing_cfg,
            )
            meas_proc = _normalize_to_peak(meas_sig)
            ax2.plot(meas_x, meas_proc, label=f"measurement ({meas_mode})", lw=1.2)

            # jede gewählte DB-Kurve nur im Überlappungsbereich einzeichnen
            for e in chosen_entries:
                p = e["path"]
                db_x, db_y = _parse_db_cached(str(p))
                lo, hi = float(np.min(db_x)), float(np.max(db_x))
                mask2 = (meas_x >= lo) & (meas_x <= hi)
                if mask2.sum() < 2:
                    continue
                x_clip = meas_x[mask2]
                db_sig = _prepare_db_signal_on_target_grid(
                    db_x,
                    db_y,
                    x_clip,
                    apply_baseline_db=apply_db_baseline_for_names,
                    baseline_cfg=db_baseline_cfg,
                    smoothing_cfg_db={"enabled": False},
                )
                db_proc = _normalize_to_peak(db_sig)
                flag = e.get("flag", "")
                fname = e.get("orig_filename", "")
                label = f"{e['name']} · {flag}{fname}".strip()
                ax2.plot(x_clip, db_proc, label=label, lw=1.0, alpha=0.65)

            ax2.set_xlabel("Raman shift (cm⁻¹)")
            ax2.set_ylabel("normalised intensity")
            ax2.legend(loc="best")
            _apply_plot_style(fig2, ax2, theme=plot_theme)
            _set_intensity_number_visibility(ax2, show_preview_intensity_numbers)
            svg_data2 = _fig_to_bytes(fig2, "svg")

            st.pyplot(fig2)

            meas_name = Path(meas_file_name).stem
            short_names = "_".join([e["name"].split()[0] for e in chosen_entries])[:60].replace("/", "_")
            filename2 = f"{meas_name}_overlay_{short_names or 'minerals'}_{range_low}-{range_high}cm-1.svg"

        except Exception as exc:
            st.error(f"Error plotting name overlays: {exc}")
        finally:
            try:
                if fig2 is not None:
                    plt.close(fig2)
            except Exception:
                pass

    if svg_data2 and filename2:
        st.download_button(
            label="📥 Export as SVG",
            data=svg_data2,
            file_name=filename2,
            mime="image/svg+xml",
            width="stretch",
        )

    if names_in and missing_names:
        for n in missing_names:
            st.info(f"Not found in selected databases: {n}")

    if names_in and overlay_specs:
        st.markdown("**Phase entry selection**")
        overlay_changed = False

        def _onehot(length: int, idx: int) -> list[bool]:
            return [i == idx for i in range(length)]

        for n, lst_sorted, sel_key, ed_key in overlay_specs:
            df_miner = pd.DataFrame(
                {
                    "Pick": _onehot(len(lst_sorted), st.session_state[sel_key]),
                    "Flag": [e.get("flag", "") for e in lst_sorted],
                    "File": [e.get("orig_filename", "") for e in lst_sorted],
                }
            )

            st.markdown(f"**{n}** – choose one entry from active databases")
            edited_m = st.data_editor(
                df_miner,
                key=ed_key,
                num_rows="fixed",
                column_config={
                    "Pick": st.column_config.CheckboxColumn("Pick", help="Select one entry per mineral"),
                },
                disabled=["Flag", "File"],
                hide_index=True,
                width="stretch",
            )

            checked = edited_m.index[edited_m["Pick"]].tolist()
            if checked:
                prev_idx = st.session_state[sel_key]
                if len(checked) == 1:
                    new_idx = checked[0]
                else:
                    new_idx = next((i for i in checked if i != prev_idx), checked[0])
                if new_idx != prev_idx:
                    st.session_state[sel_key] = new_idx
                    overlay_changed = True

        if overlay_changed:
            _safe_rerun()

    st.empty()

# ────────────────────────────────────────────────────────────────────────────
# CLI-Placeholder

def _cli_main(argv: list[str]) -> None:
    pass


# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_gui = len(sys.argv) == 1 or "streamlit" in Path(sys.argv[0]).name.lower()
    if run_gui:
        _run_streamlit()
    else:
        _cli_main(sys.argv[1:])

if __name__ == "__main__":
    main()

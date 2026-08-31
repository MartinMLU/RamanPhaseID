from __future__ import annotations                      # muss oberste Zeile sein!

# ----- BLAS/OMP auf 1 Thread begrenzen (vor NumPy / SciPy-Import) ----------
import os

for var in (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

# --------------------------------------------------------------------------- #
import csv
import math
import multiprocessing as mp
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

# ---------- Zahl-Zahl-Zeile --------------------------------------------------
_NUMERIC_LINE_PATTERN = re.compile(
    r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"(?:[,\s;]+[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)+\s*$"
)
_FLOAT_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _parse_numeric_xy_line(line: str) -> Tuple[float, float] | None:
    if not _NUMERIC_LINE_PATTERN.match(line):
        return None
    vals = _FLOAT_PATTERN.findall(line)
    if len(vals) < 2:
        return None
    try:
        return float(vals[0]), float(vals[-1])
    except Exception:
        return None

# --------------------------------------------------------------------------- #
#                        Quick-Meta (Header only)                              #
# --------------------------------------------------------------------------- #
def _quick_meta_rod(path: Path) -> Tuple[str, str]:
    """
    Liest nur die ersten ~200 Zeilen, um Mineral & Formel zu finden.
    Verzichtet auf fh.tell() → kein 'telling position disabled by next() call'.
    """
    mineral = formula = None
    with path.open("r", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if line.startswith("_chemical_name_mineral") and mineral is None:
                parts = line.split(None, 1)
                if len(parts) > 1 and parts[1].strip():
                    mineral = parts[1].strip().strip("'\"")
                else:                                 # Wert steht evtl. in Folgezeile
                    nxt = fh.readline()
                    if nxt and not nxt.lstrip().startswith("_"):
                        mineral = nxt.strip().strip("'\"")
            elif line.startswith("_chemical_formula_sum") and formula is None:
                parts = line.split(None, 1)
                if len(parts) > 1 and parts[1].strip():
                    formula = parts[1].strip().strip("'\"")
                else:
                    nxt = fh.readline()
                    if nxt and not nxt.lstrip().startswith("_"):
                        formula = nxt.strip().strip("'\"")
            if mineral and formula:
                break
            if i >= 200:          # nach 200 Zeilen abbrechen (Performance)
                break
    return mineral or "Unknown", formula or "?"


def _quick_meta_rruff(path: Path) -> Tuple[str, str]:
    mineral = formula = None
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("##"):
                break
            if line.startswith("##NAMES=") and mineral is None:
                mineral = line.split("=", 1)[1].split(";")[0].strip()
            elif line.startswith("##IDEAL CHEMISTRY=") and formula is None:
                formula = line.split("=", 1)[1].strip().replace("_", "")
    return mineral or "Unknown", formula or "?"


# --------------------------------------------------------------------------- #
#                        Full-Parser (Worker-Phase)                           #
# --------------------------------------------------------------------------- #
def _parse_rod(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    shifts, intensities = [], []
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            xy = _parse_numeric_xy_line(line)
            if xy is not None:
                shifts.append(xy[0])
                intensities.append(xy[1])
    if not shifts:
        raise ValueError(f"no spectral data in {path}")
    return np.asarray(shifts), np.asarray(intensities)


def _parse_rruff(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    shifts, intensities = [], []
    with path.open("r", errors="ignore") as fh:
        in_header = True
        for line in fh:
            if in_header and line.startswith("##"):
                continue
            in_header = False
            xy = _parse_numeric_xy_line(line)
            if xy is not None:
                shifts.append(xy[0])
                intensities.append(xy[1])
    if not shifts:
        raise ValueError(f"no spectral data in {path}")
    return np.asarray(shifts), np.asarray(intensities)


# --------------------------------------------------------------------------- #
#                        Measurement-Parser                                   #
# --------------------------------------------------------------------------- #
_MEASUREMENT_DELIMITERS: tuple[str | None, ...] = (",", "\t", ";", None)
_DECIMAL_COMMA_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:,\d*)?|,\d+)(?:[eE][+-]?\d+)?$"
)


def _measurement_fields(line: str, delimiter: str | None) -> list[str]:
    """Split one measurement record using one already-selected delimiter."""
    if delimiter is None:
        return re.split(r"\s+", line.strip())
    try:
        return next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))
    except csv.Error:
        return []


def _measurement_token(token: str | None, delimiter: str | None) -> str | None:
    if token is None:
        return None
    value = token.strip().lstrip("\ufeff")
    # Decimal commas are unambiguous when comma is not the field delimiter.
    if delimiter != "," and _DECIMAL_COMMA_PATTERN.fullmatch(value):
        value = value.replace(",", ".")
    return value


def _measurement_rows(
    text: str,
    delimiter: str | None,
) -> list[tuple[int, str | None, str | None]]:
    """Return the first two physical fields of each non-comment record."""
    rows: list[tuple[int, str | None, str | None]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = _measurement_fields(line, delimiter)
        first = fields[0] if len(fields) >= 1 else None
        second = fields[1] if len(fields) >= 2 else None
        rows.append(
            (
                line_number,
                _measurement_token(first, delimiter),
                _measurement_token(second, delimiter),
            )
        )
    return rows


def _coerce_measurement_rows(
    rows: list[tuple[int, str | None, str | None]],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["line", "shift", "intensity"])
    if frame.empty:
        return frame
    # Explicit coercion prevents ordinary, uncommented header text from leaking
    # into the returned arrays as object/string values.
    frame["shift"] = pd.to_numeric(frame["shift"], errors="coerce")
    frame["intensity"] = pd.to_numeric(frame["intensity"], errors="coerce")
    return frame


def _detect_measurement_delimiter(text: str) -> str | None:
    """Choose comma, tab, semicolon, or whitespace deterministically."""
    best_delimiter: str | None = None
    best_numeric_rows = -1
    for delimiter in _MEASUREMENT_DELIMITERS:
        frame = _coerce_measurement_rows(_measurement_rows(text, delimiter))
        if frame.empty:
            numeric_rows = 0
        else:
            numeric_rows = int(
                (
                    np.isfinite(frame["shift"].to_numpy(dtype=float))
                    & np.isfinite(frame["intensity"].to_numpy(dtype=float))
                ).sum()
            )
        # Candidate order is the documented tie-breaker; do not replace an
        # equally scoring earlier candidate.
        if numeric_rows > best_numeric_rows:
            best_numeric_rows = numeric_rows
            best_delimiter = delimiter
    if best_numeric_rows <= 0:
        raise ValueError(
            "No numeric measurement rows were found in the first two columns "
            "(supported delimiters: comma, tab, semicolon, or whitespace)."
        )
    return best_delimiter


def parse_measurement(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a measurement spectrum into canonical, numeric ``(x, y)`` arrays.

    The first and second *physical* columns are always Raman shift and intensity;
    any later columns are ignored. Leading nonnumeric records are treated as a
    header. Once numeric data have started, a malformed/interspersed record is an
    error rather than being silently skipped or remapped.

    An input axis may be nondecreasing or nonincreasing. Exact duplicate shifts
    are consolidated by their arithmetic-mean intensity, and output is strictly
    increasing. A non-monotonic axis is rejected because interpolation and
    point-spacing-dependent preprocessing otherwise become ambiguous.
    """
    if not isinstance(text, str):
        raise TypeError("Measurement input must be decoded text.")

    delimiter = _detect_measurement_delimiter(text)
    frame = _coerce_measurement_rows(_measurement_rows(text, delimiter))
    shift_values = frame["shift"].to_numpy(dtype=float)
    intensity_values = frame["intensity"].to_numpy(dtype=float)
    valid = np.isfinite(shift_values) & np.isfinite(intensity_values)

    data_positions = np.flatnonzero(valid)
    first_data_position = int(data_positions[0])
    malformed_positions = np.flatnonzero(~valid & (np.arange(valid.size) > first_data_position))
    if malformed_positions.size:
        line_numbers = frame.iloc[malformed_positions]["line"].astype(int).tolist()
        preview = ", ".join(str(number) for number in line_numbers[:5])
        if len(line_numbers) > 5:
            preview += ", ..."
        raise ValueError(
            "Malformed measurement row after numeric data started "
            f"(line{'s' if len(line_numbers) != 1 else ''} {preview}); "
            "the first two columns must both be finite numbers."
        )

    x = shift_values[valid]
    y = intensity_values[valid]
    if x.size < 2:
        raise ValueError("A measurement spectrum must contain at least two numeric rows.")

    dx = np.diff(x)
    is_nondecreasing = bool(np.all(dx >= 0.0))
    is_nonincreasing = bool(np.all(dx <= 0.0))
    if not (is_nondecreasing or is_nonincreasing):
        raise ValueError(
            "Raman-shift values must be monotonic (entirely ascending or "
            "entirely descending); sort or repair the input axis first."
        )

    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if unique_x.size < 2:
        raise ValueError(
            "A measurement spectrum must contain at least two distinct Raman shifts."
        )
    summed_y = np.bincount(inverse, weights=y, minlength=unique_x.size)
    unique_y = summed_y / counts.astype(float)
    return unique_x.astype(float, copy=False), unique_y.astype(float, copy=False)


# --------------------------------------------------------------------------- #
#                    Signal-Utilities                                         #
# --------------------------------------------------------------------------- #
def _baseline_als(y: np.ndarray, lam=1e5, p=0.01, niter=10) -> np.ndarray:
    L = len(y)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sp.diags(w, 0)
        z = spsolve(W + lam * D.T @ D, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _smooth(y: np.ndarray, window=11, poly=3):
    return savgol_filter(y, window, poly) if len(y) >= window else y


def _normalize(y: np.ndarray):
    y = y - np.min(y)
    ymax = np.max(y)
    return y / ymax if ymax else y


def _resample(src_x, src_y, tgt_x):
    return interp1d(src_x, src_y, kind="linear", bounds_error=False, fill_value=0.0)(tgt_x)


def _resample_uniform_support(src_x, src_y, step_cm1: float = 1.0):
    """Resample finite data to a globally anchored physical Raman-shift grid."""
    x = np.asarray(src_x, dtype=float).reshape(-1)
    y = np.asarray(src_y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must contain the same number of values")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return x, y

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        sums = np.bincount(inverse, weights=y)
        counts = np.bincount(inverse)
        x = unique_x
        y = sums / np.maximum(counts, 1)
    if x.size < 2:
        return x, y

    step = float(step_cm1)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("uniform resampling step must be positive")
    first = math.ceil((float(x[0]) / step) - 1e-9)
    last = math.floor((float(x[-1]) / step) + 1e-9)
    if last < first:
        return x, y
    grid = np.arange(first, last + 1, dtype=float) * step
    return grid, np.interp(grid, x, y)


def _cosine(a, b):
    dot = np.dot(a, b)
    nrm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / nrm if nrm else 0.0


# --------------------------------------------------------------------------- #
#                        Header-Loader (rekursiv)                              #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def load_reference_folders(
    folders: Tuple[str | Path, ...],
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """
    Rekursiv *.rod & *.txt scannen, nur Header meta einlesen.

    • Qualitäts-Präfix s / x wird erkannt  
    • Rückgabe: (entries, skipped)
    """
    paths: list[Path] = []
    skipped: list[Tuple[str, str]] = []

    for folder in folders:
        f = Path(folder)
        if not f.exists():
            skipped.append((str(f), "folder does not exist"))
            continue
        paths += list(f.rglob("*.rod"))
        paths += list(f.rglob("*.txt"))

    paths.sort(key=lambda p: p.stem.lstrip("sSxX"))

    entries: List[Dict] = []
    for p in paths:
        try:
            mineral, formula = (
                _quick_meta_rod(p) if p.suffix.lower() == ".rod" else _quick_meta_rruff(p)
            )
            flag = p.stem[0].lower() if p.stem and p.stem[0].lower() in ("s", "x") else ""
            numeric = p.stem.lstrip("sSxX")
            entries.append(
                {
                    "name": mineral,
                    "formula": formula,
                    "filename": f"{numeric}{p.suffix}",
                    "orig_filename": p.name,
                    "flag": flag,
                    "path": p,
                }
            )
        except Exception as exc:  # pragma: no cover
            skipped.append((p.name, str(exc)))

    return entries, skipped


# --------------------------------------------------------------------------- #
#                        Worker-Funktion                                      #
# --------------------------------------------------------------------------- #
def _similarity_worker(entry: Dict, meas_x: np.ndarray, meas_proc: np.ndarray):
    p = entry["path"]
    db_x, db_y = (_parse_rruff if p.suffix.lower() == ".txt" else _parse_rod)(p)
    # Reference spectra are interpolated to common support but never smoothed.
    db_proc = _normalize(_resample(db_x, db_y, meas_x))
    return _cosine(meas_proc, db_proc), entry


# --------------------------------------------------------------------------- #
#                        Matching-Engine                                      #
# --------------------------------------------------------------------------- #
def rank_matches(
    meas_x: np.ndarray,
    meas_y: np.ndarray,
    db_entries: List[Dict],
    *,
    top_n: int = 30,
    workers: int | None = None,
    chunk: int = 500,
    safe: bool = False,
):
    """
    Chunk-basierte, parallele Cosinus-Suche.

    safe=True   → workers=1, chunk=100  (Low-RAM, Freeze-sicher)
    """
    if safe:
        workers, chunk = 1, 100

    if workers is None:
        workers = max(1, (mp.cpu_count() or 1) - 1)
    workers = max(1, workers)

    # Legacy API: keep lambda/window semantics fixed by preprocessing the
    # measurement on the same 1 cm⁻¹ physical grid used by the main app.
    meas_x, meas_y = _resample_uniform_support(meas_x, meas_y, step_cm1=1.0)
    meas_proc = _normalize(_smooth(meas_y - _baseline_als(meas_y)))

    results: List[Dict] = []
    for i in range(0, len(db_entries), chunk):
        batch = db_entries[i : i + chunk]

        if workers == 1:
            sims = [_similarity_worker(e, meas_x, meas_proc) for e in batch]
        else:
            with mp.get_context("spawn").Pool(workers, maxtasksperchild=100) as pool:
                sims = pool.starmap(
                    _similarity_worker,
                    ((e, meas_x, meas_proc) for e in batch),
                    chunksize=max(1, len(batch) // workers // 4),
                )

        results.extend({**e, "similarity": s} for s, e in sims)

    results.sort(key=lambda d: d["similarity"], reverse=True)
    return results[:top_n]

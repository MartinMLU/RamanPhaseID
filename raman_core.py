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
import io
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
def parse_measurement(text: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"\t|,|\s+",
        engine="python",
        comment="#",
        header=None,
        names=["shift", "intensity"],
    ).dropna()
    return df["shift"].to_numpy(), df["intensity"].to_numpy()


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
    db_proc = _normalize(_smooth(_resample(db_x, db_y, meas_x)))
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raman-Cluster-Auswahl (nach vorgelagerter Dedup-Stufe)

- Gruppiert Spektren pro Mineral (name)
- Bestimmt automatisch k in [3..9] via Silhouette-Scan (Cosinus), wenn n>4
- Clustert per K-Medoids (Cosinus) mit robuster Initialisierung; Fallback: GMM (BIC-Minimum, k>=3)
- Wählt pro Cluster den Medoid als Repräsentant (Qualitätstiebreak: 's' < '' < 'x')
- Verwirft keine Spektren aufgrund zu weniger Peaks/Banden
- Spektren außerhalb REF_X werden immer übernommen (ungeclustert)
- Kopiert nur Repräsentanten + Outside-Spektren in den Zielordner
- Schreibt CSV-Reports (Gesamt & je Mineral)

Standardlauf (z. B. VS Code ohne Argumente):
    Eingabe:  databases/OWN (Fallback: CleanDB)
    Ausgabe:  ClusterDB
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Externe Abhängigkeiten
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

try:
    from sklearn_extra.cluster import KMedoids
    HAVE_KMEDOIDS = True
except Exception:
    HAVE_KMEDOIDS = False

# gezielt die bekannte Warnung von sklearn-extra unterdrücken (falls der Lib-Code sie doch wirft)
warnings.filterwarnings(
    "ignore",
    message=r"Cluster \d+ is empty! .*",
    module=r"sklearn_extra\.cluster\._k_medoids",
)

# Deine Hilfsbibliothek (wie im vorhandenen Skript)
import raman_core as rc

# ---------------------------------------------------------------------------
# Standard-Eingabeordner robust auf altes und neues Layout auflösen.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CANDIDATES = (
    BASE_DIR / "databases" / "OWN",
    BASE_DIR / "CleanDB",
)

def _default_input_folder() -> str:
    for p in DEFAULT_INPUT_CANDIDATES:
        if p.exists():
            return str(p)
    return str(DEFAULT_INPUT_CANDIDATES[0])

# ---------------------------------------------------------------------------
# Referenzachse für Fingerprints (50–2000 cm⁻1 in 1er-Schritten)
REF_X = np.arange(50, 2001, 1)

# ---------------------------------------------------------------------------
# Einfache ROD-Parser (wie in deinem Skript), falls benötigt

_FLOATS_RE = re.compile(r'[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?').findall

def _rod_parse_prefer_loop(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    in_loop = False
    have_tags = False
    tags: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                s = raw.strip()
                if s.startswith("loop_"):
                    in_loop = True; have_tags = False; tags = []; continue
                if not in_loop:
                    continue
                if s.startswith("_"):
                    tags.append(s); continue
                if not have_tags:
                    if "_raman_spectrum.raman_shift" in tags:
                        have_tags = True
                    else:
                        in_loop = False
                        continue
                if s == "" or s.startswith(("loop_", "data_", "_", "#", ";")):
                    if x_vals:
                        break
                    in_loop = s.startswith("loop_"); have_tags = False; tags = []
                    if not in_loop:
                        in_loop = False
                    continue
                nums = _FLOATS_RE(s)
                if len(nums) >= 2:
                    try:
                        x_vals.append(float(nums[0])); y_vals.append(float(nums[-1]))
                    except ValueError:
                        pass
    except Exception as exc:
        return np.array([]), np.array([]), f"io error: {exc}"
    if x_vals:
        return np.array(x_vals), np.array(y_vals), None
    return np.array([]), np.array([]), "no _raman_spectrum loop found"

def _rod_parse_any_pairs(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    x_vals: List[float] = []; y_vals: List[float] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                s = raw.strip()
                if not s or s.startswith(("#", "data_")):
                    continue
                nums = _FLOATS_RE(s)
                if len(nums) >= 2:
                    try:
                        x_vals.append(float(nums[0])); y_vals.append(float(nums[-1]))
                    except ValueError:
                        continue
    except Exception as exc:
        return np.array([]), np.array([]), f"io error: {exc}"
    if len(x_vals) >= 5:
        return np.array(x_vals), np.array(y_vals), None
    return np.array([]), np.array([]), "no numeric pairs found"

def _parse_rod(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    x, y, err = _rod_parse_prefer_loop(path)
    if err is None:
        return x, y, None
    x2, y2, err2 = _rod_parse_any_pairs(path)
    if err2 is None:
        return x2, y2, None
    return np.array([]), np.array([]), f"rod parse failed: {err}; fallback: {err2}"

# ---------------------------------------------------------------------------

@dataclass
class Entry:
    """Ein Eintrag aus der Referenzdatenbank (nach deiner ersten Stufe)."""
    name: str
    path: Path
    orig_filename: str
    flag: str  # 's', '', 'x'

@dataclass
class FPInfo:
    """Fingerprint-Infos für Clustering."""
    fp: Optional[np.ndarray]  # L2-normalisierter Fingerprint auf REF_X oder None
    xmin: float
    xmax: float
    kept_outside: bool        # Außerhalb REF_X → wird immer behalten (ungeclustert)
    err: Optional[str]

@dataclass
class ClusterResult:
    labels: np.ndarray        # shape (n_valid,)
    reps_idx: List[int]       # Indizes relativ zu validen Items
    scores: Dict[str, float]  # z.B. silhouette, bic, k, method

# ---------------------------------------------------------------------------

def _sort_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    return x[idx], y[idx]

def _make_fingerprint(entry: Entry) -> FPInfo:
    try:
        p = entry.path
        if p.suffix.lower() == ".txt":
            try:
                x, y = rc._parse_rruff(p); err = None
            except Exception as exc:
                return FPInfo(None, 0.0, 0.0, False, f"rruff parse failed: {exc}")
        else:
            x, y, err = _parse_rod(p)
            if err:
                return FPInfo(None, 0.0, 0.0, False, err)

        if len(x) == 0 or len(y) == 0:
            return FPInfo(None, 0.0, 0.0, False, "empty spectrum")

        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        x, y = _sort_xy(x, y)

        xmin, xmax = float(x[0]), float(x[-1])
        ref_min, ref_max = float(REF_X[0]), float(REF_X[-1])
        if xmax < ref_min or xmin > ref_max:
            return FPInfo(None, xmin, xmax, True, None)

        y_corr = rc._normalize(y - rc._baseline_als(y))
        fp = rc._resample(x, y_corr, REF_X)

        if not np.isfinite(fp).all():
            return FPInfo(None, xmin, xmax, False, "non-finite values")
        if np.linalg.norm(fp) == 0.0:
            return FPInfo(None, xmin, xmax, True, None)

        nrm = np.linalg.norm(fp)
        if nrm > 0:
            fp = fp / nrm

        return FPInfo(fp, xmin, xmax, False, None)

    except Exception as exc:
        return FPInfo(None, 0.0, 0.0, False, f"unexpected error: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# Robustes PCA-Embedding

def pca_embed(X: np.ndarray, n_comp: int = 30, seed: int = 42) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Liefert (Z, pca). Falls PCA aufgrund geringer Stichprobe keine echte Reduktion
    erlaubt/sinnvoll ist, wird Z = X zurückgegeben und pca=None.
    Regel: n_components_eff <= min(n_samples-1, n_features), mindestens 1.
    """
    n_samples, n_features = X.shape
    n_comp_eff = max(1, min(int(n_comp), n_features, n_samples - 1))
    if n_comp_eff >= n_features:
        return X, None
    pca = PCA(n_components=n_comp_eff, random_state=seed)
    Z = pca.fit_transform(X)
    return Z, pca

# ---------------------------------------------------------------------------
# K-Scan mit KMeans (stabil, keine sklearn_extra-Warnungen)

def _fit_kmeans_labels(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    try:
        labels = KMeans(n_clusters=k, n_init='auto', random_state=seed).fit_predict(Z)
    except TypeError:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return labels

def choose_k_by_silhouette(Z: np.ndarray, k_min=3, k_max=9, metric="cosine", seed=42) -> Tuple[int, float]:
    """
    Wählt k in [max(2, k_min) .. min(k_max, n-1, n_unique-1)] anhand des Silhouette-Scores.
    Labels stammen bewusst **von KMeans**, um leere K-Medoids-Cluster beim Scan zu vermeiden.
    """
    n = len(Z)
    if n < 3:
        return 2, -1.0

    n_unique = np.unique(Z, axis=0).shape[0]
    k_min_eff = max(2, min(k_min, n - 1, n_unique - 1))
    k_max_eff = max(k_min_eff, min(k_max, n - 1, n_unique - 1))

    best_k, best_s = k_min_eff, -1.0

    for k in range(k_min_eff, k_max_eff + 1):
        try:
            labels = _fit_kmeans_labels(Z, k, seed)
            n_labels = len(np.unique(labels))
            if n_labels < 2 or n_labels >= n:
                continue
            s = silhouette_score(Z, labels, metric=metric)
            if s > best_s:
                best_k, best_s = k, s
        except Exception:
            continue

    return best_k, best_s

# ---------------------------------------------------------------------------
# Robuster K-Medoids-Fit (mehrere Seeds; keine leeren Cluster)

def _fit_kmedoids_robust(Z: np.ndarray, k: int, seed: int) -> Optional[np.ndarray]:
    """
    Versucht K-Medoids mehrfach mit verschiedenen Seeds.
    Gibt labels (shape (n,)) zurück, wenn alle k Cluster belegt sind – sonst None.
    """
    if not HAVE_KMEDOIDS or k < 2:
        return None

    for attempt in range(6):  # 1 + 5 Versuche
        rs = seed + attempt
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Cluster \d+ is empty! .*",
                                    module=r"sklearn_extra\.cluster\._k_medoids")
            km = KMedoids(
                n_clusters=k,
                metric="cosine",
                init="k-medoids++",
                random_state=rs,
            )
            labels = km.fit_predict(Z)

        labs = np.unique(labels)
        if len(labs) == k and set(labs.tolist()) == set(range(k)):
            return labels

    return None

# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    labels: np.ndarray
    reps_idx: List[int]
    scores: Dict[str, float]

def cluster_group(X: np.ndarray, flags: List[str], method='auto',
                  k_min=3, k_max=9, seed=42, dim: int = 30) -> ClusterResult:
    n = X.shape[0]
    if n <= 4:
        labels = np.arange(n)
        return ClusterResult(labels=labels, reps_idx=list(range(n)),
                             scores={'method': 'trivial', 'k': n})

    Z, _ = pca_embed(X, n_comp=min(dim, X.shape[1]), seed=seed)

    # k bestimmen (Silhouette via KMeans)
    k, sil = choose_k_by_silhouette(Z, k_min=max(3, k_min), k_max=min(9, k_max), metric="cosine", seed=seed)

    # k an Zahl einzigartiger Punkte anpassen (verhindert triviale Leercuster)
    n_unique = np.unique(Z, axis=0).shape[0]
    k = min(k, max(2, n_unique - 1, 3))  # mind. 3, sofern möglich

    labels = None
    chosen: Dict[str, float] = {}

    # Versuche K-Medoids robust
    if method in ('auto', 'kmedoids') and HAVE_KMEDOIDS:
        kk = k
        while kk >= 2 and labels is None:
            labels = _fit_kmedoids_robust(Z, kk, seed)
            if labels is None:
                kk -= 1
        if labels is not None:
            if (n > 4) and (kk < 3):
                labels = None  # Mindest-k nicht unterschreiten; später GMM
            else:
                chosen.update({'method': 'kmedoids', 'k': float(kk), 'silhouette': float(sil)})

    # Fallback: GMM
    if labels is None and method in ('auto', 'gmm'):
        best_bic, best_model = np.inf, None
        for kk in range(max(3, k_min), min(9, k_max, n - 1) + 1):
            gmm = GaussianMixture(n_components=kk, covariance_type='diag', random_state=seed)
            gmm.fit(Z)
            bic = gmm.bic(Z)
            if bic < best_bic:
                best_bic, best_model = bic, gmm
        labels = best_model.predict(Z)
        chosen.update({'method': 'gmm', 'k': float(len(np.unique(labels))), 'bic': float(best_bic)})

    # Medoide je Cluster (Cosinus), Qualitäts-Tiebreak
    reps: List[int] = []
    qrank = {'s': 0, '': 1, 'x': 2}.get

    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        if len(idx) == 1:
            reps.append(idx[0]); continue
        Xi = normalize(X[idx], norm='l2', axis=1)
        sim = Xi @ Xi.T
        dist_mean = (1.0 - sim).mean(axis=0)
        cand_local = np.where(dist_mean == dist_mean.min())[0]
        cand_idx = idx[cand_local]
        med = min(cand_idx, key=lambda j: qrank(flags[j], 1))
        reps.append(med)

    reps = sorted(set(reps))
    return ClusterResult(labels=labels, reps_idx=reps, scores=chosen)

def enforce_rep_quota(labels: np.ndarray, reps_idx: List[int], X: np.ndarray,
                      flags: List[str], max_reps: int = 9) -> Tuple[np.ndarray, List[int]]:
    if len(reps_idx) <= max_reps:
        return labels, sorted(set(reps_idx))

    qrank = {'s': 0, '': 1, 'x': 2}.get
    reps = list(reps_idx)
    labels = labels.copy()

    while len(reps) > max_reps:
        Xi = normalize(np.vstack([X[i] for i in reps]), norm='l2', axis=1)
        sim = Xi @ Xi.T
        np.fill_diagonal(sim, -np.inf)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        if i > j:
            i, j = j, i

        lab_i, lab_j = labels[reps[i]], labels[reps[j]]
        labels[labels == lab_j] = lab_i

        members = np.where(labels == lab_i)[0]
        if len(members) > 1:
            Xm = normalize(X[members], norm='l2', axis=1)
            sim_m = Xm @ Xm.T
            dist_mean = (1.0 - sim_m).mean(axis=0)
            best_local = np.where(dist_mean == dist_mean.min())[0]
            cand = members[best_local]
            new_med = min(cand, key=lambda jdx: qrank(flags[jdx], 1))
        else:
            new_med = members[0]

        reps[i] = new_med
        reps.pop(j)

    return labels, sorted(set(reps))

# ---------------------------------------------------------------------------

def load_entries_from_folders(folders: List[Path]) -> List[Entry]:
    entries_raw, skipped = rc.load_reference_folders(tuple(folders))
    if skipped:
        logging.info("ℹ️  %d Dateien beim Scan übersprungen (siehe raman_core-Logik).", len(skipped))
    entries: List[Entry] = []
    for e in entries_raw:
        entries.append(Entry(
            name=e.get('name', ''),
            path=Path(e.get('path', '')),
            orig_filename=e.get('orig_filename', Path(e.get('path', '')).name if e.get('path') else ''),
            flag=e.get('flag', ''),
        ))
    return entries

def compute_fingerprints(entries: List[Entry], workers: int = 0) -> List[FPInfo]:
    if workers and workers > 1:
        with mp.Pool(workers) as pool:
            fps = list(pool.imap(_make_fingerprint, entries, chunksize=32))
    else:
        fps = [_make_fingerprint(e) for e in entries]
    return fps

# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cluster-Auswahl repräsentativer Raman-Spektren je Mineral (k ≤ 9, keine Peak-Filter).")
    ap.add_argument("folders", nargs="+", help="Eingabeverzeichnisse (bereits dedupliziert/vorgefiltert)")
    ap.add_argument("-o", "--output", required=True, help="Zielordner für Repräsentanten & Reports")
    ap.add_argument("--method", choices=["auto", "kmedoids", "gmm"], default="auto",
                    help="Clustermethode (default: auto → K-Medoids, sonst GMM)")
    ap.add_argument("--k-min", type=int, default=3, help="Minimale Clusterzahl (nur wenn n>4; default 3)")
    ap.add_argument("--k-max", type=int, default=9, help="Maximale Clusterzahl/Quote (default 9)")
    ap.add_argument("--dim", type=int, default=30, help="Ziel-Dimension für PCA (default 30)")
    default_workers = max(1, min(8, (mp.cpu_count() or 1) - 1))
    ap.add_argument("--workers", type=int, default=default_workers,
                    help=f"Parallelisierung beim Fingerprinting (default min(8, CPU-1) → {default_workers})")
    ap.add_argument("--max-reps-per-mineral", type=int, default=9, help="Maximale Repräsentanten pro Mineral (default 9)")
    ap.add_argument("--seed", type=int, default=42, help="Zufallssamen (Reproduzierbarkeit)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Mehr Logausgabe")

    # VS-Code-Defaults ohne Parameter
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = [_default_input_folder(), "-o", "ClusterDB"]

    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)

    # 1) Laden
    logging.info("🔍  Scanne Eingabeordner …")
    entries = load_entries_from_folders([Path(f) for f in args.folders])
    logging.info("📑  %d Spektren geladen.", len(entries))

    # 2) Fingerprints
    logging.info("🧬  Erzeuge Fingerprints … (workers=%d)", args.workers)
    fp_infos = compute_fingerprints(entries, workers=args.workers)

    # 3) Gruppenbildung nach Mineral
    groups: Dict[str, List[int]] = {}
    for i, e in enumerate(entries):
        groups.setdefault(e.name, []).append(i)
    logging.info("🔬  %d Mineral-Gruppen erkannt.", len(groups))

    # 4) CSV: alle Spektren
    all_csv = target / "report_all_spectra.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "flag", "orig_filename", "path"])
        for e in entries:
            w.writerow([e.name, e.flag, e.orig_filename, str(e.path)])
    logging.info("📝  %s geschrieben.", all_csv.resolve())

    # 5) Verarbeitung je Gruppe
    per_group_summary_rows = []
    per_group_detail_dir = target / "per_group_details"
    per_group_detail_dir.mkdir(parents=True, exist_ok=True)

    chosen_indices_global: List[int] = []
    kept_outside_indices_global: List[int] = []

    for name, idxs in tqdm(groups.items(), desc="Gruppen", unit="group"):
        # Split in outside / valid
        valid_local: List[int] = []
        outside_local: List[int] = []
        failed_local: List[int] = []

        for i in idxs:
            info = fp_infos[i]
            if info.kept_outside:
                outside_local.append(i)
            elif info.fp is None:
                failed_local.append(i)  # Parsing/FP fehlgeschlagen → als Singleton behalten
            else:
                valid_local.append(i)

        # Datenmatrix X und Flags für gültige
        X = None
        flags_valid: List[str] = []
        if valid_local:
            X = np.vstack([fp_infos[i].fp for i in valid_local]).astype(float)
            X = normalize(X, norm='l2', axis=1)  # sicherheitshalber
            flags_valid = [entries[i].flag for i in valid_local]

        # Clustering (nur wenn n_valid > 4)
        chosen_local: List[int] = []
        method_used = "n/a"
        chosen_k = len(valid_local) if len(valid_local) <= 4 else 0
        score_sil = ""
        score_bic = ""

        if X is not None and len(valid_local) > 4:
            res = cluster_group(
                X, flags_valid,
                method=args.method,
                k_min=args.k_min,
                k_max=args.k_max,
                seed=args.seed,
                dim=args.dim,
            )
            method_used = str(res.scores.get("method", "auto"))
            chosen_k = int(res.scores.get("k", len(set(res.labels))))
            if "silhouette" in res.scores:
                score_sil = f"{res.scores['silhouette']:.4f}"
            if "bic" in res.scores:
                score_bic = f"{res.scores['bic']:.2f}"

            # Repräsentanten-Quote durchsetzen (≤ max_reps_per_mineral)
            labels_adj, reps_adj = enforce_rep_quota(
                labels=res.labels,
                reps_idx=res.reps_idx,
                X=X,
                flags=flags_valid,
                max_reps=args.max_reps_per_mineral
            )
            chosen_local = [valid_local[i] for i in reps_adj]

        else:
            # Kein Clustering (n_valid <= 4): alle behalten
            chosen_local = valid_local[:]
            method_used = "trivial"
            chosen_k = len(chosen_local)

        # „Outside REF_X“ und Parsing-Fehler immer behalten
        chosen_local.extend(failed_local)
        kept_outside_indices_global.extend(outside_local)

        # Für globale Kopierliste
        chosen_indices_global.extend(chosen_local)

        # Detail-CSV je Mineral
        safe_name = re.sub(r'[^A-Za-z0-9_-]+','_',name.strip()) or 'unknown'
        detail_csv = per_group_detail_dir / f"{safe_name}.csv"
        with detail_csv.open("w", newline="", encoding="utf-8") as df:
            w = csv.writer(df)
            w.writerow(["name", "flag", "orig_filename", "path",
                        "status", "cluster_label", "is_medoid", "dist_to_medoid"])
            if X is not None and len(valid_local) > 0:
                # Labels/Reps für Details
                if len(valid_local) > 4 and method_used != "trivial":
                    labels_for_detail = labels_adj
                    reps_for_detail = reps_adj
                else:
                    labels_for_detail = np.arange(len(valid_local))
                    reps_for_detail = list(range(len(valid_local)))

                Xi = normalize(X, norm='l2', axis=1)
                sim_all = Xi @ Xi.T
                for local_idx, i in enumerate(valid_local):
                    lab = labels_for_detail[local_idx]
                    cand_meds = [r for r in reps_for_detail if labels_for_detail[r] == lab]
                    if cand_meds:
                        dists = [(1.0 - sim_all[local_idx, r], r) for r in cand_meds]
                        med_local = min(dists, key=lambda t: t[0])[1]
                        dist = 1.0 - sim_all[local_idx, med_local]
                        is_med = (valid_local[med_local] == i)
                    else:
                        dist, is_med = "", False

                    w.writerow([entries[i].name, entries[i].flag, entries[i].orig_filename, str(entries[i].path),
                                "valid", lab if lab is not None else "", "YES" if is_med else "NO",
                                f"{dist:.6f}" if dist != "" else ""])
            for i in outside_local:
                w.writerow([entries[i].name, entries[i].flag, entries[i].orig_filename, str(entries[i].path),
                            "outside_REF", "", "YES", ""])
            for i in failed_local:
                w.writerow([entries[i].name, entries[i].flag, entries[i].orig_filename, str(entries[i].path),
                            "parse_failed_fp_singleton", "", "YES", ""])

        per_group_summary_rows.append([
            name,
            len(idxs),
            len(valid_local),
            len(outside_local),
            len(failed_local),
            len(set(chosen_local)) + len(outside_local),
            chosen_k,
            method_used,
            score_sil,
            score_bic,
        ])

    # 6) Summary-CSV
    summary_csv = target / "report_groups_clustered.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "name",
            "count_total",
            "count_valid_for_clustering",
            "count_outside_ref",
            "count_parse_failed",
            "count_selected_final",
            "chosen_k",
            "method",
            "silhouette",
            "bic",
        ])
        for row in sorted(per_group_summary_rows, key=lambda r: r[0].lower()):
            w.writerow(row)
    logging.info("📝  %s geschrieben.", summary_csv.resolve())

    # 7) Kopieren der finalen Auswahl
    chosen_set = sorted(set(chosen_indices_global))
    outside_set = sorted(set(kept_outside_indices_global))
    total_copy = len(chosen_set) + len(outside_set)
    logging.info("📂  Kopiere %d Dateien in %s …", total_copy, target.resolve())

    copied = 0
    for i in tqdm(chosen_set + outside_set, desc="Kopieren", unit="file"):
        src = entries[i].path
        dst = target / entries[i].orig_filename
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as exc:
            logging.warning("⏭️  Konnte %s nicht kopieren (%s).", src, exc)

    logging.info("✅  %d Dateien kopiert. Fertig.", copied)


if __name__ == "__main__":
    # VS Code: ohne Args → databases/OWN (Fallback: CleanDB) → ClusterDB
    main()

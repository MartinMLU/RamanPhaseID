#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Matplotlib nur für File-Output benutzen
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent

def _default_trim_folder() -> Path:
    candidates = (
        BASE_DIR / "databases" / "OWN",
        BASE_DIR / "CleanDB",
    )
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def _default_trim_output_folder() -> Path:
    return BASE_DIR / "databases_trimmed"


_NUMERIC_LINE_PATTERN = re.compile(
    r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"(?:[\t,\s;]+[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)+\s*$"
)
_FLOAT_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _parse_numeric_xy_line(line: str) -> tuple[float, float] | None:
    if not _NUMERIC_LINE_PATTERN.match(line):
        return None
    vals = _FLOAT_PATTERN.findall(line)
    if len(vals) < 2:
        return None
    try:
        return float(vals[0]), float(vals[-1])
    except Exception:
        return None

# -----------------------------------------------------------------------------
# I/O Helpers

def parse_rod_txt(path: Path) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    shifts, intensities, lines = [], [], []
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            lines.append(line)
            xy = _parse_numeric_xy_line(line)
            if xy is not None:
                shifts.append(xy[0])
                intensities.append(xy[1])
    if not shifts:
        raise ValueError(f"no spectral data in {path}")
    return np.asarray(shifts), np.asarray(intensities), lines


def write_trimmed_norm(
    path_out: Path,
    lines: list[str],
    keep_mask: np.ndarray,
    x: np.ndarray,
    y_norm_full: np.ndarray,
) -> None:
    """
    Schreibt die Datei neu:
      - Kommentar-/Headerzeilen bleiben unverändert.
      - Für numerische Zeilen werden NUR die beibehaltenen Punkte geschrieben,
        mit x und NORMALISIERTEM y (Tab-getrennt).
      - y_norm_full enthält bereits die finale (nach dem Trimmen neu berechnete)
        Normalisierung; Werte an nicht-behaltenen Indizes werden ignoriert.
    """
    data_idx = 0
    with path_out.open("w", newline="") as fh:
        for line in lines:
            if _parse_numeric_xy_line(line) is not None:
                if keep_mask[data_idx]:
                    fh.write(f"{x[data_idx]:.10g}\t{y_norm_full[data_idx]:.10g}\n")
                data_idx += 1
            else:
                fh.write(line)

# -----------------------------------------------------------------------------
# Parameter

@dataclass
class Params:
    # Normierung (für Erkennung & finalen Output)
    norm_min: float = 0.0
    norm_max: float = 1000.0

    # --- Sprung-/Gate-Logik (deine Regel)
    jump_search_max_cm: float = 250.0      # Sprungsuche bis 250 cm^-1 ab x[0]
    post_return_check_cm: float = 300.0    # Rückfallprüfung in +300 cm^-1
    post_return_tol_rel: float = 0.20      # ±20% um pre_min (z. B. 0.15 für ±15%)
    abs_return_tol_counts: float = 40.0    # ±25 Counts um pre_min (absolut)
    first_jump_min_counts: float = 50.0   # erster Sprung ab diesem Delta zählt exklusiv
    extra_head_cm: float = 2.0  # Head-Trim nach Zero-Trim um +5 cm⁻¹ erweitern

    # --- Fenster (werden in Punkte umgerechnet)
    smooth_win_cm: float = 3.0             # y-Glättung (rolling Median) für die Sprungsuche
    slope_win_cm: float = 4.0              # Steigungs-Glättung (rolling Median) für Rampenende

    # --- Rampenende (robust, aber schlank)
    ramp_min_span_cm: float = 6.0          # Mindestabstand (x_end - x_jump) für Ende
    k_end_sigma: float = 1.0               # Ende: |slope_med| <= k * sigma_slope
    end_drop_frac_of_peak: float = 0.5     # ODER: slope <= f * peak_slope
    end_pos_frac_max: float = 0.60         # UND: Positiv-Quote im Endfenster nicht zu hoch

    # --- Rauschfenster nur für slope (Rampenende); Sprungsuche benötigt kein σ
    early_noise_cm: float = 40.0
    max_noise_pts: int = 80

    # --- Null-Trims (auf ROH-Counts)
    enable_head_zero_trim: bool = True
    enable_tail_zero_trim: bool = True
    zero_threshold: float = 0.0           # |y| <= thr gilt als Null
    before_window_pts: int = 20
    min_points_keep_if_all_zero: int = 1
    jump_min_tail: float = 30.0

    # --- IO
    folder: Path = _default_trim_folder()
    output_folder: Path = _default_trim_output_folder()
    patterns: tuple[str, ...] = ("*.rod", "*.txt")
    plot_dir: Path | None = Path("EdgeTrimPlot")
    overwrite_files: bool = False

    # --- Early-Fall-Heuristik (früheres Ende, bevor eine Ramanbande hochläuft)
    early_fall_enable: bool = True
    early_fall_frac_span: float = 0.03     # Anteil der 10–90%-Spannweite als Mindestabfall
    early_fall_k_sigma_y: float = 2.0      # k * sigma_y (MAD-basiert) als Mindestabfall
    early_fall_abs_counts: float = 20.0    # absoluter Mindestabfall in Norm-Counts
    early_fall_pos_frac_max: float = 0.55  # im Bestätigungsfenster: Anteil positiver Steigungen max. so hoch
    early_fall_confirm_cm: float = 4.0     # Breite des Bestätigungsfensters in cm^-1

# -----------------------------------------------------------------------------
# Utilities

def normalize_minmax(y: np.ndarray, p: Params) -> np.ndarray:
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if ymax - ymin == 0:
        return np.full_like(y, p.norm_min, dtype=float)
    y_norm = (y - ymin) / (ymax - ymin)
    return y_norm * (p.norm_max - p.norm_min) + p.norm_min


def estimate_noise(y: np.ndarray, n_pts: int) -> float:
    seg = y[: max(5, min(n_pts, y.size))]
    if seg.size == 0 or np.allclose(seg, seg[0]):
        return 0.0
    mad = np.median(np.abs(seg - np.median(seg)))
    return 1.4826 * mad


def robust_percentile_gap(y: np.ndarray, low: float = 10.0, high: float = 90.0) -> float:
    return float(np.percentile(y, high) - np.percentile(y, low))


def cm_to_pts(x: np.ndarray, span_cm: float, min_pts: int = 3) -> int:
    if x.size < 2:
        return min_pts
    median_dx = float(np.median(np.diff(x)))
    pts = int(round(max(min_pts, span_cm / max(median_dx, 1e-12))))
    return max(min_pts, pts)


def _rolling_median(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or a.size < w:
        return a.copy()
    med = np.array([np.median(a[i:i+w]) for i in range(0, a.size - w + 1)], dtype=float)
    pad_left = w // 2
    pad_right = a.size - med.size - pad_left
    return np.pad(med, (pad_left, pad_right), mode="edge")

# -----------------------------------------------------------------------------
# Null-Trim (Anfang und Ende) auf ROH-y

def zero_block_prefix_len(y: np.ndarray, thr: float) -> int:
    if y.size == 0:
        return 0
    mask = np.abs(y) <= thr
    if not mask[0]:
        return 0
    return int(np.argmax(~mask)) if (~mask).any() else y.size


def head_zero_trim_start_idx(y_raw: np.ndarray, p: Params) -> int:
    if not p.enable_head_zero_trim or y_raw.size == 0:
        return 0
    h_len = zero_block_prefix_len(y_raw, p.zero_threshold)
    return min(h_len, y_raw.size)


def zero_block_suffix_len(y: np.ndarray, thr: float) -> int:
    if y.size == 0:
        return 0
    mask = np.abs(y) <= thr
    if not mask[-1]:
        return 0
    rev_nonzero = (~mask[::-1])
    return int(np.argmax(rev_nonzero)) if rev_nonzero.any() else y.size


def tail_zero_trim_end_idx(y_raw: np.ndarray, start_idx: int, p: Params) -> int:
    """Finde optionales End-Index (exklusiv) durch Null-Tail-Trimming."""
    n = y_raw.size
    if not p.enable_tail_zero_trim or n == 0 or start_idx >= n:
        return n

    s_len = zero_block_suffix_len(y_raw, p.zero_threshold)
    if s_len == 0:
        return n

    tail_start = n - s_len
    if tail_start <= start_idx:
        # Fast alles wäre Null; Fallback
        if np.all(np.abs(y_raw) <= p.zero_threshold):
            return min(start_idx + max(1, p.min_points_keep_if_all_zero), n)
        return n

    tail_med = float(np.median(y_raw[tail_start:]))  # ~0
    prev_start = max(start_idx, tail_start - p.before_window_pts)
    if prev_start >= tail_start:
        return n
    prev_med = float(np.median(y_raw[prev_start:tail_start]))
    delta = prev_med - tail_med
    if delta >= p.jump_min_tail:
        return tail_start
    return n

# -----------------------------------------------------------------------------
# Kern: Sprung erkennen (≤250 cm^-1), Gate (+300 cm^-1, ±Rel ODER ±Abs um pre_min), Rampenende

def detect_jump_and_ramp_end(x: np.ndarray, y_norm: np.ndarray, p: Params) -> tuple[int | None, dict]:
    """
    1) Sprung in [x0, x0+250] cm^-1: k = argmax( median(y[k:k+wpost]) - median(y[k-wpre:k]) ).
    2) pre_min = min(y_norm[:k])   (ALLE Punkte vor dem Sprung; auf ROH-normalisierten Werten).
    3) Gate in [x_k, x_k+300] cm^-1: wenn irgendein Punkt y entweder im RELATIVEN ±Tol%-Band
       ODER im ABSOLUTEN ±Counts-Band um pre_min liegt -> NICHT schneiden; sonst schneiden.
    4) Schnittpunkt = MIN( Early-Fall-Ende, klassisches Rampenende ), mit Mindestspanne.
    """
    n = y_norm.size
    if n < 7:
        return None, {"reason": "too_short"}

    # --- Sprungsuchbereich begrenzen
    x_jump_lim = x[0] + p.jump_search_max_cm
    end_jump = int(np.searchsorted(x, x_jump_lim, side="right"))
    if end_jump < 7:
        return None, {"reason": "jump_window_small"}

    # --- Fenster in Punkte
    w_s   = cm_to_pts(x, p.smooth_win_cm, 3)
    wpre  = w_s
    wpost = w_s

    # --- Glättung (nur für robuste Sprungsuche)
    y_s   = y_norm[:end_jump]
    x_s   = x[:end_jump]
    y_sm  = _rolling_median(y_s, w_s)

    # --- 1) Sprung: frühester Sprung ≥ first_jump_min_counts; sonst größter Sprung
    first_thr = max(0.0, p.first_jump_min_counts)

    best_k: int | None = None
    best_delta: float = -np.inf
    best_k_fallback: int | None = None

    for k in range(max(wpre, 1), y_sm.size - wpost):
        pre_med  = float(np.median(y_sm[k - wpre:k]))
        post_med = float(np.median(y_sm[k:k + wpost]))
        delta = post_med - pre_med

        # Merke größten Sprung als Fallback
        if delta > best_delta:
            best_delta = delta
            best_k_fallback = k

        # Nimm den ERSTEN Sprung, der den 50-Counts-Schwellenwert erreicht/überschreitet
        if delta >= first_thr:
            best_k = k
            break

    # Falls kein Sprung ≥ 50 gefunden wurde, nimm den größten Sprung im Fenster
    if best_k is None:
        best_k = best_k_fallback

    if best_k is None or best_k <= 0:
        return None, {"reason": "no_jump"}


    # --- 2) Minimum vor dem Sprung (ALLE Punkte vor k, auf ROH-normalisierten Werten)
    pre_min = float(np.min(y_norm[:best_k]))

    # --- 3) Gate: Rückfallprüfung in +300 cm^-1 (einzelner Punkt reicht)
    ret_lim = x[best_k] + p.post_return_check_cm
    ret_end = int(np.searchsorted(x, ret_lim, side="right"))
    ret_end = min(ret_end, y_norm.size)
    if ret_end <= best_k + 1:
        return None, {"reason": "no_post_window"}

    post_seg_raw = y_norm[best_k:ret_end]

    # Relatives ±Tol%-Band um pre_min
    tol_rel = max(0.0, p.post_return_tol_rel)
    rel_lower = pre_min * (1.0 - tol_rel)
    rel_upper = pre_min * (1.0 + tol_rel)

    # Absolutes ±Counts-Band um pre_min
    abs_tol = max(0.0, p.abs_return_tol_counts)
    abs_lower = pre_min - abs_tol
    abs_upper = pre_min + abs_tol

    in_rel_band = (post_seg_raw >= rel_lower) & (post_seg_raw <= rel_upper)
    in_abs_band = (post_seg_raw >= abs_lower) & (post_seg_raw <= abs_upper)

    if np.any(in_rel_band | in_abs_band):
        # Rückkehr (auch nur ein Punkt) in eines der Bänder -> kein Schnitt
        return None, {
            "reason": "returns_to_pre_min_within_300cm",
            "pre_min": pre_min,
            "post_window_min": float(np.min(post_seg_raw)),
            "rel_band": (rel_lower, rel_upper),
            "abs_band": (abs_lower, abs_upper),
            "jump_x": float(x[best_k]),
        }

    # --- 4) Rampenende bestimmen (nur für Schnittposition)
    w_m   = cm_to_pts(x, p.slope_win_cm, 3)
    dy    = np.diff(y_sm)
    dx    = np.diff(x_s)
    dx    = np.maximum(dx, 1e-12)
    slope = dy / dx
    slope_med = _rolling_median(slope, w_m)

    # Rauschmaß der slope aus frühem Bereich (bis early_noise_cm)
    noise_end_idx = max(5, np.searchsorted(x_s, x[0] + p.early_noise_cm, side="right"))
    noise_end_idx = min(noise_end_idx, min(p.max_noise_pts, slope_med.size))
    sigma_slope = estimate_noise(slope_med[:noise_end_idx], noise_end_idx)
    k_end = p.k_end_sigma * max(sigma_slope, 1e-12)

    # Lokaler slope-Peak nach dem Sprung
    start_idx_slope = max(0, best_k - 1)
    if start_idx_slope >= slope_med.size - 1:
        return None, {"reason": "no_slope_window_after_jump"}

    peak_after = int(np.argmax(slope_med[start_idx_slope:]) + start_idx_slope)
    peak_val   = float(slope_med[peak_after])
    drop_level = max(k_end, p.end_drop_frac_of_peak * peak_val)

    # Kandidaten: (A) Ruhebedingung + niedrige Positiv-Quote, (B) relativer Abfall vom Peak
    pos_mask = (slope_med > 0).astype(float)
    pos_frac = _rolling_median(pos_mask, max(3, w_m))

    cand_end_a = np.where((np.abs(slope_med) <= k_end) & (pos_frac <= p.end_pos_frac_max))[0]
    cand_end_a = cand_end_a[cand_end_a > start_idx_slope]

    cand_end_b = np.where(slope_med <= drop_level)[0]
    cand_end_b = cand_end_b[cand_end_b > peak_after]

    def first_end_with_span(cands: np.ndarray) -> int | None:
        for c in cands:
            i_y = int(c) + 1  # slope-Index -> y-Index
            if x_s[i_y] - x_s[best_k] >= p.ramp_min_span_cm:
                return int(c)
        return None

    e_a = first_end_with_span(cand_end_a)
    e_b = first_end_with_span(cand_end_b)

    # ===================== EARLY-FALL: FRÜHERES ENDE (NEU) =====================
    early_end_idx_y = None
    if p.early_fall_enable:
        # Schwellwert für „spürbaren Abfall“ in y
        sigma_y = estimate_noise(y_sm[:min(noise_end_idx+1, y_sm.size)],
                                 min(noise_end_idx+1, y_sm.size))
        span_y  = robust_percentile_gap(y_sm, 10.0, 90.0)
        drop_thr = max(
            p.early_fall_k_sigma_y * max(sigma_y, 1e-12),
            p.early_fall_frac_span * max(span_y, 1e-12),
            p.early_fall_abs_counts
        )

        # Laufendes Maximum ab Sprung
        y_post = y_sm[best_k:]
        runmax = np.maximum.accumulate(y_post)

        # Kandidaten: erster Index i > k mit ausreichendem Abfall und Mindestspanne
        idxs = np.arange(best_k+1, y_sm.size)
        has_drop = y_sm[best_k+1:] <= (runmax[:-1] - drop_thr)
        has_span = (x_s[idxs] - x_s[best_k]) >= p.ramp_min_span_cm

        # Bestätigung: lokale Steigung nicht mehr überwiegend positiv
        conf_w = cm_to_pts(x, p.early_fall_confirm_cm, 3)
        ok = []
        for i in idxs[has_drop & has_span]:
            j1 = max(best_k, i - conf_w)
            j2 = min(i - 1, slope_med.size - 1)
            if j2 <= j1:
                continue
            pos_local = (slope_med[j1:j2] > 0).mean()
            if pos_local <= p.early_fall_pos_frac_max:
                ok.append(i)
                break  # frühesten nehmen
        if ok:
            early_end_idx_y = int(ok[0])
    # ===========================================================================
    # Klassisches Rampenende bestimmen
    end_idx_slope = None
    if e_a is not None and e_b is not None:
        end_idx_slope = min(e_a, e_b)
    elif e_a is not None:
        end_idx_slope = e_a
    elif e_b is not None:
        end_idx_slope = e_b

    if end_idx_slope is None and early_end_idx_y is None:
        return None, {"reason": "no_ramp_end"}

    # y-Index aus slope-Index
    end_idx_y_std = (int(end_idx_slope) + 1) if end_idx_slope is not None else None

    # Finale Wahl: das frühere Ende nehmen (aber mind. k+1)
    candidates_y = [idx for idx in [early_end_idx_y, end_idx_y_std] if idx is not None]
    end_idx_y = min(candidates_y) if candidates_y else None
    if end_idx_y is None:
        return None, {"reason": "no_ramp_end_after_early"}

    end_idx_y = min(max(end_idx_y, best_k + 1), y_sm.size - 1)

    info = {
        "jump_k": int(best_k),
        "jump_x": float(x[best_k]),
        "pre_min": pre_min,
        "rel_band": (rel_lower, rel_upper),
        "abs_band": (abs_lower, abs_upper),
        "end_x": float(x[end_idx_y]),
    }
    if early_end_idx_y is not None:
        info["early_fall_end_x"] = float(x[early_end_idx_y])
    return end_idx_y, info

# -----------------------------------------------------------------------------
# Plot

def make_plot(
    x: np.ndarray,
    y_orig: np.ndarray,
    y_norm_final: np.ndarray,
    keep_mask: np.ndarray,
    cut_shift_start: float | None,
    cut_shift_end: float | None,
    out_png: Path | None,
    title: str = "",
) -> None:
    if out_png is None:
        return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y_orig, lw=0.8, label="orig")
        ax.plot(x, y_norm_final, lw=0.8, label="norm (final)")
        ax.plot(x[keep_mask], y_norm_final[keep_mask], lw=0.8, label="trimmed (saved)")
        if cut_shift_start is not None:
            ax.axvline(cut_shift_start, ls="--", lw=0.8, label="start cut")
        if cut_shift_end is not None:
            ax.axvline(cut_shift_end, ls=":", lw=0.8, label="end cut")
        ax.set_xlabel("shift / cm$^{-1}$")
        ax.set_ylabel("counts (norm)")
        if title:
            ax.set_title(title)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as exc:
        tqdm.write(f"Plot failed ({out_png}): {exc}")

# -----------------------------------------------------------------------------
# File iteration

def iter_files(folder: Path, patterns: Iterable[str]) -> Iterable[Path]:
    for pat in patterns:
        yield from folder.rglob(pat)

# -----------------------------------------------------------------------------
# Main

def main() -> None:
    p = Params()

    if p.plot_dir:
        p.plot_dir.mkdir(parents=True, exist_ok=True)
    if not p.overwrite_files:
        p.output_folder.mkdir(parents=True, exist_ok=True)

    files = list(iter_files(p.folder, p.patterns))
    if not files:
        print("No files found.")
        return

    for f in tqdm(files, desc=f"Processing spectra ({p.folder.name})"):
        if p.overwrite_files:
            path_out = f
        else:
            rel = f.relative_to(p.folder)
            path_out = p.output_folder / rel
            path_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            x, y_raw, lines = parse_rod_txt(f)
        except Exception as exc:
            tqdm.write(f"[!] Skip {f}: {exc}")
            continue

        # 0) Führende Nullen (ROH) -> Start-Index (VOR der Kanten-/Sprung-Erkennung!)
        start_idx_zero = head_zero_trim_start_idx(y_raw, p)

        # Head-Trim um +extra_head_cm erweitern (nur wenn wirklich ein Zero-Trim stattfand)
        start_idx_head = start_idx_zero
        if start_idx_zero > 0 and p.extra_head_cm > 0.0:
            base_shift = x[start_idx_zero] if start_idx_zero < x.size else x[-1]
            target_shift = base_shift + p.extra_head_cm
            start_idx_head = int(np.searchsorted(x, target_shift, side="left"))
            start_idx_head = min(start_idx_head, x.size - 1)

        # 1) Vor-Normierung NUR für Erkennung auf dem bereits "head-getrimmten" Ausschnitt
        if start_idx_head >= x.size - 1:
            # Nichts Sinnvolles mehr übrig
            keep_mask = np.zeros_like(y_raw, dtype=bool)
            keep_mask[start_idx_head:] = True
            y_keep = y_raw[keep_mask]
            y_keep_norm = normalize_minmax(y_keep, p)
            y_norm_final_full = np.zeros_like(y_raw, dtype=float)
            y_norm_final_full[keep_mask] = y_keep_norm
            write_trimmed_norm(path_out, lines, keep_mask, x, y_norm_final_full)
            continue

        x_det = x[start_idx_head:]
        y_det = y_raw[start_idx_head:]
        y_det_norm = normalize_minmax(y_det, p)

        # 2) Sprung + Gate + Rampenende auf dem head-getrimmten Ausschnitt
        cut_idx_rel, _info = detect_jump_and_ramp_end(x_det, y_det_norm, p)

        # 3) Effektiver Start: max(Head-Trim, Edge-Ende) in Originalindizes
        cut_idx_edge = (start_idx_head + cut_idx_rel) if cut_idx_rel is not None else None
        start_idx_eff = max(start_idx_head, (cut_idx_edge or 0))

        # 4) Null-Suffix-Trim (auf ROH-y)
        end_idx = tail_zero_trim_end_idx(y_raw, start_idx_eff, p)
        end_idx = max(end_idx, start_idx_eff + 1)  # mind. 1 Punkt behalten

        keep_mask = np.zeros_like(y_raw, dtype=bool)
        keep_mask[start_idx_eff:end_idx] = True

        cut_shift_start = float(x[start_idx_eff]) if x.size else None
        cut_shift_end = float(x[end_idx - 1]) if end_idx < y_raw.size else None

        # 5) FINAL: Nach dem Zuschneiden erneut normalisieren -> das wird gespeichert
        y_keep = y_raw[keep_mask]
        y_keep_norm = normalize_minmax(y_keep, p)
        y_norm_final_full = np.zeros_like(y_raw, dtype=float)
        y_norm_final_full[keep_mask] = y_keep_norm

        # 6) Plot (wenn Sprung erkannt oder Head-Trim wirksam)
        if p.plot_dir and (cut_idx_rel is not None or start_idx_head > 0):
            out_png = p.plot_dir / f"{f.stem}.png"
            make_plot(
                x, y_raw, y_norm_final_full, keep_mask,
                cut_shift_start, cut_shift_end,
                out_png, title=f.name
            )

        # 7) Schreiben (IN-PLACE), mit FINAL normalisierten y-Werten
        write_trimmed_norm(path_out, lines, keep_mask, x, y_norm_final_full)

    print("Done.")


if __name__ == "__main__":
    main()

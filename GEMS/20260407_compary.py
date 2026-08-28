from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOP_KS = (1, 3, 6, 10)
BLOCK_SIZE = 10

# Alias mapping so trade/sample names can map to phase names.
TRUTH_ALIASES = {
    "aquamarine": {"beryl"},
    "beryltransparent": {"beryl"},
    "calcite": {"calcite"},
    "cordierite": {"cordierite"},
    "corundumtransparent": {"corundum"},
    "cubiczirconia": {"zirconia"},
    "diamond": {"diamond"},
    "emerald": {"beryl"},
    "moissanite": {"moissanite"},
    "olivineperidote": {"forsterite", "olivine", "peridote"},
    "pinkquartzglas": {"quartz", "quartzglass"},
    "quartz": {"quartz"},
    "saphire": {"corundum", "sapphire"},
    "synspinel": {"spinel"},
    "topaz": {"topaz"},
    "zircon": {"zircon"},
}


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_rank_cell(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def canonical_app_name(header: str, next_header: str) -> str:
    raw = " ".join(part for part in (header.strip(), next_header.strip()) if part)
    key = norm(raw)

    if "ramanmatch" in key:
        return "Raman Match"
    if "ramanphaseid" in key:
        return "RamanPhaseID"
    if "ramanlab" in key and "mineralvibration" in key:
        return "RamanLab MineralVibration"
    if "ramanlab" in key and "multiwindow" in key:
        return "RamanLab MultiWindow"
    if "ramanlab" in key:
        return "RamanLab"
    return header.strip() or "Unknown App"


def detect_app_columns(df: pd.DataFrame) -> dict[str, list[int]]:
    if df.shape[0] < 2:
        raise ValueError("Excel file must contain at least 2 header rows.")

    header_row = df.iloc[0]
    rank_row = df.iloc[1]
    n_cols = df.shape[1]
    blocks: list[tuple[int, str, list[int]]] = []

    for col in range(1, n_cols - BLOCK_SIZE + 1):
        if parse_rank_cell(rank_row.iloc[col]) != 1:
            continue
        is_block = all(parse_rank_cell(rank_row.iloc[col + i]) == i + 1 for i in range(BLOCK_SIZE))
        if not is_block:
            continue

        header = "" if pd.isna(header_row.iloc[col]) else str(header_row.iloc[col])
        next_header = ""
        if col + 1 < n_cols and not pd.isna(header_row.iloc[col + 1]):
            next_header = str(header_row.iloc[col + 1])

        app_name = canonical_app_name(header, next_header)
        blocks.append((col, app_name, list(range(col, col + BLOCK_SIZE))))

    if not blocks:
        raise ValueError("No app blocks with rank sequence 1..10 were detected.")

    blocks.sort(key=lambda x: x[0])

    # Ensure unique labels if headers are duplicated.
    counts: dict[str, int] = {}
    app_columns: dict[str, list[int]] = {}
    for _, name, cols in blocks:
        counts[name] = counts.get(name, 0) + 1
        final_name = name if counts[name] == 1 else f"{name} #{counts[name]}"
        app_columns[final_name] = cols

    return app_columns


def base_name_from_spectrum(spectrum_name: str) -> str:
    match = re.match(r"^(.*?)(?:_\d|\s\d)", spectrum_name)
    return match.group(1) if match else spectrum_name


def truth_keys_from_name(spectrum_name: str) -> set[str]:
    base_name = base_name_from_spectrum(spectrum_name)
    base_key = norm(base_name)

    if base_key in TRUTH_ALIASES:
        return TRUTH_ALIASES[base_key]

    parts = {norm(p) for p in re.split(r"[^a-zA-Z0-9]+", base_name) if len(p) >= 3}
    parts.discard("")
    if base_key:
        parts.add(base_key)
    return parts


def is_perfect_match(candidate: str, truth_keys: set[str]) -> bool:
    if not candidate:
        return False
    for truth in truth_keys:
        if candidate == truth:
            return True
        if len(truth) >= 4 and truth in candidate:
            return True
        if len(candidate) >= 4 and candidate in truth:
            return True
    return False


def clean_candidate(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "x":
        return ""
    return norm(text)


def compute_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    app_columns = detect_app_columns(df)
    stats = {app: {k: 0 for k in TOP_KS} for app in app_columns}
    total = 0

    app_order = list(app_columns.keys())
    mw_app = next((a for a in app_order if "ramanlab" in norm(a) and "multiwindow" in norm(a)), None)
    mv_app = next((a for a in app_order if "ramanlab" in norm(a) and "mineralvibration" in norm(a)), None)
    combined_app = "RamanLab Combined (MW+MV)" if mw_app and mv_app else None
    if combined_app is not None:
        stats[combined_app] = {k: 0 for k in TOP_KS}

    # Data starts on Excel row 3 => pandas index 2
    for _, row in df.iloc[2:].iterrows():
        raw_name = row.iloc[0]
        if pd.isna(raw_name):
            continue
        spectrum_name = str(raw_name).strip()
        if not spectrum_name:
            continue

        truth_keys = truth_keys_from_name(spectrum_name)
        if not truth_keys:
            continue

        total += 1

        row_hits: dict[str, dict[int, bool]] = {}
        for app, cols in app_columns.items():
            candidates = [clean_candidate(row.iloc[c]) for c in cols if c < len(row)]
            candidates = [c for c in candidates if c]

            row_hits[app] = {}
            for k in TOP_KS:
                top_candidates = candidates[:k]
                hit = any(is_perfect_match(candidate, truth_keys) for candidate in top_candidates)
                row_hits[app][k] = hit
                if hit:
                    stats[app][k] += 1

        if combined_app is not None and mw_app is not None and mv_app is not None:
            for k in TOP_KS:
                if row_hits[mw_app][k] or row_hits[mv_app][k]:
                    stats[combined_app][k] += 1

    if total == 0:
        raise ValueError("No valid spectra rows found. Please check the workbook structure.")

    output_app_order = app_order.copy()
    if combined_app is not None:
        insert_idx = output_app_order.index(mv_app) + 1
        output_app_order.insert(insert_idx, combined_app)

    records: list[dict[str, object]] = []
    for app in output_app_order:
        for k in TOP_KS:
            hits = stats[app][k]
            records.append(
                {
                    "App": app,
                    "TopN": f"Top{k}",
                    "Hits": hits,
                    "Total": total,
                    "Percent": 100.0 * hits / total,
                }
            )

    return pd.DataFrame.from_records(records), app_columns


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "figure.dpi": 300,
        }
    )


def wilson_half_width(hits: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = hits / total
    denom = 1.0 + (z ** 2) / total
    center = (phat + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * np.sqrt((phat * (1.0 - phat) / total) + (z ** 2) / (4.0 * total ** 2))
    return min(center + margin, 1.0) - center


def save_multi_format(fig: plt.Figure, output_base: Path) -> list[Path]:
    saved: list[Path] = []
    for ext in ("png", "pdf", "svg"):
        out = output_base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None)
        saved.append(out)
    return saved


def app_color_marker(app: str, index: int) -> tuple[str, str]:
    app_norm = norm(app)
    if "ramanmatch" in app_norm:
        return "#1f77b4", "o"
    if "ramanlabcombined" in app_norm:
        return "#9467bd", "P"
    if "ramanlab" in app_norm and "multiwindow" in app_norm:
        return "#ff7f0e", "s"
    if "ramanlab" in app_norm and "mineralvibration" in app_norm:
        return "#d62728", "^"
    if "ramanphaseid" in app_norm:
        return "#2ca02c", "D"

    fallback_colors = plt.get_cmap("tab10").colors
    fallback_markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    return fallback_colors[index % len(fallback_colors)], fallback_markers[index % len(fallback_markers)]


def plot_cumulative_recall_curve(summary: pd.DataFrame, output_base: Path) -> list[Path]:
    _style()
    apps = summary["App"].drop_duplicates().tolist()
    x = np.array(TOP_KS, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for idx, app in enumerate(apps):
        color, marker = app_color_marker(app, idx)
        rows = summary[summary["App"] == app].copy()
        rows["k"] = rows["TopN"].str.replace("Top", "", regex=False).astype(int)
        rows = rows.sort_values("k")

        y = rows["Percent"].to_numpy()
        y_err = np.array(
            [100.0 * wilson_half_width(int(h), int(t)) for h, t in zip(rows["Hits"], rows["Total"])],
            dtype=float,
        )

        ax.errorbar(
            x,
            y,
            yerr=y_err,
            fmt=marker + "-",
            label=app,
            color=color,
            linewidth=1.8,
            markersize=6,
            capsize=3,
            zorder=3,
        )

        for xi, yi in zip(x, y):
            ax.text(xi, yi + 1.8, f"{yi:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(TOP_KS)
    ax.set_xticklabels([f"Top{k}" for k in TOP_KS])
    ax.set_ylim(0, 108)
    ax.set_xlim(min(TOP_KS) - 0.5, max(TOP_KS) + 0.5)
    ax.set_ylabel("Perfect Match Rate (%)")
    ax.set_xlabel("Ranking Threshold")
    ax.set_title("Cumulative Phase-Match Performance (Recall@k)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False)

    saved = save_multi_format(fig, output_base)
    plt.close(fig)
    return saved


def plot_heatmap(summary: pd.DataFrame, output_base: Path) -> list[Path]:
    _style()
    apps = summary["App"].drop_duplicates().tolist()
    tops = [f"Top{k}" for k in TOP_KS]

    matrix = np.array(
        [
            [float(summary[(summary["App"] == app) & (summary["TopN"] == top)]["Percent"].iloc[0]) for top in tops]
            for app in apps
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(tops)))
    ax.set_xticklabels(tops)
    ax.set_yticks(np.arange(len(apps)))
    ax.set_yticklabels(apps)
    ax.set_xlabel("Ranking Threshold")
    ax.set_title("Phase-Match Accuracy Heatmap")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt_color = "white" if val >= 60 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Perfect Match Rate (%)")

    saved = save_multi_format(fig, output_base)
    plt.close(fig)
    return saved


def plot_grouped_bar(summary: pd.DataFrame, output_base: Path) -> list[Path]:
    _style()
    apps = summary["App"].drop_duplicates().tolist()
    tops = [f"Top{k}" for k in TOP_KS]
    x = np.arange(len(tops), dtype=float)
    group_width = 0.82
    width = group_width / max(len(apps), 1)

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")

    for idx, app in enumerate(apps):
        color, _ = app_color_marker(app, idx)
        vals = np.array(
            [float(summary[(summary["App"] == app) & (summary["TopN"] == top)]["Percent"].iloc[0]) for top in tops]
        )
        offset = (idx - (len(apps) - 1) / 2.0) * width
        hatch = "///" if "combined" in norm(app) else None
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.90,
            label=app,
            color=color,
            edgecolor="#1f1f1f",
            linewidth=0.6,
            alpha=0.95,
            hatch=hatch,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            y_text = val + 1.1
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tops)
    ax.set_ylim(0, 106)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Perfect Match Rate (%)")
    ax.set_xlabel("Ranking Threshold")
    ax.set_title("Phase-Match Performance by App")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6f6f6f")
    ax.spines["bottom"].set_color("#6f6f6f")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#9a9a9a", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )

    saved = save_multi_format(fig, output_base)
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-quality TopN phase-match plots from Raman app Excel output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Auswertung_20260407b.xlsx"),
        help="Path to input Excel workbook.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("phase_match_publication"),
        help="Common prefix for output figure files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("phase_match_publication_summary.csv"),
        help="Path to save summary table CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_excel(args.input, header=None)
    summary, app_columns = compute_summary(df)
    summary.to_csv(args.summary_csv, index=False)

    curve_base = args.output_base.with_name(args.output_base.name + "_curve")
    heatmap_base = args.output_base.with_name(args.output_base.name + "_heatmap")
    bar_base = args.output_base.with_name(args.output_base.name + "_bar")
    curve_files = plot_cumulative_recall_curve(summary, curve_base)
    heatmap_files = plot_heatmap(summary, heatmap_base)
    bar_files = plot_grouped_bar(summary, bar_base)

    print("Detected app blocks:")
    for app, cols in app_columns.items():
        print(f"- {app}: Excel columns {cols[0] + 1}-{cols[-1] + 1}")

    for fig_path in curve_files + heatmap_files + bar_files:
        print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {args.summary_csv}")
    print()
    print(summary.to_string(index=False, formatters={"Percent": lambda v: f"{v:.1f}"}))


if __name__ == "__main__":
    main()

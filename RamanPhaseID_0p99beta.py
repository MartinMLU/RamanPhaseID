"""
Raman-Matcher – Streamlit-GUI + CLI
Dual-DB mit wählbarer Messungs-Preprocessing-Variante: vergleicht Messung
(**BC oder RAW**) gegen **DB-RAW & DB-BC**.
Scoring nutzt standardmäßig **20% Gradient-Similarity** (baseline-invariant) + 80% Form.
Integriert Baseline-Parameter aus der separaten Baseline-App (IAsLS/arPLS).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re  # Regex + Unicode-Maps
import sys

import matplotlib
if os.getenv("DISPLAY", "") == "" and os.getenv("MPLBACKEND", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading
import time  # Used only by the graceful local-app shutdown callback.

import numpy as np
import raman_ai_denoiser as ai_denoiser
import raman_core as rc
import raman_database as rdb
import raman_exports as rex
import raman_matching as rmatch
import raman_plotting as rplot
import raman_preprocessing as rprep
import raman_workflow as rwf
import streamlit as st

HAVE_SCIPY_BASELINE = rprep.HAVE_SCIPY


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
APP_VERSION            = "0.99beta"
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

MATCH_FOLDERS = (
    DEFAULT_OWN_DB_DIR,
    DEFAULT_ROD_DIR,
    DEFAULT_RRUFF_DIR,
)

# Precompute-Ziel
PRECOMP_ROOT = BASE_DIR / "precomputed"
PRECOMP_PAIR_WORKER_BUDGET_DEFAULT = 6
PRECOMP_SYSTEMIC_FAILURE_FRACTION_DEFAULT = 0.90

# RamanPhaseID keeps one database-cache profile. Matching remains range-local,
# and the UI initially limits that range to the fingerprint region below.
DATABASE_GRID = {"min": 60, "max": 4000, "step": 1}
DEFAULT_MATCHING_HIGH_CM1 = 2000
MATCHING_CONTROL_SCHEMA_VERSION = 2

# Baseline and measurement preprocessing parameters are defined on this
# physical grid, rather than in native detector samples. Keeping the working
# step fixed makes lambda and a Savitzky-Golay window comparable across spectra
# from instruments/databases with different sampling intervals. The AI model
# retains its fixed 500-channel/1300 cm⁻¹ contract inside overlapping windows;
# the guarded adapter covers the complete measurement support.
PREPROCESS_GRID_STEP_CM1 = rprep.PREPROCESS_GRID_STEP_CM1
PREPROCESS_PIPELINE_VERSION = rprep.PREPROCESS_PIPELINE_VERSION
DEFAULT_PROCESSING_DIFFERENCE_MAGNIFICATION = 2.0

DEFAULT_TOP_N = 60
MAX_OVERLAY_MINERALS = 12   # max. gleichzeitig eingegebene Mineralnamen
MATCH_SELECTION_VERSION = 7
PLOT_RENDER_VERSION = 12
REFERENCE_FILTER_POLICY_VERSION = 2
TOP_PER_MINERAL_CAP = 5
PCS_MINERAL_SLOT_CAP = 12
PCS_MINERAL_MIN_PCS = 0.52
PCS_MINERAL_MIN_SIM = 0.50

# Scoring
GRAD_WEIGHT = 0.20  # 20% Gradient-Similarity
PCS_F1_WEIGHT = 0.75
PCS_PEAK_TOL = 5
FINAL_SIM_WEIGHT = 0.88
COSINE_SCREEN_CHUNK_ROWS = 1024

MATCHING_PARAMETERS = rmatch.MatchingParameters(
    gradient_weight=GRAD_WEIGHT,
    spectral_similarity_weight=FINAL_SIM_WEIGHT,
    peak_f1_weight=PCS_F1_WEIGHT,
    peak_tolerance_points=PCS_PEAK_TOL,
    screen_chunk_rows=COSINE_SCREEN_CHUNK_ROWS,
    per_phase_cap=TOP_PER_MINERAL_CAP,
    peak_phase_slot_cap=PCS_MINERAL_SLOT_CAP,
    peak_phase_minimum_score=PCS_MINERAL_MIN_PCS,
    peak_phase_minimum_similarity=PCS_MINERAL_MIN_SIM,
)
EVIDENCE_DECISION_POLICY = rmatch.EvidenceDecisionPolicy()
RESIDUAL_SEARCH_POLICY = rmatch.ResidualSearchPolicy()
RESIDUAL_MATCHING_PARAMETERS = replace(
    MATCHING_PARAMETERS,
    # A residual is physically signed: negative over-subtraction is contrary
    # evidence for adding a non-negative reference phase. The primary matcher
    # remains locally offset-invariant, but residual scoring must not lift the
    # deepest negative trough to zero and turn the rest into a broad positive
    # pseudo-spectrum.
    remove_query_local_offset=False,
    # A second crystalline phase needs a minimally coherent coincident peak
    # pattern. Broad/raw background similarity alone is not phase evidence,
    # even when its cosine score is numerically high. This is an operational,
    # uncalibrated guardrail rather than a probability threshold.
    minimum_candidate_peak_consistency=0.15,
)
RESIDUAL_REFERENCE_VARIANT_POLICY = "background-neutral-all-configured-v3"
RESIDUAL_ACTIONABLE_EVIDENCE_STATUSES = frozenset(
    {
        rmatch.EvidenceStatus.SUPPORTED_CANDIDATE.value,
        rmatch.EvidenceStatus.AMBIGUOUS.value,
    }
)
MATCHING_POLICY_SIGNATURE = rwf.payload_signature(
    {
        "v": 5,
        "matching_parameters": MATCHING_PARAMETERS.payload(),
        "evidence_decision_policy": EVIDENCE_DECISION_POLICY.payload(),
        "reference_filter_policy_version": REFERENCE_FILTER_POLICY_VERSION,
    }
)
RESIDUAL_SEARCH_POLICY_SIGNATURE = rwf.payload_signature(
    {
        "v": 6,
        "projection": RESIDUAL_SEARCH_POLICY.payload(),
        "matching_parameters": RESIDUAL_MATCHING_PARAMETERS.payload(),
        "reference_variant_policy": RESIDUAL_REFERENCE_VARIANT_POLICY,
        "actionable_evidence_statuses": sorted(
            RESIDUAL_ACTIONABLE_EVIDENCE_STATUSES
        ),
    }
)
PRIMARY_RESULT_SNAPSHOT_KEY = "primary_result_snapshot"
RESIDUAL_RESULT_SNAPSHOT_KEY = "residual_result_snapshot"

# Formel-Parser
ELEMENT_PARSER_VERSION = 1
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


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


def _resolve_systemic_failure_fraction() -> float:
    raw = os.getenv("RAMAN_PRECOMP_SYSTEMIC_FAILURE_FRACTION", "").strip()
    if not raw:
        return PRECOMP_SYSTEMIC_FAILURE_FRACTION_DEFAULT
    try:
        requested = float(raw)
    except ValueError:
        return PRECOMP_SYSTEMIC_FAILURE_FRACTION_DEFAULT
    return float(np.clip(requested, 0.50, 1.0))


def _extract_formula_elements(formula: str) -> list[str]:
    if not formula or formula.startswith("?"):
        return []
    els = {tok for tok in _ELEMENT_RE.findall(formula)}
    return sorted(els)


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


def _compute_db_signature(folders: list[Path]) -> str:
    """Return the explicitly managed, nanosecond-resolution DB inventory key."""

    return _inventory_snapshot(folders).signature


def _compute_signature_with_grid(folders: list[Path], grid_min: int, grid_max: int, grid_step: int) -> str:
    base = _compute_db_signature(folders)
    h = hashlib.sha1()
    h.update(
        (
            f"{base}|g:{grid_min}-{grid_max}-{grid_step}|p:{ELEMENT_PARSER_VERSION}"
            f"|prep:{PREPROCESS_PIPELINE_VERSION}@{PREPROCESS_GRID_STEP_CM1:g}cm-1|dbsg:off"
        ).encode()
    )
    return h.hexdigest()


def _initial_matching_range(grid_low: int, grid_high: int) -> tuple[int, int]:
    """Return the first range shown for one measurement on the fixed DB grid."""

    low = int(grid_low)
    high = int(grid_high)
    if high <= low:
        raise ValueError("matching range requires an increasing interval")
    initial_high = min(high, DEFAULT_MATCHING_HIGH_CM1)
    if initial_high <= low:
        initial_high = high
    return low, initial_high


# ────────────────────────────────────────────────────────────────────────────
# Processing-Helfer

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _toggle_bool_session_state(key: str, default: bool = False) -> None:
    """Toggle one display-only boolean before Streamlit renders the next run."""

    st.session_state[str(key)] = not bool(st.session_state.get(str(key), default))


def _download_button(*args, **kwargs):
    """Serve prepared exports without triggering an unrelated full-app rerun."""

    kwargs.setdefault("on_click", "ignore")
    return getattr(st, "download_button")(*args, **kwargs)


def _delay_then_rerun(delay_seconds: float = 0.0):
    """Compatibility shim: approval reruns must never block the server thread."""
    _safe_rerun()


def _primary_result_snapshot() -> rwf.PrimaryResultSnapshot | None:
    value = st.session_state.get(PRIMARY_RESULT_SNAPSHOT_KEY)
    return value if isinstance(value, rwf.PrimaryResultSnapshot) else None


def _residual_result_snapshot() -> rwf.ResidualResultSnapshot | None:
    value = st.session_state.get(RESIDUAL_RESULT_SNAPSHOT_KEY)
    return value if isinstance(value, rwf.ResidualResultSnapshot) else None


def _clear_residual_session_state() -> None:
    """Remove every derived residual artifact after an upstream approval changes."""

    for key in (
        RESIDUAL_RESULT_SNAPSHOT_KEY,
        "top_combined_residual",
        "residual_mode_active",
        "residual_search_info",
        "residual_parent_identity",
    ):
        st.session_state.pop(key, None)


def _render_stale_result_summary(
    snapshot: rwf.PrimaryResultSnapshot | None = None,
) -> None:
    """Keep prior evidence visible without evaluating or loading a new search."""

    previous = snapshot or _primary_result_snapshot()
    if previous is None:
        return
    config = previous.matching_approval.config
    range_text = (
        f"{config.range_cm1.low:g}–{config.range_cm1.high:g} cm⁻¹"
        if config.range_cm1 is not None
        else "unknown range"
    )
    st.warning(
        "Previous database-matching results are retained below as **stale**. "
        "They describe the last completed approval, not the draft settings currently "
        "shown. No database search has started."
    )
    if previous.is_empty:
        st.caption(
            f"Last completed search ({range_text}, result {previous.identity.token}) "
            "returned no candidate traces."
        )
        return
    names = [str(result.get("name", "unnamed")) for result in previous.results[:5]]
    suffix = "" if len(previous.results) <= 5 else f" · +{len(previous.results) - 5} more"
    st.caption(
        f"Last completed search ({range_text}, result {previous.identity.token}): "
        f"{', '.join(names)}{suffix}"
    )


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
        muted_text = "#44515E"
        hover_bg = "#DCE4EE"
        matching_idle_bg = "#D1D5DB"
        matching_idle_text = "#1F2933"
    else:
        app_bg = "#0E1117"
        sidebar_bg = "#161B22"
        text = "#E6EDF3"
        button_bg = "#1D2530"
        button_text = "#E6EDF3"
        input_bg = "#11161D"
        border = "#3B4652"
        accent = "#7DCBFF"
        muted_text = "#B7C0CA"
        hover_bg = "#26313E"
        matching_idle_bg = "#4B5563"
        matching_idle_text = "#FFFFFF"

    dark_tooltip_icon_css = ""
    if theme == "dark":
        dark_tooltip_icon_css = """
        /* Keep Streamlit's HelpCircle readable on dark displays. The broad
           SVG rule below used to fill both the circle and question mark with
           one grey; separate them into a dark disc and a light glyph. */
        [data-testid="stTooltipIcon"] svg.icon {
            fill: none !important;
        }
        [data-testid="stTooltipIcon"] svg.icon circle {
            fill: #2A333D !important;
            stroke: #667382 !important;
        }
        [data-testid="stTooltipIcon"] svg.icon path {
            fill: none !important;
            stroke: #F2F5F8 !important;
        }
        """

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {app_bg};
            color: {text};
            color-scheme: {theme};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        /* Keep scientific figures readable while Streamlit marks the previous
           run stale. The standard running indicator is paired with a progress
           cursor instead of dimming every element to one-third opacity. */
        [data-testid="stElementContainer"] {{
            opacity: 1 !important;
            transition: opacity 0s !important;
        }}
        body:has([data-testid="stStatusWidgetRunningIcon"])
            [data-testid="stAppViewContainer"],
        body:has([data-testid="stStatusWidgetRunningManIcon"])
            [data-testid="stAppViewContainer"],
        body:has([data-testid="stStatusWidgetRunningIcon"])
            [data-testid="stSidebar"],
        body:has([data-testid="stStatusWidgetRunningManIcon"])
            [data-testid="stSidebar"] {{
            cursor: progress !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {sidebar_bg};
        }}
        /* Streamlit's generated widget styles continue to use the browser's
           native Streamlit theme after the app-level theme is switched. Set
           explicit foregrounds for every text-bearing component so a native
           dark theme cannot leave white widget text on our light panels. */
        [data-testid="stAppViewContainer"] :is(
            [data-testid="stHeading"],
            [data-testid="stMarkdownContainer"],
            [data-testid="stWidgetLabel"],
            [data-testid="stText"],
            [data-testid="stImageCaption"],
            [data-testid="stSliderThumbValue"],
            [data-testid="stSliderTickBar"],
            [data-testid="stSpinner"],
            [data-testid="stProgress"],
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            .stCaptionContainer
        ),
        [data-testid="stSidebar"] :is(
            [data-testid="stHeading"],
            [data-testid="stMarkdownContainer"],
            [data-testid="stWidgetLabel"],
            [data-testid="stText"],
            [data-testid="stImageCaption"],
            [data-testid="stSliderThumbValue"],
            [data-testid="stSliderTickBar"],
            [data-testid="stSpinner"],
            [data-testid="stProgress"],
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            .stCaptionContainer
        ) {{
            color: {text} !important;
        }}
        [data-testid="stAppViewContainer"] :is(
            [data-testid="stHeading"],
            [data-testid="stMarkdownContainer"],
            [data-testid="stWidgetLabel"],
            [data-testid="stText"],
            [data-testid="stImageCaption"],
            [data-testid="stSliderThumbValue"],
            [data-testid="stSliderTickBar"],
            [data-testid="stSpinner"],
            [data-testid="stProgress"],
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            .stCaptionContainer
        ) *,
        [data-testid="stSidebar"] :is(
            [data-testid="stHeading"],
            [data-testid="stMarkdownContainer"],
            [data-testid="stWidgetLabel"],
            [data-testid="stText"],
            [data-testid="stImageCaption"],
            [data-testid="stSliderThumbValue"],
            [data-testid="stSliderTickBar"],
            [data-testid="stSpinner"],
            [data-testid="stProgress"],
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            .stCaptionContainer
        ) * {{
            color: inherit !important;
        }}
        [data-testid="stAppViewContainer"] .stCaptionContainer,
        [data-testid="stAppViewContainer"] .stCaptionContainer *,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] .stCaptionContainer * {{
            color: {muted_text} !important;
        }}
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] a,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {{
            color: {accent} !important;
        }}
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] code,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {{
            background-color: {hover_bg} !important;
            color: {text} !important;
        }}
        [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
        [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] * {{
            color: {muted_text} !important;
        }}
        [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] svg,
        [data-testid="stHeader"] button svg,
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stExpandSidebarButton"] svg {{
            fill: currentColor !important;
        }}
        {dark_tooltip_icon_css}
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] button *,
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stExpandSidebarButton"],
        [data-testid="stExpandSidebarButton"] * {{
            color: {text} !important;
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
        .stButton > button * {{
            color: {button_text} !important;
        }}
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primary"] {{
            background-color: var(--rm-primary) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--rm-primary) !important;
            transition: background-color 0.2s ease, border-color 0.2s ease, filter 0.2s ease;
        }}
        .stButton > button[kind="primary"] *,
        .stButton > button[data-testid="stBaseButton-primary"] *,
        button[data-testid="stBaseButton-primary"] * {{
            color: #FFFFFF !important;
        }}
        .st-key-update_database_matching_btn button[kind="secondary"],
        .st-key-update_database_matching_btn
            button[data-testid="stBaseButton-secondary"] {{
            background-color: {matching_idle_bg} !important;
            color: {matching_idle_text} !important;
            border-color: {matching_idle_bg} !important;
        }}
        .st-key-update_database_matching_btn button[kind="secondary"] *,
        .st-key-update_database_matching_btn
            button[data-testid="stBaseButton-secondary"] * {{
            color: {matching_idle_text} !important;
        }}
        .stDownloadButton > button {{
            background-color: {button_bg};
            color: {button_text};
            border: 1px solid {border};
        }}
        .stDownloadButton > button * {{
            color: {button_text} !important;
        }}
        .stButton > button:hover {{
            border-color: {accent};
            color: {accent};
        }}
        .stButton > button:hover * {{
            color: {accent} !important;
        }}
        .stDownloadButton > button:hover {{
            border-color: {accent};
            color: {accent};
        }}
        .stDownloadButton > button:hover * {{
            color: {accent} !important;
        }}
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {{
            background-color: var(--rm-primary) !important;
            color: #FFFFFF !important;
            border-color: var(--rm-primary) !important;
            filter: brightness(1.03);
        }}
        .stButton > button[kind="primary"]:hover *,
        .stButton > button[data-testid="stBaseButton-primary"]:hover *,
        button[data-testid="stBaseButton-primary"]:hover * {{
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"]:active,
        .stButton > button[data-testid="stBaseButton-primary"]:active,
        button[data-testid="stBaseButton-primary"]:active {{
            background-color: var(--rm-primary-pressed) !important;
            border-color: var(--rm-primary-pressed-border) !important;
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"]:active *,
        .stButton > button[data-testid="stBaseButton-primary"]:active *,
        button[data-testid="stBaseButton-primary"]:active * {{
            color: #FFFFFF !important;
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {{
            background-color: {input_bg};
            color: {text} !important;
            -webkit-text-fill-color: {text} !important;
        }}
        [data-baseweb="input"] input::placeholder,
        [data-baseweb="textarea"] textarea::placeholder,
        [data-testid="stNumberInput"] input::placeholder,
        [data-testid="stTextInput"] input::placeholder {{
            color: {muted_text} !important;
            -webkit-text-fill-color: {muted_text} !important;
            opacity: 1;
        }}
        [data-baseweb="select"] *,
        [data-baseweb="input"] *,
        [data-baseweb="textarea"] *,
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] button * {{
            color: {text} !important;
        }}
        [data-testid="stNumberInput"] button {{
            background-color: {input_bg} !important;
            border-color: {border} !important;
        }}
        [data-baseweb="select"] svg,
        [data-testid="stNumberInput"] svg {{
            color: {text} !important;
            fill: currentColor !important;
        }}
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"] {{
            background-color: {button_bg} !important;
            color: {text} !important;
        }}
        [data-baseweb="popover"] [role="option"] *,
        [data-baseweb="popover"] [role="listbox"] * {{
            color: {text} !important;
        }}
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [role="option"][aria-selected="true"] {{
            background-color: {hover_bg} !important;
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
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploaderFile"] * {{
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
        [data-testid="stTable"] th,
        [data-testid="stTable"] td {{
            background-color: {button_bg} !important;
            border-color: {border} !important;
            color: {text} !important;
        }}
        [data-testid="stDataFrame"] button,
        [data-testid="stDataFrame"] button * {{
            color: {text} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_precompute_pair_core(
    signature_raw: str,
    signature_bcb: str,
    folders: tuple[Path, ...],
    grid_min: int,
    grid_max: int,
    grid_step: int,
    *,
    baseline_cfg: dict,
    workers: int | None = None,
    invalid_commit_created_at_ns: int | None = None,
    progress_callback: rdb.PairBuildProgressCallback | None = None,
) -> rdb.PairBuildReport:
    """Thin application adapter around the typed database/cache transaction."""

    def discover(paths: tuple[Path, ...]):
        # A cache build needs header identities, but it must not implicitly
        # populate the optional long-lived sidebar catalog resource.
        loader = getattr(rc.load_reference_folders, "__wrapped__", rc.load_reference_folders)
        entries, _skipped = loader(paths)
        return entries

    def parse(path: Path) -> tuple[np.ndarray, np.ndarray]:
        return rc._parse_rruff(path) if path.suffix.casefold() == ".txt" else rc._parse_rod(path)

    def preprocess(
        x: np.ndarray,
        y: np.ndarray,
        target_grid: np.ndarray,
        apply_baseline: bool,
    ) -> np.ndarray:
        return rprep.process_db_on_target_grid(
            x,
            y,
            target_grid,
            apply_baseline_db=apply_baseline,
            baseline_cfg=baseline_cfg,
        )

    grid = rdb.GridSpec(
        minimum=float(grid_min),
        maximum=float(grid_max),
        step=float(grid_step),
        length=((int(grid_max) - int(grid_min)) // int(grid_step)) + 1,
    )
    request = rdb.PairBuildRequest(
        cache_root=PRECOMP_ROOT,
        raw_signature=signature_raw,
        baseline_signature=signature_bcb,
        roots=_database_roots(folders),
        inventory_signature=_compute_db_signature(list(folders)),
        grid=grid,
        parser_version=ELEMENT_PARSER_VERSION,
        preprocess_version=PREPROCESS_PIPELINE_VERSION,
        preprocess_step_cm1=PREPROCESS_GRID_STEP_CM1,
        workers=(
            _resolve_precompute_pair_workers() if workers is None else max(1, int(workers))
        ),
        systemic_failure_fraction=_resolve_systemic_failure_fraction(),
        invalid_commit_created_at_ns=invalid_commit_created_at_ns,
    )
    return rdb.build_precompute_pair(
        request,
        discover_references=discover,
        parse_spectrum=parse,
        preprocess_vector=preprocess,
        clean_xy=rprep.clean_xy,
        support_slices=rprep.support_slices,
        extract_elements=_extract_formula_elements,
        progress_callback=progress_callback,
    )


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
    q = _process_measurement(
        meas_x,
        meas_y,
        apply_baseline=apply_baseline,
        baseline_cfg=cfg,
        smoothing_cfg=smoothing_cfg,
        target_x=grid,
    ).astype(np.float32)
    q[~np.isfinite(q)] = 0.0
    q_mask = (grid >= range_low) & (grid <= range_high)
    q[~q_mask] = 0.0
    q_l2 = float(np.linalg.norm(q))
    return q, q_l2, q_mask


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
# Ergebnis-Berechnung (teuer) – ausgelagert & nur bei Signaturwechsel

def _result_uses_baseline_pack(result: Mapping[str, object]) -> bool:
    """Resolve the exact cache variant that produced a scored result.

    ``db_variant`` records which matrix was searched. ``db_baseline`` instead
    records whether an additional baseline operation was actually applied to
    that row, and can legitimately be false for an already-processed trace in
    the baseline pack. Therefore it must not select the reconstruction matrix.
    """

    variant = str(result.get("db_variant", "")).strip().casefold()
    if variant in {"db-bc", "baseline", "baseline-corrected"}:
        return True
    if variant in {"db-raw", "raw", "library-as-provided"}:
        return False
    return bool(result.get("db_baseline", False))


def _enrich_selected_provenance(
    candidates: list[dict],
    pack_raw: dict,
    pack_bcb: dict,
) -> list[dict]:
    """Parse source provenance only for the short displayed result list."""

    enriched: list[dict] = []
    for item in candidates:
        result = dict(item)
        pack = pack_bcb if _result_uses_baseline_pack(result) else pack_raw
        index = int(result.get("db_idx", -1))
        metadata = pack["meta"][index] if 0 <= index < len(pack["meta"]) else {}
        provenance = _reference_provenance(metadata) if metadata else {}
        if provenance:
            result["database_source"] = provenance.get("database", "")
            result["source"] = provenance.get("source") or provenance.get("database", "")
            result["accession"] = provenance.get("accession") or result.get("accession", "")
            result["quality"] = provenance.get("quality", "")
            result["quality_folder"] = provenance.get("quality_folder", "")
            result["reference_processing"] = provenance.get("processing", "")
            result["determination"] = provenance.get("determination", "")
            result["orientation"] = provenance.get("orientation", "")
            result["orientation_detail"] = provenance.get("orientation_detail", "")
            result["excitation_wavelength_nm"] = provenance.get(
                "excitation_wavelength_nm"
            )
            result["resolution_cm1"] = provenance.get("resolution_cm1")
            result["measured_chemistry"] = provenance.get("measured_chemistry", "")
            result["correction_history"] = provenance.get("correction_history", [])
        enriched.append(result)
    return enriched


def _build_residual_query_vector(
    q_sel: np.ndarray,
    q_mask: np.ndarray,
    selected_match: dict,
    pack_raw: dict,
    pack_bcb: dict,
) -> rmatch.ResidualProjection | None:
    db_idx = int(selected_match.get("db_idx", -1))
    if db_idx < 0:
        return None

    use_bcb = _result_uses_baseline_pack(selected_match)
    pack_sel = pack_bcb if use_bcb else pack_raw
    if db_idx >= len(pack_sel["meta"]):
        return None

    cand = np.asarray(pack_sel["X"][db_idx, :], dtype=float)
    shift = int(selected_match.get("shift", 0))
    start_idx = int(selected_match.get("start_idx", 0))
    end_idx = int(selected_match.get("end_idx", -1))
    try:
        return rmatch.build_residual_projection(
            q_sel,
            cand,
            q_mask,
            start_idx,
            end_idx,
            shift,
            minimum_common_points=RESIDUAL_SEARCH_POLICY.minimum_common_points,
            support_edge_guard_points=(
                RESIDUAL_SEARCH_POLICY.support_edge_guard_points
            ),
            support_runs=selected_match.get("support_runs"),
        )
    except (TypeError, ValueError):
        return None


def _residual_reference_identity(
    selected_match: dict,
    raw_database_signature: str,
    baseline_database_signature: str,
    grid_step_cm1: float,
) -> rwf.ResidualReferenceIdentity:
    """Capture the exact approved cache row and alignment being subtracted."""

    use_bcb = _result_uses_baseline_pack(selected_match)
    variant = "DB-BC" if use_bcb else "DB-RAW"
    database_index = int(selected_match.get("db_idx", -1))
    path = str(selected_match.get("path", "")).strip()
    accession = str(selected_match.get("accession", "")).strip()
    filename = str(
        selected_match.get("orig_filename", selected_match.get("filename", ""))
    ).strip()
    reference_id = str(selected_match.get("reference_id", "")).strip() or (
        path or accession or f"{variant}:{database_index}:{filename}"
    )
    start_idx = int(selected_match.get("start_idx", 0))
    end_idx = int(selected_match.get("end_idx", -1))
    support_runs = selected_match.get("support_runs")
    if not support_runs:
        support_runs = ((start_idx, end_idx),)
    fitted_shift_points = int(selected_match.get("shift", 0))
    return rwf.ResidualReferenceIdentity(
        phase_name=str(selected_match.get("name", "")),
        database_variant=variant,
        database_signature=(
            baseline_database_signature if use_bcb else raw_database_signature
        ),
        database_index=database_index,
        reference_id=reference_id,
        path=path,
        accession=accession,
        filename=filename,
        fitted_shift_points=fitted_shift_points,
        fitted_shift_cm1=float(
            selected_match.get(
                "shift_cm1",
                fitted_shift_points * float(grid_step_cm1),
            )
        ),
        start_idx=start_idx,
        end_idx=end_idx,
        support_runs=tuple(tuple(run) for run in support_runs),
    )


# Public implementations now live in focused, independently testable modules.
# These aliases preserve the pre-refactor private API while the Streamlit view
# is migrated to typed artifacts.
_split_header_data = rex.split_header_data
_parse_xy_from_data_lines = rex.parse_xy_from_data_lines
_rebuild_file_bytes = rex.rebuild_spectrum_bytes

_baseline_iasls = rprep.baseline_iasls
_baseline_arpls = rprep.baseline_arpls
_default_baseline_cfg = rprep._default_baseline_cfg
_fixed_db_baseline_cfg = rprep._fixed_db_baseline_cfg
_baseline_cfg_payload = rprep._baseline_cfg_payload
_baseline_cfg_token = rprep._baseline_cfg_token
_baseline_label = rprep._baseline_label
_compute_baseline = rprep.compute_baseline
_default_smoothing_cfg = rprep._default_smoothing_cfg
_smoothing_method = rprep._smoothing_method
_smoothing_cfg_payload = rprep._smoothing_cfg_payload
_smoothing_cfg_token = rprep._smoothing_cfg_token
_smoothing_label = rprep._smoothing_label
_smoothing_preview_ui = rprep._smoothing_preview_ui
_align_reference_to_target = rprep._align_reference_to_target
_sanitize_savgol_params = rprep.sanitize_savgol_params
_apply_smoothing = rprep.apply_smoothing
_clean_xy_for_preprocessing = rprep.clean_xy
_canonical_grid_for_support = rprep.canonical_grid
_compute_baseline_on_axis = rprep.compute_baseline_on_axis
_normalize_to_peak = rprep.normalize_to_peak
_prepare_measurement_signal = rprep.prepare_measurement_signal
_prepare_db_signal_on_target_grid = rprep.prepare_db_signal_on_target_grid
_process_measurement = rprep.process_measurement
_process_db_on_target_grid = rprep.process_db_on_target_grid

_shift_candidate = rmatch.shift_candidate
_peak_consistency_score = rmatch.peak_consistency_score
_topk_cosine_subset = rmatch.topk_cosine_subset
_refine_and_rank = rmatch.refine_and_rank
_annotate_phase_evidence = rmatch.annotate_phase_evidence
_select_diverse_top = rmatch.select_diverse_top
_mineral_key = rmatch.phase_key
_final_rank_score = rmatch.final_rank_score


def _compute_matches_from_query_vector_unlimited(
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
    excluded_phase_keys: tuple[str, ...] = (),
    matching_parameters: rmatch.MatchingParameters | None = None,
) -> list[dict]:
    results = rmatch.match_query_vector(
        q_sel,
        q_mask,
        range_low,
        range_high,
        pack_raw,
        pack_bcb,
        allowed_ids_raw,
        allowed_ids_bcb,
        meas_mode,
        top_n=top_n,
        parameters=matching_parameters or MATCHING_PARAMETERS,
        evidence_policy=EVIDENCE_DECISION_POLICY,
        excluded_phase_keys=excluded_phase_keys,
    )
    return _enrich_selected_provenance(results, pack_raw, pack_bcb)


def _compute_matches_from_query_vector(*args, **kwargs) -> list[dict]:
    """Run the memory-bound matcher with a deployment-tunable BLAS cap."""

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return _compute_matches_from_query_vector_unlimited(*args, **kwargs)
    try:
        requested = int(os.getenv("RAMAN_MATCH_BLAS_THREADS", "2"))
    except ValueError:
        requested = 2
    limit = max(1, min(requested, 4))
    with threadpool_limits(limits=limit, user_api="blas"):
        return _compute_matches_from_query_vector_unlimited(*args, **kwargs)


def _aligned_mask(
    q_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    k: int,
    M: int,
) -> np.ndarray:
    if len(q_mask) != int(M):
        raise ValueError("query mask length does not match matching grid")
    return rmatch.aligned_support_mask(q_mask, start_idx, end_idx, k)


def _best_aligned_score(
    query: np.ndarray,
    cand: np.ndarray,
    q_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    max_shift: int = 5,
    grad_weight: float = GRAD_WEIGHT,
) -> tuple[float, int]:
    aligned = rmatch.best_aligned_score(
        query,
        cand,
        q_mask,
        start_idx,
        end_idx,
        max_shift=max_shift,
        gradient_weight=grad_weight,
    )
    return aligned.spectral_similarity, aligned.evidence.fitted_shift_points


_fig_to_bytes = rplot.figure_to_bytes
_normalize_plot_theme = rplot.normalize_plot_theme
_apply_plot_style = rplot.apply_plot_style
_set_intensity_number_visibility = rplot.set_intensity_number_visibility


def _database_roots(folders: tuple[str, ...] | tuple[Path, ...]) -> tuple[rdb.DatabaseRoot, ...]:
    roots: list[rdb.DatabaseRoot] = []
    used: set[str] = set()
    for index, folder in enumerate(folders):
        path = Path(folder).resolve()
        alias = path.name or f"db{index}"
        base_alias = alias
        suffix = 2
        while alias.casefold() in used:
            alias = f"{base_alias}-{suffix}"
            suffix += 1
        used.add(alias.casefold())
        roots.append(rdb.DatabaseRoot(alias, path))
    return tuple(roots)


@st.cache_resource(show_spinner=False)
def _database_inventory_manager(
    folders_as_str: tuple[str, ...],
) -> rdb.DatabaseInventoryManager:
    return rdb.DatabaseInventoryManager(_database_roots(folders_as_str))


def _inventory_snapshot(
    folders: tuple[Path, ...] | list[Path],
    *,
    refresh: bool = False,
) -> rdb.DatabaseInventory:
    folder_strings = tuple(str(Path(folder).resolve()) for folder in folders)
    manager = _database_inventory_manager(folder_strings)
    return manager.refresh() if refresh else manager.snapshot()


@st.cache_resource(show_spinner=False, max_entries=2)
def _get_db_entries_cached(
    folders_as_str: tuple[str, ...],
    inventory_signature: str,
) -> rdb.ReferenceCatalogSummary:
    """Load the optional searchable header catalog only on explicit request."""

    del inventory_signature  # Stable invalidation input for the resource key.
    # Keep only the compact summary in Streamlit's resource cache.  Calling
    # the decorated loader directly would also retain its complete entry tuple
    # in raman_core's LRU, defeating the purpose of this summary cache.
    loader = getattr(rc.load_reference_folders, "__wrapped__", rc.load_reference_folders)
    entries, skipped = loader(tuple(Path(value) for value in folders_as_str))
    return rdb.ReferenceCatalogSummary.from_entries(
        entries,
        skipped_count=len(skipped),
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_measurement_spectrum(
    content_sha256: str,
    _content: bytes,
) -> rprep.MeasurementSpectrum:
    """Parse and assess one upload once; SHA-256 is the complete cache key."""

    if len(content_sha256) != 64:
        raise ValueError("measurement cache key must be a SHA-256 digest")
    return rprep.parse_measurement_text(_content.decode("utf-8", errors="ignore"))


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_spectrum_text_layout(
    content_sha256: str,
    _content: bytes,
) -> rex.SpectrumTextLayout:
    """Inspect original headers/body once for faithful processed exports."""

    if len(content_sha256) != 64:
        raise ValueError("export-layout cache key must be a SHA-256 digest")
    return rex.inspect_spectrum_text(_content.decode("utf-8", errors="ignore"))


def _plot_render_signature(kind: str, payload: dict[str, object]) -> str:
    body = {
        "version": PLOT_RENDER_VERSION,
        "kind": str(kind),
        "payload": payload,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_figure_render_bundle(
    render_signature: str,
    _figure_factory,
) -> rplot.FigureRenderBundle:
    """Create and serialize a figure only once per complete semantic key."""

    if len(render_signature) != 64:
        raise ValueError("figure render key must be a SHA-256 digest")
    figure = _figure_factory()
    try:
        return rplot.render_figure_bundle(figure)
    finally:
        plt.close(figure)


@st.cache_data(show_spinner=False, persist="disk", max_entries=8192)
def _provenance_payload_cached(
    path_str: str,
    size_bytes: int,
    mtime_ns: int,
    database_hint: str = "",
) -> dict[str, object]:
    del size_bytes, mtime_ns  # Stable invalidation inputs for Streamlit's key.
    try:
        value = rdb.parse_spectrum_provenance(
            Path(path_str),
            database_hint=database_hint or None,
        )
    except rdb.DatabaseError:
        return {}
    return {
        "database": value.database,
        "accession": value.accession,
        "source": value.source,
        "status": value.status,
        "quality": value.quality,
        "quality_folder": value.quality_folder,
        "processing": value.processing,
        "determination": value.determination,
        "orientation": value.orientation,
        "orientation_detail": value.orientation_detail,
        "excitation_wavelength_nm": value.excitation_wavelength_nm,
        "resolution_cm1": value.resolution_cm1,
        "measured_chemistry": value.chemistry.measured,
        "correction_history": list(value.correction_history),
    }


def _reference_provenance(metadata: Mapping[str, object]) -> Mapping[str, object]:
    embedded = metadata.get("provenance")
    if isinstance(embedded, Mapping) and embedded:
        return embedded
    path = Path(str(metadata.get("path", "")))
    try:
        stat = path.stat()
    except OSError:
        return {}
    return _provenance_payload_cached(
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        str(metadata.get("source_root", "")),
    )


def _reference_is_already_processed(metadata: Mapping[str, object]) -> bool:
    embedded = metadata.get("provenance")
    if isinstance(embedded, Mapping):
        processing = str(embedded.get("processing", "unknown"))
        if processing != "unknown":
            return processing == "processed"
    path = Path(str(metadata.get("orig_filename", "") or metadata.get("path", "")))
    filename = path.name.casefold()
    if "raman_data_processed" in filename or "raman processed" in filename:
        return True
    if "raman_data_raw" in filename or "raman raw" in filename:
        return False
    if path.suffix.casefold() == ".rod":
        return str(_reference_provenance(metadata).get("processing", "unknown")) == "processed"
    return False


def _background_neutral_residual_reference_ids(
    allowed_raw_ids: np.ndarray,
    allowed_baseline_ids: np.ndarray,
    raw_metadata,
    baseline_metadata,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one background-neutral representation for residual rematching.

    Raw-source library backgrounds are instrument/sample specific and must not
    become evidence for a second phase. Already-processed source spectra remain
    in the as-provided pack; raw sources use their paired baseline-corrected
    vector. The primary matcher still searches both representations.
    """

    raw_ids = np.asarray(allowed_raw_ids, dtype=np.int32).reshape(-1)
    baseline_ids = np.asarray(allowed_baseline_ids, dtype=np.int32).reshape(-1)
    processed_raw = np.asarray(
        [
            int(index)
            for index in raw_ids
            if 0 <= int(index) < len(raw_metadata)
            and _reference_is_already_processed(raw_metadata[int(index)])
        ],
        dtype=np.int32,
    )
    corrected_raw = np.asarray(
        [
            int(index)
            for index in baseline_ids
            if 0 <= int(index) < len(baseline_metadata)
            and bool(baseline_metadata[int(index)].get("db_baseline", False))
        ],
        dtype=np.int32,
    )
    processed_raw.setflags(write=False)
    corrected_raw.setflags(write=False)
    return processed_raw, corrected_raw


def _residual_candidates_are_actionable(candidates: list[dict]) -> bool:
    """Return whether a residual ranking clears the display-level evidence gate."""

    if not candidates:
        return False
    return str(candidates[0].get("evidence_status", "")).strip() in (
        RESIDUAL_ACTIONABLE_EVIDENCE_STATUSES
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_reference_eligibility(
    request: rdb.ReferenceEligibilityRequest,
    _metadata,
) -> rdb.ReferenceEligibilityResult:
    """Cache one complete library-filter scan by its scientific identity."""

    return rdb.compute_reference_eligibility(
        _metadata,
        request,
        is_already_processed=_reference_is_already_processed,
    )


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_precompute_pair_resource(
    signature_raw: str,
    signature_bcb: str,
    folders_as_str: tuple[str, ...],
    inventory_refresh_token: str,
    require_pair_commit: bool,
) -> tuple[dict, dict]:
    del inventory_refresh_token  # Part of the resource-cache key by design.
    roots = _database_roots(folders_as_str)
    inventory = _database_inventory_manager(folders_as_str).snapshot()
    pair = rdb.load_precompute_pair(
        PRECOMP_ROOT / signature_raw,
        PRECOMP_ROOT / signature_bcb,
        roots=roots,
        inventory=inventory,
        expected_raw_signature=signature_raw,
        expected_baseline_signature=signature_bcb,
        expected_parser_version=ELEMENT_PARSER_VERSION,
        strict_sources=False,
        require_manifest=require_pair_commit,
        load_hnsw=False,
        require_aligned_metadata=True,
        pair_commit_root=PRECOMP_ROOT if require_pair_commit else None,
        require_pair_commit=require_pair_commit,
    )
    return pair.raw.matcher_view(), pair.baseline_corrected.matcher_view()


def _ensure_precompute_pair(
    signature_raw: str,
    signature_bcb: str,
    folders: tuple[Path, ...],
    grid_min: int,
    grid_max: int,
    grid_step: int,
    baseline_cfg: dict,
) -> tuple[dict, dict]:
    """Load one safe legacy pair or build/load one committed current pair."""
    required_names = ("X.float32.npy", "meta.json", "grid.json")
    raw_directory = PRECOMP_ROOT / signature_raw
    bcb_directory = PRECOMP_ROOT / signature_bcb
    raw_ready = all((raw_directory / name).is_file() for name in required_names)
    bcb_ready = all((bcb_directory / name).is_file() for name in required_names)
    folder_strings = tuple(str(Path(folder).resolve()) for folder in folders)
    inventory = _inventory_snapshot(list(folders))

    invalid_commit_created_at_ns: int | None = None
    try:
        pair_commit = rdb.read_pair_commit(
            PRECOMP_ROOT,
            signature_raw,
            signature_bcb,
            required=False,
        )
    except rdb.DatabaseError:
        pair_commit = None
    if pair_commit is not None:
        try:
            return _load_precompute_pair_resource(
                signature_raw,
                signature_bcb,
                folder_strings,
                inventory.refresh_token,
                True,
            )
        except rdb.DatabaseError:
            invalid_commit_created_at_ns = pair_commit.created_at_ns

    # Manifest-free caches predate pair commits.  Preserve their readability
    # only when both sides exist and the typed loader validates their alignment.
    raw_has_manifest = (raw_directory / rdb.DEFAULT_MANIFEST_FILE).is_file()
    bcb_has_manifest = (bcb_directory / rdb.DEFAULT_MANIFEST_FILE).is_file()
    if (
        pair_commit is None
        and raw_ready
        and bcb_ready
        and not raw_has_manifest
        and not bcb_has_manifest
    ):
        try:
            return _load_precompute_pair_resource(
                signature_raw,
                signature_bcb,
                folder_strings,
                inventory.refresh_token,
                False,
            )
        except rdb.DatabaseError:
            pass

    build_started_at = time.monotonic()
    last_progress_render_at = 0.0
    last_progress_rendered_count = -1
    with st.status(
        "Database cache build required — preparing reference spectra…",
        expanded=True,
    ) as build_status:
        progress_bar = st.progress(
            0.0,
            text="Inspecting the cache and database sources…",
        )
        progress_details = st.empty()

        def render_build_progress(progress: rdb.PairBuildProgress) -> None:
            nonlocal last_progress_render_at, last_progress_rendered_count
            now = time.monotonic()
            terminal_processing_update = bool(
                progress.stage == "processing"
                and progress.total_sources > 0
                and progress.completed_sources >= progress.total_sources
            )
            if progress.stage == "processing" and not terminal_processing_update:
                minimum_count_step = max(1, progress.total_sources // 500)
                if (
                    now - last_progress_render_at < 0.20
                    and progress.completed_sources - last_progress_rendered_count
                    < minimum_count_step
                ):
                    return
            last_progress_render_at = now
            last_progress_rendered_count = progress.completed_sources

            elapsed_seconds = max(0, int(now - build_started_at))
            elapsed_minutes, elapsed_remainder = divmod(elapsed_seconds, 60)
            elapsed_text = (
                f"{elapsed_minutes}m {elapsed_remainder:02d}s"
                if elapsed_minutes
                else f"{elapsed_remainder}s"
            )
            stage_labels = {
                "waiting_for_lock": "Waiting for any other cache builder to finish…",
                "recovering": "Checking for an interrupted cache build…",
                "discovering": "Discovering reference spectra and reading metadata…",
                "validating": "Validating the completed reference rows…",
                "writing": "Writing cache metadata and completion manifests…",
                "publishing": "Publishing the validated cache pair atomically…",
                "complete": "Database cache is ready.",
                "failed": "Database cache build failed.",
            }
            stage_fraction = {
                "waiting_for_lock": 0.0,
                "recovering": 0.01,
                "discovering": 0.02,
                "validating": 0.96,
                "writing": 0.97,
                "publishing": 0.99,
                "complete": 1.0,
                "failed": 0.0,
            }
            if progress.stage == "processing":
                percent = 100.0 * progress.fraction
                label = (
                    f"Processed {progress.completed_sources:,} of "
                    f"{progress.total_sources:,} reference spectra ({percent:.1f}%)"
                )
                bar_fraction = 0.02 + (0.93 * progress.fraction)
            else:
                label = stage_labels.get(
                    progress.stage,
                    "Preparing the database cache…",
                )
                bar_fraction = stage_fraction.get(progress.stage, 0.0)

            progress_bar.progress(
                float(np.clip(bar_fraction, 0.0, 1.0)),
                text=label,
            )
            detail_parts = [f"Elapsed: {elapsed_text}"]
            if progress.total_sources:
                detail_parts.extend(
                    (
                        f"usable: {progress.valid_rows:,}",
                        f"skipped/failed: {progress.failed_rows:,}",
                    )
                )
            if progress.current_source:
                source_label = Path(progress.current_source).name
                if len(source_label) > 100:
                    source_label = f"{source_label[:97]}…"
                detail_parts.append(f"latest: {source_label}")
            progress_details.caption(" · ".join(detail_parts))

        try:
            build_report = _build_precompute_pair_core(
                signature_raw=signature_raw,
                signature_bcb=signature_bcb,
                folders=folders,
                grid_min=grid_min,
                grid_max=grid_max,
                grid_step=grid_step,
                baseline_cfg=baseline_cfg,
                workers=_resolve_precompute_pair_workers(),
                invalid_commit_created_at_ns=invalid_commit_created_at_ns,
                progress_callback=render_build_progress,
            )
        except Exception:
            build_status.update(
                label="Database cache build failed",
                state="error",
                expanded=True,
            )
            raise
        else:
            build_status.update(
                label=(
                    f"Database cache ready — {build_report.valid_rows:,} usable "
                    f"reference spectra"
                ),
                state="complete",
                expanded=False,
            )
    _load_precompute_pair_resource.clear()
    return _load_precompute_pair_resource(
        signature_raw,
        signature_bcb,
        folder_strings,
        inventory.refresh_token,
        True,
    )


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_processed_spectrum(
    measurement_sha256: str,
    axis_cm1: np.ndarray,
    intensity: np.ndarray,
    apply_baseline: bool,
    baseline_payload_json: str,
    smoothing_payload_json: str,
) -> rprep.ProcessedSpectrum:
    """Create one immutable approved preprocessing artifact per signature."""

    del measurement_sha256  # Included deliberately in Streamlit's cache key.
    return rprep.preprocess_spectrum(
        axis_cm1,
        intensity,
        apply_baseline=apply_baseline,
        baseline_settings=json.loads(baseline_payload_json),
        smoothing_settings=json.loads(smoothing_payload_json),
    )


@st.cache_data(show_spinner=False)
def _runtime_export_metadata(
    source_stamps: tuple[tuple[str, int, int], ...],
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Capture exact source hashes and runtime versions for run manifests."""

    source_hashes: dict[str, str] = {}
    for relative_name, _size, _mtime_ns in source_stamps:
        source_path = BASE_DIR / relative_name
        try:
            source_hashes[relative_name] = rex.sha256_bytes(source_path.read_bytes())
        except OSError:
            source_hashes[relative_name] = "unavailable"
    versions = rex.installed_package_versions(
        {
            "matplotlib": "matplotlib",
            "numpy": "numpy",
            "pandas": "pandas",
            "scipy": "scipy",
            "streamlit": "streamlit",
            "threadpoolctl": "threadpoolctl",
            "torch": "torch",
        }
    )
    return rex.resolve_git_commit(BASE_DIR), versions, source_hashes


# ────────────────────────────────────────────────────────────────────────────
# Streamlit-GUI

def _run_streamlit() -> None:
    import pandas as pd

    st.set_page_config(page_title=f"RamanPhaseID {APP_VERSION}", layout="wide")
    app_theme_key = "app_theme"
    if app_theme_key not in st.session_state:
        st.session_state[app_theme_key] = "dark"
    app_theme = _normalize_app_theme(st.session_state.get(app_theme_key))
    st.session_state[app_theme_key] = app_theme
    _apply_app_theme(app_theme)
    st.title(f"RamanPhaseID - {APP_VERSION}")

    plot_theme_key = "plot_theme"
    if plot_theme_key not in st.session_state:
        st.session_state[plot_theme_key] = app_theme
    plot_theme = _normalize_plot_theme(st.session_state.get(plot_theme_key))
    st.session_state[plot_theme_key] = plot_theme
    plot_color_scheme_key = "plot_color_scheme"
    legacy_baseline_color_scheme_key = "baseline_preview_color_scheme"
    if (
        plot_color_scheme_key not in st.session_state
        and legacy_baseline_color_scheme_key in st.session_state
    ):
        st.session_state[plot_color_scheme_key] = st.session_state[
            legacy_baseline_color_scheme_key
        ]
    st.session_state.pop(legacy_baseline_color_scheme_key, None)
    plot_color_scheme = rplot.normalize_plot_color_scheme(
        st.session_state.get(plot_color_scheme_key)
    )
    st.session_state[plot_color_scheme_key] = plot_color_scheme
    intensity_numbers_key = "show_preview_intensity_numbers"
    if intensity_numbers_key not in st.session_state:
        st.session_state[intensity_numbers_key] = True
    show_preview_intensity_numbers = bool(st.session_state.get(intensity_numbers_key, True))

    # Quit-Button
    def _shutdown():
        def _kill():
            time.sleep(0.3)
            os._exit(0)
        threading.Thread(target=_kill, daemon=True).start()

    # Grid/scope selectors existed in earlier builds. Remove their retired
    # widget state so a long-lived browser session cannot preserve an invisible
    # 60–1900 cache choice or restricted reference subset.
    for retired_matching_key in (
        "match_ultra_draft",
        "matching_grid_profile_rendered",
        "matching_grid_changed_notice",
        "match_reference_scope_draft",
    ):
        st.session_state.pop(retired_matching_key, None)
    matching_control_schema_key = "matching_control_schema_version"
    if (
        st.session_state.get(matching_control_schema_key)
        != MATCHING_CONTROL_SCHEMA_VERSION
    ):
        st.session_state.pop("matching_range_draft", None)
        st.session_state[matching_control_schema_key] = MATCHING_CONTROL_SCHEMA_VERSION

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
                "λ (10^x)", min_value=0, max_value=9, value=5, step=1,
                help=(
                    "Baseline stiffness (10^x), defined after resampling valid support to a "
                    "canonical 1 cm⁻¹ grid. Lower values follow curved backgrounds better; "
                    "high values can become almost linear."
                ),
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
        st.caption(
            "Baseline parameters are evaluated on a canonical 1 cm⁻¹ grid over valid signal support. "
            "DB baseline for DB-BC cache/overlay is fixed: arPLS, λ=10^4, iter≤50, tol=1e-3, autoscaling on."
        )

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
    baseline_typed = rwf.BaselineConfig.from_mapping(
        baseline_cfg,
        have_scipy=bool(rprep.HAVE_SCIPY),
    )
    baseline_cfg = baseline_typed.to_mapping()
    db_baseline_cfg = _fixed_db_baseline_cfg()
    db_baseline_token = _baseline_cfg_token(db_baseline_cfg)
    meas_mode = "RAW" if method == "NONE" else "BC"

    smooth_defaults = _default_smoothing_cfg()
    with st.sidebar.expander("Denoising / smoothing before matching", expanded=True):
        smoothing_options = {
            "Savitzky–Golay": "savgol",
            "AI-assisted · guarded DeepeR (full range)": "deeper_ai",
            "None (keep measurement unchanged)": "none",
        }
        default_method = _smoothing_method(smooth_defaults)
        default_option = next(
            label for label, value in smoothing_options.items() if value == default_method
        )
        smoothing_option = st.selectbox(
            "Measurement denoising method",
            options=list(smoothing_options),
            index=list(smoothing_options).index(default_option),
            help=(
                "Only the uploaded measurement is processed. Database spectra remain "
                "unsmoothed for every option."
            ),
        )
        smoothing_method = smoothing_options[smoothing_option]
        # Keep method-specific values available for the configuration payload,
        # but render only the controls that apply to the selected method.
        smoothing_window = int(smooth_defaults["window"])
        smoothing_poly = int(smooth_defaults["poly"])
        ai_max_change_sigma = float(ai_denoiser.DEFAULT_MAX_CHANGE_SIGMA)
        processing_difference_magnification = (
            DEFAULT_PROCESSING_DIFFERENCE_MAGNIFICATION
        )
        if smoothing_method == "savgol":
            smoothing_window = st.slider(
                "Window length (points; odd)",
                min_value=3,
                max_value=101,
                value=smoothing_window,
                step=2,
            )
            smooth_poly_max = max(0, min(9, int(smoothing_window) - 1))
            smoothing_poly = st.slider(
                "Polynomial order",
                min_value=0,
                max_value=smooth_poly_max,
                value=min(smoothing_poly, smooth_poly_max),
                step=1,
            )
            st.caption(
                "The samples are 1 cm⁻¹ apart on the canonical grid; for example, an "
                "5-point window spans 4 cm⁻¹ between its endpoints."
            )
        elif smoothing_method == "deeper_ai":
            ai_max_change_sigma = st.slider(
                "AI safeguard: maximum correction (× estimated noise σ)",
                min_value=0.5,
                max_value=3.0,
                value=ai_max_change_sigma,
                step=0.25,
                help=(
                    "A mandatory peak-preservation limit. The AI-assisted result can never "
                    "move a point by more than this multiple of robustly estimated noise, "
                    "and is also limited to 2% of the complete signal range."
                ),
            )
            if not ai_denoiser.model_is_ready():
                st.info(
                    f"On first use, the {ai_denoiser.MODEL_SIZE_BYTES / (1024 ** 2):.1f} MiB "
                    "checkpoint is downloaded from the authors' GitHub repository and "
                    "verified by SHA-256."
                )
            st.warning(
                "Experimental guarded DeepeR: the biomedical-cell model is not validated "
                "for minerals. Full-range windowing discards learned background changes and "
                "accepts only conservative noise-scale corrections; raw neural output never "
                "enters matching or export."
            )
            if meas_mode == "RAW":
                st.warning(
                    "The pretrained AI model expects baseline-corrected input. Choose arPLS/IAsLS "
                    "for its intended input domain, or use Savitzky–Golay/None with RAW data."
                )
            st.caption(
                "Sources: [RamanSPy](https://ramanspy.readthedocs.io/en/latest/"
                "auto_examples/plot_ii_dl_denoising.html) · "
                f"[DeepeR model]({ai_denoiser.MODEL_REPOSITORY})"
            )
        else:
            st.caption("No denoising or smoothing is applied to the measurement.")
        if smoothing_method != "none":
            processing_difference_magnification = st.slider(
                "Processing-difference line magnification",
                min_value=1.0,
                max_value=10.0,
                value=DEFAULT_PROCESSING_DIFFERENCE_MAGNIFICATION,
                step=0.5,
                key="processing_difference_magnification",
                help=(
                    "Display only: magnifies the output − input trace in the "
                    "preview. It does not alter denoising, matching, or exports."
                ),
            )
        st.caption("Step 3 is confirmed explicitly before matching starts.")

    # Reserve the matching-control position beside the other preprocessing
    # controls.  The contents are populated only after the measurement has
    # passed Steps 1–3, when its valid matching limits are known.
    matching_controls_slot = st.sidebar.container(key="matching_controls_slot")

    smoothing_cfg = {
        "method": smoothing_method,
        "enabled": smoothing_method != "none",
        "window": int(smoothing_window),
        "poly": int(smoothing_poly),
        "max_change_sigma": float(ai_max_change_sigma),
    }
    smoothing_typed = rwf.SmoothingConfig.from_mapping(smoothing_cfg)
    smoothing_cfg = smoothing_typed.to_mapping()
    smoothing_token = _smoothing_cfg_token(smoothing_cfg)

    active_folders = MATCH_FOLDERS
    active_grid_cfg = DATABASE_GRID
    measurement_upload_generation_key = "measurement_upload_generation"
    if measurement_upload_generation_key not in st.session_state:
        st.session_state[measurement_upload_generation_key] = 0
    prospective_upload_key = f"measurement_file_{st.session_state[measurement_upload_generation_key]}"
    has_active_upload = st.session_state.get(prospective_upload_key) is not None
    active_inventory: rdb.DatabaseInventory | None = None
    active_entries_sig = "not-loaded"
    if has_active_upload:
        try:
            active_inventory = _inventory_snapshot(list(active_folders))
            active_entries_sig = active_inventory.signature
        except Exception as exc:
            st.sidebar.warning(f"Could not inspect active databases: {exc}")

    with st.sidebar.expander("Input calibration", expanded=True):
        axis_unit_label = st.selectbox(
            "Spectral-axis unit",
            ("Raman shift (cm⁻¹)", "Unknown / not confirmed"),
            key="measurement_axis_unit_label",
            help="Database matching is disabled until the x axis is explicitly confirmed as Raman shift in cm⁻¹.",
        )
        measurement_axis_unit = (
            "cm^-1" if axis_unit_label.startswith("Raman shift") else "unknown"
        )
        meas_shift_cm1 = st.slider(
            "Calibrated Raman-shift offset (cm⁻¹)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            key="measurement_shift_cm1",
            help=(
                "Applies a linear axis calibration after intensity preprocessing and before "
                "projection onto the matching grid. Positive values move peaks higher."
            ),
        )
        st.caption(f"Current offset: {float(meas_shift_cm1):+.1f} cm⁻¹")
        measurement_calibrant = st.text_input(
            "Calibrant / reference peak (optional)",
            value="",
            placeholder="e.g. silicon 520.5 cm⁻¹",
            key="measurement_calibrant",
        )
        measurement_calibration_residual = st.number_input(
            "Calibration residual (cm⁻¹; 0 = not recorded)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            key="measurement_calibration_residual",
        )
        measurement_excitation_nm = st.number_input(
            "Excitation wavelength (nm; 0 = unknown)",
            min_value=0.0,
            max_value=2000.0,
            value=0.0,
            step=1.0,
            key="measurement_excitation_nm",
        )
        measurement_resolution_cm1 = st.number_input(
            "Spectral resolution (cm⁻¹; 0 = unknown)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key="measurement_resolution_cm1",
        )
        measurement_instrument = st.text_input(
            "Instrument / acquisition ID (optional)",
            value="",
            key="measurement_instrument",
        )

    with st.sidebar.expander("Current cached DB overview", expanded=False):
        st.caption(f"Active folders: {', '.join(Path(p).name for p in active_folders)}")
        if active_inventory is None:
            st.caption("Upload a measurement to inspect the active database inventory.")
        else:
            st.markdown(f"- Reference files: **{len(active_inventory.files):,}**")
            catalog_state_key = "db_catalog_loaded_signature"
            if st.button(
                "Load searchable phase-name catalog",
                key="load_db_catalog_btn",
                help=(
                    "Reads reference headers only when requested; matching uses the typed "
                    "precompute cache and does not need this optional catalog."
                ),
            ):
                st.session_state[catalog_state_key] = active_entries_sig

            if st.session_state.get(catalog_state_key) == active_entries_sig:
                try:
                    catalog_summary = _get_db_entries_cached(
                        tuple(str(path) for path in active_folders),
                        active_entries_sig,
                    )
                except Exception as exc:
                    st.warning(f"Could not load the phase-name catalog: {exc}")
                    catalog_summary = rdb.ReferenceCatalogSummary((), 0, 0)
                unique_names = catalog_summary.unique_names
                st.markdown(f"- Different mineral/phase names: **{len(unique_names):,}**")
                if catalog_summary.skipped_count:
                    st.caption(
                        "Skipped files while loading headers: "
                        f"{catalog_summary.skipped_count:,}"
                    )
                sidebar_name_query = st.text_input(
                    "Search mineral/phase in current cache",
                    value="",
                    placeholder="e.g. quartz",
                    key="sidebar_db_name_query",
                ).strip()
                if sidebar_name_query:
                    query_text = sidebar_name_query.casefold()
                    hits = [
                        name for name in unique_names if query_text in name.casefold()
                    ]
                    exact = next(
                        (name for name in hits if name.casefold() == query_text),
                        None,
                    )
                    if exact is not None:
                        st.success(f"Exact name found: {exact}")
                    if hits:
                        shown = 30
                        st.caption(f"{len(hits)} matching names:")
                        st.write(", ".join(hits[:shown]))
                        if len(hits) > shown:
                            st.caption(f"... and {len(hits) - shown} more")
                    else:
                        st.info("No matching mineral/phase found in the current cache.")
            else:
                st.caption("Phase-name headers are not loaded, keeping upload reruns fast.")

    st.sidebar.divider()
    if st.sidebar.button("Reload DB"):
        rc.load_reference_folders.cache_clear()
        _get_db_entries_cached.clear()
        _provenance_payload_cached.clear()
        _load_precompute_pair_resource.clear()
        _inventory_snapshot(list(MATCH_FOLDERS), refresh=True)
        current_workflow = st.session_state.get("workflow_state")
        if isinstance(current_workflow, rwf.WorkflowState):
            st.session_state["workflow_state"] = current_workflow.invalidate_from(
                "matching"
            )
        st.session_state.pop("db_catalog_loaded_signature", None)
        st.sidebar.success(
            "Database inventory refreshed; valid precompute files were preserved."
        )

    with st.sidebar.expander("Appearance", expanded=False):
        next_app_theme = "light" if app_theme == "dark" else "dark"
        st.caption(f"App theme: {app_theme}")
        if st.button(
            f"Switch app to {next_app_theme} mode",
            key="toggle_app_theme_btn",
            width="stretch",
        ):
            st.session_state[app_theme_key] = next_app_theme
            st.session_state[plot_theme_key] = next_app_theme
            _safe_rerun()

        next_plot_theme = "light" if plot_theme == "dark" else "dark"
        st.caption(f"Plot theme: {plot_theme}")
        if st.button(
            f"Switch plots to {next_plot_theme} theme",
            key="toggle_plot_theme_btn",
            width="stretch",
        ):
            st.session_state[plot_theme_key] = next_plot_theme
            plot_theme = next_plot_theme

        plot_color_scheme = rplot.normalize_plot_color_scheme(
            st.selectbox(
                "Plot line colors",
                options=("standard", "colorblind", "grayscale"),
                format_func={
                    "standard": "Current colors",
                    "colorblind": "Colorblind-friendly",
                    "grayscale": "Grayscale",
                }.get,
                key=plot_color_scheme_key,
                help=(
                    "Changes line colors in every plot and downloaded figure; "
                    "processing and matching are unchanged."
                ),
            )
        )

        intensity_status = "shown" if show_preview_intensity_numbers else "hidden"
        intensity_target = "hide" if show_preview_intensity_numbers else "show"
        st.caption(f"Intensity numbers: {intensity_status}")
        st.button(
            f"{intensity_target.capitalize()} intensity numbers",
            key="toggle_preview_intensity_numbers_btn",
            width="stretch",
            on_click=_toggle_bool_session_state,
            args=(intensity_numbers_key, True),
        )

    st.sidebar.button("❌ Quit application", on_click=_shutdown, width="stretch")

    # Measurement upload.  The uploader itself is the source of truth: an empty
    # visible uploader must never leave a hidden old spectrum active.
    measurement_upload_generation_key = "measurement_upload_generation"
    if measurement_upload_generation_key not in st.session_state:
        st.session_state[measurement_upload_generation_key] = 0
    measurement_upload_key = f"measurement_file_{st.session_state[measurement_upload_generation_key]}"
    meas_file = st.file_uploader(
        "Measurement spectrum (.txt / .csv)",
        key=measurement_upload_key,
        help=(
            "The first two columns are read explicitly as Raman shift and intensity. "
            "Comma, tab, semicolon, and whitespace delimiters are supported."
        ),
    )
    if meas_file is None:
        current_workflow = st.session_state.get("workflow_state")
        if (
            isinstance(current_workflow, rwf.WorkflowState)
            and current_workflow.measurement is not None
        ):
            st.session_state["workflow_state"] = current_workflow.set_measurement(None)
            st.session_state.pop(PRIMARY_RESULT_SNAPSHOT_KEY, None)
            st.session_state.pop("approved_measurement_artifact", None)
            st.session_state.pop("matching_range_draft", None)
            _clear_residual_session_state()
        st.info("Upload a measurement spectrum to begin input QC and calibration.")
        st.stop()
    meas_bytes = meas_file.getvalue()
    meas_file_name = str(getattr(meas_file, "name", "") or "measurement.txt")
    measurement_sha256 = rex.sha256_bytes(meas_bytes)
    try:
        measurement_spectrum = _cached_measurement_spectrum(
            measurement_sha256,
            meas_bytes,
        )
    except ValueError as exc:
        st.error(f"Could not parse measurement spectrum: {exc}")
        st.stop()
    meas_x_full = measurement_spectrum.axis_cm1
    meas_y_raw_full = measurement_spectrum.intensity

    # --- NEU: Guard gegen leere Arrays/NaNs, damit _normalize nicht crasht ---
    if meas_x_full.size == 0 or meas_y_raw_full.size == 0:
        st.error("The measurement file contains no usable data points (x or y empty). Please check the file.")
        st.stop()

    input_quality = measurement_spectrum.quality
    spacing_text = (
        f"{input_quality.median_spacing_cm1:.3g} cm⁻¹"
        if input_quality.median_spacing_cm1 is not None
        else "not determined"
    )
    st.caption(
        f"{input_quality.finite_point_count:,} finite points · "
        f"{input_quality.minimum_cm1:.2f}–{input_quality.maximum_cm1:.2f} cm⁻¹ · "
        f"median spacing {spacing_text}"
    )
    for quality_warning in input_quality.warnings:
        st.warning(quality_warning)

    white_ref_file = None
    if white_ref_enabled:
        white_upload_generation_key = "white_reference_upload_generation"
        if white_upload_generation_key not in st.session_state:
            st.session_state[white_upload_generation_key] = 0
        white_ref_file = st.file_uploader(
            "White-light reference spectrum (.txt / .csv)",
            key=f"white_ref_file_{st.session_state[white_upload_generation_key]}",
            help=(
                "The first two columns are read explicitly as Raman shift and intensity; "
                "the same validation as for the measurement spectrum is applied."
            ),
        )

    white_ref_bytes = b""
    white_ref_sha256 = ""
    white_name = ""
    if white_ref_enabled and white_ref_file is not None:
        white_ref_bytes = white_ref_file.getvalue()
        white_ref_sha256 = rex.sha256_bytes(white_ref_bytes)
        white_name = str(getattr(white_ref_file, "name", "") or "white_reference.txt")
        white_info_col, white_clear_col = st.columns([4, 1])
        with white_info_col:
            st.caption(
                f"Active white-light reference: **{white_name}** · SHA-256 "
                f"`{white_ref_sha256[:12]}…`"
            )
        with white_clear_col:
            if st.button("Clear reference", key="clear_white_reference_btn", width="stretch"):
                st.session_state[white_upload_generation_key] += 1
                current_workflow = st.session_state.get("workflow_state")
                if isinstance(current_workflow, rwf.WorkflowState):
                    st.session_state["workflow_state"] = current_workflow.invalidate_from("input")
                _safe_rerun()

    white_ref_x = np.array([], dtype=float)
    white_ref_y = np.array([], dtype=float)
    white_ref_error = ""
    if white_ref_enabled and white_ref_bytes:
        try:
            white_spectrum = _cached_measurement_spectrum(
                white_ref_sha256,
                white_ref_bytes,
            )
            white_ref_x = white_spectrum.axis_cm1
            white_ref_y = white_spectrum.intensity
            finite_ref = np.isfinite(white_ref_x) & np.isfinite(white_ref_y)
            unique_x = np.unique(np.asarray(white_ref_x, dtype=float)[finite_ref]).size
            if white_ref_x.size < 2 or white_ref_y.size < 2 or unique_x < 2:
                white_ref_error = "White-light reference must contain at least 2 valid points with distinct Raman shifts."
        except ValueError as exc:
            white_ref_error = f"Could not parse white-light reference: {exc}"

    white_ref_applied = bool(white_ref_enabled and white_ref_bytes and not white_ref_error)
    if white_ref_applied:
        white_alignment = rprep.align_reference_to_target(meas_x_full, white_ref_x, white_ref_y)
        white_ref_aligned_full = white_alignment.values
        if white_alignment.overlap_fraction < 0.95:
            st.warning(
                "White-light reference covers only "
                f"{100.0 * white_alignment.overlap_fraction:.1f}% of measurement points. "
                "Subtraction is applied only on common support; inspect both support edges."
            )
    else:
        white_ref_aligned_full = np.zeros_like(meas_y_raw_full, dtype=float)
    white_ref_scaled_full = float(white_ref_scale) * white_ref_aligned_full
    meas_y_full = meas_y_raw_full - white_ref_scaled_full if white_ref_applied else meas_y_raw_full.copy()
    meas_x_shifted_full = np.asarray(meas_x_full, dtype=float) + float(meas_shift_cm1)

    def _native_plot_trace(
        values: np.ndarray,
        point_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return native points with explicit separators at unmeasured gaps."""

        signal = np.asarray(values, dtype=float).reshape(-1)
        if signal.size != meas_x_full.size:
            raise ValueError("native plot trace must match the measurement axis")
        active = (
            np.ones(meas_x_full.size, dtype=bool)
            if point_mask is None
            else np.asarray(point_mask, dtype=bool).reshape(-1)
        )
        if active.size != meas_x_full.size:
            raise ValueError("native plot mask must match the measurement axis")
        native_axis = np.asarray(meas_x_full, dtype=float)[active]
        shifted_axis = np.asarray(meas_x_shifted_full, dtype=float)[active]
        return rplot.segmented_line_data(
            shifted_axis,
            signal[active],
            rprep.support_slices(native_axis),
        )

    white_ref_sha1 = hashlib.sha1(white_ref_bytes).hexdigest() if white_ref_bytes else ""
    white_ref_cfg = {
        "enabled": bool(white_ref_enabled),
        "scale": float(white_ref_scale),
        "ref_sha1": white_ref_sha1 if white_ref_enabled else "",
    }
    measurement_identity = rwf.UploadIdentity.from_bytes(meas_file_name, meas_bytes)
    white_identity = (
        rwf.UploadIdentity.from_bytes(white_name or "white_reference.txt", white_ref_bytes)
        if white_ref_enabled and white_ref_bytes
        else None
    )
    white_ref_typed = rwf.WhiteReferenceConfig(
        enabled=bool(white_ref_enabled),
        scale=float(white_ref_scale),
        reference=white_identity,
    )
    calibration_typed = rwf.CalibrationConfig(
        shift_cm1=float(meas_shift_cm1),
        axis_unit=measurement_axis_unit,
        calibrant=measurement_calibrant,
        residual_cm1=(
            float(measurement_calibration_residual)
            if float(measurement_calibration_residual) > 0.0
            else None
        ),
        excitation_wavelength_nm=(
            float(measurement_excitation_nm)
            if float(measurement_excitation_nm) > 0.0
            else None
        ),
        spectral_resolution_cm1=(
            float(measurement_resolution_cm1)
            if float(measurement_resolution_cm1) > 0.0
            else None
        ),
        instrument=measurement_instrument,
    )
    workflow_state = st.session_state.get("workflow_state")
    if not isinstance(workflow_state, rwf.WorkflowState):
        workflow_state = rwf.WorkflowState()
    previous_measurement_identity = workflow_state.measurement
    workflow_state = (
        workflow_state.set_measurement(measurement_identity)
        .with_white_reference(white_ref_typed)
        .with_calibration(calibration_typed)
        .with_baseline(baseline_typed)
        .with_smoothing(smoothing_typed)
    )
    if previous_measurement_identity != measurement_identity:
        st.session_state.pop(PRIMARY_RESULT_SNAPSHOT_KEY, None)
        st.session_state.pop("matching_range_draft", None)
        for state_key in (
            RESIDUAL_RESULT_SNAPSHOT_KEY,
            "top_combined_residual",
            "residual_mode_active",
            "residual_search_info",
            "residual_parent_identity",
        ):
            st.session_state.pop(state_key, None)
    st.session_state["workflow_state"] = workflow_state
    white_ref_token = _white_ref_cfg_token(white_ref_cfg)

    base_name = Path(meas_file_name).stem
    export_layout = _cached_spectrum_text_layout(measurement_sha256, meas_bytes)
    header_lines = list(export_layout.header_lines)
    delimiter_hint = export_layout.delimiter_hint
    export_x = meas_x_full.copy()
    export_y_raw = meas_y_raw_full.copy()
    export_header_exact_ok = export_layout.exact_body_available
    if export_header_exact_ok:
        export_x = export_layout.axis
        export_y_raw = export_layout.intensity

    if white_ref_applied:
        white_ref_aligned_export = _align_reference_to_target(export_x, white_ref_x, white_ref_y)
        export_y = export_y_raw - float(white_ref_scale) * white_ref_aligned_export
    else:
        white_ref_aligned_export = np.zeros_like(export_y_raw, dtype=float)
        export_y = export_y_raw.copy()

    st.subheader("Measurement spectrum (raw)")
    raw_plot_signature = _plot_render_signature(
        "measurement-input",
        {
            "measurement": measurement_sha256,
            "white_reference": white_ref_token,
            "axis_shift_cm1": float(meas_shift_cm1),
            "theme": plot_theme,
            "line_color_scheme": plot_color_scheme,
            "intensity_numbers": show_preview_intensity_numbers,
        },
    )

    def _render_measurement_input_figure():
        fig_white, ax_white = plt.subplots(figsize=(11, 4.6))
        plot_x, plot_y = _native_plot_trace(meas_y_raw_full)
        ax_white.plot(
            plot_x,
            plot_y,
            label="measurement (raw)",
            linewidth=rplot.PLOT_LINEWIDTH,
        )
        if white_ref_applied:
            ref_x, ref_y = _native_plot_trace(
                white_ref_scaled_full,
                white_alignment.overlap_mask,
            )
            ax_white.plot(
                ref_x,
                ref_y,
                label=f"white-light reference × {float(white_ref_scale):.3f}",
                linewidth=rplot.PLOT_LINEWIDTH,
            )
            corrected_x, corrected_y = _native_plot_trace(meas_y_full)
            ax_white.plot(
                corrected_x,
                corrected_y,
                label="measurement − white-light reference",
                linewidth=rplot.PLOT_LINEWIDTH,
            )
        ax_white.legend(loc="best")
        _apply_plot_style(
            fig_white,
            ax_white,
            theme=plot_theme,
            color_scheme=plot_color_scheme,
        )
        _set_intensity_number_visibility(ax_white, show_preview_intensity_numbers)
        return fig_white

    raw_figure = _cached_figure_render_bundle(
        raw_plot_signature,
        _render_measurement_input_figure,
    )
    st.image(raw_figure.png, width="stretch")
    if white_ref_applied:
        col_white_png, col_white_svg, col_white_txt, _ = st.columns([1, 1, 1, 1])
    else:
        col_white_png, col_white_svg, _ = st.columns([1, 1, 2])
    with col_white_png:
        _download_button(
            "⬇️ Measurement figure (PNG)",
            data=raw_figure.png,
            file_name=f"{base_name}_measurement_raw.png",
            mime="image/png",
            width="stretch",
        )
    with col_white_svg:
        _download_button(
            "⬇️ Measurement figure (SVG)",
            data=raw_figure.svg,
            file_name=f"{base_name}_measurement_raw.svg",
            mime="image/svg+xml",
            width="stretch",
        )
    if white_ref_applied:
        with col_white_txt:
            _download_button(
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

    if calibration_typed.axis_unit != "cm^-1":
        st.error(
            "Confirm that the spectral x axis is Raman shift in cm⁻¹ before applying "
            "input settings or starting library matching."
        )
    can_confirm_white_ref = (
        ((not white_ref_enabled) or white_ref_applied)
        and calibration_typed.axis_unit == "cm^-1"
    )

    if workflow_state.input_dirty:
        st.info("Step 1/4: Adjust white-light subtraction and confirm to unlock baseline correction.")
        if st.button(
            "Apply white-light subtraction and/or continue to baseline",
            type="primary",
            width="stretch",
            key="approve_white_ref_btn",
            disabled=not can_confirm_white_ref,
        ):
            workflow_state = workflow_state.apply_input()
            st.session_state["workflow_state"] = workflow_state
            _delay_then_rerun()
        _render_stale_result_summary()
        st.stop()
    else:
        st.success("Step 1/4 complete: white-light subtraction settings applied.")
        if st.button(
            "Adjust white-light subtraction settings again",
            width="stretch",
            key="edit_white_ref_again_btn",
        ):
            workflow_state = workflow_state.invalidate_from("input")
            st.session_state["workflow_state"] = workflow_state
            _safe_rerun()

    # Baseline-Preview + Export (wie baseline_app_01c.py)
    baseline_preview_artifact = _cached_processed_spectrum(
        measurement_sha256,
        np.asarray(meas_x_full, dtype=float),
        np.asarray(meas_y_full, dtype=float),
        meas_mode == "BC",
        json.dumps(_baseline_cfg_payload(baseline_cfg), sort_keys=True),
        json.dumps(_smoothing_cfg_payload({"method": "none"}), sort_keys=True),
    )
    baseline_full, _baseline_valid = baseline_preview_artifact.project(
        meas_x_full,
        field_name="baseline",
    )
    meas_corr_full, _corrected_valid = baseline_preview_artifact.project(meas_x_full)
    baseline_plot_full = rplot.mask_unsupported_line_values(
        baseline_full,
        _baseline_valid,
    )
    corrected_plot_full = rplot.mask_unsupported_line_values(
        meas_corr_full,
        _corrected_valid,
    )
    baseline_export, _baseline_export_valid = baseline_preview_artifact.project(
        export_x,
        field_name="baseline",
    )
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
        "Baseline display viewport (cm⁻¹)",
        min_value=x_min_f,
        max_value=x_max_f,
        value=(x_min_f, x_max_f),
        step=step_guess,
        key=range_key,
    )
    workflow_state = workflow_state.with_display_viewport(rng_preview)
    st.session_state["workflow_state"] = workflow_state
    mask_prev = (meas_x_shifted_full >= rng_preview[0]) & (meas_x_shifted_full <= rng_preview[1])
    baseline_input_label = "measurement − white-light reference" if white_ref_applied else "raw measurement"
    baseline_preview_colors = rplot.baseline_preview_colors(
        plot_color_scheme,
        plot_theme,
    )
    baseline_plot_signature = _plot_render_signature(
        "baseline-preview",
        {
            "measurement": measurement_sha256,
            "white_reference": white_ref_token,
            "baseline": _baseline_cfg_token(baseline_cfg),
            "measurement_mode": meas_mode,
            "axis_shift_cm1": float(meas_shift_cm1),
            "viewport": [float(rng_preview[0]), float(rng_preview[1])],
            "show_raw": show_raw,
            "show_baseline": show_baseline,
            "show_corrected": show_corrected,
            "theme": plot_theme,
            "line_color_scheme": plot_color_scheme,
            "intensity_numbers": show_preview_intensity_numbers,
        },
    )

    def _render_baseline_preview_figure():
        fig_prev, ax_prev = plt.subplots(figsize=(11, 4.6))
        if show_raw:
            plot_x, plot_y = _native_plot_trace(meas_y_full, mask_prev)
            ax_prev.plot(
                plot_x,
                plot_y,
                label=baseline_input_label,
                color=baseline_preview_colors.input_signal,
                linestyle="-",
                linewidth=rplot.PLOT_LINEWIDTH,
            )
        if show_baseline:
            plot_x, plot_y = _native_plot_trace(baseline_plot_full, mask_prev)
            ax_prev.plot(
                plot_x,
                plot_y,
                label=f"baseline · {_baseline_label(baseline_cfg)}",
                color=baseline_preview_colors.fitted_baseline,
                linestyle=rplot.BASELINE_DOTTED_LINESTYLE,
                linewidth=rplot.PLOT_LINEWIDTH,
            )
        if show_corrected:
            plot_x, plot_y = _native_plot_trace(corrected_plot_full, mask_prev)
            ax_prev.plot(
                plot_x,
                plot_y,
                label="corrected = input − baseline",
                color=baseline_preview_colors.corrected_signal,
                linestyle="-",
                linewidth=rplot.PLOT_LINEWIDTH,
            )
        ax_prev.legend(loc="best")
        _apply_plot_style(
            fig_prev,
            ax_prev,
            theme=plot_theme,
            color_scheme=plot_color_scheme,
            preserve_line_appearance=True,
        )
        _set_intensity_number_visibility(ax_prev, show_preview_intensity_numbers)
        return fig_prev

    baseline_figure = _cached_figure_render_bundle(
        baseline_plot_signature,
        _render_baseline_preview_figure,
    )
    st.image(baseline_figure.png, width="stretch")
    base_name = Path(meas_file_name).stem
    col_prev_png, col_prev_svg, _ = st.columns([1, 1, 2])
    with col_prev_png:
        _download_button(
            "⬇️ Baseline preview (PNG)",
            data=baseline_figure.png,
            file_name=f"{base_name}_baseline_preview.png",
            mime="image/png",
            width="stretch",
        )
    with col_prev_svg:
        _download_button(
            "⬇️ Baseline preview (SVG)",
            data=baseline_figure.svg,
            file_name=f"{base_name}_baseline_preview.svg",
            mime="image/svg+xml",
            width="stretch",
        )

    baseline_negative_fraction = float(
        baseline_preview_artifact.diagnostics.get("negative_fraction", 0.0)
    )
    baseline_material_negative_fraction = float(
        baseline_preview_artifact.diagnostics.get(
            "material_negative_fraction",
            baseline_negative_fraction,
        )
    )
    baseline_negative_threshold = float(
        baseline_preview_artifact.diagnostics.get(
            "material_negative_threshold",
            0.0,
        )
    )
    baseline_change_fraction = float(
        baseline_preview_artifact.diagnostics.get("change_rms_fraction", 0.0)
    )
    st.caption(
        f"Baseline diagnostics: {100.0 * baseline_negative_fraction:.1f}% below zero · "
        f"{100.0 * baseline_material_negative_fraction:.1f}% materially negative "
        f"(below −{baseline_negative_threshold:.3g} a.u.) · "
        f"RMS change {100.0 * baseline_change_fraction:.1f}% of the input dynamic range."
    )
    if meas_mode == "BC" and baseline_material_negative_fraction > 0.10:
        st.warning(
            "More than 10% of baseline-corrected points are negative beyond the "
            "noise-aware tolerance. Inspect possible overcorrection and weak-peak "
            "loss before approval."
        )
    if keep_header and not export_header_exact_ok:
        st.info("Exact header export could not be determined; exporting without unchanged original header.")
    export_note = note_text if ((not keep_header) and add_note and note_text.strip()) else None
    keep_header_effective = bool(keep_header and export_header_exact_ok)
    header_lines_export = header_lines if export_header_exact_ok else []
    col_exp_a, col_exp_b, _ = st.columns([1, 1, 2])
    with col_exp_a:
        _download_button(
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
        _download_button(
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

    if workflow_state.baseline_dirty:
        st.info(
            "Step 2/4: The preview above has been recalculated from the current "
            "baseline draft. Confirm it to invalidate and replace the previously "
            "approved denoising/matching result."
        )
        if st.button(
            "Apply baseline settings and continue to denoising / smoothing",
            type="primary",
            width="stretch",
            key="approve_baseline_btn",
        ):
            workflow_state = workflow_state.apply_baseline()
            st.session_state["workflow_state"] = workflow_state
            _clear_residual_session_state()
            _delay_then_rerun()
        _render_stale_result_summary()
        st.stop()
    else:
        st.success("Step 2/4 complete: baseline settings applied.")
        if st.button(
            "Adjust baseline settings again",
            width="stretch",
            key="edit_baseline_again_btn",
        ):
            workflow_state = workflow_state.invalidate_from("baseline")
            st.session_state["workflow_state"] = workflow_state
            _safe_rerun()

    smooth_input_full = meas_corr_full if meas_mode == "BC" else meas_y_full
    smooth_input_label = rprep.smoothing_input_curve_label(
        meas_mode,
        white_reference_applied=white_ref_applied,
    )
    try:
        if smoothing_method == "deeper_ai":
            with st.spinner("Applying guarded AI-assisted denoising over the complete spectrum…"):
                approved_processed_spectrum = _cached_processed_spectrum(
                    measurement_sha256,
                    np.asarray(meas_x_full, dtype=float),
                    np.asarray(meas_y_full, dtype=float),
                    meas_mode == "BC",
                    json.dumps(_baseline_cfg_payload(baseline_cfg), sort_keys=True),
                    json.dumps(_smoothing_cfg_payload(smoothing_cfg), sort_keys=True),
                )
        else:
            approved_processed_spectrum = _cached_processed_spectrum(
                measurement_sha256,
                np.asarray(meas_x_full, dtype=float),
                np.asarray(meas_y_full, dtype=float),
                meas_mode == "BC",
                json.dumps(_baseline_cfg_payload(baseline_cfg), sort_keys=True),
                json.dumps(_smoothing_cfg_payload(smoothing_cfg), sort_keys=True),
            )
        smooth_output_full, _smooth_valid_full = approved_processed_spectrum.project(
            meas_x_full
        )
        # Reproject the single approved artifact onto the original export row
        # order.  This avoids a second baseline/smoothing pass and, crucially,
        # a second DeepeR inference.
        smooth_export_full, _smooth_export_valid = approved_processed_spectrum.project(
            export_x
        )
    except ai_denoiser.AIDenoiserError as exc:
        st.error(f"AI denoising could not be applied: {exc}")
        st.info("Choose Savitzky–Golay or None in the sidebar to continue without the AI model.")
        st.stop()
    savgol_unsmoothed_segment_count = (
        sum(
            _sanitize_savgol_params(
                segment.axis_cm1.size,
                int(smoothing_cfg.get("window", 5)),
                int(smoothing_cfg.get("poly", 3)),
            )
            is None
            for segment in approved_processed_spectrum.segments
        )
        if smoothing_method == "savgol"
        else 0
    )

    preview_ui = _smoothing_preview_ui(smoothing_cfg)
    st.subheader(preview_ui["title"])
    if smoothing_method == "deeper_ai":
        preview_sigma = ai_denoiser.estimate_noise_sigma(
            np.asarray(meas_x_full, dtype=float),
            np.asarray(smooth_input_full, dtype=float),
        )
        preview_max_change = float(
            np.max(
                np.abs(
                    np.asarray(smooth_output_full, dtype=float)
                    - np.asarray(smooth_input_full, dtype=float)
                )
            )
        )
        st.caption(
            f"Guarded full support: {float(np.min(meas_x_shifted_full)):g}–"
            f"{float(np.max(meas_x_shifted_full)):g} cm⁻¹ · robust noise σ ≈ "
            f"{preview_sigma:.3g} a.u. · largest accepted point correction "
            f"{preview_max_change:.3g} a.u. Raw neural predictions are discarded."
        )
    elif smoothing_method == "savgol":
        st.caption(
            f"Selected settings: {int(smoothing_cfg.get('window', 11))}-point window "
            f"at {PREPROCESS_GRID_STEP_CM1:g} cm⁻¹ spacing · polynomial order "
            f"{int(smoothing_cfg.get('poly', 3))}."
        )
    else:
        st.caption(
            "No denoising or smoothing is applied; matching and text export use the "
            "input trace shown."
        )
    processing_difference = np.asarray(smooth_output_full) - np.asarray(smooth_input_full)
    processing_difference_magnified = (
        float(processing_difference_magnification) * processing_difference
    )
    processing_difference_display, _processing_difference_display_offset = (
        rplot.offset_trace_below(
            processing_difference_magnified,
            (smooth_input_full, smooth_output_full),
            mask_prev,
        )
    )
    input_dynamic_range = max(float(np.ptp(smooth_input_full)), 1e-12)
    largest_change_fraction = float(
        np.max(np.abs(processing_difference)) / input_dynamic_range
    )
    if largest_change_fraction > 0.10:
        st.warning(
            "The selected denoising settings change at least one point by more than "
            "10% of the signal range. Inspect the difference trace carefully."
        )
    smoothing_plot_signature = _plot_render_signature(
        "smoothing-preview",
        {
            "measurement": measurement_sha256,
            "white_reference": white_ref_token,
            "baseline": _baseline_cfg_token(baseline_cfg),
            "smoothing": smoothing_token,
            "measurement_mode": meas_mode,
            "axis_shift_cm1": float(meas_shift_cm1),
            "viewport": [float(rng_preview[0]), float(rng_preview[1])],
            "theme": plot_theme,
            "line_color_scheme": plot_color_scheme,
            "processing_difference_magnification": float(
                processing_difference_magnification
            ),
            "intensity_numbers": show_preview_intensity_numbers,
        },
    )

    def _render_smoothing_preview_figure():
        fig_smooth, ax_smooth = plt.subplots(figsize=(11, 4.6))
        plot_x, plot_y = _native_plot_trace(smooth_input_full, mask_prev)
        ax_smooth.plot(
            plot_x,
            plot_y,
            label=smooth_input_label,
            linewidth=rplot.PLOT_LINEWIDTH,
        )
        if smoothing_method != "none":
            plot_x, plot_y = _native_plot_trace(smooth_output_full, mask_prev)
            ax_smooth.plot(
                plot_x,
                plot_y,
                label=preview_ui["curve_label"],
                linewidth=rplot.PLOT_LINEWIDTH,
            )
            plot_x, plot_y = _native_plot_trace(
                processing_difference_display,
                mask_prev,
            )
            ax_smooth.plot(
                plot_x,
                plot_y,
                label=rprep.processing_difference_curve_label(
                    processing_difference_magnification
                ),
                linewidth=rplot.PLOT_LINEWIDTH,
                alpha=0.7,
                linestyle=":",
            )
        ax_smooth.legend(loc="best")
        _apply_plot_style(
            fig_smooth,
            ax_smooth,
            theme=plot_theme,
            color_scheme=plot_color_scheme,
        )
        _set_intensity_number_visibility(ax_smooth, show_preview_intensity_numbers)
        return fig_smooth

    smoothing_figure = _cached_figure_render_bundle(
        smoothing_plot_signature,
        _render_smoothing_preview_figure,
    )
    st.image(smoothing_figure.png, width="stretch")
    base_name = Path(meas_file_name).stem
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
        _download_button(
            f"⬇️ {preview_ui['preview_label']} (PNG)",
            data=smoothing_figure.png,
            file_name=f"{base_name}_{preview_ui['preview_file_tag']}.png",
            mime="image/png",
            width="stretch",
        )
    with col_smooth_svg:
        _download_button(
            f"⬇️ {preview_ui['preview_label']} (SVG)",
            data=smoothing_figure.svg,
            file_name=f"{base_name}_{preview_ui['preview_file_tag']}.svg",
            mime="image/svg+xml",
            width="stretch",
        )
    with col_smooth_txt:
        _download_button(
            f"⬇️ {preview_ui['spectrum_label']} (TXT)",
            data=smooth_txt,
            file_name=(
                f"{base_name}_{preview_ui['spectrum_file_tag']}_{smooth_suffix}.txt"
            ),
            mime="text/plain",
            width="stretch",
        )

    if savgol_unsmoothed_segment_count:
        st.warning(
            "Selected smoothing settings cannot be applied to "
            f"{savgol_unsmoothed_segment_count} short detector segment(s); those "
            "segments remain unsmoothed."
        )

    if workflow_state.smoothing_dirty:
        st.info("Step 3/4: Adjust denoising / smoothing and confirm to unlock database matching.")
        if st.button(
            "Apply denoising / smoothing settings and continue to database matching",
            type="primary",
            width="stretch",
            key="approve_smoothing_btn",
        ):
            workflow_state = workflow_state.apply_smoothing()
            st.session_state["workflow_state"] = workflow_state
            if workflow_state.smoothing_approval is not None:
                # Matching limits depend on the processed measurement and can
                # only be completed on the following rerun. Carry the exact
                # smoothing identity forward so only this approval transition
                # triggers the requested automatic first search.
                st.session_state["auto_match_smoothing_signature"] = (
                    workflow_state.smoothing_approval.signature
                )
            _delay_then_rerun()
        _render_stale_result_summary()
        st.stop()
    else:
        st.success("Step 3/4 complete: denoising / smoothing settings applied.")
        if st.button(
            "Adjust denoising / smoothing settings again",
            width="stretch",
            key="edit_smoothing_again_btn",
        ):
            workflow_state = workflow_state.invalidate_from("smoothing")
            st.session_state["workflow_state"] = workflow_state
            _safe_rerun()

    # Step 4: batch every setting that changes scientific matching.  The last
    # applied result remains available while a new draft is being edited.
    grid_low = max(
        int(math.ceil(float(np.min(meas_x_shifted_full)))),
        int(active_grid_cfg["min"]),
    )
    grid_high = min(
        int(math.floor(float(np.max(meas_x_shifted_full)))),
        int(active_grid_cfg["max"]),
    )
    if grid_high <= grid_low:
        st.error(
            "The calibrated measurement has no usable overlap with the active "
            f"database grid ({active_grid_cfg['min']}–{active_grid_cfg['max']} cm⁻¹)."
        )
        st.stop()

    st.subheader("Database matching")
    matching_controls_panel = matching_controls_slot.expander(
        "Matching parameters and controls",
        expanded=True,
    )
    existing_range_draft = st.session_state.get("matching_range_draft")
    if isinstance(existing_range_draft, (tuple, list)) and len(existing_range_draft) == 2:
        clipped_low = int(np.clip(existing_range_draft[0], grid_low, grid_high))
        clipped_high = int(np.clip(existing_range_draft[1], grid_low, grid_high))
        if clipped_high < clipped_low:
            clipped_low, clipped_high = grid_low, grid_high
        st.session_state["matching_range_draft"] = (clipped_low, clipped_high)
    with matching_controls_panel:
        matching_range_draft = st.slider(
            "Applied matching range (cm⁻¹)",
            min_value=grid_low,
            max_value=grid_high,
            value=_initial_matching_range(grid_low, grid_high),
            step=max(1, int(active_grid_cfg["step"])),
            key="matching_range_draft",
            help=(
                "The database cache always spans 60–4000 cm⁻¹. New measurements "
                "start with matching limited to 2000 cm⁻¹ (or their measured upper "
                "limit); move the upper handle toward 4000 cm⁻¹ to include the "
                "high-wavenumber region in scoring."
            ),
        )
        include_draft = st.text_input(
            "Include elements (comma-separated, optional)",
            key="match_include_draft",
        )
        filter_mode_draft = st.radio(
            "Element constraint",
            ["Must include all", "Only from this list", "Exactly this set"],
            key="match_filter_mode_draft",
        )
        exclude_draft = st.text_input(
            "Exclude elements (optional)",
            key="match_exclude_draft",
        )
        allow_no_formula_draft = st.checkbox(
            "Include entries without a formula",
            value=True,
            key="match_allow_no_formula_draft",
            help="Disable when chemistry constraints should exclude unannotated references.",
        )

    draft_matching_settings = {
        "range": (int(matching_range_draft[0]), int(matching_range_draft[1])),
        "include": str(include_draft).strip(),
        "exclude": str(exclude_draft).strip(),
        "mode": str(filter_mode_draft),
        "allow_no_formula": bool(allow_no_formula_draft),
    }
    draft_sig_base = _compute_signature_with_grid(
        list(active_folders),
        int(active_grid_cfg["min"]),
        int(active_grid_cfg["max"]),
        int(active_grid_cfg["step"]),
    )
    draft_sig_raw = f"{draft_sig_base}-dbbc0"
    draft_sig_bcb = f"{draft_sig_base}-b{db_baseline_token}-dbbc1"
    matching_typed = rwf.MatchingConfig.from_mapping(
        {
            **draft_matching_settings,
            "allow": bool(allow_no_formula_draft),
            "folders": tuple(str(path) for path in active_folders),
            "sig_raw": draft_sig_raw,
            "sig_bcb": draft_sig_bcb,
            "top_n": DEFAULT_TOP_N,
            "grad_w": GRAD_WEIGHT,
            "peak_f1_weight": PCS_F1_WEIGHT,
            "peak_tol": PCS_PEAK_TOL,
            "match_selection_v": MATCH_SELECTION_VERSION,
            "policy_signature": MATCHING_POLICY_SIGNATURE,
        }
    )
    workflow_state = workflow_state.with_matching(matching_typed)
    st.session_state["workflow_state"] = workflow_state

    auto_match_smoothing_signature = st.session_state.pop(
        "auto_match_smoothing_signature",
        None,
    )
    active_smoothing_signature = (
        workflow_state.smoothing_approval.signature
        if workflow_state.smoothing_approval is not None
        else None
    )
    if (
        auto_match_smoothing_signature is not None
        and auto_match_smoothing_signature == active_smoothing_signature
    ):
        workflow_state = workflow_state.apply_matching()
        st.session_state["workflow_state"] = workflow_state

    matching_update_required = workflow_state.matching_dirty
    with matching_controls_panel:
        update_matching = st.button(
            "Update database matching",
            type="primary" if matching_update_required else "secondary",
            width="stretch",
            key="update_database_matching_btn",
        )
        st.caption(
            "Highlighted when selected parameters differ from the current match."
        )

    if update_matching:
        workflow_state = workflow_state.apply_matching()
        st.session_state["workflow_state"] = workflow_state
        _safe_rerun()

    if workflow_state.matching_approval is None or workflow_state.matching_dirty:
        st.info(
            "Step 4/4: Matching parameters or database contents changed. Press the "
            "highlighted Update database matching button in the left sidebar."
        )
        _render_stale_result_summary()
        st.stop()

    applied_matching = workflow_state.matching_approval.config
    if applied_matching.range_cm1 is None:
        st.error("The applied matching approval does not contain a spectral range.")
        st.stop()
    range_low = int(applied_matching.range_cm1.low)
    range_high = int(applied_matching.range_cm1.high)

    mask = (meas_x_shifted_full >= range_low) & (meas_x_shifted_full <= range_high)
    if not np.any(mask):
        st.error("Selected range contains no points. Please adjust the range.")
        st.stop()
    selected_point_count = int(np.count_nonzero(mask))
    if selected_point_count < 30:
        st.error(
            f"Only {selected_point_count} measured points fall in the applied range; "
            "at least 30 are required for a defensible candidate search."
        )
        st.stop()

    # Preprocess the complete approved measurement, exactly as in the previews.
    # The range is a scoring mask only; it must not change the fitted baseline or
    # create range-dependent Savitzky-Golay edge behaviour.
    meas_x = meas_x_shifted_full
    meas_y = meas_y_full

    # Referenz-Ordner/Grids
    folders = tuple(Path(path) for path in applied_matching.database_folders)
    grid_cfg = DATABASE_GRID
    sig_base = applied_matching.raw_database_signature.removesuffix("-dbbc0")
    # Zwei Caches: DB-RAW und DB-BC (gleiche Grid-Config, anderer Signature-Suffix)
    sig_raw = applied_matching.raw_database_signature
    sig_bcb = applied_matching.baseline_database_signature
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

    filter_active = bool(
        applied_matching.include_elements or applied_matching.exclude_elements
    )
    eligibility_common = {
        "raw_signature": sig_raw,
        "baseline_signature": sig_bcb,
        "include_elements": tuple(applied_matching.include_elements),
        "exclude_elements": tuple(applied_matching.exclude_elements),
        "element_mode": applied_matching.element_mode,
        "allow_missing_formula": bool(applied_matching.allow_missing_formula),
        "filtering_policy_version": REFERENCE_FILTER_POLICY_VERSION,
    }
    raw_eligibility = _cached_reference_eligibility(
        rdb.ReferenceEligibilityRequest(
            **eligibility_common,
            library_variant="raw",
        ),
        pack_raw["meta"],
    )
    baseline_eligibility = _cached_reference_eligibility(
        rdb.ReferenceEligibilityRequest(
            **eligibility_common,
            library_variant="baseline_corrected",
        ),
        pack_bcb["meta"],
    )
    allowed_ids_raw = np.asarray(raw_eligibility.row_ids, dtype=np.int32)
    allowed_ids_bcb = np.asarray(baseline_eligibility.row_ids, dtype=np.int32)

    # A baseline-corrected measurement must not earn a phase match from an
    # instrument-specific raw-source background. Keep already-processed source
    # traces as supplied and use the paired DB-BC row for every raw source.
    if meas_mode == "BC":
        primary_ids_raw, primary_ids_bcb = (
            _background_neutral_residual_reference_ids(
                allowed_ids_raw,
                allowed_ids_bcb,
                pack_raw["meta"],
                pack_bcb["meta"],
            )
        )
    else:
        primary_ids_raw = allowed_ids_raw
        primary_ids_bcb = allowed_ids_bcb

    residual_ids_raw, residual_ids_bcb = (
        _background_neutral_residual_reference_ids(
            allowed_ids_raw,
            allowed_ids_bcb,
            pack_raw["meta"],
            pack_bcb["meta"],
        )
    )

    approved_query_vector, approved_query_valid = rprep.project_processed_spectrum(
        approved_processed_spectrum,
        pack_raw["grid"],
        axis_shift_cm1=float(meas_shift_cm1),
        normalize=True,
    )
    approved_query_vector = np.asarray(approved_query_vector, dtype=np.float32)
    approved_query_vector[~np.isfinite(approved_query_vector)] = 0.0
    approved_query_mask = (
        (pack_raw["grid"] >= range_low)
        & (pack_raw["grid"] <= range_high)
        & np.asarray(approved_query_valid, dtype=bool)
    )
    approved_query_vector[~approved_query_mask] = 0.0

    if filter_active:
        st.write(
            f"🔎 **Filter active – searched references:** as-provided "
            f"{primary_ids_raw.size}/{len(pack_raw['meta'])} · baseline-corrected "
            f"{primary_ids_bcb.size}/{len(pack_bcb['meta'])}"
        )
    if primary_ids_raw.size == 0 and primary_ids_bcb.size == 0:
        st.error("No reference spectrum matches the element filter.")
        st.stop()

    residual_matches_key = "top_combined_residual"
    residual_mode_key = "residual_mode_active"
    residual_info_key = "residual_search_info"
    residual_parent_identity_key = "residual_parent_identity"
    overlay_reset_key = "overlay_idx_reset_pending"

    expected_result_identity = workflow_state.expected_result_identity
    if expected_result_identity is None:
        st.error("The current matching request is not fully approved.")
        st.stop()
    primary_snapshot = _primary_result_snapshot()
    if (
        primary_snapshot is None
        or primary_snapshot.identity != expected_result_identity
    ):
        with st.status("Screening and refining library candidates…", expanded=False):
            top_primary = _compute_matches_from_query_vector(
                approved_query_vector,
                approved_query_mask,
                range_low,
                range_high,
                pack_raw,
                pack_bcb,
                primary_ids_raw,
                primary_ids_bcb,
                meas_mode,
                top_n=DEFAULT_TOP_N,
            )
        primary_snapshot = rwf.PrimaryResultSnapshot.from_workflow(
            workflow_state,
            top_primary,
            approved_query_vector,
            approved_query_mask,
        )
        workflow_state = workflow_state.record_result(primary_snapshot)
        st.session_state["workflow_state"] = workflow_state
        st.session_state[PRIMARY_RESULT_SNAPSHOT_KEY] = primary_snapshot
        st.session_state["overlay_idx"] = 0
        st.session_state.pop(RESIDUAL_RESULT_SNAPSHOT_KEY, None)
        st.session_state.pop(residual_matches_key, None)
        st.session_state.pop(residual_mode_key, None)
        st.session_state.pop(residual_info_key, None)
        st.session_state.pop(residual_parent_identity_key, None)
    else:
        workflow_state = workflow_state.record_result(primary_snapshot)
        st.session_state["workflow_state"] = workflow_state

    # Render and export the exact completed query, not a mutable draft-derived array.
    approved_query_vector = np.asarray(primary_snapshot.query_vector, dtype=np.float32)
    approved_query_mask = np.asarray(primary_snapshot.query_mask, dtype=bool)
    top_primary = primary_snapshot.result_mappings()
    result_identity = primary_snapshot.identity

    residual_snapshot = _residual_result_snapshot()
    residual_snapshot_is_current = bool(
        residual_snapshot is not None
        and residual_snapshot.primary_identity == primary_snapshot.identity
        and residual_snapshot.identity.residual_policy_signature
        == RESIDUAL_SEARCH_POLICY_SIGNATURE
    )
    if not residual_snapshot_is_current:
        residual_snapshot = None
        st.session_state.pop(RESIDUAL_RESULT_SNAPSHOT_KEY, None)
        st.session_state.pop(residual_matches_key, None)
        st.session_state.pop(residual_mode_key, None)
        st.session_state.pop(residual_info_key, None)
        st.session_state.pop(residual_parent_identity_key, None)
    residual_matches = residual_snapshot.result_mappings() if residual_snapshot else []
    residual_mode_active = (
        bool(st.session_state.get(residual_mode_key, False))
        and residual_snapshot is not None
        and bool(residual_matches)
    )
    top_combined = residual_matches if residual_mode_active else top_primary
    residual_info = (
        residual_snapshot.diagnostics_mapping() if residual_mode_active else {}
    )
    result_query_vector = np.asarray(
        residual_snapshot.query_vector if residual_mode_active else approved_query_vector,
        dtype=float,
    )
    result_query_mask = np.asarray(
        residual_snapshot.query_mask if residual_mode_active else approved_query_mask,
        dtype=bool,
    )
    active_result_signature = (
        residual_snapshot.identity.signature
        if residual_mode_active
        else result_identity.signature
    )

    # Statuszeile
    db_names = ", ".join(Path(p).name for p in folders)
    if residual_mode_active:
        base_name = str(residual_info.get("base_name", "selected match"))
        base_file = str(residual_info.get("base_file", "")).strip()
        alpha = float(residual_info.get("alpha", 1.0))
        improvement = 100.0 * float(residual_info.get("fit_improvement_fraction", 0.0))
        negative_points = 100.0 * float(residual_info.get("negative_point_fraction", 0.0))
        evidence_gate_cleared = bool(
            residual_info.get("evidence_gate_cleared", False)
        )
        evidence_text = (
            "evidence guardrails cleared"
            if evidence_gate_cleared
            else "exploratory only; evidence guardrails not cleared"
        )
        src_txt = f" ({base_file})" if base_file else ""
        st.warning(
            f"Exploratory residual-candidate search active. Subtracted: {base_name}{src_txt}; "
            f"least-squares scale {alpha:.2f} (not abundance); fit SSE reduction {improvement:.1f}%; "
            f"negative residual at {negative_points:.1f}% of common points. "
            f"Status: {evidence_text}. Top {DEFAULT_TOP_N} residual candidates are "
            "hypotheses, not mixture quantification."
        )
    else:
        library_mode_text = (
            "background-neutral references matched to BC measurement"
            if meas_mode == "BC"
            else "library as provided + baseline-corrected raw sources"
        )
        st.success(
            f"Top {DEFAULT_TOP_N} candidate traces (measurement: {meas_mode} · "
            f"{library_mode_text} · grad {int(GRAD_WEIGHT*100)}%)  |  DBs: "
            f"{db_names}  |  White ref: {_white_ref_label(white_ref_cfg)} "
            f"[{white_ref_token}]  |  Baseline: {_baseline_label(baseline_cfg)}  |  "
            f"Denoising: {_smoothing_label(smoothing_cfg)} [{smoothing_token}]  |  "
            f"Calibration: {float(meas_shift_cm1):+.1f} cm⁻¹  |  Range: "
            f"{range_low}–{range_high} cm⁻¹"
        )

    if not top_combined:
        st.info("No matches found. Check range, filter, or input file.")
        st.stop()

    if residual_mode_active and residual_snapshot is not None:
        with st.expander("Audit signed residual used for exploratory rematching", expanded=True):
            residual_fig, residual_ax = plt.subplots(figsize=(11, 3.2))
            residual_ax.plot(
                np.asarray(pack_raw["grid"], dtype=float),
                np.where(
                    result_query_mask,
                    result_query_vector,
                    np.nan,
                ),
                lw=rplot.PLOT_LINEWIDTH,
                label="normalised signed residual (matching query)",
            )
            residual_ax.axhline(
                0.0,
                color="#f59e0b",
                lw=rplot.PLOT_LINEWIDTH,
                ls="--",
                label="zero",
            )
            residual_ax.set_xlim(range_low, range_high)
            residual_ax.legend(loc="best")
            _apply_plot_style(
                residual_fig,
                residual_ax,
                theme=plot_theme,
                color_scheme=plot_color_scheme,
            )
            _set_intensity_number_visibility(
                residual_ax,
                show_preview_intensity_numbers,
            )
            st.pyplot(residual_fig)
            plt.close(residual_fig)
            st.caption(
                "Short gaps at subtraction-support boundaries are deliberate. "
                "Those transition samples are excluded from this audit plot, the "
                "overlay, and rematching so truncated non-zero reference edges cannot "
                "be interpreted as residual peaks. The solid residual curve is the "
                "same normalised signed array shown in the match overlay and supplied "
                "to the matcher."
            )

    # ——— Overlay figure first (same footprint as baseline/smoothing previews) ———
    if "overlay_idx" not in st.session_state:
        st.session_state.overlay_idx = 0
    if st.session_state.pop(overlay_reset_key, False):
        st.session_state["overlay_idx"] = 0

    def _fmt_opt(i: int) -> str:
        d = top_combined[i]
        formula = _format_formula(d.get("formula", "") or "—")
        return (
            f"{i+1:02d} · {d['name']} · {formula} · "
            f"phase:{float(d.get('phase_score', 0.0)):.3f} · "
            f"trace:{float(d.get('rank_score', _final_rank_score(d))):.3f}"
        )

    if st.session_state.overlay_idx >= len(top_combined):
        st.session_state.overlay_idx = 0

    st.markdown("**Overlay selection**")
    st.selectbox(
        "Select a match to audit",
        options=list(range(len(top_combined))),
        index=min(st.session_state.overlay_idx, len(top_combined) - 1),
        format_func=_fmt_opt,
        key="overlay_idx",
    )

    # Render the exact cached vector and shifted support that earned the score.
    sel = top_combined[st.session_state.overlay_idx]
    selected_reference_excitation = sel.get("excitation_wavelength_nm")
    selected_reference_resolution = sel.get("resolution_cm1")
    st.caption(
        f"Selected reference provenance: {sel.get('database_source') or 'unknown database'} · "
        f"accession {sel.get('accession') or 'unknown'} · "
        f"quality {sel.get('quality') or 'unknown'} · "
        f"processing {sel.get('reference_processing') or 'unknown'} · "
        f"excitation {selected_reference_excitation or 'unknown'} nm · "
        f"resolution {selected_reference_resolution or 'unknown'} cm⁻¹."
    )
    if (
        calibration_typed.excitation_wavelength_nm is not None
        and selected_reference_excitation is not None
        and abs(
            calibration_typed.excitation_wavelength_nm
            - float(selected_reference_excitation)
        )
        >= 20.0
    ):
        st.warning(
            "Measurement and selected reference used substantially different excitation "
            "wavelengths. Raman shifts remain comparable, but resonance, fluorescence, "
            "and relative band intensities can differ."
        )
    svg_data, png_data, filename_svg, filename_png = b"", b"", "", ""
    db_overlay_txt_data, filename_db_overlay_txt = b"", ""
    try:
        use_baseline_pack = _result_uses_baseline_pack(sel)
        apply_baseline_db = bool(sel.get("db_baseline", use_baseline_pack))
        selected_pack = pack_bcb if use_baseline_pack else pack_raw
        selected_index = int(sel.get("db_idx", -1))
        if selected_index < 0 or selected_index >= len(selected_pack["meta"]):
            raise IndexError("selected reference row is not available in the active cache")
        # The faint source curve is always the true as-provided vector. The
        # aligned curve comes from the exact RAW/DB-BC matrix that earned the
        # score; these differ when database baseline correction was selected.
        library_as_provided = np.asarray(
            pack_raw["X"][selected_index, :],
            dtype=float,
        )
        library_scored = np.asarray(
            selected_pack["X"][selected_index, :],
            dtype=float,
        )
        fitted_shift = int(sel.get("shift", 0))
        library_aligned = rmatch.shift_candidate(library_scored, fitted_shift)
        common_mask = rmatch.aligned_support_mask(
            result_query_mask,
            int(sel.get("start_idx", 0)),
            int(sel.get("end_idx", -1)),
            fitted_shift,
            support_runs=sel.get("support_runs"),
        )
        provided_metadata = pack_raw["meta"][selected_index]
        provided_support_mask = result_query_mask & rmatch.reference_support_mask(
            result_query_mask.size,
            int(provided_metadata.get("start_idx", 0)),
            int(provided_metadata.get("end_idx", -1)),
            support_runs=provided_metadata.get("support_runs"),
        )
        overlay_plot_signature = _plot_render_signature(
            "selected-match-overlay",
            {
                "results": active_result_signature,
                "residual": residual_mode_active,
                "query_content_sha256": rwf.residual_query_content_sha256(
                    result_query_vector,
                    result_query_mask,
                ),
                "db_index": selected_index,
                "db_variant": str(sel.get("db_variant", "")),
                "db_baseline": apply_baseline_db,
                "shift": fitted_shift,
                "range": [range_low, range_high],
                "theme": plot_theme,
                "line_color_scheme": plot_color_scheme,
                "intensity_numbers": show_preview_intensity_numbers,
            },
        )

        def _render_selected_match_figure():
            overlay_data = rplot.AlignmentOverlay(
                axis_cm1=np.asarray(selected_pack["grid"], dtype=float),
                measurement=result_query_vector,
                library_as_provided=library_as_provided,
                library_aligned=library_aligned,
                valid_mask=common_mask,
                label=str(sel["name"]),
                shift_cm1=float(
                    sel.get(
                        "shift_cm1",
                        fitted_shift * selected_pack["grid_info"]["step"],
                    )
                ),
                library_as_provided_mask=provided_support_mask,
                measurement_mask=result_query_mask,
                score=float(sel.get("rank_score", _final_rank_score(sel))),
                coverage_fraction=float(sel.get("coverage_fraction", 0.0)),
                shift_at_boundary=bool(sel.get("shift_boundary_hit", False)),
                measurement_label=(
                    "normalised signed residual (matching query)"
                    if residual_mode_active
                    else "measurement"
                ),
                peak_consistency=float(sel.get("pcs", 0.0)),
                aligned_treatment=(
                    "baseline corrected" if apply_baseline_db else "as provided"
                ),
            )
            figure, axis = plt.subplots(figsize=(11, 4.6))
            rplot.plot_alignment_evidence(axis, overlay_data)
            axis.set_xlim(range_low, range_high)
            _apply_plot_style(
                figure,
                axis,
                theme=plot_theme,
                color_scheme=plot_color_scheme,
            )
            if not residual_mode_active:
                y_min, _y_max = axis.get_ylim()
                if np.isfinite(y_min):
                    axis.set_ylim(bottom=max(-0.25, float(y_min)))
            _set_intensity_number_visibility(
                axis,
                show_preview_intensity_numbers,
            )
            return figure

        selected_figure = _cached_figure_render_bundle(
            overlay_plot_signature,
            _render_selected_match_figure,
        )
        svg_data = selected_figure.svg
        png_data = selected_figure.png
        st.image(selected_figure.png, width="stretch")
        if residual_mode_active:
            st.caption(
                "Residual overlay key: the solid curve is the exact normalised signed "
                "residual shown in the audit above; the dashed curve is the exact "
                "shifted reference variant that earned the score. The faint dotted "
                "curve is the original as-provided source trace for provenance only."
            )

        meas_name    = Path(meas_file_name).stem
        mineral_name = sel["name"].replace(" ", "_").replace("/", "_")
        var_tag      = f"meas-{sel.get('meas_variant','?')}_db-{'BC' if sel.get('db_baseline', False) else 'RAW'}"
        filename_svg = f"{meas_name}_fit_{mineral_name}_{var_tag}_{range_low}-{range_high}cm-1.svg"
        filename_png = f"{meas_name}_fit_{mineral_name}_{var_tag}_{range_low}-{range_high}cm-1.png"

        # Export the exact aligned cache vector used in scoring.
        if int(np.count_nonzero(common_mask)) >= 2:
            x_clip = np.asarray(selected_pack["grid"], dtype=float)[common_mask]
            db_proc = library_aligned[common_mask]
            db_note = (
                f"Score-aligned library trace ({'baseline corrected' if apply_baseline_db else 'as provided'}), "
                f"shift={float(sel.get('shift_cm1', fitted_shift)):+g} cm-1, "
                f"source={sel.get('orig_filename', '')}"
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
            "Explore residual candidates after subtracting selected trace",
            key="run_residual_phase_search_btn",
            width="stretch",
        ):
            sel_for_residual = top_combined[int(st.session_state.overlay_idx)]
            residual_payload = _build_residual_query_vector(
                approved_query_vector,
                approved_query_mask,
                sel_for_residual,
                pack_raw,
                pack_bcb,
            )
            if residual_payload is None:
                st.warning("Residual search could not be built from the selected match.")
            elif (
                residual_payload.fit_improvement_fraction
                < RESIDUAL_SEARCH_POLICY.minimum_fit_improvement_fraction
            ):
                st.warning(
                    "Residual search was not run because the selected trace reduces "
                    "common-support squared error by less than "
                    f"{100.0 * RESIDUAL_SEARCH_POLICY.minimum_fit_improvement_fraction:g}%."
                )
            else:
                residual_q = np.asarray(residual_payload.matching_vector, dtype=np.float32)
                residual_query_mask = np.asarray(
                    residual_payload.residual_mask,
                    dtype=bool,
                )
                selected_mineral = _mineral_key(sel_for_residual.get("name", ""))
                residual_references_available = bool(
                    residual_ids_raw.size or residual_ids_bcb.size
                )
                if not residual_references_available:
                    residual_top = []
                    st.warning(
                        "No background-neutral reference traces are available in "
                        "the configured databases."
                    )
                else:
                    with st.spinner("Searching for additional phase on residual spectrum…"):
                        residual_top = _compute_matches_from_query_vector(
                            residual_q,
                            residual_query_mask,
                            range_low,
                            range_high,
                            pack_raw,
                            pack_bcb,
                            residual_ids_raw,
                            residual_ids_bcb,
                            meas_mode,
                            top_n=DEFAULT_TOP_N,
                            excluded_phase_keys=(selected_mineral,),
                            matching_parameters=RESIDUAL_MATCHING_PARAMETERS,
                        )
                for d in residual_top:
                    d["residual_phase"] = True
                residual_evidence_gate_cleared = (
                    _residual_candidates_are_actionable(residual_top)
                )
                if residual_top and not residual_evidence_gate_cleared:
                    leading_residual = residual_top[0]
                    leading_status = str(
                        leading_residual.get("evidence_status", "insufficient_evidence")
                    ).replace("_", " ")
                    st.warning(
                        "The exploratory residual ranking is shown, but no additional "
                        "phase cleared the conservative evidence guardrails. The "
                        "highest mathematical ranking is "
                        f"{leading_residual.get('name', 'unnamed')} "
                        f"(phase score {float(leading_residual.get('phase_score', 0.0)):.3f}, "
                        f"peak agreement {float(leading_residual.get('pcs', 0.0)):.3f}), "
                        f"with evidence state {leading_status}. It remains inspectable "
                        "as a hypothesis, not an identified second phase."
                    )
                if not residual_top:
                    if residual_references_available:
                        st.warning(
                            "No residual candidate met the minimum detected-peak "
                            "agreement needed for exploratory display."
                        )
                else:
                    residual_diagnostics = {
                        "base_name": sel_for_residual.get("name", ""),
                        "base_file": sel_for_residual.get("orig_filename", ""),
                        "alpha": float(residual_payload.scale_factor),
                        "fit_improvement_fraction": float(
                            residual_payload.fit_improvement_fraction
                        ),
                        "negative_point_fraction": float(
                            residual_payload.negative_point_fraction
                        ),
                        "negative_energy_fraction": float(
                            residual_payload.negative_energy_fraction
                        ),
                        "common_point_count": int(residual_payload.common_point_count),
                        "background_neutral_raw_reference_count": int(
                            residual_ids_raw.size
                        ),
                        "background_corrected_reference_count": int(
                            residual_ids_bcb.size
                        ),
                        "minimum_candidate_peak_consistency": float(
                            RESIDUAL_MATCHING_PARAMETERS.minimum_candidate_peak_consistency
                        ),
                        "evidence_gate_cleared": bool(
                            residual_evidence_gate_cleared
                        ),
                    }
                    residual_snapshot = rwf.ResidualResultSnapshot.from_primary(
                        primary_snapshot,
                        _residual_reference_identity(
                            sel_for_residual,
                            sig_raw,
                            sig_bcb,
                            float(pack_raw.get("grid_info", {}).get("step", 1.0)),
                        ),
                        float(residual_payload.scale_factor),
                        residual_top[:DEFAULT_TOP_N],
                        residual_q,
                        residual_query_mask,
                        np.asarray(residual_payload.signed_residual, dtype=float),
                        RESIDUAL_SEARCH_POLICY_SIGNATURE,
                        residual_diagnostics,
                    )
                    st.session_state[RESIDUAL_RESULT_SNAPSHOT_KEY] = residual_snapshot
                    st.session_state[residual_mode_key] = True
                    st.session_state[residual_parent_identity_key] = (
                        residual_snapshot.identity
                    )
                    st.session_state.pop(residual_matches_key, None)
                    st.session_state.pop(residual_info_key, None)
                    st.session_state[overlay_reset_key] = True
                    _safe_rerun()
    if svg_data and png_data and filename_svg and filename_png:
        col_ov_svg, col_ov_png, col_ov_txt, _ = st.columns([1, 1, 1, 1])
        with col_ov_svg:
            _download_button(
                label="📊 Export as SVG",
                data=svg_data,
                file_name=filename_svg,
                mime="image/svg+xml",
                width="stretch",
            )
        with col_ov_png:
            _download_button(
                label="📊 Export as PNG",
                data=png_data,
                file_name=filename_png,
                mime="image/png",
                width="stretch",
            )
        with col_ov_txt:
            if db_overlay_txt_data and filename_db_overlay_txt:
                _download_button(
                    label="📄 Overlay DB (TXT)",
                    data=db_overlay_txt_data,
                    file_name=filename_db_overlay_txt,
                    mime="text/plain",
                    width="stretch",
                )

    with st.expander("Details table", expanded=False):
        df_full = pd.DataFrame(
            {
                "ID":         list(range(1, len(top_combined) + 1)),  # passt zur Dropdown-Nummer
                "Mineral":    [d["name"]                              for d in top_combined],
                "Formula":    [_format_formula(d.get("formula",""))   for d in top_combined],
                "Flag":       [d.get("flag","")                       for d in top_combined],
                "Library treatment": [
                    "baseline corrected" if d.get("db_baseline", False) else "as provided"
                    for d in top_combined
                ],
                "Phase rank": [int(d.get("phase_rank", 0)) for d in top_combined],
                "Phase score": [float(d.get("phase_score", 0.0)) for d in top_combined],
                "Independent refs": [
                    int(d.get("phase_independent_reference_count", 0))
                    for d in top_combined
                ],
                "Reference variants": [
                    int(d.get("phase_reference_variant_count", 0))
                    for d in top_combined
                ],
                "Rank score": [float(d.get("rank_score", _final_rank_score(d))) for d in top_combined],
                "Similarity": [d["similarity"]                        for d in top_combined],
                "Shape":      [float(d.get("shape_similarity", 0.0))   for d in top_combined],
                "Gradient":   [float(d.get("gradient_similarity", 0.0)) for d in top_combined],
                "PCS":        [float(d.get("pcs", 0.0))               for d in top_combined],
                "Shift (cm⁻¹)": [float(d.get("shift_cm1", d.get("shift", 0))) for d in top_combined],
                "Coverage (%)": [100.0 * float(d.get("coverage_fraction", 0.0)) for d in top_combined],
                "Common points": [int(d.get("common_point_count", 0)) for d in top_combined],
                "Alignment warning": [
                    "shift limit" if d.get("shift_boundary_hit", False) else ""
                    for d in top_combined
                ],
                "Source":     [d.get("source", "")                     for d in top_combined],
                "Accession":  [d.get("accession", "")                  for d in top_combined],
                "Quality":    [d.get("quality", "")                    for d in top_combined],
                "Determination": [d.get("determination", "") for d in top_combined],
                "Reference processing": [
                    d.get("reference_processing", "") for d in top_combined
                ],
                "Excitation (nm)": [
                    d.get("excitation_wavelength_nm") for d in top_combined
                ],
                "File":       [d.get("orig_filename","")              for d in top_combined],
            }
        )
        st.dataframe(df_full, hide_index=True, width="stretch", height=420)

    source_names = (
        Path(__file__).name,
        "raman_workflow.py",
        "raman_preprocessing.py",
        "raman_database.py",
        "raman_matching.py",
        "raman_plotting.py",
        "raman_exports.py",
        "raman_core.py",
        "raman_ai_denoiser.py",
    )
    source_stamps: list[tuple[str, int, int]] = []
    for source_name in source_names:
        try:
            source_stat = (BASE_DIR / source_name).stat()
            source_stamps.append(
                (source_name, int(source_stat.st_size), int(source_stat.st_mtime_ns))
            )
        except OSError:
            source_stamps.append((source_name, -1, -1))
    app_commit, package_versions, source_hashes = _runtime_export_metadata(
        tuple(source_stamps)
    )
    result_input_approval = primary_snapshot.input_approval
    result_baseline_approval = primary_snapshot.baseline_approval
    result_smoothing_approval = primary_snapshot.smoothing_approval
    result_matching_approval = primary_snapshot.matching_approval
    manifest_settings = {
        "run_kind": "exploratory_signed_residual" if residual_mode_active else "primary",
        "result_identity": (
            residual_snapshot.identity.payload()
            if residual_mode_active and residual_snapshot is not None
            else result_identity.payload()
        ),
        "primary_result_identity": result_identity.payload(),
        "residual_result_identity": (
            residual_snapshot.identity.payload()
            if residual_mode_active and residual_snapshot is not None
            else None
        ),
        "active_result_signature": active_result_signature,
        "active_query_mask_sha256": rwf.residual_query_content_sha256(
            result_query_vector,
            result_query_mask,
        ),
        "workflow_approval_signatures": {
            "input": result_input_approval.signature,
            "baseline": result_baseline_approval.signature,
            "smoothing": result_smoothing_approval.signature,
            "matching": result_matching_approval.signature,
        },
        "white_reference": result_input_approval.white_reference.payload(),
        "calibration": result_input_approval.calibration.payload(),
        "baseline": result_baseline_approval.config.payload(),
        "smoothing": result_smoothing_approval.config.payload(),
        "matching": {
            **result_matching_approval.config.payload(),
            "reference_selection_policy": "all_configured_spectra",
            "raw_database_signature": sig_raw,
            "baseline_database_signature": sig_bcb,
            "final_similarity_weight": FINAL_SIM_WEIGHT,
        },
        "matching_policy_signature": MATCHING_POLICY_SIGNATURE,
        "reference_filter_policy_version": REFERENCE_FILTER_POLICY_VERSION,
        "matching_parameters": MATCHING_PARAMETERS.payload(),
        "evidence_policy_uncalibrated": EVIDENCE_DECISION_POLICY.payload(),
        "residual_search_policy_signature": RESIDUAL_SEARCH_POLICY_SIGNATURE,
        "residual_search_policy": {
            "projection": RESIDUAL_SEARCH_POLICY.payload(),
            "matching_parameters": RESIDUAL_MATCHING_PARAMETERS.payload(),
            "reference_variant_policy": RESIDUAL_REFERENCE_VARIANT_POLICY,
            "reference_selection_policy": "all_configured_spectra",
            "actionable_evidence_statuses": sorted(
                RESIDUAL_ACTIONABLE_EVIDENCE_STATUSES
            ),
        },
        "input_quality": {
            "point_count": input_quality.point_count,
            "finite_point_count": input_quality.finite_point_count,
            "minimum_cm1": input_quality.minimum_cm1,
            "maximum_cm1": input_quality.maximum_cm1,
            "median_spacing_cm1": input_quality.median_spacing_cm1,
            "spacing_cv": input_quality.spacing_cv,
            "duplicate_count": input_quality.duplicate_count,
            "gap_intervals_cm1": input_quality.gap_intervals_cm1,
            "saturation_fraction": input_quality.saturation_fraction,
            "spike_indices": input_quality.spike_indices,
            "warnings": input_quality.warnings,
        },
        "database_inventory": {
            "base_signature": sig_base,
            "eligible_library_as_provided": int(allowed_ids_raw.size),
            "eligible_additional_baseline_corrected": int(allowed_ids_bcb.size),
            "primary_searched_as_provided": int(primary_ids_raw.size),
            "primary_searched_baseline_corrected": int(primary_ids_bcb.size),
            "residual_searched_as_provided": int(residual_ids_raw.size),
            "residual_searched_baseline_corrected": int(residual_ids_bcb.size),
        },
        "residual_diagnostics": (
            {
                key: residual_info.get(key)
                for key in (
                    "base_name",
                    "base_file",
                    "alpha",
                    "fit_improvement_fraction",
                    "negative_point_fraction",
                    "negative_energy_fraction",
                    "common_point_count",
                    "background_neutral_raw_reference_count",
                    "background_corrected_reference_count",
                    "minimum_candidate_peak_consistency",
                    "evidence_gate_cleared",
                )
            }
            if residual_mode_active
            else None
        ),
        "source_sha256": source_hashes,
    }
    run_manifest = rex.RunManifest(
        app_version=APP_VERSION,
        app_commit=app_commit,
        measurement_name=meas_file_name,
        measurement_sha256=measurement_sha256,
        database_signature=f"{sig_raw}|{sig_bcb}",
        settings=manifest_settings,
        results=tuple(top_combined),
        package_versions=package_versions,
    )
    _download_button(
        "⬇️ Reproducibility manifest (JSON)",
        data=rex.manifest_json_bytes(run_manifest),
        file_name=f"{Path(meas_file_name).stem}_RamanPhaseID_manifest.json",
        mime="application/json",
        width="stretch",
    )


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

        from collections import defaultdict

        requested_keys = {name.casefold() for name in requested_unique}
        name_index = defaultdict(list)
        # The matching cache already contains the parsed catalog.  Reuse its
        # row order so selecting a name never reparses tens of thousands of
        # source files or builds a second full catalog in memory.
        for db_index, cached_entry in enumerate(pack_raw["meta"]):
            cached_name = str(cached_entry.get("name", "")).strip()
            if cached_name.casefold() not in requested_keys:
                continue
            entry = dict(cached_entry)
            entry["_db_idx"] = db_index
            name_index[cached_name.casefold()].append(entry)

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
            help=(
                "Uses the precomputed fixed-baseline library for raw references. "
                "References already supplied as processed spectra remain as provided."
            ),
        )

    # Plot in voller Breite, Auswahlfelder darunter
    svg_data2, filename2 = b"", ""
    if names_in and chosen_entries:
        try:
            name_plot_signature = _plot_render_signature(
                "name-overlay",
                {
                    "results": result_identity.signature,
                    "rows": [int(entry.get("_db_idx", -1)) for entry in chosen_entries],
                    "database_baseline": apply_db_baseline_for_names,
                    "range": [range_low, range_high],
                    "theme": plot_theme,
                    "line_color_scheme": plot_color_scheme,
                    "intensity_numbers": show_preview_intensity_numbers,
                },
            )

            def _render_name_overlay_figure():
                figure, axis = plt.subplots(figsize=(11, 4.6))
                name_overlay_pack = (
                    pack_bcb if apply_db_baseline_for_names else pack_raw
                )
                overlay_grid = np.asarray(name_overlay_pack["grid"], dtype=float)
                measurement_trace = np.where(
                    approved_query_mask,
                    np.asarray(approved_query_vector, dtype=float),
                    np.nan,
                )
                axis.plot(
                    overlay_grid,
                    measurement_trace,
                    label=f"measurement ({meas_mode})",
                    lw=rplot.PLOT_LINEWIDTH,
                )

                # NaNs preserve real acquisition gaps instead of drawing a
                # line through unsupported detector/library regions.
                for entry in chosen_entries:
                    db_index = int(entry.get("_db_idx", -1))
                    if db_index < 0 or db_index >= len(name_overlay_pack["meta"]):
                        continue
                    cached_meta = name_overlay_pack["meta"][db_index]
                    support_mask = rmatch.reference_support_mask(
                        overlay_grid.size,
                        int(cached_meta.get("start_idx", 0)),
                        int(cached_meta.get("end_idx", -1)),
                        support_runs=cached_meta.get("support_runs"),
                    )
                    visible_mask = (
                        support_mask
                        & (overlay_grid >= range_low)
                        & (overlay_grid <= range_high)
                    )
                    if np.count_nonzero(visible_mask) < 2:
                        continue
                    db_vector = np.asarray(
                        name_overlay_pack["X"][db_index],
                        dtype=float,
                    )
                    db_trace = np.where(visible_mask, db_vector, np.nan)
                    flag = entry.get("flag", "")
                    filename = entry.get("orig_filename", "")
                    treatment = (
                        "baseline corrected"
                        if bool(cached_meta.get("db_baseline", False))
                        else "as provided"
                    )
                    label = (
                        f"{entry['name']} · {flag}{filename} · {treatment}"
                    ).strip()
                    axis.plot(
                        overlay_grid,
                        db_trace,
                        label=label,
                        lw=rplot.PLOT_LINEWIDTH,
                        alpha=0.65,
                    )

                axis.set_xlim(range_low, range_high)
                axis.legend(loc="best")
                _apply_plot_style(
                    figure,
                    axis,
                    theme=plot_theme,
                    color_scheme=plot_color_scheme,
                )
                _set_intensity_number_visibility(
                    axis,
                    show_preview_intensity_numbers,
                )
                return figure

            name_figure = _cached_figure_render_bundle(
                name_plot_signature,
                _render_name_overlay_figure,
            )
            svg_data2 = name_figure.svg
            st.image(name_figure.png, width="stretch")

            meas_name = Path(meas_file_name).stem
            short_names = "_".join([e["name"].split()[0] for e in chosen_entries])[:60].replace("/", "_")
            filename2 = f"{meas_name}_overlay_{short_names or 'minerals'}_{range_low}-{range_high}cm-1.svg"

        except Exception as exc:
            st.error(f"Error plotting name overlays: {exc}")

    if svg_data2 and filename2:
        _download_button(
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

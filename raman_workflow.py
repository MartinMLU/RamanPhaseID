"""Typed, UI-independent workflow state for RamanPhaseID.

The Streamlit application currently keeps approval hashes and draft widget
values directly in ``st.session_state``.  This module models the same workflow
without importing Streamlit:

``input/white-reference + calibration -> baseline -> denoising -> matching``.

Configuration edits are drafts.  Applying a stage snapshots its draft and
invalidates only the applied stages below it.  Consequently, an already
computed result can remain visible while the UI reports pending draft changes;
it is never mistaken for a result produced from those drafts.

Display-only state, notably the plot viewport, is deliberately outside every
scientific signature.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np


PREPROCESS_GRID_STEP_CM1 = 1.0

BaselineMethod: TypeAlias = Literal["arPLS", "ALS", "NONE"]
SmoothingMethod: TypeAlias = Literal["none", "savgol", "deeper_ai"]
ElementMode: TypeAlias = Literal[
    "must_include_all",
    "only_from_list",
    "exact_set",
]
WorkflowStage: TypeAlias = Literal["input", "baseline", "smoothing", "matching"]

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


class WorkflowError(RuntimeError):
    """Base class for invalid workflow transitions."""


class WorkflowOrderError(WorkflowError):
    """Raised when a downstream stage is applied before its upstream stage."""


class WorkflowValidationError(WorkflowError):
    """Raised when a draft cannot be applied safely."""


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0", ""}:
            return False
    return bool(value)


def _payload_of(value: Any) -> Any:
    payload_method = getattr(value, "payload", None)
    return payload_method() if callable(payload_method) else value


def canonical_json(value: Any) -> str:
    """Serialize a payload deterministically for cache and approval hashes."""

    return json.dumps(
        _payload_of(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def payload_signature(value: Any) -> str:
    """Return the full SHA-1 signature used by workflow approvals."""

    return hashlib.sha1(canonical_json(value).encode("utf-8")).hexdigest()


def payload_token(value: Any, length: int = 12) -> str:
    """Return a short deterministic token suitable for labels/cache folders."""

    if length < 1 or length > 40:
        raise ValueError("token length must be between 1 and 40")
    return payload_signature(value)[:length]


@dataclass(frozen=True, slots=True)
class UploadIdentity:
    """Stable identity of an active upload without retaining its bytes."""

    filename: str
    sha1: str
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        filename = str(self.filename or "").strip()
        digest = str(self.sha1 or "").strip().casefold()
        if not _SHA1_RE.fullmatch(digest):
            raise ValueError("upload sha1 must contain exactly 40 hexadecimal characters")
        if self.size_bytes is not None and int(self.size_bytes) < 0:
            raise ValueError("upload size_bytes cannot be negative")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "sha1", digest)
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", int(self.size_bytes))

    @classmethod
    def from_bytes(cls, filename: str, data: bytes | bytearray | memoryview) -> "UploadIdentity":
        raw = bytes(data)
        return cls(
            filename=str(filename),
            sha1=hashlib.sha1(raw).hexdigest(),
            size_bytes=len(raw),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UploadIdentity":
        digest = next(
            (
                value[key]
                for key in ("sha1", "content_sha1", "meas_sha1", "ref_sha1")
                if value.get(key)
            ),
            "",
        )
        filename = next(
            (
                value[key]
                for key in ("filename", "file_name", "name", "ref_filename")
                if value.get(key) is not None
            ),
            "",
        )
        size = next(
            (
                value[key]
                for key in ("size_bytes", "size", "ref_size_bytes")
                if value.get(key) is not None
            ),
            None,
        )
        return cls(filename=str(filename), sha1=str(digest), size_bytes=size)

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "filename": self.filename,
            "sha1": self.sha1,
            "size_bytes": self.size_bytes,
        }

    @property
    def token(self) -> str:
        return payload_token(self)


@dataclass(frozen=True, slots=True)
class SpectralRange:
    """Closed Raman-shift interval in cm^-1."""

    low: float
    high: float

    def __post_init__(self) -> None:
        low = _finite_float(self.low, "range low")
        high = _finite_float(self.high, "range high")
        if high <= low:
            raise ValueError("spectral range high must be greater than low")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @classmethod
    def from_value(
        cls,
        value: "SpectralRange | Sequence[float] | Mapping[str, Any]",
    ) -> "SpectralRange":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            low = value.get("low", value.get("min"))
            high = value.get("high", value.get("max"))
            return cls(float(low), float(high))
        if len(value) != 2:
            raise ValueError("spectral range must have exactly two values")
        return cls(float(value[0]), float(value[1]))

    def payload(self) -> list[float]:
        return [round(self.low, 6), round(self.high, 6)]


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Axis-only calibration approved before processing, projected afterward.

    Approval belongs to the input stage, while numerical baseline/denoising is
    evaluated on native intensities and the shift is applied only when those
    approved intensities are projected onto plotting/matching coordinates.
    """

    shift_cm1: float = 0.0
    axis_unit: str = "cm^-1"
    calibrant: str = ""
    residual_cm1: float | None = None
    excitation_wavelength_nm: float | None = None
    spectral_resolution_cm1: float | None = None
    instrument: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shift_cm1", _finite_float(self.shift_cm1, "shift_cm1"))
        unit = str(self.axis_unit).strip().casefold().replace("cm⁻¹", "cm^-1")
        if unit not in {"cm^-1", "unknown"}:
            raise ValueError("axis_unit must be 'cm^-1' or 'unknown'")
        object.__setattr__(self, "axis_unit", unit)
        for field_name in (
            "residual_cm1",
            "excitation_wavelength_nm",
            "spectral_resolution_cm1",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            number = _finite_float(value, field_name)
            if number < 0.0 or (
                field_name in {"excitation_wavelength_nm", "spectral_resolution_cm1"}
                and number == 0.0
            ):
                raise ValueError(f"{field_name} must be positive when provided")
            object.__setattr__(self, field_name, number)
        object.__setattr__(self, "calibrant", str(self.calibrant).strip())
        object.__setattr__(self, "instrument", str(self.instrument).strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CalibrationConfig":
        return cls(
            shift_cm1=value.get(
                "shift_cm1",
                value.get("meas_shift_cm1", value.get("measurement_shift_cm1", 0.0)),
            ),
            axis_unit=value.get("axis_unit", "cm^-1"),
            calibrant=value.get("calibrant", ""),
            residual_cm1=value.get("residual_cm1"),
            excitation_wavelength_nm=value.get("excitation_wavelength_nm"),
            spectral_resolution_cm1=value.get("spectral_resolution_cm1"),
            instrument=value.get("instrument", ""),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "v": 2,
            "shift_cm1": round(self.shift_cm1, 6),
            "axis_unit": self.axis_unit,
            "calibrant": self.calibrant,
            "residual_cm1": self.residual_cm1,
            "excitation_wavelength_nm": self.excitation_wavelength_nm,
            "spectral_resolution_cm1": self.spectral_resolution_cm1,
            "instrument": self.instrument,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "meas_shift_cm1": self.shift_cm1,
            "axis_unit": self.axis_unit,
            "calibrant": self.calibrant,
            "residual_cm1": self.residual_cm1,
            "excitation_wavelength_nm": self.excitation_wavelength_nm,
            "spectral_resolution_cm1": self.spectral_resolution_cm1,
            "instrument": self.instrument,
        }

    @property
    def token(self) -> str:
        return payload_token(self)


@dataclass(frozen=True, slots=True)
class WhiteReferenceConfig:
    """Optional white-light subtraction draft or applied configuration."""

    enabled: bool = False
    scale: float = 1.0
    reference: UploadIdentity | None = None

    def __post_init__(self) -> None:
        scale = _finite_float(self.scale, "white-reference scale")
        if scale < 0.0:
            raise ValueError("white-reference scale cannot be negative")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WhiteReferenceConfig":
        reference_value = value.get("reference")
        reference: UploadIdentity | None
        if isinstance(reference_value, UploadIdentity):
            reference = reference_value
        elif isinstance(reference_value, Mapping):
            reference = UploadIdentity.from_mapping(reference_value)
        elif value.get("ref_sha1"):
            reference = UploadIdentity.from_mapping(value)
        else:
            reference = None
        return cls(
            enabled=_bool(value.get("enabled", False)),
            scale=value.get("scale", 1.0),
            reference=reference,
        )

    @property
    def is_ready(self) -> bool:
        return not self.enabled or self.reference is not None

    def with_reference(self, reference: UploadIdentity | None) -> "WhiteReferenceConfig":
        return replace(self, reference=reference)

    def payload(self) -> dict[str, Any]:
        """Legacy-compatible processing payload used by the current app."""

        return {
            "v": 1,
            "enabled": self.enabled,
            "scale": round(self.scale, 6),
            "ref_sha1": self.reference.sha1 if self.enabled and self.reference else "",
        }

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "scale": self.scale,
            "ref_sha1": self.reference.sha1 if self.enabled and self.reference else "",
        }
        if self.reference is not None:
            result["ref_filename"] = self.reference.filename
            result["ref_size_bytes"] = self.reference.size_bytes
        return result

    @property
    def token(self) -> str:
        return payload_token(self)


def _baseline_method(value: Any) -> BaselineMethod:
    normalized = str(value or "arPLS").strip().casefold()
    aliases: dict[str, BaselineMethod] = {
        "arpls": "arPLS",
        "als": "ALS",
        "als (iasls)": "ALS",
        "iasls": "ALS",
        "none": "NONE",
        "raw": "NONE",
        "raw (no baseline)": "NONE",
        "off": "NONE",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown baseline method: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Baseline settings on the canonical physical processing grid."""

    method: BaselineMethod = "arPLS"
    lam_exp: int = 5
    lam: float = 1e5
    itermax: int = 50
    tol: float = 1e-3
    p: float = 0.010
    niter: int = 20
    lam1_exp: int = 2
    lam1: float = 1e2
    autoscale: bool = True
    db_strength: float = 1.0
    have_scipy: bool = True
    banded_solver: bool | None = None
    grid_step_cm1: float = PREPROCESS_GRID_STEP_CM1

    def __post_init__(self) -> None:
        method = _baseline_method(self.method)
        lam = _finite_float(self.lam, "baseline lambda")
        lam1 = _finite_float(self.lam1, "baseline lambda1")
        tol = _finite_float(self.tol, "baseline tolerance")
        p = _finite_float(self.p, "baseline asymmetry")
        strength = _finite_float(self.db_strength, "database baseline strength")
        grid_step = _finite_float(self.grid_step_cm1, "processing grid step")
        if lam <= 0.0 or lam1 <= 0.0 or tol <= 0.0 or grid_step <= 0.0:
            raise ValueError("baseline lambdas, tolerance, and grid step must be positive")
        if int(self.itermax) < 1 or int(self.niter) < 1:
            raise ValueError("baseline iteration counts must be positive")
        if not 0.0 <= p <= 1.0:
            raise ValueError("baseline asymmetry p must be between 0 and 1")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("database baseline strength must be between 0 and 1")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "lam_exp", int(self.lam_exp))
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "itermax", int(self.itermax))
        object.__setattr__(self, "tol", tol)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "niter", int(self.niter))
        object.__setattr__(self, "lam1_exp", int(self.lam1_exp))
        object.__setattr__(self, "lam1", lam1)
        object.__setattr__(self, "autoscale", bool(self.autoscale))
        object.__setattr__(self, "db_strength", strength)
        object.__setattr__(self, "have_scipy", bool(self.have_scipy))
        object.__setattr__(
            self,
            "banded_solver",
            bool(self.have_scipy) if self.banded_solver is None else bool(self.banded_solver),
        )
        object.__setattr__(self, "grid_step_cm1", grid_step)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        have_scipy: bool | None = None,
    ) -> "BaselineConfig":
        raw_method = value.get("method", value.get("baseline_mode", "arPLS"))
        lam_exp = int(value.get("lam_exp", 5))
        lam1_exp = int(value.get("lam1_exp", 2))
        return cls(
            method=_baseline_method(raw_method),
            lam_exp=lam_exp,
            lam=value.get("lam", 10.0**lam_exp),
            itermax=value.get("itermax", 50),
            tol=value.get("tol", 1e-3),
            p=value.get("p", 0.010),
            niter=value.get("niter", 20),
            lam1_exp=lam1_exp,
            lam1=value.get("lam1", 10.0**lam1_exp),
            autoscale=_bool(value.get("autoscale", True)),
            db_strength=value.get("db_strength", 1.0),
            have_scipy=(
                _bool(value.get("have_scipy", True))
                if have_scipy is None
                else bool(have_scipy)
            ),
            banded_solver=(
                _bool(value["banded_solver"])
                if value.get("banded_solver") is not None
                else None
            ),
            grid_step_cm1=value.get("grid_step_cm1", PREPROCESS_GRID_STEP_CM1),
        )

    @property
    def measurement_mode(self) -> Literal["BC", "RAW"]:
        return "RAW" if self.method == "NONE" else "BC"

    def payload(self) -> dict[str, Any]:
        """Payload compatible with ``raman_preprocessing.baseline_payload``."""

        result: dict[str, Any] = {
            "v": 5,
            "grid_step_cm1": self.grid_step_cm1,
            "method": self.method,
            "lam_exp": self.lam_exp,
            "lam": self.lam,
            "autoscale": self.autoscale,
            "db_strength": round(self.db_strength, 4),
            "have_scipy": self.have_scipy,
            "banded_solver": self.banded_solver,
        }
        if self.method == "arPLS":
            result.update({"itermax": self.itermax, "tol": self.tol})
        elif self.method == "ALS":
            result.update(
                {
                    "p": self.p,
                    "niter": self.niter,
                    "lam1_exp": self.lam1_exp,
                    "lam1": self.lam1,
                }
            )
        return result

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "lam_exp": self.lam_exp,
            "lam": self.lam,
            "itermax": self.itermax,
            "tol": self.tol,
            "p": self.p,
            "niter": self.niter,
            "lam1_exp": self.lam1_exp,
            "lam1": self.lam1,
            "autoscale": self.autoscale,
            "db_strength": self.db_strength,
        }

    @property
    def token(self) -> str:
        return payload_token(self)


@dataclass(frozen=True, slots=True)
class AIDenoiserSpec:
    """Identity of the pinned guarded DeepeR model/adapter contract."""

    model_id: str = "DeepeR-ResUNet-500"
    model_sha256: str = "23d11061fce98656f32f8d604d2e58973853a3f79ce69e9f08dac4d8ef9747b2"
    model_points: int = 500
    model_training_range_cm1: tuple[float, float] = (500.0, 1800.0)
    full_range_adapter_v: int = 1
    window_span_cm1: float = 1300.0
    window_overlap_cm1: float = 325.0
    guard_guide_span_cm1: float = 10.0
    guard_trend_span_cm1: float = 20.0
    guard_max_range_fraction: float = 0.02

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AIDenoiserSpec":
        defaults = cls()
        training_range = value.get(
            "model_training_range_cm1",
            defaults.model_training_range_cm1,
        )
        return cls(
            model_id=str(value.get("model_id", defaults.model_id)),
            model_sha256=str(value.get("model_sha256", defaults.model_sha256)),
            model_points=int(value.get("model_points", defaults.model_points)),
            model_training_range_cm1=(float(training_range[0]), float(training_range[1])),
            full_range_adapter_v=int(
                value.get("full_range_adapter_v", defaults.full_range_adapter_v)
            ),
            window_span_cm1=float(value.get("window_span_cm1", defaults.window_span_cm1)),
            window_overlap_cm1=float(
                value.get("window_overlap_cm1", defaults.window_overlap_cm1)
            ),
            guard_guide_span_cm1=float(
                value.get("guard_guide_span_cm1", defaults.guard_guide_span_cm1)
            ),
            guard_trend_span_cm1=float(
                value.get("guard_trend_span_cm1", defaults.guard_trend_span_cm1)
            ),
            guard_max_range_fraction=float(
                value.get(
                    "guard_max_range_fraction",
                    defaults.guard_max_range_fraction,
                )
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "model_points": self.model_points,
            "model_training_range_cm1": list(self.model_training_range_cm1),
            "full_range_adapter_v": self.full_range_adapter_v,
            "window_span_cm1": self.window_span_cm1,
            "window_overlap_cm1": self.window_overlap_cm1,
            "guard_guide_span_cm1": self.guard_guide_span_cm1,
            "guard_trend_span_cm1": self.guard_trend_span_cm1,
            "guard_max_range_fraction": self.guard_max_range_fraction,
        }


def _smoothing_method(value: Any, *, enabled: Any = True) -> SmoothingMethod:
    if value is None:
        return "savgol" if _bool(enabled) else "none"
    normalized = str(value).strip().casefold()
    aliases: dict[str, SmoothingMethod] = {
        "none": "none",
        "off": "none",
        "none (keep measurement unchanged)": "none",
        "savgol": "savgol",
        "savitzky-golay": "savgol",
        "savitzky–golay": "savgol",
        "ai": "deeper_ai",
        "deeper": "deeper_ai",
        "deeper_ai": "deeper_ai",
        "ai-assisted · guarded deeper (full range)": "deeper_ai",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown denoising method: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """Measurement-only denoising/smoothing settings."""

    method: SmoothingMethod = "savgol"
    window: int = 5
    poly: int = 3
    max_change_sigma: float = 1.0
    grid_step_cm1: float = PREPROCESS_GRID_STEP_CM1
    ai_spec: AIDenoiserSpec = field(default_factory=AIDenoiserSpec)

    def __post_init__(self) -> None:
        method = _smoothing_method(self.method)
        window = int(self.window)
        poly = int(self.poly)
        max_change = _finite_float(self.max_change_sigma, "AI maximum change sigma")
        grid_step = _finite_float(self.grid_step_cm1, "processing grid step")
        if window < 1 or poly < 0:
            raise ValueError("smoothing window must be positive and polynomial non-negative")
        if max_change < 0.0 or grid_step <= 0.0:
            raise ValueError(
                "AI correction limit cannot be negative and grid step must be positive"
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "poly", poly)
        object.__setattr__(self, "max_change_sigma", max_change)
        object.__setattr__(self, "grid_step_cm1", grid_step)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SmoothingConfig":
        if value is None:
            return cls(method="none")
        method = _smoothing_method(value.get("method"), enabled=value.get("enabled", True))
        ai_value = value.get("ai_spec")
        ai_spec = (
            ai_value
            if isinstance(ai_value, AIDenoiserSpec)
            else AIDenoiserSpec.from_mapping(
                ai_value if isinstance(ai_value, Mapping) else value
            )
        )
        return cls(
            method=method,
            window=value.get("window", 5),
            poly=value.get("poly", 3),
            max_change_sigma=value.get("max_change_sigma", 1.0),
            grid_step_cm1=value.get("grid_step_cm1", PREPROCESS_GRID_STEP_CM1),
            ai_spec=ai_spec,
        )

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    def payload(self) -> dict[str, Any]:
        """Payload compatible with ``raman_preprocessing.smoothing_payload``."""

        result: dict[str, Any] = {
            "v": 5,
            "grid_step_cm1": self.grid_step_cm1,
            "method": self.method,
        }
        if self.method == "savgol":
            result.update({"window": self.window, "poly": self.poly})
        elif self.method == "deeper_ai":
            result.update(self.ai_spec.payload())
            result["max_change_sigma"] = round(self.max_change_sigma, 3)
        return result

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "enabled": self.enabled,
            "window": self.window,
            "poly": self.poly,
            "max_change_sigma": self.max_change_sigma,
        }

    @property
    def token(self) -> str:
        return payload_token(self)


def _element_mode(value: Any) -> ElementMode:
    normalized = str(value or "Must include all").strip().casefold()
    aliases: dict[str, ElementMode] = {
        "must include all": "must_include_all",
        "must_include_all": "must_include_all",
        "only from this list": "only_from_list",
        "only_from_list": "only_from_list",
        "exactly this set": "exact_set",
        "exact_set": "exact_set",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown element filter mode: {value!r}") from exc


def _elements(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    tokens = re.split(r"[,;\s]+", value.strip()) if isinstance(value, str) else value
    normalized = {
        str(token).strip().capitalize()
        for token in tokens
        if str(token).strip()
    }
    return tuple(sorted(token for token in normalized if _ELEMENT_RE.fullmatch(token)))


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    """Applied scientific matching settings; display zoom is not included."""

    range_cm1: SpectralRange | None = None
    include_elements: tuple[str, ...] = ()
    exclude_elements: tuple[str, ...] = ()
    element_mode: ElementMode = "must_include_all"
    allow_missing_formula: bool = True
    database_folders: tuple[str, ...] = ()
    raw_database_signature: str = ""
    baseline_database_signature: str = ""
    top_n: int = 60
    gradient_weight: float = 0.20
    peak_f1_weight: float = 0.75
    peak_tolerance_cm1: int = 5
    selection_version: int = 5
    policy_signature: str = ""

    def __post_init__(self) -> None:
        range_cm1 = (
            SpectralRange.from_value(self.range_cm1)
            if self.range_cm1 is not None
            else None
        )
        gradient_weight = _finite_float(self.gradient_weight, "gradient weight")
        peak_f1_weight = _finite_float(self.peak_f1_weight, "peak F1 weight")
        if not 0.0 <= gradient_weight <= 1.0 or not 0.0 <= peak_f1_weight <= 1.0:
            raise ValueError("matching weights must be between 0 and 1")
        if int(self.top_n) < 1 or int(self.peak_tolerance_cm1) < 0:
            raise ValueError("top_n must be positive and peak tolerance non-negative")
        object.__setattr__(self, "range_cm1", range_cm1)
        object.__setattr__(self, "include_elements", _elements(self.include_elements))
        object.__setattr__(self, "exclude_elements", _elements(self.exclude_elements))
        object.__setattr__(self, "element_mode", _element_mode(self.element_mode))
        object.__setattr__(self, "allow_missing_formula", bool(self.allow_missing_formula))
        object.__setattr__(
            self,
            "database_folders",
            tuple(str(folder) for folder in self.database_folders),
        )
        object.__setattr__(self, "raw_database_signature", str(self.raw_database_signature))
        object.__setattr__(
            self,
            "baseline_database_signature",
            str(self.baseline_database_signature),
        )
        object.__setattr__(self, "top_n", int(self.top_n))
        object.__setattr__(self, "gradient_weight", gradient_weight)
        object.__setattr__(self, "peak_f1_weight", peak_f1_weight)
        object.__setattr__(self, "peak_tolerance_cm1", int(self.peak_tolerance_cm1))
        object.__setattr__(self, "selection_version", int(self.selection_version))
        object.__setattr__(self, "policy_signature", str(self.policy_signature).strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MatchingConfig":
        # Legacy mappings may still contain ``ultra``/``full_range`` and
        # ``reference_scope``. They are intentionally ignored: the application
        # now has one full-range cache and one all-configured-references policy.
        raw_range = value.get("range_cm1", value.get("range"))
        if raw_range is None and value.get("range_low") is not None:
            raw_range = (value["range_low"], value["range_high"])
        return cls(
            range_cm1=SpectralRange.from_value(raw_range) if raw_range is not None else None,
            include_elements=_elements(
                value.get("include_elements", value.get("include", ""))
            ),
            exclude_elements=_elements(
                value.get("exclude_elements", value.get("exclude", ""))
            ),
            element_mode=_element_mode(value.get("element_mode", value.get("mode"))),
            allow_missing_formula=_bool(
                value.get("allow_missing_formula", value.get("allow", True))
            ),
            database_folders=tuple(
                str(folder)
                for folder in value.get("database_folders", value.get("folders", ()))
            ),
            raw_database_signature=str(
                value.get("raw_database_signature", value.get("sig_raw", ""))
            ),
            baseline_database_signature=str(
                value.get("baseline_database_signature", value.get("sig_bcb", ""))
            ),
            top_n=value.get("top_n", 60),
            gradient_weight=value.get("gradient_weight", value.get("grad_w", 0.20)),
            peak_f1_weight=value.get("peak_f1_weight", 0.75),
            peak_tolerance_cm1=value.get("peak_tolerance_cm1", value.get("peak_tol", 5)),
            selection_version=value.get(
                "selection_version",
                value.get("match_selection_v", 5),
            ),
            policy_signature=str(
                value.get(
                    "policy_signature",
                    value.get("matching_policy_signature", ""),
                )
            ),
        )

    @property
    def is_ready(self) -> bool:
        return self.range_cm1 is not None

    def with_range(
        self,
        value: SpectralRange | Sequence[float] | Mapping[str, Any],
    ) -> "MatchingConfig":
        return replace(self, range_cm1=SpectralRange.from_value(value))

    def payload(self) -> dict[str, Any]:
        return {
            "v": 2,
            "range_cm1": self.range_cm1.payload() if self.range_cm1 else None,
            "include_elements": list(self.include_elements),
            "exclude_elements": list(self.exclude_elements),
            "element_mode": self.element_mode,
            "allow_missing_formula": self.allow_missing_formula,
            "database_folders": list(self.database_folders),
            "raw_database_signature": self.raw_database_signature,
            "baseline_database_signature": self.baseline_database_signature,
            "top_n": self.top_n,
            "gradient_weight": self.gradient_weight,
            "peak_f1_weight": self.peak_f1_weight,
            "peak_tolerance_cm1": self.peak_tolerance_cm1,
            "selection_version": self.selection_version,
            "policy_signature": self.policy_signature,
        }

    def to_mapping(self) -> dict[str, Any]:
        mode_labels = {
            "must_include_all": "Must include all",
            "only_from_list": "Only from this list",
            "exact_set": "Exactly this set",
        }
        result: dict[str, Any] = {
            "range": (
                (self.range_cm1.low, self.range_cm1.high)
                if self.range_cm1
                else None
            ),
            "include": ",".join(self.include_elements),
            "exclude": ",".join(self.exclude_elements),
            "mode": mode_labels[self.element_mode],
            "allow": self.allow_missing_formula,
            "folders": self.database_folders,
            "sig_raw": self.raw_database_signature,
            "sig_bcb": self.baseline_database_signature,
            "top_n": self.top_n,
            "policy_signature": self.policy_signature,
        }
        return result

    @property
    def token(self) -> str:
        return payload_token(self)


@dataclass(frozen=True, slots=True)
class InputApproval:
    """Applied measurement, white-reference, and calibration snapshot."""

    measurement: UploadIdentity
    white_reference: WhiteReferenceConfig
    calibration: CalibrationConfig
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.measurement.filename:
            raise WorkflowValidationError("the active measurement needs a filename")
        if not self.white_reference.is_ready:
            raise WorkflowValidationError(
                "white-reference subtraction is enabled but no reference upload is active"
            )
        if self.calibration.axis_unit != "cm^-1":
            raise WorkflowValidationError(
                "matching requires an axis explicitly confirmed as Raman shift in cm^-1"
            )
        object.__setattr__(self, "signature", payload_signature(self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "measurement": self.measurement.payload(),
            "white_reference": self.white_reference.payload(),
            "white_reference_upload": (
                self.white_reference.reference.payload()
                if self.white_reference.enabled and self.white_reference.reference
                else None
            ),
            "calibration": self.calibration.payload(),
        }

    @property
    def token(self) -> str:
        return self.signature[:12]


@dataclass(frozen=True, slots=True)
class BaselineApproval:
    input_signature: str
    config: BaselineConfig
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.input_signature:
            raise WorkflowValidationError("baseline approval needs an input approval")
        object.__setattr__(self, "signature", payload_signature(self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "input_signature": self.input_signature,
            "baseline": self.config.payload(),
            "measurement_mode": self.config.measurement_mode,
        }

    @property
    def token(self) -> str:
        return self.signature[:12]


@dataclass(frozen=True, slots=True)
class SmoothingApproval:
    baseline_signature: str
    config: SmoothingConfig
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.baseline_signature:
            raise WorkflowValidationError("smoothing approval needs a baseline approval")
        object.__setattr__(self, "signature", payload_signature(self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "baseline_signature": self.baseline_signature,
            "smoothing": self.config.payload(),
        }

    @property
    def token(self) -> str:
        return self.signature[:12]


@dataclass(frozen=True, slots=True)
class MatchingApproval:
    smoothing_signature: str
    config: MatchingConfig
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.smoothing_signature:
            raise WorkflowValidationError("matching approval needs a smoothing approval")
        if not self.config.is_ready:
            raise WorkflowValidationError("matching range must be set before matching")
        object.__setattr__(self, "signature", payload_signature(self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "smoothing_signature": self.smoothing_signature,
            "matching": self.config.payload(),
        }

    @property
    def token(self) -> str:
        return self.signature[:12]


@dataclass(frozen=True, slots=True)
class ResultIdentity:
    """Identity of one completed primary search.

    The matching approval already chains the measurement, white-reference,
    calibration, baseline, smoothing, database, range, and matching-policy
    identities.  Keeping that signature explicit here distinguishes an
    *approved request* from a search that actually completed.
    """

    matching_signature: str
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        matching_signature = str(self.matching_signature).strip().casefold()
        if not _SHA1_RE.fullmatch(matching_signature):
            raise ValueError(
                "result identity needs a 40-character matching approval signature"
            )
        object.__setattr__(self, "matching_signature", matching_signature)
        object.__setattr__(self, "signature", payload_signature(self.payload()))

    @classmethod
    def from_matching_approval(cls, approval: MatchingApproval) -> "ResultIdentity":
        return cls(approval.signature)

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "matching_signature": self.matching_signature,
        }

    @property
    def token(self) -> str:
        return self.signature[:12]


def residual_query_content_sha256(
    query_vector: np.ndarray,
    query_mask: np.ndarray,
) -> str:
    """Hash the exact float32 residual query and boolean support deterministically."""

    vector = np.asarray(query_vector, dtype="<f4").reshape(-1)
    mask = np.asarray(query_mask, dtype=bool).reshape(-1)
    if vector.size != mask.size:
        raise ValueError("residual query vector and mask must have equal length")
    if not np.all(np.isfinite(vector)):
        raise ValueError("residual query vector must contain only finite values")
    digest = hashlib.sha256()
    digest.update(b"RamanPhaseID-residual-query-mask-v1\0")
    digest.update(int(vector.size).to_bytes(8, "little", signed=False))
    digest.update(vector.tobytes(order="C"))
    digest.update(mask.astype(np.uint8, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _float_array_content_sha256(values: np.ndarray, *, label: bytes) -> str:
    array = np.asarray(values, dtype="<f8").reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("scientific result arrays must contain only finite values")
    digest = hashlib.sha256()
    digest.update(label)
    digest.update(int(array.size).to_bytes(8, "little", signed=False))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_support_runs(value: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    runs = sorted((int(run[0]), int(run[1])) for run in value if len(run) == 2)
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if start < 0 or end < start:
            raise ValueError("support runs must be non-negative inclusive index pairs")
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class ResidualReferenceIdentity:
    """Exact library row and alignment subtracted before residual rematching."""

    phase_name: str
    database_variant: str
    database_signature: str
    database_index: int
    reference_id: str
    path: str
    accession: str
    filename: str
    fitted_shift_points: int
    fitted_shift_cm1: float
    start_idx: int
    end_idx: int
    support_runs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        phase_name = str(self.phase_name).strip()
        database_variant = str(self.database_variant).strip()
        database_signature = str(self.database_signature).strip()
        reference_id = str(self.reference_id).strip()
        if not phase_name or not database_variant or not database_signature:
            raise ValueError("residual reference phase, variant, and database signature are required")
        if int(self.database_index) < 0:
            raise ValueError("residual reference database_index must be non-negative")
        if not reference_id:
            raise ValueError("residual reference_id must not be empty")
        shift_cm1 = _finite_float(self.fitted_shift_cm1, "fitted shift")
        start_idx = int(self.start_idx)
        end_idx = int(self.end_idx)
        if start_idx < 0 or end_idx < start_idx:
            raise ValueError("residual reference bounds must be a non-empty inclusive interval")
        runs = _canonical_support_runs(self.support_runs)
        if not runs:
            runs = ((start_idx, end_idx),)
        if runs[0][0] < start_idx or runs[-1][1] > end_idx:
            raise ValueError("residual reference support runs must lie inside its bounds")
        object.__setattr__(self, "phase_name", phase_name)
        object.__setattr__(self, "database_variant", database_variant)
        object.__setattr__(self, "database_signature", database_signature)
        object.__setattr__(self, "database_index", int(self.database_index))
        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "path", str(self.path).strip())
        object.__setattr__(self, "accession", str(self.accession).strip())
        object.__setattr__(self, "filename", str(self.filename).strip())
        object.__setattr__(self, "fitted_shift_points", int(self.fitted_shift_points))
        object.__setattr__(self, "fitted_shift_cm1", shift_cm1)
        object.__setattr__(self, "start_idx", start_idx)
        object.__setattr__(self, "end_idx", end_idx)
        object.__setattr__(self, "support_runs", runs)

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "phase_name": self.phase_name,
            "database_variant": self.database_variant,
            "database_signature": self.database_signature,
            "database_index": self.database_index,
            "reference_id": self.reference_id,
            "path": self.path,
            "accession": self.accession,
            "filename": self.filename,
            "fitted_shift_points": self.fitted_shift_points,
            "fitted_shift_cm1": self.fitted_shift_cm1,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "support_runs": [list(run) for run in self.support_runs],
        }


@dataclass(frozen=True, slots=True)
class ResidualResultIdentity:
    """Deterministic identity of one exact exploratory residual query."""

    primary_result_signature: str
    subtracted_reference: ResidualReferenceIdentity
    scale_factor: float
    residual_query_mask_sha256: str
    signed_residual_sha256: str
    query_point_count: int
    residual_policy_signature: str
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        primary = str(self.primary_result_signature).strip().casefold()
        query_hash = str(self.residual_query_mask_sha256).strip().casefold()
        residual_hash = str(self.signed_residual_sha256).strip().casefold()
        policy = str(self.residual_policy_signature).strip().casefold()
        if not _SHA1_RE.fullmatch(primary):
            raise ValueError("residual identity needs a primary result signature")
        if not _SHA256_RE.fullmatch(query_hash) or not _SHA256_RE.fullmatch(residual_hash):
            raise ValueError("residual identity needs valid SHA-256 content hashes")
        if not _SHA1_RE.fullmatch(policy):
            raise ValueError("residual identity needs a signed residual-search policy")
        scale_factor = _finite_float(self.scale_factor, "residual scale factor")
        if scale_factor < 0.0:
            raise ValueError("residual scale factor must be non-negative")
        if int(self.query_point_count) < 1:
            raise ValueError("residual query must contain at least one point")
        object.__setattr__(self, "primary_result_signature", primary)
        object.__setattr__(self, "scale_factor", scale_factor)
        object.__setattr__(self, "residual_query_mask_sha256", query_hash)
        object.__setattr__(self, "signed_residual_sha256", residual_hash)
        object.__setattr__(self, "query_point_count", int(self.query_point_count))
        object.__setattr__(self, "residual_policy_signature", policy)
        object.__setattr__(self, "signature", payload_signature(self.payload()))

    @classmethod
    def from_components(
        cls,
        primary_identity: ResultIdentity,
        subtracted_reference: ResidualReferenceIdentity,
        scale_factor: float,
        query_vector: np.ndarray,
        query_mask: np.ndarray,
        signed_residual: np.ndarray,
        residual_policy_signature: str,
    ) -> "ResidualResultIdentity":
        vector = np.asarray(query_vector).reshape(-1)
        mask = np.asarray(query_mask).reshape(-1)
        residual = np.asarray(signed_residual).reshape(-1)
        if residual.size != vector.size:
            raise ValueError("signed residual and matching query must have equal length")
        return cls(
            primary_result_signature=primary_identity.signature,
            subtracted_reference=subtracted_reference,
            scale_factor=scale_factor,
            residual_query_mask_sha256=residual_query_content_sha256(vector, mask),
            signed_residual_sha256=_float_array_content_sha256(
                residual,
                label=b"RamanPhaseID-signed-residual-v1\0",
            ),
            query_point_count=int(vector.size),
            residual_policy_signature=residual_policy_signature,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "v": 1,
            "primary_result_signature": self.primary_result_signature,
            "subtracted_reference": self.subtracted_reference.payload(),
            "scale_factor": self.scale_factor,
            "residual_query_mask_sha256": self.residual_query_mask_sha256,
            "signed_residual_sha256": self.signed_residual_sha256,
            "query_point_count": self.query_point_count,
            "residual_policy_signature": self.residual_policy_signature,
        }

    @property
    def token(self) -> str:
        return self.signature[:12]


def _freeze_result_value(value: Any) -> Any:
    """Recursively freeze result metadata without coercing scientific scalars."""

    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_result_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_result_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_result_value(item) for item in value)
    return value


def _thaw_result_value(value: Any) -> Any:
    """Return a detached conventional container for UI/export consumers."""

    if isinstance(value, MappingABC):
        return {str(key): _thaw_result_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_result_value(item) for item in value)
    if isinstance(value, frozenset):
        return set(_thaw_result_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PrimaryResultSnapshot:
    """Immutable primary-match output bound to the exact approval chain.

    Results and query support are copied on construction.  The numerical
    arrays are read-only and nested result mappings are recursively frozen, so
    later widget reruns cannot silently mutate the evidence represented by the
    identity.
    """

    identity: ResultIdentity
    input_approval: InputApproval
    baseline_approval: BaselineApproval
    smoothing_approval: SmoothingApproval
    matching_approval: MatchingApproval
    results: tuple[Mapping[str, Any], ...]
    query_vector: np.ndarray
    query_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.baseline_approval.input_signature != self.input_approval.signature:
            raise WorkflowValidationError("result baseline approval chain is inconsistent")
        if (
            self.smoothing_approval.baseline_signature
            != self.baseline_approval.signature
        ):
            raise WorkflowValidationError("result smoothing approval chain is inconsistent")
        if (
            self.matching_approval.smoothing_signature
            != self.smoothing_approval.signature
        ):
            raise WorkflowValidationError("result matching approval chain is inconsistent")
        if self.identity.matching_signature != self.matching_approval.signature:
            raise WorkflowValidationError(
                "result identity does not match its matching approval"
            )

        query_vector = np.array(self.query_vector, dtype=np.float32, copy=True).reshape(-1)
        query_mask = np.array(self.query_mask, dtype=bool, copy=True).reshape(-1)
        if query_vector.size != query_mask.size:
            raise ValueError("result query vector and mask must have equal length")
        if not np.all(np.isfinite(query_vector)):
            raise ValueError("result query vector must contain only finite values")
        query_vector.setflags(write=False)
        query_mask.setflags(write=False)
        frozen_results = tuple(
            _freeze_result_value(dict(result)) for result in tuple(self.results)
        )
        object.__setattr__(self, "results", frozen_results)
        object.__setattr__(self, "query_vector", query_vector)
        object.__setattr__(self, "query_mask", query_mask)

    @classmethod
    def from_workflow(
        cls,
        workflow: "WorkflowState",
        results: Sequence[Mapping[str, Any]],
        query_vector: np.ndarray,
        query_mask: np.ndarray,
    ) -> "PrimaryResultSnapshot":
        identity = workflow.expected_result_identity
        if (
            identity is None
            or workflow.input_approval is None
            or workflow.baseline_approval is None
            or workflow.smoothing_approval is None
            or workflow.matching_approval is None
        ):
            raise WorkflowOrderError(
                "apply all current settings before recording primary results"
            )
        return cls(
            identity=identity,
            input_approval=workflow.input_approval,
            baseline_approval=workflow.baseline_approval,
            smoothing_approval=workflow.smoothing_approval,
            matching_approval=workflow.matching_approval,
            results=tuple(results),
            query_vector=query_vector,
            query_mask=query_mask,
        )

    @property
    def is_empty(self) -> bool:
        return not self.results

    def result_mappings(self) -> list[dict[str, Any]]:
        """Return detached result dictionaries for existing UI/export code."""

        return [_thaw_result_value(result) for result in self.results]


@dataclass(frozen=True, slots=True)
class ResidualResultSnapshot:
    """Immutable residual matches and plotted residual bound to one primary result."""

    identity: ResidualResultIdentity
    primary_identity: ResultIdentity
    results: tuple[Mapping[str, Any], ...]
    query_vector: np.ndarray
    query_mask: np.ndarray
    signed_residual: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.identity.primary_result_signature != self.primary_identity.signature:
            raise WorkflowValidationError(
                "residual result identity does not match its primary result"
            )
        query_vector = np.array(self.query_vector, dtype=np.float32, copy=True).reshape(-1)
        query_mask = np.array(self.query_mask, dtype=bool, copy=True).reshape(-1)
        signed_residual = np.array(
            self.signed_residual,
            dtype=float,
            copy=True,
        ).reshape(-1)
        if not (
            query_vector.size == query_mask.size == signed_residual.size
        ):
            raise ValueError("residual snapshot arrays must have equal length")
        if not np.all(np.isfinite(query_vector)) or not np.all(
            np.isfinite(signed_residual)
        ):
            raise ValueError("residual snapshot arrays must contain only finite values")
        expected_identity = ResidualResultIdentity.from_components(
            self.primary_identity,
            self.identity.subtracted_reference,
            self.identity.scale_factor,
            query_vector,
            query_mask,
            signed_residual,
            self.identity.residual_policy_signature,
        )
        if expected_identity != self.identity:
            raise WorkflowValidationError(
                "residual result identity does not match its immutable numerical content"
            )
        query_vector.setflags(write=False)
        query_mask.setflags(write=False)
        signed_residual.setflags(write=False)
        object.__setattr__(
            self,
            "results",
            tuple(_freeze_result_value(dict(result)) for result in tuple(self.results)),
        )
        object.__setattr__(self, "query_vector", query_vector)
        object.__setattr__(self, "query_mask", query_mask)
        object.__setattr__(self, "signed_residual", signed_residual)
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_result_value(dict(self.diagnostics)),
        )

    @classmethod
    def from_primary(
        cls,
        primary_snapshot: PrimaryResultSnapshot,
        subtracted_reference: ResidualReferenceIdentity,
        scale_factor: float,
        results: Sequence[Mapping[str, Any]],
        query_vector: np.ndarray,
        query_mask: np.ndarray,
        signed_residual: np.ndarray,
        residual_policy_signature: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ResidualResultSnapshot":
        identity = ResidualResultIdentity.from_components(
            primary_snapshot.identity,
            subtracted_reference,
            scale_factor,
            query_vector,
            query_mask,
            signed_residual,
            residual_policy_signature,
        )
        return cls(
            identity=identity,
            primary_identity=primary_snapshot.identity,
            results=tuple(results),
            query_vector=query_vector,
            query_mask=query_mask,
            signed_residual=signed_residual,
            diagnostics=diagnostics or {},
        )

    @property
    def is_empty(self) -> bool:
        return not self.results

    def result_mappings(self) -> list[dict[str, Any]]:
        return [_thaw_result_value(result) for result in self.results]

    def diagnostics_mapping(self) -> dict[str, Any]:
        return _thaw_result_value(self.diagnostics)


@dataclass(frozen=True, slots=True)
class WorkflowState:
    """Immutable drafts and applied snapshots for one active measurement."""

    measurement: UploadIdentity | None = None
    white_reference_draft: WhiteReferenceConfig = field(
        default_factory=WhiteReferenceConfig
    )
    calibration_draft: CalibrationConfig = field(default_factory=CalibrationConfig)
    baseline_draft: BaselineConfig = field(default_factory=BaselineConfig)
    smoothing_draft: SmoothingConfig = field(default_factory=SmoothingConfig)
    matching_draft: MatchingConfig = field(default_factory=MatchingConfig)
    display_viewport: SpectralRange | None = None
    input_approval: InputApproval | None = None
    baseline_approval: BaselineApproval | None = None
    smoothing_approval: SmoothingApproval | None = None
    matching_approval: MatchingApproval | None = None
    result_identity: ResultIdentity | None = None

    def set_measurement(self, measurement: UploadIdentity | None) -> "WorkflowState":
        """Activate an upload and invalidate every applied scientific stage."""

        if measurement == self.measurement:
            return self
        return replace(
            self,
            measurement=measurement,
            display_viewport=None,
            input_approval=None,
            baseline_approval=None,
            smoothing_approval=None,
            matching_approval=None,
            result_identity=None,
        )

    def set_measurement_bytes(
        self,
        filename: str,
        data: bytes | bytearray | memoryview,
    ) -> "WorkflowState":
        return self.set_measurement(UploadIdentity.from_bytes(filename, data))

    def with_white_reference(
        self,
        value: WhiteReferenceConfig | Mapping[str, Any],
    ) -> "WorkflowState":
        config = (
            value
            if isinstance(value, WhiteReferenceConfig)
            else WhiteReferenceConfig.from_mapping(value)
        )
        return replace(self, white_reference_draft=config)

    def with_white_reference_upload(
        self,
        upload: UploadIdentity | None,
    ) -> "WorkflowState":
        return replace(
            self,
            white_reference_draft=self.white_reference_draft.with_reference(upload),
        )

    def with_calibration(
        self,
        value: CalibrationConfig | Mapping[str, Any],
    ) -> "WorkflowState":
        config = (
            value
            if isinstance(value, CalibrationConfig)
            else CalibrationConfig.from_mapping(value)
        )
        return replace(self, calibration_draft=config)

    def with_baseline(
        self,
        value: BaselineConfig | Mapping[str, Any],
    ) -> "WorkflowState":
        config = value if isinstance(value, BaselineConfig) else BaselineConfig.from_mapping(value)
        return replace(self, baseline_draft=config)

    def with_smoothing(
        self,
        value: SmoothingConfig | Mapping[str, Any] | None,
    ) -> "WorkflowState":
        config = (
            value
            if isinstance(value, SmoothingConfig)
            else SmoothingConfig.from_mapping(value)
        )
        return replace(self, smoothing_draft=config)

    def with_matching(
        self,
        value: MatchingConfig | Mapping[str, Any],
    ) -> "WorkflowState":
        config = value if isinstance(value, MatchingConfig) else MatchingConfig.from_mapping(value)
        return replace(self, matching_draft=config)

    def with_display_viewport(
        self,
        value: SpectralRange | Sequence[float] | Mapping[str, Any] | None,
    ) -> "WorkflowState":
        viewport = None if value is None else SpectralRange.from_value(value)
        return replace(self, display_viewport=viewport)

    def _prospective_input(self) -> InputApproval:
        if self.measurement is None:
            raise WorkflowValidationError("upload a measurement before applying input settings")
        return InputApproval(
            measurement=self.measurement,
            white_reference=self.white_reference_draft,
            calibration=self.calibration_draft,
        )

    @property
    def input_dirty(self) -> bool:
        if self.input_approval is None:
            return True
        try:
            return self._prospective_input().signature != self.input_approval.signature
        except WorkflowValidationError:
            return True

    @property
    def baseline_dirty(self) -> bool:
        if self.input_dirty or self.input_approval is None or self.baseline_approval is None:
            return True
        candidate = BaselineApproval(self.input_approval.signature, self.baseline_draft)
        return candidate.signature != self.baseline_approval.signature

    @property
    def smoothing_dirty(self) -> bool:
        if self.baseline_dirty or self.baseline_approval is None or self.smoothing_approval is None:
            return True
        candidate = SmoothingApproval(self.baseline_approval.signature, self.smoothing_draft)
        return candidate.signature != self.smoothing_approval.signature

    @property
    def matching_dirty(self) -> bool:
        if (
            self.smoothing_dirty
            or self.smoothing_approval is None
            or self.matching_approval is None
        ):
            return True
        if not self.matching_draft.is_ready:
            return True
        candidate = MatchingApproval(self.smoothing_approval.signature, self.matching_draft)
        return candidate.signature != self.matching_approval.signature

    @property
    def has_current_result(self) -> bool:
        expected = self.expected_result_identity
        return expected is not None and self.result_identity == expected

    @property
    def has_recorded_result(self) -> bool:
        return self.result_identity is not None

    @property
    def result_is_stale(self) -> bool:
        return self.result_identity is not None and not self.has_current_result

    @property
    def expected_result_identity(self) -> ResultIdentity | None:
        if self.matching_approval is None or self.matching_dirty:
            return None
        return ResultIdentity.from_matching_approval(self.matching_approval)

    @property
    def active_result_signature(self) -> str | None:
        return self.result_identity.signature if self.has_current_result else None

    @property
    def next_required_stage(self) -> Literal[
        "measurement", "input", "baseline", "smoothing", "matching", "complete"
    ]:
        if self.measurement is None:
            return "measurement"
        if self.input_dirty:
            return "input"
        if self.baseline_dirty:
            return "baseline"
        if self.smoothing_dirty:
            return "smoothing"
        if self.matching_dirty:
            return "matching"
        if not self.has_current_result:
            return "matching"
        return "complete"

    def record_result(
        self,
        result: ResultIdentity | PrimaryResultSnapshot | None = None,
    ) -> "WorkflowState":
        """Record a successfully completed search for the current approval.

        An empty result list is still a successful scientific result; callers
        therefore record completion independently of result count.
        """

        expected = self.expected_result_identity
        if expected is None:
            raise WorkflowOrderError(
                "apply all current matching settings before recording results"
            )
        identity = (
            expected
            if result is None
            else result.identity
            if isinstance(result, PrimaryResultSnapshot)
            else result
        )
        if identity != expected:
            raise WorkflowValidationError(
                "computed result identity does not match the current matching approval"
            )
        if identity == self.result_identity:
            return self
        return replace(self, result_identity=identity)

    def clear_result(self) -> "WorkflowState":
        if self.result_identity is None:
            return self
        return replace(self, result_identity=None)

    def apply_input(self) -> "WorkflowState":
        approval = self._prospective_input()
        if (
            self.input_approval is not None
            and approval.signature == self.input_approval.signature
        ):
            return self
        return replace(
            self,
            input_approval=approval,
            baseline_approval=None,
            smoothing_approval=None,
            matching_approval=None,
        )

    def apply_white_reference_and_calibration(self) -> "WorkflowState":
        """Alias matching the app's first guided approval step."""

        return self.apply_input()

    def apply_baseline(self) -> "WorkflowState":
        if self.input_approval is None or self.input_dirty:
            raise WorkflowOrderError("apply current input/calibration settings first")
        approval = BaselineApproval(self.input_approval.signature, self.baseline_draft)
        if (
            self.baseline_approval is not None
            and approval.signature == self.baseline_approval.signature
        ):
            return self
        return replace(
            self,
            baseline_approval=approval,
            smoothing_approval=None,
            matching_approval=None,
        )

    def apply_smoothing(self) -> "WorkflowState":
        if self.baseline_approval is None or self.baseline_dirty:
            raise WorkflowOrderError("apply the current baseline settings first")
        approval = SmoothingApproval(
            self.baseline_approval.signature,
            self.smoothing_draft,
        )
        if (
            self.smoothing_approval is not None
            and approval.signature == self.smoothing_approval.signature
        ):
            return self
        return replace(self, smoothing_approval=approval, matching_approval=None)

    def apply_matching(self) -> "WorkflowState":
        if self.smoothing_approval is None or self.smoothing_dirty:
            raise WorkflowOrderError("apply the current smoothing settings first")
        approval = MatchingApproval(
            self.smoothing_approval.signature,
            self.matching_draft,
        )
        if (
            self.matching_approval is not None
            and approval.signature == self.matching_approval.signature
        ):
            return self
        return replace(self, matching_approval=approval)

    def invalidate_downstream(self, stage: WorkflowStage) -> "WorkflowState":
        """Keep ``stage`` applied but explicitly clear all applied stages below it."""

        if stage == "input":
            return replace(
                self,
                baseline_approval=None,
                smoothing_approval=None,
                matching_approval=None,
            )
        if stage == "baseline":
            return replace(self, smoothing_approval=None, matching_approval=None)
        if stage == "smoothing":
            return replace(self, matching_approval=None)
        if stage == "matching":
            return self
        raise ValueError(f"unknown workflow stage: {stage!r}")

    def invalidate_from(self, stage: WorkflowStage) -> "WorkflowState":
        """Clear ``stage`` and every applied stage below it."""

        if stage == "input":
            return replace(
                self,
                input_approval=None,
                baseline_approval=None,
                smoothing_approval=None,
                matching_approval=None,
            )
        if stage == "baseline":
            return replace(
                self,
                baseline_approval=None,
                smoothing_approval=None,
                matching_approval=None,
            )
        if stage == "smoothing":
            return replace(self, smoothing_approval=None, matching_approval=None)
        if stage == "matching":
            return replace(self, matching_approval=None)
        raise ValueError(f"unknown workflow stage: {stage!r}")


__all__ = [
    "AIDenoiserSpec",
    "BaselineApproval",
    "BaselineConfig",
    "CalibrationConfig",
    "InputApproval",
    "MatchingApproval",
    "MatchingConfig",
    "PREPROCESS_GRID_STEP_CM1",
    "PrimaryResultSnapshot",
    "ResidualReferenceIdentity",
    "ResidualResultIdentity",
    "ResidualResultSnapshot",
    "ResultIdentity",
    "SmoothingApproval",
    "SmoothingConfig",
    "SpectralRange",
    "UploadIdentity",
    "WhiteReferenceConfig",
    "WorkflowError",
    "WorkflowOrderError",
    "WorkflowState",
    "WorkflowValidationError",
    "canonical_json",
    "payload_signature",
    "payload_token",
    "residual_query_content_sha256",
]

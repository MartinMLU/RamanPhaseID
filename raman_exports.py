"""Typed export helpers for RamanPhaseID.

This module deliberately has no Streamlit dependency.  Export generation can
therefore be cached, tested, or run from the command line without importing the
web application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_NUMERIC_XY_RE = re.compile(
    r"^\s*([+-]?\d+(?:[.,]\d+)?)(?:[,\t; ]+)([+-]?\d+(?:[.,]\d+)?)\s*$"
)


@dataclass(frozen=True, slots=True)
class SpectrumExport:
    """A processed spectrum and the provenance needed to reproduce it."""

    axis_cm1: np.ndarray
    intensity: np.ndarray
    label: str
    processing_note: str = ""
    source_name: str = ""
    source_sha256: str = ""

    def __post_init__(self) -> None:
        axis = np.array(self.axis_cm1, dtype=float, copy=True).reshape(-1)
        intensity = np.array(self.intensity, dtype=float, copy=True).reshape(-1)
        if axis.size != intensity.size:
            raise ValueError("axis_cm1 and intensity must have equal length")
        axis.setflags(write=False)
        intensity.setflags(write=False)
        object.__setattr__(self, "axis_cm1", axis)
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True, slots=True)
class SpectrumTextLayout:
    """Header-preserving view of the original two-column export body."""

    header_lines: tuple[str, ...]
    axis: np.ndarray
    intensity: np.ndarray
    delimiter_hint: str
    exact_body_available: bool

    def __post_init__(self) -> None:
        axis = np.array(self.axis, dtype=float, copy=True).reshape(-1)
        intensity = np.array(self.intensity, dtype=float, copy=True).reshape(-1)
        if axis.size != intensity.size:
            raise ValueError("export axis and intensity must have equal length")
        delimiter = str(self.delimiter_hint)
        if not delimiter:
            raise ValueError("delimiter_hint must not be empty")
        axis.setflags(write=False)
        intensity.setflags(write=False)
        object.__setattr__(self, "header_lines", tuple(str(line) for line in self.header_lines))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "delimiter_hint", delimiter)
        object.__setattr__(self, "exact_body_available", bool(self.exact_body_available))


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Portable audit record for one matching run.

    ``settings`` and ``results`` are intentionally JSON-shaped mappings so the
    manifest remains forwards compatible when controls or evidence fields are
    added.
    """

    app_version: str
    app_commit: str
    measurement_name: str
    measurement_sha256: str
    database_signature: str
    settings: Mapping[str, Any]
    results: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    package_versions: Mapping[str, str] = field(default_factory=dict)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: int = 1

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _json_safe(payload)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def installed_package_versions(
    packages: Mapping[str, str] | Sequence[str],
) -> dict[str, str]:
    """Return deterministic distribution versions without importing packages."""

    requested = (
        dict(packages)
        if isinstance(packages, Mapping)
        else {str(package): str(package) for package in packages}
    )
    versions: dict[str, str] = {}
    for label, distribution in sorted(requested.items()):
        try:
            versions[str(label)] = importlib_metadata.version(str(distribution))
        except importlib_metadata.PackageNotFoundError:
            versions[str(label)] = "not-installed"
    return versions


def resolve_git_commit(project_root: str | Path) -> str:
    """Resolve the current commit read-only, including linked Git worktrees."""

    root = Path(project_root).resolve()
    git_entry = root / ".git"
    if git_entry.is_file():
        try:
            marker = git_entry.read_text(encoding="utf-8").strip()
        except OSError:
            return "unknown"
        if not marker.lower().startswith("gitdir:"):
            return "unknown"
        git_dir = Path(marker.split(":", 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = (root / git_dir).resolve()
    elif git_entry.is_dir():
        git_dir = git_entry
    else:
        return "unknown"

    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        return head.lower()
    if not head.startswith("ref:"):
        return "unknown"
    reference = head.split(":", 1)[1].strip()
    try:
        commit = (git_dir / reference).read_text(encoding="utf-8").strip()
    except OSError:
        commit = ""
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        return commit.lower()
    try:
        packed_lines = (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unknown"
    for line in packed_lines:
        if not line or line.startswith(("#", "^")):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == reference:
            packed_commit = parts[0].strip()
            if re.fullmatch(r"[0-9a-fA-F]{40,64}", packed_commit):
                return packed_commit.lower()
    return "unknown"


def split_header_data(text: str) -> tuple[list[str], list[str], str]:
    """Split a simple two-column spectrum while preserving header lines."""

    header: list[str] = []
    data: list[str] = []
    delimiter_hint: str | None = None
    in_data = False
    for line in text.splitlines():
        match = _NUMERIC_XY_RE.match(line)
        if match:
            if not in_data:
                if "," in line and "." not in line:
                    separators = re.findall(r"[,\t; ]", line)
                    delimiter_hint = separators[0] if separators else "\t"
                else:
                    parts = re.split(r"([,\t; ]+)", line.strip())
                    delimiter_hint = parts[1] if len(parts) > 2 else "\t"
                in_data = True
            data.append(line)
        else:
            header.append(line)
    return header, data, delimiter_hint or "\t"


def parse_xy_from_data_lines(data_lines: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for line in data_lines:
        match = _NUMERIC_XY_RE.match(line)
        if not match:
            continue
        xs.append(float(match.group(1).replace(",", ".")))
        ys.append(float(match.group(2).replace(",", ".")))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def inspect_spectrum_text(text: str) -> SpectrumTextLayout:
    """Inspect an original file once for header-preserving processed exports."""

    header, data, delimiter = split_header_data(str(text))
    axis, intensity = parse_xy_from_data_lines(data)
    exact = bool(axis.size > 0 and axis.size == intensity.size)
    return SpectrumTextLayout(
        header_lines=tuple(header),
        axis=axis if exact else np.array([], dtype=float),
        intensity=intensity if exact else np.array([], dtype=float),
        delimiter_hint=delimiter,
        exact_body_available=exact,
    )


def rebuild_spectrum_bytes(
    header_lines: Sequence[str],
    axis: np.ndarray,
    intensity: np.ndarray,
    *,
    decimals: int = 6,
    delimiter: str = "\t",
    keep_header_exact: bool = True,
    processing_note: str | None = None,
    extra_note: str | None = None,
) -> bytes:
    """Serialize a processed spectrum without silently losing provenance."""

    axis_arr = np.asarray(axis, dtype=float).reshape(-1)
    intensity_arr = np.asarray(intensity, dtype=float).reshape(-1)
    if axis_arr.size != intensity_arr.size:
        raise ValueError("axis and intensity must have equal length")
    if decimals < 0:
        raise ValueError("decimals must be non-negative")

    if processing_note and extra_note and processing_note != extra_note:
        raise ValueError("processing_note and legacy extra_note disagree")
    note = processing_note or extra_note
    out = io.StringIO()
    for line in header_lines:
        out.write(str(line) + "\n")
    if note and not keep_header_exact:
        out.write(f"# {note}\n")
    number_format = f"{{:.{int(decimals)}f}}"
    for shift, value in zip(axis_arr, intensity_arr):
        out.write(number_format.format(float(shift)))
        out.write(delimiter)
        out.write(number_format.format(float(value)))
        out.write("\n")
    return out.getvalue().encode("utf-8")


def spectrum_tsv_bytes(spectrum: SpectrumExport, *, decimals: int = 6) -> bytes:
    note = spectrum.processing_note.strip()
    header = [f"# {spectrum.label}"]
    if spectrum.source_name:
        header.append(f"# source: {spectrum.source_name}")
    if spectrum.source_sha256:
        header.append(f"# source_sha256: {spectrum.source_sha256}")
    if note:
        header.append(f"# processing: {note}")
    return rebuild_spectrum_bytes(
        header,
        spectrum.axis_cm1,
        spectrum.intensity,
        decimals=decimals,
        delimiter="\t",
        keep_header_exact=True,
    )


def manifest_json_bytes(manifest: RunManifest, *, indent: int = 2) -> bytes:
    return (
        json.dumps(
            manifest.to_json_dict(),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if hasattr(value, "as_posix"):
        return value.as_posix()
    return value


# Compatibility names used by the pre-refactor application and existing tests.
_split_header_data = split_header_data
_parse_xy_from_data_lines = parse_xy_from_data_lines
_rebuild_file_bytes = rebuild_spectrum_bytes


__all__ = [
    "RunManifest",
    "SpectrumExport",
    "SpectrumTextLayout",
    "inspect_spectrum_text",
    "installed_package_versions",
    "manifest_json_bytes",
    "parse_xy_from_data_lines",
    "rebuild_spectrum_bytes",
    "resolve_git_commit",
    "sha256_bytes",
    "spectrum_tsv_bytes",
    "split_header_data",
]

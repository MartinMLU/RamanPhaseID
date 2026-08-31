"""Typed database inventory and precompute-cache primitives for RamanPhaseID.

This module deliberately has no Streamlit dependency.  A UI can keep one
``DatabaseInventoryManager`` in ``st.cache_resource`` and call ``refresh`` only
when the user explicitly requests a database reload.  Precompute matrices are
opened read-only and HNSW indexes are ignored unless a caller opts in.

The loader accepts legacy cache directories, while current cache writers use a
validated manifest and portable ``source_root``/``source_relpath`` metadata.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Iterator, Mapping, Sequence

import numpy as np


INVENTORY_SCHEMA_VERSION = 1
CACHE_MANIFEST_SCHEMA_VERSION = 1
PAIR_COMMIT_SCHEMA_VERSION = 1
DEFAULT_VECTOR_FILE = "X.float32.npy"
DEFAULT_METADATA_FILE = "meta.json"
DEFAULT_GRID_FILE = "grid.json"
DEFAULT_MANIFEST_FILE = "manifest.json"
DEFAULT_HNSW_FILE = "ann_hnsw.bin"
DEFAULT_EXTENSIONS = (".rod", ".txt")


class DatabaseError(RuntimeError):
    """Base exception for database inventory/cache failures."""


class InventoryError(DatabaseError):
    """Raised when database roots or inventory records are invalid."""


class CacheValidationError(DatabaseError):
    """Raised when a precompute directory is incomplete or inconsistent."""


class SourceResolutionError(DatabaseError):
    """Raised when a cached reference cannot be mapped to a current source."""


class PairBuildError(DatabaseError):
    """Raised when an aligned RAW/baseline cache pair cannot be published safely."""


def _normalise_relative_path(value: str | Path) -> str:
    """Return one safe, platform-neutral relative path."""
    text = str(value).strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Expected a safe relative path, got {value!r}")
    return path.as_posix()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class DatabaseRoot:
    """One named database root.

    ``alias`` is persisted in signatures and metadata; ``path`` is deliberately
    excluded from content signatures so caches remain portable after a workspace
    move.
    """

    alias: str
    path: Path

    def __post_init__(self) -> None:
        alias = str(self.alias).strip()
        if not alias:
            raise ValueError("Database root aliases must not be empty")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve(strict=False))


DatabaseRootsInput = (
    Mapping[str, str | Path]
    | Sequence[DatabaseRoot | str | Path]
)


def coerce_database_roots(roots: DatabaseRootsInput) -> tuple[DatabaseRoot, ...]:
    """Normalise database roots and require unambiguous aliases."""
    if isinstance(roots, Mapping):
        values = [DatabaseRoot(str(alias), Path(path)) for alias, path in roots.items()]
    else:
        values = []
        for item in roots:
            if isinstance(item, DatabaseRoot):
                values.append(item)
            else:
                path = Path(item)
                values.append(DatabaseRoot(path.name, path))

    aliases: dict[str, str] = {}
    for root in values:
        folded = root.alias.casefold()
        if folded in aliases:
            raise ValueError(
                f"Duplicate database root alias {root.alias!r}; pass an alias-to-path mapping"
            )
        aliases[folded] = root.alias
    return tuple(sorted(values, key=lambda root: root.alias.casefold()))


@dataclass(frozen=True, slots=True)
class DatabaseFileRecord:
    root_alias: str
    relative_path: str
    size: int
    mtime_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _normalise_relative_path(self.relative_path))
        if self.size < 0 or self.mtime_ns < 0:
            raise ValueError("Database file size and mtime_ns must be non-negative")

    @property
    def key(self) -> tuple[str, str]:
        return self.root_alias, self.relative_path


@dataclass(frozen=True, slots=True)
class InventoryIssue:
    root_alias: str
    relative_path: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DatabaseInventory:
    roots: tuple[DatabaseRoot, ...]
    files: tuple[DatabaseFileRecord, ...]
    missing_roots: tuple[str, ...]
    issues: tuple[InventoryIssue, ...]
    signature: str
    generation: int
    scanned_at_ns: int

    @property
    def refresh_token(self) -> str:
        """Token that changes even when an explicit refresh finds no file changes."""
        return f"{self.signature}:generation:{self.generation}"

    @property
    def root_map(self) -> Mapping[str, DatabaseRoot]:
        return MappingProxyType({root.alias: root for root in self.roots})

    def resolve(self, source: "PortableSource", *, require_exists: bool = False) -> Path | None:
        return source.resolve(self.roots, require_exists=require_exists)

    def derive_signature(self, namespace: str, payload: Mapping[str, Any]) -> str:
        """Derive a deterministic cache signature from inventory plus settings."""
        body = {
            "inventory": self.signature,
            "namespace": str(namespace),
            "payload": payload,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


def _inventory_signature(
    roots: Sequence[DatabaseRoot],
    files: Sequence[DatabaseFileRecord],
    missing_roots: Sequence[str],
    issues: Sequence[InventoryIssue],
) -> str:
    digest = sha256()
    digest.update(f"raman-db-inventory-v{INVENTORY_SCHEMA_VERSION}\0".encode("ascii"))
    missing = set(missing_roots)
    for root in sorted(roots, key=lambda value: value.alias.casefold()):
        state = "missing" if root.alias in missing else "present"
        digest.update(f"root\0{root.alias}\0{state}\0".encode("utf-8"))
    for record in sorted(
        files,
        key=lambda value: (value.root_alias.casefold(), value.relative_path),
    ):
        digest.update(
            (
                f"file\0{record.root_alias}\0{record.relative_path}\0"
                f"{record.size}\0{record.mtime_ns}\0"
            ).encode("utf-8")
        )
    for issue in sorted(
        issues,
        key=lambda value: (value.root_alias.casefold(), value.relative_path, value.code),
    ):
        # Messages can contain machine-specific absolute paths.  The stable code
        # and portable relative location are sufficient to invalidate a cache.
        digest.update(
            f"issue\0{issue.root_alias}\0{issue.relative_path}\0{issue.code}\0".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def scan_database_inventory(
    roots: DatabaseRootsInput,
    *,
    generation: int = 0,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> DatabaseInventory:
    """Scan roots once and return a portable mtime-nanosecond inventory."""
    normalised_roots = coerce_database_roots(roots)
    suffixes = {
        str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in extensions
    }
    records: list[DatabaseFileRecord] = []
    missing: list[str] = []
    issues: list[InventoryIssue] = []

    for root in normalised_roots:
        if not root.path.is_dir():
            missing.append(root.alias)
            continue
        try:
            candidates = root.path.rglob("*")
            for candidate in candidates:
                if candidate.suffix.lower() not in suffixes:
                    continue
                try:
                    if not candidate.is_file():
                        continue
                    stat = candidate.stat()
                    relative = candidate.relative_to(root.path).as_posix()
                    records.append(
                        DatabaseFileRecord(
                            root_alias=root.alias,
                            relative_path=relative,
                            size=int(stat.st_size),
                            mtime_ns=int(stat.st_mtime_ns),
                        )
                    )
                except OSError as exc:
                    try:
                        relative = candidate.relative_to(root.path).as_posix()
                    except ValueError:
                        relative = candidate.name
                    issues.append(
                        InventoryIssue(
                            root_alias=root.alias,
                            relative_path=relative,
                            code=type(exc).__name__,
                            message=str(exc),
                        )
                    )
        except OSError as exc:
            issues.append(
                InventoryIssue(
                    root_alias=root.alias,
                    relative_path=".",
                    code=type(exc).__name__,
                    message=str(exc),
                )
            )

    records.sort(key=lambda value: (value.root_alias.casefold(), value.relative_path))
    issues.sort(key=lambda value: (value.root_alias.casefold(), value.relative_path, value.code))
    signature = _inventory_signature(normalised_roots, records, missing, issues)
    return DatabaseInventory(
        roots=normalised_roots,
        files=tuple(records),
        missing_roots=tuple(sorted(missing, key=str.casefold)),
        issues=tuple(issues),
        signature=signature,
        generation=max(0, int(generation)),
        scanned_at_ns=time.time_ns(),
    )


class DatabaseInventoryManager:
    """Thread-safe inventory snapshot with explicit refresh semantics."""

    def __init__(
        self,
        roots: DatabaseRootsInput,
        *,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    ) -> None:
        self._roots = coerce_database_roots(roots)
        self._extensions = tuple(extensions)
        self._generation = 0
        self._snapshot: DatabaseInventory | None = None
        self._lock = threading.RLock()

    @property
    def roots(self) -> tuple[DatabaseRoot, ...]:
        return self._roots

    def snapshot(self) -> DatabaseInventory:
        """Return the current snapshot, scanning only on the first call."""
        with self._lock:
            if self._snapshot is None:
                self._snapshot = scan_database_inventory(
                    self._roots,
                    generation=self._generation,
                    extensions=self._extensions,
                )
            return self._snapshot

    def refresh(self) -> DatabaseInventory:
        """Force a rescan and increment the UI/cache generation."""
        with self._lock:
            self._generation += 1
            self._snapshot = scan_database_inventory(
                self._roots,
                generation=self._generation,
                extensions=self._extensions,
            )
            return self._snapshot


@dataclass(frozen=True, slots=True)
class PortableSource:
    """A source path anchored to a logical database root."""

    root_alias: str = ""
    relative_path: str = ""
    legacy_path: str = ""

    def __post_init__(self) -> None:
        alias = str(self.root_alias).strip()
        relative = str(self.relative_path).strip()
        if alias:
            relative = _normalise_relative_path(relative)
        object.__setattr__(self, "root_alias", alias)
        object.__setattr__(self, "relative_path", relative)
        object.__setattr__(self, "legacy_path", str(self.legacy_path).strip())

    @property
    def is_portable(self) -> bool:
        return bool(self.root_alias and self.relative_path)

    @property
    def identity_key(self) -> tuple[str, str]:
        if self.is_portable:
            return self.root_alias.casefold(), self.relative_path
        return "legacy", self.legacy_path

    def resolve(
        self,
        roots: DatabaseRootsInput,
        *,
        require_exists: bool = False,
    ) -> Path | None:
        root_values = coerce_database_roots(roots)
        root = next(
            (
                value
                for value in root_values
                if value.alias.casefold() == self.root_alias.casefold()
            ),
            None,
        )
        candidate: Path | None = None
        if root is not None and self.relative_path:
            candidate = (root.path / Path(*PurePosixPath(self.relative_path).parts)).resolve(
                strict=False
            )
            if not _path_is_within(candidate, root.path):
                raise SourceResolutionError(
                    f"Cached source escapes database root {root.alias!r}: {self.relative_path}"
                )
        elif self.legacy_path:
            candidate = Path(self.legacy_path).expanduser().resolve(strict=False)

        if candidate is None or (require_exists and not candidate.is_file()):
            return None
        return candidate

    def as_metadata_fields(self) -> dict[str, str]:
        if self.is_portable:
            return {
                "source_root": self.root_alias,
                "source_relpath": self.relative_path,
            }
        return {"path": self.legacy_path} if self.legacy_path else {}


def portable_source_for_path(path: str | Path, roots: DatabaseRootsInput) -> PortableSource:
    """Convert a current source path to portable root-relative metadata."""
    candidate = Path(path).expanduser().resolve(strict=False)
    matches: list[tuple[int, DatabaseRoot, Path]] = []
    for root in coerce_database_roots(roots):
        try:
            relative = candidate.relative_to(root.path)
        except ValueError:
            continue
        matches.append((len(root.path.parts), root, relative))
    if not matches:
        return PortableSource(legacy_path=str(candidate))
    _, root, relative = max(matches, key=lambda value: value[0])
    return PortableSource(root.alias, relative.as_posix(), str(candidate))


@dataclass(frozen=True, slots=True)
class GridSpec:
    minimum: float
    maximum: float
    step: float
    length: int

    def __post_init__(self) -> None:
        values = (float(self.minimum), float(self.maximum), float(self.step))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Grid bounds and step must be finite")
        if self.step <= 0.0 or self.maximum < self.minimum or int(self.length) <= 0:
            raise ValueError("Grid requires max >= min, positive step, and positive length")
        expected = int(math.floor(((self.maximum - self.minimum) / self.step) + 1e-9)) + 1
        if expected != int(self.length):
            raise ValueError(
                f"Grid length {self.length} does not match min/max/step (expected {expected})"
            )
        object.__setattr__(self, "minimum", values[0])
        object.__setattr__(self, "maximum", values[1])
        object.__setattr__(self, "step", values[2])
        object.__setattr__(self, "length", int(self.length))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GridSpec":
        try:
            return cls(
                minimum=float(value["min"]),
                maximum=float(value["max"]),
                step=float(value["step"]),
                length=int(value["len"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheValidationError(f"Invalid grid metadata: {exc}") from exc

    def values(self, *, dtype: np.dtype[Any] | type = np.float32) -> np.ndarray:
        return (self.minimum + np.arange(self.length, dtype=dtype) * self.step).astype(
            dtype, copy=False
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "len": self.length,
        }


def derive_precompute_signature(
    inventory: DatabaseInventory,
    grid: GridSpec,
    *,
    variant: str,
    preprocessing: Mapping[str, Any] | str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Build a portable precompute signature from typed inputs."""
    return inventory.derive_signature(
        "raman-precompute-v1",
        {
            "grid": grid.as_dict(),
            "variant": str(variant),
            "preprocessing": preprocessing,
            "extra": dict(extra or {}),
        },
    )


@dataclass(frozen=True, slots=True)
class CachedProvenance(Mapping[str, Any]):
    """Small normalized provenance payload persisted with each cache identity."""

    _KEYS: ClassVar[tuple[str, ...]] = (
        "database",
        "accession",
        "source",
        "status",
        "quality",
        "quality_folder",
        "processing",
        "determination",
        "orientation",
        "orientation_detail",
        "excitation_wavelength_nm",
        "resolution_cm1",
        "measured_chemistry",
        "correction_history",
    )

    database: str = ""
    accession: str = ""
    source: str = ""
    status: str = ""
    quality: str = "unknown"
    quality_folder: str = ""
    processing: str = "unknown"
    determination: str = "unknown"
    orientation: str = "unknown"
    orientation_detail: str = ""
    excitation_wavelength_nm: float | None = None
    resolution_cm1: float | None = None
    measured_chemistry: str = ""
    correction_history: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CachedProvenance":
        payload = value or {}

        def optional_float(key: str) -> float | None:
            raw = payload.get(key)
            if raw is None or raw == "":
                return None
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"provenance {key} must be finite")
            return number

        history = payload.get("correction_history", ())
        if not isinstance(history, (list, tuple)):
            raise ValueError("provenance correction_history must be a list")
        return cls(
            database=str(payload.get("database", "")),
            accession=str(payload.get("accession", "")),
            source=str(payload.get("source", "")),
            status=str(payload.get("status", "")),
            quality=str(payload.get("quality", "unknown")),
            quality_folder=str(payload.get("quality_folder", "")),
            processing=str(payload.get("processing", "unknown")),
            determination=str(payload.get("determination", "unknown")),
            orientation=str(payload.get("orientation", "unknown")),
            orientation_detail=str(payload.get("orientation_detail", "")),
            excitation_wavelength_nm=optional_float("excitation_wavelength_nm"),
            resolution_cm1=optional_float("resolution_cm1"),
            measured_chemistry=str(payload.get("measured_chemistry", "")),
            correction_history=tuple(str(item) for item in history),
        )

    @classmethod
    def from_spectrum(cls, value: "SpectrumProvenance") -> "CachedProvenance":
        """Compress a parsed source header into the fields persisted per row."""

        return cls(
            database=value.database,
            accession=value.accession,
            source=value.source,
            status=value.status,
            quality=value.quality,
            quality_folder=value.quality_folder,
            processing=value.processing,
            determination=value.determination,
            orientation=value.orientation,
            orientation_detail=value.orientation_detail,
            excitation_wavelength_nm=value.excitation_wavelength_nm,
            resolution_cm1=value.resolution_cm1,
            measured_chemistry=value.chemistry.measured,
            correction_history=value.correction_history,
        )

    def __getitem__(self, key: str) -> Any:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    @property
    def available(self) -> bool:
        return bool(self.database or self.accession or self.source)

    def as_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "accession": self.accession,
            "source": self.source,
            "status": self.status,
            "quality": self.quality,
            "quality_folder": self.quality_folder,
            "processing": self.processing,
            "determination": self.determination,
            "orientation": self.orientation,
            "orientation_detail": self.orientation_detail,
            "excitation_wavelength_nm": self.excitation_wavelength_nm,
            "resolution_cm1": self.resolution_cm1,
            "measured_chemistry": self.measured_chemistry,
            "correction_history": list(self.correction_history),
        }


@dataclass(frozen=True, slots=True)
class ReferenceIdentity:
    name: str
    formula: str
    flag: str
    filename: str
    orig_filename: str
    source: PortableSource
    elements: tuple[str, ...]
    has_formula: bool
    parser_version: int | None
    source_label: str = ""
    provenance: CachedProvenance = field(default_factory=CachedProvenance)

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.name,
            self.formula,
            self.flag,
            self.filename,
            self.orig_filename,
            self.source.identity_key,
            self.elements,
            self.has_formula,
            self.parser_version,
            self.source_label,
            self.provenance,
        )


@dataclass(frozen=True, slots=True)
class ReferenceVectorStats:
    start_idx: int
    end_idx: int
    l2: float
    db_baseline: bool
    support_runs: tuple[tuple[int, int], ...] = ()
    error: str = ""


@dataclass(frozen=True, slots=True)
class MetadataIndex:
    entries: tuple[ReferenceIdentity, ...]
    by_element: Mapping[str, tuple[int, ...]] = field(compare=False, repr=False)

    @classmethod
    def build(cls, entries: Sequence[ReferenceIdentity]) -> "MetadataIndex":
        frozen_entries = tuple(entries)
        inverse: dict[str, list[int]] = {}
        for index, entry in enumerate(frozen_entries):
            for element in entry.elements:
                inverse.setdefault(element, []).append(index)
        immutable = MappingProxyType(
            {element: tuple(indices) for element, indices in sorted(inverse.items())}
        )
        return cls(entries=frozen_entries, by_element=immutable)


_ELIGIBILITY_VARIANT_ALIASES = {
    "raw": "raw",
    "db-raw": "raw",
    "library_as_provided": "raw",
    "library-as-provided": "raw",
    "baseline": "baseline_corrected",
    "baseline_corrected": "baseline_corrected",
    "baseline-corrected": "baseline_corrected",
    "db-bc": "baseline_corrected",
}
_ELEMENT_MODE_ALIASES = {
    "must include all": "must_include_all",
    "must_include_all": "must_include_all",
    "only from this list": "only_from_list",
    "only_from_list": "only_from_list",
    "exactly this set": "exact_set",
    "exact_set": "exact_set",
}


def _normalise_element_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


@dataclass(frozen=True, slots=True)
class ReferenceEligibilityRequest:
    """Complete cache identity for one library-eligibility calculation.

    Both pair signatures are retained even though one request evaluates only
    one variant.  This prevents a valid row-id array from being reused with a
    different aligned partner pack after a cache migration or repair.
    """

    raw_signature: str
    baseline_signature: str
    library_variant: str
    include_elements: tuple[str, ...] = ()
    exclude_elements: tuple[str, ...] = ()
    element_mode: str = "must_include_all"
    allow_missing_formula: bool = True
    filtering_policy_version: int = 1

    def __post_init__(self) -> None:
        raw_signature = str(self.raw_signature).strip()
        baseline_signature = str(self.baseline_signature).strip()
        if not raw_signature or not baseline_signature:
            raise ValueError("eligibility requests require both cache-pair signatures")
        variant_key = str(self.library_variant).strip().casefold()
        try:
            variant = _ELIGIBILITY_VARIANT_ALIASES[variant_key]
        except KeyError as exc:
            raise ValueError(f"unknown eligibility library variant: {self.library_variant!r}") from exc
        mode_key = str(self.element_mode).strip().casefold()
        try:
            mode = _ELEMENT_MODE_ALIASES[mode_key]
        except KeyError as exc:
            raise ValueError(f"unknown element-filter mode: {self.element_mode!r}") from exc
        policy_version = int(self.filtering_policy_version)
        if policy_version < 1:
            raise ValueError("filtering policy version must be positive")
        object.__setattr__(self, "raw_signature", raw_signature)
        object.__setattr__(self, "baseline_signature", baseline_signature)
        object.__setattr__(self, "library_variant", variant)
        object.__setattr__(
            self,
            "include_elements",
            _normalise_element_tuple(self.include_elements),
        )
        object.__setattr__(
            self,
            "exclude_elements",
            _normalise_element_tuple(self.exclude_elements),
        )
        object.__setattr__(self, "element_mode", mode)
        object.__setattr__(self, "allow_missing_formula", bool(self.allow_missing_formula))
        object.__setattr__(self, "filtering_policy_version", policy_version)

    @property
    def library_signature(self) -> str:
        return (
            self.raw_signature
            if self.library_variant == "raw"
            else self.baseline_signature
        )


@dataclass(frozen=True, slots=True)
class ReferenceEligibilityResult:
    """Immutable eligible row ids bound to their complete request identity."""

    request: ReferenceEligibilityRequest
    row_ids: np.ndarray = field(compare=False, repr=False)
    scanned_count: int = 0

    def __post_init__(self) -> None:
        row_ids = np.array(self.row_ids, dtype=np.int32, copy=True).reshape(-1)
        if np.any(row_ids < 0):
            raise ValueError("eligible reference row ids must be non-negative")
        if row_ids.size > 1 and np.any(np.diff(row_ids.astype(np.int64)) <= 0):
            raise ValueError("eligible reference row ids must be strictly increasing")
        scanned_count = int(self.scanned_count)
        if scanned_count < row_ids.size:
            raise ValueError("eligibility scan count cannot be smaller than its result")
        row_ids.setflags(write=False)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "scanned_count", scanned_count)

    @property
    def eligible_count(self) -> int:
        return int(self.row_ids.size)


ReferenceProcessedPredicate = Callable[[Mapping[str, Any]], bool]


def _element_policy_allows(
    metadata: Mapping[str, Any],
    request: ReferenceEligibilityRequest,
    included: frozenset[str],
    excluded: frozenset[str],
) -> bool:
    elements = {str(value) for value in metadata.get("elements", ())}
    if not bool(metadata.get("has_formula", False)):
        return request.allow_missing_formula
    if excluded and elements & excluded:
        return False
    if not included:
        return True
    if request.element_mode == "must_include_all":
        return included.issubset(elements)
    if request.element_mode == "only_from_list":
        return elements.issubset(included)
    return elements == included


def compute_reference_eligibility(
    metadata: Sequence[Mapping[str, Any]],
    request: ReferenceEligibilityRequest,
    *,
    is_already_processed: ReferenceProcessedPredicate,
) -> ReferenceEligibilityResult:
    """Evaluate one pack once; callers may cache the typed result by request.

    All configured reference spectra are eligible before explicit chemistry
    filters are applied. The baseline-corrected variant excludes references
    already supplied as processed spectra because those rows remain available
    unchanged through the library-as-provided variant.
    """

    eligible: list[int] = []
    baseline_variant = request.library_variant == "baseline_corrected"
    included = frozenset(request.include_elements)
    excluded = frozenset(request.exclude_elements)
    for row_id, row in enumerate(metadata):
        if not isinstance(row, Mapping):
            raise TypeError(f"reference metadata row {row_id} is not a mapping")
        if not _element_policy_allows(row, request, included, excluded):
            continue
        if baseline_variant and is_already_processed(row):
            continue
        eligible.append(row_id)
    return ReferenceEligibilityResult(
        request=request,
        row_ids=np.asarray(eligible, dtype=np.int32),
        scanned_count=len(metadata),
    )


@dataclass(frozen=True, slots=True)
class CacheManifest:
    signature: str
    inventory_signature: str
    variant: str
    grid: GridSpec
    metadata_rows: int
    vector_bytes: int
    created_at_ns: int
    complete: bool = True
    schema_version: int = CACHE_MANIFEST_SCHEMA_VERSION
    vector_dtype: str = "float32"
    vector_file: str = DEFAULT_VECTOR_FILE
    metadata_file: str = DEFAULT_METADATA_FILE
    grid_file: str = DEFAULT_GRID_FILE

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cache manifest schema {self.schema_version}")
        if not self.signature:
            raise ValueError("Cache manifest signature must not be empty")
        if self.metadata_rows < 0 or self.vector_bytes < 0 or self.created_at_ns < 0:
            raise ValueError("Manifest counts and timestamps must be non-negative")
        for name in (self.vector_file, self.metadata_file, self.grid_file):
            if Path(name).name != name:
                raise ValueError(f"Manifest cache filenames must be simple names: {name!r}")
        try:
            dtype = np.dtype(self.vector_dtype)
        except TypeError as exc:
            raise ValueError(f"Unsupported vector dtype {self.vector_dtype!r}") from exc
        if dtype.hasobject:
            raise ValueError("Object dtypes are not valid precompute vectors")

    @classmethod
    def create(
        cls,
        *,
        signature: str,
        inventory_signature: str,
        variant: str,
        grid: GridSpec,
        metadata_rows: int,
        vector_bytes: int,
        vector_dtype: str = "float32",
    ) -> "CacheManifest":
        return cls(
            signature=signature,
            inventory_signature=inventory_signature,
            variant=variant,
            grid=grid,
            metadata_rows=int(metadata_rows),
            vector_bytes=int(vector_bytes),
            created_at_ns=time.time_ns(),
            vector_dtype=vector_dtype,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CacheManifest":
        try:
            grid_value = value["grid"]
            if not isinstance(grid_value, Mapping):
                raise TypeError("manifest grid must be an object")
            return cls(
                schema_version=int(value.get("schema_version", CACHE_MANIFEST_SCHEMA_VERSION)),
                signature=str(value["signature"]),
                inventory_signature=str(value.get("inventory_signature", "")),
                variant=str(value.get("variant", "")),
                grid=GridSpec.from_mapping(grid_value),
                metadata_rows=int(value["metadata_rows"]),
                vector_bytes=int(value["vector_bytes"]),
                created_at_ns=int(value.get("created_at_ns", 0)),
                complete=bool(value.get("complete", False)),
                vector_dtype=str(value.get("vector_dtype", "float32")),
                vector_file=str(value.get("vector_file", DEFAULT_VECTOR_FILE)),
                metadata_file=str(value.get("metadata_file", DEFAULT_METADATA_FILE)),
                grid_file=str(value.get("grid_file", DEFAULT_GRID_FILE)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheValidationError(f"Invalid cache manifest: {exc}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "signature": self.signature,
            "inventory_signature": self.inventory_signature,
            "variant": self.variant,
            "grid": self.grid.as_dict(),
            "metadata_rows": self.metadata_rows,
            "vector_bytes": self.vector_bytes,
            "created_at_ns": self.created_at_ns,
            "complete": self.complete,
            "vector_dtype": self.vector_dtype,
            "vector_file": self.vector_file,
            "metadata_file": self.metadata_file,
            "grid_file": self.grid_file,
        }


@dataclass(frozen=True, slots=True)
class PairCommitManifest:
    """Last-written marker proving that two aligned packs were published together."""

    raw_signature: str
    baseline_signature: str
    inventory_signature: str
    grid: GridSpec
    row_count: int
    valid_rows: int
    failed_rows: int
    failure_counts: tuple[tuple[str, int], ...]
    created_at_ns: int
    complete: bool = True
    schema_version: int = PAIR_COMMIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for signature in (self.raw_signature, self.baseline_signature):
            if not signature or Path(signature).name != signature:
                raise ValueError(f"Pair signatures must be simple names: {signature!r}")
        if self.schema_version != PAIR_COMMIT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported pair commit schema {self.schema_version}")
        if self.row_count <= 0 or self.valid_rows <= 0 or self.failed_rows < 0:
            raise ValueError("A committed pair requires at least one valid row")
        if self.valid_rows + self.failed_rows != self.row_count:
            raise ValueError("Pair valid/failed counts do not add up to row_count")
        if self.created_at_ns < 0:
            raise ValueError("Pair commit timestamp must be non-negative")
        failure_total = 0
        seen: set[str] = set()
        for name, count in self.failure_counts:
            normalized = str(name).strip()
            if not normalized or normalized in seen or int(count) <= 0:
                raise ValueError("Pair failure counts require unique names and positive counts")
            seen.add(normalized)
            failure_total += int(count)
        if failure_total != self.failed_rows:
            raise ValueError("Pair failure counts do not add up to failed_rows")

    @classmethod
    def create(
        cls,
        *,
        raw_signature: str,
        baseline_signature: str,
        inventory_signature: str,
        grid: GridSpec,
        row_count: int,
        valid_rows: int,
        failure_counts: Mapping[str, int] | None = None,
    ) -> "PairCommitManifest":
        failures = tuple(
            sorted(
                (
                    (str(name), int(count))
                    for name, count in (failure_counts or {}).items()
                    if int(count) > 0
                ),
                key=lambda item: item[0],
            )
        )
        failed_rows = sum(count for _, count in failures)
        return cls(
            raw_signature=raw_signature,
            baseline_signature=baseline_signature,
            inventory_signature=inventory_signature,
            grid=grid,
            row_count=int(row_count),
            valid_rows=int(valid_rows),
            failed_rows=failed_rows,
            failure_counts=failures,
            created_at_ns=time.time_ns(),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PairCommitManifest":
        try:
            grid_value = value["grid"]
            if not isinstance(grid_value, Mapping):
                raise TypeError("pair commit grid must be an object")
            counts_value = value.get("failure_counts", {})
            if not isinstance(counts_value, Mapping):
                raise TypeError("pair failure_counts must be an object")
            return cls(
                raw_signature=str(value["raw_signature"]),
                baseline_signature=str(value["baseline_signature"]),
                inventory_signature=str(value.get("inventory_signature", "")),
                grid=GridSpec.from_mapping(grid_value),
                row_count=int(value["row_count"]),
                valid_rows=int(value["valid_rows"]),
                failed_rows=int(value.get("failed_rows", 0)),
                failure_counts=tuple(
                    sorted(
                        ((str(name), int(count)) for name, count in counts_value.items()),
                        key=lambda item: item[0],
                    )
                ),
                created_at_ns=int(value.get("created_at_ns", 0)),
                complete=bool(value.get("complete", False)),
                schema_version=int(value.get("schema_version", PAIR_COMMIT_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheValidationError(f"Invalid pair commit manifest: {exc}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "raw_signature": self.raw_signature,
            "baseline_signature": self.baseline_signature,
            "inventory_signature": self.inventory_signature,
            "grid": self.grid.as_dict(),
            "row_count": self.row_count,
            "valid_rows": self.valid_rows,
            "failed_rows": self.failed_rows,
            "failure_counts": dict(self.failure_counts),
            "created_at_ns": self.created_at_ns,
            "complete": self.complete,
        }


def _pair_artifact_key(raw_signature: str, baseline_signature: str) -> str:
    for signature in (raw_signature, baseline_signature):
        if not signature or Path(signature).name != signature:
            raise ValueError(f"Pair signatures must be simple names: {signature!r}")
    encoded = json.dumps(
        [raw_signature, baseline_signature], separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:32]


def pair_commit_path(
    cache_root: str | Path,
    raw_signature: str,
    baseline_signature: str,
) -> Path:
    root = Path(cache_root).expanduser().resolve(strict=False)
    key = _pair_artifact_key(raw_signature, baseline_signature)
    return root / f".pair-{key}.commit.json"


def pair_lock_path(
    cache_root: str | Path,
    raw_signature: str,
    baseline_signature: str,
) -> Path:
    root = Path(cache_root).expanduser().resolve(strict=False)
    key = _pair_artifact_key(raw_signature, baseline_signature)
    return root / f".pair-{key}.lock"


def write_pair_commit(
    cache_root: str | Path,
    commit: PairCommitManifest,
) -> Path:
    """Atomically publish the last-written logical commit for a cache pair."""
    root = Path(cache_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    target = pair_commit_path(root, commit.raw_signature, commit.baseline_signature)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(commit.as_dict(), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(root)
        return target
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def read_pair_commit(
    cache_root: str | Path,
    raw_signature: str,
    baseline_signature: str,
    *,
    required: bool = False,
) -> PairCommitManifest | None:
    path = pair_commit_path(cache_root, raw_signature, baseline_signature)
    if not path.is_file():
        if required:
            raise CacheValidationError(f"Pair commit marker is missing: {path}")
        return None
    value = _read_json(path, "pair commit marker")
    if not isinstance(value, Mapping):
        raise CacheValidationError(f"Pair commit marker must be a JSON object: {path}")
    commit = PairCommitManifest.from_mapping(value)
    if (
        commit.raw_signature != raw_signature
        or commit.baseline_signature != baseline_signature
    ):
        raise CacheValidationError("Pair commit signatures do not match its filename key")
    if not commit.complete:
        raise CacheValidationError("Pair commit marker is not complete")
    return commit


_PAIR_THREAD_LOCKS_GUARD = threading.Lock()
_PAIR_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _pair_thread_lock(lock_path: Path) -> threading.RLock:
    key = str(lock_path)
    with _PAIR_THREAD_LOCKS_GUARD:
        return _PAIR_THREAD_LOCKS.setdefault(key, threading.RLock())


def _acquire_file_lock(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on packaged Windows builds
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_file_lock(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on packaged Windows builds
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def pair_build_lock(
    cache_root: str | Path,
    raw_signature: str,
    baseline_signature: str,
) -> Iterator[Path]:
    """Serialize one pair build across Streamlit threads and local processes."""
    path = pair_lock_path(cache_root, raw_signature, baseline_signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _pair_thread_lock(path):
        with path.open("a+b") as handle:
            _acquire_file_lock(handle)
            try:
                yield path
            finally:
                _release_file_lock(handle)


def _fsync_directory(directory: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_cache_manifest(
    directory: str | Path,
    manifest: CacheManifest,
    *,
    filename: str = DEFAULT_MANIFEST_FILE,
) -> Path:
    """Atomically write a cache-completion manifest."""
    cache_dir = Path(directory)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if Path(filename).name != filename:
        raise ValueError("Manifest filename must be a simple filename")
    target = cache_dir / filename
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{filename}.",
            suffix=".tmp",
            dir=cache_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest.as_dict(), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(cache_dir)
        return target
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def read_cache_manifest(
    directory: str | Path,
    *,
    required: bool = False,
    filename: str = DEFAULT_MANIFEST_FILE,
) -> CacheManifest | None:
    path = Path(directory) / filename
    if not path.is_file():
        if required:
            raise CacheValidationError(f"Cache manifest is missing: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"Could not read cache manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CacheValidationError(f"Cache manifest must be a JSON object: {path}")
    return CacheManifest.from_mapping(value)


def _validate_manifest_files(directory: Path, manifest: CacheManifest) -> None:
    """Validate a completed manifest without opening source databases."""
    vector_path = directory / manifest.vector_file
    metadata_path = directory / manifest.metadata_file
    grid_path = directory / manifest.grid_file
    for path, label in (
        (vector_path, "vector file"),
        (metadata_path, "metadata file"),
        (grid_path, "grid file"),
    ):
        if not path.is_file():
            raise CacheValidationError(f"Manifest {label} is missing: {path}")

    grid_value = _read_json(grid_path, "grid metadata")
    if not isinstance(grid_value, Mapping) or GridSpec.from_mapping(grid_value) != manifest.grid:
        raise CacheValidationError("Manifest grid does not match grid.json")
    metadata_value = _read_json(metadata_path, "reference metadata")
    if not isinstance(metadata_value, list) or len(metadata_value) != manifest.metadata_rows:
        raise CacheValidationError("Manifest metadata row count does not match metadata file")
    dtype = np.dtype(manifest.vector_dtype)
    expected_bytes = manifest.metadata_rows * manifest.grid.length * dtype.itemsize
    actual_bytes = vector_path.stat().st_size
    if manifest.vector_bytes != expected_bytes or actual_bytes != expected_bytes:
        raise CacheValidationError(
            "Manifest/vector byte counts do not match rows, grid length, and dtype"
        )


def validate_pair_commit_files(
    cache_root: str | Path,
    commit: PairCommitManifest,
) -> None:
    """Validate the two manifest-backed packs named by a pair commit marker."""
    root = Path(cache_root).expanduser().resolve(strict=False)
    for signature, label in (
        (commit.raw_signature, "RAW"),
        (commit.baseline_signature, "baseline-corrected"),
    ):
        directory = root / signature
        manifest = read_cache_manifest(directory, required=True)
        if manifest is None:
            raise CacheValidationError(f"Committed {label} pack has no manifest")
        if manifest.signature != signature:
            raise CacheValidationError(f"Committed {label} pack signature is inconsistent")
        if manifest.grid != commit.grid or manifest.metadata_rows != commit.row_count:
            raise CacheValidationError(f"Committed {label} pack does not match pair geometry")
        if manifest.inventory_signature != commit.inventory_signature:
            raise CacheValidationError(
                f"Committed {label} pack has a different inventory signature"
            )
        _validate_manifest_files(directory, manifest)


def committed_pair_available(
    cache_root: str | Path,
    raw_signature: str,
    baseline_signature: str,
) -> bool:
    """Return true only for a complete marker whose two packs still validate."""
    try:
        commit = read_pair_commit(
            cache_root,
            raw_signature,
            baseline_signature,
            required=True,
        )
        if commit is None:
            return False
        validate_pair_commit_files(cache_root, commit)
    except (OSError, DatabaseError, ValueError):
        return False
    return True


class _SourceResolver:
    def __init__(
        self,
        roots: tuple[DatabaseRoot, ...],
        inventory: DatabaseInventory | None,
    ) -> None:
        self.roots = roots
        self.root_by_alias = {root.alias.casefold(): root for root in roots}
        self.root_prefixes = tuple(
            (
                root,
                root.path.as_posix().rstrip("/") + "/",
            )
            for root in roots
        )
        self.inventory_keys = (
            {(record.root_alias.casefold(), record.relative_path) for record in inventory.files}
            if inventory is not None
            else set()
        )
        self.has_inventory = inventory is not None
        basename_rows: dict[str, list[DatabaseFileRecord]] = {}
        if inventory is not None:
            for record in inventory.files:
                basename = PurePosixPath(record.relative_path).name.casefold()
                basename_rows.setdefault(basename, []).append(record)
        self.unique_basename = {
            name: rows[0] for name, rows in basename_rows.items() if len(rows) == 1
        }

    def resolve_source(
        self,
        source: PortableSource,
        *,
        require_exists: bool,
    ) -> Path | None:
        """Resolve against already-normalized roots without repeated filesystem work."""
        root = self.root_by_alias.get(source.root_alias.casefold())
        if root is not None and source.relative_path:
            # PortableSource has already rejected absolute and parent-traversal
            # paths, while every DatabaseRoot is absolute and normalized.
            candidate = root.path.joinpath(*PurePosixPath(source.relative_path).parts)
        elif source.legacy_path:
            candidate = Path(source.legacy_path).expanduser()
            if not candidate.is_absolute():
                candidate = candidate.resolve(strict=False)
        else:
            return None
        if require_exists and not candidate.is_file():
            return None
        return candidate

    def source_is_known(self, source: PortableSource) -> bool | None:
        """Check a portable source against inventory without per-row stat calls."""
        if not source.is_portable or not self.has_inventory:
            return None
        return (source.root_alias.casefold(), source.relative_path) in self.inventory_keys

    def _explicit_source(self, value: Mapping[str, Any]) -> PortableSource | None:
        source_value = value.get("source")
        if isinstance(source_value, Mapping):
            alias = source_value.get("root_alias", source_value.get("root", ""))
            relative = source_value.get(
                "relative_path", source_value.get("relpath", source_value.get("path", ""))
            )
            if alias and relative:
                return PortableSource(str(alias), str(relative), str(value.get("path", "")))
        alias = value.get("source_root", value.get("root_alias", ""))
        relative = value.get("source_relpath", value.get("relative_path", ""))
        if alias and relative:
            return PortableSource(str(alias), str(relative), str(value.get("path", "")))
        return None

    def resolve_metadata(self, value: Mapping[str, Any]) -> PortableSource:
        explicit = self._explicit_source(value)
        if explicit is not None:
            return explicit

        legacy_text = str(value.get("path", "")).strip()
        if legacy_text:
            normalized_legacy = legacy_text.replace("\\", "/")
            folded_legacy = normalized_legacy.casefold()
            for root, prefix in self.root_prefixes:
                if folded_legacy.startswith(prefix.casefold()):
                    return PortableSource(
                        root.alias,
                        normalized_legacy[len(prefix) :],
                        legacy_text,
                    )

            # A legacy absolute path can point into an older workspace.  Match a
            # logical root component, then validate the suffix against the current
            # inventory when one is available.
            components = tuple(part for part in legacy_text.replace("\\", "/").split("/") if part)
            for root in self.roots:
                aliases = {root.alias.casefold(), root.path.name.casefold()}
                for position in range(len(components) - 1, -1, -1):
                    if components[position].casefold() not in aliases:
                        continue
                    tail = "/".join(components[position + 1 :])
                    if not tail:
                        continue
                    try:
                        relative = _normalise_relative_path(tail)
                    except ValueError:
                        continue
                    key = (root.alias.casefold(), relative)
                    if not self.inventory_keys or key in self.inventory_keys:
                        return PortableSource(root.alias, relative, legacy_text)

        basename = str(value.get("orig_filename", value.get("filename", ""))).strip()
        if basename:
            record = self.unique_basename.get(Path(basename).name.casefold())
            if record is not None:
                root = self.root_by_alias.get(record.root_alias.casefold())
                if root is not None:
                    return PortableSource(root.alias, record.relative_path, legacy_text)
        return PortableSource(legacy_path=legacy_text or basename)


@dataclass(frozen=True, slots=True)
class ReferenceCatalogSummary:
    """Small phase-name catalog suitable for a long-lived UI cache."""

    unique_names: tuple[str, ...]
    reference_count: int
    skipped_count: int = 0

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[Mapping[str, Any]],
        *,
        skipped_count: int = 0,
    ) -> "ReferenceCatalogSummary":
        names: dict[str, str] = {}
        for entry in entries:
            name = str(entry.get("name", "")).strip()
            if name:
                names.setdefault(name.casefold(), name)
        return cls(
            unique_names=tuple(sorted(names.values(), key=str.casefold)),
            reference_count=len(entries),
            skipped_count=max(0, int(skipped_count)),
        )


class MatcherMetadataRow(Mapping[str, Any]):
    """Lazy legacy row view backed by shared typed identity/vector metadata."""

    __slots__ = ("_identity", "_stats", "_root_by_alias")

    _BASE_KEYS: ClassVar[tuple[str, ...]] = (
        "name",
        "formula",
        "flag",
        "filename",
        "orig_filename",
        "path",
        "start_idx",
        "end_idx",
        "l2",
        "elements",
        "has_formula",
        "parser_version",
        "source",
        "db_baseline",
        "support_runs",
        "source_root",
        "source_relpath",
    )

    def __init__(
        self,
        identity: ReferenceIdentity,
        stats: ReferenceVectorStats,
        root_by_alias: Mapping[str, DatabaseRoot],
    ) -> None:
        self._identity = identity
        self._stats = stats
        self._root_by_alias = root_by_alias

    def _resolved_path(self) -> Path:
        source = self._identity.source
        root = self._root_by_alias.get(source.root_alias.casefold())
        if root is not None and source.relative_path:
            return root.path.joinpath(*PurePosixPath(source.relative_path).parts)
        if source.legacy_path:
            return Path(source.legacy_path).expanduser()
        return Path("__unresolved_source__")

    def __getitem__(self, key: str) -> Any:
        identity = self._identity
        stats = self._stats
        if key == "name":
            return identity.name
        if key == "formula":
            return identity.formula
        if key == "flag":
            return identity.flag
        if key == "filename":
            return identity.filename
        if key == "orig_filename":
            return identity.orig_filename
        if key == "path":
            return self._resolved_path()
        if key == "start_idx":
            return stats.start_idx
        if key == "end_idx":
            return stats.end_idx
        if key == "l2":
            return stats.l2
        if key == "elements":
            return identity.elements
        if key == "has_formula":
            return identity.has_formula
        if key == "parser_version":
            return identity.parser_version
        if key == "source":
            return identity.source_label
        if key == "db_baseline":
            return stats.db_baseline
        if key == "support_runs":
            return stats.support_runs
        if key == "source_root" and identity.source.root_alias:
            return identity.source.root_alias
        if key == "source_relpath" and identity.source.relative_path:
            return identity.source.relative_path
        if key == "provenance" and identity.provenance.available:
            return identity.provenance
        if key == "error" and stats.error:
            return stats.error
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        identity = self._identity
        keys = list(self._BASE_KEYS)
        if not identity.source.root_alias:
            keys.remove("source_root")
        if not identity.source.relative_path:
            keys.remove("source_relpath")
        if identity.provenance.available:
            keys.append("provenance")
        if self._stats.error:
            keys.append("error")
        return iter(keys)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


class MatcherMetadataSequence(Sequence[MatcherMetadataRow]):
    """Indexable row adapter without constructing two full dict catalogs."""

    __slots__ = ("_metadata", "_rows", "_root_by_alias")

    def __init__(
        self,
        metadata: MetadataIndex,
        rows: tuple[ReferenceVectorStats, ...],
        roots: tuple[DatabaseRoot, ...],
    ) -> None:
        self._metadata = metadata
        self._rows = rows
        self._root_by_alias = MappingProxyType(
            {root.alias.casefold(): root for root in roots}
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(
        self, index: int | slice
    ) -> MatcherMetadataRow | tuple[MatcherMetadataRow, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(index)
        return MatcherMetadataRow(
            self._metadata.entries[position],
            self._rows[position],
            self._root_by_alias,
        )

    def __iter__(self) -> Iterator[MatcherMetadataRow]:
        for position in range(len(self)):
            yield self[position]


@dataclass(frozen=True, slots=True)
class PrecomputePack:
    directory: Path
    signature: str
    grid: GridSpec
    matrix: np.memmap = field(compare=False, repr=False)
    metadata: MetadataIndex
    rows: tuple[ReferenceVectorStats, ...]
    roots: tuple[DatabaseRoot, ...]
    manifest: CacheManifest | None = None
    hnsw_index: Any | None = field(default=None, compare=False, repr=False)
    warnings: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def resolve_source(self, row: int, *, require_exists: bool = True) -> Path:
        try:
            source = self.metadata.entries[int(row)].source
        except (IndexError, ValueError) as exc:
            raise SourceResolutionError(f"Reference row is out of range: {row}") from exc
        path = source.resolve(self.roots, require_exists=require_exists)
        if path is None:
            raise SourceResolutionError(
                "Could not resolve source for row "
                f"{row}: {source.root_alias}/{source.relative_path}"
            )
        return path

    def matcher_view(self) -> dict[str, Any]:
        """Return a lazy adapter matching the current app's pack mapping."""
        meta = MatcherMetadataSequence(self.metadata, self.rows, self.roots)
        inverse = {
            element: np.asarray(indices, dtype=np.int32)
            for element, indices in self.metadata.by_element.items()
        }
        return {
            "X": self.matrix,
            "meta": meta,
            "grid": self.grid.values(dtype=np.float32),
            "grid_info": self.grid.as_dict(),
            "ann": self.hnsw_index,
            "dir": self.directory,
            "inv_elements": inverse,
        }


@dataclass(frozen=True, slots=True)
class PrecomputePair:
    raw: PrecomputePack
    baseline_corrected: PrecomputePack

    @property
    def metadata(self) -> MetadataIndex:
        return self.raw.metadata


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"Could not read {description} {path}: {exc}") from exc


def _load_hnsw_index(path: Path, dimension: int) -> Any | None:
    if not path.is_file():
        return None
    try:
        import hnswlib  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise CacheValidationError("HNSW loading was requested but hnswlib is unavailable") from exc
    try:
        index = hnswlib.Index(space="cosine", dim=dimension)
        index.load_index(str(path))
        index.set_ef(200)
        return index
    except Exception as exc:  # pragma: no cover - optional dependency/data
        raise CacheValidationError(f"Could not load HNSW index {path}: {exc}") from exc


def _parse_reference_rows(
    metadata_value: Any,
    *,
    grid: GridSpec,
    resolver: _SourceResolver,
    expected_parser_version: int | None,
    strict_sources: bool,
) -> tuple[MetadataIndex, tuple[ReferenceVectorStats, ...], tuple[str, ...]]:
    if not isinstance(metadata_value, list) or not metadata_value:
        raise CacheValidationError("Reference metadata must be a non-empty JSON list")

    identities: list[ReferenceIdentity] = []
    stats_rows: list[ReferenceVectorStats] = []
    warnings: list[str] = []
    for row_number, value in enumerate(metadata_value):
        if not isinstance(value, Mapping):
            raise CacheValidationError(f"Metadata row {row_number} is not an object")
        try:
            start_idx = int(value.get("start_idx", 0))
            end_idx = int(value.get("end_idx", -1))
            l2 = float(value.get("l2", 0.0))
            if not math.isfinite(l2) or l2 < 0.0:
                raise ValueError("l2 must be finite and non-negative")
            if end_idx >= start_idx:
                if start_idx < 0 or end_idx >= grid.length:
                    raise ValueError(
                        f"support {start_idx}:{end_idx} is outside grid length {grid.length}"
                    )
            elif l2 > 0.0:
                raise ValueError("a positive-norm row must have non-empty support")

            runs_raw = value.get("support_runs")
            if runs_raw is None:
                support_runs = (
                    ((start_idx, end_idx),) if end_idx >= start_idx else ()
                )
            else:
                if not isinstance(runs_raw, (list, tuple)):
                    raise ValueError("support_runs must be a list of [start, end] pairs")
                parsed_runs: list[tuple[int, int]] = []
                previous_end = -1
                for run in runs_raw:
                    if not isinstance(run, (list, tuple)) or len(run) != 2:
                        raise ValueError("support_runs entries must contain two indices")
                    run_start, run_end = int(run[0]), int(run[1])
                    if (
                        run_start < 0
                        or run_end < run_start
                        or run_end >= grid.length
                        or run_start <= previous_end
                    ):
                        raise ValueError("support_runs must be ordered, disjoint, and in-grid")
                    parsed_runs.append((run_start, run_end))
                    previous_end = run_end
                support_runs = tuple(parsed_runs)
                if support_runs and (
                    start_idx != support_runs[0][0] or end_idx != support_runs[-1][1]
                ):
                    raise ValueError("start_idx/end_idx must summarize support_runs")
                if not support_runs and end_idx >= start_idx:
                    raise ValueError("non-empty support summary requires support_runs")

            parser_raw = value.get("parser_version")
            parser_version = int(parser_raw) if parser_raw is not None else None
            if expected_parser_version is not None and parser_version != expected_parser_version:
                raise ValueError(
                    f"parser version {parser_version!r} != expected {expected_parser_version}"
                )

            elements_raw = value.get("elements", [])
            if not isinstance(elements_raw, (list, tuple)):
                raise ValueError("elements must be a list")
            elements = tuple(sorted({str(element) for element in elements_raw if str(element)}))
            provenance_raw = value.get("provenance", {})
            if not isinstance(provenance_raw, Mapping):
                raise ValueError("provenance must be an object")
            provenance = CachedProvenance.from_mapping(provenance_raw)
            source = resolver.resolve_metadata(value)
            known = resolver.source_is_known(source)
            if known is False or (
                strict_sources
                and known is None
                and resolver.resolve_source(source, require_exists=True) is None
            ):
                message = f"Metadata row {row_number} source is unresolved: {source}"
                if strict_sources:
                    raise SourceResolutionError(message)
                warnings.append(message)

            identities.append(
                ReferenceIdentity(
                    name=str(value.get("name", "Unknown")),
                    formula=str(value.get("formula", "?")),
                    flag=str(value.get("flag", "")),
                    filename=str(value.get("filename", "")),
                    orig_filename=str(value.get("orig_filename", value.get("filename", ""))),
                    source=source,
                    elements=elements,
                    has_formula=bool(value.get("has_formula", bool(elements))),
                    parser_version=parser_version,
                    source_label=(
                        str(value.get("source", ""))
                        if not isinstance(value.get("source"), Mapping)
                        else ""
                    ),
                    provenance=provenance,
                )
            )
            stats_rows.append(
                ReferenceVectorStats(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    l2=l2,
                    db_baseline=bool(value.get("db_baseline", False)),
                    support_runs=support_runs,
                    error=str(value.get("error", "")),
                )
            )
        except SourceResolutionError:
            raise
        except (TypeError, ValueError) as exc:
            raise CacheValidationError(f"Invalid metadata row {row_number}: {exc}") from exc

    return MetadataIndex.build(identities), tuple(stats_rows), tuple(warnings)


def load_precompute_pack(
    directory: str | Path,
    *,
    roots: DatabaseRootsInput,
    inventory: DatabaseInventory | None = None,
    expected_signature: str | None = None,
    expected_parser_version: int | None = None,
    strict_sources: bool = False,
    require_manifest: bool = False,
    load_hnsw: bool = False,
) -> PrecomputePack:
    """Validate and open one precompute pack read-only.

    Legacy directories without a manifest remain supported.  HNSW is not loaded
    by default because the current range-local matcher performs an exact scan.
    """
    cache_dir = Path(directory).expanduser().resolve(strict=False)
    if not cache_dir.is_dir():
        raise CacheValidationError(f"Precompute directory does not exist: {cache_dir}")
    root_values = coerce_database_roots(roots)
    manifest = read_cache_manifest(cache_dir, required=require_manifest)
    if manifest is not None and not manifest.complete:
        raise CacheValidationError(f"Precompute manifest is not marked complete: {cache_dir}")

    signature = manifest.signature if manifest is not None else cache_dir.name
    if expected_signature is not None and signature != expected_signature:
        raise CacheValidationError(
            f"Cache signature mismatch: found {signature!r}, expected {expected_signature!r}"
        )

    vector_name = manifest.vector_file if manifest is not None else DEFAULT_VECTOR_FILE
    metadata_name = manifest.metadata_file if manifest is not None else DEFAULT_METADATA_FILE
    grid_name = manifest.grid_file if manifest is not None else DEFAULT_GRID_FILE
    vector_path = cache_dir / vector_name
    metadata_path = cache_dir / metadata_name
    grid_path = cache_dir / grid_name
    for path, label in (
        (vector_path, "vector file"),
        (metadata_path, "metadata file"),
        (grid_path, "grid file"),
    ):
        if not path.is_file():
            raise CacheValidationError(f"Precompute {label} is missing: {path}")

    grid_value = _read_json(grid_path, "grid metadata")
    if not isinstance(grid_value, Mapping):
        raise CacheValidationError(f"Grid metadata must be a JSON object: {grid_path}")
    grid = GridSpec.from_mapping(grid_value)
    if manifest is not None and grid != manifest.grid:
        raise CacheValidationError("Manifest grid does not match grid.json")

    dtype = np.dtype(manifest.vector_dtype if manifest is not None else "float32")
    if dtype.hasobject:
        raise CacheValidationError("Object dtypes are not valid precompute vectors")
    row_bytes = grid.length * dtype.itemsize
    vector_bytes = vector_path.stat().st_size
    if row_bytes <= 0 or vector_bytes % row_bytes != 0:
        raise CacheValidationError(
            f"Vector byte size {vector_bytes} is not divisible by row size {row_bytes}"
        )
    vector_rows = vector_bytes // row_bytes
    metadata_value = _read_json(metadata_path, "reference metadata")
    if not isinstance(metadata_value, list):
        raise CacheValidationError("Reference metadata must be a JSON list")
    if vector_rows != len(metadata_value):
        raise CacheValidationError(
            f"Vector rows ({vector_rows}) do not match metadata rows ({len(metadata_value)})"
        )
    if vector_rows <= 0:
        raise CacheValidationError("Precompute pack contains no reference rows")
    if manifest is not None:
        if manifest.vector_bytes != vector_bytes:
            raise CacheValidationError(
                f"Manifest vector byte count {manifest.vector_bytes} != file size {vector_bytes}"
            )
        if manifest.metadata_rows != len(metadata_value):
            raise CacheValidationError(
                "Manifest metadata row count does not match metadata file"
            )

    resolver = _SourceResolver(root_values, inventory)
    metadata, stats_rows, warnings = _parse_reference_rows(
        metadata_value,
        grid=grid,
        resolver=resolver,
        expected_parser_version=expected_parser_version,
        strict_sources=strict_sources,
    )
    matrix = np.memmap(
        vector_path,
        mode="r",
        dtype=dtype,
        shape=(vector_rows, grid.length),
    )
    matrix.flags.writeable = False
    hnsw_index = (
        _load_hnsw_index(cache_dir / DEFAULT_HNSW_FILE, grid.length) if load_hnsw else None
    )
    return PrecomputePack(
        directory=cache_dir,
        signature=signature,
        grid=grid,
        matrix=matrix,
        metadata=metadata,
        rows=stats_rows,
        roots=root_values,
        manifest=manifest,
        hnsw_index=hnsw_index,
        warnings=warnings,
    )


def load_precompute_pair(
    raw_directory: str | Path,
    baseline_directory: str | Path,
    *,
    roots: DatabaseRootsInput,
    inventory: DatabaseInventory | None = None,
    expected_raw_signature: str | None = None,
    expected_baseline_signature: str | None = None,
    expected_parser_version: int | None = None,
    strict_sources: bool = False,
    require_manifest: bool = False,
    load_hnsw: bool = False,
    require_aligned_metadata: bool = True,
    pair_commit_root: str | Path | None = None,
    require_pair_commit: bool = False,
) -> PrecomputePair:
    """Load RAW/BC packs and intern their immutable static metadata."""
    commit: PairCommitManifest | None = None
    if require_pair_commit:
        if pair_commit_root is None:
            raise ValueError("pair_commit_root is required with require_pair_commit=True")
        if expected_raw_signature is None or expected_baseline_signature is None:
            raise ValueError("Expected pair signatures are required for commit validation")
        commit = read_pair_commit(
            pair_commit_root,
            expected_raw_signature,
            expected_baseline_signature,
            required=True,
        )
        if commit is None:
            raise CacheValidationError("Required pair commit marker is missing")
        validate_pair_commit_files(pair_commit_root, commit)
        require_manifest = True
    root_values = coerce_database_roots(roots)
    raw = load_precompute_pack(
        raw_directory,
        roots=root_values,
        inventory=inventory,
        expected_signature=expected_raw_signature,
        expected_parser_version=expected_parser_version,
        strict_sources=strict_sources,
        require_manifest=require_manifest,
        load_hnsw=load_hnsw,
    )
    baseline = load_precompute_pack(
        baseline_directory,
        roots=root_values,
        inventory=inventory,
        expected_signature=expected_baseline_signature,
        expected_parser_version=expected_parser_version,
        strict_sources=strict_sources,
        require_manifest=require_manifest,
        load_hnsw=load_hnsw,
    )
    if raw.grid != baseline.grid:
        raise CacheValidationError("RAW and baseline-corrected packs use different grids")
    if raw.row_count != baseline.row_count:
        raise CacheValidationError("RAW and baseline-corrected packs have different row counts")
    if commit is not None and (
        raw.grid != commit.grid or raw.row_count != commit.row_count
    ):
        raise CacheValidationError("Loaded cache pair does not match its commit marker")

    identities_align = all(
        raw_entry.cache_key == baseline_entry.cache_key
        for raw_entry, baseline_entry in zip(
            raw.metadata.entries,
            baseline.metadata.entries,
            strict=True,
        )
    )
    if identities_align:
        baseline = replace(baseline, metadata=raw.metadata)
    elif require_aligned_metadata:
        raise CacheValidationError(
            "RAW and baseline-corrected packs do not have identical reference ordering"
        )
    return PrecomputePair(raw=raw, baseline_corrected=baseline)


def create_cache_staging_directory(final_directory: str | Path) -> Path:
    """Create a same-filesystem staging directory next to the final cache."""
    final = Path(final_directory).expanduser().resolve(strict=False)
    final.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent)
    )


def promote_cache_directory(
    staging_directory: str | Path,
    final_directory: str | Path,
    *,
    require_complete_manifest: bool = True,
) -> Path:
    """Atomically rename one completed staging directory into place.

    Existing destinations are never replaced.  Versioned/signature cache paths
    make this safer than deleting or overwriting a cache potentially used by
    another Streamlit session.
    """
    staging = Path(staging_directory).expanduser().resolve(strict=False)
    final = Path(final_directory).expanduser().resolve(strict=False)
    if not staging.is_dir() or staging.is_symlink():
        raise CacheValidationError(f"Staging cache is not a real directory: {staging}")
    if staging.parent != final.parent:
        raise CacheValidationError("Staging and final cache must share one parent filesystem")
    if final.exists():
        raise FileExistsError(f"Refusing to replace existing cache directory: {final}")
    if require_complete_manifest:
        manifest = read_cache_manifest(staging, required=True)
        if manifest is None or not manifest.complete:
            raise CacheValidationError("Staging cache has no complete manifest")
        _validate_manifest_files(staging, manifest)
    os.rename(staging, final)
    _fsync_directory(final.parent)
    return final


@contextmanager
def atomic_cache_directory(
    final_directory: str | Path,
    *,
    require_complete_manifest: bool = True,
    preserve_failed_staging: bool = True,
) -> Iterator[Path]:
    """Yield a staging directory and atomically promote it on successful exit.

    Failed staging content is preserved by default for inspection and recovery.
    Passing ``preserve_failed_staging=False`` is an explicit instruction to
    discard only the temporary directory created by this context manager.
    """
    staging = create_cache_staging_directory(final_directory)
    promoted = False
    try:
        yield staging
        promote_cache_directory(
            staging,
            final_directory,
            require_complete_manifest=require_complete_manifest,
        )
        promoted = True
    finally:
        if not promoted and staging.exists() and not preserve_failed_staging:
            shutil.rmtree(staging)


@dataclass(frozen=True, slots=True)
class CacheEntryInfo:
    path: Path
    signature: str
    size_bytes: int
    modified_ns: int
    complete: bool
    manifest: CacheManifest | None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    entry: CacheEntryInfo
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheCleanupPlan:
    cache_root: Path
    keep: tuple[CacheEntryInfo, ...]
    candidates: tuple[CleanupCandidate, ...]
    total_bytes: int
    reclaimable_bytes: int


@dataclass(frozen=True, slots=True)
class QuarantinedCacheEntry:
    original_path: Path
    quarantine_path: Path
    size_bytes: int


def _directory_size_and_mtime(directory: Path) -> tuple[int, int, tuple[str, ...]]:
    total = 0
    latest = 0
    issues: list[str] = []
    try:
        for path in directory.rglob("*"):
            try:
                if path.is_file():
                    stat = path.stat()
                    total += int(stat.st_size)
                    latest = max(latest, int(stat.st_mtime_ns))
            except OSError as exc:
                issues.append(f"{path.name}: {exc}")
    except OSError as exc:
        issues.append(str(exc))
    return total, latest, tuple(issues)


def inspect_cache_entries(cache_root: str | Path) -> tuple[CacheEntryInfo, ...]:
    """Inspect immediate cache directories without deleting or loading vectors."""
    root = Path(cache_root).expanduser().resolve(strict=False)
    if not root.exists():
        return ()
    entries: list[CacheEntryInfo] = []
    directories = (path for path in root.iterdir() if path.is_dir())
    for directory in sorted(directories, key=lambda path: path.name):
        issues: list[str] = []
        try:
            manifest = read_cache_manifest(directory, required=False)
        except CacheValidationError as exc:
            manifest = None
            issues.append(str(exc))
        size, modified, size_issues = _directory_size_and_mtime(directory)
        issues.extend(size_issues)
        if manifest is not None:
            required_files = (
                directory / manifest.vector_file,
                directory / manifest.metadata_file,
                directory / manifest.grid_file,
            )
            complete = manifest.complete and all(path.is_file() for path in required_files)
            signature = manifest.signature
        else:
            required_files = (
                directory / DEFAULT_VECTOR_FILE,
                directory / DEFAULT_METADATA_FILE,
                directory / DEFAULT_GRID_FILE,
            )
            complete = all(path.is_file() for path in required_files)
            signature = directory.name
        entries.append(
            CacheEntryInfo(
                path=directory,
                signature=signature,
                size_bytes=size,
                modified_ns=modified,
                complete=complete,
                manifest=manifest,
                issues=tuple(issues),
            )
        )
    return tuple(entries)


def plan_cache_cleanup(
    cache_root: str | Path,
    *,
    active_signatures: Sequence[str] = (),
    candidate_signatures: Sequence[str] = (),
    include_incomplete: bool = True,
    max_total_bytes: int | None = None,
    retain_newest_complete: int = 2,
) -> CacheCleanupPlan:
    """Produce a read-only cleanup proposal.

    Nothing is removed.  Active signatures are always retained.  Complete old
    entries become candidates only when explicitly named or when a byte budget is
    supplied; incomplete entries can be proposed separately.
    """
    entries = inspect_cache_entries(cache_root)
    active = {str(value) for value in active_signatures}
    explicit = {str(value) for value in candidate_signatures}
    reasons: dict[Path, list[str]] = {}

    for entry in entries:
        if entry.signature in active:
            continue
        if entry.signature in explicit:
            reasons.setdefault(entry.path, []).append("explicit candidate")
        if include_incomplete and not entry.complete:
            reasons.setdefault(entry.path, []).append("incomplete cache")

    total = sum(entry.size_bytes for entry in entries)
    if max_total_bytes is not None:
        budget = max(0, int(max_total_bytes))
        already_reclaimable = sum(
            entry.size_bytes for entry in entries if entry.path in reasons
        )
        projected = total - already_reclaimable
        protected_complete = {
            entry.path
            for entry in sorted(
                (
                    value
                    for value in entries
                    if value.complete
                    and value.signature not in active
                    and value.path not in reasons
                ),
                key=lambda value: value.modified_ns,
                reverse=True,
            )[: max(0, int(retain_newest_complete))]
        }
        old_first = sorted(entries, key=lambda value: value.modified_ns)
        for entry in old_first:
            if projected <= budget:
                break
            if (
                entry.signature in active
                or entry.path in reasons
                or entry.path in protected_complete
            ):
                continue
            reasons.setdefault(entry.path, []).append("cache-size budget")
            projected -= entry.size_bytes

    candidates = tuple(
        CleanupCandidate(entry=entry, reasons=tuple(reasons[entry.path]))
        for entry in entries
        if entry.path in reasons
    )
    candidate_paths = {candidate.entry.path for candidate in candidates}
    keep = tuple(entry for entry in entries if entry.path not in candidate_paths)
    reclaimable = sum(candidate.entry.size_bytes for candidate in candidates)
    return CacheCleanupPlan(
        cache_root=Path(cache_root).expanduser().resolve(strict=False),
        keep=keep,
        candidates=candidates,
        total_bytes=total,
        reclaimable_bytes=reclaimable,
    )


def quarantine_cleanup_candidates(
    plan: CacheCleanupPlan,
) -> tuple[QuarantinedCacheEntry, ...]:
    """Move planned candidates to a recoverable sibling quarantine directory.

    The function accepts only the exact immediate child directories captured in
    ``plan``.  It never follows symlinks, overwrites an existing destination,
    or permanently deletes cache data.
    """

    root = Path(plan.cache_root).resolve(strict=False)
    quarantine_root = root.parent / f".{root.name}-quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    moved: list[QuarantinedCacheEntry] = []
    for candidate in plan.candidates:
        source = Path(candidate.entry.path)
        resolved_source = source.resolve(strict=False)
        if source.is_symlink() or resolved_source.parent != root:
            raise CacheValidationError(
                f"cleanup candidate is not a real immediate cache child: {source}"
            )
        if not source.is_dir():
            raise CacheValidationError(f"cleanup candidate no longer exists: {source}")
        suffix = 0
        while True:
            extra = "" if suffix == 0 else f"-{suffix}"
            destination = quarantine_root / f"{source.name}{extra}"
            if not destination.exists():
                break
            suffix += 1
        os.rename(source, destination)
        moved.append(
            QuarantinedCacheEntry(
                original_path=source,
                quarantine_path=destination,
                size_bytes=int(candidate.entry.size_bytes),
            )
        )
    return tuple(moved)


def quarantine_cache_directories(
    cache_root: str | Path,
    directories: Sequence[str | Path],
) -> tuple[QuarantinedCacheEntry, ...]:
    """Recoverably move exact immediate cache children out of a rebuild's way."""
    root = Path(cache_root).expanduser().resolve(strict=False)
    entries: list[CacheEntryInfo] = []
    for value in directories:
        requested_source = Path(value).expanduser()
        if requested_source.is_symlink():
            raise CacheValidationError(
                f"recovery target must not be a symlink: {requested_source}"
            )
        source = requested_source.resolve(strict=False)
        if not source.exists():
            continue
        if source.parent != root or not source.is_dir():
            raise CacheValidationError(
                f"recovery target is not a real immediate cache child: {source}"
            )
        size, modified, issues = _directory_size_and_mtime(source)
        entries.append(
            CacheEntryInfo(
                path=source,
                signature=source.name,
                size_bytes=size,
                modified_ns=modified,
                complete=False,
                manifest=None,
                issues=issues,
            )
        )
    if not entries:
        return ()
    plan = CacheCleanupPlan(
        cache_root=root,
        keep=(),
        candidates=tuple(
            CleanupCandidate(entry=entry, reasons=("incomplete pair recovery",))
            for entry in entries
        ),
        total_bytes=sum(entry.size_bytes for entry in entries),
        reclaimable_bytes=sum(entry.size_bytes for entry in entries),
    )
    return quarantine_cleanup_candidates(plan)


# ---------------------------------------------------------------------------
# Lightweight provenance parsing


@dataclass(frozen=True, slots=True)
class CorrectionStep:
    """One declared correction and its optional free-text detail."""

    name: str
    applied: bool | None
    details: str = ""


@dataclass(frozen=True, slots=True)
class ChemistryProvenance:
    ideal: str = ""
    measured: str = ""
    structural: str = ""
    sample_source: str = ""


@dataclass(frozen=True, slots=True)
class SpectrumProvenance:
    """Normalized provenance fields used to assess reference suitability.

    Empty strings and ``None`` mean the source did not state the field; they do
    not imply a negative result.  ``raw_fields`` preserves the lightweight
    header values for future UI details without requiring a second file read.
    """

    database: str
    accession: str = ""
    source: str = ""
    source_accession: str = ""
    owner: str = ""
    url: str = ""
    quality_folder: str = ""
    quality: str = "unknown"
    status: str = ""
    processing: str = "unknown"
    determination: str = "unknown"
    orientation: str = "unknown"
    orientation_detail: str = ""
    scan_type: str = ""
    excitation_wavelength_nm: float | None = None
    resolution_cm1: float | None = None
    chemistry: ChemistryProvenance = ChemistryProvenance()
    corrections: tuple[CorrectionStep, ...] = ()
    raw_intensity_available: bool = False
    processed_intensity_available: bool = True
    notes: tuple[str, ...] = ()
    raw_fields: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.raw_fields, MappingProxyType):
            object.__setattr__(
                self,
                "raw_fields",
                MappingProxyType(dict(self.raw_fields)),
            )

    @property
    def correction_history(self) -> tuple[str, ...]:
        history: list[str] = []
        for step in self.corrections:
            if step.applied is True:
                state = "applied"
            elif step.applied is False:
                state = "not applied"
            else:
                state = "not stated"
            text = f"{step.name}: {state}"
            if step.details:
                text += f" ({step.details})"
            history.append(text)
        return tuple(history)


_FLOAT_VALUE_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_PROVENANCE_NOTE_PATTERN = re.compile(
    r"\b(?:baseline|background|fluorescen|correct(?:ed|ion)?|"
    r"process(?:ed|ing)?|smooth(?:ed|ing)?)\b",
    re.IGNORECASE,
)


def _strip_header_value(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    if text in ("?", "."):
        return ""
    return " ".join(text.split())


def _optional_float(value: str) -> float | None:
    match = _FLOAT_VALUE_PATTERN.search(str(value).replace(",", "."))
    if match is None:
        return None
    try:
        result = float(match.group(0))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _optional_claim(value: str) -> bool | None:
    normalized = _strip_header_value(value).casefold()
    if normalized in {"yes", "y", "true", "1", "applied"}:
        return True
    if normalized in {"no", "n", "false", "0", "none", "not applied"}:
        return False
    return None


def _normalise_processing(value: str) -> str:
    normalized = str(value).casefold()
    if "raw" in normalized:
        return "raw"
    if "process" in normalized or "correct" in normalized:
        return "processed"
    return "unknown"


def _normalise_determination(value: str) -> str:
    normalized = _strip_header_value(value).casefold()
    if "experiment" in normalized or normalized in {"measured", "measurement"}:
        return "experimental"
    if "theor" in normalized or "calculat" in normalized or "predict" in normalized:
        return "theoretical"
    return "unknown"


def _normalise_orientation(value: str, folder_hint: str = "") -> tuple[str, str]:
    detail = _strip_header_value(value)
    normalized = detail.casefold()
    folder = folder_hint.casefold()
    if "unoriented" in normalized or folder.endswith("_unoriented"):
        return "unoriented", detail or "unoriented"
    if normalized or folder.endswith("_oriented"):
        return "oriented", detail
    return "unknown", detail


def _rruff_filename_fields(path: Path) -> Mapping[str, str]:
    tokens = path.stem.split("__")
    if len(tokens) != 8:
        return MappingProxyType({})
    return MappingProxyType(
        {
            "mineral": tokens[0],
            "accession": tokens[1],
            "scan_type": tokens[2].replace("_", " "),
            "wavelength": tokens[3],
            "rotation": tokens[4],
            "orientation": tokens[5],
            "filetype": tokens[6].replace("_", " "),
        }
    )


def _rruff_wavelength(value: str) -> float | None:
    text = str(value).strip().casefold().removesuffix("nm")
    # RRUFF filename tokens such as 514-5 and 458-5nm represent a decimal
    # wavelength, while ordinary header values are already numeric.
    if re.fullmatch(r"\d{3,4}-\d", text):
        text = text.replace("-", ".")
    return _optional_float(text)


def _quality_from_folder(path: Path) -> tuple[str, str]:
    folder = path.parent.name
    folded = folder.casefold()
    if folded == "lr-raman":
        return folder, "long-range"
    match = re.fullmatch(r"(excellent|fair|poor|unrated)_(?:un)?oriented", folded)
    if match:
        return folder, match.group(1)
    return folder, "unknown"


def parse_rruff_provenance(
    path: str | Path,
    *,
    max_header_lines: int = 256,
) -> SpectrumProvenance:
    """Read only the leading RRUFF header/comments and normalize provenance."""
    source_path = Path(path)
    fields: dict[str, str] = {}
    notes: list[str] = []
    try:
        with source_path.open("r", errors="ignore") as handle:
            for line_number, line in enumerate(handle):
                if line_number >= max(1, int(max_header_lines)):
                    break
                stripped = line.strip()
                if stripped.startswith("##"):
                    key, separator, value = stripped[2:].partition("=")
                    if separator:
                        fields[key.strip().upper()] = _strip_header_value(value)
                    continue
                if stripped.startswith("#"):
                    text = stripped.lstrip("#").strip()
                    if text and _PROVENANCE_NOTE_PATTERN.search(text):
                        notes.append(text)
                    continue
                if stripped:
                    break
    except OSError as exc:
        raise InventoryError(
            f"Could not read RRUFF provenance header {source_path}: {exc}"
        ) from exc

    filename = _rruff_filename_fields(source_path)
    quality_folder, quality = _quality_from_folder(source_path)
    filetype = fields.get("FILETYPE", filename.get("filetype", ""))
    processing = _normalise_processing(filetype)
    orientation_hint = fields.get("ORIENTATION", filename.get("orientation", ""))
    orientation, orientation_detail = _normalise_orientation(
        orientation_hint, quality_folder
    )
    wavelength = _rruff_wavelength(
        fields.get("RAMAN WAVELENGTH", filename.get("wavelength", ""))
    )
    accession = fields.get("RRUFFID", filename.get("accession", ""))
    corrections: list[CorrectionStep] = []
    if processing != "unknown":
        corrections.append(
            CorrectionStep(
                name="RRUFF processing state",
                applied=(processing == "processed"),
                details=filetype,
            )
        )
    return SpectrumProvenance(
        database="RRUFF",
        accession=accession,
        source=fields.get("SOURCE", ""),
        owner=fields.get("OWNER", ""),
        url=fields.get("URL", ""),
        quality_folder=quality_folder,
        quality=quality,
        status=fields.get("STATUS", ""),
        processing=processing,
        determination="experimental",
        orientation=orientation,
        orientation_detail=orientation_detail,
        scan_type=filename.get("scan_type", ""),
        excitation_wavelength_nm=wavelength,
        resolution_cm1=_optional_float(fields.get("RAMAN RESOLUTION", "")),
        chemistry=ChemistryProvenance(
            ideal=fields.get("IDEAL CHEMISTRY", ""),
            measured=fields.get("MEASURED CHEMISTRY", ""),
            sample_source=fields.get("SOURCE", ""),
        ),
        corrections=tuple(corrections),
        raw_intensity_available=(processing == "raw"),
        processed_intensity_available=(processing != "raw"),
        notes=tuple(dict.fromkeys(notes)),
        raw_fields=MappingProxyType(dict(fields)),
    )


def _read_rod_scalar_fields(
    path: Path,
    *,
    max_header_lines: int,
) -> tuple[dict[str, str], set[str]]:
    fields: dict[str, str] = {}
    present_tags: set[str] = set()
    try:
        with path.open("r", errors="ignore") as handle:
            lines: list[str] = []
            spectrum_header_seen = False
            for line_number, line in enumerate(handle):
                if line_number >= max(1, int(max_header_lines)):
                    break
                lines.append(line.rstrip("\r\n"))
                stripped = line.strip()
                if stripped.casefold().startswith("_raman_spectrum."):
                    spectrum_header_seen = True
                elif (
                    spectrum_header_seen
                    and stripped
                    and not stripped.startswith(("_", "#", ";", "loop_"))
                    and _FLOAT_VALUE_PATTERN.match(stripped)
                ):
                    break
    except OSError as exc:
        raise InventoryError(f"Could not read ROD provenance header {path}: {exc}") from exc

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("_"):
            index += 1
            continue
        parts = stripped.split(None, 1)
        tag = parts[0]
        tag = tag.casefold()
        present_tags.add(tag)
        inline = parts[1].strip() if len(parts) > 1 else ""
        if inline:
            fields[tag] = _strip_header_value(inline)
            index += 1
            continue
        if index + 1 < len(lines) and lines[index + 1].startswith(";"):
            index += 2
            block: list[str] = []
            while index < len(lines) and not lines[index].startswith(";"):
                block.append(lines[index].strip())
                index += 1
            fields[tag] = _strip_header_value(" ".join(block))
            if index < len(lines):
                index += 1
            continue
        fields.setdefault(tag, "")
        index += 1
    return fields, present_tags


def parse_rod_provenance(
    path: str | Path,
    *,
    max_header_lines: int = 512,
) -> SpectrumProvenance:
    """Read scalar ROD/CIF Raman fields without parsing spectral data."""
    source_path = Path(path)
    fields, tags = _read_rod_scalar_fields(
        source_path, max_header_lines=max_header_lines
    )

    def get(*names: str) -> str:
        for name in names:
            value = fields.get(name.casefold(), "")
            if value:
                return value
        return ""

    determination = _normalise_determination(
        get("_raman_determination_method", "_raman_determination.method")
    )
    orientation_detail = get("_raman_measurement_device.direction_polarization")
    orientation, orientation_detail = _normalise_orientation(orientation_detail)
    raw_available = "_raman_spectrum.raw_intensity" in tags
    processed_available = "_raman_spectrum.intensity" in tags
    if raw_available and processed_available:
        processing = "processed"
    elif raw_available:
        processing = "raw"
    else:
        processing = "unknown"

    background_value = get("_raman_measurement.background_subtraction")
    background_details = get("_raman_measurement.background_subtraction_details")
    baseline_value = get("_raman_measurement.baseline_correction")
    baseline_details = get("_raman_measurement.baseline_correction_details")
    corrections = (
        CorrectionStep(
            "background subtraction",
            _optional_claim(background_value),
            background_details,
        ),
        CorrectionStep(
            "baseline correction",
            _optional_claim(baseline_value),
            baseline_details,
        ),
    )
    source_file = get("_rod_data_source.file", "_cod_data_source_file")
    source_block = get("_rod_data_source.block", "_cod_data_source_block")
    source_text = source_file or get("_chemical_compound_source")
    accession = get("_rod_database.code") or source_path.stem
    status_parts = [
        part
        for part in (
            f"determination: {determination}" if determination != "unknown" else "",
            get("_rod_maintainer_comment.text"),
        )
        if part
    ]
    return SpectrumProvenance(
        database="ROD",
        accession=accession,
        source=source_text,
        source_accession=source_block,
        quality="unknown",
        status="; ".join(status_parts),
        processing=processing,
        determination=determination,
        orientation=orientation,
        orientation_detail=orientation_detail,
        excitation_wavelength_nm=_optional_float(
            get("_raman_measurement_device.excitation_laser_wavelength")
        ),
        resolution_cm1=_optional_float(
            get("_raman_measurement_device.resolution")
        ),
        chemistry=ChemistryProvenance(
            ideal=get("_chemical_formula_sum"),
            structural=get("_chemical_formula_structural"),
            sample_source=get("_chemical_compound_source"),
        ),
        corrections=corrections,
        raw_intensity_available=raw_available,
        processed_intensity_available=processed_available,
        raw_fields=MappingProxyType(dict(fields)),
    )


def parse_spectrum_provenance(
    path: str | Path,
    *,
    max_header_lines: int | None = None,
    database_hint: str | DatabaseRoot | None = None,
) -> SpectrumProvenance:
    """Parse a lightweight header and retain the caller's logical DB identity.

    TXT describes a file format here, not necessarily the RRUFF collection:
    user-owned libraries commonly contain RRUFF-shaped TXT files.  A root alias
    supplied by the inventory/builder therefore overrides the format parser's
    default database label.
    """
    source_path = Path(path)
    provenance: SpectrumProvenance
    if source_path.suffix.casefold() == ".rod":
        provenance = parse_rod_provenance(
            source_path,
            max_header_lines=512 if max_header_lines is None else max_header_lines,
        )
    elif source_path.suffix.casefold() == ".txt":
        provenance = parse_rruff_provenance(
            source_path,
            max_header_lines=256 if max_header_lines is None else max_header_lines,
        )
    else:
        raise InventoryError(
            f"Unsupported provenance file type: {source_path.suffix or '<none>'}"
        )

    hinted_database = (
        database_hint.alias
        if isinstance(database_hint, DatabaseRoot)
        else str(database_hint or "").strip()
    )
    if hinted_database:
        provenance = replace(provenance, database=hinted_database)
    return provenance


# ---------------------------------------------------------------------------
# Typed paired-cache builder


ReferenceDiscovery = Callable[[tuple[Path, ...]], Sequence[Mapping[str, Any]]]
SpectrumParser = Callable[[Path], tuple[np.ndarray, np.ndarray]]
VectorPreprocessor = Callable[
    [np.ndarray, np.ndarray, np.ndarray, bool], np.ndarray
]
CleanXY = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
SupportSliceProvider = Callable[[np.ndarray], Sequence[slice]]
ElementExtractor = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class PairBuildRequest:
    """Validated, UI-independent specification for one aligned cache pair."""

    cache_root: Path
    raw_signature: str
    baseline_signature: str
    roots: tuple[DatabaseRoot, ...]
    inventory_signature: str
    grid: GridSpec
    parser_version: int
    preprocess_version: int
    preprocess_step_cm1: float
    workers: int = 1
    systemic_failure_fraction: float = 0.90
    invalid_commit_created_at_ns: int | None = None
    raw_variant: str = "library_as_provided"
    baseline_variant: str = "baseline_corrected_raw_sources"

    def __post_init__(self) -> None:
        root = Path(self.cache_root).expanduser().resolve(strict=False)
        roots = coerce_database_roots(self.roots)
        for signature in (self.raw_signature, self.baseline_signature):
            if not signature or Path(signature).name != signature:
                raise ValueError(f"Pair signatures must be simple names: {signature!r}")
        if self.raw_signature == self.baseline_signature:
            raise ValueError("RAW and baseline cache signatures must differ")
        if not str(self.inventory_signature).strip():
            raise ValueError("Pair builds require an inventory signature")
        if int(self.parser_version) < 0 or int(self.preprocess_version) < 0:
            raise ValueError("Parser and preprocessing versions must be non-negative")
        step = float(self.preprocess_step_cm1)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("Preprocessing grid spacing must be finite and positive")
        failure_fraction = float(self.systemic_failure_fraction)
        if not math.isfinite(failure_fraction) or not 0.50 <= failure_fraction <= 1.0:
            raise ValueError("Systemic failure fraction must be between 0.50 and 1.0")
        object.__setattr__(self, "cache_root", root)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "workers", max(1, int(self.workers)))
        object.__setattr__(self, "parser_version", int(self.parser_version))
        object.__setattr__(self, "preprocess_version", int(self.preprocess_version))
        object.__setattr__(self, "preprocess_step_cm1", step)
        object.__setattr__(self, "systemic_failure_fraction", failure_fraction)

    @property
    def raw_directory(self) -> Path:
        return self.cache_root / self.raw_signature

    @property
    def baseline_directory(self) -> Path:
        return self.cache_root / self.baseline_signature

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(root.path for root in self.roots)

    @property
    def grid_metadata(self) -> dict[str, Any]:
        return {
            **self.grid.as_dict(),
            "preprocess_version": self.preprocess_version,
            "preprocess_step_cm1": self.preprocess_step_cm1,
            "db_smoothing": False,
            "db_ai_denoising": False,
        }


@dataclass(frozen=True, slots=True)
class ReferenceBuildSource:
    """One discovered reference before its spectral data are parsed."""

    path: Path
    name: str
    formula: str
    flag: str
    filename: str
    orig_filename: str
    source: PortableSource
    source_label: str = ""

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        roots: DatabaseRootsInput,
    ) -> "ReferenceBuildSource":
        try:
            path = Path(value["path"]).expanduser().resolve(strict=False)
        except (KeyError, TypeError, ValueError) as exc:
            raise PairBuildError(f"Reference discovery returned an invalid path: {exc}") from exc
        portable = portable_source_for_path(path, roots)
        return cls(
            path=path,
            name=str(value.get("name", "Unknown")),
            formula=str(value.get("formula", "?")),
            flag=str(value.get("flag", "")),
            filename=str(value.get("filename", "")),
            orig_filename=str(value.get("orig_filename", path.name)),
            source=portable,
            source_label=str(value.get("source", "")),
        )


@dataclass(frozen=True, slots=True)
class PreparedReferenceVector:
    """One normalized cache vector and its exact gap-aware grid support."""

    values: np.ndarray = field(compare=False, repr=False)
    start_idx: int
    end_idx: int
    l2: float
    support_runs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("prepared reference vector contains non-finite values")
        runs = tuple((int(start), int(end)) for start, end in self.support_runs)
        for index, (start, end) in enumerate(runs):
            if start < 0 or end < start or end >= values.size:
                raise ValueError("prepared support runs must be non-negative and ordered")
            if index and start <= runs[index - 1][1]:
                raise ValueError("prepared support runs must be disjoint and ordered")
        expected_start = runs[0][0] if runs else 0
        expected_end = runs[-1][1] if runs else -1
        if int(self.start_idx) != expected_start or int(self.end_idx) != expected_end:
            raise ValueError("prepared start/end indices must summarize support runs")
        norm = float(self.l2)
        if not math.isfinite(norm) or norm < 0.0:
            raise ValueError("prepared vector norm must be finite and non-negative")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "start_idx", expected_start)
        object.__setattr__(self, "end_idx", expected_end)
        object.__setattr__(self, "l2", norm)
        object.__setattr__(self, "support_runs", runs)


@dataclass(frozen=True, slots=True)
class PreparedPairRow:
    """Typed worker result shared by RAW and baseline-corrected packs."""

    source: ReferenceBuildSource
    identity: ReferenceIdentity
    raw: PreparedReferenceVector
    baseline: PreparedReferenceVector
    baseline_applied: bool
    failure_category: str = ""
    error: str = ""

    @property
    def valid(self) -> bool:
        return not self.failure_category


@dataclass(frozen=True, slots=True)
class PairBuildReport:
    """Auditable result of a completed or concurrently reused pair build."""

    raw_directory: Path
    baseline_directory: Path
    commit: PairCommitManifest
    reused_existing: bool = False
    quarantined: tuple[QuarantinedCacheEntry, ...] = ()

    @property
    def valid_rows(self) -> int:
        return self.commit.valid_rows

    @property
    def failed_rows(self) -> int:
        return self.commit.failed_rows

    @property
    def failure_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.commit.failure_counts))


@dataclass(frozen=True, slots=True)
class PairBuildProgress:
    """Read-only progress emitted by the paired-cache build coordinator.

    Callbacks run only on the coordinating thread, including when individual
    references are prepared by worker threads.  A UI can therefore report an
    exact completed/total source count without reading partially written cache
    files or weakening commit-last publication.
    """

    stage: str
    completed_sources: int = 0
    total_sources: int = 0
    valid_rows: int = 0
    failed_rows: int = 0
    current_source: str = ""

    def __post_init__(self) -> None:
        stage = str(self.stage).strip().casefold().replace("-", "_")
        completed = int(self.completed_sources)
        total = int(self.total_sources)
        valid = int(self.valid_rows)
        failed = int(self.failed_rows)
        if not stage:
            raise ValueError("pair-build progress stage must not be empty")
        if min(completed, total, valid, failed) < 0:
            raise ValueError("pair-build progress counts must be non-negative")
        if total and completed > total:
            raise ValueError("completed pair-build sources cannot exceed the total")
        if valid + failed > completed:
            raise ValueError("valid/failed pair-build rows cannot exceed completed sources")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "completed_sources", completed)
        object.__setattr__(self, "total_sources", total)
        object.__setattr__(self, "valid_rows", valid)
        object.__setattr__(self, "failed_rows", failed)
        object.__setattr__(self, "current_source", str(self.current_source).strip())

    @property
    def fraction(self) -> float:
        if self.total_sources <= 0:
            return 0.0
        return float(np.clip(self.completed_sources / self.total_sources, 0.0, 1.0))


PairBuildProgressCallback = Callable[[PairBuildProgress], None]


def _emit_pair_build_progress(
    callback: PairBuildProgressCallback | None,
    stage: str,
    *,
    completed_sources: int = 0,
    total_sources: int = 0,
    valid_rows: int = 0,
    failed_rows: int = 0,
    current_source: str = "",
) -> None:
    """Emit best-effort observer state without making UI failure scientific failure."""

    if callback is None:
        return
    progress = PairBuildProgress(
        stage=stage,
        completed_sources=completed_sources,
        total_sources=total_sources,
        valid_rows=valid_rows,
        failed_rows=failed_rows,
        current_source=current_source,
    )
    try:
        callback(progress)
    except Exception:
        # Progress is an observational side channel. A disconnected UI must
        # not leave an otherwise valid cache pair unpublished or quarantined.
        return


def _prepare_reference_vector(
    x: np.ndarray,
    y: np.ndarray,
    request: PairBuildRequest,
    *,
    apply_baseline: bool,
    preprocess_vector: VectorPreprocessor,
    clean_xy: CleanXY,
    support_slices: SupportSliceProvider,
) -> PreparedReferenceVector:
    grid_values = request.grid.values(dtype=np.float32)
    prepared = np.asarray(
        preprocess_vector(x, y, grid_values, bool(apply_baseline)),
        dtype=np.float32,
    ).reshape(-1)
    if prepared.size != request.grid.length:
        raise ValueError(
            f"prepared vector has {prepared.size} points; expected {request.grid.length}"
        )

    clean_x, _clean_y = clean_xy(x, y)
    clean_axis = np.asarray(clean_x, dtype=float).reshape(-1)
    runs: list[tuple[int, int]] = []
    for support_slice in support_slices(clean_axis):
        segment = clean_axis[support_slice]
        if segment.size == 0:
            continue
        start = int(
            max(
                0,
                math.ceil(
                    (float(segment[0]) - request.grid.minimum) / request.grid.step
                ),
            )
        )
        end = int(
            min(
                request.grid.length - 1,
                math.floor(
                    (float(segment[-1]) - request.grid.minimum) / request.grid.step
                ),
            )
        )
        if end < start or end < 0 or start >= request.grid.length:
            continue
        if runs and start <= runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((start, end))
    frozen_runs = tuple(runs)
    return PreparedReferenceVector(
        values=prepared,
        start_idx=frozen_runs[0][0] if frozen_runs else 0,
        end_idx=frozen_runs[-1][1] if frozen_runs else -1,
        l2=float(np.linalg.norm(prepared)),
        support_runs=frozen_runs,
    )


def _source_provenance(source: ReferenceBuildSource) -> CachedProvenance:
    database_hint = source.source.root_alias
    try:
        parsed = parse_spectrum_provenance(
            source.path,
            database_hint=database_hint or None,
        )
    except Exception:
        return CachedProvenance(database=database_hint)
    return CachedProvenance.from_spectrum(parsed)


def _source_identity(
    source: ReferenceBuildSource,
    provenance: CachedProvenance,
    request: PairBuildRequest,
    extract_elements: ElementExtractor,
    *,
    skipped: bool = False,
) -> ReferenceIdentity:
    elements = (
        ()
        if skipped
        else tuple(sorted({str(value) for value in extract_elements(source.formula)}))
    )
    return ReferenceIdentity(
        name="SKIPPED" if skipped else source.name,
        formula="?" if skipped else source.formula,
        flag="" if skipped else source.flag,
        filename="" if skipped else source.filename,
        orig_filename=source.orig_filename,
        source=source.source,
        elements=elements,
        has_formula=bool(elements),
        parser_version=request.parser_version,
        source_label=source.source_label,
        provenance=provenance,
    )


def _zero_prepared_vector(grid_length: int) -> PreparedReferenceVector:
    return PreparedReferenceVector(
        values=np.zeros(grid_length, dtype=np.float32),
        start_idx=0,
        end_idx=-1,
        l2=0.0,
        support_runs=(),
    )


def _failed_pair_row(
    source: ReferenceBuildSource,
    provenance: CachedProvenance,
    request: PairBuildRequest,
    extract_elements: ElementExtractor,
    category: str,
    message: str,
) -> PreparedPairRow:
    zero = _zero_prepared_vector(request.grid.length)
    return PreparedPairRow(
        source=source,
        identity=_source_identity(
            source,
            provenance,
            request,
            extract_elements,
            skipped=True,
        ),
        raw=zero,
        baseline=zero,
        baseline_applied=False,
        failure_category=category,
        error=message,
    )


def _source_is_processed(
    source: ReferenceBuildSource,
    provenance: CachedProvenance,
) -> bool:
    folded_name = source.path.name.casefold()
    if "raman_data_processed" in folded_name or "raman processed" in folded_name:
        return True
    if "raman_data_raw" in folded_name or "raman raw" in folded_name:
        return False
    return provenance.processing == "processed"


def _process_pair_source(
    source: ReferenceBuildSource,
    request: PairBuildRequest,
    *,
    parse_spectrum: SpectrumParser,
    preprocess_vector: VectorPreprocessor,
    clean_xy: CleanXY,
    support_slices: SupportSliceProvider,
    extract_elements: ElementExtractor,
) -> PreparedPairRow:
    provenance = _source_provenance(source)
    try:
        identity = _source_identity(source, provenance, request, extract_elements)
    except Exception as exc:
        category = f"metadata:{type(exc).__name__}"
        return _failed_pair_row(
            source,
            provenance,
            request,
            extract_elements,
            category,
            f"{category}: {exc}",
        )
    try:
        x, y = parse_spectrum(source.path)
    except Exception as exc:
        category = f"parse:{type(exc).__name__}"
        return _failed_pair_row(
            source,
            provenance,
            request,
            extract_elements,
            category,
            f"{category}: {exc}",
        )
    try:
        raw = _prepare_reference_vector(
            x,
            y,
            request,
            apply_baseline=False,
            preprocess_vector=preprocess_vector,
            clean_xy=clean_xy,
            support_slices=support_slices,
        )
    except Exception as exc:
        category = f"raw-preprocess:{type(exc).__name__}"
        return _failed_pair_row(
            source,
            provenance,
            request,
            extract_elements,
            category,
            f"{category}: {exc}",
        )
    try:
        if _source_is_processed(source, provenance):
            baseline = raw
            baseline_applied = False
        else:
            baseline = _prepare_reference_vector(
                x,
                y,
                request,
                apply_baseline=True,
                preprocess_vector=preprocess_vector,
                clean_xy=clean_xy,
                support_slices=support_slices,
            )
            baseline_applied = True
    except Exception as exc:
        category = f"baseline-preprocess:{type(exc).__name__}"
        return _failed_pair_row(
            source,
            provenance,
            request,
            extract_elements,
            category,
            f"{category}: {exc}",
        )

    row = PreparedPairRow(
        source=source,
        identity=identity,
        raw=raw,
        baseline=baseline,
        baseline_applied=baseline_applied,
    )
    if raw.l2 <= 0.0 or not raw.support_runs:
        category = "validation:unmatchable-row"
        return _failed_pair_row(
            source,
            provenance,
            request,
            extract_elements,
            category,
            f"{category}: empty support or zero norm",
        )
    return row


def _pair_row_metadata(
    row: PreparedPairRow,
    *,
    baseline: bool,
) -> dict[str, Any]:
    identity = row.identity
    vector = row.baseline if baseline else row.raw
    stats = ReferenceVectorStats(
        start_idx=vector.start_idx,
        end_idx=vector.end_idx,
        l2=vector.l2,
        db_baseline=bool(row.baseline_applied) if baseline else False,
        support_runs=vector.support_runs,
        error=row.error,
    )
    metadata: dict[str, Any] = {
        "name": identity.name,
        "formula": identity.formula,
        "flag": identity.flag,
        "filename": identity.filename,
        "orig_filename": identity.orig_filename,
        "path": str(row.source.path),
        "start_idx": stats.start_idx,
        "end_idx": stats.end_idx,
        "l2": stats.l2,
        "elements": list(identity.elements),
        "has_formula": identity.has_formula,
        "parser_version": identity.parser_version,
        "source": identity.source_label,
        "db_baseline": stats.db_baseline,
        "support_runs": stats.support_runs,
        **identity.source.as_metadata_fields(),
    }
    if identity.provenance.available:
        metadata["provenance"] = identity.provenance.as_dict()
    if stats.error:
        metadata["error"] = stats.error
    return metadata


def _recover_failed_pair_build(
    request: PairBuildRequest,
    paths: Sequence[Path | None],
    error: BaseException,
) -> tuple[QuarantinedCacheEntry, ...]:
    if committed_pair_available(
        request.cache_root,
        request.raw_signature,
        request.baseline_signature,
    ):
        return ()
    targets = tuple(path for path in paths if path is not None and path.exists())
    if not targets:
        return ()
    try:
        return quarantine_cache_directories(request.cache_root, targets)
    except Exception as recovery_error:
        if hasattr(error, "add_note"):
            error.add_note(f"Failed cache staging recovery: {recovery_error}")
        return ()


def _build_precompute_pair_unlocked(
    request: PairBuildRequest,
    *,
    discover_references: ReferenceDiscovery,
    parse_spectrum: SpectrumParser,
    preprocess_vector: VectorPreprocessor,
    clean_xy: CleanXY,
    support_slices: SupportSliceProvider,
    extract_elements: ElementExtractor,
    prior_quarantine: tuple[QuarantinedCacheEntry, ...] = (),
    progress_callback: PairBuildProgressCallback | None = None,
) -> PairBuildReport:
    raw_stage: Path | None = None
    baseline_stage: Path | None = None
    raw_matrix: np.memmap | None = None
    baseline_matrix: np.memmap | None = None
    row_count = 0
    completed_sources = 0
    valid_rows = 0
    failure_counts: dict[str, int] = {}
    _emit_pair_build_progress(progress_callback, "discovering")
    try:
        discovered = discover_references(request.source_paths)
        sources = tuple(
            ReferenceBuildSource.from_mapping(value, request.roots)
            for value in discovered
        )
        if not sources:
            raise PairBuildError("Precompute: no references found")
        row_count = len(sources)
        _emit_pair_build_progress(
            progress_callback,
            "processing",
            total_sources=row_count,
        )
        raw_stage = create_cache_staging_directory(request.raw_directory)
        baseline_stage = create_cache_staging_directory(request.baseline_directory)
        try:
            raw_matrix = np.memmap(
                raw_stage / DEFAULT_VECTOR_FILE,
                mode="w+",
                dtype=np.float32,
                shape=(row_count, request.grid.length),
            )
            baseline_matrix = np.memmap(
                baseline_stage / DEFAULT_VECTOR_FILE,
                mode="w+",
                dtype=np.float32,
                shape=(row_count, request.grid.length),
            )
            raw_metadata: list[dict[str, Any] | None] = [None] * row_count
            baseline_metadata: list[dict[str, Any] | None] = [None] * row_count

            def store(index: int, row: PreparedPairRow) -> None:
                nonlocal completed_sources, valid_rows
                assert raw_matrix is not None and baseline_matrix is not None
                raw_matrix[index, :] = row.raw.values
                baseline_matrix[index, :] = row.baseline.values
                raw_metadata[index] = _pair_row_metadata(row, baseline=False)
                baseline_metadata[index] = _pair_row_metadata(row, baseline=True)
                if row.valid:
                    valid_rows += 1
                else:
                    failure_counts[row.failure_category] = (
                        failure_counts.get(row.failure_category, 0) + 1
                    )
                completed_sources += 1
                _emit_pair_build_progress(
                    progress_callback,
                    "processing",
                    completed_sources=completed_sources,
                    total_sources=row_count,
                    valid_rows=valid_rows,
                    failed_rows=completed_sources - valid_rows,
                    current_source=(
                        row.source.orig_filename or row.source.path.name
                    ),
                )

            def process(source: ReferenceBuildSource) -> PreparedPairRow:
                return _process_pair_source(
                    source,
                    request,
                    parse_spectrum=parse_spectrum,
                    preprocess_vector=preprocess_vector,
                    clean_xy=clean_xy,
                    support_slices=support_slices,
                    extract_elements=extract_elements,
                )

            if request.workers == 1 or row_count == 1:
                for index, source in enumerate(sources):
                    store(index, process(source))
            else:
                with ThreadPoolExecutor(max_workers=request.workers) as pool:
                    source_iter = iter(enumerate(sources))
                    futures: dict[Any, int] = {}
                    for _ in range(min(row_count, request.workers * 2)):
                        try:
                            index, source = next(source_iter)
                        except StopIteration:
                            break
                        futures[pool.submit(process, source)] = index
                    while futures:
                        future = next(as_completed(tuple(futures)))
                        index = futures.pop(future)
                        store(index, future.result())
                        try:
                            next_index, next_source = next(source_iter)
                        except StopIteration:
                            continue
                        futures[pool.submit(process, next_source)] = next_index
        finally:
            if raw_matrix is not None:
                raw_matrix.flush()
            if baseline_matrix is not None:
                baseline_matrix.flush()
            raw_matrix = None
            baseline_matrix = None

        _emit_pair_build_progress(
            progress_callback,
            "validating",
            completed_sources=completed_sources,
            total_sources=row_count,
            valid_rows=valid_rows,
            failed_rows=completed_sources - valid_rows,
        )
        if any(value is None for value in raw_metadata + baseline_metadata):
            raise PairBuildError("Precompute: incomplete paired worker results")
        if valid_rows <= 0:
            summary = ", ".join(
                f"{name}={count}" for name, count in sorted(failure_counts.items())
            )
            raise PairBuildError(
                "Refusing to publish a cache pair with no valid reference rows"
                + (f" ({summary})" if summary else "")
            )
        if failure_counts:
            dominant_name, dominant_count = max(
                failure_counts.items(), key=lambda item: item[1]
            )
            dominant_fraction = dominant_count / row_count
            if dominant_fraction >= request.systemic_failure_fraction:
                failure_kind = (
                    "preprocessing"
                    if dominant_name.startswith(
                        ("raw-preprocess:", "baseline-preprocess:")
                    )
                    else "reference-build"
                )
                raise PairBuildError(
                    f"Refusing to publish a likely systemic {failure_kind} failure: "
                    f"{dominant_name} affected {dominant_count}/{row_count} rows "
                    f"({dominant_fraction:.1%}, limit "
                    f"{request.systemic_failure_fraction:.1%})"
                )

        raw_rows = [value for value in raw_metadata if value is not None]
        baseline_rows = [value for value in baseline_metadata if value is not None]
        _emit_pair_build_progress(
            progress_callback,
            "writing",
            completed_sources=completed_sources,
            total_sources=row_count,
            valid_rows=valid_rows,
            failed_rows=completed_sources - valid_rows,
        )
        for directory, rows in (
            (raw_stage, raw_rows),
            (baseline_stage, baseline_rows),
        ):
            (directory / DEFAULT_GRID_FILE).write_text(
                json.dumps(request.grid_metadata, sort_keys=True),
                encoding="utf-8",
            )
            (directory / DEFAULT_METADATA_FILE).write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )
        vector_bytes = (
            row_count * request.grid.length * np.dtype(np.float32).itemsize
        )
        write_cache_manifest(
            raw_stage,
            CacheManifest.create(
                signature=request.raw_signature,
                inventory_signature=request.inventory_signature,
                variant=request.raw_variant,
                grid=request.grid,
                metadata_rows=row_count,
                vector_bytes=vector_bytes,
            ),
        )
        write_cache_manifest(
            baseline_stage,
            CacheManifest.create(
                signature=request.baseline_signature,
                inventory_signature=request.inventory_signature,
                variant=request.baseline_variant,
                grid=request.grid,
                metadata_rows=row_count,
                vector_bytes=vector_bytes,
            ),
        )
        _emit_pair_build_progress(
            progress_callback,
            "publishing",
            completed_sources=completed_sources,
            total_sources=row_count,
            valid_rows=valid_rows,
            failed_rows=completed_sources - valid_rows,
        )
        promote_cache_directory(raw_stage, request.raw_directory)
        promote_cache_directory(baseline_stage, request.baseline_directory)
        commit = PairCommitManifest.create(
            raw_signature=request.raw_signature,
            baseline_signature=request.baseline_signature,
            inventory_signature=request.inventory_signature,
            grid=request.grid,
            row_count=row_count,
            valid_rows=valid_rows,
            failure_counts=failure_counts,
        )
        write_pair_commit(request.cache_root, commit)
        _emit_pair_build_progress(
            progress_callback,
            "complete",
            completed_sources=row_count,
            total_sources=row_count,
            valid_rows=commit.valid_rows,
            failed_rows=commit.failed_rows,
        )
        return PairBuildReport(
            raw_directory=request.raw_directory,
            baseline_directory=request.baseline_directory,
            commit=commit,
            quarantined=prior_quarantine,
        )
    except Exception as exc:
        if committed_pair_available(
            request.cache_root,
            request.raw_signature,
            request.baseline_signature,
        ):
            commit = read_pair_commit(
                request.cache_root,
                request.raw_signature,
                request.baseline_signature,
                required=True,
            )
            assert commit is not None
            _emit_pair_build_progress(
                progress_callback,
                "complete",
                completed_sources=commit.row_count,
                total_sources=commit.row_count,
                valid_rows=commit.valid_rows,
                failed_rows=commit.failed_rows,
            )
            return PairBuildReport(
                raw_directory=request.raw_directory,
                baseline_directory=request.baseline_directory,
                commit=commit,
                reused_existing=True,
                quarantined=prior_quarantine,
            )
        _emit_pair_build_progress(
            progress_callback,
            "failed",
            completed_sources=completed_sources,
            total_sources=row_count,
            valid_rows=valid_rows,
            failed_rows=completed_sources - valid_rows,
        )
        _recover_failed_pair_build(
            request,
            (
                raw_stage,
                baseline_stage,
                request.raw_directory,
                request.baseline_directory,
            ),
            exc,
        )
        raise


def build_precompute_pair(
    request: PairBuildRequest,
    *,
    discover_references: ReferenceDiscovery,
    parse_spectrum: SpectrumParser,
    preprocess_vector: VectorPreprocessor,
    clean_xy: CleanXY,
    support_slices: SupportSliceProvider,
    extract_elements: ElementExtractor,
    progress_callback: PairBuildProgressCallback | None = None,
) -> PairBuildReport:
    """Serialize one pair build and publish only after a commit-last marker."""

    _emit_pair_build_progress(progress_callback, "waiting_for_lock")
    with pair_build_lock(
        request.cache_root,
        request.raw_signature,
        request.baseline_signature,
    ):
        if committed_pair_available(
            request.cache_root,
            request.raw_signature,
            request.baseline_signature,
        ):
            current = read_pair_commit(
                request.cache_root,
                request.raw_signature,
                request.baseline_signature,
                required=True,
            )
            if (
                current is not None
                and current.created_at_ns != request.invalid_commit_created_at_ns
            ):
                _emit_pair_build_progress(
                    progress_callback,
                    "complete",
                    completed_sources=current.row_count,
                    total_sources=current.row_count,
                    valid_rows=current.valid_rows,
                    failed_rows=current.failed_rows,
                )
                return PairBuildReport(
                    raw_directory=request.raw_directory,
                    baseline_directory=request.baseline_directory,
                    commit=current,
                    reused_existing=True,
                )
        _emit_pair_build_progress(progress_callback, "recovering")
        quarantined = quarantine_cache_directories(
            request.cache_root,
            (request.raw_directory, request.baseline_directory),
        )
        return _build_precompute_pair_unlocked(
            request,
            discover_references=discover_references,
            parse_spectrum=parse_spectrum,
            preprocess_vector=preprocess_vector,
            clean_xy=clean_xy,
            support_slices=support_slices,
            extract_elements=extract_elements,
            prior_quarantine=quarantined,
            progress_callback=progress_callback,
        )


__all__ = [
    "CACHE_MANIFEST_SCHEMA_VERSION",
    "PAIR_COMMIT_SCHEMA_VERSION",
    "DEFAULT_GRID_FILE",
    "DEFAULT_HNSW_FILE",
    "DEFAULT_MANIFEST_FILE",
    "DEFAULT_METADATA_FILE",
    "DEFAULT_VECTOR_FILE",
    "CacheCleanupPlan",
    "CacheEntryInfo",
    "CacheManifest",
    "CacheValidationError",
    "CachedProvenance",
    "ChemistryProvenance",
    "CleanupCandidate",
    "CorrectionStep",
    "DatabaseError",
    "DatabaseFileRecord",
    "DatabaseInventory",
    "DatabaseInventoryManager",
    "DatabaseRoot",
    "GridSpec",
    "InventoryError",
    "InventoryIssue",
    "MetadataIndex",
    "PairBuildError",
    "PairBuildProgress",
    "PairBuildProgressCallback",
    "PairBuildReport",
    "PairBuildRequest",
    "PairCommitManifest",
    "PortableSource",
    "PrecomputePack",
    "PrecomputePair",
    "PreparedPairRow",
    "PreparedReferenceVector",
    "ReferenceBuildSource",
    "ReferenceCatalogSummary",
    "ReferenceEligibilityRequest",
    "ReferenceEligibilityResult",
    "ReferenceIdentity",
    "ReferenceVectorStats",
    "MatcherMetadataRow",
    "MatcherMetadataSequence",
    "QuarantinedCacheEntry",
    "SourceResolutionError",
    "SpectrumProvenance",
    "atomic_cache_directory",
    "build_precompute_pair",
    "coerce_database_roots",
    "committed_pair_available",
    "compute_reference_eligibility",
    "create_cache_staging_directory",
    "derive_precompute_signature",
    "inspect_cache_entries",
    "load_precompute_pack",
    "load_precompute_pair",
    "pair_build_lock",
    "pair_commit_path",
    "pair_lock_path",
    "plan_cache_cleanup",
    "quarantine_cleanup_candidates",
    "quarantine_cache_directories",
    "portable_source_for_path",
    "parse_rod_provenance",
    "parse_rruff_provenance",
    "parse_spectrum_provenance",
    "promote_cache_directory",
    "read_cache_manifest",
    "read_pair_commit",
    "scan_database_inventory",
    "write_cache_manifest",
    "write_pair_commit",
    "validate_pair_commit_files",
]

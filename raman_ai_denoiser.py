"""DeepeR ResUNet inference used by the RamanSPy denoising example.

The network architecture and pretrained checkpoint originate from
``conor-horgan/DeepeR`` (MIT license).  PyTorch is imported lazily so the rest
of RamanPhaseID remains importable when the optional AI runtime is unavailable.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable
import urllib.request

import numpy as np


MODEL_ID = "DeepeR-ResUNet-500"
MODEL_POINTS = 500
MODEL_RANGE_CM1 = (500.0, 1800.0)
MODEL_WINDOW_SPAN_CM1 = MODEL_RANGE_CM1[1] - MODEL_RANGE_CM1[0]
MODEL_WINDOW_OVERLAP_CM1 = 0.25 * MODEL_WINDOW_SPAN_CM1
GUARD_GUIDE_SPAN_CM1 = 10.0
GUARD_TREND_SPAN_CM1 = 20.0
GUARD_MAX_DYNAMIC_RANGE_FRACTION = 0.02
DEFAULT_MAX_CHANGE_SIGMA = 1.0
FULL_RANGE_ADAPTER_VERSION = 1
MODEL_REPOSITORY = "https://github.com/conor-horgan/DeepeR"
MODEL_COMMIT = "87da149b2cdc8b4d98af60f6211f3b35d3c21493"
MODEL_URL = (
    "https://raw.githubusercontent.com/conor-horgan/DeepeR/"
    f"{MODEL_COMMIT}/Raman%20Spectral%20Denoising/ResUNet.pt"
)
MODEL_SHA256 = "23d11061fce98656f32f8d604d2e58973853a3f79ce69e9f08dac4d8ef9747b2"
MODEL_SIZE_BYTES = 8_342_235
MODEL_FILENAME = "deeper_resunet_500_87da149.pt"
MODEL_PATH_ENV = "RAMANPHASEID_DEEPER_MODEL"


class AIDenoiserError(RuntimeError):
    """Base class for actionable AI-denoising failures."""


class AIDenoiserDependencyError(AIDenoiserError):
    """Raised when the PyTorch inference runtime is not installed."""


class AIDenoiserModelError(AIDenoiserError):
    """Raised when the checkpoint cannot be obtained or verified."""


class AIDenoiserInputError(AIDenoiserError):
    """Raised when a spectrum is outside the pretrained model contract."""


def _default_cache_dir() -> Path:
    if os.name == "nt":
        root = os.getenv("LOCALAPPDATA", "").strip()
        if root:
            return Path(root) / "RamanPhaseID" / "models"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "RamanPhaseID" / "models"

    root = os.getenv("XDG_CACHE_HOME", "").strip()
    if root:
        return Path(root) / "RamanPhaseID" / "models"
    return Path.home() / ".cache" / "RamanPhaseID" / "models"


def model_path() -> Path:
    """Return the configured checkpoint path without touching the network."""
    override = os.getenv(MODEL_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_cache_dir() / MODEL_FILENAME


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_is_ready(path: Path | None = None) -> bool:
    """Return whether a local checkpoint has the expected size and digest."""
    candidate = model_path() if path is None else Path(path)
    try:
        return (
            candidate.is_file()
            and candidate.stat().st_size == MODEL_SIZE_BYTES
            and _file_sha256(candidate) == MODEL_SHA256
        )
    except OSError:
        return False


def ensure_model(path: Path | None = None, *, timeout: float = 45.0) -> Path:
    """Return a verified checkpoint, downloading it atomically if necessary."""
    candidate = model_path() if path is None else Path(path)
    if model_is_ready(candidate):
        return candidate

    override = os.getenv(MODEL_PATH_ENV, "").strip()
    if path is None and override:
        raise AIDenoiserModelError(
            f"{MODEL_PATH_ENV} points to a missing or unverified checkpoint: {candidate}. "
            f"Expected SHA-256 {MODEL_SHA256}."
        )

    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AIDenoiserModelError(
            f"Cannot create the AI model cache directory {candidate.parent}: {exc}"
        ) from exc

    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "RamanPhaseID-DeepeR-model-loader/1"},
    )
    temp_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            advertised = response.headers.get("Content-Length")
            if advertised is not None and int(advertised) != MODEL_SIZE_BYTES:
                raise AIDenoiserModelError(
                    "The DeepeR server returned an unexpected model size "
                    f"({advertised} bytes; expected {MODEL_SIZE_BYTES})."
                )

            digest = hashlib.sha256()
            total = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{MODEL_FILENAME}.",
                suffix=".part",
                dir=candidate.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MODEL_SIZE_BYTES:
                        raise AIDenoiserModelError(
                            "The downloaded DeepeR checkpoint is larger than expected."
                        )
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())

        if total != MODEL_SIZE_BYTES or digest.hexdigest() != MODEL_SHA256:
            raise AIDenoiserModelError(
                "The downloaded DeepeR checkpoint failed its size/SHA-256 check; "
                "it was not installed."
            )
        os.replace(temp_path, candidate)
        temp_path = None
        return candidate
    except AIDenoiserModelError:
        raise
    except Exception as exc:
        raise AIDenoiserModelError(
            "Could not download the pretrained DeepeR checkpoint from GitHub. "
            f"Check the network connection or set {MODEL_PATH_ENV} to a verified local copy. "
            f"Details: {exc}"
        ) from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise AIDenoiserDependencyError(
            "AI denoising requires PyTorch. Install the project requirements "
            "(`pip install -r requirements.txt`) and restart RamanPhaseID."
        ) from exc
    return torch


def _build_resunet(torch):
    """Build the exact 500-channel 1D ResUNet from the DeepeR repository."""
    nn = torch.nn

    class BasicConv(nn.Module):
        def __init__(self, channels_in: int, channels_out: int, batch_norm: bool):
            super().__init__()
            body = [
                nn.Conv1d(
                    channels_in,
                    channels_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PReLU(),
            ]
            if batch_norm:
                body.append(nn.BatchNorm1d(channels_out))
            self.body = nn.Sequential(*body)

        def forward(self, x):
            return self.body(x)

    class ResUNetConv(nn.Module):
        def __init__(self, num_convs: int, channels: int, batch_norm: bool):
            super().__init__()
            body = []
            for _ in range(num_convs):
                body.extend(
                    [
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                        nn.PReLU(),
                    ]
                )
                if batch_norm:
                    body.append(nn.BatchNorm1d(channels))
            self.body = nn.Sequential(*body)

        def forward(self, x):
            return self.body(x) + x

    class UNetLinear(nn.Module):
        def __init__(self, repeats: int, channels_in: int, channels_out: int):
            super().__init__()
            body = []
            for _ in range(repeats):
                body.extend([nn.Linear(channels_in, channels_out), nn.PReLU()])
            self.body = nn.Sequential(*body)

        def forward(self, x):
            return self.body(x)

    class ResUNet(nn.Module):
        def __init__(self, num_convs: int, batch_norm: bool):
            super().__init__()
            self.conv1 = nn.Sequential(
                BasicConv(1, 64, batch_norm),
                ResUNetConv(num_convs, 64, batch_norm),
            )
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Sequential(
                BasicConv(64, 128, batch_norm),
                ResUNetConv(num_convs, 128, batch_norm),
            )
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Sequential(
                BasicConv(128, 256, batch_norm),
                ResUNetConv(num_convs, 256, batch_norm),
                BasicConv(256, 128, batch_norm),
            )
            self.up3 = nn.Upsample(scale_factor=2)
            self.conv4 = nn.Sequential(
                BasicConv(256, 128, batch_norm),
                ResUNetConv(num_convs, 128, batch_norm),
                BasicConv(128, 64, batch_norm),
            )
            self.up4 = nn.Upsample(scale_factor=2)
            self.conv5 = nn.Sequential(
                BasicConv(128, 64, batch_norm),
                ResUNetConv(num_convs, 64, batch_norm),
            )
            self.conv6 = nn.Sequential(BasicConv(64, 1, batch_norm))
            self.linear7 = UNetLinear(3, MODEL_POINTS, MODEL_POINTS)

        def forward(self, x):
            x = self.conv1(x)
            x1 = self.pool1(x)

            x2 = self.conv2(x1)
            x3 = self.pool1(x2)
            x3 = self.conv3(x3)
            x3 = self.up3(x3)

            x4 = torch.cat((x2, x3), dim=1)
            x4 = self.conv4(x4)
            x5 = self.up4(x4)

            x6 = torch.cat((x, x5), dim=1)
            x6 = self.conv5(x6)
            x7 = self.conv6(x6)
            return self.linear7(x7)

    return ResUNet(3, False).float()


@lru_cache(maxsize=2)
def _load_model_cached(path_text: str, size: int, mtime_ns: int):
    del size, mtime_ns  # only used to invalidate this cache when the file changes
    torch = _import_torch()
    model = _build_resunet(torch)
    try:
        state = torch.load(path_text, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise AIDenoiserModelError(
            f"The verified DeepeR checkpoint could not be loaded: {exc}"
        ) from exc
    model.eval()
    return model


def load_model(path: Path | None = None):
    checkpoint = ensure_model(path)
    stat = checkpoint.stat()
    return _load_model_cached(str(checkpoint), stat.st_size, stat.st_mtime_ns)


def _run_pretrained_model_batch(
    values: np.ndarray,
    path: Path | None = None,
) -> np.ndarray:
    torch = _import_torch()
    model = load_model(path)
    values32 = np.ascontiguousarray(values, dtype=np.float32)
    if values32.ndim == 1:
        values32 = values32.reshape(1, -1)
    if values32.ndim != 2 or values32.shape[1] != MODEL_POINTS:
        raise AIDenoiserModelError(
            f"DeepeR inference requires rows of {MODEL_POINTS} values; got {values32.shape}."
        )
    tensor = torch.from_numpy(values32).reshape(values32.shape[0], 1, MODEL_POINTS)
    try:
        with torch.inference_mode():
            output = model(tensor).detach().cpu().numpy().reshape(values32.shape[0], MODEL_POINTS)
    except Exception as exc:
        raise AIDenoiserModelError(f"DeepeR inference failed: {exc}") from exc
    return np.asarray(output, dtype=float)


def _run_pretrained_model(values: np.ndarray, path: Path | None = None) -> np.ndarray:
    return _run_pretrained_model_batch(values, path)[0]


def _full_support_windows(axis: np.ndarray) -> list[tuple[float, float]]:
    """Cover the complete axis with overlapping windows of the training span."""
    low = float(axis[0])
    high = float(axis[-1])
    width = high - low
    span = float(MODEL_WINDOW_SPAN_CM1)
    if width <= span:
        return [(low, high)]

    max_step = span - float(MODEL_WINDOW_OVERLAP_CM1)
    window_count = int(np.ceil((width - span) / max_step)) + 1
    starts = np.linspace(low, high - span, window_count, dtype=float)
    return [(float(start), float(start + span)) for start in starts]


def _window_weights(
    positions: np.ndarray,
    windows: list[tuple[float, float]],
    index: int,
) -> np.ndarray:
    """Cosine cross-fades for a window while retaining both global edges."""
    start, end = windows[index]
    weight = np.ones(positions.size, dtype=float)
    if index > 0:
        previous_end = windows[index - 1][1]
        overlap = previous_end - start
        if overlap > 0.0:
            in_overlap = positions < previous_end
            phase = np.clip((positions[in_overlap] - start) / overlap, 0.0, 1.0)
            weight[in_overlap] *= 0.5 - (0.5 * np.cos(np.pi * phase))
    if index + 1 < len(windows):
        next_start = windows[index + 1][0]
        overlap = end - next_start
        if overlap > 0.0:
            in_overlap = positions > next_start
            phase = np.clip((end - positions[in_overlap]) / overlap, 0.0, 1.0)
            weight[in_overlap] *= 0.5 - (0.5 * np.cos(np.pi * phase))
    return weight


def _physical_savgol(
    axis: np.ndarray,
    values: np.ndarray,
    span_cm1: float,
    *,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky–Golay helper with a window specified in physical units."""
    if values.size < 3:
        return values.copy()
    spacing = float(np.median(np.diff(axis)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return values.copy()
    window = max(3, int(round(float(span_cm1) / spacing)) + 1)
    if window % 2 == 0:
        window += 1
    maximum = values.size if values.size % 2 == 1 else values.size - 1
    window = min(window, maximum)
    if window < 3:
        return values.copy()
    order = min(max(0, int(polyorder)), window - 1)
    try:
        from scipy.signal import savgol_filter

        return np.asarray(savgol_filter(values, window, order, mode="interp"), dtype=float)
    except Exception:
        return values.copy()


def estimate_noise_sigma(axis: np.ndarray, values: np.ndarray) -> float:
    """Robustly estimate point noise while discounting sparse Raman peaks."""
    axis_arr = np.asarray(axis, dtype=float).reshape(-1)
    signal = np.asarray(values, dtype=float).reshape(-1)
    if axis_arr.size != signal.size or signal.size < 3:
        return 0.0

    differences = np.diff(signal)
    diff_median = float(np.median(differences))
    sigma_diff = (
        1.4826
        * float(np.median(np.abs(differences - diff_median)))
        / np.sqrt(2.0)
    )
    guide = _physical_savgol(axis_arr, signal, GUARD_GUIDE_SPAN_CM1)
    residual = signal - guide
    residual_median = float(np.median(residual))
    sigma_residual = 1.4826 * float(np.median(np.abs(residual - residual_median)))
    valid = [
        value
        for value in (sigma_diff, sigma_residual)
        if np.isfinite(value) and value > 0.0
    ]
    return float(min(valid)) if valid else 0.0


def _guard_full_range_candidate(
    axis: np.ndarray,
    signal: np.ndarray,
    candidate: np.ndarray,
    *,
    max_change_sigma: float,
) -> np.ndarray:
    """Accept only bounded, high-frequency, smoothing-consistent corrections.

    A raw out-of-domain DeepeR prediction is never returned. Its low-frequency
    difference from the measurement is removed, preventing learned biological
    backgrounds from entering mineral spectra. A correction is then accepted
    only where it moves toward a conservative 10 cm⁻¹ Savitzky–Golay guide, and
    its pointwise magnitude is capped by both the measured noise and 2% of the
    complete signal range. Thus the AI can help identify noise-like residuals,
    but it cannot freely redraw peaks or baselines.
    """
    delta = np.asarray(candidate, dtype=float) - signal
    delta_trend = _physical_savgol(axis, delta, GUARD_TREND_SPAN_CM1)
    high_frequency_delta = delta - delta_trend

    conservative_guide = _physical_savgol(axis, signal, GUARD_GUIDE_SPAN_CM1)
    guide_delta = conservative_guide - signal
    agrees_with_smoothing = (high_frequency_delta * guide_delta) > 0.0

    sigma = estimate_noise_sigma(axis, signal)
    dynamic_range = float(np.ptp(signal))
    sigma_limit = max(0.0, float(max_change_sigma)) * sigma
    range_limit = GUARD_MAX_DYNAMIC_RANGE_FRACTION * dynamic_range
    change_limit = min(sigma_limit, range_limit)
    if not np.isfinite(change_limit) or change_limit <= 0.0:
        return signal.copy()

    accepted = np.zeros_like(signal, dtype=float)
    accepted[agrees_with_smoothing] = (
        np.sign(guide_delta[agrees_with_smoothing])
        * np.minimum(
            np.abs(high_frequency_delta[agrees_with_smoothing]),
            np.abs(guide_delta[agrees_with_smoothing]),
        )
    )
    accepted = np.clip(accepted, -change_limit, change_limit)
    return signal + accepted


def denoise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    path: Path | None = None,
    runner: Callable[[np.ndarray], np.ndarray] | None = None,
    max_change_sigma: float = DEFAULT_MAX_CHANGE_SIGMA,
) -> np.ndarray:
    """Apply guarded DeepeR-assisted denoising over the complete input axis.

    The pretrained network still receives 500 values spanning 1300 cm⁻¹, its
    original physical input width, but overlapping windows cover every input
    point. The blended raw prediction is passed through a mandatory peak and
    background guard before anything is returned. ``runner`` exists for
    deterministic tests and receives one normalised 500-value window at a time.
    """
    axis = np.asarray(x, dtype=float).reshape(-1)
    signal = np.asarray(y, dtype=float).reshape(-1)
    if axis.size != signal.size:
        raise AIDenoiserInputError("AI denoising requires x and y with equal lengths.")
    if axis.size < 2 or not np.all(np.isfinite(axis)) or not np.all(np.isfinite(signal)):
        raise AIDenoiserInputError("AI denoising requires a finite spectrum with at least two points.")
    if np.any(np.diff(axis) <= 0.0):
        raise AIDenoiserInputError("AI denoising requires a strictly increasing Raman-shift axis.")

    signal_range = float(np.ptp(signal))
    scale_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(signal))))
    if not np.isfinite(signal_range) or signal_range <= scale_floor:
        return signal.copy()

    windows = _full_support_windows(axis)
    model_axes: list[np.ndarray] = []
    offsets: list[float] = []
    scales: list[float] = []
    normalised_windows: list[np.ndarray] = []
    for start, end in windows:
        model_axis = np.linspace(start, end, MODEL_POINTS, dtype=float)
        model_input = np.interp(model_axis, axis, signal)
        offset = float(np.min(model_input))
        scale = float(np.max(model_input) - offset)
        model_axes.append(model_axis)
        offsets.append(offset)
        scales.append(scale)
        if not np.isfinite(scale) or scale <= scale_floor:
            normalised_windows.append(np.zeros(MODEL_POINTS, dtype=float))
        else:
            normalised_windows.append((model_input - offset) / scale)

    if runner is None:
        predictions = _run_pretrained_model_batch(np.vstack(normalised_windows), path)
    else:
        predictions = np.vstack(
            [np.asarray(runner(values.copy()), dtype=float).reshape(-1) for values in normalised_windows]
        )
    if predictions.shape != (len(windows), MODEL_POINTS) or not np.all(np.isfinite(predictions)):
        raise AIDenoiserModelError(
            "The DeepeR model must return one finite 500-value prediction per full-range window; "
            f"got {predictions.shape}."
        )

    accumulated = np.zeros_like(signal, dtype=float)
    accumulated_weight = np.zeros_like(signal, dtype=float)
    for index, ((start, end), model_axis, offset, scale, prediction) in enumerate(
        zip(windows, model_axes, offsets, scales, predictions)
    ):
        active = (axis >= start) & (axis <= end)
        active_axis = axis[active]
        if scale <= scale_floor:
            restored = np.interp(model_axis, axis, signal)
        else:
            restored = prediction * scale + offset
        projected = np.interp(active_axis, model_axis, restored)
        weight = _window_weights(active_axis, windows, index)
        accumulated[active] += weight * projected
        accumulated_weight[active] += weight

    candidate = np.divide(
        accumulated,
        accumulated_weight,
        out=signal.copy(),
        where=accumulated_weight > 1e-12,
    )
    return _guard_full_range_candidate(
        axis,
        signal,
        candidate,
        max_change_sigma=max_change_sigma,
    )


def clear_model_cache() -> None:
    """Clear only the in-memory network cache (primarily useful in tests)."""
    _load_model_cached.cache_clear()

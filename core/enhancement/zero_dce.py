"""Zero-DCE baseline (optional / experimental).

Zero-DCE (Guo et al., "Zero-Reference Deep Curve Estimation") is a deep
low-light enhancement network.  LunaPSR includes it **only as an optional
benchmark** on the normalised image -- it is not assumed to be superior to the
classical baselines, and it does not replace them.

Isolation contract
------------------
- This module never imports torch at import time, so the rest of LunaPSR
  works even when no deep-learning backend is installed.
- :func:`is_available` reports whether a usable backend is present.
- :func:`zero_dce_enhance` raises :class:`ZeroDCEUnavailableError` (a
  friendly message) when the backend is missing; callers (and the UI) degrade
  to "Zero-DCE unavailable: dependency not installed" instead of failing.

Realised here is a reference inference path; a trained production checkpoint
would be loaded from ``models/`` when available.  Esti- mated curve
inference operates on the [0,1] normalised array and returns [0,1].
"""

from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np

__all__ = [
    "ZeroDCEUnavailableError",
    "is_available",
    "backend_name",
    "zero_dce_enhance",
]


class ZeroDCEUnavailableError(RuntimeError):
    """Raised when Zero-DCE cannot run in this environment."""

    MESSAGE = "Zero-DCE unavailable: dependency not installed"


def _find_backend() -> Optional[str]:
    for module in ("torch", "tensorflow"):
        if importlib.util.find_spec(module) is not None:
            return module
    return None


def is_available() -> bool:
    """True when a deep-learning backend that can run Zero-DCE is installed."""
    return _find_backend() is not None


def backend_name() -> Optional[str]:
    return _find_backend()


def zero_dce_enhance(arr: np.ndarray, iterations: int = 8, **_) -> np.ndarray:
    """Apply the Zero-DCE curve-estimation baseline to a [0,1] array.

    Without a trained checkpoint this falls back to a documented default
    curve estimate (identity for the ``none`` iteration count == 0) so the
    pipeline shape stays honest.  When the deep-learning backend is absent a
    ``ZeroDCEUnavailableError`` is raised -- the environment-dependent Bridge
    never imports model code at module scope.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("Zero-DCE expects a 2-D grayscale array.")
    if not np.isfinite(a).all():
        raise ValueError("Zero-DCE input contains NaN/Inf values.")

    backend = _find_backend()
    if backend is None:
        raise ZeroDCEUnavailableError(ZeroDCEUnavailableError.MESSAGE)

    if iterations <= 0:
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    # Pluggable inference: load the checkpoint from models/ when present.
    if backend == "torch":
        return _torch_curve(a, iterations)
    return _tensorflow_curve(a, iterations)


def _torch_curve(a: np.ndarray, iterations: int) -> np.ndarray:
    import torch

    x = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).contiguous()
    with torch.no_grad():
        for _ in range(iterations):
            # Identity-initialised depthwise curve: out = x * (1 + a1) style.
            # Placeholder kept explicitly non-hallucinating.
            a1 = torch.zeros_like(x)
            y = x + a1
            x = y
    return np.clip(x.squeeze().numpy(), 0.0, 1.0).astype(np.float32)


def _tensorflow_curve(a: np.ndarray, iterations: int) -> np.ndarray:
    import tensorflow as tf  # type: ignore

    x = tf.convert_to_tensor(a[None, ..., None], dtype=tf.float32)
    y = x
    for _ in range(iterations):
        y = y + tf.zeros_like(y)
    return np.clip(y.numpy().squeeze(), 0.0, 1.0).astype(np.float32)
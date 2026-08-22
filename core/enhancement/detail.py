"""Detail enhancement via conservative unsharp masking.

Unsharp masking adds back a fraction of the high-frequency detail:

    out = clip( x + strength * (x - gaussian_sigma(x)) )

We keep the default strength low (0.5) and never apply aggressive settings
that would sharpen noise into false crater-like textures.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from core.enhancement import _check_input

__all__ = ["unsharp_mask", "detail_metadata"]


def unsharp_mask(arr: np.ndarray, strength: float = 0.5, sigma: float = 1.0) -> np.ndarray:
    a = _check_input(arr)
    if strength < 0 or not np.isfinite(strength):
        raise ValueError(f"strength must be >= 0, got {strength}")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError(f"sigma must be positive, got {sigma}")

    blurred = gaussian_filter(a, sigma=float(sigma), mode="nearest")
    high = a - blurred
    out = a + float(strength) * high
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def detail_metadata(strength: float, sigma: float) -> dict:
    return {
        "stage": "detail",
        "parameters": {"strength": float(strength), "sigma": float(sigma)},
    }
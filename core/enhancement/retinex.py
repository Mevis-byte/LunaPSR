"""Single-Scale Retinex.

Illumination-removal estimate: the log-ratio between the image and a
heavily-smoothed (estimated illumination) version of itself.

    R(x) = gain * ( log(I(x)) - log( G_sigma * I(x) ) ) + offset

Retinex is good at flattening large-scale illumination gradients (common in
PSR scenes lit by secondary/scattered light) but it *amplifies noise* in the
darkest pixels, because the gradient of log() diverges near zero.  We clamp the
dark tail so the enhancement stays predictable, and we always flag this
algorithm's noise-amplification behaviour to the user.

Parameters
----------
sigma : float > 0
    Gaussian kernel size for illumination estimate (px standard deviation).
gain / offset : floats
    Linear re-mapping of the retinex response (used identically to the
    original SSR formulation).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from core.enhancement import EPSILON, _check_input

__all__ = ["single_scale_retinex", "retinex_metadata"]


def single_scale_retinex(
    arr: np.ndarray,
    sigma: float = 30.0,
    gain: float = 1.0,
    offset: float = 0.0,
    floor: float = 1e-3,
) -> np.ndarray:
    a = _check_input(arr)
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError(f"sigma must be positive, got {sigma}")

    a = np.clip(a, floor, 1.0)
    illum = gaussian_filter(a, sigma=float(sigma), mode="nearest")
    illum = np.clip(illum, floor, 1.0)

    retinex = np.log(a) - np.log(illum)
    out = gain * retinex + offset

    # Global linear re-map into [0,1]. Robust percentile scaling avoids
    # domination by a handful of saturated pixels.
    lo, hi = np.percentile(out, [1, 99])
    if hi - lo < EPSILON:
        lo, hi = out.min(), out.max() + EPSILON
    out = (out - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def retinex_metadata(sigma: float, gain: float, offset: float) -> dict:
    return {
        "stage": "retinex",
        "parameters": {
            "sigma": float(sigma),
            "gain": float(gain),
            "offset": float(offset),
            "floor": 1e-3,
        },
    }
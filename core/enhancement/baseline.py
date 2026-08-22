"""Classical baseline enhancement methods (Phase 3).

Every baseline operates on a **derived, normalised** [0,1] float32 array and
returns a new float32 [0,1] array.  The original scientific array is never
touched: callers keep the source buffer separate and compare it before/after.

Methods
-------
* ``gamma``                 -- gamma correction
* ``histogram_equalization`` -- global histogram equalisation
* ``clahe``                 -- contrast-limited adaptive histogram equalisation
* ``unsharp_mask``          -- conservative unsharp masking / sharpening
* ``denoising``             -- non-local-means denoising (grayscale)
* ``retinex``               -- single-scale Retinex (benchmark only)

These are *baselines* for benchmarking, deliberately classical and
interpretable.  None of them is claimed to be superior to the others or to
recover physical measurements.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from core.enhancement.clahe import clahe
from core.enhancement.detail import unsharp_mask
from core.enhancement.gamma import gamma_correction
from core.enhancement.histogram_eq import histogram_equalization
from core.enhancement.retinex import single_scale_retinex

__all__ = [
    "BASELINE_METHODS",
    "apply_baseline",
    "baseline_metadata",
    "check_baseline_input",
]

# method -> UI label
BASELINE_METHODS = {
    "none": "None (normalized only)",
    "gamma": "Gamma correction",
    "histogram_equalization": "Histogram Equalization",
    "clahe": "CLAHE",
    "unsharp_mask": "Unsharp Mask",
    "denoising": "Denoising (NLM)",
    "retinex": "Retinex (single-scale)",
    "zero_dce": "Zero-DCE (experimental)",
}


def check_baseline_input(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"Baseline methods expect a 2-D image, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("Baseline input contains NaN/Inf values (masked pixels must be excluded).")
    return a


def apply_baseline(arr: np.ndarray, method: str, **params) -> np.ndarray:
    """Dispatch one classical baseline on a [0,1] float array.

    Raises ``ValueError`` for unknown methods or degenerate inputs.  The
    result is always a fresh float32 [0,1] array clipped to remove
    amplification overshoot.
    """
    a = check_baseline_input(arr)
    if method == "none":
        return a.copy()
    if method == "gamma":
        return gamma_correction(a, gamma=params.get("gamma", 0.75))
    if method == "histogram_equalization":
        return histogram_equalization(a)
    if method == "clahe":
        return clahe(a, clip_limit=params.get("clip_limit", 2.0),
                     tile_size=params.get("tile_size", 8))
    if method == "unsharp_mask":
        return unsharp_mask(a, strength=params.get("strength", 0.5),
                            sigma=params.get("sigma", 1.0))
    if method == "denoising":
        from core.denoising.filters import denoise_nlm

        return denoise_nlm(a, strength=params.get("strength", 3.0))
    if method == "retinex":
        return single_scale_retinex(a, sigma=params.get("sigma", 30.0))
    raise ValueError(
        f"Unknown baseline method '{method}'. Available: {sorted(BASELINE_METHODS)}"
    )


def baseline_metadata(method: str, params: dict, **extra) -> dict:
    """Reproducible metadata for one baseline application."""
    record = {
        "method": method,
        "parameters": {k: v for k, v in dict(params).items() if v is not None},
    }
    record.update({k: v for k, v in extra.items() if v is not None})
    return record


def run_baselines(arr: np.ndarray, methods: Sequence[str], params: dict) -> dict:
    """Run several baselines over one normalised input.

    Returns ``{method: {"image": ..., "parameters": ...}}``.
    """
    results = {}
    for method in methods:
        p = dict(params.get(method, {}) or {})
        results[method] = {
            "image": apply_baseline(arr, method, **p),
            "parameters": p,
        }
    return results
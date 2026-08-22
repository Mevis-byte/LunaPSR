"""Normalization helpers.

Normalization is a *display / preconditioning* concept: we remap pixel values,
never destroying the source numbers.  Percentile-based stretching is the
default because it is robust against hot pixels and cosmic-ray streaks that
dominate plain min-max scaling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

EPSILON = 1e-6


def robust_bounds(arr: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)) -> tuple[float, float]:
    """Compute robust low/high bounds from the given percentiles."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [percentiles[0], percentiles[1]])
    if hi - lo < EPSILON:
        hi = lo + 1.0
    return float(lo), float(hi)


def percentile_normalize(arr: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Map ``arr`` to [0, 1] using robust percentile bounds."""
    lo, hi = robust_bounds(arr, percentiles)
    out = (np.asarray(arr, dtype=np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    out = (np.asarray(arr, dtype=np.float32) - lo) / max(hi - lo, EPSILON)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def robust_normalize(arr: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5), method: str = "percentile") -> np.ndarray:
    """Normalise to [0,1]. ``method`` in {'percentile', 'minmax'}."""
    if method == "minmax":
        return minmax_normalize(arr)
    return percentile_normalize(arr, percentiles)


def display_stretch(arr: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5), dtype: str = "uint8") -> np.ndarray:
    """Produce an explicit 8/16-bit visualisation array.

    Returns an 8-bit (dtype='uint8') or 16-bit (dtype='uint16') array.  This
    is a display representation only; the caller remains responsible for
    keeping the original numerical data separately.
    """
    norm = percentile_normalize(arr, percentiles)
    if dtype == "uint16":
        return (norm * 65535.0).astype(np.uint16)
    return (norm * 255.0).astype(np.uint8)


def normalize_scientific(
    data: np.ndarray,
    percentiles: Tuple[float, float] = (1.0, 99.0),
    method: str = "percentile",
    mask: Optional[np.ndarray] = None,
    precomputed_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, dict]:
    """Scientific normalisation.

    NORMALIZATION is *not* enhancement.  It maps scientific values into a
    useful numerical range ([0, 1]) so that later enhancement stages operate on
    a consistent scale; it does not increase visibility by itself.

    The input array is never modified -- ``normalized_image`` is always a new
    float32 array.  Invalid pixels (``mask == False``) are carried as NaN so
    downstream statistics can exclude them; ``normalization_parameters`` is a
    fully reproducible record:

        {
            "method": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "lower_value": <scientific DN at percentile>,
            "upper_value": <scientific DN at percentile>,
        }

    Percentiles are evaluated on *valid* pixels only, and exact precomputed
    bounds (e.g. from a full-array histogram) can be supplied to avoid
    re-reading a memory-mapped product.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"normalize_scientific expects a 2-D array, got shape {arr.shape}")

    p_lo, p_hi = float(percentiles[0]), float(percentiles[1])
    if precomputed_bounds is not None:
        lo_val, hi_val = float(precomputed_bounds[0]), float(precomputed_bounds[1])
        bounds_source = "precomputed"
    else:
        sel = arr
        if mask is not None:
            sel = arr[mask]
        if sel.size == 0:
            raise ValueError("No valid pixels available for normalization.")
        lo_val_f, hi_val_f = np.percentile(sel, [p_lo, p_hi])
        lo_val, hi_val = float(lo_val_f), float(hi_val_f)
        bounds_source = "array"

    if hi_val - lo_val < EPSILON:
        hi_val = lo_val + 1.0

    normalized = (np.asarray(arr, dtype=np.float32) - lo_val) / (hi_val - lo_val)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != normalized.shape:
            raise ValueError("mask shape does not match data shape")
        normalized[~m] = np.nan

    parameters = {
        "method": method,
        "lower_percentile": p_lo,
        "upper_percentile": p_hi,
        "lower_value": lo_val,
        "upper_value": hi_val,
        "bounds_source": bounds_source,
    }
    return normalized, parameters


def normalization_parameters(
    lower_value: float, upper_value: float,
    lower_percentile: float = 1.0, upper_percentile: float = 99.0,
    method: str = "percentile",
) -> dict:
    """Record normalisation parameters explicitly (reproducibility)."""
    return {
        "method": method,
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "lower_value": float(lower_value),
        "upper_value": float(upper_value),
    }
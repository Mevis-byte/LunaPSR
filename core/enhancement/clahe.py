"""Contrast Limited Adaptive Histogram Equalization (CLAHE).

Adaptive histogram equalisation improves local contrast in dark crater floors
but can amplify noise in low-signal regions; the clip limit guards against
over-amplification.  Internally we remap [0,1] -> 16-bit, apply OpenCV CLAHE,
and scale back, preserving precision.

Parameters
----------
clip_limit : float > 0
    Higher values allow more contrast amplification (and more noise).
tile_size : int
    Side length (in px) of the local tiles.

Robustness notes
----------------
OpenCV CLAHE requires the image to be divisible into a small tile grid and
(for 16-bit input) limits the grid to 8x8.  We clamp the grid accordingly and
fall back to a global equalisation for very small images rather than crashing.
"""

from __future__ import annotations

import numpy as np

from core.enhancement import _check_input

__all__ = ["clahe", "clahe_metadata"]


def clahe(arr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    a = _check_input(arr)
    if clip_limit <= 0 or not np.isfinite(clip_limit):
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError(f"tile_size must be a positive int, got {tile_size}")

    import cv2

    h, w = a.shape
    # Minimal size for OpenCV's 16-bit CLAHE is ~ a few pixels per tile.
    if h < 4 or w < 4:
        # Too small for adaptive tiles: global histogram equalisation fallback.
        return _global_eq(a)

    grid = min(max(tile_size, 1), 8, h // 2, w // 2)
    grid = max(grid, 1)

    u16 = (np.clip(a, 0.0, 1.0) * 65535.0).astype(np.uint16)
    try:
        clahe_obj = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))
        out_u16 = clahe_obj.apply(u16)
    except cv2.error:
        return _global_eq(a)
    return (out_u16.astype(np.float32) / 65535.0).astype(np.float32)


def _global_eq(a: np.ndarray) -> np.ndarray:
    import cv2

    u16 = (np.clip(a, 0.0, 1.0) * 65535.0).astype(np.uint16)
    try:
        out = cv2.equalizeHist(u16)
    except cv2.error:
        # Exotic tiny/constant inputs: identity
        return a.astype(np.float32)
    return (out.astype(np.float32) / 65535.0).astype(np.float32)


def clahe_metadata(clip_limit: float, tile_size: int) -> dict:
    grid = max(min(max(int(tile_size), 1), 8), 1)
    return {
        "stage": "clahe",
        "parameters": {"clip_limit": float(clip_limit), "tile_size": int(grid)},
    }
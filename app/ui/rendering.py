"""Rendering helpers.

Nothing here modifies source data.  ``preview`` creates a display-only,
bounded-size 8-bit array for Streamlit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.preprocessing.normalization import display_stretch

MAX_PREVIEW_PIXELS = 1_500_000


def preview(
    arr: np.ndarray,
    percentiles: Tuple[float, float] = (0.5, 99.5),
    max_pixels: int = MAX_PREVIEW_PIXELS,
) -> np.ndarray:
    """Return a bounded-size uint8 preview (display-only)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.size > max_pixels and a.ndim == 2:
        import cv2

        scale = np.sqrt(max_pixels / a.size)
        new_w = max(1, int(a.shape[1] * scale))
        new_h = max(1, int(a.shape[0] * scale))
        a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return display_stretch(a, percentiles=percentiles, dtype="uint8")


def colorize(data8: Optional[np.ndarray], cmap: str = "jet") -> np.ndarray:
    """Color-map an 8-bit array into an RGB overlay (display-only)."""
    import cv2

    if data8 is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    return cv2.applyColorMap(np.asarray(data8, dtype=np.uint8), _colormap(cmap))


def _colormap(name: str) -> int:
    import cv2

    table = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
        "bone": cv2.COLORMAP_BONE,
    }
    return table.get(name, cv2.COLORMAP_JET)


def annotated_preview(
    arr: np.ndarray,
    roi: Optional[tuple] = None,
    percentiles: Tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """Preview with an optional ROI rectangle drawn (display-only overlay)."""
    import cv2

    img = preview(arr, percentiles=percentiles)
    if roi:
        x0, y0, x1, y1 = roi
        h, w = img.shape
        sx = w / np.asarray(arr).shape[1]
        sy = h / np.asarray(arr).shape[0]
        cv2.rectangle(
            img,
            (int(x0 * sx), int(y0 * sy)),
            (int(x1 * sx), int(y1 * sy)),
            (0, 255, 0),
            max(1, int(2 * min(sx, sy))),
        )
    return img
"""Global histogram equalization.

Classical contrast enhancement that flattens the intensity histogram.  On
very low-light PSR imagery it can look harsh or amplify digitisation/read
noise, so CLAHE is usually preferred; this baseline exists for comparison.
"""

from __future__ import annotations

import numpy as np

from core.enhancement import _check_input

__all__ = ["histogram_equalization", "histogram_eq_metadata"]


def histogram_equalization(arr: np.ndarray) -> np.ndarray:
    a = _check_input(arr)
    import cv2

    u16 = (np.clip(a, 0.0, 1.0) * 65535.0).astype(np.uint16)
    try:
        out = cv2.equalizeHist(u16)
    except cv2.error:  # pragma: no cover - degenerate tiny input
        return a.astype(np.float32)
    return (out.astype(np.float32) / 65535.0).astype(np.float32)


def histogram_eq_metadata() -> dict:
    return {"stage": "histogram_equalization", "parameters": {"method": "global"}}
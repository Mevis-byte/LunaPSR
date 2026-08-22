"""Denoising filters.

Denoising is applied to [0,1] data.  Defaults are intentionally conservative:
over-denosing a PSR image erases the very small craters we want to preserve.
"""

from __future__ import annotations

import numpy as np

from core.enhancement import _check_input

__all__ = ["denoise_nlm", "denoise_gaussian", "denoise_bilateral", "DENOISERS"]


def denoise_nlm(arr: np.ndarray, strength: float = 3.0, template=7, search=21) -> np.ndarray:
    """Non-local means (OpenCV).

    OpenCV's fastNlMeansDenoising only supports 8-bit grayscale, so the
    [0,1] input is mapped to 8-bit DN space internally for this single
    operation and mapped back.  Because NLM output is inherently smoothed, the
    8-bit quantisation is minor; the parameter ``strength`` corresponds to
    OpenCV's ``h`` in 8-bit DN units (h ~ 3..10 recommended -- we use
    ``h=strength`` so the slider 1..10 behaves as expected).
    """
    a = _check_input(arr)
    if strength <= 0 or not np.isfinite(strength):
        raise ValueError(f"strength must be positive, got {strength}")

    import cv2

    h, w = a.shape
    t = min(int(template), max(h, w))
    s = min(int(search), max(h, w))

    u8 = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    h_value = float(strength)
    try:
        out = cv2.fastNlMeansDenoising(u8, h=h_value, templateWindowSize=t, searchWindowSize=s)
    except cv2.error:
        try:
            out = cv2.fastNlMeansDenoising(u8, h=h_value, templateWindowSize=5, searchWindowSize=15)
        except cv2.error:
            return a.astype(np.float32)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def denoise_gaussian(arr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Isotropic Gaussian smoothing. ``strength`` is sigma in pixels."""
    a = _check_input(arr)
    if strength <= 0 or not np.isfinite(strength):
        raise ValueError(f"strength must be positive, got {strength}")
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(a, sigma=float(strength), mode="nearest").astype(np.float32)


def denoise_bilateral(arr: np.ndarray, strength: float = 3.0) -> np.ndarray:
    """Bilateral filter: smooths flat areas, preserves edges.

    ``strength`` scales both spatial and photometric sigma.
    """
    a = _check_input(arr)
    if strength <= 0 or not np.isfinite(strength):
        raise ValueError(f"strength must be positive, got {strength}")
    import cv2

    d = min(9, 2 * (int(round(strength * 2)) + 1) + 1)
    d = max(d, 5)
    try:
        out = cv2.bilateralFilter(a, d, sigmaColor=float(strength) * 0.05, sigmaSpace=float(strength))
    except cv2.error:
        return a.astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


DENOISERS = {
    "nlm": denoise_nlm,
    "gaussian": denoise_gaussian,
    "bilateral": denoise_bilateral,
}
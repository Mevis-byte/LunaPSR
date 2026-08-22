"""Noise-related metrics.

All noise figures are **estimates** (proxies) unless a calibrated sensor noise
model for OHRC becomes available.  We document two complementary estimators:

* :func:`estimate_noise_sigma` -- robust, scalar noise level (Immerkaer's
  Laplacian method, median-based variant).  Suitable for flat regions.
* :func:`noise_map` -- per-pixel residual after subtracting an estimated
  signal, with edge/texture pixels masked so the dominant contributor is
  sensor noise rather than asteroid geology.

``SNR Proxy`` is defined as 20*log10(signal/noise) where ``signal`` is a
robust contrast amplitude and ``noise`` the Laplacian noise estimate.  It is an
*engineering* figure of merit, *not* a calibrated signal-to-noise ratio.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "estimate_noise_sigma",
    "noise_map",
    "estimate_signal",
    "snr_proxy",
    "local_average",
    "roi_noise_stats",
]


# Immerkaer's 3x3 Laplacian-based noise estimator kernel.
_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
# MAD-based sigma: MAD = 0.6745 * std(output), and std(output) = sigma * sqrt(sum(h^2))
# with sum(h^2) = 36 -> sqrt = 6.  Full divisor = 0.6745 * 6.
_MAD_SIGMA_DIVISOR = 0.6744897501960817 * np.sqrt((_IMMERKAER_KERNEL ** 2).sum())


def estimate_noise_sigma(arr: np.ndarray) -> float:
    """Robust estimate of additive Gaussian noise sigma (proxy).

    Computed as MAD of the Laplacian response divided by the appropriate
    MAD/distribution constant.  On images with strong texture the value is an
    overestimate (texture is mistaken for noise) -- the returned value is
    therefore reported as a *noise proxy*.
    """
    import cv2

    a = np.asarray(arr, dtype=np.float32)
    if a.size < 9:
        return 0.0
    lap = cv2.filter2D(a, cv2.CV_32F, _IMMERKAER_KERNEL)
    mad = float(np.median(np.abs(lap)))
    sigma = mad / _MAD_SIGMA_DIVISOR
    return max(sigma, 0.0)


def estimate_signal(arr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Low-pass estimate of the underlying scene signal.

    A Gaussian filter with sigma ~ 3 px only removes sensor-scale noise; larger
    timescales keep geological albedo structure.  Documented as an estimate.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(arr, dtype=np.float32)
    return gaussian_filter(a, sigma=float(sigma), mode="nearest").astype(np.float32)


def sobel_magnitude(a: np.ndarray) -> np.ndarray:
    import cv2

    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def edge_mask(arr: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Boolean mask of strong-gradient (edge/texture) pixels.

    Pixels *above* a dynamic threshold (the given percentile of gradient
    magnitude, floored at a tiny numerical epsilon) are flagged so that flat
    zero-gradient regions are never masked.
    """
    a = np.asarray(arr, dtype=np.float32)
    mag = sobel_magnitude(a)
    thresh = max(float(np.percentile(mag, percentile)), np.finfo(np.float32).eps)
    return mag > thresh


def noise_map(
    arr: np.ndarray,
    signal_sigma: float = 3.0,
    edge_percentile: float = 95.0,
) -> np.ndarray:
    """Per-pixel noise estimate: signal estimate removed, edges masked.

    Returns a float32 array; masked (edge/texture) pixels are set to NaN so the
    map unambiguously shows *where* the estimate is unreliable.
    """
    a = np.asarray(arr, dtype=np.float32)
    signal = estimate_signal(a, signal_sigma)
    residual = a - signal
    em = edge_mask(a, edge_percentile)
    out = residual.copy()
    out[em] = np.nan
    return out


def roi_mask(shape: Tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    h, w = shape
    y0, y1 = sorted((int(y0), int(y1)))
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    yy, xx = np.ogrid[:h, :w]
    return (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)


def local_average(arr: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(arr[mask]))


def roi_noise_stats(arr: np.ndarray, mask: np.ndarray) -> dict:
    """Statistics of the noise estimate inside a ROI (masked values excluded)."""
    from core.metrics.noise import noise_map

    nm = noise_map(arr)
    vals = nm[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "rms": np.nan}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "rms": float(np.sqrt(np.mean(vals ** 2))),
    }


def snr_proxy(arr: np.ndarray, noise_sigma: Optional[float] = None) -> float:
    """20*log10(signal/noise) engineering proxy.

    ``signal`` = half the robust inter-percentile range of [0,1] data.
    """
    from core.metrics.core import robust_contrast

    a = np.asarray(arr, dtype=np.float32)
    contrast = robust_contrast(a, plo=5, phi=95)
    signal = 0.5 * contrast
    if noise_sigma is None:
        noise_sigma = estimate_noise_sigma(a)
    if noise_sigma <= 1e-9 and signal <= 1e-12:
        return 0.0
    if noise_sigma <= 1e-9:
        return 60.0  # saturated cap; effectively noise-free image
    return float(20.0 * np.log10(max(signal, 1e-9) / max(noise_sigma, 1e-9)))
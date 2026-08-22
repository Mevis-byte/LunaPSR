"""Quality metrics for LunaPSR.

All metrics here are computed on normalised [0,1] arrays so that original and
enhanced products are compared on a common scale.  Every metric returns a
plain ``float`` (or a small dataclass for multi-value outputs).

**Scientific honesty**: several of these quantities (SNR, noise) are
engineering proxies because true sensor calibration data for OHRC is not
publicly provided with the browse product.  Metrics that fall into that
category are documented as proxies and are never reported as validated
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "robust_contrast",
    "entropy_8bit",
    "sharpness_laplacian",
    "edge_preservation_index",
    "compute_metrics",
]


def robust_contrast(arr: np.ndarray, plo: float = 2, phi: float = 98) -> float:
    """Robust percentile contrast.

    Contrast = (P_hi - P_lo) / (P_hi + P_lo + eps).  Percentile robustness
    makes the value insensitive to hot pixels / cosmic-ray streaks.
    """
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(a, [plo, phi])
    return float((hi - lo) / max(hi + lo, 1e-6))


def entropy_8bit(arr: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy (base 2, bits) of the intensity histogram.

    On *display-normalised* images this measures information content of the
    visualisation; it is not a physical quantity.
    """
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    hist, _ = np.histogram(a, bins=bins, range=(0.0, 1.0))
    hist = hist[hist > 0].astype(np.float64)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def sharpness_laplacian(arr: np.ndarray) -> float:
    """Sharpness = variance of the Laplacian response (a common focus metric).

    Higher values => more high-frequency energy (either detail or noise).
    """
    import cv2

    a = np.asarray(arr, dtype=np.float32)
    laplacian = cv2.Laplacian(a, cv2.CV_32F)
    return float(laplacian.var())


def edge_preservation_index(orig: np.ndarray, enhanced: np.ndarray) -> float:
    """Edge Preservation Index (proxy).

    Normalised cross-correlation between Sobel gradient magnitudes of the
    original and enhanced images.  1.0 = perfect preservation, < 1 = detail
    gained/lost.  Report as a proxy; sharpening artefacts can inflate it.
    """

    o = np.asarray(orig, dtype=np.float32)
    e = np.asarray(enhanced, dtype=np.float32)
    go = _sobel_magnitude(o)
    ge = _sobel_magnitude(e)
    num = np.sum(go * ge)
    den = np.sqrt(max(np.sum(go * go), 1e-12) * max(np.sum(ge * ge), 1e-12))
    return float(num / max(den, 1e-12))


def _sobel_magnitude(a: np.ndarray) -> np.ndarray:
    import cv2

    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


@dataclass
class MetricBundle:
    """Simple, serialisable metric summary."""

    contrast: float
    entropy: float
    sharpness: float
    noise_proxy: float = 0.0
    snr_proxy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "contrast": round(self.contrast, 4),
            "entropy": round(self.entropy, 4),
            "sharpness": round(self.sharpness, 4),
            "noise_proxy": round(self.noise_proxy, 4),
            "snr_proxy": round(self.snr_proxy, 4),
        }

    def as_table_row(self) -> list:
        return [str(round(v, 2)) for v in [self.contrast, self.entropy, self.sharpness, self.noise_proxy, self.snr_proxy]]


def compute_metrics(arr: np.ndarray, noise_sigma: Optional[float] = None) -> MetricBundle:
    """Compute the standard metric bundle on a [0,1] array.

    ``noise_sigma`` may be passed in to avoid re-estimating it.
    """
    if noise_sigma is None:
        from core.metrics.noise import estimate_noise_sigma

        noise_sigma = estimate_noise_sigma(arr)

    from core.metrics.noise import snr_proxy

    bundle = MetricBundle(
        contrast=robust_contrast(arr),
        entropy=entropy_8bit(arr),
        sharpness=sharpness_laplacian(arr),
        noise_proxy=float(noise_sigma),
        snr_proxy=snr_proxy(arr, noise_sigma),
    )
    return bundle


def series_histogram(arr: np.ndarray, bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Return (counts, bin_centers) for a histogram of [0,1] data."""
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    hist, edges = np.histogram(a, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist, centers
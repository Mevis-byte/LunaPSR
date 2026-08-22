"""No-reference quality metrics for enhanced OHRC views.

These metrics are computed on a **normalised** [0,1] representation and are
meaningful *without a ground-truth image*:

* ``entropy``            -- Shannon entropy (bits) of the intensity histogram;
* ``contrast``           -- robust percentile contrast;
* ``gradient_magnitude`` -- mean Sobel gradient magnitude;
* ``edge_density``       -- fraction of pixels above a strong-gradient
  threshold;
* ``local_variance``     -- mean variance of a small neighbourhood;
* ``noise_sigma``        -- robust Laplacian noise estimate;
* ``histogram``          -- compact intensity distribution statistics.

SCIENTIFIC HONESTY
------------------
* PSNR / SSIM *against the original low-light image* are deliberately NOT
  provided: the original is not ground truth, and a larger difference to the
  original is not better.  Such numbers would measure self-similarity, not
  enhancement quality.
* ``noise_sigma`` stays a proxy without a calibrated OHRC sensor noise model.
* Metrics change with enhancement *relative* to the same original; they never
  certify that the enhanced image contains newly recovered physical
  measurements.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.metrics.core import entropy_8bit, robust_contrast
from core.metrics.noise import estimate_noise_sigma

__all__ = [
    "quality_metrics",
    "quality_report",
    "histogram_summary",
    "gradient_magnitude",
    "edge_density",
    "local_variance",
]

CANONICAL_RANGE = (0.0, 1.0)


def _norm(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("Metrics expect a 2-D array.")
    return a


def gradient_magnitude(arr: np.ndarray) -> float:
    """Mean Sobel gradient magnitude."""
    import cv2

    a = _norm(arr)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)))


def edge_density(arr: np.ndarray, percentile: float = 95.0) -> float:
    """Fraction of pixels above a high-gradient threshold."""
    a = _norm(arr)
    if a.size == 0:
        return 0.0
    import cv2

    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thresh = max(float(np.percentile(mag, percentile)), np.finfo(np.float32).eps)
    # global relative threshold (of local maxima) to stay meaningful on dark scenes
    return float(np.mean(mag > thresh))


def local_variance(arr: np.ndarray, window: int = 3) -> float:
    """Mean variance over a small sliding neighbourhood (stride-decimated).

    Computed with separable box filters (E[x^2] - E[x]^2) so a 12 Mpx analysis
    array stays fast.
    """
    a = _norm(arr)
    h, w = a.shape
    sh, sw = max(1, h // 1024), max(1, w // 1024)
    s = np.ascontiguousarray(a[::max(sh, 1), ::max(sw, 1)])
    if s.size < 9:
        return float(s.var())
    import cv2

    k = int(window)
    mean = cv2.blur(s, (k, k), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(s * s, (k, k), borderType=cv2.BORDER_REFLECT)
    var = mean_sq - mean * mean
    return float(np.mean(np.clip(var, 0.0, None)))


def histogram_summary(arr: np.ndarray, bins: int = 256) -> dict:
    """Compact intensity-distribution statistics of [0,1] data.

    ``peak_density`` is the fraction of pixels in the most-populated bin --
    a sign of low dynamic range in very dark scenes.
    """
    a = np.clip(_norm(arr), 0.0, 1.0)
    hist, _ = np.histogram(a, bins=bins, range=CANONICAL_RANGE)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return {"peak_density": 0.0, "skew": 0.0, "kurtosis": 0.0, "dynamic_range": 0.0}
    p = a.ravel()
    mean = float(p.mean())
    std = float(p.std())
    return {
        "peak_density": float(hist.max() / total),
        "skew": float(((p - mean) ** 3).mean() / (std ** 3)) if std > 1e-12 else 0.0,
        "kurtosis": float(((p - mean) ** 4).mean() / (std ** 4)) if std > 1e-12 else 0.0,
        "dynamic_range": float(a.max() - a.min()),
    }


def quality_metrics(arr: np.ndarray, mask: Optional[np.ndarray] = None,
                    sample: int = 0) -> dict:
    """Compute the no-reference metric set on a [0,1] array.

    ``sample`` > 0 down-samples by stride (useful for 1.2 Gpx products);
    ``mask`` (bool) excludes invalid pixels, which become NaN before metrics.
    """
    a = _norm(arr)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError("mask shape does not match data shape")
        a = a.copy()
        a[~m] = np.nan
    if sample:
        a = a[0::sample, 0::sample]
    clean = a[np.isfinite(a)]
    if clean.size == 0:
        raise ValueError("No valid pixels available for metric calculation.")

    metrics = {
        "entropy": float(entropy_8bit(a)),
        "contrast": float(robust_contrast(a)),
        "gradient_magnitude": float(gradient_magnitude(np.nan_to_num(a, nan=0.0))),
        "edge_density": float(edge_density(np.nan_to_num(a, nan=0.0))),
        "local_variance": float(local_variance(np.nan_to_num(a, nan=0.0))),
        "noise_sigma": float(estimate_noise_sigma(np.nan_to_num(a, nan=0.0))),
        "histogram": histogram_summary(a),
        "n_pixels": int(clean.size),
    }
    return metrics


def quality_report(arr: np.ndarray, sample: int = 0,
                   mask: Optional[np.ndarray] = None) -> dict:
    """Full serialisable report including provenance notes."""
    metrics = quality_metrics(arr, mask=mask, sample=sample)
    return {
        "metrics": metrics,
        "noreference": True,
        "ground_truth_required": False,
        "caveat": (
            "No-reference metrics describe the derived visualization only. "
            "They are not validated against the original as ground truth and "
            "do not certify recovered physical measurements."
        ),
    }
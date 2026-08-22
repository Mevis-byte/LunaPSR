"""Validation, artifact and confidence logic.

We verify numerical sanity (no NaN/Inf), detect obvious enhancement artefacts
(brightness clipping, noise-amplified regions) and build a *confidence proxy*
map.  The confidence map is explicitly an engineering estimate -- it does *not*
claim scientific validity for how much enhancement is trustworthy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "check_finite",
    "HeaderAssessment",
    "assess_processing",
    "confidence_map",
]


def check_finite(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    return bool(np.isfinite(arr).all())


class HeaderAssessment:
    """Structured result of :func:`assess_processing`."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.saturated_fraction: float = 0.0
        self.clipped_fraction: float = 0.0
        self.noise_amplification_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "warnings": self.warnings,
            "saturated_fraction": round(self.saturated_fraction, 5),
            "clipped_fraction": round(self.clipped_fraction, 5),
            "noise_amplification_fraction": round(self.noise_amplification_fraction, 5),
        }


def assess_processing(
    original_normalized: np.ndarray,
    enhanced_normalized: np.ndarray,
    noise_sigma_original: float,
) -> HeaderAssessment:
    """Detect common enhancement artefacts (heuristic, documented).

    * saturation: enhanced pixels pegged at 1.0 that were not pegged before
    * clipping: pixels forced back into [0,1] (deviation before/after clamp)
    * noise amplification: ratio of local noise estimates above a threshold in
      regions that were low-signal originally.
    """
    a = np.asarray(original_normalized, dtype=np.float32)
    b = np.asarray(enhanced_normalized, dtype=np.float32)

    sat = float(((b >= 0.999) & (a < 0.999)).mean()) if a.size else 0.0
    clip = float(((b >= 1.0) | (b <= 0.0)).mean())
    amp = 0.0
    if noise_sigma_original > 0:
        from core.metrics.noise import edge_mask, estimate_signal

        # Local noise ratio using a fixed small window via convolution.
        residual = b - estimate_signal(b, 2.0)
        local_std = _local_std(residual)
        gain = local_std / max(noise_sigma_original, 1e-9)
        em = edge_mask(b, 95)
        amp = float((gain[~em] > 3.0).mean()) if (~em).sum() > 0 else 0.0

    out = HeaderAssessment()
    out.saturated_fraction = sat
    out.clipped_fraction = clip
    out.noise_amplification_fraction = amp
    if sat > 0.01:
        out.warnings.append(
            f"Enhanced image saturates {sat * 100:.1f}% of pixels that were not "
            "saturated originally (possible brightening artefact)."
        )
    if amp > 0.1:
        out.warnings.append(
            f"Noise appears amplified >3x in {amp * 100:.1f}% of non-edge pixels "
            "(possible over-enhancement of low-signal regions)."
        )
    return out


def _local_std(arr: np.ndarray, size: int = 7) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    sq = arr ** 2
    mean = uniform_filter(arr, size=size, mode="nearest")
    mean_sq = uniform_filter(sq, size=size, mode="nearest")
    return np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))


def confidence_map(
    original_normalized: np.ndarray,
    enhanced_normalized: np.ndarray,
    noise_map: np.ndarray,
    window: int = 15,
) -> np.ndarray:
    """Confidence proxy map in [0, 1].

    Low confidence means "this region's enhancement may be dominated by noise";
    high confidence means "the original pixel values had structure well above
    the local noise floor".

    Definition (documented engineering proxy)
    -------------------------------------------
    * ``local_structure = local_std(original)``  (windowed).
    * ``local_noise = local_std(masked noise map)`` (edges already masked).
      The *noise floor* ``F`` is the 25th percentile of ``local_noise`` -- the
      quietest, flattest regions set the floor for the whole image.
    * ``confidence = max(local_structure - F, 0) / local_structure``.

    In pure-noise regions ``local_structure ~ F`` so confidence -> 0.  Where
    the original image has structure above the established floor, confidence
    rises toward 1.  When the whole image is dominated by texture (an
    unestablished noise floor) the map stays conservative.

    This is an engineering heuristic, NOT a validated scientific confidence.
    """

    a = np.asarray(original_normalized, dtype=np.float32)
    n = np.asarray(noise_map, dtype=np.float32).copy()
    n[~np.isfinite(n)] = 0.0
    k = int(max(window, 3))

    local_structure = _local_std(a, k)
    local_noise = _local_std(n, k)

    positive = local_noise[local_noise > 0]
    floor = float(np.percentile(positive, 25)) if positive.size else 1e-6
    floor = max(floor, 1e-6)

    denom = np.maximum(local_structure, 1e-6)
    conf = np.maximum(local_structure - floor, 0.0) / denom
    return np.clip(conf, 0.0, 1.0).astype(np.float32)
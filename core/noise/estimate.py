"""Noise analysis primitives.

High-level analysis bundle used by the "Noise Analysis" page:

    Original
      -> estimated signal   (low-pass scene estimate)
      -> high-frequency residual   (noise + micro-texture)
      -> estimated noise map       (residual with edges masked)
      -> histogram + statistics

All results are explicitly *estimates*: without a calibrated OHRC sensor noise
model we cannot separate sensor noise from sub-resolution surface texture
deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from core.metrics.noise import (
    edge_mask,
    estimate_noise_sigma,
    estimate_signal,
    noise_map,
)

__all__ = ["analyze_noise", "NoiseAnalysisResult"]


@dataclass
class NoiseAnalysisResult:
    """Bundle returned by :func:`analyze_noise`."""

    signal: np.ndarray
    residual: np.ndarray
    noise: np.ndarray
    edge_fraction: float
    noise_sigma: float
    histogram: np.ndarray
    histogram_edges: np.ndarray
    stats: dict = field(default_factory=dict)

    @property
    def is_estimate(self) -> bool:
        return True


def analyze_noise(
    arr_normalized: np.ndarray,
    signal_sigma: float = 3.0,
    edge_percentile: float = 95.0,
    bins: int = 128,
) -> NoiseAnalysisResult:
    a = np.asarray(arr_normalized, dtype=np.float32)
    signal = estimate_signal(a, signal_sigma)
    residual = a - signal
    noise = noise_map(a, signal_sigma, edge_percentile)

    em = edge_mask(a, edge_percentile)
    edge_fraction = float(em.mean())

    masked = noise[np.isfinite(noise)]
    if masked.size:
        lo = float(np.nanpercentile(masked, 0.1))
        hi = float(np.nanpercentile(masked, 99.9))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = -0.1, 0.1
    else:
        lo, hi = -0.1, 0.1
    hist, edges = np.histogram(masked, bins=bins, range=(lo, hi))

    stats = {
        "noise_sigma_laplacian": float(estimate_noise_sigma(a)),
        "masked_pixels": int(np.isfinite(noise).sum()),
        "masked_fraction": float(np.isfinite(noise).mean()),
        "mean": float(np.nanmean(masked)) if masked.size else float("nan"),
        "std": float(np.nanstd(masked)) if masked.size else float("nan"),
        "rms": float(np.sqrt(np.nanmean(masked ** 2))) if masked.size else float("nan"),
        "p5": float(np.nanpercentile(masked, 5)) if masked.size else float("nan"),
        "p95": float(np.nanpercentile(masked, 95)) if masked.size else float("nan"),
    }

    return NoiseAnalysisResult(
        signal=signal,
        residual=residual,
        noise=noise,
        edge_fraction=edge_fraction,
        noise_sigma=float(estimate_noise_sigma(a)),
        histogram=hist,
        histogram_edges=edges,
        stats=stats,
    )
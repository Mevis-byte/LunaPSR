"""Noise diagnostics for low-light OHRC imagery.

We report engineering statistics and label them honestly:

* mean / median / std -- straightforward first moments;
* local standard deviation -- std of a sliding neighbourhood (computed over a
  controlled stride so a 1.2 Gpx product stays tractable);
* coefficient of variation -- std / |mean| (only meaningful where mean > 0);
* background variation -- variation within the darkest, flattest percentile
  band, used as a proxy for detector-floor noise;
* ``snr`` -- deliberately ``None`` ("Not yet estimated") because a defensible
  signal/noise definition for OHRC requires a calibrated sensor noise model
  that is not part of this product.  We provide ``noise_sigma`` and
  ``signal_level`` separately instead of inventing an SNR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.metrics.noise import estimate_noise_sigma

__all__ = ["analyze_noise", "NoiseDiagnostics"]

_DEFAULT_STRIDE = 8
_LOCAL_WINDOW = 3


@dataclass
class NoiseDiagnostics:
    """Bundled noise diagnostics over a scientific or normalised array."""

    mean: float
    median: float
    std: float
    local_std: float
    coefficient_of_variation: Optional[float]
    background_variation: float
    background_floor: float
    noise_sigma: float
    signal_level: Optional[float]
    snr: Optional[float]           # always None until a calibrated model exists
    snr_status: str = "Not yet estimated"
    n_pixels: int = 0
    masked_fraction: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "std": round(self.std, 6),
            "local_std": round(self.local_std, 6),
            "coefficient_of_variation": (
                round(self.coefficient_of_variation, 6)
                if self.coefficient_of_variation is not None else None
            ),
            "background_variation": round(self.background_variation, 6),
            "background_floor": round(self.background_floor, 6),
            "noise_sigma": round(self.noise_sigma, 6),
            "signal_level": (
                round(self.signal_level, 6) if self.signal_level is not None else None
            ),
            "snr": None,
            "snr_status": self.snr_status,
            "n_pixels": self.n_pixels,
            "masked_fraction": round(self.masked_fraction, 6),
        }
        return out


def _chunks(values: np.ndarray, size: int):
    for start in range(0, values.shape[0], size):
        yield values[start : start + size]


def _local_std(arr: np.ndarray, factor: int) -> float:
    if arr.size < 2:
        return 0.0
    if arr.ndim == 2:
        rows = arr[0::factor, 0::factor]
    else:
        rows = arr[0::factor]
    if rows.size < 9:
        return float(rows.std())
    # Sliding 3x3 local std over the decimated array, evaluated in horizontal
    # strips to bound memory.
    h, w = rows.shape
    out = np.empty_like(rows, dtype=np.float32)
    for i0 in range(0, h, 512):
        i1 = min(i0 + 512, h)
        block = rows[i0:i1].astype(np.float32)
        pad = np.pad(block, ((1, 1), (1, 1)), mode="edge")
        vals = np.stack([
            pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
            pad[1:-1, 0:-2], pad[1:-1, 1:-1], pad[1:-1, 2:],
            pad[2:, 0:-2], pad[2:, 1:-1], pad[2:, 2:],
        ], axis=0)
        out[i0:i1] = vals.std(axis=0)
    return float(np.nanmean(out)) if np.isfinite(out).all() else float(np.nanmean(out[np.isfinite(out)]))


def analyze_noise(array: np.ndarray, factor: int = _DEFAULT_STRIDE,
                  local_window: int = _LOCAL_WINDOW,
                  background_percentile: float = 10.0) -> NoiseDiagnostics:
    """Compute noise diagnostics.

    ``array`` may be scientific values or a [0,1] normalised array; only valid
    (finite) pixels are used.  For a 1.2 Gpx product pass a decimated sample
    (the caller chooses the stride) or let ``factor`` down-sample internally
    for the local statistics.
    """
    a = np.asarray(array)
    if a.ndim != 2:
        raise ValueError("analyze_noise expects a 2-D array.")

    flat = a.ravel()
    finite = np.isfinite(flat.astype(np.float64))
    vals = flat[finite]
    total = int(flat.size)
    n_finite = int(vals.size)
    masked_fraction = float((total - n_finite) / total) if total else 0.0

    mean = float(vals.mean()) if n_finite else float("nan")
    median = float(np.median(vals)) if n_finite else float("nan")
    std = float(vals.std()) if n_finite else float("nan")
    cv = float(std / abs(mean)) if n_finite and abs(mean) > 1e-12 else None

    # Background (dark floor) variability: the lowest-valued percentile band.
    bfloor = float(np.percentile(vals, background_percentile)) if n_finite else float("nan")
    bg = vals[vals <= bfloor]
    background_variation = float(bg.std()) if bg.size else 0.0

    # Noise sigma from the 2-D structure (Laplacian MAD).  NaN/Inf are
    # replaced by the median so the filter is stable and the reported sigma
    # reflects true pixel-level variation, not border effects.
    a2d = a[0::factor, 0::factor].astype(np.float32)
    finite2d = np.isfinite(a2d)
    fill = float(np.nanmedian(a2d)) if finite2d.any() else 0.0
    a_est = np.where(finite2d, a2d, fill)
    if a_est.size >= 9:
        noise_sigma = estimate_noise_sigma(a_est)
        if not np.isfinite(noise_sigma):
            noise_sigma = float("nan")
    else:
        noise_sigma = float("nan")

    loc_std = _local_std(a, factor)

    # A signal proxy: robust (5-95 percentile) amplitude.  Kept separate from
    # SNR so nobody misreads it as a validated SNR.
    if n_finite:
        signal_level = float(np.percentile(vals, 95) - np.percentile(vals, 5))
    else:
        signal_level = None

    diagnostics = NoiseDiagnostics(
        mean=mean,
        median=median,
        std=std,
        local_std=loc_std,
        coefficient_of_variation=cv,
        background_variation=background_variation,
        background_floor=bfloor,
        noise_sigma=float(noise_sigma) if np.isfinite(noise_sigma) else float("nan"),
        signal_level=signal_level,
        snr=None,
        snr_status="Not yet estimated",
        n_pixels=n_finite,
        masked_fraction=masked_fraction,
        extra={"local_window": local_window, "stride_factor": factor},
    )
    return diagnostics
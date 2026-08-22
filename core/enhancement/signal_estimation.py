"""LunaPSR -- local signal estimation (Phase 5).

Separates the image into three conceptual scales so the adaptive enhancer can
treat them differently:

    slow illumination      -> low-frequency, smooth component
    medium-scale structure -> local geology / albedo features
    high-frequency residual-> sensor-noise-dominated detail

The robust local signal estimate is the DN level that feeds the Phase 4B noise
model (``core.analysis.ohrc_noise_model``), because shot/read/quantization
noise depends on the *local* signal level, not on a single global value.

Assumptions (documented, not derived)
-------------------------------------
1. Illumination (lighting geometry, limb gradient, vignetting-like falloff)
   varies on scales larger than ``illumination_sigma`` pixels, so a Gaussian
   low-pass is a reasonable slowly-varying estimate.
2. The local median over ``local_window`` pixels is a robust estimate of the
   underlying signal: it is nearly insensitive to Poisson outliers and single
   hot pixels (median of an odd-sized window tolerates up to ~half the samples
   being outliers), unlike the mean.
3. The high-frequency residual ``data - robust_local`` is dominated by sensor
   noise plus sub-window texture.  It is treated as a noise-dominated residual;
   it is never automatically interpreted as geological structure.

None of these components is claimed to be a physical measurement: they are
*derived estimates* used to make enhancement adaptive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

__all__ = [
    "SignalEstimationParams",
    "LocalSignalComponents",
    "estimate_local_signal",
]

_ODD = (3, 5, 7, 9, 11, 15, 21, 31, 45)


def _nearest_odd(value: int) -> int:
    value = int(value)
    return min((odd for odd in _ODD if odd >= value), default=value if value % 2 else value + 1)


@dataclass(frozen=True)
class SignalEstimationParams:
    """Configurable parameters for local signal estimation.

    ``local_window`` -- odd window (px) for the robust local median.
    ``illumination_sigma`` -- Gaussian sigma (px) for the slowly-varying
        illumination estimate.
    ``use_median`` -- use median (default) instead of a trimmed mean as the
        robust local representation.
    """

    local_window: int = 9
    illumination_sigma: float = 9.0
    use_median: bool = True

    def __post_init__(self) -> None:
        if self.local_window < 3:
            raise ValueError("local_window must be >= 3")
        if self.illumination_sigma <= 0.0:
            raise ValueError("illumination_sigma must be > 0")


@dataclass
class LocalSignalComponents:
    """Separated signal scales (all same shape as input, float32).

    ``signal`` is the robust local signal level (DN) used by the noise model.
    ``illumination``  -- slowly-varying component (low-frequency).
    ``structure``     -- medium-scale structure = signal - illumination.
    ``residual``      -- high-frequency residual = data - signal.
    """

    signal: np.ndarray
    illumination: np.ndarray
    structure: np.ndarray
    residual: np.ndarray
    valid: np.ndarray
    params: SignalEstimationParams

    def to_dict(self) -> dict:
        return {
            "local_window": self.params.local_window,
            "illumination_sigma": self.params.illumination_sigma,
            "use_median": self.params.use_median,
            "valid_fraction": float(self.valid.mean()) if self.valid.size else 0.0,
        }


def _fill_invalid(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if valid.all():
        return out
    fill = float(np.median(out[valid])) if valid.any() else 0.0
    out[~valid] = fill
    return out


def estimate_local_signal(
    data: np.ndarray,
    params: Optional[SignalEstimationParams] = None,
    mask: Optional[np.ndarray] = None,
) -> LocalSignalComponents:
    """Estimate the three separated signal components of a 2-D scientific array.

    ``data`` is expected to be DN counts (float or int).  The input array is
    never modified.  Invalid pixels (NaN/Inf or ``mask == False``) are excluded
    from filtering and flagged in ``components.valid``; the returned components
    carry NaN at invalid positions.

    ``signal`` = robust local median (or, if ``use_median`` is False, the local
    mean) over ``local_window``.  ``illumination`` = Gaussian low-pass with
    ``illumination_sigma`` applied to the robust signal.  ``structure`` and
    ``residual`` follow by subtraction.
    """
    p = params or SignalEstimationParams()
    a = np.asarray(data)
    if a.ndim != 2:
        raise ValueError(f"estimate_local_signal expects a 2-D array, got {a.shape}")

    valid = np.isfinite(a)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError("mask shape does not match data shape")
        valid &= m
    if not valid.any():
        raise ValueError("No valid pixels available for local signal estimation.")

    filled = _fill_invalid(a, valid)
    window = _nearest_odd(p.local_window)

    if p.use_median:
        signal = median_filter(filled, size=window, mode="nearest").astype(np.float32)
    else:
        # local mean (box filter); kept for comparison / ablation
        from scipy.ndimage import uniform_filter

        signal = uniform_filter(filled, size=window, mode="nearest").astype(np.float32)

    illumination = gaussian_filter(
        signal, sigma=float(p.illumination_sigma), mode="nearest"
    ).astype(np.float32)
    structure = signal - illumination
    residual = filled - signal

    out_valid = np.full(a.shape, True, dtype=bool)
    if not valid.all():
        out_valid = valid

    return LocalSignalComponents(
        signal=np.where(out_valid, signal, np.nan),
        illumination=np.where(out_valid, illumination, np.nan),
        structure=np.where(out_valid, structure, np.nan),
        residual=np.where(out_valid, residual, np.nan),
        valid=out_valid,
        params=p,
    )
"""Phase 4B -- physics-aware OHRC noise model and model-based SNR (read-only).

This module implements the model recommended in ``docs/phase4a_ohrc_radiometry_noise.md``
section 15:

    Poisson shot noise
    + Gaussian read noise
    + 8-bit quantization noise

The result is a **MODEL-BASED ESTIMATE**, never a measured absolute SNR.

Units are explicit throughout and never mixed silently:

* ``DN``      -- 8-bit counts as delivered in the scientific product;
* ``e``       -- electrons at the detector;
* ``DN**2``   -- variance in count units (the quantization term lives here);
* ``e**2``    -- variance in electron units.

CRITICAL CAVEAT (see also ``docs/phase4b_noise_model.md``):

The gain of this model is **INFERRED** (Phase 4A, section 12): the published
full-well capacity (26.6 ke-) is taken to map onto the 10-bit full scale at the
published saturation radiance, giving ~26 e-/10-bit count, i.e. ~104 e-/8-bit
count for this MSB-selected product.  It is **NOT an officially published OHRC
calibration coefficient**.  It must never be presented as one.

TDI SCOPE: the current product was acquired with ``tdi_stages = TDI64``
(product label).  The model is scoped to that acquisition.  No TDI-dependent
scaling is applied because the available metadata does not justify it; the
limitation is exposed in the configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "OHRCNoiseModelConfig",
    "ModelNoiseReport",
    "FlatWindowResult",
    "shot_variance_e",
    "read_variance_e",
    "quantization_variance_dn2",
    "shot_sigma_dn",
    "read_sigma_dn",
    "quant_sigma_dn",
    "variance_total_dn",
    "sigma_total_dn",
    "model_snr",
    "noise_breakdown",
    "estimate_flat_window_noise",
]

_GAIN_NOTE = (
    "INFERRED (Phase 4A §12): full well 26.6 ke- assumed to map onto the 10-bit "
    "full scale at the published saturation radiance -> ~26 e-/10-bit count, "
    "~104 e-/8-bit count (MSB selection). NOT an official published OHRC "
    "calibration coefficient."
)

_TDI_NOTE = (
    "Model is scoped to this product's TDI64 acquisition (product label "
    "tdi_stages=TDI64). No TDI-dependent scaling is applied: the available "
    "metadata does not justify generalising across TDI settings."
)


@dataclass(frozen=True)
class OHRCNoiseModelConfig:
    """Configuration / assumptions object for the Poisson+Read+Quantization model.

    Every assumption lives here so it is never silently embedded in code.
    """

    type: str = "poisson_read_quantization"
    gain_e_per_dn: float = 104.0
    gain_status: str = "inferred"
    read_noise_e: float = 40.0
    quantization_variance_dn2: float = 1.0 / 12.0
    status: str = "model_based"
    tdi_stages: int = 64
    gain_note: str = _GAIN_NOTE
    tdi_scope_note: str = _TDI_NOTE

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "gain_e_per_dn": self.gain_e_per_dn,
            "gain_status": self.gain_status,
            "read_noise_e": self.read_noise_e,
            "quantization_variance_dn2": self.quantization_variance_dn2,
            "status": self.status,
            "tdi_stages": self.tdi_stages,
            "gain_note": self.gain_note,
            "tdi_scope_note": self.tdi_scope_note,
        }

    @classmethod
    def default(cls) -> "OHRCNoiseModelConfig":
        return cls()


# ---------------------------------------------------------------------------
# Per-component variance in electron units
# ---------------------------------------------------------------------------


def shot_variance_e(signal_e: float) -> float:
    """Poisson (shot) variance in electrons: ``var = signal``."""
    return float(max(signal_e, 0.0))


def read_variance_e(read_noise_e: float) -> float:
    """Gaussian read variance in electrons: ``var = read_noise**2``."""
    return float(read_noise_e ** 2)


def quantization_variance_dn2() -> float:
    """Uniform quantization variance in DN^2: ``1/12`` of one LSB."""
    return 1.0 / 12.0


# ---------------------------------------------------------------------------
# Per-component sigma in DN units (electron variance divided by gain^2)
# ---------------------------------------------------------------------------


def shot_sigma_dn(dn: float, gain_e_per_dn: float) -> float:
    """Shot-noise sigma in DN: ``sqrt(signal_e / gain**2) = sqrt(dn/gain)``."""
    dn = float(max(dn, 0.0))
    return float(np.sqrt(max(dn / gain_e_per_dn, 0.0)))


def read_sigma_dn(read_noise_e: float, gain_e_per_dn: float) -> float:
    """Read-noise sigma in DN: ``read_noise / gain``."""
    return float(read_noise_e / gain_e_per_dn)


def quant_sigma_dn() -> float:
    """Quantization sigma in DN: ``sqrt(1/12)``."""
    return float(np.sqrt(quantization_variance_dn2()))


# ---------------------------------------------------------------------------
# Total variance / sigma / model-based SNR
# ---------------------------------------------------------------------------


def variance_total_dn(dn: float, config: Optional[OHRCNoiseModelConfig] = None) -> float:
    """Total modeled variance in DN^2.

    ``var_total = shot(dn/gain) + read(noise^2/gain^2) + quant(1/12)``
    """
    cfg = config or OHRCNoiseModelConfig.default()
    dn = float(max(dn, 0.0))
    shot = shot_sigma_dn(dn, cfg.gain_e_per_dn) ** 2
    read = read_sigma_dn(cfg.read_noise_e, cfg.gain_e_per_dn) ** 2
    return float(shot + read + quantization_variance_dn2())


def sigma_total_dn(dn: float, config: Optional[OHRCNoiseModelConfig] = None) -> float:
    """Total modeled noise sigma in DN (includes shot, read, quantization)."""
    return float(np.sqrt(variance_total_dn(dn, config)))


def model_snr(dn: float, config: Optional[OHRCNoiseModelConfig] = None) -> float:
    """Model-based SNR (signal/noise, unit-less).

    ``SNR = signal_e / noise_e = (dn*gain) / (gain * sigma_dn) = dn / sigma_dn``
    """
    cfg = config or OHRCNoiseModelConfig.default()
    sigma = sigma_total_dn(dn, cfg)
    if sigma <= 0.0:
        return float("inf") if dn > 0 else 0.0
    return float(dn / sigma)


@dataclass
class ModelNoiseReport:
    """Per-pixel breakdown for one DN value, with explicit units."""

    dn: float
    gain_e_per_dn: float
    read_noise_e: float
    quantization_variance_dn2: float
    signal_e: float
    shot_variance_e: float
    read_variance_e: float
    shot_sigma_dn: float
    read_sigma_dn: float
    quant_sigma_dn: float
    total_variance_dn2: float
    total_sigma_dn: float
    model_snr: float
    status: str = "model_based"

    def to_dict(self) -> dict:
        return {
            "dn": round(self.dn, 6),
            "signal_e": round(self.signal_e, 3),
            "shot_variance_e": round(self.shot_variance_e, 3),
            "read_variance_e": round(self.read_variance_e, 3),
            "shot_sigma_dn": round(self.shot_sigma_dn, 6),
            "read_sigma_dn": round(self.read_sigma_dn, 6),
            "quant_sigma_dn": round(self.quant_sigma_dn, 6),
            "total_variance_dn2": round(self.total_variance_dn2, 6),
            "total_sigma_dn": round(self.total_sigma_dn, 6),
            "model_snr": round(self.model_snr, 4),
            "status": self.status,
        }


def noise_breakdown(dn: float, config: Optional[OHRCNoiseModelConfig] = None) -> ModelNoiseReport:
    """Full per-component breakdown for a DN value (units explicit)."""
    cfg = config or OHRCNoiseModelConfig.default()
    dn = float(max(dn, 0.0))
    signal_e = dn * cfg.gain_e_per_dn
    shot_var = shot_variance_e(signal_e)
    read_var = read_variance_e(cfg.read_noise_e)
    s_dn = shot_sigma_dn(dn, cfg.gain_e_per_dn)
    r_dn = read_sigma_dn(cfg.read_noise_e, cfg.gain_e_per_dn)
    q_dn = quant_sigma_dn()
    total_var = s_dn ** 2 + r_dn ** 2 + q_dn ** 2
    return ModelNoiseReport(
        dn=dn,
        gain_e_per_dn=cfg.gain_e_per_dn,
        read_noise_e=cfg.read_noise_e,
        quantization_variance_dn2=cfg.quantization_variance_dn2,
        signal_e=signal_e,
        shot_variance_e=shot_var,
        read_variance_e=read_var,
        shot_sigma_dn=s_dn,
        read_sigma_dn=r_dn,
        quant_sigma_dn=q_dn,
        total_variance_dn2=total_var,
        total_sigma_dn=float(np.sqrt(total_var)),
        model_snr=model_snr(dn, cfg),
        status=cfg.status,
    )


# ---------------------------------------------------------------------------
# Flat-window measured-noise estimator (Phase 4A §10.5 methodology)
# ---------------------------------------------------------------------------


@dataclass
class FlatWindowResult:
    """Measured vs modeled noise over selected low-structure (low-mean) windows.

    ``measured_sigma_dn`` is the mean window std; it is an estimate of the
    total random component (read + shot + quantization + any residual
    fixed-pattern).  It is **not** automatically pure sensor noise -- terrain
    structure can contribute, so low-structure (dark, flat) windows are used.
    """

    window_size: int
    stride: int
    n_windows: int
    mean_dn: float
    measured_sigma_dn: float
    model_sigma_dn: float
    residual_dn: float
    relative_error: float
    note: str = (
        "Measured spread includes possible terrain structure; not automatically "
        "pure sensor noise."
    )

    def to_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "stride": self.stride,
            "n_windows": self.n_windows,
            "mean_dn": round(self.mean_dn, 6),
            "measured_sigma_dn": round(self.measured_sigma_dn, 6),
            "model_sigma_dn": round(self.model_sigma_dn, 6),
            "residual_dn": round(self.residual_dn, 6),
            "relative_error": round(self.relative_error, 6),
            "note": self.note,
        }


def estimate_flat_window_noise(
    region: np.ndarray,
    config: Optional[OHRCNoiseModelConfig] = None,
    window: int = 11,
    stride: Optional[int] = None,
    mean_max: float = 5.0,
) -> FlatWindowResult:
    """Flat-window measured-vs-model noise comparison (Phase 4A methodology).

    Slides ``window x window`` windows (step ``stride``) over a 2-D scientific
    region, keeps the **low-signal / low-structure** windows (mean ``< mean_max``),
    and compares their measured spread with the model prediction at their mean.

    The ``mean_max`` selection is the same darkness-based flatness criterion used
    in Phase 4A (the "0-5 DN" bin): darkest regions are flattest, so terrain
    structure contributes least there.  The input is never modified.
    """
    cfg = config or OHRCNoiseModelConfig.default()
    if stride is None:
        stride = window
    a = np.asarray(region)
    if a.ndim != 2:
        raise ValueError("estimate_flat_window_noise expects a 2-D array.")
    if a.shape[0] < window or a.shape[1] < window:
        raise ValueError(
            f"region too small: need at least {window}x{window}, got {a.shape}"
        )
    means: list = []
    stds: list = []
    rows = range(0, a.shape[0] - window + 1, stride)
    cols = range(0, a.shape[1] - window + 1, stride)
    for r in rows:
        for c in cols:
            win = a[r : r + window, c : c + window].astype(np.float64)
            if not np.isfinite(win).all():
                continue
            means.append(float(win.mean()))
            stds.append(float(win.std()))
    if not means:
        raise ValueError("no finite windows in region")
    means = np.asarray(means)
    stds = np.asarray(stds)
    sel = means < mean_max
    if not sel.any():
        raise ValueError(f"no windows below mean_max={mean_max} (region not dark/flat enough)")
    mean_dn = float(means[sel].mean())
    measured = float(stds[sel].mean())
    model = sigma_total_dn(mean_dn, cfg)
    residual = measured - model
    rel = residual / model if model else float("nan")
    return FlatWindowResult(
        window_size=window,
        stride=stride,
        n_windows=int(sel.sum()),
        mean_dn=mean_dn,
        measured_sigma_dn=measured,
        model_sigma_dn=model,
        residual_dn=residual,
        relative_error=rel,
    )
"""LunaPSR -- model-based local confidence map (Phase 5).

Uses the Phase 4B physics-aware noise model (``core.analysis.ohrc_noise_model``)
to turn a local DN signal level into a *model-based* signal-to-noise figure and
a bounded, numerically stable confidence value.

Distinction (documented, not optional)
--------------------------------------
* ``model_based_local_snr`` is the ratio ``signal / modeled_noise_sigma`` in DN
  units.  Because the modeled noise is never zero (read + quantization floor
  alone give sigma ~ 0.48 DN), the ratio is always finite.
* ``local_model_confidence`` is the bounded, monotone map ``snr/(1 + snr)``
  (equivalently ``signal/(signal + noise)``), which lies in ``[0, 1)``.  It is
  **NOT** an absolute measured SNR and is **NOT** computed from array
  ``mean/std``.  It is a *research diagnostic* describing how trustworthy the
  local signal is relative to the modeled noise.

Confidence bands (LOW / MEDIUM / HIGH) are display categories with configurable
thresholds.  They make no independent scientific claim about the surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig

__all__ = [
    "ConfidenceMap",
    "model_based_local_snr",
    "local_model_confidence",
    "confidence_categories",
    "sigma_map_dn",
]

_DEFAULT_BANDS = (0.3, 0.7)  # (LOW, HIGH) confidence thresholds


def sigma_map_dn(
    signal_dn: np.ndarray,
    config: Optional[OHRCNoiseModelConfig] = None,
) -> np.ndarray:
    """Vectorised modeled-noise sigma (DN) for an array of local DN levels.

    Reuses the Phase 4B model: shot + read + quantization variance in DN^2.
    The read + quantization floor guarantees sigma > 0 everywhere, so
    signal/noise is always finite.
    """
    cfg = config or OHRCNoiseModelConfig.default()
    sig = np.asarray(signal_dn, dtype=np.float32)
    sig = np.maximum(np.where(np.isfinite(sig), sig, 0.0), 0.0)
    shot_var = sig / cfg.gain_e_per_dn
    read_var = (cfg.read_noise_e / cfg.gain_e_per_dn) ** 2
    var = shot_var + read_var + cfg.quantization_variance_dn2
    sigma = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    return np.where(np.isfinite(signal_dn), sigma, np.nan)


def model_based_local_snr(
    signal_dn: np.ndarray,
    config: Optional[OHRCNoiseModelConfig] = None,
    noise_floor_dn: float = 0.0,
) -> np.ndarray:
    """Model-based local SNR map (DN): ``signal / modeled_noise_sigma``.

    ``noise_floor_dn`` optionally adds a guard floor to the modeled noise
    (default 0.0 keeps the pure Phase 4B model).  The result is a research
    diagnostic, not an absolute measured SNR.
    """
    cfg = config or OHRCNoiseModelConfig.default()
    sig = np.asarray(signal_dn, dtype=np.float32)
    sigma = sigma_map_dn(sig, cfg)
    sigma_guard = sigma + float(max(noise_floor_dn, 0.0))
    snr = np.divide(
        np.maximum(np.where(np.isfinite(sig), sig, 0.0), 0.0),
        sigma_guard,
        out=np.zeros_like(sig, dtype=np.float32),
        where=sigma_guard > 0,
    )
    return np.where(np.isfinite(sig), snr, np.nan)


def local_model_confidence(
    signal_dn: np.ndarray,
    config: Optional[OHRCNoiseModelConfig] = None,
    noise_floor_dn: float = 0.0,
) -> np.ndarray:
    """Bounded local confidence map in ``[0, 1)``: ``snr/(1 + snr)``.

    Monotone in the model-based SNR, numerically stable, finite everywhere
    (at signal 0 the confidence is 0; at large signal it saturates towards 1).
    """
    snr = model_based_local_snr(signal_dn, config, noise_floor_dn)
    return np.where(np.isfinite(snr), snr / (1.0 + np.maximum(snr, 0.0)), np.nan)


def confidence_categories(
    confidence: np.ndarray,
    bands: tuple[float, float] = _DEFAULT_BANDS,
) -> np.ndarray:
    """Integer band map: 0 = LOW, 1 = MEDIUM, 2 = HIGH confidence.

    Thresholds are display categories only; they carry no independent
    scientific claim.  ``bands = (low, high)`` with ``0 <= low < high <= 1``.
    """
    low, high = float(bands[0]), float(bands[1])
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("confidence bands must satisfy 0 <= low < high <= 1")
    c = np.asarray(confidence, dtype=np.float32)
    out = np.zeros(c.shape, dtype=np.uint8)
    out[c >= high] = 2
    out[(c >= low) & (c < high)] = 1
    out[np.isnan(c)] = 255  # invalid pixel marker (not a category)
    return out


@dataclass
class ConfidenceMap:
    """Bundled confidence outputs (all derived, all diagnostic)."""

    confidence: np.ndarray          # [0,1) bounded confidence
    model_based_local_snr: np.ndarray
    noise_sigma_dn: np.ndarray      # modeled noise sigma (DN)
    categories: np.ndarray          # 0/1/2/255 band map
    bands: tuple = field(default_factory=lambda: _DEFAULT_BANDS)
    formula: str = "confidence = snr/(1+snr); snr = signal/modeled_noise_sigma"
    status: str = "model_based"

    def fractions(self) -> dict:
        valid = self.categories < 255
        n = int(valid.sum())
        if n == 0:
            return {"low": 0.0, "medium": 0.0, "high": 0.0}
        return {
            "low": float((self.categories[valid] == 0).mean()),
            "medium": float((self.categories[valid] == 1).mean()),
            "high": float((self.categories[valid] == 2).mean()),
        }

    def to_dict(self) -> dict:
        return {
            "confidence_mean": float(np.nanmean(self.confidence)),
            "confidence_median": float(np.nanmedian(self.confidence)),
            "snr_mean": float(np.nanmean(self.model_based_local_snr)),
            "snr_median": float(np.nanmedian(self.model_based_local_snr)),
            "noise_sigma_dn_mean": float(np.nanmean(self.noise_sigma_dn)),
            "fractions": self.fractions(),
            "bands": list(self.bands),
            "formula": self.formula,
            "status": self.status,
        }
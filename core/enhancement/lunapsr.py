"""LunaPSR -- physics-guided adaptive enhancement (Phase 5, research prototype).

Core idea
---------
Enhancement strength is **not** applied identically everywhere.  It depends on
the local signal and the Phase 4B modeled noise:

    scientific DN -> local signal estimate -> modeled noise -> confidence
        -> adaptive strength -> tone curve -> structure-aware local contrast
        -> enhanced representation

Every number this module produces is a **derived representation** or a
**research diagnostic**.  LunaPSR does not claim to recover information that
OHRC did not capture, does not fabricate structure, and does not measure
absolute SNR.

Fairness contract
-----------------
``enhance_lunapsr`` takes the *same* normalised [0,1] representation that every
Phase 3 baseline receives (``normalized``).  The DN-level local signal is used
only to derive the spatially adaptive strength, the confidence map and the
diagnostics -- never to change the tone-mapping input.  No special
preprocessing is given to LunaPSR that the other methods do not receive.

Never mutated: ``scientific`` and ``normalized`` are read-only here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
from core.enhancement.confidence import (
    local_model_confidence,
    model_based_local_snr,
    sigma_map_dn,
)
from core.enhancement.signal_estimation import (
    SignalEstimationParams,
    estimate_local_signal,
)

__all__ = [
    "LunaPSRParams",
    "LunaPSRResult",
    "LunaPSRDiagnostics",
    "enhance_lunapsr",
    "adaptive_strength_map",
    "smoothstep",
    "noise_amplification_ratio",
    "structure_preservation_metrics",
    "artifact_diagnostics",
    "lunapsr_score",
    "full_diagnostics",
    "derive_dn",
    "benchmark_lunapsr",
]

_COMPONENTS = ("tone_only", "tone_adaptive", "adaptive_contrast", "full")


@dataclass(frozen=True)
class LunaPSRParams:
    """Enhancement parameters (every value documented and configurable).

    ``local_window`` -- odd window (px) for the robust local signal median.
    ``illumination_sigma`` -- Gaussian sigma (px) for the slowly-varying
        illumination estimate.
    ``min_strength`` -- adaptive strength floor (0..1): conservative
        enhancement even where confidence is low.
    ``max_strength`` -- adaptive strength ceiling (0..1): strongest effect
        where confidence is high.
    ``confidence_low`` / ``confidence_high`` -- confidence band thresholds;
        the strength ramp is a smoothstep between them.
    ``gamma_min`` -- floor for the spatially varying gamma (gamma in
        [gamma_min, 1]; gamma = 1 means no tone change).
    ``local_contrast_strength`` -- max gain of the confidence-weighted local
        contrast term.
    ``contrast_window`` -- odd window (px) for the structure-aware median
        detail.
    ``structure_preservation`` -- scaling of the structure-aware contrast gain
        (LunaPSR full mode).
    ``noise_floor_dn`` -- optional guard floor added to the modeled noise for
        the confidence computation (default 0 keeps the pure Phase 4B model).
    ``components`` -- ablation switch: 'tone_only' | 'tone_adaptive' |
        'adaptive_contrast' | 'full'.
    ``version`` -- algorithm version string recorded with experiments.
    """

    local_window: int = 9
    illumination_sigma: float = 9.0
    min_strength: float = 0.25
    max_strength: float = 0.90
    confidence_low: float = 0.30
    confidence_high: float = 0.70
    gamma_min: float = 0.30
    local_contrast_strength: float = 0.35
    contrast_window: int = 5
    structure_preservation: float = 1.0
    noise_floor_dn: float = 0.0
    components: str = "full"
    version: str = "lunapsr-0.1.0"
    status: str = "model_based"

    def __post_init__(self) -> None:
        if self.local_window < 3:
            raise ValueError("local_window must be >= 3")
        if self.contrast_window < 3:
            raise ValueError("contrast_window must be >= 3")
        if not (0.0 <= self.min_strength <= self.max_strength <= 1.0):
            raise ValueError("strength bounds must satisfy 0 <= min <= max <= 1")
        if not (0.0 < self.gamma_min <= 1.0):
            raise ValueError("gamma_min must be in (0, 1]")
        if not (0.0 <= self.confidence_low < self.confidence_high <= 1.0):
            raise ValueError("confidence bands must satisfy 0 <= low < high <= 1")
        if self.components not in _COMPONENTS:
            raise ValueError(f"components must be one of {_COMPONENTS}")
        if self.noise_floor_dn < 0.0:
            raise ValueError("noise_floor_dn must be >= 0")

    def to_dict(self) -> dict:
        d = {
            "local_window": self.local_window,
            "illumination_sigma": self.illumination_sigma,
            "min_strength": self.min_strength,
            "max_strength": self.max_strength,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "gamma_min": self.gamma_min,
            "local_contrast_strength": self.local_contrast_strength,
            "contrast_window": self.contrast_window,
            "structure_preservation": self.structure_preservation,
            "noise_floor_dn": self.noise_floor_dn,
            "components": self.components,
            "version": self.version,
            "status": self.status,
        }
        return d


# ---------------------------------------------------------------------------
# Adaptive strength (the defining LunaPSR idea)
# ---------------------------------------------------------------------------


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Smooth, continuous, monotone ramp: ``t^2(3 - 2t)`` for t in [0,1]."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def adaptive_strength_map(
    confidence: np.ndarray,
    params: Optional[LunaPSRParams] = None,
) -> np.ndarray:
    """Spatially adaptive, smooth, bounded enhancement strength in [0,1].

    ``strength = min + (max - min) * smoothstep(t)`` with
    ``t = clip((confidence - low)/(high - low), 0, 1)``.

    Continuous and monotone in confidence: LOW confidence -> strength near
    ``min_strength`` (conservative); HIGH confidence -> strength near
    ``max_strength`` (stronger contrast/detail).
    """
    p = params or LunaPSRParams()
    c = np.asarray(confidence, dtype=np.float32)
    t = (c - p.confidence_low) / max(p.confidence_high - p.confidence_low, 1e-9)
    strength = p.min_strength + (p.max_strength - p.min_strength) * smoothstep(t)
    return np.where(np.isfinite(c), strength, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# Tone curve and local contrast
# ---------------------------------------------------------------------------


def _fill_valid(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if not valid.all():
        fill = float(np.median(out[valid])) if valid.any() else 0.0
        out[~valid] = fill
    return out


def _gamma_map(strength: np.ndarray, params: LunaPSRParams) -> np.ndarray:
    gamma = 1.0 - strength * (1.0 - params.gamma_min)
    return np.clip(gamma, params.gamma_min, 1.0).astype(np.float32)


def _tone_curve(normalized: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.power(normalized, gamma).astype(np.float32)


def _structure_weight(structure_dn: np.ndarray, noise_sigma_dn: np.ndarray) -> np.ndarray:
    """Weight favouring medium-scale structure over noise-dominated residual.

    ``w_s = |structure| / (|structure| + noise_sigma)`` (element-wise, [0,1]).
    High where the medium-scale structure clearly exceeds the modeled noise.
    """
    s = np.abs(np.asarray(structure_dn, dtype=np.float32))
    n = np.maximum(np.asarray(noise_sigma_dn, dtype=np.float32), 1e-9)
    w = s / (s + n)
    return np.where(np.isfinite(s) & np.isfinite(n), w, 0.0).astype(np.float32)


def _local_contrast(
    tone: np.ndarray,
    confidence: np.ndarray,
    structure_weight: np.ndarray,
    valid: np.ndarray,
    params: LunaPSRParams,
) -> np.ndarray:
    """Structure-aware, confidence-weighted local contrast enhancement.

    ``detail = tone - median(tone, contrast_window)`` (median subtraction
    avoids Gaussian-unsharp halos).  ``gain = local_contrast_strength *
    confidence * structure_preservation * structure_weight`` where
    ``structure_weight`` is 1 for the ``adaptive_contrast`` ablation variant
    and the medium-scale structure weight ``w_s`` for the full LunaPSR mode.
    Where confidence is low or the residual is noise-dominated (``w_s ~ 0``),
    contrast amplification is small.
    """
    base = _fill_valid(tone, valid)
    window = int(params.contrast_window)
    if window % 2 == 0:
        window += 1
    smooth = median_filter(base, size=window, mode="nearest").astype(np.float32)
    detail = base - smooth
    conf = _fill_valid(np.asarray(confidence, dtype=np.float32), valid)
    conf = np.nan_to_num(conf, nan=0.0)
    w = np.nan_to_num(np.asarray(structure_weight, dtype=np.float32), nan=0.0)
    gain = (
        params.local_contrast_strength * conf * params.structure_preservation * w
    ).astype(np.float32)
    enhanced = tone + gain * detail
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def derive_dn(normalized: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    """Map a [0,1] array back to derived DN using the normalization bounds."""
    lo, hi = float(bounds[0]), float(bounds[1])
    return np.nan_to_num(np.asarray(normalized, dtype=np.float32) * (hi - lo) + lo)


def _flat_window_ratio(
    orig_dn: np.ndarray,
    enh_dn: np.ndarray,
    window: int = 11,
    stride: Optional[int] = None,
    mean_max: float = 5.0,
) -> dict:
    """Measured flat-window noise before/after on the SAME spatial windows.

    Windows are selected on the original DN (low-signal, flat).  The std of the
    same windows in the enhanced derived DN gives an *empirical* noise
    amplification ratio.  The tone curve changes local statistics, so the ratio
    is a diagnostic, not a physical measurement.
    """
    stride = stride or window
    a = np.asarray(orig_dn, dtype=np.float32)
    b = np.asarray(enh_dn, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("orig_dn and enh_dn must have the same shape")
    h, w = a.shape
    if h < window or w < window:
        raise ValueError(f"array too small for {window}x{window} windows")
    before = []
    after = []
    for r in range(0, h - window + 1, stride):
        for c in range(0, w - window + 1, stride):
            wo = a[r : r + window, c : c + window]
            if not np.isfinite(wo).all():
                continue
            if float(wo.mean()) >= mean_max:
                continue
            before.append(float(wo.std()))
            we = b[r : r + window, c : c + window]
            if np.isfinite(we).all():
                after.append(float(we.std()))
            else:
                after.append(float("nan"))
    if not before:
        return {"n_windows": 0, "sigma_before": float("nan"),
                "sigma_after": float("nan"), "ratio": float("nan")}
    sigma_before = float(np.mean(before))
    valid_after = [x for x in after if np.isfinite(x)]
    if not valid_after:
        return {"n_windows": len(before), "sigma_before": sigma_before,
                "sigma_after": float("nan"), "ratio": float("nan")}
    sigma_after = float(np.mean(valid_after))
    return {
        "n_windows": len(before),
        "sigma_before": sigma_before,
        "sigma_after": sigma_after,
        "ratio": sigma_after / max(sigma_before, 1e-9),
    }


def noise_amplification_ratio(
    orig_norm: np.ndarray,
    enh_norm: np.ndarray,
    orig_dn: np.ndarray,
    enh_dn: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
) -> dict:
    """Noise amplification diagnostics (research diagnostics, not verdicts).

    - ``laplacian_ratio``: Immerkaer/Laplacian noise estimate after/before in
      [0,1] units (proxy).
    - ``flat_window``: measured ratio on the same low-signal windows, computed
      from derived DN.  Requires ``enh_dn`` (the enhanced array mapped back to
      DN with the exact normalization bounds); otherwise reported as not
      computed.
    An increase is not automatically bad: enhancement can legitimately raise
    local contrast; the ratio is a benchmark quantity.
    """
    from core.metrics.noise import estimate_noise_sigma

    if valid is None:
        valid = np.isfinite(np.asarray(orig_norm))
    o = _fill_valid(np.asarray(orig_norm, dtype=np.float32), valid)
    e = _fill_valid(np.asarray(enh_norm, dtype=np.float32), valid)
    sb = max(estimate_noise_sigma(o), 1e-9)
    sa = max(estimate_noise_sigma(e), 1e-9)
    if enh_dn is None:
        fw = {
            "n_windows": 0,
            "sigma_before": float("nan"),
            "sigma_after": float("nan"),
            "ratio": float("nan"),
            "note": "Not computed: normalization bounds required to derive DN.",
        }
    else:
        fw = _flat_window_ratio(orig_dn, enh_dn)
    return {
        "sigma_before": float(sb),
        "sigma_after": float(sa),
        "laplacian_ratio": float(sa / sb),
        "flat_window": fw,
    }


def _sobel_components(a: np.ndarray) -> tuple:
    import cv2

    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def structure_preservation_metrics(
    orig: np.ndarray,
    enhanced: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Structure-alignment metrics between input and enhanced (both [0,1]).

    - ``edge_preservation_index``: normalized cross-correlation of Sobel
      magnitudes (proxy; sharpening can inflate it).
    - ``gradient_orientation_similarity``: mean |cos(delta_theta)| over pixels
      where both gradients are significant (0..1).
    - ``edge_displacement_px``: mean distance from enhanced edges to the
      nearest original edge (0 = perfectly aligned).
    - ``structure_correlation``: Pearson correlation of the medium-scale
      structure components (input vs enhanced).

    These measure *alignment*, not scientific correctness.
    """
    from core.metrics.core import edge_preservation_index

    o = _fill_valid(np.asarray(orig, dtype=np.float32), valid)
    e = _fill_valid(np.asarray(enhanced, dtype=np.float32), valid)
    epi = edge_preservation_index(o, e)

    gxo, gyo, mago = _sobel_components(o)
    gxe, gye, mage = _sobel_components(e)
    th_o = max(float(np.percentile(mago, 75)), 1e-9)
    th_e = max(float(np.percentile(mage, 75)), 1e-9)
    both = (mago > th_o) & (mage > th_e)
    if both.sum() == 0:
        grad_sim = float("nan")
    else:
        denom = (mago * mage)[both] + 1e-9
        cosd = ((gxo * gxe + gyo * gye)[both]) / denom
        grad_sim = float(np.mean(np.abs(cosd)))

    edges_o = mago > max(float(np.percentile(mago, 90)), 1e-9)
    edges_e = mage > max(float(np.percentile(mage, 90)), 1e-9)
    if edges_o.any() and edges_e.any():
        import cv2

        dist = cv2.distanceTransform((~edges_o).astype(np.uint8), cv2.DIST_L2, 3)
        displacement = float(np.mean(dist[edges_e]))
    else:
        displacement = float("nan")

    struct_o = o - gaussian_filter(o, sigma=3.0, mode="nearest")
    struct_e = e - gaussian_filter(e, sigma=3.0, mode="nearest")
    v = valid
    if struct_o[v].size < 2 or float(np.std(struct_o[v])) == 0.0:
        struct_corr = float("nan")
    else:
        struct_corr = float(
            np.corrcoef(struct_o[v], struct_e[v])[0, 1]
        )

    return {
        "edge_preservation_index": float(epi),
        "gradient_orientation_similarity": grad_sim,
        "edge_displacement_px": displacement,
        "structure_correlation": struct_corr,
    }


def artifact_diagnostics(
    orig: np.ndarray,
    enhanced: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Artifact safeguards.  Returns diagnostics plus a human-readable warning.

    Flags (each documented as a diagnostic):
    - ``clipping_fraction`` before/after and ``new_clipping`` (fraction newly
      clipped to 0/1).
    - ``isolated_pixel_amplification``: fraction of pixels that became isolated
      bright spikes after enhancement.
    - ``high_frequency_energy_ratio``: Laplacian-variance ratio (proxy for
      excessive high-frequency energy / ringing).
    - ``halo_score``: mean overshoot near strong edges relative to edge
      strength (halos).
    - ``ringing_proxy``: oscillation of the residual in the edge band.
    """
    from core.metrics.core import sharpness_laplacian

    o = _fill_valid(np.asarray(orig, dtype=np.float32), valid)
    e = _fill_valid(np.asarray(enhanced, dtype=np.float32), valid)

    clip_before = float(np.mean((o == 0.0) | (o == 1.0)))
    clip_after = float(np.mean((e == 0.0) | (e == 1.0)))
    new_clipping = max(0.0, clip_after - clip_before)

    med3 = median_filter(e, size=3, mode="nearest").astype(np.float32)
    isolated = np.mean((e > med3 + 0.02))
    iso_before = np.mean((o > median_filter(o, size=3, mode="nearest") + 0.02))
    isolated_amplification = max(0.0, float(isolated - iso_before))

    hf_ratio = max(
        sharpness_laplacian(e) / max(sharpness_laplacian(o), 1e-9), 0.0
    )

    _, _, mago = _sobel_components(o)
    edge_band = mago > max(float(np.percentile(mago, 90)), 1e-9)
    smooth = gaussian_filter(e, sigma=3.0, mode="nearest")
    overshoot = np.abs(e - smooth)
    if edge_band.sum() > 0:
        edge_strength = float(np.mean(mago[edge_band]))
        halo_score = float(np.mean(overshoot[edge_band])) / max(edge_strength, 1e-9)
        ringing_proxy = float(np.std((e - smooth)[edge_band]))
    else:
        halo_score = float("nan")
        ringing_proxy = float("nan")

    warnings = []
    if new_clipping > 0.005:
        warnings.append("New clipping detected at the extremes of the range.")
    if np.isfinite(halo_score) and halo_score > 0.05:
        warnings.append("Possible halo artifacts near strong edges.")
    if hf_ratio > 3.0:
        warnings.append("Excessive high-frequency energy relative to input.")
    if isolated_amplification > 0.001:
        warnings.append("Isolated-pixel amplification detected.")

    return {
        "clipping_fraction_before": clip_before,
        "clipping_fraction_after": clip_after,
        "new_clipping": new_clipping,
        "isolated_pixel_amplification": isolated_amplification,
        "high_frequency_energy_ratio": float(hf_ratio),
        "halo_score": halo_score,
        "ringing_proxy": ringing_proxy,
        "warnings": warnings,
    }


def lunapsr_score(
    orig: np.ndarray,
    enhanced: np.ndarray,
    structure: dict,
    noise_amp: dict,
    valid: np.ndarray,
) -> dict:
    """LunaPSR enhancement score (RESEARCH DIAGNOSTIC, not a measurement).

    ``score = visibility_gain_norm + structure_preservation - noise_penalty``
    with weights (1, 1, 1), each term saturated into [0,1] so the score is
    bounded.  It is explicitly not scientifically validated and is only a
    benchmark quantity for the ablation study.
    """
    from core.metrics.core import robust_contrast

    o = _fill_valid(np.asarray(orig, dtype=np.float32), valid)
    e = _fill_valid(np.asarray(enhanced, dtype=np.float32), valid)
    gain = max(0.0, robust_contrast(e) - robust_contrast(o))
    visibility_norm = gain / (gain + 0.1)

    pres_terms = [
        structure.get("edge_preservation_index"),
        structure.get("gradient_orientation_similarity"),
    ]
    finite_pres = [t for t in pres_terms if t is not None and np.isfinite(t)]
    structure_score = float(np.mean(finite_pres)) if finite_pres else 0.0

    ratio = noise_amp.get("laplacian_ratio")
    if ratio is not None and np.isfinite(ratio):
        penalty = max(0.0, ratio - 1.0)
        noise_penalty = penalty / (penalty + 1.0)
    else:
        noise_penalty = 0.0

    score = 1.0 * visibility_norm + 1.0 * structure_score - 1.0 * noise_penalty
    return {
        "visibility_gain_norm": float(visibility_norm),
        "structure_preservation": float(structure_score),
        "noise_penalty_norm": float(noise_penalty),
        "score": float(score),
        "status": "research_diagnostic",
    }


# ---------------------------------------------------------------------------
# Public enhancement API
# ---------------------------------------------------------------------------


@dataclass
class LunaPSRDiagnostics:
    """Bundled diagnostics produced by :func:`full_diagnostics`."""

    noise_amplification: dict
    structure_preservation: dict
    artifacts: dict
    score: dict

    def to_dict(self) -> dict:
        return {
            "noise_amplification": self.noise_amplification,
            "structure_preservation": self.structure_preservation,
            "artifacts": self.artifacts,
            "score": self.score,
        }

    def warnings(self) -> list:
        return list(self.artifacts.get("warnings", []))


@dataclass
class LunaPSRResult:
    """Result of :func:`enhance_lunapsr`."""

    enhanced: np.ndarray            # [0,1] float32, NaN at invalid pixels
    confidence: np.ndarray          # [0,1) confidence map
    model_based_local_snr: np.ndarray
    noise_sigma_dn: np.ndarray      # modeled noise sigma (DN)
    strength_map: np.ndarray        # [0,1] adaptive strength
    gamma_map: np.ndarray           # spatially varying gamma
    diagnostics: LunaPSRDiagnostics
    parameters: dict
    metadata: dict
    status: str = "model_based"
    caveat: str = (
        "Enhanced representation of the available OHRC signal; no claim of "
        "recovering information OHRC did not capture."
    )

    def to_dict(self) -> dict:
        return {
            "enhanced_shape": list(self.enhanced.shape),
            "enhanced_min": float(np.nanmin(self.enhanced)),
            "enhanced_max": float(np.nanmax(self.enhanced)),
            "enhanced_mean": float(np.nanmean(self.enhanced)),
            "confidence_mean": float(np.nanmean(self.confidence)),
            "snr_mean": float(np.nanmean(self.model_based_local_snr)),
            "noise_sigma_dn_mean": float(np.nanmean(self.noise_sigma_dn)),
            "strength_mean": float(np.nanmean(self.strength_map)),
            "gamma_mean": float(np.nanmean(self.gamma_map)),
            "diagnostics": self.diagnostics.to_dict(),
            "parameters": self.parameters,
            "metadata": self.metadata,
            "status": self.status,
            "caveat": self.caveat,
        }


def _common_valid(scientific: np.ndarray, normalized: np.ndarray, mask) -> np.ndarray:
    valid = np.isfinite(np.asarray(scientific))
    valid &= np.isfinite(np.asarray(normalized))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != valid.shape:
            raise ValueError("mask shape does not match data shape")
        valid &= m
    if not valid.any():
        raise ValueError("No valid pixels available for LunaPSR enhancement.")
    return valid


def enhance_lunapsr(
    scientific: np.ndarray,
    normalized: np.ndarray,
    mask: Optional[np.ndarray] = None,
    noise_model: Optional[OHRCNoiseModelConfig] = None,
    params: Optional[LunaPSRParams] = None,
    bounds: Optional[Sequence[float]] = None,
    compute_diagnostics: bool = True,
) -> LunaPSRResult:
    """Run the LunaPSR adaptive enhancement (research prototype).

    ``scientific`` -- DN counts (2-D).  ``normalized`` -- the [0,1] float32
    representation (identical to what every Phase 3 baseline receives).
    ``mask`` -- optional boolean valid-pixel mask.  ``noise_model`` -- Phase 4B
    ``OHRCNoiseModelConfig``.  ``params`` -- ``LunaPSRParams``.  ``bounds`` --
    the exact (lo, hi) normalization bounds used to derive DN for the
    flat-window noise-amplification diagnostic (recommended; when omitted that
    single diagnostic is reported as not computed).  ``compute_diagnostics`` --
    set False for tiled full-resolution processing where the heavy research
    diagnostics (edge displacement, distance transforms) would dominate cost;
    the maps and enhanced image are unaffected.

    Neither ``scientific`` nor ``normalized`` is ever mutated.  The returned
    ``enhanced`` keeps NaN where pixels are invalid.
    """
    cfg = noise_model or OHRCNoiseModelConfig.default()
    p = params or LunaPSRParams()

    sci = np.asarray(scientific)
    norm = np.asarray(normalized, dtype=np.float32)
    if sci.ndim != 2 or norm.ndim != 2:
        raise ValueError("scientific and normalized must be 2-D")
    if sci.shape != norm.shape:
        raise ValueError("scientific and normalized must have the same shape")

    valid = _common_valid(sci, norm, mask)

    # 1. local signal + 2. modeled noise (Phase 4B)
    components = estimate_local_signal(
        sci,
        SignalEstimationParams(
            local_window=p.local_window,
            illumination_sigma=p.illumination_sigma,
        ),
        mask=valid,
    )
    signal_dn = components.signal
    noise_sigma_dn = sigma_map_dn(signal_dn, cfg)

    # 3. confidence map
    snr = model_based_local_snr(signal_dn, cfg, p.noise_floor_dn)
    confidence = local_model_confidence(signal_dn, cfg, p.noise_floor_dn)

    # 4. adaptive strength + 5. tone curve
    strength = adaptive_strength_map(confidence, p)
    if p.components == "tone_only":
        # constant strength (max) -> constant gamma; no confidence adaptation.
        strength = np.full_like(confidence, p.max_strength)
    gamma = _gamma_map(strength, p)
    tone = _tone_curve(norm, gamma)

    # 6. local contrast (confidence- and structure-constrained)
    structure_w = _structure_weight(components.structure, noise_sigma_dn)
    if p.components == "adaptive_contrast":
        enhanced = _local_contrast(tone, confidence, np.ones_like(structure_w), valid, p)
    elif p.components == "full":
        enhanced = _local_contrast(tone, confidence, structure_w, valid, p)
    else:
        enhanced = tone.copy()

    # invalid pixels -> NaN (restore)
    if not valid.all():
        enhanced = np.where(valid, enhanced, np.nan)
        confidence = np.where(valid, confidence, np.nan)
        snr = np.where(valid, snr, np.nan)
        noise_sigma_dn = np.where(valid, noise_sigma_dn, np.nan)
        strength = np.where(valid, strength, np.nan)
        gamma = np.where(valid, gamma, np.nan)

    # 9./10./11. diagnostics
    if compute_diagnostics:
        diag = full_diagnostics(
            orig_norm=norm, enh_norm=enhanced,
            orig_dn=sci,
            enh_dn=None if bounds is None else derive_dn(enhanced, bounds),
            valid=valid, params=p,
        )
    else:
        diag = LunaPSRDiagnostics(
            noise_amplification={"laplacian_ratio": float("nan"), "flat_window": {}},
            structure_preservation={},
            artifacts={"warnings": []},
            score={"score": float("nan"), "status": "research_diagnostic"},
        )

    metadata = {
        "method": "lunapsr",
        "version": p.version,
        "components": p.components,
        "noise_model": cfg.to_dict(),
        "caveat": (
            "Research prototype. Enhanced representation of the available "
            "OHRC signal; no claim of recovering information OHRC did not capture."
        ),
    }
    return LunaPSRResult(
        enhanced=enhanced,
        confidence=confidence,
        model_based_local_snr=snr,
        noise_sigma_dn=noise_sigma_dn,
        strength_map=strength,
        gamma_map=gamma,
        diagnostics=diag,
        parameters=p.to_dict(),
        metadata=metadata,
        status=cfg.status,
    )


def full_diagnostics(
    orig_norm: np.ndarray,
    enh_norm: np.ndarray,
    orig_dn: np.ndarray,
    enh_dn: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    params: Optional[LunaPSRParams] = None,
    bounds: Optional[Sequence[float]] = None,
) -> LunaPSRDiagnostics:
    """Bundle noise-amplification, structure-preservation, artifact and score
    diagnostics.  ``bounds`` are the normalization bounds used to derive DN
    from the enhanced [0,1] array (exact in the lab / runner)."""
    if bounds is not None and enh_dn is None:
        enh_dn = derive_dn(enh_norm, bounds)
    noise_amp = noise_amplification_ratio(orig_norm, enh_norm, orig_dn, enh_dn, valid)
    structure = structure_preservation_metrics(orig_norm, enh_norm, valid)
    artifacts = artifact_diagnostics(orig_norm, enh_norm, valid)
    score = lunapsr_score(orig_norm, enh_norm, structure, noise_amp, valid)
    return LunaPSRDiagnostics(
        noise_amplification=noise_amp,
        structure_preservation=structure,
        artifacts=artifacts,
        score=score,
    )


# ---------------------------------------------------------------------------
# Benchmark: all methods on the same normalized input (fair comparison)
# ---------------------------------------------------------------------------

_BENCHMARK_METHODS = [
    "none",
    "gamma",
    "histogram_equalization",
    "clahe",
    "unsharp_mask",
    "denoising",
    "retinex",
    "zero_dce",
    "lunapsr",
]


def benchmark_lunapsr(
    normalized: np.ndarray,
    base_dn: np.ndarray,
    mask: Optional[np.ndarray],
    bounds: Sequence[float],
    params: Optional[LunaPSRParams] = None,
    noise_model: Optional[OHRCNoiseModelConfig] = None,
    methods: Optional[Sequence[str]] = None,
) -> dict:
    """Run every Phase 3 baseline *and* LunaPSR on the identical normalized
    input and report the same evaluation framework for each.

    Never declares a winner: it returns a flat dict of per-method metric
    rows for the UI / ablation study.  Model-based SNR is reported for every
    method (derived from the enhanced DN and the Phase 4B model).
    """
    from core.analysis.metrics import quality_metrics
    from core.enhancement.baseline import apply_baseline

    cfg = noise_model or OHRCNoiseModelConfig.default()
    p = params or LunaPSRParams()
    methods = methods or _BENCHMARK_METHODS

    valid = np.isfinite(np.asarray(base_dn))
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    results: Dict[str, dict] = {}
    lo, hi = float(bounds[0]), float(bounds[1])

    for method in methods:
        if method == "lunapsr":
            res = enhance_lunapsr(base_dn, normalized, mask=mask,
                                  noise_model=cfg, params=p)
            enh = res.enhanced
        elif method == "zero_dce":
            try:
                from core.enhancement.zero_dce import is_available as zdce

                if not zdce():
                    results[method] = {"error": "Zero-DCE unavailable"}
                    continue
                from core.enhancement.zero_dce import zero_dce_enhance

                enh = zero_dce_enhance(np.nan_to_num(normalized, nan=0.0))
            except Exception as exc:  # noqa: BLE001 - surfaced in the table
                results[method] = {"error": str(exc)}
                continue
        else:
            enh = apply_baseline(np.nan_to_num(normalized, nan=0.0), method,
                                 **({} if method == "none" else _baseline_params(method)))

        enh = np.asarray(enh, dtype=np.float32)
        metrics = quality_metrics(np.nan_to_num(enh, nan=0.0))
        enh_dn = derive_dn(enh, (lo, hi))
        model_snr = model_based_local_snr(enh_dn, cfg, p.noise_floor_dn)

        diag = full_diagnostics(
            np.nan_to_num(normalized, nan=0.0), np.nan_to_num(enh, nan=0.0),
            base_dn, enh_dn, valid, params=p, bounds=(lo, hi),
        )
        row = {
            "method": method,
            "contrast": metrics["contrast"],
            "entropy": metrics["entropy"],
            "gradient_magnitude": metrics["gradient_magnitude"],
            "edge_density": metrics["edge_density"],
            "noise_sigma": metrics["noise_sigma"],
            "model_snr_median": float(np.nanmedian(model_snr)),
            "noise_amplification_ratio": diag.noise_amplification["laplacian_ratio"],
            "structure_preservation": diag.structure_preservation[
                "edge_preservation_index"
            ],
            "clipping_fraction": diag.artifacts["clipping_fraction_after"],
            "score": diag.score["score"],
            "warnings": diag.warnings(),
        }
        results[method] = row

    return results


def _baseline_params(method: str) -> dict:
    from configs.defaults import BASELINE_DEFAULTS

    return dict(BASELINE_DEFAULTS.get(method, {}))


def benchmark_table(results: dict) -> list:
    """Flatten the benchmark dict into a list of dicts for a DataFrame."""
    rows = []
    for method, row in results.items():
        if "error" in row:
            rows.append({"method": method, "error": row["error"]})
            continue
        rows.append({k: v for k, v in row.items()})
    return rows
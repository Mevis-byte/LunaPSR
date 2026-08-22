"""LunaPSR Adaptive Retinex -- noise-aware Retinex enhancement (Phase 5B).

The first LunaPSR algorithm prototype built on a Retinex foundation.  Standard
Single-Scale Retinex (``core.enhancement.retinex``, unchanged, still available
as a baseline) divides the image by its low-frequency illumination in log
space, which makes weak dark-region texture *visible* -- but Phase 5A showed
that inside a noise-dominated five-DN region that "revealed structure" is
mostly amplified read/quantization noise (31 % new edges, 391 isolated
components).  LunaPSR therefore gates the Retinex strength with the Phase 4B
physics noise model and only enhances where the data supports it.

Pipeline (all float32, all derived representations)
---------------------------------------------------
    OHRC DN  ->  local signal estimation (median / illumination / structure /
                 residual; core.enhancement.signal_estimation)
             ->  modeled noise sigma at the LOCAL DN (Phase 4B model;
                 core.analysis.ohrc_noise_model)
             ->  model-based confidence (core.enhancement.confidence)
             ->  spatially varying Retinex strength alpha(x,y) in [0,1]
             ->  controlled blend + noise-aware detail control
             ->  bounded enhanced representation + artifact diagnostics

Mathematical formulation
------------------------
Retinex response (identical formulation to the SSR baseline):

    R(x,y) = log(I(x,y) + eps) - log(G_sigma(I)(x,y) + eps)

with ``I`` the normalized [0,1] image and ``G_sigma`` a Gaussian illumination
estimate.  ``R`` is remapped to display space ``Rd in [0,1]`` by the same
robust global percentile re-map the SSR baseline uses (comparability).

Model-based confidence (two factors, both required):

    C_level(x,y)  = S / (S + k*N)          signal-level evidence
    C_struct(x,y) = |M| / (|M| + N)        variation evidence

where ``S`` is the robust local signal (DN), ``M`` the medium-scale structure
component (DN), ``N`` the modeled total noise sigma at ``S`` (DN), and ``k``
the confidence transition parameter.  The Phase 5A lesson is encoded here: a
DN-3 pixel has C_level ~ 0.86 ("confident") while its window-level variation
is fully explained by noise (measured/modeled ratio 0.90); only evidence that
medium-scale variation EXCEEDS the modeled noise justifies strong enhancement.

Low-confidence protection gate (continuous, no thresholds):

    C_eff = (1 - g) + g * C_struct          g = low_signal_protection
    C     = C_level * C_eff                 composite confidence in [0,1)

Adaptive strength and output:

    alpha(x,y) = alpha_min + (alpha_max - alpha_min) * smoothstep(C)
    base       = (1 - alpha) * I + alpha * Rd
    out        = clip(base + detail_strength * C * (I - G_d(I)), 0, 1)

``smoothstep(t) = t*t*(3 - 2t)`` keeps the transition continuous with zero
slope at both ends.  The detail term preserves/boosts high-frequency content
ONLY where confidence is high; where confidence is low it vanishes, so no new
noise texture is injected.

Scientific status
-----------------
LunaPSR is an ENHANCEMENT algorithm producing a DERIVED REPRESENTATION.
It does not recover unmeasured photons, does not reconstruct true lunar
surface reflectance, does not prove visible structures are geological, and
does not create new physical measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
from core.enhancement.confidence import (
    local_model_confidence,
    sigma_map_dn,
)
from core.enhancement.signal_estimation import (
    SignalEstimationParams,
    estimate_local_signal,
)

__all__ = [
    "LunaPSRRetinexConfig",
    "LunaPSRRetinexResult",
    "enhance_lunapsr_retinex",
    "smoothstep",
    "required_halo",
]

_ALGORITHM = "LunaPSR Adaptive Retinex"
_VERSION = "0.1"

# Internal fixed constants (documented; not exposed as tuning knobs to keep
# the parameter set small).
_DETAIL_SIGMA_PX = 1.5      # Gaussian sigma of the detail-control low-pass
_SMOOTH_EPS = 1e-12         # guards divisions in the evidence ratios


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Continuous ``t*t*(3-2t)`` on [0,1]; zero slope at both ends."""
    x = np.clip(np.nan_to_num(np.asarray(t, dtype=np.float32), nan=0.0), 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


@dataclass(frozen=True)
class LunaPSRRetinexConfig:
    """Configuration for the LunaPSR Adaptive Retinex prototype.

    Every parameter lists name, default, valid range, description and
    scientific rationale.  There are no undocumented magic constants.

    sigma
        default 30.0, range (0, 200] px.
        Gaussian sigma of the illumination estimate G_sigma, identical role
        to the SSR baseline so results stay comparable.  Rationale: must be
        larger than the structures of interest but smaller than the
        illumination gradient scale (Phase 5A used 30).
    epsilon
        default 1e-3, range (0, 0.1].
        Log-stability floor applied to I and G_sigma(I) before the logarithm,
        identical to the SSR baseline floor.  Rationale: log diverges at 0;
        the floor bounds the response in the darkest pixels.
    alpha_min
        default 0.15, range [0, 1].
        Retinex strength in zero-confidence regions.  Rationale: Phase 5A
        showed noise-dominated regions must stay conservative; a small
        non-zero floor keeps the output continuous without inventing
        structure.
    alpha_max
        default 0.85, range [0, 1], >= alpha_min.
        Retinex strength at full confidence.  Rationale: even high-confidence
        blending keeps part of the original measurement visible.
    confidence_transition
        default 1.0, range (0, 10].
        Transition parameter k in C_level = S/(S + k*N).  k=1 gives the
        Phase 5 formulation snr/(1+snr); larger k makes confidence more
        conservative.  Rationale: single smooth knob for the level-evidence
        transition; no hard thresholds.
    detail_strength
        default 0.5, range [0, 2].
        Gain of the noise-aware detail term ``detail_strength * C * (I -
        G_d(I))``.  0 disables detail control.  Rationale: selective
        preservation of supported high-frequency content; suppressed exactly
        where confidence is low (Phase 5A protection requirement).
    low_signal_protection
        default 0.7, range [0, 1].
        Weight g of the variation-evidence gate C_eff = (1-g) + g*C_struct.
        0 ignores medium-scale evidence (level-only confidence); 1 requires
        structure above the modeled noise for strong enhancement.  Rationale:
        direct implementation of the Phase 5A finding that per-pixel level
        alone over-trusts noise-dominated dark regions.
    """

    sigma: float = 30.0
    epsilon: float = 1e-3
    alpha_min: float = 0.15
    alpha_max: float = 0.85
    confidence_transition: float = 1.0
    detail_strength: float = 0.5
    low_signal_protection: float = 0.7
    version: str = _VERSION

    def __post_init__(self) -> None:
        if not (0.0 < self.sigma <= 200.0):
            raise ValueError(f"sigma must be in (0, 200], got {self.sigma}")
        if not (0.0 < self.epsilon <= 0.1):
            raise ValueError(f"epsilon must be in (0, 0.1], got {self.epsilon}")
        if not (0.0 <= self.alpha_min <= 1.0):
            raise ValueError(f"alpha_min must be in [0, 1], got {self.alpha_min}")
        if not (0.0 <= self.alpha_max <= 1.0):
            raise ValueError(f"alpha_max must be in [0, 1], got {self.alpha_max}")
        if self.alpha_max < self.alpha_min:
            raise ValueError("alpha_max must be >= alpha_min")
        if not (0.0 < self.confidence_transition <= 10.0):
            raise ValueError(
                f"confidence_transition must be in (0, 10], got {self.confidence_transition}"
            )
        if not (0.0 <= self.detail_strength <= 2.0):
            raise ValueError(
                f"detail_strength must be in [0, 2], got {self.detail_strength}"
            )
        if not (0.0 <= self.low_signal_protection <= 1.0):
            raise ValueError(
                f"low_signal_protection must be in [0, 1], got {self.low_signal_protection}"
            )

    def to_dict(self) -> dict:
        return {
            "algorithm": _ALGORITHM,
            "version": self.version,
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "confidence_transition": self.confidence_transition,
            "detail_strength": self.detail_strength,
            "low_signal_protection": self.low_signal_protection,
            "detail_sigma_px": _DETAIL_SIGMA_PX,
            "formulation": (
                "R=log(I+eps)-log(G_sigma(I)+eps); Rd=robust[0,1] remap(R); "
                "C_level=S/(S+k*N); C_struct=|M|/(|M|+N); "
                "C=C_level*((1-g)+g*C_struct); "
                "alpha=amin+(amax-amin)*smoothstep(C); "
                "out=clip((1-alpha)*I+alpha*Rd+detail*C*(I-G_d(I)),0,1)"
            ),
        }


@dataclass
class LunaPSRRetinexResult:
    """Enhanced output plus every derived diagnostic map.

    All arrays are float32 and same-shaped as the input.  Invalid pixels are
    NaN everywhere except ``valid``.  None of these products replaces the
    original scientific data.
    """

    enhanced: np.ndarray
    retinex_response: np.ndarray      # raw log-ratio R (diagnostic)
    retinex_display: np.ndarray       # Rd, robust [0,1] remap of R
    illumination: np.ndarray          # G_sigma(I) on normalized input
    confidence: np.ndarray            # composite C in [0,1)
    alpha_map: np.ndarray             # adaptive strength in [0,1]
    noise_sigma_dn: np.ndarray        # modeled N(S) in DN
    structure_dn: np.ndarray          # medium-scale component M in DN
    diagnostics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "parameters": self.parameters,
            "metadata": self.metadata,
            "diagnostics": self.diagnostics,
            "artifacts": self.artifacts,
            "warnings": list(self.warnings),
        }


def required_halo(config: Optional[LunaPSRRetinexConfig] = None) -> int:
    """Padding (px) a tiled caller must add around each strip.

    Covers the illumination Gaussian (4*sigma), the signal-estimation median
    window and the detail low-pass, so tiled full-resolution processing is
    numerically equivalent to whole-image processing away from true borders.
    """
    cfg = config or LunaPSRRetinexConfig()
    return int(4 * cfg.sigma + 8)


# ---------------------------------------------------------------------------
# internal building blocks (also used by the ablation runner)
# ---------------------------------------------------------------------------


def _retinex_response(image_norm: np.ndarray, valid: np.ndarray,
                      config: LunaPSRRetinexConfig) -> tuple:
    """Raw Retinex response R and illumination map (normalized domain)."""
    cfg = config
    filled = np.where(valid, image_norm, 0.0).astype(np.float32)
    illum = gaussian_filter(filled, sigma=float(cfg.sigma), mode="nearest")
    illum = np.clip(illum, cfg.epsilon, 1.0).astype(np.float32)
    img = np.clip(filled, cfg.epsilon, 1.0)
    resp = (np.log(img) - np.log(illum)).astype(np.float32)
    return resp, illum


def _retinex_display(
    resp: np.ndarray,
    valid: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Robust global [0,1] re-map of the response (same style as SSR).

    ``bounds`` overrides the per-call percentiles; tiled full-resolution
    processing passes globally-determined bounds so every tile shares one
    consistent tone mapping.
    """
    from core.enhancement import EPSILON as CORE_EPS

    if bounds is None:
        vals = resp[valid]
        lo, hi = np.percentile(vals, [1, 99])
        if hi - lo < CORE_EPS:
            lo, hi = float(vals.min()), float(vals.max()) + CORE_EPS
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
    out = (resp - lo) / max(hi - lo, CORE_EPS)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _confidence_fields(
    components,
    config: LunaPSRRetinexConfig,
    noise_model: OHRCNoiseModelConfig,
) -> Dict[str, np.ndarray]:
    """Level confidence, variation evidence, composite C and alpha map."""
    cfg = config
    valid = components.valid
    sig_dn = np.where(valid, components.signal, 0.0).astype(np.float32)
    struct = np.where(valid, components.structure, 0.0).astype(np.float32)

    n_map = sigma_map_dn(sig_dn, noise_model)
    c_level = local_model_confidence(sig_dn, noise_model)
    c_level = np.clip(np.nan_to_num(c_level, nan=0.0), 0.0, 1.0)

    # variation evidence: does medium-scale structure exceed modeled noise?
    m_abs = np.abs(struct)
    c_struct = m_abs / (m_abs + n_map + _SMOOTH_EPS)
    g = float(cfg.low_signal_protection)
    c_eff = (1.0 - g) + g * c_struct
    c_comp = np.clip(c_level * c_eff, 0.0, 1.0).astype(np.float32)

    span = cfg.alpha_max - cfg.alpha_min
    alpha = cfg.alpha_min + span * smoothstep(c_comp)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return {"c_level": c_level, "c_struct": c_struct, "c": c_comp,
            "alpha": alpha, "noise_sigma_dn": n_map}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def enhance_lunapsr_retinex(
    scientific_dn: np.ndarray,
    normalized_image: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    noise_model: Optional[OHRCNoiseModelConfig] = None,
    config: Optional[LunaPSRRetinexConfig] = None,
    compute_diagnostics: bool = True,
    display_bounds: Optional[Tuple[float, float]] = None,
) -> LunaPSRRetinexResult:
    """Noise-aware adaptive Retinex (the LunaPSR prototype).

    Parameters
    ----------
    scientific_dn
        ORIGINAL scientific DN array (read-only; never modified).  Used for
        local signal estimation and the Phase 4B noise model.  Normalized
        values are never fed to the noise model.
    normalized_image
        The same scene normalized to [0,1] (percentile normalization), the
        domain Retinex operates on.
    valid_mask
        Optional boolean mask of valid pixels; NaN/Inf are always invalid.
    noise_model
        Phase 4B configuration (defaults to ``OHRCNoiseModelConfig.default()``).
    config
        ``LunaPSRRetinexConfig`` (defaults for the standard prototype).
    compute_diagnostics
        Set False inside tiled full-resolution processing; maps and the
        enhanced image are unaffected, heavy research diagnostics are skipped.
    display_bounds
        Optional ``(lo, hi)`` response remap bounds.  Tiled full-resolution
        processing passes globally-determined bounds so all tiles share one
        consistent tone mapping (defaults to per-call 1/99 percentiles).

    Returns
    -------
    LunaPSRRetinexResult with the enhanced derived representation, all
    diagnostic maps and artifact warnings.  The source arrays are untouched.
    """
    cfg = config or LunaPSRRetinexConfig()
    nm = noise_model or OHRCNoiseModelConfig.default()

    dn = np.asarray(scientific_dn)
    img = np.asarray(normalized_image, dtype=np.float32)
    if dn.shape != img.shape or dn.ndim != 2:
        raise ValueError(
            f"shape mismatch: scientific {dn.shape} vs normalized {img.shape}"
        )

    valid = np.isfinite(dn) & np.isfinite(img)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        if m.shape != dn.shape:
            raise ValueError("valid_mask shape does not match input shape")
        valid &= m
    if not valid.any():
        raise ValueError("No valid pixels for LunaPSR Adaptive Retinex.")

    # 1. Retinex foundation (existing SSR formulation, unchanged mathematics)
    resp, illum = _retinex_response(img, valid, cfg)
    rd = _retinex_display(resp, valid, bounds=display_bounds)

    # 2./3./4. local signal estimation + modeled noise + confidence (DN domain)
    params = SignalEstimationParams()
    components = estimate_local_signal(dn, params=params, mask=valid)
    fields = _confidence_fields(components, cfg, nm)
    alpha = fields["alpha"]
    conf = fields["c"]

    # 5. controlled blend between original and Retinex
    img_f = np.where(valid, img, 0.0).astype(np.float32)
    base = (1.0 - alpha) * img_f + alpha * rd

    # 6. noise-aware detail control (vanishes where confidence is low)
    if cfg.detail_strength > 0.0:
        detail = img_f - gaussian_filter(img_f, sigma=_DETAIL_SIGMA_PX, mode="nearest")
        enhanced = base + float(cfg.detail_strength) * conf * detail
    else:
        enhanced = base
    enhanced = np.clip(enhanced, 0.0, 1.0).astype(np.float32)
    enhanced = np.where(valid, enhanced, np.nan).astype(np.float32)

    result = LunaPSRRetinexResult(
        enhanced=enhanced,
        retinex_response=np.where(valid, resp, np.nan).astype(np.float32),
        retinex_display=np.where(valid, rd, np.nan).astype(np.float32),
        illumination=np.where(valid, illum, np.nan).astype(np.float32),
        confidence=np.where(valid, conf, np.nan).astype(np.float32),
        alpha_map=np.where(valid, alpha, np.nan).astype(np.float32),
        noise_sigma_dn=np.where(valid, fields["noise_sigma_dn"], np.nan).astype(np.float32),
        structure_dn=np.where(valid, components.structure, np.nan).astype(np.float32),
        parameters=cfg.to_dict(),
        metadata={
            "algorithm": _ALGORITHM,
            "version": cfg.version,
            "status": "model_based",
            "signal_estimation": components.to_dict(),
            "caveat": ("Derived enhanced representation; no recovery of "
                       "unmeasured photons or new physical measurements."),
        },
    )

    if compute_diagnostics:
        _attach_diagnostics(result, img, enhanced, valid, fields, cfg, nm)
    return result


# ---------------------------------------------------------------------------
# diagnostics / artifact detection
# ---------------------------------------------------------------------------

# Heuristic warning thresholds (research flags, not verdicts).
_WARN_NEW_EDGE = 0.25
_WARN_ISOLATED = 100
_WARN_AMP = 1.5
_WARN_HF_RATIO = 1.5


def _attach_diagnostics(result, img, enh, valid, fields, cfg, nm) -> None:
    from core.analysis.ssr_investigation import (
        feature_consistency,
        hf_energy,
        ssr_metrics,
    )

    metrics = ssr_metrics(img, enh, valid=valid)
    feat = feature_consistency(img, enh)
    hf_before = hf_energy(img, valid=valid)
    hf_after = hf_energy(enh, valid=valid)
    hf_ratio = float(hf_after / max(hf_before, 1e-12))

    result.diagnostics = {
        "metrics": metrics,
        "feature_consistency": feat,
        "hf_energy_ratio": round(hf_ratio, 4),
        "confidence_mean": round(float(np.nanmean(fields["c"])), 4),
        "confidence_median": round(float(np.nanmedian(fields["c"])), 4),
        "alpha_mean": round(float(np.nanmean(fields["alpha"])), 4),
        "alpha_min_observed": round(float(np.nanmin(fields["alpha"])), 4),
        "alpha_max_observed": round(float(np.nanmax(fields["alpha"])), 4),
        "noise_sigma_dn_median": round(
            float(np.nanmedian(fields["noise_sigma_dn"])), 4),
    }
    result.artifacts = {
        "clipping_fraction_after": metrics["ssr"]["clipping_fraction"],
        "new_edge_fraction": feat["new_edge_fraction_of_ssr_edges"],
        "isolated_new_components": feat["isolated_new_components"],
        "halo_like_profile": feat["halo_like_profile"],
        "noise_amplification_ratio": metrics["noise_amplification_ratio"],
        "hf_energy_ratio": round(hf_ratio, 4),
    }

    warnings: list = []
    clip_before = metrics["original"]["clipping_fraction"]
    clip_after = metrics["ssr"]["clipping_fraction"]
    if clip_after > clip_before + 0.01:
        warnings.append(
            f"Potential enhancement artifact detected: new clipping introduced "
            f"({clip_before:.3f} -> {clip_after:.3f})."
        )
    if feat["new_edge_fraction_of_ssr_edges"] > _WARN_NEW_EDGE:
        warnings.append(
            f"Potential enhancement artifact detected: new-edge fraction "
            f"{feat['new_edge_fraction_of_ssr_edges']:.3f} exceeds {_WARN_NEW_EDGE}."
        )
    if feat["isolated_new_components"] > _WARN_ISOLATED:
        warnings.append(
            f"Potential enhancement artifact detected: "
            f"{feat['isolated_new_components']} isolated new components."
        )
    if metrics["noise_amplification_ratio"] > _WARN_AMP:
        warnings.append(
            f"Potential enhancement artifact detected: noise amplification "
            f"ratio {metrics['noise_amplification_ratio']:.2f}."
        )
    if hf_ratio > _WARN_HF_RATIO:
        warnings.append(
            f"Potential enhancement artifact detected: high-frequency energy "
            f"ratio {hf_ratio:.2f}."
        )
    if feat["halo_like_profile"]:
        warnings.append(
            "Potential enhancement artifact detected: halo-like difference "
            "profile around original edges."
        )
    result.warnings = warnings
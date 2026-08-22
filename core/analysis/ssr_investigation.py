"""Phase 5A -- SSR (Single-Scale Retinex) scientific investigation utilities.

Pure, testable analysis functions used to determine whether the structure
SSR reveals inside a dark OHRC region is

    A. spatial signal already present in the original DN data,
    B. amplified sensor/image noise, or
    C. a mixture of both.

All functions are read-only: they never modify the input arrays and never
touch the product tree.  Every map/table produced here is a DERIVED
representation.

Scientific language rules (project-wide):
- "model-based SNR" (never "measured SNR": the gain ~104 e-/DN is INFERRED).
- Windows whose measured variation exceeds the Phase 4B model prediction are
  labelled "spatial variation above the current noise model" -- NOT
  "geological signal" (that would require independent verification).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label,
    sobel,
    uniform_filter,
)

from core.analysis.ohrc_noise_model import (
    OHRCNoiseModelConfig,
    model_snr,
    noise_breakdown,
)
from core.enhancement.confidence import local_model_confidence, sigma_map_dn

__all__ = [
    "dn_statistics",
    "model_noise_analysis",
    "window_analysis_table",
    "structure_maps",
    "gradient_energy",
    "hf_energy",
    "local_variance_map",
    "ssr_metrics",
    "feature_consistency",
]

# A window is flagged when its measured variation exceeds the modeled total
# sigma by this factor.  The flag wording is deliberately cautious.
ABOVE_MODEL_FACTOR = 1.5


# ---------------------------------------------------------------------------
# 2. Raw DN statistics
# ---------------------------------------------------------------------------


def dn_statistics(dn: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Descriptive statistics of the ORIGINAL scientific DN values.

    ``mask`` (optional) selects valid pixels.  The input array is never
    modified.
    """
    a = np.asarray(dn, dtype=np.float64)
    if mask is not None:
        a = a[np.asarray(mask, dtype=bool)]
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("no valid pixels for dn_statistics")
    pct = {p: float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    return {
        "n_pixels": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "std": float(a.std()),
        "percentiles": pct,
        "zero_fraction": float((a == 0).mean()),
        "saturation_fraction": float((a >= 255).mean()),
    }


# ---------------------------------------------------------------------------
# 3. Physics-aware noise analysis (Phase 4B model, on ORIGINAL DN)
# ---------------------------------------------------------------------------


def model_noise_analysis(
    dn: np.ndarray,
    config: Optional[OHRCNoiseModelConfig] = None,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Phase 4B model evaluation at the ROI mean DN (original DN, not normalized).

    Returns signal electrons, per-component sigmas (DN), total modeled sigma
    and the MODEL-BASED SNR.  The gain (~104 e-/DN) is INFERRED; every number
    here is model-based, not measured.
    """
    cfg = config or OHRCNoiseModelConfig.default()
    a = np.asarray(dn, dtype=np.float64)
    if mask is not None:
        a = a[np.asarray(mask, dtype=bool)]
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("no valid pixels for model_noise_analysis")
    rep = noise_breakdown(float(a.mean()), cfg)
    d = rep.to_dict()
    d["mean_dn"] = d.pop("dn")
    return d


# ---------------------------------------------------------------------------
# 4. Windowed structure-vs-noise table
# ---------------------------------------------------------------------------


def _grad_magnitude(a: np.ndarray) -> np.ndarray:
    a32 = np.asarray(a, dtype=np.float32)
    gx = sobel(a32, axis=1, mode="nearest") / 8.0
    gy = sobel(a32, axis=0, mode="nearest") / 8.0
    return np.hypot(gx, gy)


def gradient_energy(arr: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """Mean squared gradient magnitude (Sobel, [array units]^2)."""
    g = _grad_magnitude(arr)
    if valid is not None:
        g = g[np.asarray(valid, dtype=bool)]
    else:
        g = g[np.isfinite(g)]
    if g.size == 0:
        return float("nan")
    return float(np.mean(g.astype(np.float64) ** 2))


def hf_energy(arr: np.ndarray, sigma: float = 2.0,
              valid: Optional[np.ndarray] = None) -> float:
    """High-frequency energy: mean((x - gaussian(x))**2) in [array units]^2."""
    a = np.asarray(arr, dtype=np.float32)
    filled = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    low = gaussian_filter(filled, sigma=sigma, mode="nearest")
    hf2 = (filled - low).astype(np.float64) ** 2
    sel = valid if valid is not None else np.isfinite(a)
    sel = np.asarray(sel, dtype=bool)
    if not sel.any():
        return float("nan")
    return float(hf2[sel].mean())


def local_variance_map(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Per-pixel local variance map (uniform window; derived representation)."""
    a = np.asarray(arr, dtype=np.float32)
    filled = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    mu = uniform_filter(filled, size=window, mode="nearest")
    mu2 = uniform_filter(filled * filled, size=window, mode="nearest")
    var = np.clip(mu2 - mu * mu, 0.0, None)
    return var.astype(np.float32)


def window_analysis_table(
    dn: np.ndarray,
    config: Optional[OHRCNoiseModelConfig] = None,
    window: int = 20,
) -> List[dict]:
    """Non-overlapping window table: measured variation vs Phase 4B model.

    Per window: mean DN, measured std, modeled total sigma, ratio,
    model-based SNR, gradient energy, edge density, local variance and a
    cautious category label:

    - ``consistent_with_modeled_noise``      when ratio < 1.5
    - ``spatial_variation_above_the_current_noise_model`` otherwise

    The second category is NOT interpreted as geology.
    """
    cfg = config or OHRCNoiseModelConfig.default()
    a = np.asarray(dn, dtype=np.float32)
    h, w = a.shape
    rows: List[dict] = []
    for r0 in range(0, h - window + 1, window):
        for c0 in range(0, w - window + 1, window):
            win = a[r0:r0 + window, c0:c0 + window]
            vals = win[np.isfinite(win)].astype(np.float64)
            if vals.size < max(10, window * window // 4):
                continue
            mean_dn = float(vals.mean())
            measured_sigma = float(vals.std())
            model_sigma = float(noise_breakdown(mean_dn, cfg).total_sigma_dn)
            ratio = measured_sigma / max(model_sigma, 1e-9)
            g = _grad_magnitude(win)
            gthr = float(np.percentile(g, 95))
            rows.append({
                "row0": int(r0),
                "col0": int(c0),
                "mean_dn": round(mean_dn, 4),
                "measured_sigma": round(measured_sigma, 4),
                "model_sigma": round(model_sigma, 4),
                "ratio": round(ratio, 3),
                "model_based_snr": round(float(model_snr(mean_dn, cfg)), 3),
                "gradient_energy": round(float(np.mean(g.astype(np.float64) ** 2)), 6),
                "edge_density": round(float((g >= gthr).mean()), 4),
                "local_variance": round(float(vals.var()), 6),
                "category": (
                    "consistent_with_modeled_noise"
                    if ratio < ABOVE_MODEL_FACTOR
                    else "spatial_variation_above_the_current_noise_model"
                ),
            })
    return rows


# ---------------------------------------------------------------------------
# 5. Structure maps (derived representations of the ORIGINAL DN)
# ---------------------------------------------------------------------------


def structure_maps(
    dn: np.ndarray,
    config: Optional[OHRCNoiseModelConfig] = None,
    window: int = 5,
) -> Dict[str, np.ndarray]:
    """Derived maps for the original DN ROI (inputs never modified).

    Keys: ``gradient_magnitude``, ``local_variance``, ``local_mean``,
    ``model_noise_sigma_dn`` (Phase 4B sigma at each pixel's DN),
    ``model_confidence`` (snr/(1+snr)).
    """
    cfg = config or OHRCNoiseModelConfig.default()
    a = np.asarray(dn, dtype=np.float32)
    finite = np.isfinite(a)
    filled = np.where(finite, a, 0.0).astype(np.float32)
    mu = uniform_filter(filled, size=window, mode="nearest")
    return {
        "gradient_magnitude": _grad_magnitude(filled),
        "local_variance": local_variance_map(filled, window),
        "local_mean": mu.astype(np.float32),
        "model_noise_sigma_dn": sigma_map_dn(filled, cfg),
        "model_confidence": local_model_confidence(filled, cfg),
    }


# ---------------------------------------------------------------------------
# 7. SSR metrics
# ---------------------------------------------------------------------------

_SSR_METRIC_KEYS = (
    "gradient_energy",
    "edge_density",
    "local_variance",
    "hf_energy",
    "clipping_fraction",
)


def _single_metrics(arr: np.ndarray, valid: Optional[np.ndarray]) -> dict:
    a = np.asarray(arr, dtype=np.float32)
    sel = valid if valid is not None else np.isfinite(a)
    sel = np.asarray(sel, dtype=bool)
    v = a[sel].astype(np.float64)
    g = _grad_magnitude(a)
    gthr = float(np.percentile(g[sel], 95)) if sel.any() else 0.0
    lv = local_variance_map(a)[sel]
    return {
        "gradient_energy": gradient_energy(a, sel),
        "edge_density": float((g[sel] >= gthr).mean()) if sel.any() else float("nan"),
        "local_variance": float(lv.mean()) if lv.size else float("nan"),
        "hf_energy": hf_energy(a, valid=sel),
        "clipping_fraction": float(((v <= 0.0) | (v >= 1.0)).mean()) if v.size else float("nan"),
    }


def ssr_metrics(
    orig_norm: np.ndarray,
    enhanced_norm: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> dict:
    """Original vs SSR metrics on the SAME [0,1]-normalized ROI.

    ``noise_amplification_ratio`` compares robust noise-sigma estimates
    after/before (proxy; an increase is not automatically bad).
    """
    from core.metrics.noise import estimate_noise_sigma

    m_o = _single_metrics(orig_norm, valid)
    m_e = _single_metrics(enhanced_norm, valid)
    o = np.asarray(orig_norm, dtype=np.float32)
    e = np.asarray(enhanced_norm, dtype=np.float32)
    sel = valid if valid is not None else np.isfinite(o)
    sel = np.asarray(sel, dtype=bool)
    # keep the 2-D shape: the Immerkaer kernel is spatial -- a flattened
    # array would mix unrelated rows and corrupt the estimate
    sb = max(estimate_noise_sigma(np.where(sel, o, 0.0)), 1e-9)
    sa = max(estimate_noise_sigma(np.where(sel, e, 0.0)), 1e-9)
    return {
        "original": m_o,
        "ssr": m_e,
        "noise_amplification_ratio": float(sa / sb),
        "note": ("Increased high-frequency energy is not automatically useful "
                 "structure; see feature_consistency."),
    }


# ---------------------------------------------------------------------------
# 8. Feature consistency test
# ---------------------------------------------------------------------------


def feature_consistency(
    orig: np.ndarray,
    enhanced: np.ndarray,
    edge_percentile: float = 95.0,
) -> dict:
    """Compare spatial features before vs after SSR ([0,1] arrays).

    Answers, quantitatively:
    - which fraction of ORIGINAL edges became stronger,
    - which fraction of SSR edges appear only after SSR (new),
    - how many NEW isolated small components appear,
    - whether the change is edge-localized (halo-like profile),
    - gradient-orientation agreement where both have gradients.

    All outputs are research diagnostics, not verdicts.
    """
    o = np.asarray(orig, dtype=np.float32)
    e = np.asarray(enhanced, dtype=np.float32)
    mag_o = _grad_magnitude(o)
    mag_e = _grad_magnitude(e)

    # percentile threshold with an absolute floor: on near-binary images the
    # raw percentile can collapse to ~0 and mark every pixel as an edge
    thr_o = max(float(np.percentile(mag_o, edge_percentile)),
                1e-6 * float(mag_o.max()))
    thr_e = max(float(np.percentile(mag_e, edge_percentile)),
                1e-6 * float(mag_e.max()))
    edge_o = mag_o >= thr_o
    edge_e = mag_e >= thr_e

    n_edge_o = int(edge_o.sum())
    n_edge_e = int(edge_e.sum())

    strengthened = edge_o & (mag_e > mag_o)
    dilated_o = binary_dilation(edge_o, structure=np.ones((3, 3), dtype=bool))
    new_edges = edge_e & ~dilated_o

    # isolated new structures: connected components of new edges with <=4 px
    lab, n_comp = label(new_edges)
    iso_count = 0
    iso_px = 0
    if n_comp:
        sizes = np.bincount(lab.ravel())[1:]
        small = sizes <= 4
        iso_count = int(small.sum())
        iso_px = int(sizes[small].sum())

    # gradient orientation agreement (circular, doubled angle), weighted by
    # the weaker of the two magnitudes, evaluated where both exceed threshold
    both = edge_o & edge_e
    if both.any():
        gy_o, gx_o = np.gradient(o.astype(np.float64))
        gy_e, gx_e = np.gradient(e.astype(np.float64))
        th_o = np.arctan2(gy_o, gx_o)
        th_e = np.arctan2(gy_e, gx_e)
        wgt = np.minimum(mag_o, mag_e)[both].astype(np.float64)
        dth = np.abs(th_e[both] - th_o[both])
        orient = float((np.cos(2.0 * dth) * wgt).sum() / max(wgt.sum(), 1e-12))
    else:
        orient = float("nan")

    # halo-like profile: mean |enhanced-original| as a function of distance
    # from ORIGINAL edges (bands 1..5+ px).  A ring at distance 2-4 higher
    # than the adjacent band suggests halo/ringing rather than pure
    # edge-localized amplification.
    dist = distance_transform_edt(~edge_o)
    diff = np.abs(e.astype(np.float64) - o.astype(np.float64))
    profile = {}
    for d in range(1, 6):
        band = (dist >= d) & (dist < d + 1)
        profile[str(d)] = round(float(diff[band].mean()), 6) if band.any() else None
    far = dist >= 6
    profile["6+"] = round(float(diff[far].mean()), 6) if far.any() else None
    b1, b3, bfar = profile.get("1"), profile.get("3"), profile.get("6+")
    # classic halo/ringing: an elevated ring at distance 2-4 that decays with
    # distance AND exceeds the change right at the edge
    mid = [profile.get(str(d)) for d in (2, 3, 4)]
    halo_like = bool(
        b1 is not None and bfar is not None and all(v is not None for v in mid)
        and max(mid) > 1.25 * bfar and max(mid) > b1
    )
    # change concentrated at original edges vs a global re-map
    edge_localized = bool(b1 is not None and b3 is not None and b1 > b3)

    from core.metrics.core import edge_preservation_index

    return {
        "n_edges_original": n_edge_o,
        "n_edges_ssr": n_edge_e,
        "strengthened_fraction_of_original_edges":
            round(float(strengthened.sum()) / max(n_edge_o, 1), 4),
        "new_edge_fraction_of_ssr_edges":
            round(float(new_edges.sum()) / max(n_edge_e, 1), 4),
        "isolated_new_components": iso_count,
        "isolated_new_pixels": iso_px,
        "orientation_agreement_doubled_angle": None if np.isnan(orient) else round(orient, 4),
        "edge_preservation_index": round(float(edge_preservation_index(o, e)), 4),
        "abs_difference_by_distance_from_original_edges": profile,
        "halo_like_profile": halo_like,
        "change_concentrated_at_original_edges": edge_localized,
        "note": ("Research diagnostics only; 'new' edges are relative to the "
                 "original edge threshold and do not imply real features."),
    }
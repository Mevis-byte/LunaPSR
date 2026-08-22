"""Phase 3 - OHRC Enhancement Lab.

A scientifically careful preprocessing + baseline-enhancement benchmark tab.

Pipeline (matching docs/phase3.md):

    real OHRC product
      -> discovery / reader
      -> scientific image        (read-only memory-map, 1.2 Gpx)
      -> analysis representation (block-averaged, size-reduced float32)
      -> valid-pixel mask
      -> robust normalization    (percentile-based, reproducible)
      -> noise analysis
      -> baseline enhancement
      -> side-by-side / difference / before-after
      -> metrics
      -> experiment record + derived outputs

CONTRACT: the original scientific array is never modified.  All processing
operates on derived arrays (the analysis representation, masks, normalised
and enhanced copies).  Derived outputs are written under ``outputs/`` only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st

from configs.defaults import (
    BASELINE_DEFAULTS,
    ENHANCE_PERCENTILES,
    OHRC_ANALYSIS_FACTOR,
    OHRC_DISPLAY_PERCENTILES,
)
from core.analysis.experiment import ExperimentRecord, ExperimentStore
from core.analysis.metrics import quality_report
from core.analysis.noise import analyze_noise
from core.analysis.ohrc_noise_model import (
    OHRCNoiseModelConfig,
    estimate_flat_window_noise,
    noise_breakdown,
)
from core.enhancement.baseline import BASELINE_METHODS, apply_baseline
from core.enhancement.lunapsr import (
    LunaPSRParams,
    benchmark_lunapsr,
    benchmark_table,
    enhance_lunapsr,
)
from core.enhancement.lunapsr_retinex import (
    LunaPSRRetinexConfig,
    enhance_lunapsr_retinex,
)
from core.enhancement.zero_dce import ZeroDCEUnavailableError, is_available as zdce_available
from core.ohrc import OHRCProduct, find_ohrc_products, validate_product
from core.ohrc.statistics import exact_histogram, histogram_statistics
from core.preprocessing.mask import build_valid_mask
from core.preprocessing.normalization import normalize_scientific

logger = logging.getLogger(__name__)

_LAB_KEY = "ohrc_lab"
_LAB_DERIVED = "ohrc_lab_derived"
_LAB_RESULT = "ohrc_lab_result"
_LAB_SCAN = "ohrc_lab_scan"
_LAB_LUNAPSR = "ohrc_lab_lunapsr"
_LAB_LPRT = "ohrc_lab_lunapsr_retinex"
_METHODS = sorted(BASELINE_METHODS)

# Phase 4A report §10.3 dark patch (row0, col0, size) at full resolution,
# used for the measured-vs-model flat-window validation.
_P4A_DARK_PATCH = (61750, 6000, 200)


# ---------------------------------------------------------------------------
# Pure computation (unit-testable, no streamlit)
# ---------------------------------------------------------------------------


def _bounds_from_histogram(counts, edges, lo_pct: float, hi_pct: float):
    from core.ohrc.statistics import percentile_from_histogram

    lo = percentile_from_histogram(counts, edges, float(lo_pct))
    hi = percentile_from_histogram(counts, edges, float(hi_pct))
    if lo is None or hi is None or hi <= lo:
        return None, None
    return lo, hi


def build_analysis(
    product: OHRCProduct,
    factor: int = OHRC_ANALYSIS_FACTOR,
    lower_percentile: float = ENHANCE_PERCENTILES[0],
    upper_percentile: float = ENHANCE_PERCENTILES[1],
    count_hist: Optional[Tuple] = None,
) -> dict:
    """Precompute the analysis representation for one product.

    Returns:
        base                  -- block-averaged scientific float32 (analysis rep)
        mask, mask_details    -- valid-pixel mask + explanation
        normalized            -- [0,1] float32 with invalid pixels = NaN
        norm_params           -- reproducible normalization parameters
        scientific_stats      -- exact statistics over the full scientific data
        bounds                -- (lo_value, hi_value) from the exact histogram
    """
    sci = product.image.scientific_array()

    if count_hist is None:
        counts, edges, _, _ = exact_histogram(sci)
    else:
        counts, edges = count_hist

    lo_val, hi_val = _bounds_from_histogram(counts, edges, lower_percentile, upper_percentile)
    if lo_val is None:  # degenerate range: guard against divide-by-zero below
        lo_val, hi_val = 0.0, 1.0

    base = product.decimate_average(factor=factor)
    mask, mask_details = build_valid_mask(base, metadata=product.metadata)

    normalized, norm_params = normalize_scientific(
        base,
        percentiles=(lower_percentile, upper_percentile),
        mask=mask,
        precomputed_bounds=(lo_val, hi_val),
    )

    sci_stats = histogram_statistics(counts, edges)
    sci_stats["total_pixels"] = int(counts.sum())
    sci_stats["valid_pixels"] = int(counts.sum())
    sci_stats["invalid_pixels"] = 0

    return {
        "base": base,
        "mask": mask,
        "mask_details": mask_details,
        "normalized": normalized,
        "norm_params": norm_params,
        "scientific_stats": sci_stats,
        "bounds": (lo_val, hi_val),
        "histogram": (counts, edges),
    }


def apply_enhancement(normalized: np.ndarray, method: str, params: dict) -> dict:
    """Run one baseline on the [0,1] normalised array.

    Returns the enhanced array, its metadata and the no-reference metrics of
    both original-normalised and enhanced views.  Input is never mutated.
    """
    if method == "zero_dce":
        if not zdce_available():
            raise ZeroDCEUnavailableError(ZeroDCEUnavailableError.MESSAGE)
        from core.enhancement.zero_dce import zero_dce_enhance

        enhanced = zero_dce_enhance(normalized, iterations=int(params.get("iterations", 8)))
    else:
        enhanced = apply_baseline(normalized, method, **params)

    meta = {"method": method, "parameters": dict(params)}
    return {
        "enhanced": enhanced,
        "metadata": meta,
        "metrics_original": quality_report(normalized),
        "metrics_enhanced": quality_report(enhanced),
    }


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _scan_products() -> dict:
    if _LAB_SCAN not in st.session_state:
        scan = find_ohrc_products()
        st.session_state[_LAB_SCAN] = {
            "products": scan.products,
            "default_path": str(scan.default().path) if scan.default() else None,
            "root": str(scan.root),
        }
    return st.session_state[_LAB_SCAN]


def _rescan() -> None:
    st.session_state.pop(_LAB_SCAN, None)
    st.session_state.pop(_LAB_KEY, None)
    st.session_state.pop(_LAB_DERIVED, None)
    st.session_state.pop(_LAB_RESULT, None)
    st.session_state.pop(_LAB_LUNAPSR, None)


def _load_chosen(product_path: str) -> Optional[OHRCProduct]:
    state = st.session_state.get(_LAB_KEY)
    key = str(Path(product_path).resolve())
    if state is not None and state.get("path") == key:
        return state["product"]  # already loaded this session
    try:
        product = OHRCProduct.load(key)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        st.error(f"Could not load OHRC product: {exc}")
        return None
    st.session_state[_LAB_KEY] = {"path": key, "product": product}
    st.session_state.pop(_LAB_DERIVED, None)
    st.session_state.pop(_LAB_RESULT, None)
    return product


def _derived(product: OHRCProduct, factor: int, lo_pct: float, hi_pct: float) -> dict:
    key = (product.path, factor)
    cache = st.session_state.get(_LAB_DERIVED, {})
    entry = cache.get(key)
    if entry is None:
        with st.spinner("Building size-reduced analysis representation (read-only)..."):
            entry = build_analysis(product, factor=factor)
        hist = entry["histogram"]
        entry["_hist_cache"] = (hist[0], hist[1])
        entry["_analysis_factor"] = int(factor)
        cache[key] = entry
        st.session_state[_LAB_DERIVED] = cache
    if (lo_pct, hi_pct) != (ENHANCE_PERCENTILES[0], ENHANCE_PERCENTILES[1]):
        counts, edges = entry["_hist_cache"]
        bounds = _bounds_from_histogram(counts, edges, lo_pct, hi_pct)
        if bounds is not None and bounds != entry["bounds"]:
            base = entry["base"]
            mask = entry["mask"]
            normalized, norm_params = normalize_scientific(
                base, percentiles=(lo_pct, hi_pct), mask=mask,
                precomputed_bounds=bounds,
            )
            entry = {**entry, "normalized": normalized, "norm_params": norm_params,
                     "bounds": bounds}
            st.session_state[_LAB_DERIVED] = cache
    return entry


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_source_selector() -> Optional[OHRCProduct]:
    st.markdown("#### OHRC Data Source")
    st.caption(
        "Products are discovered automatically in the project directory — "
        "no upload required, and no product file is ever modified."
    )
    scan = _scan_products()
    products = scan["products"]

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        available = [p for p in products if p.valid]
        if not available:
            st.info(
                "No valid OHRC product detected in this environment. The "
                "1.2-billion-pixel dataset is intentionally not part of the "
                "repository — place an OHRC product folder under the project "
                "root locally, or add one below."
            )
        else:
            default_name = scan["default_path"]
            default_idx = 0
            for i, p in enumerate(available):
                if str(p.path) == default_name:
                    default_idx = i
                    break
            selected = st.selectbox(
                "Detected OHRC Products",
                available,
                format_func=lambda p: f"{p.name}  ·  VALID",
                index=default_idx,
            )
            st.markdown(
                f"● **{selected.name}** — Status: `VALID` · "
                f"{selected.lines} × {selected.samples} px · {selected.datatype}"
            )
    with c2:
        st.markdown("&nbsp;")
        if available and st.button("Use this OHRC Product", type="primary", key="lab_use"):
            return _load_chosen(str(selected.path))
    with c3:
        st.markdown("&nbsp;")
        if st.button("Rescan", key="lab_rescan"):
            _rescan()
            st.rerun()

    if not available:
        return None

    with st.expander("Additional OHRC product (add later)", expanded=False):
        extra = st.text_input("Product directory path (project-relative or absolute)")
        if st.button("Add & validate", key="lab_add_extra"):
            if extra.strip():
                entry = validate_product(Path(extra.strip()))
                if entry.valid:
                    return _load_chosen(str(entry.path))
                st.error(f"Not a valid OHRC product: {entry.error}")

    return None


def _render_experiment_controls(entry: dict, method: str, params: dict, product, res: dict,
                                normalized) -> None:
    st.divider()
    st.markdown("#### Reproducibility & export")
    st.caption(
        "Enhancement results are derived visualization/analysis products. "
        "The original scientific array is preserved internally and on disk."
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        save = st.button("💾 Save experiment (outputs/)", key="lab_save_experiment")
    with c2:
        from app.exporters import json_bytes

        meta = {
            "product": product.path.name,
            "method": method,
            "parameters": params,
            "normalization": entry["norm_params"],
            "metrics": {
                "original": res["metrics_original"]["metrics"],
                "enhanced": res["metrics_enhanced"]["metrics"],
            },
            "analysis_factor": entry.get("_analysis_factor", OHRC_ANALYSIS_FACTOR),
            "bounds": entry["bounds"],
            "scientific_stats": entry["scientific_stats"]["percentiles"] if "scientific_stats" in entry else {},
            "noise_model": OHRCNoiseModelConfig.default().to_dict(),
        }
        st.download_button(
            "⬇️ Experiment record (JSON)",
            json_bytes(meta),
            file_name=f"{product.path.name}_{method}_records.json",
            mime="application/json",
            key="lab_dl_record",
        )
    with c3:
        st.slider(
            "Derived display opacity (orig ⇄ enhanced)", 0.0, 1.0, 0.5, 0.1,
            key="lab_opacity", help="Before/after wipe: 0 = normalized original, 1 = enhanced.",
        )

    if save:
        store = ExperimentStore()
        rec = ExperimentRecord(
            product=product.path.name,
            method=method,
            parameters=params,
            normalization=entry["norm_params"],
            metrics={
                "original": res["metrics_original"]["metrics"],
                "enhanced": res["metrics_enhanced"]["metrics"],
            },
            analysis={**entry["mask_details"],
                      "bound_lo": entry["bounds"][0], "bound_hi": entry["bounds"][1],
                      "scientific_stats": entry["scientific_stats"] if "scientific_stats" in entry else {}},
            noise_model=OHRCNoiseModelConfig.default().to_dict(),
            notes=st.session_state.get("lab_notes", ""),
        )
        path = store.save_record(rec)
        st.success(f"Experiment recorded → `{path.relative_to(store.root)}`")

    c1, c2, c3, c4 = st.columns(4)
    enhanced = res["enhanced"]
    from app.exporters import (difference_png_bytes, json_bytes as _jb,
                               png_bytes, tiff16_bytes)
    with c1:
        st.download_button(
            "Enhanced PNG (derived 8-bit)", png_bytes(enhanced),
            file_name=f"{product.path.name}_{method}.png", mime="image/png", key="lab_png")
    with c2:
        try:
            st.download_button(
                "Enhanced TIFF (16-bit, derived)", tiff16_bytes(enhanced, {"product": product.path.name, "method": method}),
                file_name=f"{product.path.name}_{method}.tiff", mime="image/tiff", key="lab_tiff")
        except ValueError as exc:
            st.caption(str(exc))
    with c3:
        st.download_button(
            "Difference PNG", difference_png_bytes(np.nan_to_num(normalized, nan=0.0), enhanced),
            file_name=f"{product.path.name}_{method}_difference.png", mime="image/png", key="lab_diff")
    with c4:
        st.download_button(
            "Metrics JSON", _jb(res["metrics_enhanced"]),
            file_name=f"{product.path.name}_{method}_metrics.json", mime="application/json", key="lab_mjson")
    st.caption(
        "8-bit PNGs and 16-bit TIFFs are derived visualizations of the "
        "enhancement result.  Raw scientific values are only available from "
        "the original, unmodified product files."
    )


def _render_quality_tables(res: dict, entry: dict, product) -> None:
    st.divider()
    st.markdown("#### Image Statistics")
    st.caption(
        "No-reference metrics on the derived [0,1] analysis representation. "
        "Metrics describe the derived visualization; they do not certify "
        "newly recovered physical measurements."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original (normalized)**")
    with col2:
        st.markdown("**Enhanced**")
    m_o, m_e = res["metrics_original"]["metrics"], res["metrics_enhanced"]["metrics"]
    rows = sorted(set(m_o.keys()) & set(m_e.keys()))
    data = {"metric": [], "original": [], "enhanced": []}
    for key in rows:
        if isinstance(m_e[key], dict):
            continue
        data["metric"].append(key)
        data["original"].append(f"{m_o[key]:.4f}")
        data["enhanced"].append(f"{m_e[key]:.4f}")
    import pandas as pd

    st.table(pd.DataFrame(data))

    with st.expander("Per-view details"):
        for label, rep in (("Original (normalized)", res["metrics_original"]),
                           ("Enhanced", res["metrics_enhanced"])):
            st.markdown(f"**{label}**")
            st.json(rep)

    if "scientific_stats" in entry:
        with st.expander("Scientific statistics (full product, exact)"):
            p = entry["scientific_stats"]
            cols = st.columns(5)
            vals = [("Min", p["min"]), ("Median", p["median"]), ("Mean", p["mean"]),
                    ("Std", p["std"]), ("Max", p["max"])]
            for col, (k, v) in zip(cols, vals):
                col.metric(k, f"{v:.4g}")
            st.caption("Computed exactly from the full scientific histogram (1.2 Gpx).")


def _render_display(product, entry, res, normalized, shown: float) -> None:
    st.divider()
    st.markdown("#### Result")
    enhanced = res["enhanced"]
    disp_orig = _preview_8(normalized)
    disp_enh = _preview_8(enhanced)
    blended = (disp_orig.astype(np.float32) * (1.0 - shown)
               + disp_enh.astype(np.float32) * shown).astype(np.uint8)
    c1, c2 = st.columns(2)
    with c1:
        st.image(disp_orig, caption="Normalized scientific (derived analysis rep)",
                 width="stretch", channels="GRAY")
    with c2:
        st.image(disp_enh, caption=f"{baseline_label(res['metadata']['method'])} (enhanced, derived)",
                 width="stretch", channels="GRAY")
    st.caption("Both views are derived visualizations; the original scientific data is untouched.")

    st.markdown("#### Comparison / difference")
    diff = np.abs(np.nan_to_num(enhanced, nan=0.0) - np.nan_to_num(normalized, nan=0.0))
    c1, c2 = st.columns(2)
    with c1:
        st.image(_preview_8(blended), caption="Before/after wipe (slider above)",
                 width="stretch", channels="GRAY")
    with c2:
        st.image(_preview_8(diff), caption="Enhancement Difference (|enhanced − normalized|)",
                 width="stretch", channels="GRAY")
    st.caption(
        "The difference view shows what the enhancement changed.  It is NOT "
        "interpreted as newly discovered information."
    )


def _render_roi(entry, product) -> None:
    st.divider()
    st.markdown("#### Region of interest (ROI)")
    st.caption("Rectangular ROI on the analysis representation. Coordinates scale by ×factor to the full product.")
    base = entry["base"]
    h, w = base.shape
    c1, c2, c3, c4 = st.columns(4)
    x0 = c1.number_input("x0 (col)", 0, w - 1, 0, 1, key="lab_roi_x0")
    y0 = c2.number_input("y0 (row)", 0, h - 1, 0, 1, key="lab_roi_y0")
    x1 = c3.number_input("x1 (col)", 0, w, min(w, 200), 1, key="lab_roi_x1")
    y1 = c4.number_input("y1 (row)", 0, h, min(h, 200), 1, key="lab_roi_y1")
    if st.button("Analyze ROI", key="lab_roi_go"):
        x0, x1 = sorted((int(x0), int(x1))); y0, y1 = sorted((int(y0), int(y1)))
        region = base[y0:y1, x0:x1]
        norm = entry["normalized"][y0:y1, x0:x1]
        finite = np.isfinite(region)
        if finite.sum() == 0:
            st.warning("ROI contains no valid pixels.")
            return
        vals = region[finite].astype(np.float32)
        from core.analysis.metrics import local_variance
        from core.metrics.core import robust_contrast
        from core.metrics.noise import estimate_noise_sigma
        stats = {
            "roi_width": x1 - x0,
            "roi_height": y1 - y0,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "local_contrast": float(robust_contrast(norm)),
            "local_noise_estimate": float(estimate_noise_sigma(norm)),
            "local_variance": float(local_variance(norm)),
            "noise_model": noise_breakdown(float(vals.mean()), OHRCNoiseModelConfig.default()).to_dict(),
        }
        st.json(stats)
        c1, c2, c3 = st.columns(3)
        c1.metric("ROI width", stats["roi_width"])
        c2.metric("ROI height", stats["roi_height"])
        c3.metric("ROI mean", f"{stats['mean']:.4g}")
        c1, c2, c3 = st.columns(3)
        c1.metric("ROI median", f"{stats['median']:.4g}")
        c2.metric("ROI std", f"{stats['std']:.4g}")
        c3.metric("ROI local contrast", f"{stats['local_contrast']:.4g}")
        c1, c2, c3 = st.columns(3)
        rep = noise_breakdown(float(vals.mean()), OHRCNoiseModelConfig.default())
        c1.metric("Total modeled noise σ (DN)", f"{rep.total_sigma_dn:.4g}")
        c2.metric("Model-based SNR", f"{rep.model_snr:.3g}")
        c3.metric("Signal (e⁻)", f"{rep.signal_e:.3g}")
        with st.expander("ROI measurements (full)"):
            st.json(stats)


def _flat_window_result(sci: np.ndarray, row0: int, col0: int, size: int,
                        config: OHRCNoiseModelConfig, mean_max: float = 5.0):
    """Flat-window measured-vs-model on a full-resolution patch (read-only)."""
    patch = np.asarray(sci[row0:row0 + size, col0:col0 + size])
    return estimate_flat_window_noise(patch, config=config, mean_max=mean_max)


def _render_noise_snr(entry, product) -> None:
    st.divider()
    st.markdown("#### 🔬 Noise & SNR (physics-aware, model-based)")
    cfg = OHRCNoiseModelConfig.default()

    c1, c2 = st.columns([2, 2])
    with c1:
        st.markdown("**OHRC Product**")
        st.code(product.path.name)
        st.markdown("**Noise Model:** Poisson + Read + Quantization")
        st.markdown("**Gain:** ~104 e⁻/DN  `(INFERRED)`")
        st.markdown("**Read noise:** 40 e⁻")
        st.markdown("**Quantization:** 1/12 LSB²")
    with c2:
        st.markdown("**Model status:** `MODEL-BASED ESTIMATE`")
        st.caption(cfg.gain_note)
        st.caption(cfg.tdi_scope_note)

    # Estimated noise sigma / model SNR at the analysis-representation median DN
    base = entry["base"]
    finite = np.isfinite(base)
    if finite.any():
        med_dn = float(np.nanmedian(base[finite]))
        rep = noise_breakdown(med_dn, cfg)
        c1, c2, c3 = st.columns(3)
        c1.metric("Shot noise", f"{rep.shot_sigma_dn:.4g} DN")
        c2.metric("Read noise", f"{rep.read_sigma_dn:.4g} DN")
        c3.metric("Quantization", f"{rep.quant_sigma_dn:.4g} DN")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total σ (model)", f"{rep.total_sigma_dn:.4g} DN")
        c2.metric("Model-based SNR", f"{rep.model_snr:.3g}")
        c3.metric("Median DN (analysis rep)", f"{med_dn:.3g}")
        c4.metric("Signal (e⁻)", f"{rep.signal_e:.3g}")
        st.warning(
            "**⚠ MODEL-BASED ESTIMATE** — These values are *not* measured "
            "sensor quantities. They are evaluated from the Phase 4B physics "
            "noise model (Poisson ⊕ read ⊕ quantization) at the median DN of "
            "the derived analysis representation."
        )
        st.caption(
            "Noise σ and SNR at the median DN of the derived analysis "
            "representation (block-averaged DN counts)."
        )

    st.markdown("**Measured vs model (flat windows)**")
    st.caption(
        "Low-signal, low-structure windows on the full-resolution product "
        "(Phase 4A §10.5 methodology).  Measured spread is an estimate of the "
        "total random component; it is **not automatically pure sensor noise** "
        "— terrain structure can contribute."
    )
    try:
        sci = product.image.scientific_array()
        r0, c0, sz = _P4A_DARK_PATCH
        fw = _flat_window_result(sci, r0, c0, sz, cfg)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Windows (n)", f"{fw.n_windows}")
        c2.metric("Mean DN", f"{fw.mean_dn:.3g}")
        c3.metric("Measured σ (DN)", f"{fw.measured_sigma_dn:.4g}")
        c4.metric("Model σ (DN)", f"{fw.model_sigma_dn:.4g}")
        c5.metric("Rel. error", f"{fw.relative_error * 100:.2f} %")
        c1, c2, c3 = st.columns(3)
        c1.metric("Difference (meas − model)", f"{fw.residual_dn:.4g} DN")
        c2.metric("Window size", f"{fw.window_size}×{fw.window_size}")
        c3.metric("Stride", f"{fw.stride}")
        st.caption(f"Region: full-res lines {r0}–{r0 + sz - 1}, samples {c0}–{c0 + sz - 1} (Phase 4A dark patch).")
    except (ValueError, IndexError) as exc:
        st.warning(f"Flat-window comparison unavailable for this region: {exc}")

    _render_noise_roi(entry, product, cfg)


def _render_noise_roi(entry, product, cfg: OHRCNoiseModelConfig) -> None:
    st.markdown("**ROI noise / SNR (model-based)**")
    base = entry["base"]
    h, w = base.shape
    c1, c2, c3, c4 = st.columns(4)
    x0 = c1.number_input("x0 (col)", 0, w - 1, 0, 1, key="lab_nr_x0")
    y0 = c2.number_input("y0 (row)", 0, h - 1, 0, 1, key="lab_nr_y0")
    x1 = c3.number_input("x1 (col)", 0, w, min(w, 200), 1, key="lab_nr_x1")
    y1 = c4.number_input("y1 (row)", 0, h, min(h, 200), 1, key="lab_nr_y1")
    if not st.button("Analyze ROI noise/SNR", key="lab_nr_go"):
        return
    x0, x1 = sorted((int(x0), int(x1))); y0, y1 = sorted((int(y0), int(y1)))
    region = base[y0:y1, x0:x1]
    finite = np.isfinite(region)
    if finite.sum() == 0:
        st.warning("ROI contains no valid pixels.")
        return
    mean_dn = float(region[finite].mean())
    rep = noise_breakdown(mean_dn, cfg)
    c1, c2, c3 = st.columns(3)
    c1.metric("ROI mean DN", f"{mean_dn:.3g}")
    c2.metric("Shot noise σ (DN)", f"{rep.shot_sigma_dn:.4g}")
    c3.metric("Read noise σ (DN)", f"{rep.read_sigma_dn:.4g}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Quantization σ (DN)", f"{rep.quant_sigma_dn:.4g}")
    c2.metric("Total modeled noise σ (DN)", f"{rep.total_sigma_dn:.4g}")
    c3.metric("Model-based SNR", f"{rep.model_snr:.3g}")
    with st.expander("ROI noise breakdown (units explicit)"):
        st.json(rep.to_dict())
        st.caption(
            "All figures are MODEL-BASED ESTIMATES (status: "
            f"`{rep.status}`). Signal in electrons uses the INFERRED gain "
            f"({cfg.gain_e_per_dn:g} e⁻/DN); it is not an official calibration."
        )

    # Flat-window measured-vs-model on the full-resolution ROI
    try:
        sci = product.image.scientific_array()
        factor = entry.get("_analysis_factor", 10)
        fw = _flat_window_result(sci, y0 * factor, x0 * factor,
                                 max(1, min(int(x1 - x0), int(y1 - y0))) * factor,
                                 cfg)
        st.markdown("**ROI: measured vs model (flat windows, full resolution)**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Windows (n)", f"{fw.n_windows}")
        c2.metric("Mean DN", f"{fw.mean_dn:.3g}")
        c3.metric("Measured σ (DN)", f"{fw.measured_sigma_dn:.4g}")
        c4.metric("Model σ (DN)", f"{fw.model_sigma_dn:.4g}")
        c5.metric("Rel. error", f"{fw.relative_error * 100:.2f} %")
        st.caption(
            "Measured ≠ automatically pure sensor noise: terrain structure can "
            "contribute to the measured spread."
        )
    except (ValueError, IndexError) as exc:
        st.warning(f"ROI flat-window comparison unavailable: {exc}")


def _heat_map(arr: np.ndarray) -> np.ndarray:
    """Pseudo-colour a [0,1] map for display (derived diagnostic view only)."""
    import cv2

    a = np.clip(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0), 0.0, 1.0)
    u8 = (a * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_LUNAPSR_COMPONENT_LABELS = {
    "tone_only": "A · Tone curve only (constant strength)",
    "tone_adaptive": "B · Tone curve + confidence adaptation",
    "adaptive_contrast": "C · Tone + confidence + local contrast",
    "full": "D · Full LunaPSR (structure-aware)",
}


def _render_lunapsr_params() -> LunaPSRParams:
    st.markdown("**Enhancement parameters** (each documented)")
    c1, c2, c3 = st.columns(3)
    local_window = c1.slider(
        "Local window (px)", 5, 31, 9, 2, key="lab_lp_window",
        help="Robust local signal window. Larger = smoother signal estimate, less "
             "sensitive to noise; smaller = more local detail.")
    illumination_sigma = c2.slider(
        "Illumination sigma (px)", 3.0, 30.0, 9.0, 1.0, key="lab_lp_illum",
        help="Gaussian sigma for the slowly-varying illumination estimate.")
    min_strength = c3.slider(
        "Min strength", 0.0, 0.5, 0.25, 0.05, key="lab_lp_min",
        help="Conservative enhancement floor applied even in the lowest-confidence "
             "regions.")
    c1, c2, c3 = st.columns(3)
    max_strength = c1.slider(
        "Max strength", 0.3, 1.0, 0.9, 0.05, key="lab_lp_max",
        help="Strongest effect in high-confidence regions.")
    confidence_low = c2.slider(
        "Confidence low", 0.05, 0.6, 0.30, 0.05, key="lab_lp_clow",
        help="Below this confidence the adaptive strength saturates at min.")
    confidence_high = c3.slider(
        "Confidence high", 0.4, 0.95, 0.70, 0.05, key="lab_lp_chigh",
        help="Above this confidence the adaptive strength saturates at max.")
    c1, c2, c3 = st.columns(3)
    local_contrast_strength = c1.slider(
        "Local contrast strength", 0.0, 1.0, 0.35, 0.05, key="lab_lp_contrast",
        help="Max gain of the confidence-weighted local contrast term "
             "(0 disables local contrast).")
    structure_preservation = c2.slider(
        "Structure preservation", 0.0, 2.0, 1.0, 0.1, key="lab_lp_structure",
        help="Scaling of the structure-aware contrast gain; favours medium-scale "
             "structure over noise-dominated residuals.")
    noise_floor_dn = c3.slider(
        "Noise floor (DN, optional)", 0.0, 2.0, 0.0, 0.1, key="lab_lp_floor",
        help="Optional guard floor added to the modeled noise for the confidence "
             "computation (0 keeps the pure Phase 4B model).")
    components = st.selectbox(
        "Algorithm components (ablation)",
        list(_LUNAPSR_COMPONENT_LABELS),
        index=list(_LUNAPSR_COMPONENT_LABELS).index("full"),
        format_func=lambda c: _LUNAPSR_COMPONENT_LABELS[c],
        key="lab_lp_components",
    )
    return LunaPSRParams(
        local_window=local_window,
        illumination_sigma=illumination_sigma,
        min_strength=min_strength,
        max_strength=max_strength,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        local_contrast_strength=local_contrast_strength,
        structure_preservation=structure_preservation,
        noise_floor_dn=noise_floor_dn,
        components=components,
    )


def _render_lunapsr(entry, product) -> None:
    st.divider()
    st.markdown("#### 🌑 LunaPSR Enhancement")
    st.caption(
        "Research prototype.  Adaptive enhancement whose strength depends on the "
        "local signal and the Phase 4B modeled noise.  Result is a derived "
        "representation of the available OHRC signal; it does not claim to "
        "recover information OHRC did not capture."
    )
    cfg = OHRCNoiseModelConfig.default()
    params = _render_lunapsr_params()

    c1, c2 = st.columns([1, 1])
    run = c1.button("⚙️ Run LunaPSR", type="primary", key="lab_lp_run")
    c2.caption(f"Algorithm version: `{params.version}` · model status: `{cfg.status}`")

    res = st.session_state.get(_LAB_LUNAPSR)
    if run:
        with st.spinner("Running LunaPSR on the analysis representation (read-only)..."):
            result = enhance_lunapsr(
                entry["base"], entry["normalized"], mask=entry["mask"],
                noise_model=cfg, params=params, bounds=entry["bounds"],
            )
        res = {"result": result, "params": params.to_dict()}
        st.session_state[_LAB_LUNAPSR] = res

    if res is None:
        st.info("Configure the parameters above and run LunaPSR.")
        return

    result = res["result"]
    if res["params"] != params.to_dict():
        st.caption("Note: parameters changed since the last run; press 'Run LunaPSR' to update.")

    warnings = result.diagnostics.warnings()
    if warnings:
        for w in warnings:
            st.warning(f"Potential enhancement artifact detected: {w}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean confidence", f"{np.nanmean(result.confidence):.3f}")
    c2.metric("Median model SNR", f"{np.nanmedian(result.model_based_local_snr):.2f}")
    c3.metric("Noise σ (model, DN)", f"{np.nanmean(result.noise_sigma_dn):.3f}")
    c4.metric("Enhancement score*", f"{result.diagnostics.score['score']:.3f}")
    st.caption("*Research diagnostic only — not a physical measurement.")

    # --- Views: Original / LunaPSR / CLAHE / Retinex / Zero-DCE -----------
    norm = np.nan_to_num(entry["normalized"], nan=0.0)
    disp_orig = _preview_8(norm)
    disp_lun = _preview_8(result.enhanced)

    others = {}
    try:
        from core.enhancement.baseline import apply_baseline as _ab

        others["CLAHE"] = _preview_8(_ab(norm, "clahe"))
        others["Retinex"] = _preview_8(_ab(norm, "retinex"))
        if zdce_available():
            from core.enhancement.zero_dce import zero_dce_enhance

            others["Zero-DCE"] = _preview_8(zero_dce_enhance(norm))
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Some baseline views unavailable: {exc}")

    tabs = st.tabs(["Original", "LunaPSR"] + list(others.keys()))
    with tabs[0]:
        st.image(disp_orig, caption="Original (normalized, derived)", channels="GRAY")
    with tabs[1]:
        st.image(disp_lun, caption="LunaPSR (enhanced, derived)", channels="GRAY")
    for i, (name, img) in enumerate(others.items()):
        with tabs[2 + i]:
            st.image(img, caption=f"{name} (enhanced, derived)", channels="GRAY")

    # --- Before / after + difference --------------------------------------
    shown = st.slider(
        "Before/after wipe (0 = original, 1 = LunaPSR)", 0.0, 1.0, 0.5, 0.05,
        key="lab_lp_wipe",
    )
    blended = (disp_orig.astype(np.float32) * (1.0 - shown)
               + disp_lun.astype(np.float32) * shown).astype(np.uint8)
    diff = _preview_8(np.abs(np.nan_to_num(result.enhanced, nan=0.0) - norm))
    c1, c2 = st.columns(2)
    with c1:
        st.image(blended, caption="Before/after wipe", channels="GRAY")
    with c2:
        st.image(diff, caption="Difference (|LunaPSR − original|, derived)",
                 channels="GRAY")
    st.caption(
        "The difference view shows what the enhancement changed; it is NOT "
        "interpreted as newly discovered information."
    )

    # --- Diagnostic maps ---------------------------------------------------
    with st.expander("Diagnostic maps (derived)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.image(_heat_map(result.confidence), caption="Local model confidence")
        c2.image(_heat_map(result.noise_sigma_dn / max(2.0, float(np.nanmax(result.noise_sigma_dn)))),
                 caption="Modeled noise σ (DN, normalized display)")
        c3.image(_heat_map(result.strength_map), caption="Adaptive strength")
        st.caption(
            "Confidence = snr/(1+snr) with snr = local signal / modeled noise "
            "(Phase 4B).  Maps are display diagnostics, not measurements."
        )

    # --- Diagnostics summary -----------------------------------------------
    with st.expander("LunaPSR diagnostics (research)", expanded=False):
        st.json(result.to_dict())

    # --- Benchmark: same evaluation framework for every method -------------
    with st.expander("Benchmark: LunaPSR vs Phase 3 methods", expanded=True):
        if st.button("Run full benchmark (all methods, same normalized input)",
                     key="lab_lp_bench"):
            with st.spinner("Benchmarking all methods..."):
                bench = benchmark_lunapsr(
                    entry["normalized"], entry["base"], entry["mask"],
                    entry["bounds"], params, cfg,
                )
            st.session_state["lab_lp_bench_result"] = bench
        bench = st.session_state.get("lab_lp_bench_result")
        if bench is not None:
            import pandas as pd

            rows = [r for r in benchmark_table(bench) if "error" not in r]
            st.table(pd.DataFrame(rows))
            st.caption(
                "Every method starts from the identical normalized input.  No "
                "winner is declared automatically.  'score' is a research "
                "diagnostic; model SNR is the Phase 4B model-based estimate "
                "evaluated on each method's derived DN."
            )

    # --- Experiment logging ------------------------------------------------
    _render_lunapsr_experiment(entry, product, result, params, cfg)


def _render_lunapsr_experiment(entry, product, result, params: LunaPSRParams, cfg) -> None:
    st.markdown("**Reproducibility & export (LunaPSR)**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save LunaPSR experiment (outputs/)", key="lab_lp_save"):
            from core.analysis.experiment import ExperimentRecord, ExperimentStore
            from core.analysis.metrics import quality_report

            rec = ExperimentRecord(
                product=product.path.name,
                method="lunapsr",
                parameters=params.to_dict(),
                normalization=entry["norm_params"],
                metrics={
                    "enhanced": quality_report(np.nan_to_num(result.enhanced, nan=0.0)),
                },
                analysis={
                    **entry["mask_details"],
                    "confidence": {
                        "mean": float(np.nanmean(result.confidence)),
                        "median": float(np.nanmedian(result.confidence)),
                    },
                    "diagnostics": result.diagnostics.to_dict(),
                    "enhanced_summary": result.to_dict(),
                },
                noise_model=cfg.to_dict(),
                notes=st.session_state.get("lab_notes", ""),
            )
            store = ExperimentStore()
            path = store.save_record(rec)
            st.success(f"Experiment recorded → `{path.relative_to(store.root)}`")
    with c2:
        import json

        payload = {
            "product": product.path.name,
            "processing_level": product.metadata.processing_level,
            "normalization": entry["norm_params"],
            "noise_model": cfg.to_dict(),
            "enhancement": params.to_dict(),
            "algorithm_version": params.version,
            "diagnostics": result.diagnostics.to_dict(),
            "caveat": result.caveat,
        }
        st.download_button(
            "⬇️ LunaPSR experiment JSON",
            json.dumps(payload, indent=2, default=str),
            file_name=f"{product.path.name}_lunapsr.json",
            mime="application/json",
            key="lab_lp_dl",
        )


def _render_lprt_params() -> LunaPSRRetinexConfig:
    st.markdown("**Adaptive Retinex parameters** (each documented)")
    c1, c2, c3 = st.columns(3)
    sigma = c1.slider(
        "Illumination sigma (px)", 5.0, 100.0, 30.0, 5.0, key="lab_lrt_sigma",
        help="Gaussian sigma of the illumination estimate inside the log-Retinex "
             "ratio. Larger = broader tone structure captured.")
    alpha_min = c2.slider(
        "Alpha min", 0.0, 0.6, 0.15, 0.05, key="lab_lrt_amin",
        help="Conservative enhancement floor applied even where confidence is "
             "lowest (noise-dominated dark regions stay near this).")
    alpha_max = c3.slider(
        "Alpha max", 0.4, 1.0, 0.85, 0.05, key="lab_lrt_amax",
        help="Strongest effect in high-confidence regions.")
    c1, c2, c3 = st.columns(3)
    k_trans = c1.slider(
        "Confidence transition k", 0.1, 4.0, 1.0, 0.1, key="lab_lrt_k",
        help="Sharpness of S/(S+k·N): larger k demands more signal per unit "
             "modeled noise before trusting a pixel.")
    detail = c2.slider(
        "Detail strength", 0.0, 1.0, 0.5, 0.05, key="lab_lrt_detail",
        help="Confidence-weighted local detail term; suppressed where variation "
             "is consistent with modeled noise.")
    protection = c3.slider(
        "Low-signal protection", 0.0, 1.0, 0.7, 0.05, key="lab_lrt_prot",
        help="Phase 5A lesson: level alone must not unlock enhancement. Higher "
             "values demand spatial-variation evidence above the noise model.")
    return LunaPSRRetinexConfig(
        sigma=sigma, alpha_min=alpha_min, alpha_max=alpha_max,
        confidence_transition=k_trans, detail_strength=detail,
        low_signal_protection=protection,
    )


def _render_lunapsr_retinex(entry, product) -> None:
    st.divider()
    st.markdown("#### 🌑 LunaPSR Adaptive Retinex")
    st.caption(
        "Research prototype (Phase 5B). Noise-aware adaptive Retinex: the "
        "enhancement strength α varies per pixel between conservative and strong "
        "according to a two-factor confidence built from the Phase 4B physics "
        "noise model (signal level AND spatial-variation evidence). Result is a "
        "derived representation of the available OHRC signal; it does not claim "
        "to recover information OHRC did not capture."
    )
    cfg_nm = OHRCNoiseModelConfig.default()
    cfg_rt = _render_lprt_params()

    from core.enhancement.retinex import single_scale_retinex

    c1, c2 = st.columns([1, 1])
    run = c1.button("⚙️ Run Adaptive Retinex", type="primary", key="lab_lrt_run")
    c2.caption(f"Algorithm version: `{cfg_rt.version}` · model status: `{cfg_nm.status}`")

    res = st.session_state.get(_LAB_LPRT)
    if run:
        with st.spinner("Running LunaPSR Adaptive Retinex on the analysis "
                        "representation (read-only)..."):
            ssr_out = single_scale_retinex(
                np.nan_to_num(entry["normalized"], nan=0.0), sigma=cfg_rt.sigma)
            result = enhance_lunapsr_retinex(
                entry["base"], entry["normalized"], valid_mask=entry["mask"],
                noise_model=cfg_nm, config=cfg_rt,
            )
        res = {"result": result, "ssr": ssr_out, "config": cfg_rt.to_dict()}
        st.session_state[_LAB_LPRT] = res

    if res is None:
        st.info("Configure the parameters above and run Adaptive Retinex.")
        return

    result = res["result"]
    if res["config"] != cfg_rt.to_dict():
        st.caption("Note: parameters changed since the last run; press 'Run Adaptive Retinex' to update.")

    for w in result.warnings:
        st.warning(w)

    art = result.artifacts
    fc = result.diagnostics.get("feature_consistency", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean alpha", f"{float(np.nanmean(result.alpha_map)):.3f}")
    c2.metric("Mean confidence", f"{result.diagnostics.get('confidence_mean', float('nan')):.3f}")
    c3.metric("New-edge fraction", f"{art.get('new_edge_fraction', float('nan')):.3f}",
              help="Fraction of original edges replaced by new edges (lower = more faithful).")
    c4.metric("EPI", f"{fc.get('edge_preservation_index', float('nan')):.3f}",
              help="Edge preservation index vs the original (higher = better).")
    st.caption("All values are research diagnostics on derived data — not physical measurements.")

    # --- Views: Original / SSR / LunaPSR ----------------------------------
    norm = np.nan_to_num(entry["normalized"], nan=0.0)
    disp_orig = _preview_8(norm)
    disp_ssr = _preview_8(res["ssr"])
    disp_lp = _preview_8(result.enhanced)

    tabs = st.tabs(["Original", "SSR (standard)", "LunaPSR Adaptive"])
    with tabs[0]:
        st.image(disp_orig, caption="Original (normalized, derived)", channels="GRAY")
    with tabs[1]:
        st.image(disp_ssr, caption="Standard single-scale Retinex (derived)", channels="GRAY")
    with tabs[2]:
        st.image(disp_lp, caption="LunaPSR Adaptive Retinex (derived)", channels="GRAY")

    # --- Before / after + difference --------------------------------------
    shown = st.slider(
        "Before/after wipe (0 = original, 1 = Adaptive Retinex)", 0.0, 1.0, 0.5, 0.05,
        key="lab_lrt_wipe",
    )
    blended = (disp_orig.astype(np.float32) * (1.0 - shown)
               + disp_lp.astype(np.float32) * shown).astype(np.uint8)
    diff = _preview_8(np.abs(np.nan_to_num(result.enhanced, nan=0.0) - norm))
    c1, c2 = st.columns(2)
    with c1:
        st.image(blended, caption="Before/after wipe", channels="GRAY")
    with c2:
        st.image(diff, caption="Difference (|Adaptive Retinex − original|, derived)",
                 channels="GRAY")
    st.caption(
        "The difference view shows what the enhancement changed; it is NOT "
        "interpreted as newly discovered information."
    )

    # --- Diagnostic maps ---------------------------------------------------
    with st.expander("Diagnostic maps (derived)", expanded=False):
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.image(_heat_map(result.confidence), caption="Composite confidence C")
        r1c2.image(_heat_map(result.alpha_map), caption="Adaptive strength α")
        r1c3.image(_heat_map(
            result.noise_sigma_dn / max(2.0, float(np.nanmax(result.noise_sigma_dn)))),
            caption="Modeled noise σ (DN, normalized display)")
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.image(_heat_map(result.illumination), caption="Illumination estimate G_σ(I)")
        r2c2.image(_heat_map(np.clip(result.retinex_response, 0.0, 1.0)),
                   caption="Retinex response R_d")
        struct = result.structure_dn
        r2c3.image(_heat_map(struct / max(1e-6, float(np.nanmax(struct)))),
                   caption="Structure map |M| (DN, normalized display)")
        st.caption(
            "C combines signal level S/(S+k·N) with variation evidence "
            "|M|/(|M|+N) through the low-signal protection gate. Maps are "
            "display diagnostics, not measurements."
        )

    # --- Diagnostics summary -----------------------------------------------
    with st.expander("Adaptive Retinex diagnostics (research)", expanded=False):
        st.json(result.to_dict())

    # --- Experiment logging ------------------------------------------------
    st.markdown("**Reproducibility & export (Adaptive Retinex)**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save Adaptive Retinex experiment (outputs/)", key="lab_lrt_save"):
            rec = ExperimentRecord(
                product=product.path.name,
                method="lunapsr_retinex",
                parameters=cfg_rt.to_dict(),
                normalization=entry["norm_params"],
                metrics={
                    "enhanced": quality_report(np.nan_to_num(result.enhanced, nan=0.0)),
                    **{f"artifact_{k}": v for k, v in art.items()
                       if isinstance(v, (int, float))},
                },
                analysis={
                    **entry["mask_details"],
                    "confidence": {
                        "mean": float(np.nanmean(result.confidence)),
                        "alpha_mean": float(np.nanmean(result.alpha_map)),
                    },
                    "diagnostics": result.diagnostics,
                    "warnings": list(result.warnings),
                    "enhanced_summary": result.to_dict(),
                },
                noise_model=cfg_nm.to_dict(),
                notes=st.session_state.get("lab_notes", ""),
            )
            store = ExperimentStore()
            path = store.save_record(rec)
            st.success(f"Experiment recorded → `{path.relative_to(store.root)}`")
    with c2:
        import json

        payload = {
            "product": product.path.name,
            "processing_level": product.metadata.processing_level,
            "normalization": entry["norm_params"],
            "noise_model": cfg_nm.to_dict(),
            "algorithm": cfg_rt.to_dict(),
            "diagnostics": result.diagnostics,
            "artifacts": {k: v for k, v in art.items()
                          if isinstance(v, (int, float, bool))},
            "warnings": list(result.warnings),
        }
        st.download_button(
            "⬇️ Adaptive Retinex experiment JSON",
            json.dumps(payload, indent=2, default=str),
            file_name=f"{product.path.name}_lunapsr_retinex.json",
            mime="application/json",
            key="lab_lrt_dl",
        )


baseline_label = lambda method: str(BASELINE_METHODS.get(method, method))  # noqa: E731


def _preview_8(arr: np.ndarray) -> np.ndarray:
    """8-bit display of a [0,1] array (NaN-safe, display-only)."""
    from app.ui.rendering import preview

    return preview(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0),
                   percentiles=OHRC_DISPLAY_PERCENTILES)


def _render_method_controls(entry: dict) -> Tuple[str, dict]:
    st.divider()
    st.markdown("#### Enhancement method")
    st.caption("Classical baselines only; all operate on the derived normalized representation.")
    method = st.selectbox(
        "Method", _METHODS, format_func=lambda m: BASELINE_METHODS.get(m, m),
        index=_METHODS.index("clahe") if "clahe" in _METHODS else 0, key="lab_method")
    if method == "zero_dce":
        if zdce_available():
            st.info("Zero-DCE backend detected.")
        else:
            st.warning("Zero-DCE unavailable: dependency not installed")
        iterations = st.slider("Iterations", 0, 16, 8, 1, key="lab_zdce_iter")
        return method, {"iterations": iterations}
    params = dict(BASELINE_DEFAULTS.get(method, {}))
    if method == "gamma":
        params["gamma"] = st.slider("Gamma", 0.1, 3.0, 0.75, 0.05, key="lab_gamma")
    if method == "clahe":
        params["clip_limit"] = st.slider("CLAHE clip limit", 0.5, 6.0, 2.0, 0.1, key="lab_clip")
        params["tile_size"] = st.slider("CLAHE tile size", 2, 16, 8, 1, key="lab_tile")
    if method == "unsharp_mask":
        params["strength"] = st.slider("Unsharp strength", 0.0, 2.0, 0.5, 0.1, key="lab_us")
        params["sigma"] = st.slider("Unsharp sigma", 0.5, 5.0, 1.0, 0.25, key="lab_usigma")
    if method == "denoising":
        params["strength"] = st.slider("Denoise strength", 1.0, 10.0, 3.0, 0.5, key="lab_ds")
    if method == "retinex":
        params["sigma"] = st.slider("Retinex sigma (px)", 5, 200, 30, 5, key="lab_rsigma")
        st.info(
            "**About SSR:** Retinex can increase apparent detail in very dark "
            "regions. Phase 5A analysis found that some newly generated "
            "high-frequency structure in the documented dark ROI was consistent "
            "with amplified read/quantization noise rather than demonstrated "
            "recovered lunar structure. See `docs/phase5a_ssr_investigation.md`."
        )
    return method, params


def render() -> None:
    st.subheader("OHRC Enhancement Lab")
    st.caption(
        "Phase 3 pipeline: product discovery → scientific image (read-only) → "
        "valid-pixel mask → robust normalization → baseline enhancement → "
        "metrics & comparison.  The original scientific array is never modified."
    )

    product = _render_source_selector()
    if product is None and st.session_state.get(_LAB_KEY):
        product = st.session_state[_LAB_KEY]["product"]
    if product is None:
        st.info("Select an OHRC product to begin.")
        return

    product = st.session_state[_LAB_KEY]["product"]
    st.markdown(f"**Product:** `{product.path.name}`")

    c1, c2 = st.columns(2)
    factor = c1.selectbox("Analysis factor (block-average)", [1, 5, 10, 20, 40],
                          index=[1, 5, 10, 20, 40].index(OHRC_ANALYSIS_FACTOR), key="lab_factor")
    lo_pct = c2.slider("Lower normalization percentile", 0.0, 50.0, float(ENHANCE_PERCENTILES[0]), 0.1,
                       key="lab_lo_pct")
    st.text_input("Processing note (recorded with the experiment)", key="lab_notes")

    entry = _derived(product, int(factor), float(lo_pct), float(ENHANCE_PERCENTILES[1]))

    # --- Noise analysis -------------------------------------------------
    st.divider()
    st.markdown("#### Noise analysis (normalized scientific)")
    noise = analyze_noise(np.nan_to_num(entry["normalized"], nan=np.nan), factor=4)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{noise.mean:.4g}")
    c2.metric("Std", f"{noise.std:.4g}")
    c3.metric("Local std", f"{noise.local_std:.4g}")
    c4.metric("Noise sigma (proxy)", f"{noise.noise_sigma:.4g}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median", f"{noise.median:.4g}")
    c2.metric("CV", f"{noise.coefficient_of_variation:.4g}" if noise.coefficient_of_variation is not None else "n/a")
    c3.metric("Background variation", f"{noise.background_variation:.4g}")
    c4.metric("SNR", noise.snr_status)
    with st.expander("Noise notes"):
        st.json(noise.to_dict())
        st.caption(
            "SNR is deliberately reported as 'Not yet estimated': a defensible "
            "SNR requires a calibrated OHRC sensor noise model that is not part "
            "of this product."
        )

    # --- Noise & SNR (model-based) -------------------------------------
    _render_noise_snr(entry, product)

    # --- LunaPSR adaptive enhancement ----------------------------------
    _render_lunapsr(entry, product)

    # --- LunaPSR Adaptive Retinex (Phase 5B) ----------------------------
    _render_lunapsr_retinex(entry, product)

    # --- Valid pixel mask ----------------------------------------------
    with st.expander("Valid-pixel mask"):
        st.json(entry["mask_details"])
        st.caption(
            "For this product the label declares no special constants, so the "
            "mask reflects that -- no nodata value is assumed."
        )

    # --- Enhancement ---------------------------------------------------
    method, params = _render_method_controls(entry)
    if st.button("Apply enhancement", type="primary", key="lab_apply"):
        try:
            res = apply_enhancement(entry["normalized"], method, params)
            st.session_state[_LAB_RESULT] = res
        except ZeroDCEUnavailableError as exc:
            st.error(str(exc))
            st.session_state[_LAB_RESULT] = None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Enhancement failed")
            st.error(f"Enhancement failed: {exc}")
            st.session_state[_LAB_RESULT] = None

    res = st.session_state.get(_LAB_RESULT)
    if res is not None:
        _render_display(product, entry, res, entry["normalized"], st.session_state.get("lab_opacity", 0.5))
        _render_quality_tables(res, entry, product)
        _render_roi(entry, product)
        _render_experiment_controls(entry, method, params, product, res, entry["normalized"])
    else:
        st.info("Enhancement result will appear here after applying a method.")
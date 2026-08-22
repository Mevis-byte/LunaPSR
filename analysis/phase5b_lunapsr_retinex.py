"""Phase 5B -- LunaPSR Adaptive Retinex validation on the REAL OHRC product.

Runs the exact Phase 5A ROI (full-resolution lines 61750-61949, samples
6000-6199 -- NOT a different region) through:

    A. Standard SSR                (existing baseline, unchanged)
    B. SSR + confidence map        (level-confidence alpha, no protection,
                                    no detail control)
    C. SSR + adaptive alpha        (+ low-signal protection gate)
    D. + noise-aware detail control (full pixel pipeline)
    E. Full LunaPSR                (complete public API incl. diagnostics)

and benchmarks Original / SSR / LunaPSR against the other existing baselines.
All derived outputs go to outputs/phase5b/ (Phase 5A outputs are untouched).
Read-only with respect to the product.

Usage:
    python analysis/phase5b_lunapsr_retinex.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "phase5b"

# Exact Phase 5A ROI (mandatory, do not change)
ROI_ROW0, ROI_COL0, ROI_SIZE = 61750, 6000, 200

# Phase 5A reference observations (benchmarks, NOT optimization targets)
P5A_REFERENCE = {
    "ssr_new_edge_fraction": 0.314,
    "ssr_isolated_new_components": 391,
    "ssr_edge_preservation_index": 0.827,
    "ssr_hf_energy_increase_fraction": 0.18,
}


def _display8(arr: np.ndarray) -> np.ndarray:
    """8-bit display stretch (1-99 percentile) -- visualization only."""
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    lo, hi = np.percentile(finite, [1, 99])
    if hi - lo < 1e-9:
        lo, hi = float(finite.min()), float(finite.max()) + 1e-9
    disp = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return (disp * 255.0).astype(np.uint8)


def _method_row(name: str, orig_norm: np.ndarray, out: np.ndarray,
                valid: np.ndarray, bounds: tuple, cfg_nm) -> dict:
    """Full metric set for one method output ([0,1] array)."""
    from core.analysis.ssr_investigation import (
        feature_consistency,
        gradient_energy,
        hf_energy,
        ssr_metrics,
    )
    from core.enhancement.confidence import local_model_confidence
    from core.enhancement.lunapsr import derive_dn
    from core.metrics.noise import estimate_noise_sigma

    v = out[valid].astype(np.float64)
    from core.analysis.metrics import local_variance

    feat = feature_consistency(orig_norm, out)
    m = ssr_metrics(orig_norm, out, valid=valid)
    enh_dn = derive_dn(out, bounds)
    conf_med = float(np.nanmedian(local_model_confidence(enh_dn, cfg_nm)))
    row = {
        "method": name,
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "std": round(float(v.std()), 4),
        "local_variance": round(float(local_variance(out)), 6),
        "gradient_energy": round(gradient_energy(out, valid), 6),
        "edge_density": round(_edge_density(out, valid), 4),
        "hf_energy": round(hf_energy(out, valid=valid), 6),
        "noise_estimate_mad": round(float(
            estimate_noise_sigma(np.where(valid, out, 0.0))), 5),
        "model_based_confidence_median": round(conf_med, 4),
        "noise_amplification_ratio": round(m["noise_amplification_ratio"], 3),
        "clipping_fraction": round(m["ssr"]["clipping_fraction"], 4),
        "new_edge_fraction": feat["new_edge_fraction_of_ssr_edges"],
        "isolated_new_components": feat["isolated_new_components"],
        "epi": feat["edge_preservation_index"],
        "orientation_agreement": feat["orientation_agreement_doubled_angle"],
        "halo_like_profile": feat["halo_like_profile"],
    }
    return row


def _edge_density(arr: np.ndarray, valid: np.ndarray) -> float:
    from core.analysis.ssr_investigation import _grad_magnitude

    g = _grad_magnitude(arr)[valid]
    return float((g >= np.percentile(g, 95)).mean())


def main() -> int:
    from PIL import Image

    from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
    from core.enhancement.lunapsr_retinex import (
        LunaPSRRetinexConfig,
        enhance_lunapsr_retinex,
    )
    from core.enhancement.retinex import single_scale_retinex
    from core.preprocessing.normalization import normalize_scientific

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    nm = OHRCNoiseModelConfig.default()

    # ---- load real product (read-only) and cut the EXACT Phase 5A ROI -----
    from core.ohrc import OHRCProduct

    product = OHRCProduct.load(
        PROJECT_ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18",
        load_geometry=False,
    )
    sci = product.image.scientific_array()
    roi = np.asarray(sci[ROI_ROW0:ROI_ROW0 + ROI_SIZE,
                         ROI_COL0:ROI_COL0 + ROI_SIZE], dtype=np.float32)
    norm, norm_params = normalize_scientific(roi, percentiles=(1, 99))
    bounds = (float(norm_params["lower_value"]), float(norm_params["upper_value"]))
    valid = np.isfinite(roi) & np.isfinite(norm)

    # ---- ablation A-E ------------------------------------------------------
    ssr_out = single_scale_retinex(norm, sigma=30.0)

    cfg_b = LunaPSRRetinexConfig(low_signal_protection=0.0, detail_strength=0.0)
    res_b = enhance_lunapsr_retinex(roi, norm, valid, nm, cfg_b)

    cfg_c = LunaPSRRetinexConfig(low_signal_protection=0.7, detail_strength=0.0)
    res_c = enhance_lunapsr_retinex(roi, norm, valid, nm, cfg_c)

    cfg_full = LunaPSRRetinexConfig()
    res_e = enhance_lunapsr_retinex(roi, norm, valid, nm, cfg_full)

    variants = [
        ("A_standard_ssr", "existing single_scale_retinex(sigma=30), unchanged",
         {"sigma": 30.0}, ssr_out, None),
        ("B_ssr_confidence", "alpha from level confidence only "
         "(low_signal_protection=0, detail_strength=0)",
         {"low_signal_protection": 0.0, "detail_strength": 0.0},
         res_b.enhanced, res_b),
        ("C_adaptive_alpha", "+ low-signal protection gate "
         "(low_signal_protection=0.7, detail_strength=0)",
         {"low_signal_protection": 0.7, "detail_strength": 0.0},
         res_c.enhanced, res_c),
        ("D_detail_control", "+ noise-aware detail control "
         "(detail_strength=0.5) -- full pixel pipeline",
         {"low_signal_protection": 0.7, "detail_strength": 0.5},
         res_e.enhanced, res_e),
        ("E_full_lunapsr", "complete public API with diagnostics",
         {"full_api": True}, res_e.enhanced, res_e),
    ]

    ablation_rows = []
    for name, desc, deltas, out, res in variants:
        if res is None:
            alpha_mean = None
            warn_n = 0
            art = {}
        else:
            alpha_mean = res.diagnostics["alpha_mean"]
            warn_n = len(res.warnings)
            art = res.artifacts
        feat = _feat_for(out, norm, valid)
        ablation_rows.append({
            "variant": name,
            "description": desc,
            "config_delta": json.dumps(deltas),
            "alpha_mean": alpha_mean,
            "noise_amplification_ratio": art.get("noise_amplification_ratio"),
            "hf_energy_ratio": art.get("hf_energy_ratio"),
            "clipping_fraction": art.get("clipping_fraction_after"),
            "new_edge_fraction": feat["new_edge_fraction_of_ssr_edges"],
            "isolated_new_components": feat["isolated_new_components"],
            "epi": feat["edge_preservation_index"],
            "orientation_agreement": feat["orientation_agreement_doubled_angle"],
            "warnings": warn_n,
        })
    with open(OUT / "ablation.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)

    # ---- benchmark (Original / SSR / LunaPSR + baselines) ------------------
    bench_rows = [_method_row("original", norm, norm, valid, bounds, nm)]
    bench_rows.append(_method_row("ssr", norm, ssr_out, valid, bounds, nm))
    bench_rows.append(_method_row("lunapsr", norm, res_e.enhanced, valid, bounds, nm))

    from core.enhancement.baseline import apply_baseline

    for method in ("gamma", "clahe", "denoising"):
        try:
            out = apply_baseline(np.nan_to_num(norm, nan=0.0), method)
            bench_rows.append(_method_row(method, norm, out, valid, bounds, nm))
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            bench_rows.append({"method": method, "error": str(exc)})
    try:
        from core.enhancement.zero_dce import is_available as zdce_ok

        if zdce_ok():
            from core.enhancement.zero_dce import zero_dce_enhance

            out = zero_dce_enhance(np.nan_to_num(norm, nan=0.0))
            bench_rows.append(_method_row("zero_dce", norm, out, valid, bounds, nm))
        else:
            bench_rows.append({"method": "zero_dce", "error": "Zero-DCE unavailable"})
    except Exception as exc:  # noqa: BLE001
        bench_rows.append({"method": "zero_dce", "error": str(exc)})

    by_name = {r["method"]: r for r in bench_rows if "error" not in r}
    ssr_row, lp_row = by_name["ssr"], by_name["lunapsr"]

    result = {
        "phase": "5B",
        "algorithm": "LunaPSR Adaptive Retinex",
        "version": "0.1",
        "roi": {
            "row0": ROI_ROW0, "col0": ROI_COL0, "size": ROI_SIZE,
            "row1": ROI_ROW0 + ROI_SIZE, "col1": ROI_COL0 + ROI_SIZE,
            "pixel_count": ROI_SIZE * ROI_SIZE,
            "source": "Exact Phase 5A ROI (mandatory)",
        },
        "normalization": norm_params,
        "noise_model": nm.to_dict(),
        "parameters": cfg_full.to_dict(),
        "ablation": ablation_rows,
        "benchmark": bench_rows,
        "ssr_vs_lunapsr": {
            k: {"ssr": ssr_row.get(k), "lunapsr": lp_row.get(k)}
            for k in ("noise_amplification_ratio", "hf_energy",
                      "new_edge_fraction", "isolated_new_components", "epi",
                      "orientation_agreement", "clipping_fraction",
                      "local_variance", "gradient_energy",
                      "model_based_confidence_median")
        },
        "phase5a_reference": P5A_REFERENCE,
        "regression_check": {
            "new_edge_fraction_not_worse_than_ssr":
                lp_row["new_edge_fraction"] <= ssr_row["new_edge_fraction"],
            "isolated_not_worse_than_ssr":
                lp_row["isolated_new_components"] <= ssr_row["isolated_new_components"],
            "epi_not_worse_than_ssr": lp_row["epi"] >= ssr_row["epi"],
            "hf_energy_not_worse_than_original":
                lp_row["hf_energy"] <= _hf_of(norm, valid) * 1.18,
        },
        "lunapsr_warnings": res_e.warnings,
        "caveats": [
            "LunaPSR is an enhancement algorithm producing a DERIVED REPRESENTATION.",
            "It does not recover unmeasured photons or reconstruct true reflectance.",
            "Visible structures are not proven geological.",
            "MODEL-BASED SNR/confidence uses the INFERRED gain (~104 e-/DN).",
            "PNG files are visualization products only.",
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }

    # ---- visual outputs -----------------------------------------------------
    Image.fromarray(_display8(roi)).save(OUT / "original_roi.png")
    Image.fromarray(_display8(ssr_out)).save(OUT / "ssr_roi.png")
    Image.fromarray(_display8(res_e.enhanced)).save(OUT / "lunapsr_roi.png")
    Image.fromarray(_display8(res_e.confidence)).save(OUT / "confidence.png")
    Image.fromarray(_display8(res_e.alpha_map)).save(OUT / "alpha.png")
    Image.fromarray(_display8(res_e.noise_sigma_dn)).save(OUT / "noise.png")
    Image.fromarray(_display8(res_e.illumination)).save(OUT / "illumination.png")
    Image.fromarray(_display8(res_e.retinex_response)).save(
        OUT / "retinex_response.png")
    Image.fromarray(_display8(np.abs(ssr_out - norm))).save(
        OUT / "difference_ssr.png")
    Image.fromarray(_display8(np.abs(res_e.enhanced - norm))).save(
        OUT / "difference_original.png")

    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if hasattr(obj, "to_dict"):
            return _clean(obj.to_dict())
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    (OUT / "metrics.json").write_text(
        json.dumps(_clean(result), indent=2, default=str), encoding="utf-8")

    # ---- experiment record ---------------------------------------------------
    from core.analysis.experiment import ExperimentRecord

    rec = ExperimentRecord(
        product=product.path.name,
        method="lunapsr_retinex",
        parameters={
            **cfg_full.to_dict(),
            "algorithm": "LunaPSR Adaptive Retinex",
            "version": "0.1",
        },
        normalization=norm_params,
        metrics={"benchmark": bench_rows, "regression": result["regression_check"]},
        analysis={
            "roi": result["roi"],
            "phase5a_reference": P5A_REFERENCE,
            "diagnostics": res_e.diagnostics,
            "artifacts": res_e.artifacts,
            "warnings": res_e.warnings,
        },
        noise_model=nm.to_dict(),
        notes=("Phase 5B prototype validation on the exact Phase 5A ROI. "
               "Derived representation only."),
    )
    (OUT / "experiment.json").write_text(rec.to_json(), encoding="utf-8")

    print(f"done in {result['elapsed_s']}s -> {OUT}")
    print("regression:", result["regression_check"])
    print("SSR     :", {k: ssr_row[k] for k in ("new_edge_fraction",
          "isolated_new_components", "epi", "noise_amplification_ratio")})
    print("LunaPSR :", {k: lp_row[k] for k in ("new_edge_fraction",
          "isolated_new_components", "epi", "noise_amplification_ratio")})
    print("warnings:", res_e.warnings)
    return 0


def _feat_for(out, orig, valid):
    from core.analysis.ssr_investigation import feature_consistency

    return feature_consistency(orig, out)


def _hf_of(arr, valid):
    from core.analysis.ssr_investigation import hf_energy

    return hf_energy(arr, valid=valid)


if __name__ == "__main__":
    raise SystemExit(main())
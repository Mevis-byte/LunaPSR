"""Phase 5A -- SSR scientific investigation on the REAL OHRC product.

Investigates why Single-Scale Retinex (SSR) reveals structure inside the
dark (PSR stand-in) region while gamma / histogram equalization / CLAHE /
unsharp / denoising show substantially less visible structure, and whether
that structure is

    A. spatial signal already present in the original OHRC DN data,
    B. amplified sensor/image noise,
    C. a mixture of both.

Read-only with respect to the product.  All derived outputs go to
outputs/phase5a/.  The ROI is the documented Phase 4A §10.3 dark patch
(full-resolution lines 61750-61949, samples 6000-6199) -- the project's
established PSR stand-in and Phase 4B noise-validation region.

Usage:
    python analysis/phase5a_ssr_investigation.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "phase5a"

# Phase 4A §10.3 dark patch (full resolution)
ROI_ROW0, ROI_COL0, ROI_SIZE = 61750, 6000, 200


def _display8(arr: np.ndarray) -> np.ndarray:
    """8-bit display stretch (1-99 percentile) -- visualization only."""
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi - lo < 1e-9:
        lo, hi = float(a.min()), float(a.max()) + 1e-9
    return (np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def main() -> int:
    from PIL import Image

    from core.analysis.ssr_investigation import (
        dn_statistics,
        feature_consistency,
        model_noise_analysis,
        ssr_metrics,
        structure_maps,
        window_analysis_table,
    )
    from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
    from core.enhancement.lunapsr import LunaPSRParams, benchmark_lunapsr
    from core.enhancement.retinex import single_scale_retinex as ssr
    from core.preprocessing.normalization import normalize_scientific

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = OHRCNoiseModelConfig.default()

    # ---- load product (read-only) and cut the exact ROI -------------------
    from core.ohrc import OHRCProduct

    product = OHRCProduct.load(
        PROJECT_ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18",
        load_geometry=False,
    )
    sci = product.image.scientific_array()
    roi = np.asarray(sci[ROI_ROW0:ROI_ROW0 + ROI_SIZE,
                         ROI_COL0:ROI_COL0 + ROI_SIZE], dtype=np.float32)

    result: dict = {
        "phase": "5A",
        "purpose": ("SSR investigation: is revealed structure original signal, "
                    "amplified noise, or a mixture?"),
        "roi": {
            "row0": ROI_ROW0, "col0": ROI_COL0, "size": ROI_SIZE,
            "row1": ROI_ROW0 + ROI_SIZE, "col1": ROI_COL0 + ROI_SIZE,
            "width": ROI_SIZE, "height": ROI_SIZE,
            "pixel_count": ROI_SIZE * ROI_SIZE,
            "source": ("Phase 4A §10.3 dark patch -- documented darkest "
                       "(PSR stand-in) region; also the Phase 4B "
                       "noise-validation region"),
            "units": "original scientific DN",
        },
    }

    # ---- 2. raw DN statistics ---------------------------------------------
    raw = dn_statistics(roi)
    result["raw_dn_statistics"] = raw
    print(f"ROI DN: mean={raw['mean']:.3f} median={raw['median']:.1f} "
          f"min={raw['min']:.0f} max={raw['max']:.0f} std={raw['std']:.3f} "
          f"zero%={100*raw['zero_fraction']:.2f} sat%={100*raw['saturation_fraction']:.2f}")

    # ---- 3. physics-aware noise analysis (original DN -> Phase 4B model) ---
    noise = model_noise_analysis(roi, cfg)
    result["model_based_noise_analysis"] = {
        **noise,
        "gain_status": cfg.gain_e_per_dn and "INFERRED (~104 e-/DN)",
        "caveat": ("MODEL-BASED SNR (gain inferred); not a measured SNR. "
                   "Evaluated on ORIGINAL DN, never on normalized values."),
    }
    print(f"model: sigma_total={noise['total_sigma_dn']:.3f} DN "
          f"(shot {noise['shot_sigma_dn']:.3f} + read {noise['read_sigma_dn']:.3f} "
          f"+ quant {noise['quant_sigma_dn']:.3f}), model-based SNR={noise['model_snr']:.2f}")

    # ---- ROI-local normalization (same for every method; fairness) --------
    norm, norm_params = normalize_scientific(roi, percentiles=(1, 99))
    bounds = (float(norm_params["lower_value"]), float(norm_params["upper_value"]))
    result["normalization"] = {
        **norm_params,
        "scope": ("ROI-local percentile normalization applied identically to "
                  "every method for fairness; raw-DN statistics above are "
                  "reported separately in DN units"),
    }

    # ---- 6. SSR (existing implementation, unchanged) -----------------------
    ssr_out = ssr(norm, sigma=30.0)
    result["ssr"] = {
        "implementation": "core/enhancement/retinex.py single_scale_retinex (unchanged)",
        "parameters": {"sigma": 30.0, "gain": 1.0, "offset": 0.0, "floor": 1e-3},
        "normalization_method": "global linear re-map at output 1-99 percentiles into [0,1]",
        "output_range": "[0,1] (clipped)",
    }

    # ---- 7. SSR metrics ----------------------------------------------------
    metrics = ssr_metrics(norm, ssr_out)
    result["ssr_metrics"] = metrics
    print(f"noise amplification ratio (SSR/original): "
          f"{metrics['noise_amplification_ratio']:.2f}")

    # ---- 8. feature consistency --------------------------------------------
    feat = feature_consistency(norm, ssr_out)
    result["feature_consistency"] = feat
    print(f"strengthened orig edges: {feat['strengthened_fraction_of_original_edges']:.2f}, "
          f"new SSR edges: {feat['new_edge_fraction_of_ssr_edges']:.2f}, "
          f"isolated new comps: {feat['isolated_new_components']}")

    # ---- 4./9. windowed structure-vs-noise table ---------------------------
    rows = window_analysis_table(roi, cfg, window=20)
    n_above = sum(1 for r in rows if r["category"] != "consistent_with_modeled_noise")
    ratios = np.array([r["ratio"] for r in rows])
    result["window_analysis"] = {
        "window_size": 20,
        "n_windows": len(rows),
        "n_above_model": int(n_above),
        "ratio_median": round(float(np.median(ratios)), 3),
        "ratio_max": round(float(ratios.max()), 3),
        "label_rule": ("ratio >= 1.5 => 'spatial variation above the current "
                       "noise model' (NOT automatically geology)"),
    }
    with open(OUT / "window_analysis.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"windows: {len(rows)} total, {n_above} above model "
          f"(median ratio {np.median(ratios):.2f})")

    # ---- 5. structure maps (derived representations) -----------------------
    maps = structure_maps(roi, cfg, window=5)
    Image.fromarray(_display8(maps["gradient_magnitude"])).save(
        OUT / "gradient_original.png")
    Image.fromarray(_display8(maps["model_noise_sigma_dn"])).save(
        OUT / "noise_map.png")
    Image.fromarray(_display8(maps["model_confidence"])).save(
        OUT / "confidence_map.png")
    np.savez(OUT / "structure_maps.npz", **maps)

    # ---- 10. visual outputs -------------------------------------------------
    Image.fromarray(_display8(roi)).save(OUT / "original_roi.png")
    Image.fromarray(_display8(ssr_out)).save(OUT / "ssr_roi.png")
    from core.analysis.ssr_investigation import _grad_magnitude

    Image.fromarray(_display8(_grad_magnitude(ssr_out))).save(
        OUT / "gradient_ssr.png")
    diff = np.abs(ssr_out - norm)
    Image.fromarray(_display8(diff)).save(OUT / "difference.png")

    # ---- 12. comparison on the SAME ROI ------------------------------------
    comparison_rows = benchmark_lunapsr(
        norm, roi, None, bounds, LunaPSRParams(), cfg)
    result["comparison"] = {
        "methods": comparison_rows,
        "focus": "the selected PSR-stand-in ROI only",
    }
    print("comparison methods:", ", ".join(comparison_rows.keys()))

    # ---- metrics.json -------------------------------------------------------
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["caveats"] = [
        "All PNG files are visualization products only.",
        "MODEL-BASED SNR uses the INFERRED gain (~104 e-/DN).",
        ("Windows flagged 'spatial variation above the current noise model' "
         "are NOT classified as geological features."),
        ("SSR output is a derived representation; no claim of recovering "
         "information OHRC did not capture."),
    ]

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
    print(f"\ndone in {result['elapsed_s']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Phase 5B -- full-resolution tiled LunaPSR Adaptive Retinex run.

Read-only with respect to the product: the scientific array is memory-mapped
and processed in strips; nothing in the product tree is ever modified.  All
derived outputs go to outputs/phase5b/full_resolution/.

Tiling contract:
  * strip height STRIP rows, halo = required_halo(config) (>= 4*sigma + 8)
    on every side so every Gaussian window sees its true context;
  * the Retinex response tone remap uses GLOBALLY determined bounds
    (measured once on a factor-16 representation) so all strips share one
    consistent mapping;
  * alpha / confidence depend only on local quantities, so they are
    tile-consistent by construction.

Outputs (all derived):
  lunapsr_retinex_enhanced_f32.img   full-res float32 enhanced [0,1]
  lunapsr_retinex_enhanced_u8.img    full-res uint8 display version
  previews/*.png                     reduced-resolution display views
  maps/*.npy                         reduced-res confidence/alpha/noise maps
  metrics.json                       quality report + artifact summary
  experiment record                  ExperimentStore JSON

Usage:
  python analysis/phase5b_fullres.py
  python analysis/phase5b_fullres.py --max-lines 4096   (debug)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCT_DIR = PROJECT_ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18"
OUT = PROJECT_ROOT / "outputs" / "phase5b" / "full_resolution"

STRIP = 2048          # rows per tile
LINES, SAMPLES = 101075, 12000


def _percentile_normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    out = (a - lo) / max(hi - lo, 1e-9)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    h = a.shape[0] // factor
    w = a.shape[1] // factor
    return (a[: h * factor, : w * factor]
            .reshape(h, factor, w, factor).mean(axis=(1, 3)).astype(np.float32))


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if hasattr(obj, "to_dict"):
        return _clean(obj.to_dict())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=None,
                    help="process only the first N lines (debugging)")
    args = ap.parse_args()

    from PIL import Image

    from core.analysis.experiment import ExperimentRecord, ExperimentStore
    from core.analysis.metrics import quality_report
    from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
    from core.enhancement.lunapsr_retinex import (
        LunaPSRRetinexConfig,
        enhance_lunapsr_retinex,
        required_halo,
    )
    from core.ohrc import OHRCProduct
    from core.ohrc.statistics import exact_histogram, percentile_from_histogram
    from core.preprocessing.normalization import normalize_scientific

    for d in ("", "previews", "maps"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg_nm = OHRCNoiseModelConfig.default()
    cfg_rt = LunaPSRRetinexConfig()
    halo = max(required_halo(cfg_rt), 4 * cfg_rt.sigma + 8)

    product = OHRCProduct.load(PRODUCT_DIR, load_geometry=False)
    mm = product.image.scientific_array()
    n_lines = LINES if args.max_lines is None else args.max_lines
    assert mm.shape == (LINES, SAMPLES)

    # ---- global normalization bounds from the exact histogram (1,99) ------
    counts, edges, _, _ = exact_histogram(mm[:n_lines])
    lo_v = percentile_from_histogram(counts, edges, 1.0)
    hi_v = percentile_from_histogram(counts, edges, 99.0)
    if lo_v is None or hi_v is None or hi_v <= lo_v:
        lo_v, hi_v = 0.0, 1.0
    bounds = (float(lo_v), float(hi_v))
    print(f"normalization bounds (1,99): lo={lo_v:.4g} hi={hi_v:.4g}", flush=True)

    # ---- global Retinex response remap bounds (factor-16 estimate) --------
    red = product.decimate_average(factor=16)
    red_norm = _percentile_normalize(red, lo_v, hi_v)
    probe = enhance_lunapsr_retinex(
        red, red_norm, noise_model=cfg_nm, config=cfg_rt,
        compute_diagnostics=False,
    )
    resp_red = probe.retinex_response
    valid_red = np.isfinite(resp_red)
    r_lo, r_hi = (float(v) for v in np.percentile(resp_red[valid_red], [1, 99]))
    print(f"global response remap bounds: lo={r_lo:.5f} hi={r_hi:.5f}", flush=True)
    del resp_red, probe

    summary: dict = {
        "product": product.path.name,
        "algorithm": cfg_rt.to_dict(),
        "noise_model": cfg_nm.to_dict(),
        "normalization_bounds": bounds,
        "response_remap_bounds": [r_lo, r_hi],
        "tiling": {"strip": STRIP, "halo": halo},
        "processing": {"lines": n_lines, "samples": SAMPLES,
                       "pixels": n_lines * SAMPLES},
    }

    # ---- full-resolution tiled enhancement --------------------------------
    f32_path = OUT / "lunapsr_retinex_enhanced_f32.img"
    u8_path = OUT / "lunapsr_retinex_enhanced_u8.img"
    out_f32 = np.memmap(str(f32_path), dtype=np.float32, mode="w+",
                        shape=(n_lines, SAMPLES))
    out_u8 = np.memmap(str(u8_path), dtype=np.uint8, mode="w+",
                       shape=(n_lines, SAMPLES))
    clip_counts = 0
    alpha_sum, conf_sum, n_sum = 0.0, 0.0, 0
    for lo in range(0, n_lines, STRIP):
        hi = min(lo + STRIP, n_lines)
        y0 = max(0, lo - halo)
        y1 = min(LINES, hi + halo)
        tile = np.asarray(mm[y0:y1])
        norm = _percentile_normalize(tile.astype(np.float32), lo_v, hi_v)
        res = enhance_lunapsr_retinex(
            tile, norm, noise_model=cfg_nm, config=cfg_rt,
            compute_diagnostics=False, display_bounds=(r_lo, r_hi),
        )
        block = res.enhanced[lo - y0: hi - y0]
        out_f32[lo:hi] = block
        out_u8[lo:hi] = np.clip(np.nan_to_num(block, nan=0.0) * 255.0,
                                0.0, 255.0).astype(np.uint8)
        b = block[np.isfinite(block)]
        clip_counts += int(((b == 0.0) | (b == 1.0)).sum())
        av = res.alpha_map[lo - y0: hi - y0]
        cv = res.confidence[lo - y0: hi - y0]
        alpha_sum += float(np.nansum(av))
        conf_sum += float(np.nansum(cv))
        n_sum += int(np.isfinite(av).sum())
        print(f"  rows {lo}:{hi} done ({time.time()-t0:.1f}s)", flush=True)
    out_f32.flush()
    del out_f32
    out_u8.flush()
    del out_u8
    summary["fullres"] = {
        "path_f32": str(f32_path),
        "path_u8": str(u8_path),
        "shape": [n_lines, SAMPLES],
        "derived": ("full-resolution enhanced representation "
                    "(float32 [0,1] + uint8 display, derived)"),
        "clipped_pixels": clip_counts,
        "alpha_mean": alpha_sum / max(n_sum, 1),
        "confidence_mean": conf_sum / max(n_sum, 1),
    }
    print(f"  full-res pass done ({time.time()-t0:.1f}s)", flush=True)

    # ---- reduced-resolution diagnostics (factor 16) ------------------------
    red_norm, norm_params = normalize_scientific(
        red, percentiles=(1, 99), precomputed_bounds=bounds)
    res16 = enhance_lunapsr_retinex(red, red_norm, noise_model=cfg_nm,
                                    config=cfg_rt)
    summary["normalization"] = norm_params
    summary["diagnostics"] = {
        "confidence_mean": float(np.nanmean(res16.confidence)),
        "alpha_mean": float(np.nanmean(res16.alpha_map)),
        "artifacts": {k: v for k, v in res16.artifacts.items()
                      if isinstance(v, (int, float, bool))},
        "warnings": list(res16.warnings),
    }
    np.save(OUT / "maps" / "confidence.npy",
            res16.confidence.astype(np.float32))
    np.save(OUT / "maps" / "alpha.npy", res16.alpha_map.astype(np.float32))
    np.save(OUT / "maps" / "noise_sigma_dn.npy",
            res16.noise_sigma_dn.astype(np.float32))

    # ---- previews (display-only, downsampled) ------------------------------
    orig_dn16 = _block_average(np.asarray(mm[:n_lines]), 16)
    enh_dn16 = _block_average(
        np.memmap(str(u8_path), dtype=np.uint8, mode="r",
                  shape=(n_lines, SAMPLES)), 16) / 255.0
    p_orig = (_percentile_normalize(orig_dn16, lo_v, hi_v) * 255).astype(np.uint8)
    p_enh = (np.clip(enh_dn16, 0, 1) * 255).astype(np.uint8)
    diff = np.abs(p_enh.astype(np.float32) - p_orig.astype(np.float32))
    diff = (diff / max(diff.max(), 1e-9) * 255).astype(np.uint8)
    Image.fromarray(p_orig).save(OUT / "previews" / "original_display.png")
    Image.fromarray(p_enh).save(
        OUT / "previews" / "lunapsr_retinex_display.png")
    Image.fromarray(diff).save(OUT / "previews" / "difference_display.png")
    Image.fromarray((np.clip(np.nan_to_num(res16.confidence), 0, 1)
                     * 255).astype(np.uint8)).save(
        OUT / "previews" / "confidence_map_display.png")
    Image.fromarray((np.clip(np.nan_to_num(res16.alpha_map), 0, 1)
                     * 255).astype(np.uint8)).save(
        OUT / "previews" / "alpha_map_display.png")
    print("  previews written", flush=True)

    # ---- metrics ------------------------------------------------------------
    metrics = {
        "original": quality_report(red_norm),
        "enhanced": quality_report(res16.enhanced),
        "artifacts": summary["diagnostics"]["artifacts"],
        "caveat": "No-reference metrics describe the derived representation only.",
    }
    (OUT / "metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2, default=str), encoding="utf-8")

    # ---- experiment record ---------------------------------------------------
    rec = ExperimentRecord(
        product=product.path.name,
        method="lunapsr_retinex_fullres",
        parameters={**cfg_rt.to_dict(),
                    "tiling": {"strip": STRIP, "halo": halo},
                    "response_remap_bounds": [r_lo, r_hi]},
        normalization=norm_params,
        metrics={"enhanced": quality_report(res16.enhanced)},
        analysis={
            "processing_level": product.metadata.processing_level,
            "bounds": bounds,
            "diagnostics": summary["diagnostics"],
            "fullres_paths": [str(f32_path), str(u8_path)],
        },
        noise_model=cfg_nm.to_dict(),
        notes=(
            "Phase 5B LunaPSR Adaptive Retinex, full-resolution tiled run. "
            "Derived representation of the available OHRC signal; no claim of "
            "recovering information OHRC did not capture."
        ),
    )
    store = ExperimentStore()
    rec_path = store.save_record(rec)
    summary["experiment_record"] = str(rec_path)

    summary["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(
        json.dumps(_clean(summary), indent=2, default=str), encoding="utf-8")
    print(f"\ndone in {summary['elapsed_s']}s -> {OUT}", flush=True)
    print("alpha mean:", summary["fullres"]["alpha_mean"],
          "| conf mean:", summary["fullres"]["confidence_mean"],
          "| clipped:", clip_counts, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
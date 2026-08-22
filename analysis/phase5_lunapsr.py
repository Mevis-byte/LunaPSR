"""Phase 5 -- run LunaPSR adaptive enhancement on the real OHRC product.

Read-only with respect to the product: the scientific array is memory-mapped
and processed in tiles; nothing in the product tree is ever modified.  All
derived outputs go to outputs/phase5/:

    fullres/          full-resolution derived enhanced representation (uint8)
    previews/         8-bit display previews (downsampled, display-only)
    maps/             confidence / noise diagnostic maps (reduced-resolution)
    metrics/          metric and benchmark JSON
    ablation/         A/B/C/D ablation results
    experiments/      experiment record JSON

Previews and maps are reduced-resolution DISPLAY representations; only the
full-resolution enhanced array is a full-res derived product.

Usage:
    python analysis/phase5_lunapsr.py
    python analysis/phase5_lunapsr.py --max-lines 4096   (debug)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCT_DIR = PROJECT_ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18"
OUT = PROJECT_ROOT / "outputs" / "phase5"

STRIP = 2048          # rows per tile
HALO = 48             # padding for median/gaussian windows
DISPLAY = (1.0, 99.0)  # display percentiles for previews (display-only)
REDUCE = 16           # block-average factor for previews / maps / metrics

LINES, SAMPLES = 101075, 12000


def _preview8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Derived 8-bit display of a [0,1] array (display-only)."""
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def _percentile_normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    out = (a - lo) / max(hi - lo, 1e-9)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=None,
                    help="process only the first N lines (debugging)")
    ap.add_argument("--no-fullres", action="store_true",
                    help="skip the full-resolution tiled pass (debugging)")
    args = ap.parse_args()

    from core.ohrc import OHRCProduct
    from core.ohrc.statistics import percentile_from_histogram
    from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
    from core.enhancement.lunapsr import (
        LunaPSRParams, enhance_lunapsr, benchmark_lunapsr, benchmark_table,
    )
    from core.analysis.metrics import quality_report
    from core.analysis.experiment import ExperimentRecord
    from core.preprocessing.normalization import normalize_scientific

    for d in ("fullres", "previews", "maps", "metrics", "ablation", "experiments"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = OHRCNoiseModelConfig.default()
    params = LunaPSRParams()

    product = OHRCProduct.load(PRODUCT_DIR, load_geometry=False)
    mm = product.image.scientific_array()
    n_lines = LINES if args.max_lines is None else args.max_lines
    assert mm.shape == (LINES, SAMPLES)

    # ---- global bounds from the exact histogram (1,99) --------------------
    from core.ohrc.statistics import exact_histogram

    counts, edges, _, _ = exact_histogram(mm[:n_lines])
    lo_v = percentile_from_histogram(counts, edges, 1.0)
    hi_v = percentile_from_histogram(counts, edges, 99.0)
    if lo_v is None or hi_v is None or hi_v <= lo_v:
        lo_v, hi_v = 0.0, 1.0
    bounds = (float(lo_v), float(hi_v))
    print(f"normalization bounds (1,99): lo={lo_v:.4g} hi={hi_v:.4g}")

    summary: dict = {"product": product.path.name, "bounds": bounds,
                     "params": params.to_dict(),
                     "noise_model": cfg.to_dict(),
                     "processing": {"lines": n_lines, "samples": SAMPLES,
                                    "pixels": n_lines * SAMPLES}}

    # ---- 1. full-resolution tiled enhancement (derived uint8) ------------
    if not args.no_fullres and n_lines > 0:
        print("full-resolution tiled enhancement (read-only, chunked)...")
        out_path = OUT / "fullres" / "lunapsr_enhanced_u8.img"
        out = np.memmap(str(out_path), dtype=np.uint8, mode="w+",
                        shape=(n_lines, SAMPLES))
        clip_counts = 0
        strip = STRIP
        for lo in range(0, n_lines, strip):
            hi = min(lo + strip, n_lines)
            y0 = max(0, lo - HALO)
            y1 = min(LINES, hi + HALO)
            tile = mm[y0:y1]
            norm = _percentile_normalize(tile.astype(np.float32), lo_v, hi_v)
            res = enhance_lunapsr(
                tile, norm, noise_model=cfg, params=params,
                bounds=bounds, compute_diagnostics=False,
            )
            block = res.enhanced[lo - y0: hi - y0]
            out[lo:hi] = _preview8(block, 0.0, 1.0)
            b = block[np.isfinite(block)]
            clip_counts += int(((b == 0.0) | (b == 1.0)).sum())
            if (lo // strip) % 10 == 0:
                print(f"  rows {lo}:{hi} done ({time.time()-t0:.1f}s)")
        out.flush()
        del out
        summary["fullres"] = {
            "path": str(out_path),
            "shape": [n_lines, SAMPLES],
            "derived": "full-resolution enhanced representation (uint8, derived)",
            "clipped_pixels": clip_counts,
        }
        print(f"  full-res pass done ({time.time()-t0:.1f}s)")

    # ---- 2. reduced-resolution display / metrics representation ----------
    print("reduced-resolution diagnostics (factor 16, display/analysis)...")
    red = product.decimate_average(factor=REDUCE)
    red_norm, norm_params = normalize_scientific(
        red, percentiles=(1, 99), precomputed_bounds=bounds,
    )
    res16 = enhance_lunapsr(red, red_norm, noise_model=cfg, params=params,
                            bounds=bounds)
    summary["normalization"] = norm_params
    summary["diagnostics"] = res16.diagnostics.to_dict()
    summary["confidence"] = {
        "mean": float(np.nanmean(res16.confidence)),
        "median": float(np.nanmedian(res16.confidence)),
        "fractions": _confidence_fractions(res16.confidence),
    }
    summary["score"] = res16.diagnostics.score

    # ---- 3. previews (display-only, downsampled) --------------------------
    if args.max_lines is None:
        orig_dn16 = product.decimate_average(factor=REDUCE)
    else:
        orig_dn16 = _block_average(np.asarray(mm[:n_lines]), REDUCE)
    enh_dn16 = orig_dn16.copy()
    if (OUT / "fullres" / "lunapsr_enhanced_u8.img").exists():
        enh_full = np.memmap(str(OUT / "fullres" / "lunapsr_enhanced_u8.img"),
                             dtype=np.uint8, mode="r", shape=(n_lines, SAMPLES))
        enh_dn16 = _block_average(np.asarray(enh_full), REDUCE)
    p_orig = _preview8(_percentile_normalize(orig_dn16, lo_v, hi_v), 0.0, 1.0)
    p_enh = _preview8(_percentile_normalize(enh_dn16, lo_v, hi_v), 0.0, 1.0)
    diff = np.abs(p_enh.astype(np.float32) - p_orig.astype(np.float32))
    diff = _preview8(diff / max(diff.max(), 1e-9), 0.0, 1.0)
    from PIL import Image

    Image.fromarray(p_orig).save(OUT / "previews" / "original_display.png")
    Image.fromarray(p_enh).save(OUT / "previews" / "lunapsr_enhanced_display.png")
    Image.fromarray(diff).save(OUT / "previews" / "difference_display.png")
    Image.fromarray(_preview8(res16.confidence, 0.0, 1.0)).save(
        OUT / "previews" / "confidence_map_display.png")
    ns = res16.noise_sigma_dn / max(float(np.nanmax(res16.noise_sigma_dn)), 1e-9)
    Image.fromarray(_preview8(ns, 0.0, 1.0)).save(
        OUT / "previews" / "noise_sigma_map_display.png")
    print("  previews written")

    # ---- 4. maps (raw + display) ------------------------------------------
    np.save(OUT / "maps" / "confidence.npy", res16.confidence.astype(np.float32))
    np.save(OUT / "maps" / "noise_sigma_dn.npy", res16.noise_sigma_dn.astype(np.float32))
    np.save(OUT / "maps" / "strength.npy", res16.strength_map.astype(np.float32))
    np.save(OUT / "maps" / "model_snr.npy", res16.model_based_local_snr.astype(np.float32))
    print("  maps written")

    # ---- 5. metrics & benchmark -------------------------------------------
    metrics = {
        "original": quality_report(red_norm),
        "enhanced": quality_report(res16.enhanced),
        "diagnostics": res16.diagnostics.to_dict(),
        "caveat": "No-reference metrics describe the derived representation only.",
    }
    (OUT / "metrics" / "lunapsr_metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2, default=str), encoding="utf-8")

    bench = benchmark_lunapsr(red_norm, red, None, bounds, params, cfg)
    (OUT / "metrics" / "benchmark_all_methods.json").write_text(
        json.dumps(_clean(bench), indent=2, default=str), encoding="utf-8")
    print("  metrics + benchmark written")

    # ---- 6. ablation A/B/C/D ----------------------------------------------
    ablation = {}
    for comp in ("tone_only", "tone_adaptive", "adaptive_contrast", "full"):
        p_ab = LunaPSRParams(components=comp)
        rab = enhance_lunapsr(red, red_norm, noise_model=cfg, params=p_ab, bounds=bounds)
        Image.fromarray(_preview8(rab.enhanced, 0.0, 1.0)).save(
            OUT / "ablation" / f"ablation_{comp}_display.png")
        ablation[comp] = {
            "params": p_ab.to_dict(),
            "score": rab.diagnostics.score,
            "diagnostics": rab.diagnostics.to_dict(),
            "confidence_mean": float(np.nanmean(rab.confidence)),
        }
    (OUT / "ablation" / "ablation_summary.json").write_text(
        json.dumps(_clean(ablation), indent=2, default=str), encoding="utf-8")
    print("  ablation written")

    # ---- 7. experiment record ---------------------------------------------
    rec = ExperimentRecord(
        product=product.path.name,
        method="lunapsr",
        parameters=params.to_dict(),
        normalization=norm_params,
        metrics={"enhanced": quality_report(res16.enhanced)},
        analysis={
            "processing_level": product.metadata.processing_level,
            "bounds": bounds,
            "diagnostics": res16.diagnostics.to_dict(),
            "confidence": summary["confidence"],
            "benchmark_path": str(OUT / "metrics" / "benchmark_all_methods.json"),
            "fullres_path": summary.get("fullres", {}).get("path"),
        },
        noise_model=cfg.to_dict(),
        notes=(
            "Phase 5 LunaPSR adaptive enhancement prototype. Derived "
            "representation of the available OHRC signal; no claim of "
            "recovering information OHRC did not capture."
        ),
    )
    (OUT / "experiments" / f"lunapsr_{rec.identifier()}.json").write_text(
        rec.to_json(), encoding="utf-8")
    print("  experiment record written")

    summary["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(
        json.dumps(_clean(summary), indent=2, default=str), encoding="utf-8")
    print(f"\ndone in {summary['elapsed_s']}s -> {OUT}")
    print("score:", summary["score"]["score"], "| conf mean:",
          summary["confidence"]["mean"], "| model SNR median:",
          float(np.nanmedian(res16.model_based_local_snr)),
          "| laplacian ratio:",
          summary["diagnostics"]["noise_amplification"].get("laplacian_ratio"))
    return 0


def _confidence_fractions(confidence: np.ndarray) -> dict:
    from core.enhancement.confidence import confidence_categories

    cat = confidence_categories(confidence)
    valid = cat < 255
    n = int(valid.sum())
    if n == 0:
        return {"low": 0.0, "medium": 0.0, "high": 0.0}
    return {
        "low": float((cat[valid] == 0).mean()),
        "medium": float((cat[valid] == 1).mean()),
        "high": float((cat[valid] == 2).mean()),
    }


def _block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    h = a.shape[0] // factor
    w = a.shape[1] // factor
    return a[: h * factor, : w * factor].reshape(h, factor, w, factor).mean(axis=(1, 3)).astype(np.float32)


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


if __name__ == "__main__":
    raise SystemExit(main())
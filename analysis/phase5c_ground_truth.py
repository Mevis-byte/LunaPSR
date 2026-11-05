"""Phase 5C -- synthetic ground-truth benchmark on the REAL OHRC dark region.

Question this experiment answers
--------------------------------
Phases 5A/5B measured how much *new* structure enhancement methods add in a
dark PSR region.  They could not measure whether *real* structure is recovered,
because a single frame has no ground truth.  Here, known analytic craters and
boulders are injected at controlled amplitude (in DN and in units of the Phase
4B model sigma) into real, noise-dominated OHRC tiles.  Every method is then
asked to recover them with a *generic* difference-of-Gaussians detector whose
threshold is calibrated on feature-free control tiles at a fixed 5% false
positive rate.

Methods
-------
    raw               global 1/99 normalization only (no enhancement)
    ssr               single-scale Retinex, sigma = 30 px (Phase 5A baseline)
    lunapsr           LunaPSR Adaptive Retinex (Phase 5B, default config)
    anscombe_nlm      generalized Anscombe + non-local means (skimage)
    anscombe_bm3d     generalized Anscombe + BM3D (optional, analysis-only)

The Anscombe methods use the Phase 4B noise model (INFERRED gain) to
variance-stabilise the data before denoising.  BM3D is an *analysis-only*
dependency (``pip install bm3d``); it is not required by the app.

What is measured
----------------
* detection rate per amplitude at a fixed 5% false-positive rate on control
  tiles (16 injected features per condition: 8 tiles x 2 replicates);
* AUC of the detector statistic (features vs control locations);
* local contrast-to-noise and shape correlation of the recovered structure;
* fabrication forensics on the same control tiles (new-edge fraction,
  isolated components, edge preservation, noise amplification) so detection
  gains can be read together with hallucination costs.

Honest limitations
------------------
* Features are idealised analytic profiles, not real lunar photometry.
* Sub-DN amplitudes are a continuous-signal proxy (the product is uint8).
* Tiles come from one dark region of one product -- no terrain diversity.
* The DoG detector is generic but not exhaustive; other detectors could shift
  absolute rates (relative ordering is the informative part).
* Everything here concerns a DERIVED REPRESENTATION.  No output proves
  geological reality or recovers unmeasured photons.

Usage
-----
    python analysis/phase5c_ground_truth.py            # full run
    python analysis/phase5c_ground_truth.py --quick    # reduced smoke run

Outputs: outputs/phase5c/ (metrics.json, features.csv, summary.csv,
controls.csv, tiles.csv, montages, experiment.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "outputs" / "phase5c"
PRODUCT_DIR = PROJECT_ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18"

# Search region for noise-dominated 256x256 tiles (verified dark in Phase 4A/5A)
REGION = {"row0": 60000, "row1": 64000, "col0": 4000, "col1": 9000}
TILE = 256

# --- experiment design ------------------------------------------------------
SEED = 20260505
N_TILES = 8
AMPLITUDES_DN = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0)
SIZES_PX = {"crater": (12, 24, 40), "boulder": (6, 12, 20)}
REPLICATES = 2              # features per tile per condition
FP_RATE = 0.05              # detector calibration target on control tiles
CONTROL_PER_TILE = 400      # random control locations per tile (split cal/val)

METHODS = ("raw", "ssr", "lunapsr", "anscombe_nlm", "anscombe_bm3d")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _display_u8(arr: np.ndarray, lo: float | None = None,
                hi: float | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.uint8)
    if lo is None or hi is None:
        lo, hi = (float(v) for v in np.percentile(a[finite], [1.0, 99.5]))
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    disp = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return (disp * 255.0).astype(np.uint8)


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


def _rng(*parts) -> np.random.Generator:
    """Deterministic RNG from a stable key string."""
    key = "|".join(str(p) for p in parts)
    code = sum((i + 1) * ord(ch) for i, ch in enumerate(key)) + SEED
    return np.random.default_rng(code)


# ---------------------------------------------------------------------------
# Tile selection and normalization
# ---------------------------------------------------------------------------


def _select_tiles(sci, n_tiles: int) -> list[dict]:
    """Greedy spatially-spread selection of clean, noise-dominated tiles."""
    cands: list[dict] = []
    for r in range(REGION["row0"], REGION["row1"] - TILE + 1, TILE):
        for c in range(REGION["col0"], REGION["col1"] - TILE + 1, TILE):
            t = np.asarray(sci[r:r + TILE, c:c + TILE])
            med = float(np.median(t))
            std = float(t.std())
            p99 = float(np.percentile(t, 99))
            if 1.5 <= med <= 5.5 and std <= 1.0 and p99 <= 8.0:
                cands.append({"row": r, "col": c, "median_dn": med,
                              "std_dn": std, "p99_dn": p99})
    if len(cands) < n_tiles:
        raise RuntimeError(
            f"only {len(cands)} clean tiles found in {REGION}; need {n_tiles}"
        )

    cy = 0.5 * (REGION["row0"] + REGION["row1"])
    cx = 0.5 * (REGION["col0"] + REGION["col1"])
    cands.sort(key=lambda x: (x["row"], x["col"]))
    chosen = [min(cands, key=lambda x: (x["row"] - cy) ** 2 + (x["col"] - cx) ** 2)]
    while len(chosen) < n_tiles:
        best, best_dist = None, -1.0
        for cand in cands:
            if cand in chosen:
                continue
            d = min(float(np.hypot(cand["row"] - ch["row"],
                                   cand["col"] - ch["col"])) for ch in chosen)
            if d > best_dist:
                best, best_dist = cand, d
        chosen.append(best)
    return sorted(chosen, key=lambda x: (x["row"], x["col"]))


def _normalize(dn: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    from core.preprocessing.normalization import normalize_scientific

    norm, _ = normalize_scientific(np.asarray(dn, dtype=np.float32),
                                   percentiles=(1, 99),
                                   precomputed_bounds=bounds)
    return norm


# ---------------------------------------------------------------------------
# Anscombe (generalized) variance stabilization
# ---------------------------------------------------------------------------


def _anscombe_params(nm) -> tuple[float, float]:
    """(alpha, sigma^2) in DN^2 for Var(y|mu) = alpha*mu + sigma^2."""
    alpha = 1.0 / float(nm.gain_e_per_dn)
    sigma2 = (float(nm.read_noise_e) / float(nm.gain_e_per_dn)) ** 2 \
        + float(nm.quantization_variance_dn2)
    return alpha, sigma2


def anscombe_forward(dn: np.ndarray, nm) -> np.ndarray:
    alpha, sigma2 = _anscombe_params(nm)
    x = np.maximum(np.asarray(dn, dtype=np.float64), 0.0)
    return (2.0 / alpha) * np.sqrt(alpha * x + 0.375 * alpha ** 2 + sigma2)


def anscombe_inverse(z: np.ndarray, nm) -> np.ndarray:
    alpha, sigma2 = _anscombe_params(nm)
    y = 0.25 * alpha * np.asarray(z, dtype=np.float64) ** 2 \
        - 0.375 * alpha - sigma2 / alpha
    return np.maximum(y, 0.0).astype(np.float32)


def _anscombe_nlm(dn, nm) -> np.ndarray:
    from skimage.restoration import denoise_nl_means

    z = anscombe_forward(dn, nm)
    out = denoise_nl_means(z, patch_size=5, patch_distance=6, h=0.8,
                           sigma=1.0, fast_mode=True, preserve_range=True)
    return anscombe_inverse(out, nm)


def _anscombe_bm3d(dn, nm) -> np.ndarray:
    import bm3d

    z = anscombe_forward(dn, nm)
    out = bm3d.bm3d(z.astype(np.float32), sigma_psd=1.0, profile="np")
    return anscombe_inverse(out, nm)


# ---------------------------------------------------------------------------
# Methods under test
# ---------------------------------------------------------------------------


def _available_methods() -> list[str]:
    from importlib.util import find_spec

    methods = list(METHODS)
    if find_spec("bm3d") is None:
        methods.remove("anscombe_bm3d")
    return methods


def _apply_method(method: str, dn: np.ndarray, norm: np.ndarray,
                  bounds: tuple[float, float], nm, rt_cfg) -> np.ndarray:
    from core.enhancement.lunapsr_retinex import enhance_lunapsr_retinex
    from core.enhancement.retinex import single_scale_retinex

    if method == "raw":
        return np.asarray(norm, dtype=np.float32).copy()
    if method == "ssr":
        return single_scale_retinex(norm, sigma=30.0)
    if method == "lunapsr":
        return enhance_lunapsr_retinex(
            dn, norm, noise_model=nm, config=rt_cfg,
            compute_diagnostics=False,
        ).enhanced
    if method == "anscombe_nlm":
        return _normalize(_anscombe_nlm(dn, nm), bounds)
    if method == "anscombe_bm3d":
        return _normalize(_anscombe_bm3d(dn, nm), bounds)
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Detection-curve figures (PIL only; visualization, not analysis)
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "raw": (150, 160, 175),
    "ssr": (245, 150, 40),
    "lunapsr": (79, 195, 247),
    "anscombe_nlm": (90, 200, 120),
    "anscombe_bm3d": (220, 120, 220),
}
_METHOD_LABELS = {
    "raw": "raw (normalized)",
    "ssr": "SSR σ=30",
    "lunapsr": "LunaPSR",
    "anscombe_nlm": "Anscombe+NLM",
    "anscombe_bm3d": "Anscombe+BM3D",
}

_X_TICKS = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
_X_MIN, _X_MAX = 0.15, 12.0


def _x_map(snr: float, left: int, width: int) -> float:
    lo, hi = np.log10(_X_MIN), np.log10(_X_MAX)
    t = (np.log10(float(np.clip(snr, _X_MIN, _X_MAX))) - lo) / (hi - lo)
    return left + t * width


def _draw_curve_panel(rows: list[dict], methods: list[str], title: str,
                      width: int = 360, height: int = 250) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default()
    img = Image.new("RGB", (width, height), (14, 18, 26))
    d = ImageDraw.Draw(img)
    ml, mr, mt, mb = 42, 12, 24, 28
    pw, ph = width - ml - mr, height - mt - mb

    d.text((6, 6), title, fill=(225, 235, 245), font=font)

    # grid + axes
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = mt + (1.0 - q) * ph
        d.line([(ml, y), (ml + pw, y)], fill=(38, 46, 60))
        d.text((8, y - 5), f"{q:.2f}", fill=(120, 135, 152), font=font)
    for xv in _X_TICKS:
        x = _x_map(xv, ml, pw)
        d.line([(x, mt), (x, mt + ph)], fill=(30, 36, 48))
        d.text((x - 8, mt + ph + 6), f"{xv:g}", fill=(120, 135, 152), font=font)
    d.rectangle([ml, mt, ml + pw, mt + ph], outline=(90, 105, 125))

    for m in methods:
        pts = sorted(
            (r for r in rows if r["method"] == m),
            key=lambda r: r["input_snr_model"],
        )
        if not pts:
            continue
        color = _METHOD_COLORS.get(m, (200, 200, 200))
        xy = [
            (_x_map(r["input_snr_model"], ml, pw),
             mt + (1.0 - float(r["detection_rate"])) * ph)
            for r in pts
        ]
        for r, (x, y) in zip(pts, xy):
            lo, hi = r.get("detection_ci_lo"), r.get("detection_ci_hi")
            if lo is not None and hi is not None:
                d.line([(x, mt + (1.0 - hi) * ph), (x, mt + (1.0 - lo) * ph)],
                       fill=tuple(int(c * 0.45 + 20) for c in color))
        if len(xy) > 1:
            d.line(xy, fill=color, width=2)
        for x, y in xy:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    return img


def _detection_figure(kind: str, diameters: tuple[int, ...],
                      summary_rows: list[dict], methods: list[str],
                      path: Path) -> None:
    from PIL import ImageDraw, ImageFont

    panels = [
        _draw_curve_panel(
            [r for r in summary_rows
             if r["kind"] == kind and r["diameter_px"] == d],
            methods, f"{kind}  d={d} px",
        )
        for d in diameters
    ]
    w = sum(p.width for p in panels)
    h = max(p.height for p in panels)
    fig = Image.new("RGB", (w, h + 26), (14, 18, 26))
    x = 0
    for p in panels:
        fig.paste(p, (x, 0))
        x += p.width
    d = ImageDraw.Draw(fig)
    font = ImageFont.load_default()
    x = 10
    for m in methods:
        color = _METHOD_COLORS.get(m, (200, 200, 200))
        d.line([(x, h + 13), (x + 18, h + 13)], fill=color, width=2)
        label = _METHOD_LABELS.get(m, m)
        d.text((x + 23, h + 7), label, fill=(205, 215, 228), font=font)
        x += 34 + int(d.textlength(label, font=font))
    d.text((max(x + 20, w - 190), h + 7),
           "x: input peak SNR (σ model, log)  ·  y: detection @ 5% FP",
           fill=(120, 135, 152), font=font)
    fig.save(path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main(quick: bool = False) -> int:
    from core.analysis.experiment import ExperimentRecord
    from core.analysis.ground_truth import (
        InjectedFeature,
        auc,
        calibrate_threshold,
        detector_sigma,
        dog_response,
        false_positive_rate,
        feature_readout,
        feature_score,
        footprint_radius,
        inject_features,
        place_features,
        random_location_scores,
        wilson_interval,
    )
    from core.analysis.ohrc_noise_model import (
        OHRCNoiseModelConfig,
        estimate_flat_window_noise,
        noise_breakdown,
    )
    from core.analysis.ssr_investigation import feature_consistency
    from core.enhancement.lunapsr_retinex import LunaPSRRetinexConfig
    from core.metrics.noise import estimate_noise_sigma
    from core.ohrc import OHRCProduct
    from core.ohrc.statistics import exact_histogram, percentile_from_histogram

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    nm = OHRCNoiseModelConfig.default()
    rt_cfg = LunaPSRRetinexConfig()
    methods = _available_methods()
    amplitudes = AMPLITUDES_DN if not quick else (0.1, 0.3, 1.0)
    n_tiles = N_TILES if not quick else 2
    control_per_tile = CONTROL_PER_TILE if not quick else 80

    # ---- product + global normalization bounds ---------------------------
    print("loading product (read-only)...", flush=True)
    product = OHRCProduct.load(PRODUCT_DIR, load_geometry=False)
    sci = product.image.scientific_array()
    print(f"product shape {sci.shape}", flush=True)

    print("computing exact full-product 1/99 bounds...", flush=True)
    counts, edges, _, _ = exact_histogram(sci)
    lo_v = percentile_from_histogram(counts, edges, 1.0)
    hi_v = percentile_from_histogram(counts, edges, 99.0)
    bounds = (float(lo_v), float(hi_v))
    print(f"global bounds (1,99): lo={bounds[0]:.4g} hi={bounds[1]:.4g}", flush=True)

    # ---- tile selection ---------------------------------------------------
    print("selecting clean dark tiles...", flush=True)
    tiles = _select_tiles(sci, n_tiles)
    for t in tiles:
        t["dn"] = np.asarray(sci[t["row"]:t["row"] + TILE,
                                 t["col"]:t["col"] + TILE], dtype=np.float32)
        rep = noise_breakdown(t["median_dn"], nm)
        fw = estimate_flat_window_noise(t["dn"], nm)
        t["model_sigma_dn"] = float(rep.total_sigma_dn)
        t["model_snr"] = float(rep.model_snr)
        t["measured_sigma_dn"] = float(fw.measured_sigma_dn)
        t["measured_over_model"] = (
            float(fw.measured_sigma_dn / rep.total_sigma_dn)
            if rep.total_sigma_dn > 0 else None
        )
        t["n_flat_windows"] = int(fw.n_windows)
        print(f"  tile r{t['row']} c{t['col']}: median={t['median_dn']:.2f} "
              f"std={t['std_dn']:.3f} sigma_meas={t['measured_sigma_dn']:.3f} "
              f"sigma_model={t['model_sigma_dn']:.3f}", flush=True)

    with open(OUT / "tiles.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "row", "col", "median_dn", "std_dn", "p99_dn", "model_sigma_dn",
            "model_snr", "measured_sigma_dn", "measured_over_model",
            "n_flat_windows"])
        w.writeheader()
        for t in tiles:
            w.writerow({k: t[k] for k in w.fieldnames})

    # ---- control (feature-free) method outputs + fabrication forensics ----
    print("running methods on feature-free control tiles...", flush=True)
    control_outputs: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    control_forensics: dict[str, dict[str, list[float]]] = {
        m: defaultdict(list) for m in methods
    }
    for t in tiles:
        norm = _normalize(t["dn"], bounds)
        t["norm"] = norm
        for m in methods:
            out = _apply_method(m, t["dn"], norm, bounds, nm, rt_cfg)
            control_outputs[m].append(out)
            fc = feature_consistency(norm, out)
            vb = max(estimate_noise_sigma(np.nan_to_num(norm)), 1e-9)
            va = max(estimate_noise_sigma(np.nan_to_num(out)), 1e-9)
            for key in ("new_edge_fraction_of_ssr_edges",
                        "isolated_new_components",
                        "edge_preservation_index"):
                control_forensics[m][key].append(float(fc[key]))
            control_forensics[m]["noise_amplification_ratio"].append(float(va / vb))
        print(f"  control tile r{t['row']} c{t['col']} done", flush=True)

    # ---- per-tile, per-size detector calibration --------------------------
    # Thresholds are calibrated on each tile's own feature-free control
    # response: tile-to-tile noise differences are real (a single global
    # threshold transfers poorly -- verified in the Phase 5C pilot).  Within
    # each tile, half the control locations calibrate and the other half
    # validate the false-positive rate.
    control_stats: list[dict] = []
    for kind in ("crater", "boulder"):
        for diameter in SIZES_PX[kind]:
            probe = InjectedFeature(kind, 0.0, 0.0, diameter, 1.0)
            sigma_px = detector_sigma(probe)
            radius_px = footprint_radius(probe)
            for m in methods:
                for i, out in enumerate(control_outputs[m]):
                    resp = dog_response(out, sigma_px)
                    scores = random_location_scores(
                        resp, radius_px, control_per_tile,
                        _rng("ctrl", kind, diameter, m, i))
                    n_half = max(1, scores.size // 2)
                    cal_s, val_s = scores[:n_half], scores[n_half:]
                    if val_s.size == 0:
                        val_s = cal_s
                    thr = calibrate_threshold(cal_s, fp_rate=FP_RATE)
                    control_stats.append({
                        "method": m, "kind": kind, "diameter_px": diameter,
                        "tile_row": tiles[i]["row"], "tile_col": tiles[i]["col"],
                        "detector_sigma_px": sigma_px,
                        "footprint_radius_px": radius_px,
                        "threshold": thr,
                        "fp_rate_calibration": false_positive_rate(cal_s, thr),
                        "fp_rate_validation": false_positive_rate(val_s, thr),
                        "n_control_scores": int(cal_s.size),
                        "n_validation_scores": int(val_s.size),
                        "_cal_scores": cal_s,
                        "_val_scores": val_s,
                    })
            print(f"  calibrated {kind} d={diameter} (sigma={sigma_px:.2f}px)",
                  flush=True)

    stat_index = {(s["method"], s["kind"], s["diameter_px"],
                   s["tile_row"], s["tile_col"]): s
                  for s in control_stats}

    def _control_agg(m: str, kind: str, diameter: int) -> dict:
        rows = [s for s in control_stats if s["method"] == m
                and s["kind"] == kind and s["diameter_px"] == diameter]
        return {
            "threshold_median": float(np.median([r["threshold"] for r in rows])),
            "fp_rate_calibration_mean": float(np.mean(
                [r["fp_rate_calibration"] for r in rows])),
            "fp_rate_validation_mean": float(np.mean(
                [r["fp_rate_validation"] for r in rows])),
            "fp_rate_validation_max": float(np.max(
                [r["fp_rate_validation"] for r in rows])),
            "n_tiles": len(rows),
        }

    def _neg_scores(m: str, kind: str, diameter: int) -> np.ndarray:
        vals = [s["_val_scores"] for s in control_stats
                if s["method"] == m and s["kind"] == kind
                and s["diameter_px"] == diameter]
        return np.concatenate(vals) if vals else np.array([])

    # ---- amplitude sweep --------------------------------------------------
    feature_rows: list[dict] = []
    total_conditions = sum(len(SIZES_PX[k]) for k in SIZES_PX) * len(amplitudes)
    done = 0
    for kind in ("crater", "boulder"):
        for diameter in SIZES_PX[kind]:
            sigma_px = detector_sigma(InjectedFeature(kind, 0.0, 0.0,
                                                      diameter, 1.0))
            # fixed placement across amplitudes -> paired amplitude sweep
            placement = {}
            for i, t in enumerate(tiles):
                placement[i] = place_features(
                    (TILE, TILE), kind, diameter, 1.0, REPLICATES,
                    _rng("place", kind, diameter, i),
                )
            for amplitude in amplitudes:
                for i, t in enumerate(tiles):
                    feats = [
                        InjectedFeature(f.kind, f.row, f.col, f.diameter_px,
                                        amplitude)
                        for f in placement[i]
                    ]
                    inj, truth = inject_features(t["dn"], feats)
                    norm_inj = _normalize(inj, bounds)
                    for m in methods:
                        s = stat_index[(m, kind, diameter,
                                        t["row"], t["col"])]
                        out = _apply_method(m, inj, norm_inj, bounds, nm, rt_cfg)
                        resp = dog_response(out, sigma_px)
                        for j, f in enumerate(feats):
                            score = feature_score(resp, f)
                            ro = feature_readout(out, f, truth)
                            feature_rows.append({
                                "method": m, "kind": kind,
                                "diameter_px": diameter,
                                "amplitude_dn": amplitude,
                                "input_snr_model": amplitude / t["model_sigma_dn"],
                                "tile_row": t["row"], "tile_col": t["col"],
                                "feature_index": j,
                                "feature_row": float(f.row),
                                "feature_col": float(f.col),
                                "score": score,
                                "threshold": s["threshold"],
                                "detected": bool(score > s["threshold"]),
                                "cnr_ptp": ro["cnr_ptp"],
                                "corr_truth": ro["corr_truth"],
                                "mean_shift": ro["mean_shift"],
                            })
                done += 1
                print(f"  sweep {done}/{total_conditions}: {kind} d={diameter} "
                      f"A={amplitude} DN", flush=True)

    # ---- aggregation ------------------------------------------------------
    groups: dict[tuple, dict] = {}
    for row in feature_rows:
        key = (row["method"], row["kind"], row["diameter_px"],
               row["amplitude_dn"])
        g = groups.setdefault(key, {"scores": [], "detected": 0,
                                    "cnr": [], "corr": [], "snr": []})
        g["scores"].append(row["score"])
        g["detected"] += int(row["detected"])
        g["snr"].append(row["input_snr_model"])
        if np.isfinite(row["cnr_ptp"]):
            g["cnr"].append(row["cnr_ptp"])
        if np.isfinite(row["corr_truth"]):
            g["corr"].append(row["corr_truth"])

    summary_rows: list[dict] = []
    for (m, kind, diameter, amplitude), g in sorted(groups.items()):
        n = len(g["scores"])
        k = int(g["detected"])
        lo, hi = wilson_interval(k, n)
        agg = _control_agg(m, kind, diameter)
        summary_rows.append({
            "method": m, "kind": kind, "diameter_px": diameter,
            "amplitude_dn": amplitude,
            "input_snr_model": float(np.mean(g["snr"])),
            "n": n, "detected": k,
            "detection_rate": k / n,
            "detection_ci_lo": lo, "detection_ci_hi": hi,
            "median_cnr_ptp": float(np.median(g["cnr"])) if g["cnr"] else None,
            "median_corr_truth": float(np.median(g["corr"])) if g["corr"] else None,
            "threshold_median": agg["threshold_median"],
            "fp_rate_validation": agg["fp_rate_validation_mean"],
            "fp_rate_validation_max": agg["fp_rate_validation_max"],
        })

    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feature_rows[0].keys()))
        w.writeheader()
        w.writerows(feature_rows)

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    for kind in ("crater", "boulder"):
        _detection_figure(kind, SIZES_PX[kind], summary_rows, methods,
                          OUT / f"curves_{kind}.png")

    # ---- AUC + detection threshold per method/kind/size -------------------
    auc_table: dict[str, dict[str, dict[int, float]]] = {}
    auc_low_table: dict[str, dict[str, dict[int, float | None]]] = {}
    detection_80: dict[str, dict[str, dict[int, dict | None]]] = {}
    for m in methods:
        auc_table[m] = {}
        auc_low_table[m] = {}
        detection_80[m] = {}
        for kind in ("crater", "boulder"):
            auc_table[m][kind] = {}
            auc_low_table[m][kind] = {}
            detection_80[m][kind] = {}
            for diameter in SIZES_PX[kind]:
                pos = [r["score"] for r in feature_rows
                       if r["method"] == m and r["kind"] == kind
                       and r["diameter_px"] == diameter]
                pos_low = [r["score"] for r in feature_rows
                           if r["method"] == m and r["kind"] == kind
                           and r["diameter_px"] == diameter
                           and r["amplitude_dn"] <= 0.5]
                neg = _neg_scores(m, kind, diameter)
                auc_table[m][kind][diameter] = auc(pos, neg)
                auc_low_table[m][kind][diameter] = (
                    auc(pos_low, neg) if pos_low else None
                )

                by_amp = {}
                for row in summary_rows:
                    if (row["method"] == m and row["kind"] == kind
                            and row["diameter_px"] == diameter):
                        by_amp[row["amplitude_dn"]] = row
                hit = None
                amps = sorted(by_amp)
                for i, a in enumerate(amps):
                    if all(by_amp[b]["detection_rate"] >= 0.8 for b in amps[i:]):
                        hit = by_amp[a]
                        break
                detection_80[m][kind][diameter] = None if hit is None else {
                    "amplitude_dn": hit["amplitude_dn"],
                    "input_snr_model": hit["input_snr_model"],
                    "detection_rate": hit["detection_rate"],
                }

    # ---- control-tile fabrication forensics ------------------------------
    control_summary = {}
    for m in methods:
        control_summary[m] = {
            key: float(np.median(vals)) if vals else None
            for key, vals in control_forensics[m].items()
        }

    # ---- control rows CSV -------------------------------------------------
    with open(OUT / "controls.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "kind", "diameter_px", "tile_row", "tile_col",
                    "threshold", "fp_rate_calibration", "fp_rate_validation",
                    "n_control_scores", "n_validation_scores"])
        for s in control_stats:
            w.writerow([s["method"], s["kind"], s["diameter_px"],
                        s["tile_row"], s["tile_col"], s["threshold"],
                        s["fp_rate_calibration"], s["fp_rate_validation"],
                        s["n_control_scores"], s["n_validation_scores"]])

    control_agg = [
        {"method": m, "kind": kind, "diameter_px": diameter,
         **_control_agg(m, kind, diameter)}
        for m in methods
        for kind in ("crater", "boulder")
        for diameter in SIZES_PX[kind]
    ]
    with open(OUT / "controls_summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(control_agg[0].keys()))
        w.writeheader()
        w.writerows(control_agg)

    # ---- montages (visualization only) -----------------------------------
    def _condition_montage(kind: str, diameter: int) -> None:
        amplitudes_show = [a for a in (0.1, 0.5, 2.0) if a in amplitudes]
        t = tiles[0]
        rows_img = []
        for amplitude in amplitudes_show:
            feats = [InjectedFeature(f.kind, f.row, f.col, f.diameter_px,
                                     amplitude)
                     for f in place_features(
                         (TILE, TILE), kind, diameter, 1.0, REPLICATES,
                         _rng("place", kind, diameter, 0))]
            inj, truth = inject_features(t["dn"], feats)
            norm_inj = _normalize(inj, bounds)
            panels = [
                (np.clip((truth + amplitude) / (2.0 * amplitude), 0, 1) * 255)
                .astype(np.uint8),
                _display_u8(norm_inj),
            ]
            for m in methods:
                out = _apply_method(m, inj, norm_inj, bounds, nm, rt_cfg)
                panels.append(_display_u8(out))
            rows_img.append(np.hstack(panels))
        montage = np.vstack(rows_img)
        Image.fromarray(montage).save(OUT / f"montage_{kind}_d{diameter}.png")

    _condition_montage("crater", 24)
    _condition_montage("boulder", 12)

    # ---- headline ---------------------------------------------------------
    headline = {}
    for m in methods:
        aucs = [auc_table[m][k][d] for k in auc_table[m]
                for d in auc_table[m][k]]
        aucs_low = [auc_low_table[m][k][d] for k in auc_low_table[m]
                    for d in auc_low_table[m][k]
                    if auc_low_table[m][k][d] is not None]
        headline[m] = {
            "auc": auc_table[m],
            "auc_mean": float(np.nanmean(aucs)) if aucs else None,
            "auc_low_amplitude": auc_low_table[m],
            "auc_low_mean": float(np.nanmean(aucs_low)) if aucs_low else None,
            "detection_80": detection_80[m],
            "control_fabrication": control_summary[m],
        }

    # detection comparison LunaPSR vs SSR (paired conditions)
    comparison = {"lunapsr_better": 0, "ssr_better": 0, "equal_or_undefined": 0}
    for kind in ("crater", "boulder"):
        for diameter in SIZES_PX[kind]:
            a = detection_80["lunapsr"][kind][diameter]
            b = detection_80["ssr"][kind][diameter]
            if a is None or b is None:
                comparison["equal_or_undefined"] += 1
            elif a["input_snr_model"] < b["input_snr_model"]:
                comparison["lunapsr_better"] += 1
            elif a["input_snr_model"] > b["input_snr_model"]:
                comparison["ssr_better"] += 1
            else:
                comparison["equal_or_undefined"] += 1

    result = {
        "phase": "5C",
        "title": "Synthetic ground-truth detection benchmark in a real OHRC "
                 "dark region",
        "product": product.path.name,
        "design": {
            "tile_px": TILE,
            "n_tiles": len(tiles),
            "region": REGION,
            "amplitudes_dn": list(amplitudes),
            "sizes_px": {k: list(v) for k, v in SIZES_PX.items()},
            "replicates_per_tile": REPLICATES,
            "features_per_condition": len(tiles) * REPLICATES,
            "fp_rate_target": FP_RATE,
            "control_locations_per_tile": control_per_tile,
            "detector": "generic difference-of-Gaussians, |max| in footprint",
            "methods": methods,
            "bm3d_available": "anscombe_bm3d" in methods,
        },
        "normalization": {
            "method": "percentile", "percentiles": [1.0, 99.0],
            "lower_value": bounds[0], "upper_value": bounds[1],
            "source": "exact full-product histogram",
        },
        "noise_model": nm.to_dict(),
        "tiles": [{k: t[k] for k in (
            "row", "col", "median_dn", "std_dn", "p99_dn", "model_sigma_dn",
            "model_snr", "measured_sigma_dn", "measured_over_model",
            "n_flat_windows")} for t in tiles],
        "control_calibration": control_agg,
        "control_fabrication": control_summary,
        "headline": headline,
        "lunapsr_vs_ssr_detection_80": comparison,
        "summary_rows": summary_rows,
        "selection_note": (
            "Detection threshold calibrated on feature-free control tiles at "
            "5% false-positive rate; tiles selected for low texture and "
            "noise-dominated statistics in one dark region."
        ),
        "caveats": [
            "Injected profiles are idealised analytic shapes, not real lunar "
            "photometry.",
            "Sub-DN amplitudes are a continuous-signal proxy: the delivered "
            "product is uint8-quantised.",
            "All tiles come from one dark region of one product; no terrain "
            "diversity is measured here.",
            "The generic DoG detector is one choice; absolute rates depend on "
            "the detector, relative ordering is the informative result.",
            "Anscombe stabilization uses the INFERRED Phase 4B gain.",
            "LunaPSR produces a DERIVED REPRESENTATION; nothing here proves "
            "geological reality or recovers unmeasured photons.",
            "PNG montages are visualization only (per-image 1-99.5% "
            "stretch); quantitative results are in summary.csv/metrics.json.",
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(
        json.dumps(_clean(result), indent=2, default=str), encoding="utf-8")

    # ---- experiment record -------------------------------------------------
    rec = ExperimentRecord(
        product=product.path.name,
        method="phase5c_ground_truth_benchmark",
        parameters={"design": result["design"],
                    "methods": methods,
                    "lunapsr_config": rt_cfg.to_dict()},
        normalization=result["normalization"],
        metrics={"headline": headline,
                 "lunapsr_vs_ssr": comparison},
        analysis={"tiles": result["tiles"],
                  "selection_note": result["selection_note"],
                  "caveats": result["caveats"]},
        noise_model=nm.to_dict(),
        notes=("Phase 5C: synthetic crater/boulder injection into real OHRC "
               "dark tiles; fixed 5% false-positive detection benchmark. "
               "Derived representation only."),
    )
    (OUT / "experiment.json").write_text(rec.to_json(), encoding="utf-8")

    print(f"done in {result['elapsed_s']}s -> {OUT}", flush=True)
    for m in methods:
        print(f"{m:16s} AUC mean={headline[m]['auc_mean']:.3f} "
              f"ctrl new-edge={control_summary[m]['new_edge_fraction_of_ssr_edges']} "
              f"isolated={control_summary[m]['isolated_new_components']}",
              flush=True)
    print("LunaPSR vs SSR detection@80%:", comparison, flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="reduced smoke run (2 tiles, 3 amplitudes)")
    args = parser.parse_args()
    raise SystemExit(main(quick=args.quick))

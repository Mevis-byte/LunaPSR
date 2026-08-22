"""Phase 4A read-only statistical investigation of the actual OHRC product.

Never writes to the product tree.  The scientific array is memory-mapped and
processed in chunks; all results are derived outputs written to
outputs/phase4a/ (or printed when --json is not given).

Usage:
    python analysis/phase4a_investigate.py
    python analysis/phase4a_investigate.py --json outputs/phase4a/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCT = PROJECT_ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18"
IMG = PRODUCT / "data" / "calibrated" / "20260103" / (
    "ch2_ohr_ncp_20260103T1005176450_d_img_d18.img"
)

LINES, SAMPLES = 101075, 12000
CHUNK = 2000  # lines processed at once (~24 MB uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write derived results JSON to this path")
    ap.add_argument("--max-lines", type=int, default=None,
                    help="investigate only the first N lines (debugging)")
    args = ap.parse_args()

    if not IMG.is_file():
        print(f"product .img not found: {IMG}")
        return 1
    expected = LINES * SAMPLES
    if IMG.stat().st_size != expected:
        print(f"size mismatch: {IMG.stat().st_size} != {expected}")
        return 1

    results: dict = {}

    # ---------- 1. exact full-array histogram + global stats ----------
    hist = np.zeros(256, dtype=np.int64)
    n_lines = LINES if args.max_lines is None else args.max_lines
    mm = np.memmap(str(IMG), dtype=np.uint8, mode="r", shape=(LINES, SAMPLES))
    for lo in range(0, n_lines, CHUNK):
        block = mm[lo:lo + CHUNK]
        hist += np.bincount(block.ravel(), minlength=256)
    hist = hist.astype(np.int64)
    total = int(hist.sum())
    print(f"lines investigated: {n_lines} / {LINES}  pixels: {total:,}")

    counts = hist.astype(np.float64)
    cum = np.cumsum(counts)
    def pct(p: float) -> float:
        target = p / 100.0 * total
        return float(np.searchsorted(cum, target))

    mean = float((counts * np.arange(256)).sum() / total)
    mean_sq = float((counts * np.arange(256) ** 2).sum() / total)
    std = float(np.sqrt(max(mean_sq - mean ** 2, 0.0)))
    med = pct(50)
    q1, q99 = pct(1), pct(99)
    zero_frac = float(hist[0] / total)
    sat_frac = float(hist[255] / total)
    dark5 = float(hist[0:5].sum() / total)
    dark10 = float(hist[0:10].sum() / total)

    nonzero = np.nonzero(hist)[0]
    results["global"] = {
        "lines": n_lines,
        "samples": SAMPLES,
        "pixels": total,
        "min": int(nonzero.min()),
        "max": int(nonzero.max()),
        "mean": mean,
        "std": std,
        "median": med,
        "p01": q1,
        "p99": q99,
        "zero_fraction": zero_frac,
        "saturation_fraction": sat_frac,
        "fraction_lt_5": dark5,
        "fraction_lt_10": dark10,
        "nonzero_min": 0 if hist[0] == 0 else 1,
    }
    print(json.dumps(results["global"], indent=2))
    top = np.argsort(hist)[::-1][:6]
    print("most common values:", [(int(v), int(hist[v])) for v in top])

    # ---------- 2. row / column statistics (sampled) ----------
    step = 50
    row_idx = np.arange(0, n_lines, step)
    col_idx = np.arange(0, SAMPLES, step)
    row_means = np.empty(row_idx.size, dtype=np.float64)
    row_stds = np.empty(row_idx.size, dtype=np.float64)
    row_zero = np.empty(row_idx.size, dtype=np.float64)
    col_means = np.empty(col_idx.size, dtype=np.float64)
    col_stds = np.empty(col_idx.size, dtype=np.float64)
    col_zero = np.empty(col_idx.size, dtype=np.float64)
    for k, r in enumerate(row_idx):
        line = mm[r]
        row_means[k] = line.mean()
        row_stds[k] = line.std()
        row_zero[k] = (line == 0).mean()
    for k, c in enumerate(col_idx):
        col = mm[:, c]
        col_means[k] = col.mean()
        col_stds[k] = col.std()
        col_zero[k] = (col == 0).mean()

    results["rows"] = {
        "step": step,
        "mean_of_row_means": float(row_means.mean()),
        "std_of_row_means": float(row_means.std()),
        "row_mean_range": [float(row_means.min()), float(row_means.max())],
        "mean_of_row_stds": float(row_stds.mean()),
        "std_of_row_stds": float(row_stds.std()),
        "mean_row_zero_fraction": float(row_zero.mean()),
        "rows_with_any_zero_fraction_gt_50pct": int((row_zero > 0.5).sum()),
    }
    results["columns"] = {
        "step": step,
        "mean_of_col_means": float(col_means.mean()),
        "std_of_col_means": float(col_means.std()),
        "col_mean_range": [float(col_means.min()), float(col_means.max())],
        "mean_of_col_stds": float(col_stds.mean()),
        "std_of_col_stds": float(col_stds.std()),
        "mean_col_zero_fraction": float(col_zero.mean()),
        "cols_with_zero_fraction_gt_50pct": int((col_zero > 0.5).sum()),
        "cols_with_std_lt_1": int((col_stds < 1.0).sum()),
    }
    print("rows:", json.dumps(results["rows"], indent=2))
    print("columns:", json.dumps(results["columns"], indent=2))

    # ---------- 3. representative patches ----------
    # dark patch = lowest row-mean window; bright patch = highest row-mean window
    P = 200
    row_win = np.convolve(row_means, np.ones(4) / 4, mode="valid")
    dark_r = int(row_idx[int(np.argmin(row_win))])
    bright_r = int(row_idx[int(np.argmax(row_win))])
    # also sample a nominal mid-signal patch
    mid_r = int(row_idx[int(np.argmin(np.abs(row_means - float(np.median(row_means)))))])
    c_mid = SAMPLES // 2

    def patch_stats(r: int, c: int) -> dict:
        p = mm[r:r + P, c:c + P].astype(np.int16)
        rowdiff = float(np.abs(np.diff(p, axis=1)).mean())
        coldiff = float(np.abs(np.diff(p, axis=0)).mean())
        return {
            "row0": r, "col0": c,
            "mean": float(p.mean()),
            "std": float(p.std()),
            "median": float(np.median(p)),
            "min": int(p.min()),
            "max": int(p.max()),
            "zero_fraction": float((p == 0).mean()),
            "sat_fraction": float((p == 255).mean()),
            "mean_abs_row_diff": rowdiff,
            "mean_abs_col_diff": coldiff,
            "adjacent_diff_p99": float(np.percentile(np.abs(np.diff(p.astype(int), axis=1)), 99)),
        }

    results["patches"] = {
        "patch_size": P,
        "dark": patch_stats(dark_r, c_mid),
        "mid": patch_stats(mid_r, c_mid),
        "bright": patch_stats(bright_r, c_mid),
    }
    print("patches:", json.dumps(results["patches"], indent=2))

    # ---------- 4. spatial noise proxies on sampled lines ----------
    # local std of 7x7 windows on a strided subset (reads ~1% of the image)
    sstep = 20
    lstd_acc = 0.0
    lstd_n = 0
    flat_pool = []
    for r in range(0, n_lines, sstep * 7):
        blk = mm[r:r + 7, ::sstep].astype(np.float32)
        if blk.shape[0] < 7:
            break
        mu = blk.mean(axis=0, keepdims=True)
        lstd = np.sqrt(((blk - mu) ** 2).mean(axis=0))
        lstd_acc += float(lstd.mean())
        lstd_n += 1
        # flat windows: consecutive-sample smoothness within row centre
        row = mm[r + 3, :: sstep].astype(np.int32)
        diff = np.abs(np.diff(row))
        flat_mask = diff <= 1
        if flat_mask.size:
            flat_pool.append((row[1:][flat_mask]))
    lstd_mean = lstd_acc / max(lstd_n, 1)
    flat = np.concatenate(flat_pool) if flat_pool else np.array([], dtype=np.int32)
    results["noise_proxy"] = {
        "local_std_7x7_mean_sampled": float(lstd_mean),
        "flat_adjacent_pair_count": int(flat.size),
        "flat_pair_std": float(flat.std()) if flat.size else None,
        "flat_pair_mean": float(flat.mean()) if flat.size else None,
        "flat_pair_p99_absdiff": float(
            np.percentile(np.abs(np.diff(flat.astype(int))), 99)
        ) if flat.size else None,
    }
    print("noise_proxy:", json.dumps(results["noise_proxy"], indent=2))

    # ---------- 5. quantization / bit-depth signature ----------
    # even/odd balance and value-multiplicity of common values
    results["quantization"] = {
        "fraction_even": float(hist[0::2].sum() / total),
        "fraction_odd": float(hist[1::2].sum() / total),
        "distinct_values": int((hist > 0).sum()),
        "values_with_zero_count": int((hist == 0).sum()),
    }
    print("quantization:", json.dumps(results["quantization"], indent=2))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
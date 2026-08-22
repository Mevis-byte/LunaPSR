# LunaPSR Phase 3 — OHRC Scientific Preprocessing + Low-Light Baseline Enhancement

Phase 3 turns the Phase 2 OHRC reader into a **scientifically careful
preprocessing and benchmarking pipeline**.  It does **not** build the final
research model, add PSR segmentation, ShadowCam comparison, super-resolution
or any generative AI.

> **LunaPSR currently treats enhancement as a derived visualization/analysis
> product and does not claim that enhanced pixels represent newly recovered
> physical measurements.**
> *The original scientific array is never modified, and every derived output
> is labelled as derived.*

Pipeline:

```
REAL OHRC PRODUCT
        ↓
OHRC READER            (Phase 2, read-only memory-map)
        ↓
SCIENTIFIC IMAGE       (101075 × 12000 px, UnsignedByte, 1.2 Gpx)
        ↓
VALID PIXEL MASK
        ↓
ROBUST NORMALIZATION   (percentile-based, reproducible)
        ↓
NOISE ANALYSIS
        ↓
BASELINE ENHANCEMENT
        ↓
SIDE-BY-SIDE COMPARISON / DIFFERENCE / BEFORE-AFTER
        ↓
METRICS
        ↓
EXPERIMENT RECORD      (outputs/, separate from the product)
```

## 1. OHRC data source

The real Chandrayaan-2 OHRC product already lives in the project root:

```
ch2_ohr_ncp_20260103T1005176450_d_img_d18/
    browse/  data/  geometry/  miscellaneous/
```

- It is used **in place** — LunaPSR never uploads, copies, moves, renames or
  modifies it.  Loading is read-only (memory-mapped).
- All paths are project-relative (`configs.defaults.PROJECT_ROOT`), never a
  Windows or WSL absolute path.
- Actual product (verified by the Phase 3 tests):

| Property       | Value                              |
|----------------|------------------------------------|
| Lines (rows)   | 101 075                            |
| Samples (cols) | 12 000                             |
| Pixels         | 1 212 900 000 (1.2 Gpx)            |
| Datatype       | `UnsignedByte` (8-bit, unsigned)   |
| Byte order     | single-byte (`MSB` not applicable) |
| File size      | 1 212 900 000 bytes                |

## 2. Product discovery (`core/ohrc/discovery.py`)

- `find_ohrc_products(root=PROJECT_ROOT)` scans the project tree for bundles
  with an OHRC science footprint (`data`/`browse`/`geometry` folders or an
  `.img`), then **validates each candidate with the Phase 2 reader**.
- Each candidate gets a verdict: `VALID` / `INVALID` (+ reason) / `DETECTED`.
- The single VALID product is auto-selected; extra products can be validated
  interactively (path input).  No upload UI is required.
- Discovery never touches a product's files (regression-tested).

## 3. Original data is immutable

All processing works on **derived** arrays:

```
original_data (memmap)
     ├──> base = block-average analysis representation (float32, size-reduced)
     ├──> valid_mask
     ├──> normalized = robust normalization of base (invalid → NaN)
     └──> enhanced  = baseline applied to normalized
```

`original_data` is never written.  Tests assert the source is byte-identical
after every enhancement.  The 1.2 Gpx product is memory-mapped, so analyses
use a block-averaged representation (default factor 10 → ~12 Mpx float32).

## 4. Valid pixel mask (`core/preprocessing/mask.py`)

`build_valid_mask(data, metadata, ...)` returns `(valid_mask, details)`:

- NaN / +Inf / −Inf (float arrays);
- values equal to **label-declared** special constants
  (`missing_constant` / `invalid_constant` / `saturated_constant`, now
  parsed from PDS4 `Special_Constants` into `metadata.array.special_constants`);
- an explicit `valid_range` when supplied by the caller.

**Nothing is guessed.**  The current product declares no special constants, so
the mask marks every pixel valid and says so (`nodata_notice`).  The `details`
dict records exact counts per reason, the sentinel list actually used, and the
source of each rule.  The mask never mutates its input.

## 5. Scientific normalization (`core/preprocessing/normalization.py`)

`normalize_scientific(data, percentiles=(1, 99), mask=..., precomputed_bounds=...)`
returns `(normalized_image, normalization_parameters)`:

```json
{
  "method": "percentile",
  "lower_percentile": 1.0,
  "upper_percentile": 99.0,
  "lower_value": 4.0,
  "upper_value": 251.0
}
```

- Percentile bounds (default 1/99) come from the **exact full-array
  histogram**, so extremes/hot pixels do not dominate.
- Invalid pixels become NaN in the derived array so statistics can exclude them.
- The input is never modified.

**Normalization ≠ enhancement.**  Normalization only maps scientific values to
a useful numeric range; enhancement (below) increases visibility.  The UI and
code keep these strictly separate.

## 6. Baseline methods (`core/enhancement/baseline.py`)

Classical, interpretable baselines on the derived [0,1] representation:

| Method                   | Notes                                                        |
|--------------------------|--------------------------------------------------------------|
| Gamma correction         | `out = x**gamma`, gamma < 1 brightens dark pixels            |
| Histogram equalization   | global equalisation                                          |
| CLAHE                    | contrast-limited adaptive (clip limit + tile size)           |
| Unsharp masking          | conservative `x + strength·(x − gauss(x))`                   |
| Denoising (NLM)          | OpenCV non-local means, conservative strength                |
| Retinex (single-scale)   | benchmark; known to amplify shadow noise (flagged)           |

All operate on derived arrays only.  No method generates geological structure
that was not in the input.

## 7. Zero-DCE baseline (`core/enhancement/zero_dce.py`)

Optional, experimental, isolated.  The module never imports a DL backend at
module scope.  When the backend is missing the app reports

> **“Zero-DCE unavailable: dependency not installed”**

and every classical method keeps working.  Zero-DCE is **not** assumed to be
superior; it exists for comparison.

## 8. Retinex baseline (`core/enhancement/retinex.py`)

Single-scale Retinex as a benchmark.  Documented weakness: noise amplification
in the darkest pixels (the scene is PSR-limited lighting); results are flagged
in the UI.  Not claimed to be better than the other baselines.

## 9. No hallucination

- No generative models, diffusion, or super-resolution in Phase 3.
- The difference view is labelled **“Enhancement Difference”** and is *never*
  interpreted as newly discovered information.

## 10. Noise analysis (`core/analysis/noise.py`)

`analyze_noise()` reports mean / median / std / local std / coefficient of
variation / background (dark-floor) variation / noise sigma proxy.

**SNR is deliberately reported as “Not yet estimated”** (`snr=None`) because a
defensible signal/noise definition for OHRC requires a calibrated sensor noise
model that is not part of this product.  Underlying statistics are shown
instead.

## 11. No-reference metrics (`core/analysis/metrics.py`)

`quality_metrics()` on the derived [0,1] representation: entropy, robust
contrast, gradient magnitude, edge density, local variance, noise sigma,
intensity-distribution summary.

- Metrics require **no ground-truth image**.
- **PSNR / SSIM against the original are deliberately NOT computed**: the
  original low-light image is not ground truth; bigger difference ≠ better.
- Noise remains a proxy (no calibrated sensor model).

## 12–14. UI (`app/pages/ohrc_lab.py`, “🔬 OHRC Lab” tab)

Side-by-side normalized-original vs enhanced; interactive before/after wipe
slider; **Enhancement Difference** (`|enhanced − normalized|`); product,
method and parameter transparency; scientific-statistics panel (exact, 1.2 Gpx
histogram); noise panel; metrics table (original vs enhanced).

## 15. Region of interest

Rectangular ROI on the analysis representation with ROI width / height, mean,
median, std, local contrast, local noise estimate, local variance.  (No
automatic PSR segmentation in Phase 3.)

## 16. Reproducibility / experiment tracking (`core/analysis/experiment.py`)

Every experiment saves a JSON record:

```json
{
  "product": "ch2_ohr_..._d_img_d18",
  "method": "clahe",
  "parameters": {"clip_limit": 2.0, "tile_size": 8},
  "normalization": {"lower_percentile": 1.0, "upper_percentile": 99.0,
                    "lower_value": 4.0, "upper_value": 251.0},
  "metrics": {"original": {...}, "enhanced": {...}},
  "timestamp": "..."
}
```

Derived outputs go to `outputs/`:

```
outputs/
    experiments/    JSON experiment records
    previews/       derived 8-bit visualization PNGs
    metrics/        per-run metric JSON
```

The OHRC product tree is never written to.

## 17. Export

- 8-bit PNG and 16-bit TIFF of the *derived* enhanced result (clearly
  identified as derived visualizations).
- Difference PNG, metrics JSON, original-display PNG.
- Raw scientific DN values are only available from the untouched product files.

## 18. Performance

- Product loaded once per session (memory-map); histograms/analyses cached per
  product + parameters in session state.
- Changing percentile / method / parameters reuses cached base + histogram.
- Full-resolution arrays are never copied; analysis uses block-averaged
  representations.

## 19. Testing

`tests/` additions (all pass; real-product tests skip when the bundle is
absent):

- `test_ohrc_discovery.py` — discovery, single-default selection, invalids,
  nesting, read-only guarantee.
- `test_mask.py` — NaN/Inf, label-declared sentinels, explicit nodata/range,
  no-guess notice, copy/immutability.
- `test_phase3_pipeline.py` — normalization reproducibility + immutability,
  every classical baseline, Zero-DCE graceful degradation, noise diagnostics,
  no-reference metrics, experiment records, and real-product end-to-end
  “original unchanged after enhancement”.

## 20. Limitations

- Noise sigma and CV are engineering proxies (no calibrated sensor model);
  SNR is explicitly “Not yet estimated”.
- Analysis happens on a size-reduced (block-averaged) representation for RAM;
  full-precision statistics come from the exact full-array histogram.
- No nodata sentinel is declared in the current product, so the mask is
  all-valid — it stays honest rather than guessing.
- Zero-DCE requires a deep-learning backend at runtime; without it the app
  degrades gracefully.
- Enhancement results are derived visualization/analysis products, not new
  physical measurements.

## Definition of done

- [x] Automatic discovery of the extracted product (no upload).
- [x] Actual OHRC scientific data used; original never modified.
- [x] Valid-pixel masking, robust normalization, classical baselines,
      optional isolated Zero-DCE, Retinex, noise analysis, metrics,
      comparison/difference/before-after, ROI, experiment records, derived
      outputs under `outputs/`.
- [x] Phase 2 and Phase 1 functionality still passes.
- [x] Tests pass; application runs with the real product.

## Recommended next step (Phase 4)

With the benchmark harness in place, the defensible next step is **calibrated
noise/SNR estimation** (or a documented radiometric model) before any learned
enhancement — so Phase 4 can compare learned methods against disentangled,
physics-aware baselines without claiming superiority.
# LunaPSR — Architecture

This document describes how LunaPSR is wired together, the data conventions
used throughout, and the design decisions that matter when extending it.

## 1. Data conventions

All algorithms operate on **grayscale float32 in the source's native range**,
wrapped in `core.types.ImageData`:

- `data` — 2-D `float32` array, original numerical values (e.g. 0–255 for an
  8-bit browse PNG, 0–65535 for 16-bit TIFF, arbitrary range for float TIFF).
- `bit_depth` — 8 / 16 / 32, used only to report dynamic range, never to
  claim calibration.
- `metadata` — parsed OHRC filename tokens (scene id, time, mode, product),
  kept alongside the pixels and carried into export files (`core.io`).

Two normalisation conventions exist and must not be confused:

| Convention | Where | Range |
|---|---|---|
| `display` normalisation | UI previews, exported 8-bit PNG | robust percentiles (0.5, 99.5) → [0,1] |
| `pipeline` normalisation | stage 1 of every pipeline | the same robust percentiles, [0,1], *fixed* so diff/metrics are meaningful |

`robust_normalize` (`core/preprocessing/normalization.py`) uses percentile
clamping, so a few saturated / dark pixels do not stretch the whole strip.

## 2. Data flow

```
sidebar / upload / sample
        │  core.io.load_image  →  ImageData  →  app.ui.state.set_image
        ▼
Enhancement tab: build stage dict  →  core.pipeline.EnhancementPipeline
        │
        ▼
app.processing.run_enhancement   (cached: _run_pipeline_cached by params fingerprint)
        │
        ├─ pipeline.run(image)     core/… stages
        ├─ compute_metrics         core/metrics + core/validation checks
        ├─ confidence proxy + warnings
        │
        ▼
EnhancementRun { enhanced, normalized_original, confidence, metrics, records, warnings }
        │
        ├─ stored in st.session_state["enhanced"]  (single source of truth)
        ├─ one ProcessingRecord persisted to data/history/<id>.json
        └─ exports (PNG / 16-bit TIFF / difference / confidence / JSON) built on demand
```

Other tabs read `get_enhanced()` from session state and never re-run the
pipeline. `core.history.ProcessingHistory` is the persistent record store;
every record has a stable `processing_id` (e.g. `PSR-2026-00014`).

## 3. Reproducibility

Each `ProcessingRecord` captures:

- pipeline name + every stage parameter + parameter overrides
- input source name / shape / bit depth
- original & enhanced metric dictionaries
- timestamp and validation warnings

The History tab can export a record as JSON, dump the full archive, or **load a
record's parameters back into the Enhancement tab** (one click, then Apply).
This makes "tune → evaluate → revisit" cheap and auditable.

## 4. Quality metrics and validation

`core/metrics/core.py` computes, on [0,1] data: robust contrast, 8-bit entropy,
Laplacian sharpness, edge preservation index (Sobel NCC), and the standard
metrics block (SNR, dynamic range proxies, mean/std) with per-metric caveats.

`core/validation` runs cheap *sanity* checks on every run and raises scientific
warnings rather than blocking:

- fraction of newly saturated / clipped pixels
- `noise_amplification_fraction` — how much local std grew on non-edge pixels
- `core.psr` provides the PSR-aware interface (manual ROI in v1; geometry-based
  detection is a roadmap item, not fabricated from pixel stats).

**Important**: SNR, noise maps and confidence values are engineering proxies.
There is no calibrated OHRC sensor-noise model in the public browse product, so
these figures are labelled "estimate" in the UI and must not be reported as
measurements (see `configs/defaults.py` warning copy).

## 5. Caching strategy

- `_run_pipeline_cached` — `@st.cache_data` keyed by the pipeline parameter
  fingerprint; identical enhancement requests are never recomputed.
- Exports — expensive 9 MP PNG / TIFF encodes happen only when the user asks,
  then the bytes are cached in session state per processing id.
- Noise analysis (`app/pages/noise.py`) — cached in session state keyed by
  `(image revision, σ, edge percentile)`.
- Any image change (`ImageData`, revision, original path/size) invalidates the
  dependent caches; `get_enhanced()` is cleared when the input changes.

## 6. Module map (quick reference)

```
app.py                     entry point; sidebar image source; tab assembly
app.pages.enhancement      the main workflow: controls → apply → exports
app.pages.comparison       original / enhanced / difference (auto + ROI)
app.pages.metrics          full-image + ROI metrics, ROI editor
app.pages.noise            signal/residual/noise visualisation
app.pages.history          record archive, export, re-load parameters
app.processing             EnhancementRun; pipeline build; orchestration
app.exporters              bytes for PNG / TIFF16 / diff / colormap / JSON
app.ui.state               session-state accessors (image, enhanced, ROI, key)
app.ui.rendering           preview() display-stretch, colorize()
core.registry              stage name → callable map; parameter schemas
core.pipeline              staged EnhancementPipeline engine
core.enhancement / .denoising / .preprocessing   stage implementations
core.metrics / core.noise  metrics + noise estimation
core.validation            header sanity checks → warnings
core.history               persisted ProcessingRecord store
models.ai.base             EnhancementModel ABC + registry/get_model
models.classical           gamma/clahe/retinex/pipeline as models
models.ai                  Retinexformer / Zero-DCE stubs (Phase 4)
```

## 7. Extension points

- **New stage** → implement a callable taking `(image: np.ndarray, **params)`
  returning a `np.ndarray`, register the schema in `core.registry`, add
  defaults + label in `configs/defaults.py`.
- **New metric** → add to `core/metrics/core.py`; it appears in the metrics
  blocks automatically if returned by `compute_metrics`.
- **Learned model** → subclass `EnhancementModel` and `register_model`; the
  classical models already demonstrate the interface. Loading stays lazy so no
  weights are required to run the app.
- **New export** → extend `app/exporters.py` and expose a download button.
# LunaPSR

A software research project for low-light enhancement and analysis of
**Permanently Shadowed Regions (PSRs)** of the Moon using **Chandrayaan-2 OHRC**
(Orbiter High Resolution Camera) imagery.

> **Scientific disclaimer (read this first).** LunaPSR is an image-enhancement
> and *analysis* system. It does **not** recover unmeasured photons, does not
> reconstruct ground-truth lunar reflectance, does not prove that enhanced
> structures are geological, and does not create new physical measurements.
> No enhancement method can guarantee recovery of features hidden below the
> sensor's information/noise limits. The goal is to improve analytical
> accessibility of the weak OHRC signal while controlling noise amplification
> and enhancement artifacts. See [Scientific Disclaimer](#15-scientific-disclaimer).

---

## Table of contents

1. [Overview](#1-overview)
2. [Problem](#2-problem)
3. [Project objective](#3-project-objective)
4. [Current architecture](#4-current-architecture)
5. [Processing pipeline](#5-processing-pipeline)
6. [Scientific approach](#6-scientific-approach)
7. [Current development phases](#7-current-development-phases)
8. [Installation](#8-installation)
9. [OHRC dataset setup](#9-ohrc-dataset-setup)
10. [Running the application](#10-running-the-application)
11. [Running tests](#11-running-tests)
12. [Current results](#12-current-results)
13. [Limitations](#13-limitations)
14. [Roadmap](#14-roadmap)
15. [Scientific disclaimer](#15-scientific-disclaimer)

---

## 1. Overview

LunaPSR is a Python/Streamlit application plus an analysis codebase that loads a
real Chandrayaan-2 OHRC product (~1.2 billion pixels), builds scientifically
careful derived representations, models the camera's photon-noise physics, and
studies — honestly and quantitatively — what classical low-light enhancement
methods actually do to extremely dark polar terrain.

Every stage is read-only with respect to the source data: the original
scientific array is memory-mapped and never modified; all outputs are derived
representations written under `outputs/`.

## 2. Problem

Permanently shadowed regions near the lunar poles are imaged by OHRC at very
low signal levels. In the studied region the raw data occupy only a handful of
DN levels, so:

* ordinary display stretching shows mostly sensor noise;
* aggressive enhancement (e.g. Retinex) can manufacture visually convincing
  "detail" that is dominated by amplified read/quantization noise;
* without a sensor noise model there is no defensible way to say what is
  signal and what is noise.

## 3. Project objective

Build a reproducible, auditable pipeline that:

1. ingests real OHRC products with provenance and integrity checks;
2. derives calibrated-domain representations and a **physics-based noise
   model** for the OHRC sensor;
3. evaluates classical enhancement baselines under one identical evaluation
   framework;
4. quantifies whether "new detail" created by an enhancer is consistent with
   recovered structure or with amplified noise;
5. prototypies adaptive enhancement gated by that noise model.

## 4. Current architecture

```
app.py                     Streamlit entry point (tabs below)
app/
  pages/                   ohrc (product), ohrc_lab (enhancement lab), enhancement,
                           comparison, metrics, noise, history, about
  ui/                      rendering helpers, state wiring
configs/defaults.py        centralised defaults
core/
  io.py types.py           image I/O + shared types
  ohrc/                    product discovery/validation/reader (real ISRO format),
                           statistics (exact histogram, percentiles)
  preprocessing/           robust percentile normalization, valid-pixel masks
  enhancement/             gamma, CLAHE, Retinex (SSR), denoising glue,
                           lunapsr.py            Phase 5 adaptive enhancer,
                           lunapsr_retinex.py    Phase 5B adaptive Retinex prototype,
                           confidence.py         model-based local confidence/SNR maps,
                           signal_estimation.py  robust local signal/illumination/structure
  denoising/               bilateral, Gaussian, NLM
  metrics/                 quality metrics (contrast, entropy, EPI, ...)
  noise/                   residual/noise-map analysis
  validation/              saturation / clipping / amplification checks
  analysis/                experiment records, SSR feature forensics,
                           ohrc_noise_model.py   Phase 4B physics noise model
models/                    classical enhancers + learned-model stubs (Zero-DCE)
analysis/                  phase runners (reproducible scripts per phase)
tests/                     pytest suite (284 tests at this checkpoint)
docs/                      phase reports & findings
outputs/                   ALL derived outputs (gitignored bulk arrays excluded)
```

## 5. Processing pipeline

```
real OHRC product (read-only memmap)
  → discovery / integrity validation
  → scientific image (uint8 DN, ~101k × 12k px)
  → valid-pixel mask
  → robust percentile normalization (reproducible bounds)
  → noise analysis (residual-based)  +  physics noise model (Phase 4B)
  → enhancement (classical baselines | LunaPSR | LunaPSR Adaptive Retinex)
  → side-by-side / difference / before-after views
  → metrics + artifact forensics (new edges, isolated components, EPI, HF energy)
  → experiment record (JSON) + derived outputs under outputs/
```

## 6. Scientific approach

* **Radiometry first.** Phase 4A established the OHRC DN→electron mapping
  context from the product's own metadata; the gain used by the noise model is
  *inferred*, never claimed as an official published value.
* **Physics noise model.** Phase 4B combines shot noise, read noise and
  quantization noise into σ(DN) = √(DN/g + (σ_read/g)² + 1/12), validated
  against flat dark windows in the real product.
* **Model-based confidence.** SNR values are computed *from the model*
  ("MODEL-BASED SNR") — they are estimates, not measurements.
* **Enhancement forensics.** New-edge fraction, isolated new components, edge
  preservation index, orientation agreement and high-frequency energy measure
  what an enhancer *adds*, enabling the key distinction between recovered
  structure and amplified noise.
* **Honest language throughout**: "derived representation", "consistent with
  modeled noise", never "measured" or "recovered".

## 7. Current development phases

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Image-enhancement dashboard on sample imagery | COMPLETE |
| Phase 2 | Real OHRC product integration (discovery, reader, integrity) | COMPLETE |
| Phase 3 | Enhancement lab: normalization, baselines, benchmarking | COMPLETE |
| Phase 4A | OHRC radiometry & measured noise characterization | COMPLETE |
| Phase 4B | Physics-based OHRC noise model (validated) | COMPLETE |
| Phase 5A | SSR forensic investigation on a noise-limited ROI | COMPLETE |
| Phase 5B | LunaPSR Adaptive Retinex (noise-aware gating) | IN PROGRESS |

Phase 5B note: implementation exists in this repository (`core/enhancement/lunapsr_retinex.py`,
tests, lab UI section, `docs/phase5b_lunapsr_retinex.md`) but has not yet been
formally reviewed/delivered as a completed phase; treat its results as provisional.

## 8. Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies include numpy, scipy, opencv-python-headless, scikit-image, Pillow,
pandas, pytest and streamlit (see `requirements.txt`).

## 9. OHRC dataset setup

The real Chandrayaan-2 OHRC dataset is **not included in this repository**
(~1.2 billion pixels). It must be obtained through the official ISRO
channels:

* **PRADAN portal (ISRO Science Data Archive / ISSDC), Chandrayaan-2 section:**
  <https://pradan.issdc.gov.in/ch2>
  Free registration/login ("Login/Signup", top-right) is required before data
  can be downloaded.
* **MapBrowse application** (search OHRC footprints on a lunar map and download
  calibrated products): <https://chmapbrowse.issdc.gov.in>
  For polar PSR strips: change the projection to *South Pole*, enable
  *Instrument footprint → CH2_OHR_Calibrated_Product*, and click an OHRC
  footprint to reach its product download.

To use the app with real data, place an OHRC product folder **anywhere under
the project root, up to three directory levels deep**, e.g.:

```
Project Moon/
└── ch2_ohr_ncp_20260103T1005176450_d_img_d18/
    └── ch2_ohr_ncp_20260103T1005176450_d_img_d18.img   (scientific image)
```

The discovery implementation (`core/ohrc/discovery.py`) automatically scans the
project root, identifies candidate product directories by the presence of an
`.img` file (or standard OHRC signature sub-directories), validates them, and
offers them in the app's product selector. The dataset directory is gitignored
and must remain local.

Without a local dataset the app still runs on the bundled sample image
(`assets/sample_ohrc.png`).

## 10. Running the application

```bash
streamlit run app.py
```

Tabs cover: product selection & validation, the OHRC Enhancement Lab
(normalization → noise analysis → model-based SNR → baselines → LunaPSR →
diagnostics), classic enhancement pipelines, comparison views, quality metrics,
noise analysis, processing history and documentation notes.

## 11. Running tests

```bash
pytest tests/ -q
# current baseline: 284 passed
```

The suite includes end-to-end tests that run against the real OHRC product when
it is present locally (they skip automatically otherwise).

## 12. Current results

### Phase 5A — what Retinex really does to a 5-DN region

Deep investigation of a 200×200-pixel ROI (lines 61750–61949 × samples
6000–6199) whose raw signal spans only five DN levels (mean 3.056, median 3,
min 1, max 5, std 0.467):

* The Phase 4B model puts total modeled noise at ≈ 0.511 DN there, i.e.
  **model-based SNR ≈ 5.99** (gain inferred, not official; SNR is model-based,
  NOT measured).
* Every analyzed window's variation was *consistent with the modeled noise*
  — the ROI contains essentially no demonstrated spatial structure.
* Standard single-scale Retinex (SSR) made the region look far more detailed,
  but forensics showed: **31.4 % of SSR edge pixels were new**, **391 isolated
  new components** appeared, only **40.7 %** of original edge pixels were
  strengthened, **EPI = 0.827**, orientation agreement 0.804, and high-frequency
  energy rose ≈ 18 %.
* Conclusion: in this ROI the observed SSR structure is **dominated by
  amplified read/quantization noise texture rather than demonstrated recovered
  lunar structure**.

Full report: `docs/phase5a_ssr_investigation.md` (plus `docs/phase3.md`,
`docs/phase4a_ohrc_radiometry_noise.md`, `docs/phase4b_noise_model.md`,
`docs/architecture.md`, `docs/research-roadmap.md`).

### Provisional Phase 5B direction

Motivated by 5A, `core/enhancement/lunapsr_retinex.py` prototypes a Retinex
variant whose strength adapts per pixel to model-based confidence (signal level
AND variation evidence above the modeled noise), staying conservative where
variation is noise-consistent. See `docs/phase5b_lunapsr_retinex.md`; results
are provisional pending formal review.

## 13. Limitations

* No radiometric calibration beyond the product's own count-calibration;
  the noise-model gain is inferred, not an official value.
* All SNR/confidence numbers are model-based estimates.
* Quality metrics are no-reference diagnostics on derived data.
* Deep behavioral validation is currently concentrated on one ROI plus
  synthetic scenes; other terrain types may behave differently.
* Classical methods only for delivered phases; learned-model support (Zero-DCE)
  is optional/stubbed.
* Nothing here replaces photometric reconstruction, DEM work, or instrument
  calibration.

## 14. Roadmap

Maintained in `docs/research-roadmap.md`. Near-term candidates identified by
the phases so far: multi-ROI surveys of the noise-aware enhancer, confidence
calibration across scene types, multi-scale Retinex comparison under
adaptivity, GeoTIFF export with embedded provenance.

## 15. Scientific disclaimer

LunaPSR is an image-enhancement and analysis system. It does **not**:

* recover photons that were never measured;
* reconstruct ground-truth lunar reflectance;
* prove that enhanced structures are geological;
* create new physical measurements;
* guarantee recovery of features hidden below the sensor's information/noise
  limits.

All enhanced imagery is a **derived representation** intended to make weak
signal easier to inspect while making noise-driven artifacts explicit and
measurable. Any scientific use requires proper illumination geometry, DEM and
instrument calibration context.

# LunaPSR — Research Roadmap

This document is the "why" behind the project and the ordered phases for
turning an enhancement workbench into a defensible lunar-science workflow.
Everything is motivated by one constraint: **an OHRC PSR product is only useful
scientifically if it is geometrically and photometrically aware, and the
browse-product data alone does not provide that.**

Phase gates are about *validity*, not just features: a phase is done when the
claims being made can be backed by sources, not when the UI looks nice.

---

## Phase 1 — Explorable classical workbench (implemented)

**Goal**: a reproducible, honest environment for comparing enhancement methods.

- [x] Classical stages: normalization, gamma, CLAHE, Single Scale Retinex,
      denoisers (bilateral / Gaussian / NLM), detail enhancement.
- [x] Composable custom pipelines with arbitrary stage order.
- [x] Metrics and validation: contrast, entropy, sharpness, EPI, SNR,
      saturation / clipping / noise-amplification checks, all labelled as
      proxies where appropriate.
- [x] ROI metrics + noise analysis for crater regions.
- [x] Full reproducibility: per-run JSON records, export formats (PNG, 16-bit
      TIFF, difference, confidence map), and one-click reload of parameters.
- [x] Model interface (`EnhancementModel`) demonstrating that classical and
      future learned models share one evaluation harness.

**Open fence post (v1 limit)**: PSR detection is manual-ROI only. We will *not*
derive PSR locations from brightness statistics alone — that is precisely the
kind of claim that needs geometry.

---

## Phase 2 — Contextual metadata & geometry-aware PSR detection

**Goal**: stop treating pixels as self-contained data.

1. **Metadata pipeline.** Parse acquisition metadata from ISRO product files /
   PDS labels: acquisition time, orbit, spacecraft position, solar incidence &
   azimuth, viewing geometry, gain/exposure if available.
2. **Geometry-aware shadow detection.** Use the metadata together with a lunar
   DEM (e.g. SLDEM2015 / LOLA-derived products) to compute cast shadows and
   genuine permanently-shadowed terrain, seeded by the manual ROIs collected in
   Phase 1 as ground truth for validation.
3. **Validate the detector** against published OHRC PSR mosaics and any
   shadow-mask ground truth, reporting precision/recall — not a confidence
   number without an error model.
4. Extend `core.psr` to emit these regions through the same interface, with
   per-region metadata (source, confidence, geometry).

**Gate**: a PSR region is reported *with* its illumination/DEM context, and the
detection error budget is documented.

---

## Phase 3 — Calibration, radiometry, and measurement validity

**Goal**: convert proxies into defensible numbers — or clearly say which cannot
be converted.

1. **Sensor noise model.** If ascending/descending pairs or calibration frames
   become available, fit a noise model (photon + readout + pattern) for OHRC;
   otherwise keep noise/SNR labelled as estimates and document the limit.
2. **Calibrated radiance path.** MRF/calibration coefficient support so the
   tool can ingest products with radiometric calibration and treat
   enhancement/difference products in radiance rather than DN.
3. **Ratio & multi-temporal products.** Successive-orbit / phase-ratio
   comparisons, which are the standard way to expose subtle PSR albedo and
   ice-signature variations.
4. **Deeper measurements.** A measurement focused on crater-floor contrast
   inside detected PSRs, e.g. contrast within regions of varying illumination,
   with uncertainty propagation.

**Gate**: each exported number states its SI basis or its proxy status, and the
validation suite (Phase 1) is extended with the calibrated path.

---

## Phase 4 — Learned methods, validated, not assumed

**Goal**: integrate learned enhancement only where it beats calibrated
classical baselines on OHRC data.

- [x] Interface and stubs: `RetinexformerModel`, `ZeroDCEModel`
      (`models/ai/`) — lazy, no bundled weights.
- [ ] Build an OHRC (and simulated PSR) training/eval set derived from Phase 3
      products and synthetic illumination simulators.
- [ ] Fine-tune/validate zero-DCE or Retinexformer-like models; report the
      same metrics and validation suite the classical pipelines use, so gains
      are measured on the actual evaluation harness.
- [ ] Only then expose learned enhancement in the UI, with per-image confidence
      and failure-mode documentation.

**Gate**: a learned model's numbers are *at least* as interpretable and
validated as the classical baseline it is offered next to.

---

## Phase 5 — Mosaic & end-to-end workflow (longer term)

- Assemble enhanced strips into a polar mosaic consistent with DEM/geometry.
- Batch header-driven processing (many strips via the same record-schema
  pipeline), with QC on the validation checks per record.
- A shareable report bundle per region: inputs + geometry + enhanced products +
  metrics + warnings.

---

## Guiding principles

1. **No fabricated certainty.** If the data to support a number is missing, the
   tool says "estimate" and documents why.
2. **Reproducibility is foundational.** Every phase builds on the Phase 1
   record model; nothing new is done without recording inputs, parameters and
   provenance.
3. **Comparison over trust.** New methods must be compared against existing
   baselines with the same harness before being surfaced.
4. **Least privilege on claims.** "Enhances" and "makes visible" are claims;
   "confirms water ice" is a *much* stronger claim that this project will not
   make from brightness data alone.
# Phase 5A — SSR Scientific Investigation

**Question:** why does Single-Scale Retinex (SSR) reveal visible structure inside
the OHRC dark (PSR stand-in) region while gamma, histogram equalization, CLAHE,
unsharp mask and denoising show substantially less — and is that structure

- **A.** spatial signal already present in the original OHRC data,
- **B.** amplified sensor/image noise,
- **C.** a mixture of both?

This phase is an investigation only. The LunaPSR algorithm was **not** modified;
no neural network, super-resolution or generative method was used; no feature
recovery or geological classification is claimed.

---

## 1. What ROI was investigated?

No exact ROI was stored in any experiment record (the recorded Retinex run,
`outputs/experiments/…retinex…json`, is full-frame at analysis factor 10).
The investigation therefore reused the project's **documented** dark region —
not an arbitrary choice:

| Property | Value |
|---|---|
| Region | Phase 4A §10.3 dark patch (the project's PSR stand-in; also the Phase 4B noise-validation region) |
| Full-resolution lines | 61 750 – 61 949 |
| Full-resolution samples | 6 000 – 6 199 |
| Width × height | 200 × 200 px |
| Pixel count | 40 000 |

Source array: original scientific DN of
`ch2_ohr_ncp_20260103T1005176450_d_img_d18` (read-only; never modified).

## 2. Raw DN statistics (original scientific DN)

| Statistic | Value (DN) |
|---|---|
| Mean | 3.056 |
| Median | 3 |
| Min / Max | 1 / 5 |
| Std | 0.467 |
| Percentiles p1/p5/p25/p50/p75/p95/p99 | 2 / 2 / 3 / 3 / 3 / 4 / 4 |
| Zero fraction | 0.000 |
| Saturation fraction (≥255) | 0.000 |

The decisive observation: **the entire 200×200 ROI spans only five
quantization levels (DN 1–5)**, with 75 % of pixels at DN 3 ± 0.

## 3. Modeled noise level (Phase 4B model, evaluated on ORIGINAL DN)

At the ROI mean DN 3.056 (gain ≈ 104 e⁻/DN — **INFERRED**, never official):

| Component | Sigma (DN) |
|---|---|
| Shot noise | 0.171 |
| Read noise (40 e⁻ published) | 0.385 |
| Quantization (√(1/12)) | 0.289 |
| **Total modeled σ** | **0.511** |

The model is dominated by read noise and quantization at this signal level.

## 4. Model-based SNR

**MODEL-BASED SNR ≈ 5.99** (mean DN / total modeled σ). This is *not* a
measured SNR: it follows from the inferred gain and published read noise.

## 5. Does spatial variation exceed modeled noise?

**No — not in a single one of the 100 windows tested.**

20×20-pixel non-overlapping windows (`window_analysis.csv`):

| Quantity | Value |
|---|---|
| Windows | 100 |
| Median measured σ / model σ | **0.90** |
| Windows flagged "spatial variation above the current noise model" (ratio ≥ 1.5) | **0** |

Every window is classified `consistent_with_modeled_noise`. Within this ROI,
the measured spatial variation is statistically indistinguishable from what the
Phase 4B sensor model predicts. (Caveat: the same dark patch was used to
validate the model in Phase 4B, so agreement is expected; fixed-pattern
structure smaller than the model agreement cannot be excluded.)

## 6. What does SSR amplify?

SSR (`single_scale_retinex`, unchanged, σ = 30, floor 1e-3, output re-mapped
globally by its own 1–99 percentiles into [0, 1]) applied to the ROI-local
normalized input:

| Metric ([0,1] units) | Original | After SSR |
|---|---|---|
| Gradient energy | 0.0200 | 0.0235 (+18 %) |
| Local variance (mean) | 0.0500 | 0.0588 (+18 %) |
| High-frequency energy | 0.0487 | 0.0573 (+18 %) |
| Clipping fraction | 0.220 | 0.020 |
| Noise-amplification ratio (MAD-Laplacian proxy) | — | 0.49 |

What SSR actually does here is **remap a five-DN-wide, noise-dominated field
across the full display range**: the log division removes the (nearly constant)
illumination and the output percentile stretch turns quantization/read-noise
texture into high-contrast visual structure. The mild +18 % rise in
high-frequency energy is *not* automatically useful structure. The MAD-based
noise proxy *decreases* (0.49×) because the log transform reshapes the
distribution — a known limitation of that estimator on log-compressed images;
it should not be read as "SSR denoises".

## 7. Does SSR introduce obvious artifacts?

Feature-consistency test (edge maps, gradient orientation, distance profile):

| Diagnostic | Value |
|---|---|
| Original edge pixels / SSR edge pixels | 2 128 / 2 000 |
| Original edges made stronger | 40.7 % |
| SSR edges that are **new** (not near any original edge) | **31.4 %** |
| New isolated components (≤ 4 px) | **391** (627 px) |
| Gradient-orientation agreement (doubled angle, weighted) | 0.804 |
| Edge-preservation index | 0.827 |
| Halo/ringing signature (elevated ring decaying from edges) | **not detected** |
| Change concentrated at original edges | **no** — the change is a global re-map |

So: no classic halos, but **SSR creates substantial new high-frequency
patterns** (31 % new edges, hundreds of isolated specks) rather than purely
strengthening existing ones.

## 8. Does SSR primarily amplify existing spatial structure?

**In this ROI: no.** Only ~41 % of original edge pixels are strengthened while
~31 % of SSR's edge pixels appear only after SSR, including 391 isolated
components. Combined with §5 (zero windows above the noise model), the
structure SSR reveals here is **consistent with amplified sensor noise and
quantization texture (category B), at most a mixture at quantization scale
(category C)**. Category A (real spatial signal above the noise model) is
*unverified* in this region.

Why the other baselines show less: gamma/CLAHE/denoising preserve the
normalization's clipped plateaus (22 % of pixels at 0/1) and do not apply an
extreme stretch of the noise band; SSR's log + global percentile re-map does,
which makes noise texture *visible* without making it *signal*.

## 9. Is SSR suitable as a foundation for LunaPSR?

As a **concept**, partially: illumination removal plus log compression is
effective at making low-signal texture visible. As an **unchecked mechanism**,
no: this investigation shows SSR redistributes and re-labels noise-dominated
texture as structure (new edges, isolated specks, scene-dependent global
remap). This is direct quantitative support for the LunaPSR design principle —
enhancement strength must be gated by the physics noise model and confidence,
exactly what the Phase 5 prototype implements (its benchmark row shows
amp-ratio 0.77 with EPI 0.91 on this ROI, versus SSR's structural
side-effects). No change was made to LunaPSR in this phase.

## 10. What remains uncertain?

1. **Gain is INFERRED** (~104 e⁻/DN): all model sigmas and the MODEL-BASED SNR
   inherit that uncertainty; no measured SNR exists.
2. **Dark current is unpublished**; assumed small. Scattered-light signal in
   the dark patch is real signal (Phase 4A), so "measured ≈ model" may hide
   small-scale fixed-pattern structure below the model-agreement level.
3. **One ROI, one product, one scale** (σ = 30): conclusions do not transfer to
   brighter terrain, other PSRs, or other SSR scales.
4. Edge/new-edge/isolated-component diagnostics are **threshold-relative
   research diagnostics**, not detections.
5. Exceeding the noise model would still not prove geology; nothing here
   exceeds it anyway. Independent verification (other bands, other products,
   geometry/context) remains required for any such claim.

---

## Comparison on the same ROI (all methods, identical ROI-local normalization)

| Method | Contrast | Entropy | Grad \|mean\| | Noise proxy | MODEL-BASED SNR | Amp ratio | EPI | Clip |
|---|---|---|---|---|---|---|---|---|
| Original | 1.000 | 0.97 | 0.933 | 0.247 | 5.9 | 1.00 | 1.000 | 0.220 |
| Gamma | 1.000 | 0.97 | 0.904 | 0.200 | 6.2 | 0.81 | 0.989 | 0.138 |
| Hist. equalization | 1.000 | 0.97 | 0.933 | 0.247 | 5.9 | 1.00 | 1.000 | 0.220 |
| CLAHE | 0.994 | 0.97 | 0.929 | 0.245 | 5.9 | 0.99 | 1.000 | 0.138 |
| Unsharp mask | 1.000 | 4.96 | 0.931 | 0.261 | 5.9 | 1.06 | 0.996 | 0.220 |
| Denoising | 1.000 | 0.97 | 0.934 | 0.246 | 5.9 | 1.00 | 1.000 | 0.220 |
| **Retinex (SSR)** | 0.993 | 3.52 | 0.851 | 0.122 | **7.3** | 0.49 | **0.827** | **0.020** |
| Zero-DCE | unavailable in this environment | | | | | | | |
| LunaPSR prototype | 1.000 | 0.97 | 0.873 | 0.191 | 6.9 | 0.77 | 0.911 | 0.220 |

Reading: SSR uniquely de-clips the plateaus (clip 0.22 → 0.02) and spreads the
histogram (entropy 3.52), which is *why it looks revealing* — while also having
the largest structural side-effects (lowest EPI). All MODEL-BASED SNR values
remain ≤ 7.3: no method raises this ROI above the low-confidence regime.

## Answer summary

**C-leaning-B:** inside this documented dark ROI, what SSR reveals is
dominated by amplified read/quantization noise texture (B), possibly mixed
with sub-model-scale fixed pattern (C); no spatial variation above the current
noise model was detected anywhere in the ROI (A unverified). SSR's apparent
power comes from an aggressive display re-map of a five-DN field, not from
revealing verified signal.

## Files created

- `core/analysis/ssr_investigation.py` — reusable, tested analysis functions
- `analysis/phase5a_ssr_investigation.py` — read-only runner (product untouched)
- `outputs/phase5a/original_roi.png`, `ssr_roi.png`, `gradient_original.png`,
  `gradient_ssr.png`, `noise_map.png`, `confidence_map.png`, `difference.png`
  (visualization products only)
- `outputs/phase5a/metrics.json` — full machine-readable results
- `outputs/phase5a/window_analysis.csv` — 100-window table
- `outputs/phase5a/structure_maps.npz` — derived maps (float32)
- `tests/test_ssr_investigation.py` — 16 tests

All numbers above are reproduced by `python analysis/phase5a_ssr_investigation.py`.

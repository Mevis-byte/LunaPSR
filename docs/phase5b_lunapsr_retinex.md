# Phase 5B — LunaPSR Adaptive Retinex

**Status:** research prototype (algorithm version `0.1`)
**Product:** `ch2_ohr_ncp_20260103T1005176450_d_img_d18` (read-only throughout)
**Validation ROI:** the exact Phase 5A region of interest — lines 61750–61949 × samples 6000–6199 (200×200 px)

---

## 1. Motivation from Phase 5A findings

Phase 5A established three facts about standard single-scale Retinex (SSR) on
the dark, noise-dominated Phase 4A patch:

1. **The scene is noise-limited.** Raw DN occupies only levels 1–5; every
   analyzed window's variation is *consistent with* the Phase 4B physics noise
   model (median model/measured ratio ≈ 0.90, zero windows above the model).
   There is no evidence of real spatial structure for an enhancer to "bring out"
   in most of this region.
2. **SSR treats everything as signal.** It produced a **31.4 % new-edge
   fraction**, **391 isolated new components**, and dropped the edge
   preservation index to **0.827** — i.e. a large share of what SSR "reveals"
   is a deterministic remap of modeled sensor noise.
3. **A single global confidence number cannot fix this.** The median
   *level-based* confidence in the ROI is high (≈ 0.86) even though the
   variation there is noise-consistent. Level alone must not unlock strong
   enhancement.

Phase 5B therefore replaces SSR's constant-strength application with a
**per-pixel adaptive strength α** driven by a **two-factor confidence** built
from the Phase 4B physics noise model.

## 2. Algorithm formulation

All mathematics operate on derived arrays; the scientific array is never
modified.

```
I    = normalized image in [0,1]          (percentile normalization)
eps  = 1e-3                                (log stability)
R    = log(I + eps) − log(G_σ(I) + eps)    (SSR log-ratio, σ = 30 px)
Rd   = robust [0,1] remap of R             (global 1st/99th percentile bounds)

S    = local signal level (DN)             (Phase 5 robust estimator, w = 9)
M    = medium-scale structure component    (signal − illumination, DN)
N(S) = modeled noise sigma at level S      (Phase 4B: shot ⊕ read ⊕ quant)

C_level = S / (S + k·N)                    (k = 1.0 transition sharpness)
C_struct = |M| / (|M| + N + ε)             (variation evidence vs the model)
g      = low-signal protection (0.7)
C      = C_level · ((1−g) + g·C_struct)    (composite two-factor confidence)

α     = α_min + (α_max − α_min) · smoothstep(C)      (α_min = 0.15, α_max = 0.85)

out   = clip( (1−α)·I + α·Rd + d·C·(I − G_d(I)), 0, 1 )   (d = detail strength 0.5,
                                                            G_d = Gaussian σ_d = 1.5 px)
```

Design intent:

* **Where C → 0** (dark and/or variation consistent with modeled noise):
  α → α_min = 0.15 and the detail term vanishes — output stays close to the
  original.
* **Where C → 1** (bright with structure well above the modeled noise):
  α → α_max = 0.85 and controlled detail is added.
* `smoothstep` gives zero slope at both ends, avoiding banding at the
  saturation points.
* The Retinex foundation (`R`, `Rd`) uses the **same formulation as the SSR
  baseline** so any difference between SSR and LunaPSR is attributable to the
  adaptive gating, not to a different transform.

## 3. Why two confidence factors

| Factor | Question it answers | Failure mode if used alone |
|---|---|---|
| `C_level` (signal vs modeled noise) | "Is there enough light here to trust?" | Dark regions score high because 3 DN ≫ nothing; Phase 5A showed level-only gating barely helps (ablation B). |
| `C_struct` (variation vs modeled noise) | "Does the spatial variation exceed what the sensor model predicts?" | Textureless bright areas would be over-enhanced. |

Their gated product implements the Phase 5A lesson directly: **level alone must
not unlock enhancement; variation evidence above the physics noise model is
required.**

## 4. Parameters (all documented, all serialized)

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `sigma` | 30 px | (0, 200] | Illumination Gaussian inside the log-ratio |
| `epsilon` | 1e-3 | (0, 0.1] | Log stability constant |
| `alpha_min` | 0.15 | [0, 1] | Conservative enhancement floor |
| `alpha_max` | 0.85 | [0, 1] | Strongest effect (≥ alpha_min enforced) |
| `confidence_transition` | 1.0 | (0, 10] | k in S/(S+k·N); larger = stricter |
| `detail_strength` | 0.5 | [0, 2] | Gain of the confidence-weighted detail term |
| `low_signal_protection` | 0.7 | [0, 1] | g; how much variation evidence is demanded |
| `version` | "0.1" | — | Algorithm version string |

Invalid configurations raise `ValueError`. Every run serializes the full
parameter dict (including the formulation string) into results, experiment
records and exports.

## 5. Implementation map

| File | Role |
|---|---|
| `core/enhancement/lunapsr_retinex.py` | Config, result container, pipeline (`enhance_lunapsr_retinex`), `smoothstep`, `required_halo`, artifact warnings |
| `analysis/phase5b_lunapsr_retinex.py` | ROI validation runner: ablation A–E, benchmark, regression checks, outputs/phase5b |
| `analysis/phase5b_fullres.py` | Full-resolution tiled run → outputs/phase5b/full_resolution |
| `tests/test_lunapsr_retinex.py` | 27 tests incl. the mandatory Phase 5A regression test |
| `app/pages/ohrc_lab.py` | "🌑 LunaPSR Adaptive Retinex" lab section |

Reused unchanged: `core/enhancement/signal_estimation.py`,
`core/enhancement/confidence.py`, `core/enhancement/retinex.py`,
`core/analysis/ssr_investigation.py`, `core/analysis/ohrc_noise_model.py`.

## 6. Ablation study (exact Phase 5A ROI)

Each variant adds one mechanism; this isolates which component delivers the
Phase 5A-motivated behavior.

| Variant | Mechanisms | Mean α | Noise amp ratio | HF ratio | New edges | Isolated comps | EPI |
|---|---|---|---|---|---|---|---|
| A | Standard SSR (baseline) | — | 0.49 | 1.18 | 31.4 % | 391 | 0.827 |
| B | + level confidence only (g=0, d=0) | 0.81 | 0.70 | 1.06 | 27.5 % | 372 | 0.880 |
| C | + low-signal protection gate (g=0.7, d=0) | 0.26 | 0.79 | 0.95 | 11.8 % | 230 | 0.987 |
| D | + detail control (g=0.7, d=0.5) — full pixel pipeline | 0.26 | 0.81 | 0.97 | 7.5 % | 146 | 0.988 |
| E | D through the full public API (identical pixels + diagnostics) | 0.26 | 0.81 | 0.97 | 7.5 % | 146 | 0.988 |

Reading:

* **B confirms the Phase 5A lesson quantitatively**: level-based confidence
  alone reduces new edges only 31.4 % → 27.5 %. The median level confidence in
  this dark ROI is simply too high to be discriminating.
* **C is the decisive step**: demanding variation evidence above the modeled
  noise cuts new edges to 11.8 % and restores EPI to 0.987.
* **D adds fine-scale suppression**: the confidence-weighted detail term keeps
  unsupported high-frequency content *below* the original's HF energy
  (ratio 0.97).
* E ≡ D pixel-for-pixel, verifying the public API adds diagnostics without
  changing output.

## 7. SSR vs LunaPSR on the Phase 5A ROI

| Metric | Original | SSR | LunaPSR |
|---|---|---|---|
| Mean brightness | 0.528 | 0.834 | 0.606 |
| Std | 0.233 | 0.250 | 0.228 |
| Gradient energy | 0.0200 | 0.0235 | 0.0192 |
| HF energy | 0.0487 | 0.0573 (+18 %) | 0.0472 (−3 %) |
| Model-based confidence (median, derived DN) | 0.855 | 0.880 | 0.862 |
| Noise amplitude ratio (MAD proxy) | 1.00 | 0.49 | 0.81 |
| New-edge fraction | 0.000 | 0.314 | 0.075 |
| Isolated new components | 0 | 391 | 146 |
| Edge preservation index | 1.000 | 0.827 | 0.988 |

Interpretation (honest): LunaPSR trades some of SSR's dramatic tone lift
(mean 0.606 vs 0.834) for dramatically higher fidelity to measured structure:
~4× fewer new edges, ~2.7× fewer isolated components, near-perfect edge
preservation, and no HF amplification. Both methods *reduce* the MAD noise
proxy (both are nonlinear remaps); neither value is a physical measurement.

## 8. Regression guardrails vs Phase 5A references

The runner asserts (and the unit test enforces on the real product):

| Guardrail | Reference (SSR @ 5A) | LunaPSR result | Pass |
|---|---|---|---|
| New-edge fraction < 0.314 | 0.3135 | 0.0755 | ✅ |
| Isolated new components < 391 | 391 | 146 | ✅ |
| EPI > 0.827 | 0.827 | 0.988 | ✅ |
| HF energy ≤ original × 1.18 | +18 % | −3 % | ✅ |

These are **regression floors, not targets**: the algorithm does not encode any
of these numbers; they only guard against re-introducing the SSR failure modes
documented in Phase 5A.

## 9. Noise amplification analysis

* MAD-proxy amplitude ratio 0.81 (< 1): the remap does not amplify local
  dispersion in this ROI.
* HF energy ratio 0.97 (< 1): no high-frequency energy is added beyond the
  original — the detail term is fully suppressed where variation is
  noise-consistent.
* Caveat: both are no-reference research diagnostics on derived data, not
  measurements of sensor noise.

## 10. Artifact diagnostics & honest warnings

Every result carries machine-readable artifact flags plus human-readable
warnings prefixed *"Potential enhancement artifact detected:"*:

| Flag | Threshold | ROI value | Warning fired |
|---|---|---|---|
| Clipping introduced (after − before) | > 0.01 | ≤ 0 (inherits input plateaus) | no |
| New-edge fraction | > 0.25 | 0.075 | no |
| Isolated new components | > 100 | 146 | **yes** |
| Noise amplification ratio | > 1.5 | 0.81 | no |
| HF energy ratio | > 1.5 | 0.97 | no |
| Halo-like profile | boolean | false | no |

The remaining warning is honest and expected: a dark-scene remap inevitably
creates some small connected components that were below the edge threshold
before. It is reported as a *potential* artifact, never explained away.

## 11. Full-resolution processing design

* Strip tiling: 2048 rows, halo = `required_halo()` = 4σ+8 = **128 px** on
  every side, so every Gaussian window sees its true context.
* Tone-map consistency: the Retinex response remap bounds are measured **once**
  on a factor-16 representation and passed to every strip via
  `display_bounds`; α/C are purely local and tile-consistent by construction.
* Outputs: float32 `[0,1]` enhanced array + uint8 display copy
  (`outputs/phase5b/full_resolution/lunapsr_retinex_enhanced_{f32,u8}.img`),
  reduced-resolution previews/maps/metrics/experiment record alongside.
* The product is opened memory-mapped read-only; strips are copied into
  working arrays and never written back.

## 12. Test coverage (27 tests)

Contract: source immutability, output bounds, determinism, invalid-pixel NaN
handling, NaN/Inf prevention, shape mismatch rejection, all-invalid rejection.
Behavior: confidence/alpha bounds, low-confidence protection (dark noise scene
→ conservative α), high-confidence behavior (structured scene → strong α),
protection-gate monotonicity, LunaPSR-beats-SSR on new edges/isolated
components in a synthetic noise-dominated scene, smoothstep properties, halo
sizing. Serialization/validation: round-trip, per-field `ValueError`s.
Diagnostics: artifact keys, honest warning wording, JSON-safe export.
**Real-product regression**: exact Phase 5A ROI, SSR reproduces the published
references within tolerance, LunaPSR beats all four guardrails.

Full suite status at delivery: **284 passed** (257 pre-existing + 27 new).

## 13. UI integration

New "🌑 LunaPSR Adaptive Retinex" section in the OHRC Enhancement Lab
(`app/pages/ohrc_lab.py`), after the Phase 5 LunaPSR section:

* parameter controls with per-parameter help text;
* Original / SSR / LunaPSR tab views, before/after wipe, difference view;
* four headline metrics (mean α, mean confidence, new-edge fraction, EPI);
* six diagnostic maps (confidence, α, modeled noise σ, illumination, Retinex
  response, structure map);
* honest-warning banners;
* experiment save/download (method `lunapsr_retinex`);
* verified via Streamlit AppTest end-to-end (load product → run → metrics
  render, zero errors).

## 14. Reproducibility

* Deterministic: no RNG anywhere in the pipeline; repeated runs are
  bit-identical (tested).
* Every experiment record carries: product name, processing level, full
  algorithm parameters + formulation string, normalization params, noise-model
  dict, diagnostics, artifacts, warnings, notes.
* Runners: `analysis/phase5b_lunapsr_retinex.py` (ROI), 
  `analysis/phase5b_fullres.py [--max-lines N]` (full image).

## 15. Limitations (explicit)

1. **Derived representation only.** Nothing here recovers information OHRC did
   not capture; where variation is consistent with modeled noise, LunaPSR
   deliberately declines to enhance.
2. **Model-based ≠ measured.** All SNR/confidence values are evaluated against
   the Phase 4B *model* (gain inferred, never official). They are not
   measurements.
3. **No-reference metrics.** Gradient/HF/EPI/MAD diagnostics describe the
   derived data, not ground truth.
4. **Single ROI deep-validation.** Behavioral claims are validated deeply on
   the Phase 5A ROI and synthetically elsewhere; other scene types may behave
   differently.
5. **Heuristic thresholds.** Artifact-warning cutoffs (0.25 new edges, 100
   isolated components, …) are research conventions, not standards.
6. **Tiled tone mapping.** Full-res remap bounds come from a factor-16
   estimate; pathological scenes could shift global response percentiles.
7. **No deconvolution/super-resolution claim.** The detail term is a
   confidence-weighted high-pass of existing samples, nothing more.

## 16. Files delivered

```
core/enhancement/lunapsr_retinex.py        algorithm module (v0.1)
analysis/phase5b_lunapsr_retinex.py        ROI validation runner
analysis/phase5b_fullres.py                full-resolution tiled runner
tests/test_lunapsr_retinex.py              27 tests
app/pages/ohrc_lab.py                      🌑 Adaptive Retinex lab section
outputs/phase5b/                           ROI outputs (PNGs, metrics.json,
                                           ablation.csv, experiment.json)
outputs/phase5b/full_resolution/           f32 + u8 arrays, previews, maps,
                                           metrics.json, summary.json,
                                           experiment record
docs/phase5b_lunapsr_retinex.md            this document
```

## 17. How to reproduce

```bash
# ROI validation (fast, ~seconds)
PYTHONPATH=. python analysis/phase5b_lunapsr_retinex.py

# Full resolution (~25 min, ~5 GB float32 output)
PYTHONPATH=. python analysis/phase5b_fullres.py

# Tests
python -m pytest tests/test_lunapsr_retinex.py -q
python -m pytest tests/ -q
```

## 18. Recommendation for Phase 6

LunaPSR Adaptive Retinex meets the Phase 5B success criteria on the Phase 5A
ROI. Recommended next steps, in order of scientific value:

1. **Scene-type survey.** Apply the fixed v0.1 configuration across several
   distinct OHRC regions (bright terrain, shadow boundaries, mixed texture) to
   characterize where the protection gate helps and where it over-suppresses.
2. **Confidence calibration.** Compare `C_struct` distributions against
   window-level measured/model ratios (Phase 5A methodology) on multiple ROIs
   to check the k=1 transition placement.
3. **Multi-scale Retinex comparison.** Evaluate whether a multi-scale
   foundation changes the trade-off frontier once adaptivity is present.
4. **User-facing export formats.** GeoTIFF export of the full-res derived
   product with embedded provenance metadata.

Not recommended: treating any LunaPSR output as improved radiometry. It is a
visualization aid with explicit, model-based restraint.

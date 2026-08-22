# Phase 4B — Physics-Aware OHRC Noise Model and Model-Based SNR

Status: **implemented**.  Supersedes the Phase 3 placeholder ("SNR not yet
estimated") with a clearly-labelled *model-based* estimate.  Everything here is
a **MODEL-BASED ESTIMATE**, never a measured or officially published SNR.

The companion investigation, assumptions, and sources live in
[`phase4a_ohrc_radiometry_noise.md`](phase4a_ohrc_radiometry_noise.md)
(unchanged).  Phase 4A §15 specified this model; this document records its
implementation.

---

## 1. Model

For a delivered 8-bit count `DN` (this product is a **count-calibrated**,
PRNU-corrected Level-1 product whose pixel values are **DN counts**, Phase 4A
§3), the total random variance is

```
σ²_total(DN)  =  σ²_shot  +  σ²_read  +  σ²_quant          [DN²]

σ²_shot  =  DN / g                                  [DN²]   (Poisson)
σ²_read  =  (40 e⁻ / g)²                            [DN²]   (Gaussian)
σ²_quant =  1/12                                    [DN²]   (uniform 8-bit LSB)
```

with the *inferred* gain `g = 104 e⁻/DN` for this MSB-selected 8-bit product
(equivalently ~26 e⁻/10-bit count, Phase 4A §12).  `σ_quant = sqrt(1/12) ≈ 0.289 DN`.

Model-based SNR (unitless):

```
signal_e⁻  =  DN · g                                          [e⁻]
noise_e⁻   =  g · σ_total(DN)                                 [e⁻]
SNR_model  =  signal_e⁻ / noise_e⁻  =  DN / σ_total(DN)
```

## 2. Parameters and assumptions (`OHRCNoiseModelConfig`)

All assumptions are captured in one frozen config object in
`core/analysis/ohrc_noise_model.py` so they are never silently embedded in code:

| Parameter | Value | Status |
| --- | --- | --- |
| gain `g` | 104 e⁻/DN (8-bit) | **INFERRED** — from published full well 26.6 ke⁻ ↔ 10-bit full scale at published saturation radiance (Phase 4A §12); **not** an official OHRC calibration coefficient |
| read noise | 40 e⁻ | published (Chowdhury et al., Current Science 2020) |
| quantization | 1/12 LSB² | assumed uniform 8-bit quantization |
| TDI | TDI64 | product-scoped (label `tdi_stages`); no TDI scaling is applied or claimed |
| status | `model_based` | reported in every export |

The inferred-gain caveat and the TDI scope note are carried in the config object
and surface in the UI and in recorded experiment JSON.

## 3. What the model is NOT

- **Not** a measured/absolute SNR — it inherits the inferred gain and the
  unquantified dark-current / residual PRNU assumptions (Phase 4A §13).
- **Not** a std/mean figure of the array — SNR is never computed as
  `mean/std` of the data.
- **Not** validated across TDI settings — only the TDI64 acquisition is in scope.
- Dark lunar terrain is **never** used as a "noise-only" reference: dark pixels
  carry real scattered-light signal (Phase 4A §9, §12.4).

## 4. Flat-window measured-noise estimator (D)

`estimate_flat_window_noise` implements the Phase 4A §10.5 methodology as an
independent empirical estimate of the **total random component**:

- slide `window × window` (default 11×11, stride 11) windows over a
  full-resolution region;
- select the **low-signal / low-structure** windows (`mean < mean_max`, default
  5 DN) — darkest regions are flattest, so scene texture contributes least;
- report window count, mean DN, measured σ (mean of window stds), the model σ at
  that mean, residual, and relative error.

The result is labelled "measured spread includes possible terrain structure;
**not automatically pure sensor noise**."

## 5. Validation

- **Real product** (Phase 4A dark patch, full-res lines 61750–61949, samples
  6000–6199, 200×200): mean DN ≈ 3.06, measured σ ≈ 0.50 DN, model σ ≈ 0.51 DN
  — agreement to ~3 %, reproducing Phase 4A §12.3.  Automated in
  `tests/test_ohrc_noise_model.py`.
- **Synthetic** Poisson + read + quantized dark regions (mean DN 3) reproduce the
  model σ within 15 %.
- Dark-noise floor: model σ(DN=0) = sqrt((40/104)² + 1/12) ≈ 0.48 DN.

## 6. UI (`app/pages/ohrc_lab.py`, "🔬 OHRC Lab")

New "🔬 Noise & SNR (physics-aware, model-based)" section:

- product + model card with inferred-gain caveat and TDI scope note;
- estimated noise σ and model-based SNR at the analysis-representation median DN;
- measured-vs-model flat-window comparison on the Phase 4A dark patch;
- ROI noise/SNR selector with per-component breakdown (shot / read / quantization /
  total σ in DN, model SNR, signal in e⁻) and a full-resolution flat-window
  comparison of the ROI;
- the existing ROI analysis and the download/experiment JSON now include the
  `noise_model` block.

All figures are labelled `MODEL-BASED ESTIMATE`.

## 7. Experiment records

`ExperimentRecord` gained a `noise_model` field; saved experiments and the
downloadable JSON record include

```json
"noise_model": {
  "type": "poisson_read_quantization",
  "gain_e_per_dn": 104, "gain_status": "inferred",
  "read_noise_e": 40, "quantization_variance_dn2": 0.0833,
  "status": "model_based", "tdi_stages": 64, ...
}
```

## 8. Testing

`tests/test_ohrc_noise_model.py` (17 tests): config defaults, per-component
variance/sigma in explicit units, total = quadrature sum, SNR definitions
(equal in DN and electron space, increasing with signal, never std/mean),
DN=0 floor, flat-window estimator (synthetic match, determinism, input
immutability, input validation), and the real-product dark-patch reproduction
and TDI-scope consistency.

## 9. Limitations

- Absolute SNR inherits the inferred-gain assumption (Phase 4A §13).
- Dark-current and residual fixed-pattern terms are not included (not published).
- `162.10 µs vs 162.100 ms` exposure unit discrepancy remains unresolved (Phase
  4A §6) — exposure-dependent photometric claims are out of scope.

## 10. Next steps (outside this phase)

Phase 4A §14 lists what is still missing for a fully physics-based model
(per-pixel calibration LUT, dark signal, TDI scaling).  Resolving those with
ISSDC would let the inferred gain be replaced by an official coefficient.
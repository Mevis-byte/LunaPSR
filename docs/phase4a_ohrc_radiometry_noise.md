# Phase 4A — OHRC Radiometry and Noise Investigation

Status: investigation + documentation only (no model implemented, no application
change). Analysis date: 2026-08-20.

Product examined (in place, read-only):
`ch2_ohr_ncp_20260103T1005176450_d_img_d18/`

Purpose: determine exactly what the pixel values of the actual product represent
and what scientifically defensible information exists for building a noise/SNR
model in Phase 4B.

Source convention used throughout this report:

- **[L]** — **Product label** = the PDS4 XML label(s) shipped inside the actual
  downloaded product (`data/calibrated/20260103/*.xml`), plus
  `miscellaneous/readme.txt`. These are primary evidence for *this product*.
- **[UG]** — SAC/ISRO, *Chandrayaan-2 OHRC PDS4 Data Products User Guide*,
  High Resolution Data Processing Division, Signal & Image Processing Group,
  Space Applications Centre, Report Chandrayaan-2/DP/SAC/SIPG/HRDPD/TR-27,
  Nov 2019 (v2.0, Nov 2020).
- **[CS]** — A. R. Chowdhury et al. (SAC–ISRO), *Orbiter High Resolution Camera
  onboard Chandrayaan-2 Orbiter*, Current Science 118(4):560–565, 2020,
  DOI 10.18520/cs/v118/i4/560-565. Peer-reviewed, official SAC/ISRO paper.
- **[L20]** — A. K. Dagar et al. (SAC–ISRO), *Preliminary Observations and
  Analysis of OHRC Image, On-board Chandrayaan-2*, LPSC 51, #2412, 2020.
- **[L21]** — Amitabh et al. (SAC–ISRO), *High Resolution DEM Generation from
  Chandrayaan-2 OHRC Images*, LPSC 52, #1396, 2021.
- **[X]** — **External literature** (not official ISRO documentation), clearly
  marked where used.

---

## 1. Product Identification

| Attribute | Value | Source |
|---|---|---|
| Bundle | `ch2_ohr_ncp_20260103T1005176450_d_img_d18` | [L] |
| File name decoding | `ch2` mission; `n` = Normal Operations Phase; `c` = Calibrated data; `p` = Panchromatic HR camera; `YYYYMMDDTHHMMSSssss` = 2026-01-03T10:05:17.6450Z; `d` = under `data/`; `img` = image product; `d18` = ISSDC Bangalore station | [UG] Annexure-I |
| Logical identifier | `urn:isro:isda:ch2_cho.ohr:data_calibrated:..._d_img_d18` | [L] |
| Instrument | Orbiter High Resolution Camera (OHRC), passive panchromatic EO camera, Chandrayaan-2 Orbiter | [L][CS] |
| Observation window | 2026-01-03T10:05:17.6450Z → 10:05:33.4482Z (~15.8 s) | [L] |
| Target / area | Moon, South Polar region | [L] |
| Imaging / dumping orbit | 28372 / 28376 | [L] |
| Altitude, pixel resolution | 100.31 km, 0.25 m/pixel (nominal) | [L][CS] |
| Array | 101 075 lines × 12 000 samples × `UnsignedByte`, offset 0, row-major; 1 212 900 000 bytes (matches declared `file_size`, MD5 verified) | [L][analysis] |
| Illumination (OAT-derived) | Sun elevation 4.86–6.17°, solar incidence 83.83–85.14°, emission 0.00°, phase 93.5–94.8°, no eclipse | OAT records [L][analysis] |

The `d18` token is a **ground-station id** (ISSDC Bangalore) — the earlier
Phase-1 "unknown token" is now resolved by [UG].

---

## 2. Processing Level

**The product is Calibrated — ISRO Level 1 / PDS4 "Calibrated" / CODMAC level 3.**

Evidence:

1. `<processing_level>Calibrated</processing_level>` in the image label, with
   the description *"Calibrated Processing Level describes the radiometric
   correction (look up table) applied to the raw data"* [L].
2. The calibrated product set ("Calibrated Image", "Calibrated Browse",
   "Calibrated Geometry") is defined in [UG] Table 5; PDS4 "Calibrated" maps to
   ISRO Level 1 / CODMAC 3 [UG] Table 3.
3. [CS] defines the OHRC levels: *"Level-0 dataset consists of payload raw
   data... Level-1 data will be radiometrically corrected and geometrically
   tagged data. Radiometrically corrected datasets are free of
   photo-response non-uniformity. Level-2 datasets include both radiometric
   and geometric correction."* Our bundle has radiometric calibration +
   geometry tie-points (CSV) but is **not** map-projected/orthorectified
   (no Level-2 grid resampling): the image array is still in the
   instrument/native frame, geolocated only by the `geometry/*_g_grd_*.csv`
   tie-point grid [L].

So the pixel values have already had a radiometric (lookup-table) correction —
including removal of photo-response non-uniformity (PRNU, i.e. per-detector
response / flat-field correction) — applied at the ground processing facility.

---

## 3. Pixel Value Meaning

| Property | Value | Source | Confidence |
|---|---|---|---|
| Stored data type | `UnsignedByte` (8-bit unsigned), no scale, no offset | [L] `Array_2D_Image`/`Element_Array`; no `scaling_factor`/`value_offset` | High |
| What the file contains | *"This File contains the count calibrated image"* | [L] `File/comment` | High |
| Units | **DN counts only. No radiance unit declared in the product.** | [L] (no `unit` on array; no radiance field) | High |
| Physical meaning | Calibrated DN ≈ proportional to scene radiance (linear response measured) | [CS] Fig. 8 ("Linearity of the radiometric response... radiance vs output counts") | Medium–High |
| Calibration equation provided? | Not in the product. Numeric count↔radiance coefficients were measured ("count (10 bit) to radiance coefficients") but are not published in [CS]; not in the bundle | [CS]; [L] | High (for "not provided") |
| Electronics quantization | 10-bit (electronics), 8-bit (transmission) by bit selection | [CS] Table 4; [UG] Table 7 (`bits_selection`) | High |
| Bit selection used by this product | `bits_selection = msb` (i.e. the delivered 8-bit DN = most-significant 8 bits of the 10-bit count) | [L] `Product_Parameters/bits_selection`; [UG] Table 7 | High |
| TDI setting used | `tdi_stages = TDI64` | [L] | High |
| Integration time declared | `line_exposure_duration = 162.100 ms` (unit as labelled) | [L] | see §6 (unit discrepancy) |
| Sensor | TDI-CCD, 12 000 px × 256 TDI (selectable 64/128/192/256) | [CS] Table 3 | High |

**Answer: the pixels are digital-number (DN) counts — calibrated (Level-1,
PRNU-corrected) 8-bit counts linearly proportional to scene radiance.** They
are *not* raw detector values, *not* physical radiance units, and no
scale/offset or calibration equation accompanies the product.

Note on [UG] Table 3: the generic PDS4/CODMAC sentence *"Calibrated = Data
converted to physical units"* is a level-definition boilerplate; the **actual
delivered product is explicitly "count calibrated"** [L] and carries no radiance
unit, so the physically meaningful reading of "physical units" for this product
is a *normalised/calibrated count scale*, not SI radiance.

---

## 4. Radiometric Calibration

Items searched: radiometric calibration, gain, offset, scale factor, bias,
dark signal, detector response, flat field / non-uniformity, saturation,
integration time, TDI setting.

| Item | Status | Source |
|---|---|---|
| Radiometric calibration applied | **Yes** — Level-1 "count calibrated" via a lookup table; free of photo-response non-uniformity | [L]; [CS]; [UG] |
| Calibration equation / coefficients | **Not available in inspected product/documentation.** Coefficients were *measured* in the lab (count↔radiance, linear, Fig. 8) but numeric values are not published in [CS], [L20], [L21], [UG] or the bundle | [CS]; [L] |
| Gain / e⁻ per count | **Not published.** (Phase 4A derives a consistency estimate — see §12 — it is *not* a published calibration value) | [CS] |
| Offset / bias / dark signal | **Not available in inspected product/documentation.** No dark/bias frame in bundle; paper mentions deep-space dark observations for dark-noise study only | [L]; [CS] |
| Dark current rate | **Not available.** (Detector operating temperature range 10–30 °C is documented [CS] Table 5, but no e⁻/s rate) | [CS] |
| Flat field / non-uniformity (PRNU) | **Applied** at Level-1 (product is "free of photo-response non-uniformity"); the PRNU correction table itself is **not** bundled | [CS]; [L] |
| Linearity | **Documented** — radiometric response linear (10-bit count vs radiance) | [CS] Fig. 8 |
| Saturation radiance | **Documented** — 0.8 mW/cm²/sr/µm @256 TDI; 3.2 @64 TDI | [CS] Table 1/7; [L21] |
| Reference radiance / illumination | 0.5 mW/cm²/sr/µm @ 8% albedo, 5–6° sun elevation | [CS]; [L21] |
| Integration time options | 7 selectable values; 162.10 µs for nadir | [CS] Table 4 |
| TDI setting | **Documented for this product**: TDI64 | [L] |
| Vignetting / optics response | Not quantified in the bundle; MTF/SWR published (SWR 24% @ Nyquist) | [CS] Table 2/6 |

---

## 5. OHRC Detector Characteristics

Official values relevant to noise ([CS] Tables 1–4):

| Parameter | Value | Source |
|---|---|---|
| Detector | TDI-CCD, 12 000 px, up to 256 TDI stages | [CS] |
| Pixel size | 5.2 × 5.2 µm | [CS]; label `detector_pixel_width = 5.2 µm` [L] |
| Full-well capacity | 26.6 ke⁻ | [CS] Table 3 |
| Read noise | **40 e⁻** (system-level measurement) | [CS] Table 3 |
| Quantization | 10 bit electronics → 8 bit transmission (bit selection) | [CS] Table 4; [UG] |
| Spectral range | 450–800 nm ([CS]); product label says 500–800 nm ([L]); LPSC abstract 0.45–0.70 µm ([L20]) — reported range differs between official sources | [CS]/[L]/[L20] |
| Telescope | Ritchey–Chrétien + field-corrector, D = 300 mm, EFL = 2046 mm ([CS]); label `focal_length = 2080 mm` [L] | [CS]/[L] |
| SNR @ reference radiance | ~100 with 256 TDI (~140 at saturation) | [CS] Table 7; [L21] |
| SNR at 128 TDI | 80 (reference radiance) | [L20] Table 1 |
| Nominal integration time (nadir) | 162.10 µs | [CS] Table 4 |

Imaging geometry for this product (from [L] and OAT): roll −2.48°, pitch +9.57°,
yaw −0.006°, sun elevation ≈ 5°, incidence ≈ 84–85° (near the design reference
"5–6° sun elevation, 8% albedo"); 101 075 lines × 12 000 samples covers a
~3 km × ~25 km swath segment at the lunar South Pole.

---

## 6. TDI and Integration

- The product used **TDI64** (`tdi_stages = TDI64`) [L]. Selectable stages are
  64/128/192/256 [CS]. TDI integrates N rows of the same ground point, giving
  ~√N SNR gain for a photon-noise-limited signal — the stated purpose is
  "good signal-to-noise ratio under low illumination" [CS].
- `line_exposure_duration = 162.100 ms` [L] — **unit discrepancy**: the official
  instrument paper documents **162.10 µs** as the nadir-view integration time
  ([CS] Table 4), and the physics of a 0.25 m GSD at ~1.6 km/s orbital speed
  (~155 µs per line) is consistent with µs, not ms. The label value therefore
  appears to be the documented 162.10 µs value written with the wrong unit. As
  the label unit is authoritative *as delivered*, this is recorded as a
  label/unit inconsistency rather than assumed corrected.
- For a TDI-64 acquisition, effective exposure ≈ 64 × per-stage integration,
  i.e. **~10 ms total effective integration per output line** if the 162.10 µs
  is the per-stage value ([CS], derived — this is an interpretation, not a
  published number).

---

## 7. Noise Sources

For a TDI-CCD at Level-1 (PRNU-corrected), the physically expected noise budget
is ([CS] values where published):

1. **Photon shot noise** — Poisson, σ² = signal in e⁻. Not published as a
   number, but implied by the measured linear response and by SNR data [CS].
2. **Read noise** — **40 e⁻**, system level [CS] Table 3.
3. **Dark current noise** — rate **not published**; no dark/bias data in bundle
   [L]. Paper notes deep-space dark observations are used for dark-noise study
   [CS].
4. **Quantization noise** — 10-bit ADC → 8-bit transmission (MSB selection)
   [CS][L]. Quantization is *signal-independent* (uniform ±½ LSB) and becomes
   relatively large after the 10→8 bit reduction (§12).
5. **Residual PRNU / fixed-pattern** — minimised by Level-1 calibration ("free
   of photo-response non-uniformity") [CS]; the residual is not quantified.
6. **Scene surface variation** — *not noise* (see §10); it dominates local
   variance almost everywhere outside the darkest flats (§10, §12).

An actual, calibrated, per-pixel sensor noise model is **not published** in the
official sources inspected ([CS], [L20], [L21], [UG], product bundle). The
published ingredients (read noise, full well, SNR@reference radiance, linearity)
are **instrument-characterisation values**, measured in the lab with an
integrating sphere [CS], not per-product noise maps.

---

## 8. Saturation

- `255` is **not explicitly declared as saturation** anywhere in the product
  metadata (no `saturated_constant` in `Special_Constants`; the label declares
  no special constants at all) [L].
- Measured fraction of pixels at 255: **1.54 × 10⁻⁵** (~18 700 px of
  1.21 Gpx), scattered (row span 1 982–101 034, no concentrated hotspot; a
  bright 200×200 sample near row 97 350 had max 183, no saturation) [analysis].
- Physics cross-check: with the inferred gain of §12, 255 DN(8-bit) ≈ 1020 DN
  (10-bit) ≈ 26.5 ke⁻ ≈ full well (26.6 ke⁻) at TDI64 — i.e. 255 *is* the
  clipping level of this product even though it is not declared. The 0.0015%
  of pixels at the maximum therefore represent true saturation/clipping of the
  8-bit scale (bright specular/boulder pixels), consistent with full well at
  the MSB-selected 8-bit maximum [analysis; [CS] full well].

---

## 9. Zero / Nodata

- The label **declares no nodata/zero sentinel** (no `missing_constant` /
  `invalid_constant`; no `Special_Constants` block) [L]. Zero is **not**
  defined as nodata.
- Measured zero fraction: **1.44 × 10⁻⁵** (~17 400 px), scattered in 408 rows
  across the whole strip (row 0 → 101 074); a slight excess in sample columns
  0–400 (image edge) is visible but is not a declared pattern [analysis].
- Physics: 0 DN(8-bit) in an MSB-selected product corresponds to 0–3 DN
  (10-bit) ≈ ≤ ~78 e⁻ under the §12 gain — i.e. values at or below the
  detector/readout floor in deep shadow, or occasional readout/defect pixels.
  **Low DN values contain genuine (weak, scattered-light) signal**: a 200×200
  dark patch had mean 3.06 DN, min 1, and **no zeros** [analysis]; [L20] also
  documents OHRC "measur[ing] the weak signal coming from the floor of [a]
  crater". Therefore **zero must not be masked as nodata** for this product.

---

## 10. Actual Product Statistics

All derived by the read-only analysis script `analysis/phase4a_investigate.py`
(memmap + chunked, full-array exact histogram; no data modification).

### 10.1 Full array (1 212 900 000 px)

| Statistic | Value |
|---|---|
| min / max | 0 / 255 |
| mean / std | 49.85 / 36.05 DN |
| median | 46 DN |
| p1 / p99 | 2 / 144 DN |
| fraction < 5 DN | 9.24 % |
| fraction < 10 DN | 16.94 % |
| fraction == 0 | 1.44 × 10⁻⁵ |
| fraction == 255 | 1.54 × 10⁻⁵ |
| distinct values | 256 (full 8-bit range used) |
| even / odd balance | 49.1 % / 50.9 % |

### 10.2 Rows and columns (sampled every 50th)

| Quantity | Value |
|---|---|
| Row means | 3.18 … 115.06 DN (mean 49.8, spread σ = 25.7) |
| Column means | 45.5 … 53.1 DN (spread σ = 1.69) — no dead/striped columns |
| Columns with std < 1 | 0 — no constant (dead) columns |
| Rows with > 50 % zeros | 0 — no all-dark readout rows |

Rows vary hugely (dark → bright across the low-sun strip); columns are uniform.
There is **no evidence of column-wise dead/bad detectors or systematic stripes**.

### 10.3 Representative 200×200 patches

| Patch (line) | mean | std | min | max | zero frac | sat frac | mean \|adj diff\| |
|---|---|---|---|---|---|---|---|
| Dark (61 750) | 3.06 | 0.47 | 1 | 5 | 0.0 | 0.0 | 0.38 |
| Mid (46 250) | 24.16 | 13.78 | 5 | 88 | 0.0 | 0.0 | 1.7 / 2.6 |
| Bright (97 350) | 117.4 | 14.99 | 45 | 214 | 0.0 | 0.0 | 2.4 / 3.6 |

Dark-patch structure: 5 unique values (1–5); 66.6 % of adjacent sample-pairs
equal, 31.8 % differ by 1, 1.5 % by 2, 0 % by ≥3 — i.e. **heavily
quantization-limited** (the real signal depth is lost below ~1 DN steps).

### 10.4 Spatial noise proxies (sampled)

- Local std over 7×7 windows, whole-image sample: **4.74 DN** mean — this mixes
  scene texture and noise and must **not** be read as a sensor-noise figure.
- In flat low-signal windows (11×11, local mean < 5 DN): **std ≈ 0.50 DN**
  (412 windows, mean DN 3.26). This is the *noise-floor* scale of the darkest
  scene [analysis].

### 10.5 Variance-vs-mean (flat 11×11 windows, low-signal)

| local mean (DN) | observed std (DN) | model std* (DN) |
|---|---|---|
| 3.26 | 0.498 | 0.510 |
| 7.07 | 0.980 | 0.535 |
| 17.4 | 5.20 | 0.60 |

*model = shot + read + 8-bit quantization at inferred gain (see §12). The
0–5 DN bin matches the noise-floor model; the rapidly rising spread at > 10 DN
is **lunar surface texture** (0.25 m/pixel regolith), i.e. scene structure, not
sensor noise.

---

## 11. Candidate SNR Definitions

Each candidate is classified: **VALID NOW** / **POSSIBLE WITH ASSUMPTIONS** /
**NOT SCIENTIFICALLY DEFENSIBLE YET**.

| # | Definition | Signal | Noise | Assumptions | Classification |
|---|---|---|---|---|---|
| A | Sensor-model SNR, absolute (e⁻) | e⁻ signal = gain × DN | √(S + σ_read² + σ_dark²) | gain, dark, calibration unknown | **POSSIBLE WITH ASSUMPTIONS** (see §13) |
| B | Background-region SNR | scene DN vs "dark" region std | dark patch spread = noise | **dark patch is scattered-light signal, not pure noise** | **NOT SCIENTIFICALLY DEFENSIBLE YET** (valid only as a noise-floor bound) |
| C | Local signal/background SNR | per-pixel DN vs local floor | local floor std | scene-vs-noise confusion | **NOT SCIENTIFICALLY DEFENSIBLE YET** (same reason) |
| D | Random-noise sigma from spatial statistics (flat windows, robust MAD / high-pass residual) | n/a (estimates σ_noise only) | read + shot + quantization + residual PRNU | flat-region selection, stationarity | **VALID NOW** as a *noise-magnitude estimate* (not an SNR alone); must restrict to flats and label "total random component" |
| E | Poisson/shot-noise model (σ² = S/g) | linear DN ∝ radiance | Poisson + read + quant | gain g; linearity (measured); shot dominance (supported §12) | **POSSIBLE WITH ASSUMPTIONS** — scientifically justified *for relative / model SNR*, with stated gain assumption |

Key point: because the darkest pixels are still *signal* (scattered light, [L20]
and §10), using them as "noise" (B/C) is the classic error Phase 4A explicitly
warns against. A defensible SNR must come from either (i) the physics model (A/E)
with a validated gain, or (ii) a strictly-named "random-noise sigma" statistic (D).

---

## 12. What Can Be Computed Now

1. **Exact global statistics** of the real product (§10) — done, read-only.
2. **Random-noise sigma estimate (D)**: flat-window residual statistics
   (observed noise floor ≈ 0.50 DN at the darkest flats) — VALID NOW, must be
   labelled "total random component, flat-region restricted".
3. **A gain-consistent shot+read+quantization model** — a self-consistency
   chain from **published** values, usable as a Phase 4B starting model:

   - Full well 26.6 ke⁻ ↔ 10-bit full scale (1023 DN) at saturation ⇒
     **g ≈ 26 e⁻ / DN(10-bit)**, i.e. **≈ 104 e⁻ / DN(8-bit)** for this
     MSB-selected, TDI64 product (derived; not published). This is consistent
     with the *published* saturation radiance scaling (0.8 → 3.2 mW/cm²/sr/µm
     for 256→64 TDI, i.e. ×4) and with the 8-bit maximum ≈ full well.
   - σ²(DN) ≈ DN/104 + (40/104)² + 1/12  (shot + read + uniform quantization)
   - **Predicts** noise-floor std ≈ 0.51 DN at DN = 3.26; **observed** 0.498 DN
     (§10.4) — agreement to ~3 %.
   - **Predicts** SNR ≈ 123 at the reference radiance (TDI256); **published**
     ~100 ([CS] Table 7) — consistent to ~20 %, i.e. shot-noise-dominated as
     the paper's SNR curve implies.
   - **Predicts** the TDI64 median (46 DN ≈ 4.8 ke⁻) sits at roughly the
     reference radiance (0.5 mW/cm²/sr/µm ≈ 4.2 ke⁻ at TDI64) — consistent.
4. **Quantitative statement about the dark end**: at the inferred gain, the
   ~9 % of pixels below 5 DN correspond to signal ≲ 0.5 ke⁻ where read + shot +
   quantization give SNR ≲ 10 — the darkest fraction of the strip is
   noise-adjacent (motivates PSR enhancement), but the signal is real
   scattered light, not "no data".
5. **Row/column integrity checks** (§10.2) and **zero/saturation audits** (§8,
   §9) — no anomalies that would require masking.

---

## 13. What Requires Assumptions

- **Absolute per-pixel SNR (A/E)** requires the gain g (e⁻/DN) and the dark
  signal. g is *inferred* (§12) from the published full-well/saturation
  consistency and validated against the observed noise floor; it is **not a
  published calibration value**, so absolute SNR inherits this assumption.
- **Dark-current contribution** is unknown (no published rate); assumed small at
  ~10 ms effective integration and 10–30 °C, but unquantified.
- **Residual PRNU / fixed-pattern** after Level-1 is assumed small (product is
  PRNU-corrected [CS]) but its residual is not published.
- **10→8-bit MSB mapping** is assumed to be a simple ÷4 truncation/selection
  (label `bits_selection = msb` [L]); any rounding/dithering in ground
  processing is unknown.
- The **162.10 µs vs 162.100 ms** unit inconsistency (§6) must be resolved with
  ISSDC before any exposure-dependent absolute photometric claim.

Everything above is presented as *model-based estimates*, never as measured
calibration.

---

## 14. Missing Information Required for a Physics-Based Noise Model

Only items genuinely required by candidate models (A/E/D):

1. **Count↔radiance calibration coefficients** (gains, DN↔radiance) — measured
   at SAC [CS] but not published/bundled.
2. **Validated in-flight gain (e⁻/DN) and bias/offset** for this observation.
3. **Dark current / dark-signal rate** for the detector at the operating
   temperature and integration setting (deep-space darks exist per [CS] but are
   not in the bundle).
4. **Per-detector calibration LUT / flat-field (PRNU) table** actually applied
   at Level-1 (product is PRNU-corrected [CS]; the table itself is not bundled).
5. **Exact 10→8-bit MSB mapping** and any rounding performed during product
   generation.
6. **Temporal noise statistics** of the delivered product (no repeated
   frames / multi-frame data in a single product).
7. **Correction of the `line_exposure_duration` unit** (ms vs µs) in the label.
8. **Thermal state during acquisition** (only the 10–30 °C operating envelope
   is published [CS]).
9. **Vignetting / edge-response residual** and **MTF effect** on noise texture
   (SWR/MTF published [CS], not the per-product spatial response).

---

## 15. Recommended Phase 4B Model

Implement, in `core/analysis` (isolated, read-only, estimates clearly labelled),
a **Gain-inferred photon-shot + read-noise (Poisson + Gaussian) SNR model**:

```
DN10 = 4 · DN8                       # MSB selection [L][CS]
S_e⁻  = g10 · DN10 ,  g10 ≈ 26 e⁻/DN   # from full-well/saturation consistency [CS] + §12
σ_e⁻  = sqrt(S_e⁻ + (40 e⁻)² + σ_dark²)   # read noise published [CS]; dark term default small, flagged
σ_DN  = σ_e⁻ / g10 / 4  (quantization + 1/12 LSB² added in DN8 units)
SNR_pixel = S_e⁻ / σ_e⁻   (reported as "model-based estimate", not measured)
```

Validation harness (all already measurable, read-only):

- Dark/flat noise-floor: observed ≈ 0.50 DN vs model ≈ 0.51 DN (§10.4, §12).
- Published SNR ≈ 100 @ reference radiance / 256 TDI ([CS]; also 80 @ 128 TDI
  [L20]) as an upper-reference check.
- Variance-vs-mean on flat windows must follow the model floor at DN ≲ 10 and
  depart only where scene texture begins (scene ≠ noise, §10.5).

Also implement **D** (flat-window random-noise sigma) as an independent
empirical noise estimate, and keep the three-noise separation (§10.5, §7):
instrument noise, processing artifacts, and natural surface variation are
reported as separate quantities. Never label crater edges / regolith texture as
"noise"; never label SNR unless it is the model SNR (clearly caveated).

---

## 16. Sources

Official / primary:

1. **Product PDS4 labels + readme** — `ch2_ohr_ncp_20260103T1005176450_d_img_d18/`
   (`data/.../*.xml`, `browse/.../*.xml`, `geometry/.../*.xml`,
   `miscellaneous/readme.txt`, OAT/OATH/LBR/SPM). Primary evidence for this
   product. [L]
2. SAC/ISRO, **Chandrayaan-2 OHRC PDS4 Data Products User Guide**, HRDPD/SIPG,
   Rep. Chandrayaan-2/DP/SAC/SIPG/HRDPD/TR-27, Nov 2019 (v2.0 Nov 2020). [UG]
   (public mirror used for inspection:
   ia801806.us.archive.org/.../ch2_ohrc_data_products_user_guide.pdf)
3. A. R. Chowdhury, M. Saxena, A. Kumar, S. R. Joshi, Amitabh, A. Dagar, et al.,
   **Orbiter High Resolution Camera onboard Chandrayaan-2 Orbiter**, Current
   Science 118(4):560–565 (2020),
   DOI 10.18520/cs/v118/i4/560-565. [CS]
4. A. K. Dagar et al., **Preliminary Observations and Analysis of OHRC Image,
   On-board Chandrayaan-2**, LPSC 51, abstract #2412 (2020). [L20]
5. Amitabh et al., **High Resolution DEM Generation from Chandrayaan-2 OHRC
   Images**, LPSC 52, abstract #1396 (2021). [L21]
6. ISSDC/ISRO — PRADAN Chandrayaan-2 archive pages and FAQ
   (pradan.issdc.gov.in/ch2; chmapbrowse.issdc.gov.in); ISRO "Chandrayaan-2
   Science and Data Product documents" page. (Context; used for the availability
   of archive/SPICE and the OHRC description.)

External literature (used only for context, clearly separated):

7. Robinson et al., **ShadowCam Instrument and Investigation Overview**
   (2023) — lunar PSR imaging context; no OHRC instrument facts. [X]
8. Open-source OHRC DEM pipeline preprint (arXiv 2604.01032, 2026) — notes that
   ISRO's calibrated products apply "a scaling and a mildly non-linear response
   correction"; external, not used for instrument parameters. [X]

Note: per-pixel calibration LUT availability claims circulate anecdotally in
non-official sources; they were **not verified** from official ISRO/ISSDC
documentation during this investigation and are therefore not relied upon here.

---

## Appendix — Analysis performed

- `analysis/phase4a_investigate.py` (read-only; memmap + chunked): exact
  full-array histogram, zero/saturation fractions, row/column statistics,
  representative patches, local-std noise proxies, quantization balance.
  Derived output: `outputs/phase4a/stats.json`.
- Supplementary read-only passes: zero/saturation spatial distribution; OAT
  illumination extraction; flat-window variance-vs-mean; dark-patch
  quantization structure.
- No product file was read-write, copied, renamed or modified; Phase 3 test
  suite still passes.
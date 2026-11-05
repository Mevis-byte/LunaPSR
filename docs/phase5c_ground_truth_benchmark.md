# Phase 5C — Ground-truth detection benchmark in a real OHRC dark region

**Status:** research benchmark, complete
**Product:** `ch2_ohr_ncp_20260103T1005176450_d_img_d18` (read-only throughout)
**Background:** 8 real 256 × 256 tiles from the noise-dominated dark region
(rows 60000–64000, columns 4000–9000)
**Ground truth:** analytically known craters and boulders injected at
controlled contrast
**Runtime:** 471.8 s full run (single machine, CPU)

---

## 1. Why this experiment exists

Phases 5A and 5B could show that enhancement methods *add* structure in a dark
PSR region, and could quantify how much of that structure is consistent with
sensor noise. They could not show whether *real* structure is recovered,
because a single OHRC frame contains no ground truth. Every claim about
"revealing" a crater or boulder in that region was therefore unfalsifiable —
in both directions.

Phase 5C supplies a controlled ground truth. Real OHRC dark tiles (real noise,
real quantisation, real background statistics) are combined with analytically
known crater and boulder profiles at a known amplitude and position. Each
method is then asked to recover those features with a fixed, generic detector
at a **fixed false-positive rate**. The result is a detection-vs-contrast
curve and a shape-fidelity readout per method — including the uncomfortable
cases where enhancement does not help and where a denoiser wins.

Nothing here proves geological reality. It measures what can be recovered when
the answer is known.

---

## 2. Design

### 2.1 Real backgrounds, selected per tile

Candidate tiles are scanned on a 256 px grid inside the dark region and kept
only if `1.5 ≤ median DN ≤ 5.5`, `std ≤ 1.0 DN` and `p99 ≤ 8 DN` — i.e. tiles
whose variation is dominated by the sensor noise floor rather than terrain.
Eight spatially spread tiles are chosen greedily. All selected tiles have
median 3 DN:

| tile (row, col) | median DN | tile std | measured σ | model σ | measured/model |
|---|---|---|---|---|---|
| 60000, 6304 | 3.00 | 0.580 | 0.499 | 0.510 | 0.978 |
| 60000, 8608 | 3.00 | 0.480 | 0.458 | 0.510 | 0.898 |
| 60512, 4256 | 3.00 | 0.481 | 0.457 | 0.510 | 0.896 |
| 61536, 8096 | 3.00 | 0.446 | 0.433 | 0.510 | 0.849 |
| 62048, 6560 | 3.00 | 0.527 | 0.494 | 0.510 | 0.969 |
| 62560, 4512 | 3.00 | 0.510 | 0.442 | 0.510 | 0.867 |
| 63072, 8608 | 3.00 | 0.531 | 0.492 | 0.510 | 0.965 |
| 63328, 7072 | 3.00 | 0.517 | 0.477 | 0.510 | 0.935 |

`model σ` is the Phase 4B model-based noise estimate at the tile median
(`noise_breakdown`, Poisson + read + quantisation; **inferred gain**). The
measured window estimate is 0.85–0.98 of the model, consistent with Phase 4A:
the tiles sit at the quantisation-limited noise floor.

### 2.2 Injected features

Two idealised analytic profiles are used. They are **not** real lunar
photometry; they represent the information content a lander-relevant feature
would produce, and they let the benchmark state an exact amplitude.

**Crater** — parabolic bowl plus a raised rim ring:

```
bowl(d) = −max(0, 1 − (d/r)²)                 r = diameter/2
rim(d)  = 0.8 · exp(−(d−r)² / (2·(0.22 r)²))
profile = (bowl + rim) / max|bowl + rim|
```

The profile is normalised to unit peak, so `amplitude_dn` is the peak bowl
depth in DN; the rim rises to 0.8 × amplitude. The crater is radially
symmetric (no illumination direction) by design: this benchmark measures
contrast detection, not photometric shape inversion.

**Boulder** — Gaussian dome:

```
σ_b     = diameter/5
profile = exp(−d² / (2 σ_b²)) / max(...)
```

The visible footprint radius is about 0.5 × diameter. An optional offset shadow
is implemented but disabled in this sweep (shadow_ratio = 0).

**Injection.** `injected = real_tile + amplitude × profile`, evaluated in
floating-point DN, then floored at 0 DN. The original tile is never modified.
Positions are drawn deterministically, non-overlapping, with a margin from the
tile border. The same positions are reused across amplitudes so the amplitude
sweep is paired.

**Sub-DN amplitudes.** The delivered product is uint8-quantised, so amplitudes
below ~0.3 DN cannot be stored literally in integer counts. Injection in
floating-point DN therefore models a continuous signal superposed on real
quantised noise. This is the standard synthetic-injection methodology and the
correct way to measure a detection *limit*; it is not a claim that such a
signal exists in the product. Amplitudes ≥ 0.5 DN are physically realisable.

### 2.3 Amplitude sweep and sizes

```
amplitudes (DN):  0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0
crater diameters: 12, 24, 40 px   (~3, 6, 10 m at 0.25 m/px)
boulder diameters: 6, 12, 20 px   (~1.5, 3, 5 m)
```

With 8 tiles and 2 injected features per tile, every condition has
**16 features**. Peak input contrast is reported both in DN and in units of the
model noise σ (`amplitude / 0.510 DN`), i.e. 0.5 DN ≈ 0.98 σ, 6 DN ≈ 11.8 σ.
This "input SNR" is a *peak contrast in noise units*, not a measured SNR.

### 2.4 Methods

All methods receive the same injected DN tile. The output is always a
`[0,1]` derived representation; detection never touches the original product.

| method | definition |
|---|---|
| `raw` | global 1/99 percentile normalisation only (lower bound 2 DN, upper 144 DN, from an exact full-product histogram) |
| `ssr` | single-scale Retinex, σ = 30 px (the Phase 5A baseline) |
| `lunapsr` | LunaPSR Adaptive Retinex, default config (Phase 5B) |
| `anscombe_nlm` | generalised Anscombe transform + non-local means, inverse transform |
| `anscombe_bm3d` | generalised Anscombe transform + BM3D, inverse transform (optional analysis dependency) |

The Anscombe methods use the Phase 4B model `Var(y|μ) = α μ + σ²` with
`α = 1/gain` (gain **inferred**, not an official calibration) and
`σ² = read² + quantisation²`. Stabilised data are denoised (NLM: 5×5 patches,
6 px search, h = 0.8; BM3D: `sigma_psd = 1.0`), inverse-transformed, then
normalised with the same global bounds. BM3D is installed only for analysis
(`requirements-analysis.txt`); it is not an app dependency.

### 2.5 Detector and fixed false-positive calibration

A single generic **difference-of-Gaussians** detector is used for every method:

```
DoG(x) = G_s(x) − G_2s(x)          (reflect padding)
s = 0.18 × crater diameter   or   0.20 × boulder diameter
score = max |DoG| inside the feature footprint
footprint radius = 1.25 × crater radius   or   0.6 × boulder diameter
```

The detector is not matched to the injected template, so a detection means an
ordinary blob/ring detector can see the structure, not that the template was
correlated with itself.

**Calibration.** For each method, feature size class and **tile**, 400 random
feature-free locations are scored on the corresponding uninjected control
output. The first 200 locations set the threshold at the 95th percentile; the
other 200 validate the false-positive rate. Thresholds are per tile because
tile-to-tile noise differences are real: a single global threshold transferred
poorly in the Phase 5C pilot (raw FP 0.24 on a held-out tile against a 0.05
target). The validation FP rates actually achieved are:

| method | mean validation FP | worst per-condition mean |
|---|---|---|
| raw | 0.049 | 0.069 |
| SSR | 0.049 | 0.059 |
| LunaPSR | 0.046 | 0.054 |
| Anscombe + NLM | 0.056 | 0.070 |
| Anscombe + BM3D | 0.050 | 0.057 |

### 2.6 Metrics

* **Detection rate** at each amplitude for the fixed 5 % FP threshold
  (Wilson 95 % intervals in `summary.csv`).
* **AUC** of the detector score, features vs held-out control locations.
  Reported over the full sweep and restricted to amplitudes ≤ 0.5 DN.
* **`detection@80`**: the lowest amplitude whose detection rate and every
  higher amplitude reach ≥ 0.8.
* **Shape fidelity** `corr_truth`: Pearson correlation between the
  background-subtracted output and the injected signal inside the footprint.
  Scale invariant, so different global tone curves are comparable.
* **Local CNR** (`cnr_ptp`): footprint dynamic range over annulus standard
  deviation.
* **Control-tile forensics**: new-edge fraction, isolated new components,
  edge-preservation index and noise-amplification ratio from the Phase 5A
  toolbox, computed on the uninjected control tiles.

---

## 3. Results

### 3.1 Headline: AUC, all amplitudes and low amplitudes only

AUC is the probability that a true feature scores above a random control
location. 0.5 is chance, 1.0 is perfect separation.

| method | AUC crater | AUC boulder | AUC all | AUC ≤ 0.5 DN, crater | AUC ≤ 0.5 DN, boulder | AUC ≤ 0.5 DN, all |
|---|---|---|---|---|---|---|
| raw | **0.930** | 0.834 | **0.882** | **0.846** | 0.638 | 0.742 |
| LunaPSR | 0.897 | **0.840** | 0.868 | 0.784 | **0.704** | **0.744** |
| Anscombe + NLM | 0.913 | 0.779 | 0.846 | 0.807 | 0.559 | 0.683 |
| Anscombe + BM3D | 0.913 | 0.770 | 0.841 | 0.807 | 0.542 | 0.675 |
| SSR | 0.890 | 0.753 | 0.821 | 0.752 | 0.559 | 0.655 |

Two things stand out.

1. **The raw data are the best detector overall.** In this extremely clean dark
   region (σ ≈ 0.5 DN), a generic detector on the normalised raw tile separates
   injected craters from control locations better than any enhancement method.
   Enhancement is not required to *detect* these features.
2. **The ordering changes for faint boulders.** At ≤ 0.5 DN, LunaPSR has the
   highest boulder AUC (0.704 vs 0.638 raw) and ties raw overall (0.744 vs
   0.742). This is the one regime where the noise-aware gating measurably helps:
   it preserves compact features while holding back background noise.

### 3.2 Detection at 5 % false positives

Values are the input peak SNR (in model-σ units) at which detection reaches
80 %. Lower is better. `—` means 80 % was not reached within the tested range.

| method | crater 12 px | crater 24 px | crater 40 px | boulder 6 px | boulder 12 px | boulder 20 px |
|---|---|---|---|---|---|---|
| raw | **0.98** | **0.98** | **0.98** | **1.96** | **1.47** | **1.47** |
| SSR | **0.98** | **0.98** | **0.98** | — | 3.92 | 1.96 |
| LunaPSR | 1.47 | 1.96 | 1.96 | 7.84 | **1.47** | **0.98** |
| Anscombe + NLM | **0.98** | **0.98** | **0.98** | 3.92 | 1.96 | **1.47** |
| Anscombe + BM3D | 1.47 | **0.98** | **0.98** | 3.92 | 1.96 | **1.47** |

Key observations:

* **Craters are detectable at ~1 σ peak contrast in the raw data** and remain
  so after denoising. SSR matches raw here, but the detections are accompanied
  by much lower shape fidelity (below) and a 50× noise-amplification ratio — so
  these are "energy detections", not recoveries.
* **LunaPSR is *less* sensitive than raw for faint large craters** (1.5–2.0 σ
  instead of 1.0 σ). This is the deliberate cost of low-signal protection: the
  gate suppresses enhancement below the noise floor, which removes false
  structure but also removes some true faint structure.
* **LunaPSR is best or tied-best for 12–20 px boulders** (1.47 σ at 12 px,
  tied with raw; 0.98 σ at 20 px, ahead of every other method) and clearly
  beats SSR (3.92 and 1.96 σ).
* **LunaPSR is the *worst* method for 6 px boulders** (7.84 σ; SSR never
  reaches 80 %). The detail control and protection gate suppress structures at
  the smallest scale. This is a real limitation, not a rounding effect.

Detection rates at two representative amplitudes:

| amplitude | kind | raw | SSR | LunaPSR | NLM | BM3D |
|---|---|---|---|---|---|---|
| 0.5 DN (≈1 σ) | crater | **0.94** | 0.92 | 0.48 | 0.88 | 0.85 |
| 0.5 DN | boulder | 0.42 | 0.17 | **0.50** | 0.33 | 0.33 |
| 1.0 DN (≈2 σ) | crater | 0.96 | **1.00** | 0.98 | 0.98 | 0.96 |
| 1.0 DN | boulder | **0.96** | 0.48 | 0.67 | 0.83 | 0.75 |

### 3.3 Shape fidelity: detection is not recovery

`corr_truth` (median over 16 features, averaged over the three sizes) measures
whether the recovered pattern actually resembles the injected one.

| amplitude | kind | raw | SSR | LunaPSR | NLM | BM3D |
|---|---|---|---|---|---|---|
| 0.5 DN | crater | 0.53 | 0.50 | 0.45 | 0.82 | **0.88** |
| 0.5 DN | boulder | 0.27 | 0.20 | 0.45 | 0.42 | **0.55** |
| 1.0 DN | crater | 0.78 | 0.82 | 0.81 | 0.94 | **0.96** |
| 1.0 DN | boulder | 0.48 | 0.36 | 0.65 | 0.74 | **0.77** |
| 2.0 DN | crater | 0.91 | 0.93 | 0.94 | 0.97 | **0.98** |
| 2.0 DN | boulder | 0.74 | 0.49 | 0.76 | 0.94 | **0.95** |

This is the central scientific result of Phase 5C:

* **Detection and recovery are different questions.** Raw and SSR can "detect"
  a 0.5 DN crater, but their recovered pattern is only ~0.5 correlated with the
  truth — the detector is firing on local energy, not on the crater shape.
  The Anscombe denoisers, especially BM3D, recover the shape far better at the
  same contrast.
* **LunaPSR is the best of the three enhancers for boulder shape fidelity**
  (0.45 vs 0.27 raw and 0.20 SSR at 0.5 DN; 0.65 vs 0.48 and 0.36 at 1 DN).
  It is *not* better than raw for crater shape at 0.5 DN (0.45 vs 0.53), and
  it never beats the denoisers on fidelity.
* **SSR has the lowest boulder fidelity of all methods** and degrades with
  amplitude (0.49 at 2 DN vs 0.74 raw). Its σ = 30 px illumination estimate
  barely changes across a 6–20 px feature, so the log-ratio response is
  dominated by noise and the per-tile percentile remap then stretches that
  noise. This is the quantitative version of the Phase 5A finding.

### 3.4 Fabrication on control tiles

Median over the eight uninjected control tiles. These are Phase 5A-style
descriptors, deliberately reported as-is.

| method | new-edge fraction | isolated new components | edge preservation index | noise amplification |
|---|---|---|---|---|
| raw | 0.000 | 0 | 1.000 | 1.00 |
| SSR | 0.191 | 448.5 | 0.929 | **50.07** |
| LunaPSR | 0.248 | 367.0 | 0.918 | **14.52** |
| Anscombe + NLM | 0.294 | 339.0 | 0.673 | 0.02 |
| Anscombe + BM3D | 0.413 | 118.5 | 0.571 | 0.00 |

How to read this table honestly:

* **Noise amplification** is unambiguous and strongly favours LunaPSR over SSR
  (14.5× vs 50×). Both are far above 1 because both enhance contrast.
* **LunaPSR does not win every forensic metric.** Its new-edge fraction
  (0.248) is slightly *higher* than SSR's (0.191) in this region, even though
  it creates fewer isolated components (367 vs 448). The Phase 5B ROI gave the
  opposite ordering for new-edge fraction; the metric is sensitive to the
  region and to each output's own gradient distribution. We report the result
  as measured rather than selecting the favourable metric.
* **New-edge fraction and EPI are not meaningful verdicts for denoisers.** A
  denoiser collapses the gradient distribution, so its own 95th-percentile
  edge threshold moves and "new" edges appear almost everywhere; the 0.29–0.41
  values for NLM/BM3D are a property of the metric, not evidence that the
  denoisers hallucinate craters. Their relevant evidence is the fixed-FP
  detection/AUC result, where they do **not** beat raw.
* `noise amplification` for the denoisers is < 1 by construction (they remove
  noise); it is not a fabrication measure for them.

### 3.5 LunaPSR vs SSR, paired

At matched conditions, LunaPSR reaches 80 % detection at a lower input SNR in
2 of 6 size classes, SSR in 3, and 1 is undefined (SSR never reaches 80 % for
6 px boulders). But the aggregate hides a clean split:

* **Craters:** SSR is as sensitive as raw, LunaPSR is less sensitive. LunaPSR's
  crater shape fidelity is not better than raw at 0.5 DN.
* **Boulders:** LunaPSR is substantially better than SSR at every size except
  6 px. Detection-rate gains at 0.5 DN: +44 pp (12 px), +56 pp (20 px);
  at 1.0 DN: +50 pp (12 px). Shape fidelity at 1 DN: 0.65 vs 0.36.

So the honest summary is not "LunaPSR wins". It is: **LunaPSR trades a small
amount of crater sensitivity for a large reduction in noise amplification and a
clear gain on faint 12–20 px boulders; it does not help at the smallest scale,
and it does not beat a good denoiser on shape recovery.**

### 3.6 Figures

* `outputs/phase5c/curves_crater.png`, `curves_boulder.png` — detection rate vs
  input SNR (log axis), three size panels per kind, Wilson intervals drawn.
* `outputs/phase5c/montage_crater_d24.png`, `montage_boulder_d12.png` — rows:
  0.1, 0.5 and 2.0 DN; columns: truth (±amplitude DN), normalised input, then
  each method. The truth panel uses a fixed ±amplitude stretch; the image
  panels use a per-image 1–99.5 % stretch and are **visualization only**.
  Quantitative conclusions come from `summary.csv` / `metrics.json`.

---

## 4. What this does and does not establish

**Established (within this benchmark):**

1. In a clean OHRC dark tile, a generic detector on the normalised raw data
   detects 12–40 px craters at ~1 σ peak contrast and 12–20 px boulders at
   ~1.5 σ, at a 5 % false-positive rate. Enhancement is not needed for
   detection in this regime.
2. Detected ≠ recovered. At 0.5 DN the raw/SSR/Retinex outputs correlate only
   ~0.2–0.5 with the true feature shape, while Anscombe+BM3D reaches ~0.55–0.88.
3. SSR adds no detection power over raw, strongly distorts boulder shape, and
   amplifies noise ~50× on the same tiles.
4. LunaPSR reduces noise amplification ~3.4× relative to SSR, improves faint
   12–20 px boulder detection over both raw and SSR, but is less sensitive than
   raw for faint craters and much worse for 6 px boulders.
5. Per-tile threshold calibration is necessary; a global threshold transfers
   poorly across tiles in this region.

**Not established:**

* That any of these features exist in the product, or that any detected
  structure is geological. The injected features are synthetic by construction.
* Performance on terrain with real structure, variable albedo, illumination
  gradients or mixed noise regimes. All tiles are noise-dominated and from one
  dark region of one product (one TDI setting).
* That the ordering generalises to other detectors (multi-scale, matched
  filter, learned). The generic DoG is one defensible choice; absolute rates
  depend on it.
* That `corr_truth` captures perceptual quality or lander-relevant
  interpretability. It captures linear resemblance to the injected profile.

---

## 5. Limitations

* **Idealised profiles.** No real crater photometry, no directional
  illumination, no ejecta, no boulder shadow in this sweep.
* **Sub-quantisation amplitudes.** 0.1–0.3 DN cannot be stored in uint8 counts;
  those points are a continuous-signal proxy (see §2.2).
* **Sample size.** 16 features per condition gives 6.25 pp detection-rate
  granularity; Wilson intervals are in `summary.csv`. Do not over-read
  single-point differences below ~15 pp.
* **One region, one product.** No terrain diversity, no TDI variation, no
  seasonal or calibration variation.
* **Threshold stationarity.** Per-tile calibration assumes the noise regime is
  roughly stationary inside a 256 px tile. Strongly varying local noise would
  need a local threshold map.
* **Fabrication metrics are not detector-independent.** New-edge fraction and
  EPI were designed for enhancers on quantised data; they are not a fair
  verdict for denoisers (see §3.4).
* **Anscombe approximation.** The generalised Anscombe transform assumes the
  Poisson–Gaussian model and the inferred gain; at very low counts the
  stabilisation is approximate.

---

## 6. Reproduction

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt   # includes optional BM3D
.venv/bin/python analysis/phase5c_ground_truth.py
# smoke run (~1 min, 2 tiles, 3 amplitudes):
.venv/bin/python analysis/phase5c_ground_truth.py --quick
```

The dataset directory must be present next to the repository (see README §9).
The product is opened read-only; all outputs go to `outputs/phase5c/`.

Runtime: 471.8 s for the full run on the reference machine. The runner prints
the exact normalization bounds, the selected tiles, per-condition progress and
the headline AUCs.

---

## 7. Files

| file | contents |
|---|---|
| `core/analysis/ground_truth.py` | analytic profiles, deterministic injection, DoG detector, calibration, AUC/readout primitives |
| `analysis/phase5c_ground_truth.py` | full benchmark runner (methods, per-tile calibration, sweep, figures, record) |
| `tests/test_ground_truth.py` | 17 unit tests for the primitives (no product required) |
| `outputs/phase5c/metrics.json` | complete record: design, tiles, calibration, AUC, detection@80, forensics, caveats |
| `outputs/phase5c/summary.csv` | aggregated detection/fidelity per method × kind × size × amplitude |
| `outputs/phase5c/features.csv` | per-feature scores, thresholds, detection, fidelity |
| `outputs/phase5c/controls.csv` | per-tile calibration and validation FP rates |
| `outputs/phase5c/controls_summary.csv` | per method × size calibration summary |
| `outputs/phase5c/tiles.csv` | selected tiles and noise statistics |
| `outputs/phase5c/curves_*.png` | detection-vs-SNR figures |
| `outputs/phase5c/montage_*.png` | visualization montages (see §3.6) |
| `outputs/phase5c/experiment.json` | `ExperimentRecord` for this run |
| `requirements-analysis.txt` | analysis-only dependencies (BM3D) |

---

## 8. Scientific disclaimer

LunaPSR produces a **derived representation**. This benchmark measures
detection and fidelity against *synthetic* ground truth injected into real
noise. It does not recover unmeasured photons, does not reconstruct true
reflectance, and does not demonstrate that any structure in the product is
geological. The Phase 4B gain remains **inferred**, not an official OHRC
calibration coefficient.

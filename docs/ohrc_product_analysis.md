# Chandrayaan-2 OHRC Product Analysis

Analysis date: 2026-08-19
Product examined: `ch2_ohr_ncp_20260103T1005176450_d_img_d18/`
All checks performed read-only. No files inside the original product were
modified, renamed, converted, or deleted. MD5 checksums declared in the PDS4
labels were verified against the actual files and **all match**.

---

## Product

| Attribute | Value |
|---|---|
| Mission | Chandrayaan-2 (ISRO) |
| Instrument | Orbiter High Resolution Camera (OHRC) — visible panchromatic 500–800 nm, passive |
| Observation start | 2026-01-03T10:05:17.6450Z |
| Observation stop | 2026-01-03T10:05:33.4482Z |
| Target | Moon (South Polar region) |
| Projection (nominal) | Polar stereographic |
| Processing level | **Calibrated** |
| Imaging orbit / dumping orbit | 28372 / 28376 |
| Spacecraft altitude | 100.31 km |
| Pixel resolution | 0.25 m/pixel |
| Reference product | PDS4 (ISRO/ISDA dictionary `ch2_ldd_ISDA_1000`), `Product_Observational` |

The label explicitly defines this level: *"Calibrated Processing Level describes
the radiometric correction (look up table) applied to the raw data."*

---

## Directory Structure

```
ch2_ohr_ncp_20260103T1005176450_d_img_d18/
├── browse/
│   └── calibrated/20260103/     browse (visualisation) PNG + label
├── data/
│   └── calibrated/20260103/     scientific image array + label
├── geometry/
│   └── calibrated/20260103/     lat/lon grid tie-point CSV + label
└── miscellaneous/
    ├── calibrated/20260103/     OAT / OATH / LBR / SPM planning files
    └── readme.txt               binary record-format contract
```

Each sub-directory mirrors the structure of `data/` (per `readme.txt`).

---

## Files

All paths relative to the bundle root `ch2_ohr_ncp_20260103T1005176450_d_img_d18/`.

| Path | Extension | Size (bytes) | Purpose | Image data | Metadata | Geometry | Browse |
|---|---|---|---|---|---|---|---|
| `data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.img` | `.img` | 1,212,900,000 | Scientific image (count-calibrated) | **Yes** | — | — | — |
| `data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.xml` | `.xml` | 8,893 | PDS4 label for the scientific image | — | **Yes** | — | — |
| `browse/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_b_brw_d18.png` | `.png` | 8,180,505 | Browse/display image (10× subsampled) | **Yes** (2nd copy, reduced) | — | — | **Yes** |
| `browse/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_b_brw_d18.xml` | `.xml` | 4,770 | PDS4 label for the browse image | — | **Yes** | — | — |
| `geometry/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_g_grd_d18.csv` | `.csv` | 4,255,725 | Selenographic lat/lon ↔ (pixel, scan) tie-points | — | **Yes** | **Yes** | — |
| `geometry/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_g_grd_d18.xml` | `.xml` | 6,208 | PDS4 label for the geometry grid | — | **Yes** | — | — |
| `miscellaneous/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.oat` | `.oat` | 406,944 | Orbit & Attitude data — 648 records × 628 B | — | **Yes** | — | — |
| `miscellaneous/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.oath` | `.oath` | 201 | Orbit & Attitude Header (1 block, 201 B) | — | **Yes** | — | — |
| `miscellaneous/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.lbr` | `.lbr` | 167,184 | Lunar libration angles — 648 records × 258 B | — | **Yes** | — | — |
| `miscellaneous/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.spm` | `.spm` | 161,352 | Sun-angle parameters — 648 records × 249 B | — | **Yes** | — | — |
| `miscellaneous/readme.txt` | `.txt` | 11,119 | Documented record formats for OATH/OAT/LBR/SPM | — | **Yes** | — | — |

Integrity: MD5 of the `.img`, browse `.png` and geometry `.csv` match the checksums
declared in their PDS4 labels. Record counts in the plan files are exact integer
multiples of the documented record lengths (648 × 628 / 258 / 249).

---

## Scientific Image Format

The scientific product is `data/calibrated/20260103/..._d_img_d18.img`:

- Raw, **uncompressed binary array**, no container wrapper and no header.
- Offset to first element: **0 bytes** (per `Array_2D_Image.offset` and confirmed
  by exact file-size match).
- **2 axes**, `axis_index_order = Last Index Fastest` (sample variest fastest →
  row-major, C order, `array[line, sample]`).
- Axis 1 (`Line`): 101,075 elements. Axis 2 (`Sample`): 12,000 elements.
- Element type: `UnsignedByte` (8-bit).

File size = 101,075 × 12,000 = **1,212,900,000 bytes**, identical to the declared
`file_size` — confirming there is no trailing header, trailer, or padding.

The label comment summarises the content: *"This File contains the count
calibrated image."*

---

## Image Dimensions

- **101,075 lines (rows) × 12,000 samples (columns)** = 1,212,900,000 pixels (~1.21 gigapixels).
- Ground resolution: **0.25 m/pixel** (declared), with focal length 2080 mm,
  detector pixel width 5.2 µm, TDI-64, line exposure 162.1 ms.
- Nominal cross-track footprint at nadir: 12,000 × 0.25 m ≈ **3.0 km** (derived;
  the view is off-nadir, roll −2.48°, pitch 9.57°, so actual swath geometry
  differs).
- The strip spans the lunar South Pole region, corner coordinates
  ≈ lat −84.52°…−85.33°, long 22.79°…27.75° E.

---

## Data Type

- **Unsigned 8-bit integer** (`UnsignedByte`, PDS4; `uint8`).
- **Endianness: not applicable** — single-byte elements; no byte-order concerns.
- No sign bit. Range as delivered: 0–255 (observed min/max 0/255).
- No scaled/offset presentation values declared (`scaling_factor`, `value_offset`
  absent), so stored values are the counts to use directly.

---

## Calibration

- Processing level **Calibrated** = radiometric correction (a lookup-table) has
  been applied to raw instrument output ("count calibrated").
- The data are **DN counts**, not calibrated radiance or reflectance. **No
  radiometric (radiance) unit is declared.**
- No sensor noise model, bias/dark frame, flat field, or detector response curve
  is included in the bundle.
- Acquisition illumination (from `Product_Parameters`, corroborated by the
  `.oat` / `.spm` records):
  - Sun azimuth ≈ 337.1°
  - Sun elevation ≈ 4.9–5.5°
  - Solar incidence ≈ 84.5–85.1° (readme: incidence = 90° − sun elevation)
  - Emission angle ≈ 94.8°, phase (sun-angle w.r.t. −yaw) ≈ 9.9°
- These are extreme low-sun conditions — the intended use case for dark/PSR
  enhancement.

---

## Metadata

PDS4 XML labels (1 per product) carry identification, time, instrument, orbit,
pointing, corner coordinates, and array structure. The volume also ships
machine-readable planning/auxiliary files whose record layouts are fully
documented in `miscellaneous/readme.txt`:

| File | Records | Content |
|---|---|---|
| `.oath` | 1 header (201 B) | Record type, project, block length, station `BLR`, start/end UTC, record count (648), source & mission phase codes |
| `.oat` | 648 × 628 B, 512 ms cadence | EME2000 lunar & spacecraft position (km), velocity, inertial→body + Earth-fixed + lunar-fixed quaternions, sub-satellite lat/lon, solar azimuth/elevation, altitude, eclipse status, emission/phase/solar-zenith angles, orbit no., FoV-vs-velocity angles, X/Y/Z (yaw/roll/pitch) angles |
| `.lbr` | 648 × 258 B | Lunar libration angles Φ, Ψ, Θ and rates |
| `.spm` | 648 × 249 B | Phase angle, sun aspect, sun azimuth, sun elevation, orbit limb direction |

Timing: the `.oath` header brackets 2026-01-03T10:05:14 — 10:05:40 with 648
records, i.e. the planning data spans the instrument duty period containing the
15.8 s image acquisition.

---

## Geometry

`geometry/calibrated/20260103/..._g_grd_d18.csv` is a **grid of selenographic
tie-points** mapping image coordinates to latitudes/longitudes:

- Header (31 B): `Longitude,Latitude,Pixel,Scan`
- 122,452 data records; fields: Longitude (degrees), Latitude (degrees), Pixel
  (image sample column), Scan (image line row).
- Regular grid: **121 pixel columns** (0 … 11900 step 100, plus final 11999) ×
  **1012 scan rows** (0 … 101000 step 100, plus final 101074). Verified:
  121 × 1012 = 122,452.
  - In image terms: a tie-point every ~100 samples and ~100 lines (~25 m at
    0.25 m/px, derived).
- Corners agree with the label: pixel 0 / scan 0 ↔ (lon 27.751481°, lat
  −85.279005°); pixel 11,999 / scan 101,074 ↔ (lon 22.794260°, lat
  −84.562153°).

This makes every pixel geolocatable by bilinear interpolation over the grid.
Only map-projection *metadata* ("Polar stereographic") exists; no DEM, elevation,
or slope data is bundled.

---

## Browse Products

`browse/calibrated/20260103/..._b_brw_d18.png` is the visual companion product.

- **Actual file**: PNG, greyscale (8-bit, mode `L`), **1,200 × 10,107 px**
  (width × height), 8,180,505 bytes, MD5 matches the label.
- **Label caveat**: the browse XML declares an `Array_2D_Image` of 4,211 lines ×
  500 samples, which does **not** match the real PNG dimensions. The recorded
  `file_size` and checksum are correct; the array dimensions in the browse label
  appear nominal/stale. Actual dimensions were measured from the file, not taken
  from the label.
- Content: verified pixel-by-pixel against the scientific `.img`. The browse is a
  **10×10 block-average downsample of the count-calibrated image**:
  - sampled-strip correlation with 10×10 block means: **0.958**
  - regression slope vs. block means: ≈ 0.97 (linear, near 1:1)
  - global mean / median / percentiles of a 36 Mpix scientific sample (50.0 / 46 /
    Q1 2 / Q50 46 / Q99 144) essentially equal the browse values (49.2 / 45 / Q1
    ~2 / Q50 45)
- So the browse is a **faithful, linear, reduced-resolution representation of the
  scientific data** — suitable for fast preview and navigation, *not* the
  full-resolution science product.

---

## Important Findings

1. **The scientific image is huge.** 101,075 × 12,000 × 1 byte =
   **1,212,900,000 bytes (1.21 Gpx)** on disk; the same array as float32 would
   be ≈ 4.8 GB. Any full-image processing in RAM is impractical; LunaPSR will
   need memory-mapped / tiled / ROI strategies.
2. **Format is simple and fully specified.** Uncompressed `uint8` array at offset
   0, row-major, no scale/offset, no endianness. MD5s verify download integrity.
3. **The browse PNG is verified to represent the science data** at 10× reduction
   (`r ≈ 0.96`, linear), so it is a legitimate fast preview of the calibrated
   counts. Its XML dimensions are misleading and must not be trusted.
4. **Full geolocation is available** via the 122,452-point lat/lon ↔ (pixel,
   scan) grid — dense enough (every ~100 px/line) to support per-PSR ROI
   georeferencing without external products.
5. **Genuinely dark/PSR-like content is present.** Across a 36 Mpix sample: ~9%
   of pixels < 5 DN, ~17% < 10 DN, median 46, max 255 — a low-signal polar
   strip ideal for LunaPSR's dark-terrain enhancement workloads.
6. **The product is count-calibrated, not physically calibrated.** Values are DN
   counts after a radiometric LUT; there is no radiance unit, no noise model, no
   flat/bias data. LunaPSR's "estimates not measurements" stance is therefore
   the correct one for this data.
7. **Rich ephemeris/attitude metadata ships alongside** (OAT/LBR/SPM, 512 ms
   cadence) — real solar incidence/azimuth, emission, phase, pointing quaternions
   — available for geometry-aware processing without guessing.

---

## Unknowns

Items not determinable from the product files alone are stated as such:

- **Not determined from product:** the exact radiometric LUT coefficients used
  for "count calibrated".
- **Not determined from product:** the meaning of filename tokens `ncp` and
  `d18` (present in every filename and in the Phase-1 sample); no definition is
  supplied in the bundle.
- **Not determined from product:** whether the array has been resampled into the
  polar-stereographic grid or is still in the instrument frame. The label calls
  the projection "Polar stereographic" and supplies corner coordinates, but the
  processing level is "Calibrated" (sensor-level LUT), and the geometry grid
  provides per-pixel lat/lon — so the two are only reconciled by the tie-points.
- **Not determined from product:** the observation geometry implied by
  `line_exposure_duration` (162.1 ms) versus 101,075 lines over a ~15.8 s
  window. Reconciling line rate, TDI-64, and the 648-record (~332 s) planning
  span requires Level-0 timing/clock products or mission documentation not in
  this bundle.
- **Not determined from product:** calibration/denoising support data (bias,
  dark, flat-field, linearity). Not bundled; would have to be sourced externally.
- **Not determined from product:** browse label array dimensions (4,211×500)
  do not correspond to the actual 1,200×10,107 PNG — the label field appears
  stale/nominal, and the authoritative values are the measured file dimensions.

---

## Implications for LunaPSR

The Phase-1 assumption that OHRC browse products are the working input needs
revision:

1. **Full-resolution processing requires a lazy, tiled reader.** The 1.21 Gpx
   `.img` cannot be loaded as float32 in the current pipeline unchanged. Live
   preview should use the verified-faithful browse PNG; heavy computation should
   target the ROI or a downsampled working scale.
2. **The geometry grid unlocks Phase-2 PSR work.** Scan/pixel → lat/lon
   interpolation (plus the OAT solar incidence/azimuth and attitude quaternions)
   gives genuine illumination geometry for each ROI — the data Phase 2 needs to
   distinguish "dark because unlit" from "dark because shadow-cast", without
   inventing geometry from pixel statistics.
3. **Scientific honesty is enforced by the data itself.** No radiance units, no
   noise model → LunaPSR's `Calibrated`-level handling should treat counts as
   counts and keep all derived quantities labelled as estimates.
4. **Reader design targets (future)**: parse the PDS4 XML label to recover
   dimensions/type/offset programmatically rather than hard-coding; use the
   `file_size == dims` check as a validation invariant; surface the geometry CSV
   (lat/lon/lines/samples) as an optional per-pixel overlay; keep the browse PNG
   for navigation; expose dark-region statistics (fraction < n DN) as a QC
   signal.

No OHRC reader, PNG conversion, or enhancement was performed on the scientific
data during this inspection.
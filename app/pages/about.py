"""Tab 6: About / Scientific Notes.

Scientific integrity guardrails: what the tool can and cannot claim.
"""

from __future__ import annotations

import streamlit as st

from core import __version__


def render() -> None:
    st.subheader("About LunaPSR")
    st.markdown(f"**Version {__version__}** — lunar PSR image enhancement research prototype.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### Goal
Recover useful weak lunar-surface signal from very low-light imagery (initially
Chandrayaan-2 **OHRC** browse products) while suppressing noise, preserving
geological boundaries, minimising artefacts and providing quantitative
evidence of improvement.

### What LunaPSR does — and does not
- **Does** enhance and measure dark imagery with transparent classical
  algorithms.
- **Does not** detect PSRs automatically, find water ice, or invent terrain.
  No hallucinated claims: no surface feature or volatile is inferred from
  enhancement alone.

### Pipeline stages
Normalization → Gamma → CLAHE → (optional) Denoising → Detail enhancement.
Each stage records its parameters so experiments are reproducible.
""")
    with c2:
        st.markdown("""
### Metrics — read the labels
| Metric | Type |
|---|---|
| Contrast | robust percentile contrast |
| Entropy | Shannon entropy (display domain) |
| Sharpness | Laplacian variance |
| Noise Proxy | Laplacian-MAD noise *estimate* |
| SNR Proxy | 20·log10(signal/noise) *engineering proxy* |

**Scientific honesty**: none of these are calibrated measurements. A true SNR
requires the OHRC sensor noise model, exposure/gain metadata and radiometric
calibration coefficients — none of which are public for the browse product.

### Status
Classical baseline (Version 1). AI models are stubbed behind a clean interface
and remain off until Phase 4 of the research roadmap.

> Enhancement changes pixel statistics. It cannot create information that the
> sensor did not collect, and it amplifies noise along with signal. Trust the
> maps and proxies, then validate geologically.
""")


def render_psr_notes() -> None:
    st.markdown("""
### PSR-aware processing (planned)
Future versions will accept illumination geometry (solar incidence, lunar
latitude/longitude, DEM, acquisition time) and perform physics-aware PSR
detection. In this classical version a manually selected ROI stands in for a
PSR region so the metrics/noise/confidence plumbing can be exercised.
""")
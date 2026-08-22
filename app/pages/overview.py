"""Mission Overview — dynamic system/product status dashboard.

All values are obtained dynamically from the existing discovery/metadata and
Phase 4B model systems.  Model-based quantities are explicitly labelled as
MODEL-BASED ESTIMATES, never measurements.
"""

from __future__ import annotations

import streamlit as st

from app.ui.theme import section_label
from core.analysis.ohrc_noise_model import (
    OHRCNoiseModelConfig,
    estimate_flat_window_noise,
    noise_breakdown,
)
from core.ohrc import find_ohrc_products

# Phase 4A §10.3 documented dark patch (row0, col0, size) — same region the
# lab uses for the measured-vs-model validation.
_DARK_PATCH = (61750, 6000, 200)


@st.cache_resource(show_spinner=False)
def _cached_scan(root_str: str):
    return find_ohrc_products()


def _product_entry():
    from configs.defaults import PROJECT_ROOT

    scan = _cached_scan(str(PROJECT_ROOT))
    valids = [p for p in scan.products if p.valid]
    if valids:
        return valids[0]
    return scan.products[0] if scan.products else None


def _load_product(path):
    from core.ohrc import OHRCProduct

    try:
        return OHRCProduct.load(path, load_geometry=False)
    except Exception:  # noqa: BLE001 — overview must never crash the app
        return None


def render() -> None:
    st.subheader("Mission Overview")
    st.caption(
        "Dynamic status of the local OHRC product and the Phase 4B physics "
        "noise model.  Nothing on this page is a measurement; see the notes "
        "under each card group."
    )

    entry = _product_entry()
    cfg = OHRCNoiseModelConfig.default()

    # ---- product cards -----------------------------------------------------
    section_label("OHRC Product")
    c1, c2, c3 = st.columns(3)
    if entry is None:
        st.info(
            "No OHRC product found in the project directory.  Place an OHRC "
            "product folder (containing its .img) under the project root — it "
            "is discovered automatically."
        )
        c1.metric("Product", "—")
        c2.metric("Status", "NOT FOUND")
        c3.metric("Image size", "—")
        return

    c1.metric("Product", entry.name)
    c2.metric("Status", "CALIBRATED · VALID" if entry.valid else str(entry.status))
    shape = f"{entry.lines} × {entry.samples} px" if entry.lines else "—"
    c3.metric("Image size", shape)

    c1, c2, c3 = st.columns(3)
    pixels = entry.lines * entry.samples if (entry.lines and entry.samples) else None
    c1.metric("Pixels", f"≈ {pixels / 1e9:.3f} Gpx" if pixels else "—")
    c2.metric("Data type", entry.datatype or "UnsignedByte")
    product = _load_product(entry.path)
    level = product.metadata.processing_level if product is not None else None
    c3.metric("Processing level", level or "Level 1 / CODMAC 3")

    if not entry.valid and entry.error:
        st.warning(f"Validation reported: {entry.error}")

    # ---- scientific status cards (model-based) ------------------------------
    section_label("Scientific Status")
    fw = None
    if product is not None:
        try:
            sci = product.image.scientific_array()
            r0, c0, sz = _DARK_PATCH
            with st.spinner("Evaluating noise model on the documented dark patch..."):
                fw = estimate_flat_window_noise(sci[r0:r0 + sz, c0:c0 + sz], cfg)
        except Exception:  # noqa: BLE001
            fw = None

    if fw is not None and fw.n_windows > 0:
        rep = noise_breakdown(float(fw.mean_dn), cfg)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Shot noise", f"{rep.shot_sigma_dn:.3f} DN")
        c2.metric("Read noise", f"{rep.read_sigma_dn:.3f} DN")
        c3.metric("Quantization", f"{rep.quant_sigma_dn:.3f} DN")
        c4.metric("Total σ (model)", f"{rep.total_sigma_dn:.3f} DN")
        c5.metric("Model-based SNR", f"{rep.model_snr:.2f}")

        st.markdown(
            """
> ⚠ **MODEL-BASED ESTIMATE** — This value is *not* a measured sensor SNR.
> It is evaluated from the Phase 4B physics noise model (Poisson ⊕ read ⊕
> quantization) at the mean DN of the Phase 4A dark patch, using an INFERRED
> gain (~104 e⁻/DN).  Measured window spread in that patch: {:.4g} DN over {}
> flat windows.
""".format(fw.measured_sigma_dn, fw.n_windows)
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Dark ROI (documented)", f"{_DARK_PATCH[2]} × {_DARK_PATCH[2]} px",
                  help="Phase 4A §10.3 dark patch at full resolution.")
        c2.metric("Mean DN (dark ROI)", f"{fw.mean_dn:.3f}")
        c3.metric("Measured σ / model σ",
                  f"{fw.measured_sigma_dn:.3f} / {rep.total_sigma_dn:.3f}")
    else:
        st.info(
            "Scientific status cards appear once the local OHRC product can be "
            "opened.  All such values are MODEL-BASED ESTIMATES, not measurements."
        )

    # ---- where to go ---------------------------------------------------------
    section_label("Workspace")
    st.markdown(
        """
| Area | Purpose |
|---|---|
| 🔬 **OHRC Lab** | Full pipeline: normalization → noise → model-based SNR → baselines → LunaPSR → diagnostics |
| 🖼️ **Image Enhancement** | Classical enhancement pipelines on any loaded image |
| 🔍 **Comparison** | Original vs enhanced vs difference |
| 📊 **Quality Metrics** | No-reference metrics (descriptive only — no metric implies "better") |
| 🌫️ **Noise Analysis** | Residual/noise-map estimation views |
| 🕘 **Processing History** | Persisted experiment records |

The original scientific array is **never modified**; every displayed image is a
derived display preview, clearly separated from the full-resolution data.
"""
    )

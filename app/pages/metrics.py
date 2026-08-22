"""Tab 3: Quality Metrics (full image + ROI).

Metrics compare original and enhanced [0,1] data.  A rectangle ROI can be
selected for crater-scale evaluation.  All noise/SNR figures are proxies.
"""

from __future__ import annotations

import streamlit as st

from core.metrics.core import compute_metrics
from core.metrics.noise import estimate_noise_sigma, roi_mask
from app.ui.components import metric_comparison_table, show_warnings, metric_cards
from app.ui.rendering import annotated_preview
from app.ui.state import get_enhanced, get_image, get_roi, set_roi, image_key


def render() -> None:
    st.subheader("Quality Metrics")
    img = get_image()
    run = get_enhanced()
    if img is None:
        st.info("Load an image first.")
        return
    if run is None or not run.ok:
        st.warning("Run an enhancement first.")
        return

    orig = run.normalized_original

    c1, c2 = st.columns(2)
    c1.metric("Image", f"{orig.shape[0]}×{orig.shape[1]}")
    c2.metric("Processing ID", run.records[0].processing_id if run.records else "n/a")

    # Full-image metrics are already computed and stored by the processing run;
    # reuse them instead of recomputing on every page render.
    m_orig = run.records[0].metrics["original"]
    m_enh = run.records[0].metrics["enhanced"]
    st.markdown("### Full-image metrics")
    metric_cards(m_orig, prefix="ORIG · ")
    metric_cards(m_enh, prefix="ENH · ")
    metric_comparison_table(m_orig, m_enh)

    st.divider()
    st.markdown("### Region of Interest (ROI) analysis")

    h, w = orig.shape
    default_roi = get_roi()
    if default_roi:
        rx0, ry0, rx1, ry1 = default_roi
    else:
        rx0, ry0, rx1, ry1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rx0 = st.number_input("x0", 0, w - 1, int(rx0))
    with c2:
        ry0 = st.number_input("y0", 0, h - 1, int(ry0))
    with c3:
        rx1 = st.number_input("x1", 1, w, int(rx1))
    with c4:
        ry1 = st.number_input("y1", 1, h, int(ry1))

    if rx1 <= rx0 or ry1 <= ry0:
        st.warning("ROI must have x1 > x0 and y1 > y0.")
        return

    set_roi(rx0, ry0, rx1, ry1)

    st.image(
        annotated_preview(orig, (rx0, ry0, rx1, ry1)),
        caption="ROI shown on the original (display-only overlay; green box).",
        width="stretch",
        channels="GRAY",
    )

    mask = roi_mask(orig.shape, ry0, ry1, rx0, rx1)
    if bool(mask.sum()):
        record_id = run.records[0].processing_id if run.records else "none"
        m_roi_orig, m_roi_enh = _roi_metrics_cached(image_key(), record_id, ry0, ry1, rx0, rx1)
        st.markdown(f"#### ROI metrics (region y[{ry0}:{ry1}], x[{rx0}:{rx1}])")
        metric_comparison_table(m_roi_orig, m_roi_enh)
        st.caption("ROI metrics describe only the selected rectangle. For a crater, "
                   "choose a rectangle that contains the crater floor and rim.")
    else:
        st.warning("Empty ROI.")


@st.cache_data(show_spinner="Measuring ROI...", max_entries=16)
def _roi_metrics_cached(image_key: str, record_id: str, y0: int, y1: int, x0: int, x1: int):
    from app.ui.state import get_enhanced

    run = get_enhanced()
    if run is None or not run.ok:
        return {}, {}
    mask = roi_mask(run.normalized_original.shape, y0, y1, x0, x1)
    m_orig = compute_metrics(run.normalized_original[mask], estimate_noise_sigma(run.normalized_original[mask]))
    m_enh = compute_metrics(run.enhanced[mask])
    return m_orig.to_dict(), m_enh.to_dict()


    st.divider()
    show_warnings(run.warnings, "Scientific warnings")
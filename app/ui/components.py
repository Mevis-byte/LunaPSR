"""Reusable Streamlit components.

All widgets operate on display (uint8) arrays produced by
``app.ui.rendering``; none mutate source data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import streamlit as st

METRIC_ROWS = ["Contrast", "Entropy (bits)", "Sharpness", "Noise Proxy", "SNR Proxy (dB)"]
MEANING = {
    "Contrast": "Robust percentile contrast (P98-P2)/(P98+P2). Higher = more tonal separation.",
    "Entropy (bits)": "Shannon entropy of the display histogram (information content of the visualisation).",
    "Sharpness": "Variance of the Laplacian. Higher = more high-frequency energy (detail or noise).",
    "Noise Proxy": "Estimated noise sigma (Laplacian MAD). Engineering estimate, not calibrated.",
    "SNR Proxy (dB)": "20*log10(robust signal / noise estimate). Engineering proxy, NOT scientific SNR.",
}


def metric_comparison_table(orig: Dict, enh: Dict, roi_orig: Optional[Dict] = None, roi_enh: Optional[Dict] = None) -> None:
    """Render the ORIGINAL vs ENHANCED metric table."""
    rows = []
    for label in METRIC_ROWS:
        key_map = {
            "Contrast": "contrast",
            "Entropy (bits)": "entropy",
            "Sharpness": "sharpness",
            "Noise Proxy": "noise_proxy",
            "SNR Proxy (dB)": "snr_proxy",
        }
        key = key_map[label]
        ov, ev = float(orig[key]), float(enh[key])
        delta = ev - ov
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        rows.append({
            "Metric": label,
            "Original": round(ov, 3),
            "Enhanced": round(ev, 3),
            "Δ": f"{arrow} {delta:+.2f}",
            "Meaning": MEANING[label],
        })

    # ROI columns when provided
    if roi_orig is not None and roi_enh is not None:
        for row in rows:
            key = _key_for(row["Metric"])
            row["ROI Orig"] = round(float(roi_orig[key]) if key in roi_orig else float("nan"), 3)
            row["ROI Enh"] = round(float(roi_enh[key]) if key in roi_enh else float("nan"), 3)

    import pandas as pd

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _key_for(label: str) -> str:
    return {
        "Contrast": "contrast",
        "Entropy (bits)": "entropy",
        "Sharpness": "sharpness",
        "Noise Proxy": "noise_proxy",
        "SNR Proxy (dB)": "snr_proxy",
    }[label]


def metric_cards(metrics: Dict, prefix: str = "") -> None:
    """Render small metric cards in a single row."""
    cols = st.columns(5)
    for col, key in zip(cols, ["contrast", "entropy", "sharpness", "noise_proxy", "snr_proxy"]):
        value = float(metrics.get(key, float("nan")))
        with col:
            st.metric(label=f"{prefix}{key.replace('_', ' ').title()}", value=f"{value:.2f}")


def show_warnings(warnings: Optional[List[str]], caption: str = "Scientific warnings") -> None:
    if not warnings:
        return
    with st.expander(f"⚠️ {caption}", expanded=False):
        for w in warnings:
            st.markdown(f"- {w}")


def image_with_caption(arr: np.ndarray, caption: str, default_color: bool = False) -> None:
    st.image(arr, caption=caption, width="stretch", channels="RGB" if arr.ndim == 3 and default_color else "GRAY")


def chip(text: str) -> None:
    st.markdown(f"<span style='background:#262730;padding:2px 8px;border-radius:8px;font-size:0.8rem'>{text}</span>", unsafe_allow_html=True)
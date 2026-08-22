"""Tab 4: Noise Analysis.

Visualises original → estimated signal → high-frequency residual → estimated
noise map, plus histogram and statistics.  Everything here is explicitly an
*estimate*: without a calibrated OHRC sensor noise model we can't fully
separate sensor noise from sub-resolution surface texture.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from core.noise.estimate import analyze_noise
from core.metrics.noise import roi_mask
from app.ui.components import show_warnings
from app.ui.rendering import preview, colorize
from app.ui.state import get_enhanced, get_image, get_roi


def _display_target(run, img):
    if run and run.ok:
        return run.normalized_original
    from core.preprocessing.normalization import robust_normalize

    return robust_normalize(img.data)


def render() -> None:
    st.subheader("Noise Analysis")
    img = get_image()
    if img is None:
        st.info("Load an image first.")
        return

    run = get_enhanced()
    target = _display_target(run, img)

    st.markdown("Analysis is performed on the **original** data (display-normalised). "
                "Results are estimates, not calibrated measurements.")

    signal_sigma = st.slider("Signal-estimation smoothing σ (px)", 1, 10, 3, 1)
    edge_pct = st.slider("Edge/texture mask percentile", 80, 99, 95, 1)

    from app.ui.state import image_key

    result = _analyze_session(image_key(), signal_sigma, edge_pct)
    if result is None:
        st.warning("No analysis result available.")
        return

    # ---- visualisation ----------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.image(preview(target), caption="Original (display-normalised)", width="stretch", channels="GRAY")
    with c2:
        # Signal estimate mapped through same display stretch as target.
        st.image(preview(result.signal), caption="Estimated signal (σ={})".format(signal_sigma),
                 width="stretch", channels="GRAY")

    st.markdown("#### High-frequency residual")
    residual_disp = _residual_preview(result.residual)
    st.image(residual_disp, caption="Original − estimated signal (contains sensor noise AND micro-texture)",
             width="stretch", channels="GRAY")

    st.markdown("#### Estimated noise map (edges masked)")
    noise_disp = _noise_map_preview(result.noise)
    c1, c2 = st.columns(2)
    with c1:
        st.image(noise_disp, caption="Estimated noise map (greyscale; black = edge/texture pixels excluded)",
                 width="stretch", channels="GRAY")
    with c2:
        st.image(colorize(np.nan_to_num(result.noise, nan=0.0) / max(np.nanmax(np.abs(np.nan_to_num(result.noise, nan=0.0))), 1e-6),
                          cmap="inferno"),
                 caption="Colour view: |noise| scaled to range", width="stretch")

    st.divider()
    st.markdown("#### Statistics (estimate)")
    st.json(_format_stats(result.stats))

    st.markdown("#### Noise histogram (masked noise-map values)")
    centers = result.histogram_edges[:-1] + 0.5 * np.diff(result.histogram_edges)
    import pandas as pd

    st.bar_chart(pd.DataFrame({"intensity": centers, "count": result.histogram}).set_index("intensity"))
    st.caption("Peak near zero = regions consistent with low sensor noise; tails = texture/residual structure or events.")

    # ---- ROI ----------------------------------------------------------------
    roi = get_roi()
    st.divider()
    if roi:
        rx0, ry0, rx1, ry1 = roi
        mask_roi = roi_mask(target.shape, ry0, ry1, rx0, rx1)
        roi_vals = result.noise[mask_roi]
        roi_vals = roi_vals[np.isfinite(roi_vals)]
        if roi_vals.size:
            st.markdown(f"#### Noise inside ROI y[{ry0}:{ry1}], x[{rx0}:{rx1}]")
            cols = st.columns(4)
            cols[0].metric("n (masked)", int(roi_vals.size))
            cols[1].metric("Mean", f"{roi_vals.mean():.4f}")
            cols[2].metric("Std", f"{roi_vals.std():.4f}")
            cols[3].metric("RMS", f"{np.sqrt((roi_vals ** 2).mean()):.4f}")
            st.caption("ROI noise uses the ROI selected on the Quality Metrics tab.")

    show_warnings(
        ["Noise, SNR and residual values are ESTIMATES. A calibrated OHRC sensor "
         "noise model is required before these can be reported as measurements."],
        "Estimate disclaimer",
    )


def _analyze_session(image_key: str, signal_sigma: int, edge_pct: int):
    """Cache the analysis result in session state (no pickle overhead).

    Recomputed only when the image or the analysis parameters change.
    """
    from app.ui.state import get_enhanced, get_image
    from core.preprocessing.normalization import robust_normalize

    cache_key = ("noise_analysis", image_key, int(signal_sigma), int(edge_pct))
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached
    img = get_image()
    if img is None:
        return None
    run = get_enhanced()
    target = run.normalized_original if (run and run.ok) else robust_normalize(img.data)
    result = analyze_noise(target, signal_sigma=signal_sigma, edge_percentile=edge_pct)
    st.session_state[cache_key] = result
    return result


def _residual_preview(residual: np.ndarray, p: float = 99.5) -> np.ndarray:
    """Residual centred on zero for display."""
    a = np.asarray(residual, dtype=np.float32)
    limit = float(np.percentile(np.abs(a), p))
    if limit < 1e-9:
        limit = 1e-9
    norm = np.clip(a / (2 * limit) + 0.5, 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def _noise_map_preview(noise: np.ndarray) -> np.ndarray:
    a = np.asarray(noise, dtype=np.float32).copy()
    finite = np.isfinite(a)
    if finite.sum() == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    limit = float(np.percentile(np.abs(a[finite]), 99.5))
    if limit < 1e-9:
        limit = 1e-9
    out = np.full(a.shape, 0, dtype=np.uint8)
    vals = np.clip(np.abs(a[finite]) / limit * 255.0, 0, 255).astype(np.uint8)
    out[finite] = vals
    return out


def _format_stats(stats: dict) -> dict:
    return {k: (round(float(v), 5) if isinstance(v, float) and np.isfinite(v) else v) for k, v in stats.items()}
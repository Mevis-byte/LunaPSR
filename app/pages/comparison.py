"""Tab 2: Comparison.

Side-by-side, alpha-blend slider, difference map, flicker, and zoomed crop
inspection for crater-scale detail.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from app.ui.rendering import preview
from app.ui.state import get_enhanced, get_image


def render() -> None:
    st.subheader("Comparison")
    img = get_image()
    run = get_enhanced()
    if img is None:
        st.info("Load an image first.")
        return
    if run is None or not run.ok:
        st.warning("Run an enhancement on the 'Image Enhancement' tab first.")
        return

    orig = run.normalized_original
    enh = run.enhanced
    orig8 = preview(orig)
    enh8 = preview(enh)

    mode = st.radio(
        "Comparison mode",
        ["Side by side", "Blend slider", "Difference map", "Flicker", "Zoom inspection"],
        horizontal=True,
    )

    if mode == "Side by side":
        c1, c2 = st.columns(2)
        c1.image(orig8, caption="ORIGINAL", width="stretch", channels="GRAY")
        c2.image(enh8, caption="ENHANCED", width="stretch", channels="GRAY")

    elif mode == "Blend slider":
        alpha = st.slider("Blend original → enhanced", 0.0, 1.0, 1.0, 0.01)
        blended = (orig8.astype(np.float32) * (1 - alpha) + enh8.astype(np.float32) * alpha).astype(np.uint8)
        st.image(blended, caption=f"Blend α={alpha:.2f}", width="stretch", channels="GRAY")

    elif mode == "Difference map":
        diff = np.abs(enh - orig)
        st.image(_load_bytes_gray(diff), caption="|Enhanced − Original| (normalised)", width="stretch")
        st.caption("The difference map combines legitimate signal gain with noise/artefacts. "
                   "Bright regions deserve a sceptical look, not automatic trust.")

    elif mode == "Flicker":
        _flicker(orig8, enh8)

    elif mode == "Zoom inspection":
        _zoom(orig, enh, orig8, enh8)

    st.divider()
    st.caption("All visualisations use display-normalised [0,1] data. Source numbers are never overwritten.")


def _load_bytes_gray(arr: np.ndarray):
    import io

    from PIL import Image

    from core.preprocessing.normalization import display_stretch

    a = np.asarray(arr, dtype=np.float32)
    if a.size > 1_500_000 and a.ndim == 2:
        import cv2

        scale = np.sqrt(1_500_000 / a.size)
        a = cv2.resize(a, (max(1, int(a.shape[1] * scale)), max(1, int(a.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    norm = display_stretch(a, percentiles=(0.0, 99.0), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(norm).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _flicker(orig8: np.ndarray, enh8: np.ndarray) -> None:
    st.markdown("Press **Start** to rapidly alternate; **Stop** halts.")
    start = st.button("▶ Start flicker")
    stop = st.button("■ Stop")
    slot = st.empty()
    if start:
        st.session_state["flicker_running"] = True
    if stop:
        st.session_state["flicker_running"] = False

    if st.session_state.get("flicker_running"):
        show_orig = True
        t_end = time.time() + 2.0  # bounded burst to avoid an infinite loop
        while st.session_state.get("flicker_running") and time.time() < t_end:
            slot.image(orig8 if show_orig else enh8, caption="FLICKER", width=420, channels="GRAY")
            show_orig = not show_orig
            time.sleep(0.35)
        st.session_state["flicker_running"] = False
        slot.image(orig8, caption="Flicker ended (original)", width=420, channels="GRAY")


def _zoom(orig: np.ndarray, enh: np.ndarray, orig8: np.ndarray, enh8: np.ndarray) -> None:
    h, w = orig.shape
    max_side = 600
    c1, c2, c3 = st.columns(3)
    with c1:
        y0 = st.slider("Crop top (y0)", 0, max(h - 1, 1), max(h // 2 - max_side // 2, 0), 1)
    with c2:
        x0 = st.slider("Crop left (x0)", 0, max(w - 1, 1), 0, 1)
    with c3:
        size = st.slider("Crop size (px)", 16, min(max_side, h, w), 256, 16)

    x1 = min(x0 + size, w)
    y1 = min(y0 + size, h)
    o_crop = orig[y0:y1, x0:x1]
    e_crop = enh[y0:y1, x0:x1]

    import cv2

    o_disp = preview(o_crop)
    e_disp = preview(e_crop)
    if o_disp.shape[0] > 0:
        o_disp = cv2.resize(o_disp, (400, 400), interpolation=cv2.INTER_NEAREST)
        e_disp = cv2.resize(e_disp, (400, 400), interpolation=cv2.INTER_NEAREST)

    cc1, cc2 = st.columns(2)
    cc1.image(o_disp, caption=f"Original crop [{y0}:{y1}, {x0}:{x1}]", width=420, channels="GRAY")
    cc2.image(e_disp, caption="Enhanced crop (same region)", width=420, channels="GRAY")
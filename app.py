"""LunaPSR — Chandrayaan-2 OHRC PSR image enhancement.

Entry point:  ``streamlit run app.py``
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="LunaPSR",
    page_icon="🌑",
    layout="wide",
)


def _decode_upload(raw: bytes, name: str):
    """Decode uploaded bytes into a numpy array without touching RAM twice."""
    import io

    import cv2
    import numpy as np

    buf = np.frombuffer(raw, dtype=np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as pil:
            if pil.mode in ("P", "PO", "PA"):
                pil = pil.convert("RGBA")
            elif pil.mode in ("I", "I;16", "I;16B"):
                pil = pil.convert("I")
            arr = np.asarray(pil)
    return arr


def _image_from_array(data, bit_depth, name):
    from core.types import ImageData

    return ImageData(data=data, bit_depth=bit_depth, source_name=name)


from app.pages import (about, comparison, enhancement, history, metrics,  # noqa: E402
                       noise, ohrc, ohrc_lab, overview)
from app.ui.state import (bump_image_revision, get_image, set_image)  # noqa: E402
from app.ui.theme import apply_theme, render_header  # noqa: E402
from configs.defaults import DEFAULT_SAMPLE_PATH  # noqa: E402
from core.io import load_image  # noqa: E402

apply_theme()
render_header()

# ---------------------------------------------------------------------------
# Sidebar: image source
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
<div style='text-align:center; padding:2px 0 10px 0'>
  <div style='font-size:1.15rem;font-weight:700;letter-spacing:0.3em'>LUNAPSR</div>
  <div style='font-size:0.62rem;letter-spacing:0.12em;color:#8fa3bd'>
    CHANDRAYAAN-2 · OHRC · PSR ANALYSIS</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### 📁 Image source")

    uploaded = st.file_uploader(
        "Upload an OHRC image (PNG/JPG/TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )

    sample_label = "Use sample OHRC strip" if DEFAULT_SAMPLE_PATH.exists() else "Sample missing"

    if uploaded is not None:
        from core.types import infer_bit_depth, to_float32_grayscale

        fingerprint = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("loaded_upload") != fingerprint:
            bytes_arr = uploaded.getvalue()
            try:
                arr = _decode_upload(bytes_arr, uploaded.name)
                data = to_float32_grayscale(arr)
                img = _image_from_array(data, infer_bit_depth(arr), uploaded.name)
                if get_image() is not None and get_image().source_name != uploaded.name:
                    st.session_state.pop("enhanced", None)
                set_image(img)
                bump_image_revision()
                st.session_state["loaded_upload"] = fingerprint
                st.success(f"Loaded {uploaded.name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load '{uploaded.name}': {exc}")
        else:
            st.success(f"Loaded {uploaded.name} (cached in session)")

    elif get_image() is None:
        if DEFAULT_SAMPLE_PATH.exists():
            try:
                img = load_image(DEFAULT_SAMPLE_PATH)
                set_image(img)
                bump_image_revision()
                st.success(f"Loaded sample: {img.source_name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Sample failed to load: {exc}")
        else:
            st.warning("No sample image found. Please upload a file.")
            set_image(None)

    st.markdown("---")
    img = get_image()
    if img is not None:
        st.markdown(f"**Loaded:** `{img.source_name}`")
        st.markdown(f"{img.height} × {img.width} px · {img.bit_depth}-bit")
        if img.metadata:
            with st.expander("Parsed metadata"):
                st.json(img.metadata)
        st.caption("The original numerical data is kept internally; displayed "
                   "images are 8-bit visualisations.")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_overview, tab_lab, tab_enhance, tab_ohrc, tab_compare, tab_metrics, \
    tab_noise, tab_hist, tab_about = st.tabs(
    ["🛰️ Mission Overview", "🔬 OHRC Lab", "🖼️ Image Enhancement", "🛰️ OHRC Product",
     "🔍 Comparison", "📊 Quality Metrics", "🌫️ Noise Analysis",
     "🕘 Processing History", "📖 About / Notes"]
)

with tab_overview:
    overview.render()
with tab_lab:
    ohrc_lab.render()
with tab_enhance:
    enhancement.render()
with tab_ohrc:
    ohrc.render()
with tab_compare:
    comparison.render()
with tab_metrics:
    metrics.render()
with tab_noise:
    noise.render()
with tab_hist:
    history.render()
with tab_about:
    about.render()
"""Tab 1: Image Enhancement.

Layout: original | enhanced side-by-side, then controls, then a compact metric
report and the processing record.  The heavy computation is invoked only when
the user presses "Apply enhancement" so the history is not flooded by widget
tweaks.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from app.processing import (DENOISE_METHODS, METHODS, STAGE_OPTIONS,
                            run_enhancement)
from app.ui.components import metric_cards, show_warnings
from app.ui.rendering import preview
from app.ui.state import get_enhanced, get_image


def _default_from_resume(key: str, fallback):
    resume = st.session_state.pop("resume_pipeline", None)
    if resume and key in resume:
        return resume[key]
    return fallback


def render() -> None:
    st.subheader("Image Enhancement")
    img = get_image()
    if img is None:
        st.info("Load or upload an OHRC image in the sidebar first.")
        return

    st.markdown(
        f"**Input:** `{img.source_name}` — {img.height}×{img.width} px, "
        f"{img.bit_depth}-bit grayscale"
    )

    # --- Images ------------------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.image(
            preview(img.data),
            caption="ORIGINAL (display-stretched)",
            width="stretch",
            channels="GRAY",
        )
    with c2:
        run = get_enhanced()
        if run is not None and run.ok:
            st.image(
                preview(run.enhanced),
                caption="ENHANCED (display-stretched)",
                width="stretch",
                channels="GRAY",
            )
        else:
            st.info("Enhanced image will appear here after applying.")

    st.divider()

    # --- Controls ----------------------------------------------------------
    with st.expander("⚙️ Enhancement Controls", expanded=True):
        method_key = _default_from_resume("method", "gamma_clahe")
        method = st.selectbox(
            "Enhancement method",
            list(METHODS.keys()),
            format_func=lambda m: METHODS[m],
            index=list(METHODS.keys()).index(method_key) if method_key in METHODS else 0,
        )

        col_a, col_b, col_c = st.columns(3)
        gamma = _default_from_resume("gamma", 0.75)
        with col_a:
            gamma = st.slider("Gamma", 0.1, 3.0, float(gamma), 0.05,
                              help="<1 brightens dark pixels; >1 darkens.")

        clip_limit = _default_from_resume("clip_limit", 2.0)
        tile_size = _default_from_resume("tile_size", 8)
        with col_b:
            if method in ("clahe", "gamma_clahe", "custom"):
                clip_limit = st.slider("CLAHE clip limit", 0.5, 6.0, float(clip_limit), 0.1)
                tile_size = st.slider("CLAHE tile size", 4, 32, int(tile_size), 2)
            else:
                st.caption("CLAHE parameters inactive for this method.")

        retinex_sigma = _default_from_resume("retinex_sigma", 30.0)
        with col_c:
            if method == "retinex":
                retinex_sigma = st.slider("Retinex sigma (px)", 5, 200, int(retinex_sigma), 5)
            else:
                st.caption("Retinex parameter inactive for this method.")

        denoise = _default_from_resume("denoise", "none")
        denoise_strength = _default_from_resume("denoise_strength", 3.0)
        detail_strength = _default_from_resume("detail_strength", 0.0)
        c1, c2, c3 = st.columns(3)
        with c1:
            denoise = st.selectbox(
                "Denoising",
                list(DENOISE_METHODS.keys()),
                format_func=lambda m: DENOISE_METHODS[m],
                index=list(DENOISE_METHODS.keys()).index(denoise) if denoise in DENOISE_METHODS else 0,
            )
        with c2:
            denoise_strength = st.slider("Denoise strength", 1.0, 10.0, float(denoise_strength) if denoise != "none" else 3.0, 0.5,
                                         disabled=denoise == "none")
        with c3:
            detail_strength = st.slider("Detail enhancement (unsharp)", 0.0, 2.0, float(detail_strength), 0.1,
                                        help="Conservative unsharp masking. 0 = off.")

        custom_stages = []
        if method == "custom":
            custom_stages = st.multiselect(
                "Custom stage sequence (after normalization)",
                list(STAGE_OPTIONS.keys()),
                default=_default_from_resume("custom_stages", ["gamma", "clahe"]),
                format_func=lambda s: STAGE_OPTIONS[s],
            )

        pct_lo = _default_from_resume("pct_lo", 0.5)
        pct_hi = _default_from_resume("pct_hi", 99.5)
        c1, c2 = st.columns(2)
        with c1:
            pct_lo = st.number_input("Normalization low percentile", 0.0, 10.0, float(pct_lo), 0.1)
        with c2:
            pct_hi = st.number_input("Normalization high percentile", 90.0, 100.0, float(pct_hi), 0.1)

        notes = st.text_input("Processing note (stored in the record)", value="")

        apply = st.button("⚡ Apply enhancement", type="primary", width="stretch")

    # --- Apply & report ----------------------------------------------------
    if apply:
        params = dict(
            gamma=gamma, clip_limit=clip_limit, tile_size=tile_size, retinex_sigma=retinex_sigma,
            denoise=denoise, denoise_strength=denoise_strength,
            detail_strength=detail_strength, custom_stages=custom_stages,
            pct_lo=pct_lo, pct_hi=pct_hi, notes=notes,
        )
        with st.spinner("Enhancing and measuring image..."):
            run = run_enhancement(method, **params)
        if run.ok:
            st.session_state["enhanced"] = run
        else:
            st.error(run.error)
        run = get_enhanced()

    if (run := get_enhanced()) and run.ok:
        st.divider()
        st.markdown(f"**Processing ID:** `{run.records[0].processing_id}` · "
                    f"Pipeline: `{METHODS[run.pipeline['name']] if run.pipeline['name'] in METHODS else run.pipeline['name']}`")
        metric_cards(run.records[0].metrics["enhanced"], prefix="")
        st.caption("All metrics are computed on display-normalised [0,1] data. Noise/SNR are engineering proxies.")
        show_warnings(run.warnings, "Scientific warnings")

        _render_exports_and_confidence(run)


def _render_exports_and_confidence(run) -> None:
    from app.ui.rendering import preview

    with st.expander("📤 Exports & confidence map", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.image(preview(run.enhanced), caption="Enhanced (display preview)", width="stretch", channels="GRAY")
        with c2:
            conf_disp = preview(np.clip(np.nan_to_num(run.confidence, nan=0.0), 0.0, 1.0))
            st.image(conf_disp, caption="Confidence proxy (heuristic)", width="stretch", channels="GRAY", clamp=True)

        rec = run.records[0]
        # Encode export payloads ON demand: the byte blobs are expensive
        # (9 MP encodes) so we defer until the user asks, then cache the result
        # in session state keyed by the processing record.
        payloads = st.session_state.get(("export_payloads", rec.processing_id))
        if payloads is None:
            if st.button("⚙️ Prepare export files (PNG / TIFF / JSON)", key=f"prep_{rec.processing_id}"):
                with st.spinner("Encoding export files..."):
                    payloads = _build_payloads(rec, run)
                    st.session_state[("export_payloads", rec.processing_id)] = payloads
            else:
                st.caption("Export files are prepared on demand (saved for this session afterwards).")
                return

        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button("Enhanced PNG (8-bit display)", payloads["png"],
                               file_name=f"{rec.processing_id}_enhanced.png", mime="image/png")
        with b2:
            try:
                st.download_button("Enhanced TIFF (16-bit)", payloads["tiff"],
                                   file_name=f"{rec.processing_id}_enhanced.tiff", mime="image/tiff")
            except ValueError as exc:
                st.error(str(exc))
        with b3:
            st.download_button("Difference PNG", payloads["diff"],
                               file_name=f"{rec.processing_id}_difference.png", mime="image/png")

        b4, b5, b6 = st.columns(3)
        with b4:
            st.download_button("Confidence map PNG", payloads["conf"],
                               file_name=f"{rec.processing_id}_confidence.png", mime="image/png")
        with b5:
            st.download_button("Metrics JSON", payloads["metrics_json"],
                               file_name=f"{rec.processing_id}_metrics.json", mime="application/json")
        with b6:
            st.download_button("Processing record JSON", payloads["record_json"],
                               file_name=f"{rec.processing_id}.json", mime="application/json")


def _build_payloads(rec, run) -> dict:
    """Produce the export byte blobs."""
    from app.exporters import (difference_png_bytes, json_bytes, map_png_bytes,
                               png_bytes, tiff16_bytes)

    tiff_meta = {
        "processing_id": rec.processing_id,
        "pipeline": rec.pipeline_name,
        "source_name": rec.input_name,
        "timestamp": rec.timestamp,
    }
    return {
        "png": png_bytes(run.enhanced),
        "tiff": tiff16_bytes(run.enhanced, tiff_meta),
        "diff": difference_png_bytes(run.normalized_original, run.enhanced),
        "conf": map_png_bytes(run.confidence),
        "metrics_json": json_bytes(rec.metrics),
        "record_json": json_bytes(rec.to_dict()),
    }
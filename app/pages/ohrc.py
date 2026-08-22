"""Tab: OHRC Product ingestion and safe visualisation.

Loads a real Chandrayaan-2 OHRC PDS4 product, presents its metadata, the
scientific image (as a derived, percentile-normalised 8-bit display), basic
image statistics, a histogram comparison and a pixel inspector.

Scientific data is never modified or duplicated unnecessarily:
  - the product is loaded once per session (memory-mapped);
  - the exact full-array histogram is computed once per product;
  - percentile changes re-derive only the lightweight display mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from configs.defaults import OHRC_DISPLAY_PERCENTILES, OHRC_PRODUCT, OHRC_VIEW_FACTOR
from core.ohrc import OHRCError, OHRCProduct
from core.ohrc.display import histogram_pair, to_display_8bit
from core.ohrc.statistics import (
    exact_histogram,
    histogram_statistics,
    percentile_from_histogram,
)

logger = logging.getLogger(__name__)

_SESSION_KEY = "ohrc_product_state"


def _default_path() -> str:
    if OHRC_PRODUCT.exists():
        return str(OHRC_PRODUCT)
    return ""


def _load_product(path_str: str) -> OHRCProduct:
    state = st.session_state.get(_SESSION_KEY, {})
    key = str(Path(path_str).resolve())
    if state.get("path") == key and state.get("product") is not None:
        return state["product"]
    logger.info("Loading OHRC product from %s", key)
    product = OHRCProduct.load(key)
    build_product_cache(key, product)
    st.session_state[_SESSION_KEY]["revision"] = state.get("revision", 0) + 1
    return product


def _setup(key: str, product: OHRCProduct) -> dict:
    cache = st.session_state[_SESSION_KEY]
    for name, default in _CACHE_DEFAULTS.items():
        cache.setdefault(name, default)
    if cache.get("hist_path") != key:
        logger.info("Computing exact histogram for %s", key)
        counts, edges, _, _ = exact_histogram(product.decimate_stride(factor=1))
        cache["hist_path"] = key
        cache["hist"] = (counts, edges)
    return cache


def build_product_cache(key: str, product: OHRCProduct) -> None:
    st.session_state[_SESSION_KEY] = {
        "path": key,
        "product": product,
        "revision": 0,
        "hist_path": None,
        "hist": None,
        "blocks": {},
        "sample": None,
        "stats": None,
    }


_CACHE_DEFAULTS = {
    "hist_path": None,
    "hist": None,
    "blocks": {},
    "sample": None,
    "stats": None,
}


def _stats(cache: dict) -> dict:
    if cache.get("stats") is not None:
        return cache["stats"]
    counts, edges = cache["hist"]
    stats = histogram_statistics(counts, edges)
    stats["total_pixels"] = int(counts.sum())
    stats["valid_pixels"] = int(counts.sum())
    stats["invalid_pixels"] = 0
    cache["stats"] = stats
    return stats


def _percentile_value(cache: dict, q: float):
    counts, edges = cache["hist"]
    return percentile_from_histogram(counts, edges, q)


def _display_view(cache: dict, product: OHRCProduct, factor: int, lo_val, hi_val) -> np.ndarray:
    blocks = cache["blocks"].get(factor)
    if blocks is None:
        logger.info("Computing block-averaged view (factor=%d)", factor)
        blocks = product.decimate_average(factor=factor)
        cache["blocks"][factor] = blocks
    return to_display_8bit(blocks, lo_val, hi_val)


def _hist_sample(cache: dict, product: OHRCProduct) -> np.ndarray:
    if cache.get("sample") is None:
        logger.info("Computing stride sample for histogram")
        cache["sample"] = product.decimate_stride(factor=OHRC_VIEW_FACTOR)
    return cache["sample"]


def render() -> None:
    st.subheader("OHRC Product")

    default = _default_path()
    if default:
        st.caption(
            "Load a real Chandrayaan-2 OHRC PDS4 product (read-only). The "
            "scientific data is never modified."
        )
    path_str = st.text_input(
        "OHRC product directory",
        value=st.session_state.get("ohrc_input_path", default),
    )
    st.session_state["ohrc_input_path"] = path_str

    if st.button("Load OHRC Product", type="primary"):
        try:
            product = _load_product(path_str)
            st.success(f"Loaded: {product.path.name}")
        except OHRCError as exc:
            logger.exception("OHRC load failed")
            st.error(str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected OHRC load failure")
            st.error(
                "An unexpected error occurred while loading the OHRC product. "
                "Details have been logged to the developer log."
            )

    state = st.session_state.get(_SESSION_KEY)
    if state is None or state.get("product") is None:
        st.info("Load an OHRC product to begin.")
        return

    product: OHRCProduct = state["product"]
    cache = _setup(state["path"], product)
    _render_product_panel(product)
    _render_stats(cache)
    _render_display(cache, product)
    _render_histogram(cache, product)
    _render_pixel_inspector(cache, product)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _render_product_panel(product: OHRCProduct) -> None:
    info = product.product_info
    img = product.image

    st.divider()
    st.markdown("### OHRC Product")
    st.markdown(f"**Product:** `{info['product_name']}`")
    st.markdown(f"**Path:** `{info['product_path']}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processing Level", info["processing_level"] or "unknown")
    c2.metric("Product type", (info["product_type"] or "unknown").replace("_", " "))
    c3.metric("Lines", f"{img.lines:,}")
    c4.metric("Samples", f"{img.samples:,}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pixels", f"{int(img.lines * img.samples):,}")
    c2.metric("Datatype", img.pds_datatype)
    c3.metric("Bit depth", img.bit_depth)
    c4.metric("Signed / byte order",
              f"{'signed' if img.signed else 'unsigned'} / "
              f"{img.byte_order or 'n/a (single-byte)'}")

    units = img.units or "No radiometric unit declared"
    st.markdown(f"**Units:** {units}")
    if img.units_comment:
        st.caption(img.units_comment)

    with st.expander("Acquisition & camera", expanded=False):
        acq = info["acquisition"]
        cam = info["camera"]
        rows = [
            ("Start (UTC)", acq["start_date_time"]),
            ("Stop (UTC)", acq["stop_date_time"]),
            ("Imaging orbit", acq["imaging_orbit_number"]),
            ("Dumping orbit", acq["dumping_orbit_number"]),
            ("Spacecraft altitude", f"{acq['spacecraft_altitude_km']} km"),
            ("Target", acq["target"]),
            ("Area", acq["area"]),
            ("Instrument", cam["name"]),
            ("Band", cam["band"]),
            ("Focal length", f"{cam['focal_length_mm']} mm"),
            ("Detector pixel width", f"{cam['detector_pixel_width_um']} µm"),
            ("Pixel resolution", f"{cam['pixel_resolution_mp']} m/pixel"),
        ]
        for label, value in rows:
            st.markdown(f"- **{label}:** {value if value is not None else 'unknown'}")

    with st.expander("Observation geometry", expanded=False):
        geo = info["observation_geometry"]
        st.markdown(
            f"- Roll {geo['roll_deg']}° · Pitch {geo['pitch_deg']}° · Yaw {geo['yaw_deg']}°"
        )
        st.markdown(
            f"- Sun azimuth {geo['sun_azimuth_deg']}° · Sun elevation "
            f"{geo['sun_elevation_deg']}° · Solar incidence {geo['solar_incidence_deg']}°"
        )
        st.markdown(f"- Projection: {geo['projection'] or 'unknown'}")
        if geo["corners"]:
            st.json(geo["corners"])

    with st.expander("Calibration & integrity", expanded=False):
        cal = info["calibration"]
        st.markdown(f"**Level description:** {info['processing_level_description']}")
        st.markdown(f"**Calibration note:** {cal['calibration_note']}")
        st.markdown(f"**Label checksum (md5):** `{info['files']['md5_checksum']}`")
        browse = product.browse_summary
        if browse:
            st.markdown(
                f"**Browse product:** `{browse['path']}` — "
                f"{browse['width']} × {browse['height']} px ({browse['mode']})"
            )
        geometry = product.geometry
        if geometry is not None:
            b = geometry.bounds
            st.markdown(
                f"**Geometry grid:** {b['pixel_max'] + 1}×{b['scan_max'] + 1} px tie-point "
                f"grid; lon {b['lon_min']:.3f}°…{b['lon_max']:.3f}°, "
                f"lat {b['lat_min']:.3f}°…{b['lat_max']:.3f}°"
            )

    if st.button("Verify MD5 checksum (full read of scientific file)"):
        try:
            ok = product.verify_md5()
            if ok:
                st.success("MD5 matches the label.")
            else:
                st.error("MD5 does not match the label — product may be corrupted.")
        except OSError as exc:
            st.error(f"Could not read scientific file for verification: {exc}")


def _render_stats(cache: dict) -> None:
    stats = _stats(cache)
    st.divider()
    st.markdown("### Image statistics (scientific values)")
    st.caption(
        "Basic statistics computed from the original scientific array before "
        "any display normalisation. These are image statistics, not SNR."
    )
    cols = st.columns(7)
    cols[0].metric("Min", f"{stats['min']:.4g}")
    cols[1].metric("Max", f"{stats['max']:.4g}")
    cols[2].metric("Mean", f"{stats['mean']:.4g}")
    cols[3].metric("Median", f"{stats['median']:.4g}")
    cols[4].metric("Std", f"{stats['std']:.4g}")
    cols[5].metric("Valid px", f"{stats['valid_pixels']:,}")
    cols[6].metric("Invalid px", f"{stats['invalid_pixels']:,}")

    pcts = stats["percentiles"]
    with st.expander("Percentiles", expanded=False):
        cols = st.columns(len(pcts))
        for col, (key, value) in zip(cols, pcts.items()):
            col.metric(f"p{key}", f"{value:.4g}")


def _render_display(cache: dict, product: OHRCProduct) -> None:
    st.divider()
    st.markdown("### Image")

    lo_pct_default, hi_pct_default = OHRC_DISPLAY_PERCENTILES
    c1, c2 = st.columns(2)
    lo_pct = c1.slider("Lower display percentile", 0.0, 99.0, float(lo_pct_default), 0.1)
    hi_pct = c2.slider(
        "Upper display percentile", min(float(lo_pct) + 0.5, 100.0), 100.0,
        float(max(hi_pct_default, float(lo_pct) + 1.0)), 0.1,
    )
    if hi_pct <= lo_pct:
        hi_pct = min(lo_pct + 0.5, 100.0)

    factor_labels = {
        "1× (every pixel — heavy)": 1,
        "10× (default)": 10,
        "20×": 20,
        "40× (overview)": 40,
    }
    factor_key = st.selectbox("Display scaling", list(factor_labels.keys()), index=1)
    factor = factor_labels[factor_key]

    lo_val = _percentile_value(cache, lo_pct)
    hi_val = _percentile_value(cache, hi_pct)

    if lo_val is None or hi_val is None or hi_val <= lo_val:
        st.warning(
            "The selected percentiles do not produce a valid stretch "
            "(scientific values span too narrow a range)."
        )
        return

    view = _display_view(cache, product, factor, lo_val, hi_val)
    st.image(
        view,
        caption=(
            f"Derived 8-bit display of the scientific image "
            f"(display percentiles {lo_pct:g}–{hi_pct:g} → scientific values "
            f"{lo_val:.4g}–{hi_val:.4g}; block-averaged {factor}×). "
            f"Scaling shown as {view.shape[1]} × {view.shape[0]} px."
        ),
        width="stretch",
        channels="GRAY",
    )
    st.caption(
        "This is a derived visualization, NOT the original scientific array. "
        "Original count-calibrated values are preserved internally."
    )


def _render_histogram(cache: dict, product: OHRCProduct) -> None:
    st.divider()
    st.markdown("### Value distribution")
    lo_pct, hi_pct = OHRC_DISPLAY_PERCENTILES
    lo_val = _percentile_value(cache, lo_pct)
    hi_val = _percentile_value(cache, hi_pct)

    sample = _hist_sample(cache, product)
    display = to_display_8bit(sample, lo_val, hi_val)
    centers, (sci_density, disp_density) = histogram_pair(sample, display)

    df = pd.DataFrame(
        {
            "scientific": sci_density,
            "display": disp_density,
        },
        index=centers.round(4),
    )
    st.bar_chart(df)
    st.caption(
        "Density of scientific values vs the derived 8-bit display, computed "
        "from a stride-decimated sample of the actual scientific data "
        f"(1 of every {OHRC_VIEW_FACTOR} pixels).  The image is not modified."
    )


def _render_pixel_inspector(cache: dict, product: OHRCProduct) -> None:
    st.divider()
    st.markdown("### Pixel inspector")
    img = product.image
    default_sample = min(int(img.samples // 2), img.samples - 1)
    default_line = min(int(img.lines // 2), img.lines - 1)

    c1, c2 = st.columns(2)
    sample_x = c1.number_input(
        "X (sample column)", 0, int(img.samples) - 1, int(default_sample), 1
    )
    line_y = c2.number_input(
        "Y (line row)", 0, int(img.lines) - 1, int(default_line), 1
    )

    scientific = img.value(int(sample_x), int(line_y))
    lo_pct, hi_pct = OHRC_DISPLAY_PERCENTILES
    lo_val = _percentile_value(cache, lo_pct)
    hi_val = _percentile_value(cache, hi_pct)
    display_val = None
    if scientific is not None and lo_val is not None and hi_val is not None and hi_val > lo_val:
        display_val = float(np.clip(
            (scientific - lo_val) * (255.0 / (hi_val - lo_val)), 0.0, 255.0
        ))

    geo_text = "not available"
    if product.geometry is not None:
        ll = product.geometry.geolocate(int(sample_x), int(line_y))
        if ll is not None:
            lon, lat = ll
            geo_text = f"lon {lon:.6f}°  lat {lat:.6f}°"

    cols = st.columns(4)
    cols[0].metric("X (sample)", int(sample_x))
    cols[1].metric("Y (line)", int(line_y))
    cols[2].metric(
        "Original scientific value",
        f"{scientific}" if scientific is not None else "out of range",
    )
    cols[3].metric(
        "Display value (8-bit)",
        f"{display_val:.1f}" if display_val is not None else "n/a",
    )
    st.caption(f"Geolocation (geometry grid): {geo_text}")
    st.caption(
        "Pixel values are read directly from the original scientific array; "
        "nothing is re-scaled or stored."
    )
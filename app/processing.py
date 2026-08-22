"""Processing orchestration.

Bridges the Streamlit UI and the core algorithms.  The heavy pipeline run goes
through ``st.cache_data`` keyed on plain primitives so interactive tweaks stay
fast while results remain reproducible.

The canonical domain throughout is float32 [0,1].  Raw DN (e.g. 0..255 or
0..65535) data is mapped into [0,1] by the ``normalize`` stage; the enhanced
product stays in [0,1] and is re-scaled to 16-bit only at export time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

from configs.defaults import WARNINGS
from core.history import ProcessingRecord, utc_now_iso
from core.pipeline import EnhancementPipeline
from core.metrics.core import compute_metrics
from core.metrics.noise import estimate_noise_sigma
from core.preprocessing.normalization import robust_normalize
from core.validation import assess_processing

METHODS = {
    "gamma": "Gamma correction",
    "clahe": "Adaptive CLAHE",
    "gamma_clahe": "Gamma + CLAHE",
    "retinex": "Single Scale Retinex",
    "custom": "Custom pipeline",
}

DENOISE_METHODS = {
    "none": "None",
    "nlm": "Non-local Means",
    "gaussian": "Gaussian",
    "bilateral": "Bilateral",
}

STAGE_OPTIONS = {
    "normalize": "Normalization",
    "gamma": "Gamma",
    "clahe": "CLAHE",
    "retinex": "Single Scale Retinex",
    "denoise": "Denoising",
    "detail": "Detail enhancement",
}


@dataclass
class EnhancementRun:
    """Everything a page needs about one enhancement run."""

    ok: bool
    normalized_original: Optional[np.ndarray] = None   # [0,1]
    enhanced: Optional[np.ndarray] = None              # [0,1]
    confidence: Optional[np.ndarray] = None            # [0,1] proxy
    pipeline: Optional[dict] = None
    warnings: List[str] = None
    records: List[ProcessingRecord] = None
    error: str = ""


def _build_pipeline(
    method: str,
    gamma: float,
    clip_limit: float,
    tile_size: int,
    retinex_sigma: float,
    denoise: str,
    denoise_strength: float,
    detail_strength: float,
    normalize_percentiles: Tuple[float, float],
    custom_stages: Optional[List[str]] = None,
) -> EnhancementPipeline:
    p = EnhancementPipeline(name=method)
    p.add("normalize", percentiles=normalize_percentiles, method="percentile")

    if method == "gamma":
        p.add("gamma", gamma=gamma)
    elif method == "clahe":
        p.add("clahe", clip_limit=clip_limit, tile_size=tile_size)
    elif method == "gamma_clahe":
        p.add("gamma", gamma=gamma)
        p.add("clahe", clip_limit=clip_limit, tile_size=tile_size)
    elif method == "retinex":
        p.add("retinex", sigma=retinex_sigma)
    elif method == "custom":
        for name in custom_stages or []:
            if name == "normalize":
                continue
            if name == "gamma":
                p.add("gamma", gamma=gamma)
            elif name == "clahe":
                p.add("clahe", clip_limit=clip_limit, tile_size=tile_size)
            elif name == "retinex":
                p.add("retinex", sigma=retinex_sigma)
            elif name == "denoise":
                p.add("denoise", method=denoise, strength=denoise_strength)
            else:
                p.add(name)

    if method != "custom":
        if denoise and denoise != "none":
            p.add("denoise", method=denoise, strength=denoise_strength)
        if detail_strength > 0:
            p.add("detail", strength=detail_strength)
    return p


@st.cache_data(show_spinner="Enhancing image...", max_entries=24)
def _run_pipeline_cached(
    image_key: str, method: str, gamma: float, clip_limit: float, tile_size: int,
    retinex_sigma: float, denoise: str, denoise_strength: float, detail_strength: float,
    pct_lo: float, pct_hi: float, custom_stages: Optional[Tuple[str, ...]], raw_summary: Tuple[int, int, float, float],
) -> Tuple[np.ndarray, list, dict, list]:
    """Cached pipeline runner.

    The raw image is read from session state at call time; the cache key
    (``image_key`` = source + shape + revision) guarantees invalidation when the
    image changes.
    """
    from app.ui.state import get_image

    img = get_image()
    if img is None:
        raise RuntimeError("No image loaded.")

    p = _build_pipeline(
        method, gamma, clip_limit, tile_size, retinex_sigma,
        denoise, denoise_strength, detail_strength, (pct_lo, pct_hi),
        custom_stages=list(custom_stages) if custom_stages else None,
    )
    result = p.run(img.data)
    return result.image, result.stage_metadata_list(), result.normalization, result.warnings


def run_enhancement(method: str, **params) -> EnhancementRun:
    from app.ui.state import get_image, image_key, get_history

    img = get_image()
    if img is None:
        return EnhancementRun(ok=False, error="No image loaded.")

    h, w = img.shape
    pcts = (params.get("pct_lo", 0.5), params.get("pct_hi", 99.5))
    raw_summary = (h, w, float(img.data.min()), float(img.data.max()))
    custom_tuple = tuple(params.get("custom_stages") or [])
    key = (
        image_key(), method, params.get("gamma", 0.75), params.get("clip_limit", 2.0),
        params.get("tile_size", 8), params.get("retinex_sigma", 30.0),
        params.get("denoise", "none"), params.get("denoise_strength", 0.0),
        params.get("detail_strength", 0.0), pcts[0], pcts[1], custom_tuple, raw_summary,
    )

    try:
        enhanced, stage_metadata, normalization, warnings = _run_pipeline_cached(*key)
    except Exception as exc:  # noqa: BLE001 - surface to UI
        return EnhancementRun(ok=False, error=f"Enhancement failed: {exc}")

    normalized_orig = robust_normalize(img.data, percentiles=pcts)

    noise_sigma_orig = estimate_noise_sigma(normalized_orig)
    m_orig = compute_metrics(normalized_orig, noise_sigma_orig)
    m_enh = compute_metrics(enhanced)

    assessment = assess_processing(normalized_orig, enhanced, noise_sigma_orig)
    all_warnings = list(assessment.warnings) + warnings
    all_warnings.append(WARNINGS["display_normalization"])
    all_warnings.append(WARNINGS["calibration_unavailable"])

    from core.metrics.noise import noise_map
    from core.validation import confidence_map

    nmap = noise_map(normalized_orig)
    conf = confidence_map(normalized_orig, enhanced, nmap)
    all_warnings.append(
        "Confidence map is a heuristic proxy (local signal vs local noise); "
        "it is not a validated scientific confidence."
    )

    hist = get_history()
    rec = ProcessingRecord(
        processing_id=hist.next_id(),
        timestamp=utc_now_iso(),
        input_name=img.source_name,
        input_summary=img.summary(),
        pipeline_name=method,
        pipeline_stages=stage_metadata,
        warnings=all_warnings,
        metrics={
            "original": m_orig.to_dict(),
            "enhanced": m_enh.to_dict(),
            "artifacts": assessment.to_dict(),
        },
        notes=params.get("notes", ""),
    )
    hist.add(rec)

    run = EnhancementRun(
        ok=True,
        normalized_original=normalized_orig,
        enhanced=enhanced,
        confidence=conf,
        pipeline={"name": method, "stages": stage_metadata, "normalization": normalization},
        warnings=all_warnings,
        records=[rec],
    )
    return run
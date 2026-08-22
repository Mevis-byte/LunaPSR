"""Export helpers.

Produce in-memory byte streams for: enhanced PNG / 16-bit TIFF, difference
image, noise map, confidence map, metrics JSON and the full processing record
JSON.  Numeric data is preserved where the format allows; 8-bit previews are
explicitly labelled as visualisations.
"""

from __future__ import annotations

import io
import json
from typing import Optional

import numpy as np
from PIL import Image

from core.preprocessing.normalization import display_stretch

__all__ = [
    "png_bytes",
    "tiff16_bytes",
    "difference_png_bytes",
    "map_png_bytes",
    "json_bytes",
    "metrics_json_bytes",
]


def _png(arr8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr8)).save(buf, format="PNG")
    return buf.getvalue()


def _tiff16(arr: np.ndarray, metadata: Optional[dict] = None) -> bytes:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Cannot export an array containing NaN/Inf")
    scaled = np.clip(arr, 0.0, 1.0) * 65535.0
    buf = io.BytesIO()
    img = Image.fromarray(scaled.astype(np.uint16))
    if metadata:
        img.save(buf, format="TIFF", description=json.dumps(metadata, default=str))
    else:
        img.save(buf, format="TIFF")
    return buf.getvalue()


def png_bytes(arr_normalized: np.ndarray, percentiles=(0.5, 99.5)) -> bytes:
    """8-bit PNG visualisation of a [0,1] array."""
    return _png(display_stretch(arr_normalized, percentiles=percentiles, dtype="uint8"))


def tiff16_bytes(arr_normalized: np.ndarray, metadata: Optional[dict] = None) -> bytes:
    """16-bit TIFF of a [0,1] array (0..65535 numeric scale)."""
    return _tiff16(np.asarray(arr_normalized, dtype=np.float32), metadata)


def difference_png_bytes(orig_normalized: np.ndarray, enh_normalized: np.ndarray) -> bytes:
    """|enhanced - original| as a colour-mapped PNG."""
    import cv2

    diff = np.abs(
        np.asarray(enh_normalized, dtype=np.float32)
        - np.asarray(orig_normalized, dtype=np.float32)
    )
    diff_norm = display_stretch(diff, percentiles=(0.0, 99.0), dtype="uint8")
    color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return _png(color)


def map_png_bytes(confidence_or_noise: np.ndarray, cmap: str = "turbo") -> bytes:
    """Render a [0,1] map (NaN-safe) as a colour PNG.

    Used for confidence maps and noise maps.  NaN values render as black.
    """
    import cv2

    arr = np.asarray(confidence_or_noise, dtype=np.float32)
    nan_mask = ~np.isfinite(arr)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    frame = display_stretch(arr, percentiles=(0.0, 100.0), dtype="uint8")
    frame[nan_mask] = 0  # NaN -> black in map visualisation
    table = {
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
    }
    color = cv2.applyColorMap(frame, table.get(cmap, cv2.COLORMAP_TURBO))
    return _png(color)


def json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, indent=2, default=str).encode("utf-8")


def metrics_json_bytes(metrics: dict) -> bytes:
    return json_bytes(metrics)
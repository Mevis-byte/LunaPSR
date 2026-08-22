"""Safe display pipeline for scientific arrays.

The scientific array is never modified.  A separate, derived 8-bit display
representation is produced as follows:

    scientific image
        -> invalid (NaN / inf) handling
        -> robust percentile value mapping
        -> 8-bit visualization

Percentile *values* are supplied by the caller (e.g. from the exact histogram
of the scientific array), so the mapping can be recomputed cheaply when the
user changes percentile selections without re-reading the full scientific data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def percentile_values(counts: np.ndarray, edges: np.ndarray,
                      lower_pct: float, upper_pct: float) -> Tuple[Optional[float], Optional[float]]:
    """Map display percentiles to scientific *values* from an exact histogram."""
    lo_val = _percentile(counts, edges, lower_pct)
    hi_val = _percentile(counts, edges, upper_pct)
    return lo_val, hi_val


def _percentile(counts: np.ndarray, edges: np.ndarray, q: float) -> Optional[float]:
    from core.ohrc.statistics import percentile_from_histogram

    return percentile_from_histogram(counts, edges, q)


def to_display_8bit(
    scientific_array: np.ndarray,
    lower: float,
    upper: float,
    pixel_scale: float = 255.0,
) -> np.ndarray:
    """Derive an 8-bit display array from scientific data using value thresholds.

    ``lower`` / ``upper`` are *scientific values* (not percentages) that map to
    0 / 255.  The scientific array is not modified; NaN / inf pixels are
    replaced with 0 in the display only.  When ``lower >= upper`` an all-zero
    display is returned (no valid stretch possible).
    """
    arr = np.asarray(scientific_array, dtype=np.float32)
    if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper):
        return np.zeros(arr.shape, dtype=np.uint8)
    if upper <= lower:
        return np.zeros(arr.shape, dtype=np.uint8)

    scale = pixel_scale / (upper - lower)
    a = arr - float(lower)
    a = np.where(np.isfinite(a), a, 0.0)
    a = np.clip(a * scale, 0.0, pixel_scale)
    return a.astype(np.uint8)


def display_percentile_stretch(
    scientific_array: np.ndarray,
    counts: np.ndarray,
    edges: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Convenience: full pipeline for a normalised display image.

    Returns (display_8bit, lower_value, upper_value).  Started from an exact
    histogram so the caller can reuse it later.
    """
    lo_val, hi_val = percentile_values(counts, edges, lower_pct, upper_pct)
    display = to_display_8bit(scientific_array, lo_val, hi_val)
    return display, lo_val, hi_val


def histogram_pair(
    scientific_sample: np.ndarray,
    display_sample: np.ndarray,
    bins: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of the scientific sample vs its derived display values.

    Both histograms are normalised to density so the shapes are comparable.
    Returns (bin_centers, (scientific, display)) — arrays of equal length.

    ``display_sample`` must be the 8-bit display of ``scientific_sample``,
    otherwise the pair is not meaningful.
    """
    sci = np.asarray(scientific_sample).ravel()
    disp = np.asarray(display_sample).ravel()
    if sci.size != disp.size:
        raise ValueError("scientific and display samples must have equal size.")

    finite = np.isfinite(sci) & np.isfinite(disp)
    sci = sci[finite]
    disp = disp[finite]

    lo = float(sci.min()) if sci.size else 0.0
    hi = float(sci.max()) if sci.size else 1.0
    if hi <= lo:
        hi = lo + 1.0

    sci_counts, sci_edges = np.histogram(sci, bins=bins, range=(lo, hi))
    disp_min, disp_max = 0.0, 255.0
    disp_counts, disp_edges = np.histogram(disp, bins=bins, range=(disp_min, disp_max))

    def _density(counts, edges):
        widths = np.diff(edges)
        denom = max(float(counts.sum() * widths[0]), 1e-12)
        return counts.astype(np.float64) / denom

    sci_density = _density(sci_counts, sci_edges)
    disp_density = _density(disp_counts, disp_edges)
    centers = (sci_edges[:-1] + sci_edges[1:]) / 2.0
    return centers, (sci_density, disp_density)


def downsampled_view(
    scientific_array: np.ndarray,
    lower: float,
    upper: float,
    factor: int = 10,
) -> np.ndarray:
    """Derive a display-normalised 8-bit view, block-averaged by ``factor``.

    So a 1.2 Gpx product renders as a ~12 Mpx image without a full copy.
    """
    blocks = block_average(scientific_array, factor)
    return to_display_8bit(blocks, lower, upper)


def block_average(scientific_array: np.ndarray, factor: int = 10) -> np.ndarray:
    """Block-average decimation of scientific values (float32, no copies).

    Reads only every ``factor``-th line from the array, then averages
    ``factor``-wide groups of samples — cheap for large memory-mapped arrays.
    """
    factor = max(1, int(factor))
    arr = np.asarray(scientific_array)
    nrows, ncols = arr.shape
    lines = nrows - (nrows % factor)
    cols = ncols - (ncols % factor)
    rows = lines // factor
    if rows == 0 or cols == 0:
        return np.zeros((max(1, rows), max(1, cols // factor)), dtype=np.float32)
    step = np.asarray(arr[0:lines:factor, 0:cols], dtype=np.float32)
    cols_g = cols // factor
    return step.reshape(rows, cols_g, factor).mean(axis=2).copy()


def stride_sample(scientific_array: np.ndarray, factor: int = 10) -> np.ndarray:
    """An unbiased stride decimation of the scientific array (distribution-preserving).

    Unlike :func:`downsampled_view` this does *not* smooth; it simply keeps
    every ``factor``-th element, suitable for histograms and approximate quick
    statistics of the real data.
    """
    factor = max(1, int(factor))
    arr = np.asarray(scientific_array)
    return arr[0::factor, 0::factor]
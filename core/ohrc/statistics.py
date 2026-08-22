"""Scientific-array statistics.

Provides exact basic statistics for integer scientific arrays (the OHRC
product is 8-bit) using a chunked histogram, so mean / std / percentiles are
full-accuracy while reading a memory-mapped file block-by-block.  Float arrays
fall back to direct numpy operations (exact) and a 256-bin histogram for
display purposes.

Quantile handling follows numpy's "linear" convention so results match
``np.percentile`` on the same data.  These are *image statistics* — never SNR.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

_DEFAULT_PERCENTILES = (0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5)

_CHUNK_ROWS = 4096


def exact_histogram(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (counts, bin_edges, value_min, value_max).

    Exact full-array histogram for integer arrays, computed in row chunks so a
    1.2 Gpx memory-map is never copied wholesale.  ``bin_edges`` is one
    element longer than ``counts`` and each bin is centred on an integer
    pixel value.  Float arrays use a 256-bin numpy histogram.
    """
    arr = np.asarray(arr)
    if arr.ndim not in (1, 2):
        raise ValueError("scientific data must be 1-D or 2-D.")
    if np.issubdtype(arr.dtype, np.integer):
        lo = int(arr.min())
        hi = int(arr.max())
        n = int(hi - lo + 1)
        counts = np.zeros(n, dtype=np.int64)
        if arr.ndim == 2:
            nrows, _ = arr.shape
            for start in range(0, nrows, _CHUNK_ROWS):
                block = arr[start : start + _CHUNK_ROWS]
                vals = np.ravel(block).astype(np.int64, copy=False) - lo
                counts += np.bincount(vals, minlength=n)
        else:
            vals = arr.astype(np.int64, copy=False).ravel() - lo
            counts += np.bincount(vals, minlength=n)
        edges = np.arange(lo - 0.5, hi + 1.5, 1.0, dtype=np.float64)
        return counts, edges, lo, hi

    counts, edges = np.histogram(arr, bins=256)
    return counts.astype(np.int64), edges, float(edges[0]), float(edges[-1])


def _center_values(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def percentile_from_histogram(counts: np.ndarray, edges: np.ndarray,
                              q: float) -> Optional[float]:
    """Quantile of the data represented by an exact histogram.

    Uses linear interpolation between ranked values (numpy convention).  For
    integer histograms (integer bins) this is exact.  Returns ``None`` for an
    empty array.
    """
    total = int(counts.sum())
    if total <= 0:
        return None
    centers = _center_values(edges)
    q = max(0.0, min(100.0, float(q)))
    rank = q / 100.0 * (total - 1)
    lower = int(np.floor(rank))
    upper = min(lower + 1, total - 1)
    v_low = _value_at_rank(counts, centers, lower)
    v_high = _value_at_rank(counts, centers, upper)
    return float(v_low + (rank - lower) * (v_high - v_low))


def _value_at_rank(counts: np.ndarray, centers: np.ndarray, rank: int) -> float:
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, rank, side="right"))
    idx = min(idx, len(centers) - 1)
    return float(centers[idx])


def histogram_statistics(counts: np.ndarray, edges: np.ndarray) -> Dict[str, float]:
    """Derive exact statistics from an integer-representation histogram."""
    total = int(counts.sum())
    centers = _center_values(edges)
    empty_stats = {
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "median": 0.0,
        "percentiles": {str(q): 0.0 for q in _DEFAULT_PERCENTILES},
    }
    if total == 0:
        return empty_stats

    weights = counts.astype(np.float64) / total
    mean = float(np.dot(centers, weights))
    var = float(np.dot(centers * centers, weights)) - mean * mean
    std = float(np.sqrt(max(var, 0.0)))

    nz = np.flatnonzero(counts > 0)
    percentiles = {str(q): percentile_from_histogram(counts, edges, q)
                   for q in _DEFAULT_PERCENTILES}
    return {
        "min": float(centers[int(nz[0])]),
        "max": float(centers[int(nz[-1])]),
        "mean": mean,
        "std": std,
        "median": percentiles["50"],
        "percentiles": percentiles,
    }


def _float_statistics(arr: np.ndarray) -> Dict[str, float]:
    flat = arr.ravel().astype(np.float64, copy=False)
    finite = np.isfinite(flat)
    a = flat[finite]
    if a.size == 0:
        qs = {str(q): 0.0 for q in _DEFAULT_PERCENTILES}
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
                "median": 0.0, "percentiles": qs}
    percentiles = np.percentile(a, _DEFAULT_PERCENTILES)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.percentile(a, 50)),
        "percentiles": {str(q): float(v)
                        for q, v in zip(_DEFAULT_PERCENTILES, percentiles)},
    }


def scientific_statistics(arr: np.ndarray) -> dict:
    """Basic statistics of a scientific array.  The scientific values are used
    directly; no display normalisation is applied."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("scientific data must be 2-D (line, sample).")

    flat = arr.ravel()
    total = int(flat.size)
    if np.issubdtype(arr.dtype, np.integer):
        counts, edges, _, _ = exact_histogram(arr)
        stats = histogram_statistics(counts, edges)
        valid = int(counts.sum())
        invalid = total - valid
    else:
        finite = np.isfinite(flat)
        valid = int(finite.sum())
        invalid = total - valid
        if valid:
            counts, edges, _, _ = exact_histogram(arr)
        else:
            counts = np.zeros(1, dtype=np.int64)
            edges = np.array([-0.5, 0.5], dtype=np.float64)
        stats = _float_statistics(arr)

    stats["total_pixels"] = total
    stats["valid_pixels"] = valid
    stats["invalid_pixels"] = invalid
    stats["histogram"] = {"counts": counts, "edges": edges}
    return stats
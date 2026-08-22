"""Classical enhancement algorithms.

Convention
----------
All functions in this package operate on **normalised float32 arrays in
[0, 1]** and return float32 arrays in [0, 1].  Converting a raw data-number
(DN) array into this canonical domain is the job of the pipeline's
``normalize`` stage (see :mod:`core.preprocessing.normalization`).  Mapping a
pipeline result back to 16-bit DN for export is handled explicitly at export
time.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-6


def _check_input(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"Enhancement algorithms expect a 2-D image, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("Enhancement input contains NaN/Inf values")
    return a
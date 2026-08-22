"""Gamma correction.

Simple, transparent brightness/contrast adjustment on [0,1] data:

    out = x ** gamma

We store *gamma* directly so that ``gamma < 1`` brightens dark pixels and
``gamma > 1`` darkens them -- the convention researchers expect for very
low-light imagery (a PSR image is enhanced with a low gamma, default 0.75).

Parameters
----------
gamma : float > 0
    Values < 1 brighten, values > 1 darken the output.
"""

from __future__ import annotations

import numpy as np

from core.enhancement import EPSILON, _check_input

__all__ = ["gamma_correction", "gamma_metadata"]


def gamma_correction(arr: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    a = _check_input(arr)
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError(f"gamma must be a positive finite number, got {gamma}")
    out = np.power(np.clip(a, EPSILON, 1.0), float(gamma))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def gamma_metadata(gamma: float) -> dict:
    return {
        "stage": "gamma",
        "parameters": {"gamma": float(gamma)},
    }
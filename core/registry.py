"""Pipeline stage registry.

Maps stage names to callables (input: [0,1] float32 array -> output).  Each
stage has a ``metadata`` builder producing a reproducible record.  Registering
new algorithms here exposes them to the UI and the processing history.
"""

from __future__ import annotations

from configs.defaults import STAGE_LABELS

from core.enhancement.clahe import clahe, clahe_metadata
from core.enhancement.detail import detail_metadata, unsharp_mask
from core.enhancement.gamma import gamma_correction, gamma_metadata
from core.enhancement.retinex import retinex_metadata, single_scale_retinex
from core.denoising.filters import (denoise_bilateral,
                                     denoise_gaussian, denoise_nlm)
from core.preprocessing.normalization import robust_normalize

# Each entry: {"run": callable, "params": dict of defaults}

def _denoise_dispatch(img, method="nlm", strength=3.0):
    if method == "gaussian":
        return denoise_gaussian(img, strength=strength)
    if method == "bilateral":
        return denoise_bilateral(img, strength=strength)
    return denoise_nlm(img, strength=strength)


def _denoise_metadata(method="nlm", strength=3.0):
    return {"stage": "denoise", "parameters": {"method": method, "strength": float(strength)}}


STAGES: dict = {
    "normalize": {
        "run": lambda img, percentiles=(0.5, 99.5), method="percentile": robust_normalize(img, percentiles, method),
        "defaults": {"percentiles": (0.5, 99.5), "method": "percentile"},
        "metadata": lambda **kw: {"stage": "normalize", "parameters": {"percentiles": list(kw.get("percentiles", (0.5, 99.5))), "method": kw.get("method", "percentile")}},
    },
    "gamma": {
        "run": gamma_correction,
        "defaults": {"gamma": 0.75},
        "metadata": gamma_metadata,
    },
    "clahe": {
        "run": clahe,
        "defaults": {"clip_limit": 2.0, "tile_size": 8},
        "metadata": clahe_metadata,
    },
    "retinex": {
        "run": single_scale_retinex,
        "defaults": {"sigma": 30.0, "gain": 1.0, "offset": 0.0},
        "metadata": retinex_metadata,
    },
    "denoise": {
        "run": _denoise_dispatch,
        "defaults": {"method": "nlm", "strength": 3.0},
        "metadata": _denoise_metadata,
    },
    "detail": {
        "run": unsharp_mask,
        "defaults": {"strength": 0.5, "sigma": 1.0},
        "metadata": detail_metadata,
    },
}


def stage_parameters(name: str) -> dict:
    params = dict(STAGES[name]["defaults"])
    return params


AVAILABLE_STAGES = tuple(STAGES.keys())


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


def validate_stage(name: str) -> None:
    if name not in STAGES:
        raise KeyError(f"Unknown pipeline stage '{name}'. Available: {AVAILABLE_STAGES}")
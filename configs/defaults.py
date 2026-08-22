"""Default parameters and constants for LunaPSR.

Centralised defaults keep algorithms, the UI and the documentation in sync.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSET_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
CONFIG_DIR = PROJECT_ROOT / "configs"

# Phase 3: derived outputs live OUTSIDE the (read-only) OHRC product tree.
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_ROOT / "experiments"
PREVIEWS_DIR = OUTPUTS_ROOT / "previews"
METRICS_DIR = OUTPUTS_ROOT / "metrics"

DEFAULT_SAMPLE_PATH = ASSET_DIR / "sample_ohrc.png"

# Chandrayaan-2 OHRC PDS4 products (Phase 2).  These directories are treated as
# read-only scientific inputs; LunaPSR never modifies their contents.
OHRC_PRODUCT = PROJECT_ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18"
OHRC_DISPLAY_PERCENTILES = (1.0, 99.0)
OHRC_VIEW_FACTOR = 10

# Phase 3: automatic discovery scans the project root (project-relative only).
OHRC_SEARCH_ROOT = PROJECT_ROOT

# Phase 3: scientific (robust) normalization defaults.  Extremes must not
# dominate the visualization, so min/max is avoided by default.
ENHANCE_PERCENTILES = (1.0, 99.0)

# Phase 3: size-reduced scientific analysis representation used by the lab.
# Block-averaging preserves mean structure while bounding memory.
OHRC_ANALYSIS_FACTOR = 10

# Phase 3: classical baseline defaults (see core/enhancement/baseline.py).
BASELINE_DEFAULTS = {
    "gamma": {"gamma": 0.75},
    "clahe": {"clip_limit": 2.0, "tile_size": 8},
    "unsharp_mask": {"strength": 0.5, "sigma": 1.0},
    "denoising": {"strength": 3.0},
    "retinex": {"sigma": 30.0},
}

# Reference value used internally to represent "full scale" brightness for a
# pixel when an image has no other calibration information (raw data numbers).
# We deliberately avoid claiming this is calibrated radiance.
DEFAULT_BIT_DEPTHS = {
    "uint8": 8,
    "uint16": 16,
    "float32": 32,
}

# Display normalization used only for on-screen / exported 8-bit visualisation.
DISPLAY_PERCENTILES = (0.5, 99.5)

# Pipeline stage defaults.  Keys are the registered stage names (core.registry).
PIPELINE_DEFAULTS = {
    "normalize": {
        "percentiles": (0.5, 99.5),
        "method": "percentile",
    },
    "gamma": {
        "gamma": 0.75,
    },
    "clahe": {
        "clip_limit": 2.0,
        "tile_size": 8,
    },
    "retinex": {
        "sigma": 30.0,
        "gain": 1.0,
        "offset": 0.0,
    },
    "denoise": {
        "method": "nlm",
        "strength": 3.0,
    },
    "detail": {
        "strength": 0.5,
        "sigma": 1.0,
    },
}

# Stage ordering used by the recommended "Gamma + CLAHE" quick pipeline.
RECOMMENDED_PIPELINE = ["normalize", "gamma", "clahe"]

STAGE_LABELS = {
    "normalize": "Normalization",
    "gamma": "Gamma",
    "clahe": "CLAHE",
    "retinex": "Single Scale Retinex",
    "denoise": "Denoising",
    "detail": "Detail enhancement",
}

WARNINGS = {
    "display_normalization": "Display normalization applied for visualisation only; "
    "original numerical values are preserved internally.",
    "calibration_unavailable": "Scientific calibration / true sensor noise model is unavailable; "
    "SNR and noise figures are engineering proxies, not validated measurements.",
    "proxy_metric": "This metric is an engineering approximation (proxy) and must not be "
    "reported as a calibrated scientific measurement.",
}

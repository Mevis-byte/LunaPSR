"""Core data types for LunaPSR.

We carry images around as floating point numpy arrays.  The *original*
numerical representation is preserved in ``ImageData.data`` (e.g. 0..65535
for 16-bit data).  A separate, explicitly-labelled 8-bit visualisation array is
produced only when needed for display or export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ImageData:
    """Wrapper around a single-channel scientific image.

    ``data`` is always a float32 2-D array in the *original* data-number (DN)
    scale.  Bit depth records the native precision of the source file so we can
    restore units on export.
    """

    data: np.ndarray
    bit_depth: int = 8
    source_name: str = ""
    source_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = to_float32_grayscale(self.data)
        if self.data.ndim != 2:
            raise ValueError(
                f"ImageData expects a 2-D grayscale array, got shape {self.data.shape}"
            )
        if not np.isfinite(self.data).all():
            raise ValueError("ImageData contains NaN or Inf values")

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def max_value(self) -> float:
        """Maximum representable value for the recorded bit depth."""
        if self.bit_depth >= 32:
            return float(max(1.0, float(self.data.max())))
        return float(2 ** self.bit_depth - 1)

    def to_8bit(self, percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
        """Return a uint8 visualisation array (percentile-stretched).

        This is explicitly a *display* representation, not a replacement for
        the original numerical data.
        """
        from core.preprocessing.normalization import display_stretch

        return display_stretch(self.data, percentiles=percentiles, dtype="uint8")

    def normalized(self, percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
        """Return a float32 copy normalised to [0, 1] for algorithm input."""
        from core.preprocessing.normalization import robust_normalize

        return robust_normalize(self.data, percentiles=percentiles)

    def summary(self) -> dict:
        arr = self.data
        p = np.percentile(arr, [0, 1, 50, 99, 100])
        return {
            "shape": list(arr.shape),
            "bit_depth": self.bit_depth,
            "dtype_original": "float32",
            "min": float(p[0]),
            "p1": float(p[1]),
            "median": float(p[2]),
            "p99": float(p[3]),
            "max": float(p[4]),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "source_name": self.source_name,
        }


def to_float32_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert any common input array to a float32 2-D grayscale array.

    Accepts 2-D, 3-D (H, W) and (H, W, C) / (H, W, 1) inputs.  Colour images are
    converted to luminance with a documented, standard transform.
    """
    if arr is None:
        raise ValueError("No image data provided")
    arr = np.asarray(arr)

    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3 and (arr.shape[2] == 1 or arr.shape[2] == 2):
        out = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # BT.601 luma -- standard, reversible-in-principle mapping.
        rgb = arr.astype(np.float32)
        out = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        out = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported array shape for grayscale conversion: {arr.shape}")

    return out.astype(np.float32)


def infer_bit_depth(arr: np.ndarray) -> int:
    """Infer a native bit depth from a numpy dtype where possible."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return 8
    if arr.dtype in (np.uint16, np.int16):
        return 16
    if arr.dtype in (np.uint32, np.int32):
        return 32
    return 32

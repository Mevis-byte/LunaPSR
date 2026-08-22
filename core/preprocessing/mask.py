"""Valid-pixel mask for scientific OHRC data.

Before any enhancement we establish *which pixels are valid measurements*.
Invalid pixels are those that must not feed statistics or enhancement:

* NaN / +Inf / -Inf values (float data);
* values equal to label-declared sentinels (``missing_constant`` /
  ``invalid_constant`` / ``saturated_constant`` read from the PDS4 label's
  ``Special_Constants``);
* values outside an explicit ``valid_range`` if the caller supplies one.

**Nothing is ever guessed.**  If the label does not declare a nodata value
(the current Chandrayaan-2 product does not), no nodata sentinel is applied
and the mask reflects exactly what the metadata supports.

CRITICAL CONTRACT: ``build_valid_mask`` never copies or modifies the input
array -- it returns a boolean mask and a description of the reasons.  The
scientific array returned by the OHRC reader stays untouched.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from core.ohrc.metadata import OHRCMetadata

__all__ = [
    "build_valid_mask",
    "masked_float",
    "mask_summary",
    "count_invalid",
]


def _label_constants(metadata: Optional[OHRCMetadata]) -> Dict[str, float]:
    if metadata is None:
        return {}
    return dict(getattr(metadata.array, "special_constants", {}) or {})


def build_valid_mask(
    data: np.ndarray,
    metadata: Optional[OHRCMetadata] = None,
    special_constants: Optional[Dict[str, float]] = None,
    nodata_values: Sequence[float] = (),
    saturated_values: Sequence[float] = (),
    valid_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return ``(valid_mask, details)``.

    ``details`` explains exactly why pixels were marked invalid:

    * ``nan`` / ``positive_inf`` / ``negative_inf`` counts;
    * ``nodata`` / ``saturated`` counts from declared constants;
    * ``out_of_range`` count when ``valid_range`` is given;
    * ``total`` / ``valid`` / ``invalid`` pixel counts;
    * ``declared_sentinels`` -- the sentinel values actually used (empty if
      none were declared).

    The input array is read-only; a boolean mask is returned, never a data
    copy.
    """
    arr = np.asarray(data)
    reason_counts: Dict[str, int] = {}

    constants = dict(_label_constants(metadata))
    if special_constants is not None:
        constants.update(special_constants)

    # 1. Non-finite values (float data only).
    if np.issubdtype(arr.dtype, np.floating):
        nonfinite = ~np.isfinite(arr)
        nan = np.isnan(arr)
        pins = np.isposinf(arr)
        ninfs = np.isneginf(arr)
        reason_counts["nan"] = int(nan.sum())
        reason_counts["positive_inf"] = int(pins.sum())
        reason_counts["negative_inf"] = int(ninfs.sum())
    else:
        nonfinite = np.zeros(arr.shape, dtype=bool)
        reason_counts["nan"] = 0
        reason_counts["positive_inf"] = 0
        reason_counts["negative_inf"] = 0

    # 2. Label-declared missing / invalid sentinels (never guessed).
    nodata_values = tuple(nodata_values or ())
    if not nodata_values:
        nodata_values = tuple(
            float(v) for k, v in constants.items()
            if k in ("missing_constant", "invalid_constant", "not_applicable_constant")
        )
        nodata_reason = "missing_constant/invalid_constant"
    else:
        nodata_reason = "caller-provided nodata"

    saturated_values = tuple(saturated_values or ())
    if not saturated_values:
        saturated_values = tuple(
            float(v) for k, v in constants.items()
            if k in ("saturated_constant", "high_instrument_saturation")
        )
        saturated_reason = "saturated_constant"
    else:
        saturated_reason = "caller-provided saturated values"

    nodata_mask = np.zeros(arr.shape, dtype=bool)
    if nodata_values:
        for value in nodata_values:
            nodata_mask |= arr == value
    reason_counts["nodata"] = int(nodata_mask.sum())

    saturated_mask = np.zeros(arr.shape, dtype=bool)
    if saturated_values:
        for value in saturated_values:
            saturated_mask |= arr == value
    reason_counts["saturated"] = int(saturated_mask.sum())

    # 3. Explicit valid range.
    range_mask = np.zeros(arr.shape, dtype=bool)
    if valid_range is not None:
        lo, hi = sorted((float(valid_range[0]), float(valid_range[1])))
        range_mask = (arr < lo) | (arr > hi)
    reason_counts["out_of_range"] = int(range_mask.sum())

    valid = ~(nonfinite | nodata_mask | saturated_mask | range_mask)

    details: Dict[str, object] = {
        "total": int(arr.size),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
        "reasons": reason_counts,
        "nodata_source": nodata_reason,
        "saturated_source": saturated_reason,
        "declared_sentinels": dict(constants),
        "nodata_values": list(nodata_values),
        "saturated_values": list(saturated_values),
        "valid_range": None if valid_range is None else list(valid_range),
    }
    return valid, details


def mask_summary(valid_mask: np.ndarray, total: Optional[int] = None) -> dict:
    """Compact, serialisable summary of a mask."""
    valid = bool(np.all(valid_mask))
    n_valid = int(np.count_nonzero(valid_mask))
    total = int(total) if total is not None else int(valid_mask.size)
    return {
        "all_valid": valid,
        "valid_pixels": n_valid,
        "invalid_pixels": int(total - n_valid),
        "valid_fraction": float(n_valid / total) if total else 0.0,
    }


def count_invalid(valid_mask: np.ndarray) -> int:
    return int(np.count_nonzero(~valid_mask))


def masked_float(data: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """A private float32 copy of the data with invalid pixels set to NaN.

    Returns a **copy** (expression result), never the input buffer, so
    downstream processing can never mutate the original array.
    """
    out = np.asarray(data, dtype=np.float32).copy()
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        if m.shape != out.shape:
            raise ValueError(
                f"mask shape {m.shape} does not match data shape {out.shape}"
            )
        out[~m] = np.nan
    return out


_NODATA_NOTICE = (
    "The label declares no special constants for this product; no nodata "
    "sentinel is assumed."
)


def nodata_notice(metadata: Optional[OHRCMetadata] = None,
                  special_constants: Optional[Dict[str, float]] = None) -> str:
    """Human-readable statement of how invalid pixels were identified."""
    constants = dict(_label_constants(metadata))
    if special_constants is not None:
        constants.update(special_constants)
    if not constants:
        return _NODATA_NOTICE
    return (
        "Invalid pixels were identified only from label-declared special "
        f"constants: {constants}. No other values were assumed invalid."
    )
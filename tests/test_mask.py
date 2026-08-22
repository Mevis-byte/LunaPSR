"""Tests for the valid-pixel mask (Phase 3).

The mask must never mutate the input array, must only use label-declared
sentinels (never guessed), and must flag NaN/Inf/nodata/saturated/out-of-range
pixels exactly as documented.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.preprocessing.mask import (
    build_valid_mask,
    mask_summary,
    masked_float,
    nodata_notice,
)


def test_all_valid_when_no_sentinels_declared():
    data = np.arange(64, dtype=np.uint8).reshape(8, 8)
    before = data.copy()
    mask, details = build_valid_mask(data, metadata=None)
    assert mask.shape == data.shape
    assert mask.all()
    assert details["valid"] == 64
    assert details["invalid"] == 0
    assert details["declared_sentinels"] == {}
    assert details["nodata_values"] == []
    # original untouched
    assert np.array_equal(data, before)


def test_uses_label_declared_sentinels():
    data = np.array([[0, 1, 2], [3, 0, 5], [6, 7, 255]], dtype=np.uint8)
    mask, details = build_valid_mask(
        data,
        special_constants={"missing_constant": 0.0, "saturated_constant": 255.0},
    )
    assert not mask.all()
    assert (mask == (data != 0) & (data != 255)).all()
    assert details["reasons"]["nodata"] == 2   # two zeros
    assert details["reasons"]["saturated"] == 1
    assert details["nodata_values"] == [0.0]
    assert details["valid"] == 6


def test_flags_nan_inf_float_data():
    data = np.ones((8, 8), dtype=np.float32)
    data[0, 0] = np.nan
    data[1, 1] = np.inf
    data[2, 2] = -np.inf
    mask, details = build_valid_mask(data)
    assert details["reasons"]["nan"] == 1
    assert details["reasons"]["positive_inf"] == 1
    assert details["reasons"]["negative_inf"] == 1
    assert details["invalid"] == 3
    assert not mask[0, 0] and not mask[1, 1] and not mask[2, 2]


def test_explicit_nodata_and_range():
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask, details = build_valid_mask(data, nodata_values=[0], valid_range=(5, 12))
    assert details["reasons"]["nodata"] == 1
    assert details["reasons"]["out_of_range"] == 8   # {0..4, 13..15} (value 0 double-counted)
    assert details["valid"] == 8                     # 5..12 inclusive (8 px)
    assert details["invalid"] == 8                   # union of nodata + out-of-range
    assert not mask[0, 0]                            # value 0 -> nodata
    assert details["valid_range"] == [5, 12]


def test_nodata_notice_preserves_no_guess_constraint():
    assert "no nodata sentinel is assumed" in nodata_notice()
    assert "missing_constant" in nodata_notice(special_constants={"missing_constant": 0.0})


def test_mask_summary_counts():
    data = np.arange(64, dtype=np.uint8).reshape(8, 8)
    mask, _ = build_valid_mask(data, special_constants={"missing_constant": 0.0})
    summary = mask_summary(mask, total=64)
    assert summary["valid_pixels"] == 63
    assert summary["invalid_pixels"] == 1
    assert summary["valid_fraction"] == pytest.approx(63 / 64)


def test_masked_float_is_a_copy_never_the_input():
    data = np.arange(9, dtype=np.uint8).reshape(3, 3)
    mask, _ = build_valid_mask(data, special_constants={"missing_constant": 0.0})
    out = masked_float(data, mask)
    assert out is not data
    assert out.dtype == np.float32
    assert np.isnan(out[0, 0])          # 0 was invalid
    assert out[1, 1] == 4.0
    assert data.dtype == np.uint8       # original untouched
    assert data[0, 0] == 0


def test_mask_never_mutates_input_bytes():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(32, 32)).astype(np.uint8)
    orig = data.tobytes()
    build_valid_mask(data, nodata_values=[0], special_constants={"missing_constant": 0.0})
    assert data.tobytes() == orig
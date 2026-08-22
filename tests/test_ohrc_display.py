"""Tests for the OHRC safe-display pipeline.

Verifies that display generation never alters the scientific array, that
percentile mapping behaves as specified, and that statistics / histograms
operate on the original values.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.ohrc.display import (
    block_average,
    display_percentile_stretch,
    downsampled_view,
    histogram_pair,
    percentile_values,
    stride_sample,
    to_display_8bit,
)
from core.ohrc.statistics import (
    exact_histogram,
    percentile_from_histogram,
    scientific_statistics,
)

from conftest import build_ohrc_bundle


def test_display_does_not_modify_scientific_array():
    arr = np.array([[0, 50], [100, 255]], dtype=np.uint8)
    original = arr.copy()
    counts, edges, _, _ = exact_histogram(arr)
    display, lo, hi = display_percentile_stretch(arr, counts, edges, 0.0, 100.0)
    assert display.dtype == np.uint8
    np.testing.assert_array_equal(arr, original)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(255.0)


def test_to_display_8bit_percentile_mapping():
    arr = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    out = to_display_8bit(arr, 0.0, 255.0)
    expected = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_to_display_8bit_clips_extremes():
    arr = np.array([-100, 0, 255, 400], dtype=np.int16)
    out = to_display_8bit(arr, 0.0, 255.0)
    assert int(out[0]) == 0
    assert int(out[1]) == 0
    assert int(out[2]) == 255
    assert int(out[3]) == 255


def test_to_display_8bit_nan_handled():
    arr = np.array([[1.0, np.nan], [np.inf, 10.0]], dtype=np.float64)
    original = arr.copy()
    out = to_display_8bit(arr, 0.0, 255.0)
    assert out.shape == arr.shape
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(arr, original)


def test_to_display_8bit_invalid_stretch_returns_zero():
    arr = np.arange(10, dtype=np.uint8)
    out = to_display_8bit(arr, 5.0, 5.0)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_percentile_values_from_histogram():
    arr = np.arange(256, dtype=np.uint8)
    counts, edges, _, _ = exact_histogram(arr)
    lo, hi = percentile_values(counts, edges, 1.0, 99.0)
    assert lo == pytest.approx(2.55, abs=0.01)
    assert hi == pytest.approx(252.45, abs=0.01)


def test_stride_sample_preserves_values():
    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    sample = stride_sample(arr, factor=2)
    expected = arr[0::2, 0::2]
    np.testing.assert_array_equal(sample, expected)


def test_block_average_matches_numpy_averaging():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(40, 40)).astype(np.uint8)
    got = block_average(arr, factor=4)
    rows = arr[:40:4]
    want = rows.reshape(10, 10, 4).mean(axis=2)
    np.testing.assert_allclose(got, want, atol=1e-4)
    assert got.shape == (10, 10)


def test_display_view_preserves_scientific():
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=(200, 100)).astype(np.uint8)
    original = arr.copy()
    counts, edges, _, _ = exact_histogram(arr)
    lo, hi = percentile_values(counts, edges, 1.0, 99.0)
    view = downsampled_view(arr, lo, hi, factor=10)
    assert view.shape == (20, 10)
    assert view.dtype == np.uint8
    np.testing.assert_array_equal(arr, original)


def test_histogram_pair_compares_scientific_vs_display():
    rng = np.random.default_rng(3)
    sample = rng.integers(0, 256, size=(5000,)).astype(np.uint8)
    display = to_display_8bit(sample, 10.0, 240.0)
    centers, (sci, disp) = histogram_pair(sample, display, bins=32)
    assert centers.shape == sci.shape == disp.shape == (32,)
    assert sci.sum() > 0
    assert disp.sum() > 0


def test_statistics_from_scientific_array():
    arr = np.array([0, 1, 2, 3, 4], dtype=np.float32).reshape(1, 5)
    stats = scientific_statistics(arr)
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(4.0)
    assert stats["valid_pixels"] == 5
    assert stats["invalid_pixels"] == 0


def test_16bit_statistics():
    arr = (np.arange(100, dtype=np.uint16).reshape(10, 10) * 100)
    counts, edges, lo, hi = exact_histogram(arr)
    assert lo == 0
    assert hi == 9900
    stats = scientific_statistics(arr)
    assert stats["mean"] == pytest.approx(4950.0)
    assert stats["percentiles"]["50"] == pytest.approx(4950.0)


# ---------------------------------------------------------------------------
# Full product round-trips
# ---------------------------------------------------------------------------


def test_product_display_and_preservation(tmp_path):
    values = np.arange(80, dtype=np.uint8).reshape(10, 8)
    bundle = build_ohrc_bundle(tmp_path, lines=10, samples=8, values=values)
    from core.ohrc import OHRCProduct

    product = OHRCProduct.load(bundle["bundle"])
    arr = product.image.scientific_array()
    original = np.asarray(arr).copy()

    counts, edges, _, _ = exact_histogram(product.decimate_stride(factor=1))
    lo, hi = percentile_values(counts, edges, 0.0, 100.0)
    view = downsampled_view(product.decimate_stride(factor=1), lo, hi, factor=2)
    assert view.shape == (5, 4)
    np.testing.assert_array_equal(np.asarray(product.image.scientific_array()), original)


def test_real_product_stats_are_sane(real_ohrc_product):
    sample = real_ohrc_product.decimate_stride(factor=100)  # ~121k px
    counts, edges, lo, hi = exact_histogram(sample)
    stats = scientific_statistics(sample)
    assert stats["valid_pixels"] == sample.size
    assert 0.0 <= stats["mean"] <= 255.0
    assert lo >= 0 and hi <= 255
    p50 = percentile_from_histogram(counts, edges, 50.0)
    assert p50 == pytest.approx(stats["median"], abs=1.0)


def test_real_product_display_derivation(real_ohrc_product):
    sample = real_ohrc_product.decimate_average(factor=100)
    counts, edges, _, _ = exact_histogram(real_ohrc_product.decimate_stride(factor=100))
    lo, hi = percentile_values(counts, edges, 1.0, 99.0)
    view = downsampled_view(real_ohrc_product.decimate_stride(factor=1), lo, hi,
                            factor=100)
    assert view.dtype == np.uint8
    assert view.shape == sample.shape
    assert np.isfinite(view).all()
    # and the underlying scientific data is untouched
    arr = real_ohrc_product.image.scientific_array()
    assert arr.dtype == np.uint8
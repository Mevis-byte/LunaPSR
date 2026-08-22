"""Tests for the OHRC product reader: discovery, loading, validation."""

from __future__ import annotations

import numpy as np
import pytest

from core.ohrc import OHRCProduct
from core.ohrc.exceptions import (
    DimensionMismatchError,
    ProductDiscoveryError,
    ScientificImageNotFoundError,
)
from core.ohrc.reader import (
    array_from_bytes,
    discover_product,
    open_scientific_array,
    validate_file_size,
)

from conftest import build_ohrc_bundle


def _make(tmp_path, **kw):
    return build_ohrc_bundle(tmp_path, **kw)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_finds_all_members(tmp_path):
    bundle = _make(tmp_path, lines=10, samples=8)
    layout = discover_product(bundle["bundle"])
    assert layout.scientific_image.name.endswith(".img")
    assert layout.scientific_label.suffix == ".xml"
    assert layout.browse_image is not None
    assert layout.geometry_grid is not None
    assert layout.complete_science_chain


def test_discover_missing_root_raises(tmp_path):
    with pytest.raises(ProductDiscoveryError):
        discover_product(tmp_path / "does_not_exist")


def test_discover_empty_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ProductDiscoveryError):
        discover_product(empty)


def test_discover_no_img_raises(tmp_path):
    bundle = _make(tmp_path)
    bundle["img"].unlink()
    with pytest.raises(ScientificImageNotFoundError):
        discover_product(bundle["bundle"])


def test_discover_img_without_label_raises(tmp_path):
    bundle = _make(tmp_path)
    bundle["xml"].unlink()
    with pytest.raises(Exception) as excinfo:
        discover_product(bundle["bundle"])
    assert type(excinfo.value).__name__ == "MetadataNotFoundError"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_returns_values_unchanged(tmp_path):
    values = np.arange(60, dtype=np.uint8).reshape(10, 6)
    bundle = _make(tmp_path, lines=10, samples=6, values=values)
    product = OHRCProduct.load(bundle["bundle"])
    arr = product.image.scientific_array()
    assert arr.shape == (10, 6)
    assert arr.dtype == np.dtype("u1")
    np.testing.assert_array_equal(np.asarray(arr), values)


def test_load_16bit_big_endian(tmp_path):
    values = (np.arange(80, dtype=np.uint16).reshape(10, 8) * 257).astype(np.uint16)
    bundle = _make(tmp_path, data_type="UnsignedMSB2", lines=10, samples=8,
                   values=values)
    product = OHRCProduct.load(bundle["bundle"])
    arr = product.image.scientific_array()
    assert arr.dtype == np.dtype(">u2")
    assert product.image.bit_depth == 16
    assert product.image.byte_order == "big"
    np.testing.assert_array_equal(np.asarray(arr), values.astype(">u2"))


def test_load_16bit_little_endian_signed(tmp_path):
    values = np.arange(-40, 40, dtype=np.int16).reshape(8, 10)
    bundle = _make(tmp_path, data_type="SignedLSB2", lines=8, samples=10,
                   values=values)
    product = OHRCProduct.load(bundle["bundle"])
    arr = product.image.scientific_array()
    assert arr.dtype == np.dtype("<i2")
    assert product.image.signed is True
    assert product.image.byte_order == "little"
    np.testing.assert_array_equal(np.asarray(arr), values)


def test_load_with_offset_header(tmp_path):
    values = np.arange(24, dtype=np.uint8).reshape(6, 4)
    bundle = _make(tmp_path, lines=6, samples=4, values=values, offset=4)
    product = OHRCProduct.load(bundle["bundle"])
    np.testing.assert_array_equal(
        np.asarray(product.image.scientific_array()), values
    )


def test_truncated_image_raises(tmp_path):
    bundle = _make(tmp_path, lines=10, samples=8)
    img = bundle["img"]
    data = img.read_bytes()[:10]
    img.write_bytes(data)
    with pytest.raises(DimensionMismatchError):
        OHRCProduct.load(bundle["bundle"])


def test_extra_bytes_image_raises(tmp_path):
    bundle = _make(tmp_path, lines=10, samples=8)
    img = bundle["img"]
    img.write_bytes(img.read_bytes() + b"\x01\x02")
    with pytest.raises(DimensionMismatchError):
        OHRCProduct.load(bundle["bundle"])


def test_unsupported_datatype_raises(tmp_path):
    bundle = _make(tmp_path, data_type="MSB2UnsignedWeird")
    with pytest.raises(Exception) as excinfo:
        OHRCProduct.load(bundle["bundle"])
    assert "Unsupported" in str(excinfo.value)


def test_metadata_missing_raises(tmp_path):
    bundle = _make(tmp_path, lines=4, samples=4)
    bundle["xml"].unlink()
    with pytest.raises(Exception) as excinfo:
        OHRCProduct.load(bundle["bundle"])
    assert "metadata" in str(excinfo.value).lower()


def test_image_file_missing_raises(tmp_path):
    bundle = _make(tmp_path, lines=4, samples=4)
    bundle["img"].unlink()
    with pytest.raises(Exception) as excinfo:
        OHRCProduct.load(bundle["bundle"])
    assert "scientific image file not found" in str(excinfo.value).lower()


def test_validate_file_size_ok(tmp_path):
    bundle = _make(tmp_path, lines=10, samples=8)
    product = OHRCProduct.load(bundle["bundle"])
    validate_file_size(bundle["img"], product.metadata)  # no raise


def test_array_from_bytes_honours_layout(tmp_path):
    values = np.arange(24, dtype=np.uint8).reshape(6, 4)
    bundle = _make(tmp_path, lines=6, samples=4, values=values)
    product = OHRCProduct.load(bundle["bundle"])
    arr = array_from_bytes(bundle["img"].read_bytes(), product.metadata.array)
    np.testing.assert_array_equal(arr, values)


def test_open_scientific_array_is_memmap(tmp_path):
    bundle = _make(tmp_path, lines=10, samples=8)
    product = OHRCProduct.load(bundle["bundle"])
    arr = open_scientific_array(bundle["img"], product.metadata.array)
    assert isinstance(arr, np.memmap)


# ---------------------------------------------------------------------------
# Real product integration
# ---------------------------------------------------------------------------


def test_real_product_loads(real_ohrc_product):
    image = real_ohrc_product.image
    assert image.lines == 101075
    assert image.samples == 12000
    assert image.pds_datatype == "UnsignedByte"
    assert image.bit_depth == 8
    assert image.signed is False


def test_real_product_file_size_matches(real_ohrc_product):
    assert real_ohrc_product.layout.scientific_image.stat().st_size == 1212900000


def test_real_product_md5_matches_label(real_ohrc_product):
    assert real_ohrc_product.verify_md5() is True


def test_real_product_reads_sample_sane(real_ohrc_product):
    arr = real_ohrc_product.image.scientific_array()
    assert np.isfinite(np.asarray(arr[:: 500, :: 500])).all()
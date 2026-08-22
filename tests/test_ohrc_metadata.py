"""Tests for the OHRC PDS4 metadata parser."""

from __future__ import annotations

import pytest

from core.ohrc.exceptions import MetadataNotFoundError, UnsupportedDataTypeError
from core.ohrc.metadata import (
    OHRCMetadata,
    interpret_data_type,
    parse_scientific_label,
)

from conftest import build_ohrc_bundle


def _parse_bundle(tmp_path, **kw):
    bundle = build_ohrc_bundle(tmp_path, **kw)
    return bundle, parse_scientific_label(bundle["xml"].read_bytes())


def test_parse_basic_fields(tmp_path):
    _, meta = _parse_bundle(tmp_path, lines=10, samples=8)
    assert isinstance(meta, OHRCMetadata)
    assert meta.array.lines == 10
    assert meta.array.samples == 8
    assert meta.array.total_pixels == 80
    assert meta.processing_level == "Calibrated"
    assert meta.product_class == "Product_Observational"
    assert meta.acquisition.imaging_orbit_number == 28372
    assert meta.acquisition.area == "South Pole"
    assert meta.geometry.solar_incidence_deg == pytest.approx(85.0)
    assert meta.acquisition.start_date_time == "2026-01-03T10:05:17.6450Z"


def test_parse_data_type_interpretation():
    info = interpret_data_type("UnsignedByte")
    assert info.numpy_dtype == "u1"
    assert info.bit_depth == 8
    assert info.signed is False
    assert info.byte_order is None

    info = interpret_data_type("UnsignedLSB2")
    assert info.numpy_dtype == "<u2"
    assert info.bit_depth == 16
    assert info.byte_order == "little"

    info = interpret_data_type("SignedMSB4")
    assert info.numpy_dtype == ">i4"
    assert info.signed is True
    assert info.byte_order == "big"


def test_unsupported_data_type_raises():
    with pytest.raises(UnsupportedDataTypeError):
        interpret_data_type("ThisIsNotARealType")


def test_empty_label_raises():
    with pytest.raises(MetadataNotFoundError):
        parse_scientific_label(b"")


def test_garbage_label_raises():
    with pytest.raises(MetadataNotFoundError):
        parse_scientific_label(b"<not-xml>")


def test_missing_values_are_none(tmp_path):
    bundle = build_ohrc_bundle(tmp_path, lines=4, samples=4)
    xml = bundle["xml"].read_text()
    xml = xml.replace("<isda:solar_incidence unit=\"deg\">85.0</isda:solar_incidence>", "")
    xml = xml.replace("<isda:imaging_orbit_number>28372</isda:imaging_orbit_number>", "")
    meta = parse_scientific_label(xml.encode())
    assert meta.geometry.solar_incidence_deg is None
    assert meta.acquisition.imaging_orbit_number is None


def test_units_not_fabricated(tmp_path):
    _, meta = _parse_bundle(tmp_path, lines=4, samples=4)
    assert meta.calibration.units is None
    assert "count calibrated" in meta.calibration.units_comment.lower()


# ---------------------------------------------------------------------------
# Real product label
# ---------------------------------------------------------------------------


def test_real_label_parses(real_ohrc_product):
    meta = real_ohrc_product.metadata
    assert meta.array.lines == 101075
    assert meta.array.samples == 12000
    assert meta.array.data_type.pds_type == "UnsignedByte"
    assert meta.array.offset_bytes == 0
    assert meta.calibration.processing_level == "Calibrated"
    assert meta.acquisition.start_date_time.startswith("2026-01-03T10:05:17")
    assert meta.acquisition.imaging_orbit_number == 28372
    assert meta.camera.name is not None and "high resolution camera" in meta.camera.name.lower()
    assert meta.geometry.solar_incidence_deg is not None
    assert meta.geometry.corners["upper_left"]["lat"] is not None
    assert meta.source_file.md5_checksum == "40347cb3a97e4fce72e04526643098e1"


def test_real_product_info_summary(real_ohrc_product):
    info = real_ohrc_product.product_info
    assert info["dimensions"]["lines"] == 101075
    assert info["dimensions"]["samples"] == 12000
    assert info["dimensions"]["total_pixels"] == 101075 * 12000
    assert info["datatype"] == "UnsignedByte"
    assert info["bit_depth"] == 8
    assert info["byte_order"] is None
    assert info["units"] is None
    assert info["files"]["md5_checksum"] == "40347cb3a97e4fce72e04526643098e1"
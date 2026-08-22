"""Tests for automatic OHRC product discovery (Phase 3)."""

from __future__ import annotations

from core.ohrc.discovery import (
    DiscoveryScan,
    find_ohrc_products,
    select_product,
    validate_product,
)
from core.ohrc.reader import discover_product

from conftest import build_ohrc_bundle


def test_discovers_bundle_in_root(tmp_path):
    bundle = build_ohrc_bundle(tmp_path, lines=10, samples=8)
    scan = find_ohrc_products(root=tmp_path)
    assert isinstance(scan, DiscoveryScan)
    names = [p.name for p in scan.products]
    assert bundle["bundle"].name in names
    entry = next(p for p in scan.products if p.name == bundle["bundle"].name)
    assert entry.valid
    assert entry.status == "VALID"
    assert entry.lines == 10 and entry.samples == 8
    assert entry.datatype == "UnsignedByte"
    assert entry.extra["complete_science_chain"]


def test_single_valid_product_is_default(tmp_path):
    build_ohrc_bundle(tmp_path, lines=10, samples=8)
    scan = find_ohrc_products(root=tmp_path)
    default = scan.default()
    assert default is not None
    assert default.valid
    assert select_product(scan) == default


def test_select_by_name(tmp_path):
    bundle = build_ohrc_bundle(tmp_path, lines=12, samples=9)
    scan = find_ohrc_products(root=tmp_path)
    chosen = select_product(scan, bundle["bundle"].name)
    assert chosen is not None and chosen.path == bundle["bundle"]


def test_multiple_products_have_no_default(tmp_path):
    build_ohrc_bundle(tmp_path / "p1_sub", lines=10, samples=8)
    build_ohrc_bundle(tmp_path / "p2_sub", lines=12, samples=9)
    scan = find_ohrc_products(root=tmp_path)
    assert len(scan.valid_products) >= 2
    assert scan.default() is None


def test_empty_and_invalid_dirs_excluded(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "not_a_product").mkdir()
    (tmp_path / "not_a_product" / "notes.txt").write_text("hello")
    scan = find_ohrc_products(root=tmp_path)
    names = [p.name for p in scan.products]
    assert "empty_dir" not in names
    assert "not_a_product" not in names


def test_validate_invalid_candidate_reports_error(tmp_path):
    bad = tmp_path / "bad_product"
    bad.mkdir()
    (bad / "thing.img").write_bytes(b"\x00")
    entry = validate_product(bad)
    assert not entry.valid
    assert entry.status == "INVALID"
    assert entry.error is not None


def test_discovers_nested_product(tmp_path):
    bundle = build_ohrc_bundle(tmp_path / "nested", lines=10, samples=8)
    scan = find_ohrc_products(root=tmp_path)
    names = [p.name for p in scan.products]
    assert any("ch2_ohr" in n for n in names)
    assert any(p.path == bundle["bundle"] for p in scan.products)


def test_discovery_never_modifies_product(tmp_path):
    bundle = build_ohrc_bundle(tmp_path, lines=10, samples=8)
    before = bundle["img"].read_bytes()
    find_ohrc_products(root=tmp_path)
    after = bundle["img"].read_bytes()
    assert before == after


def test_discovered_products_load_with_phase2_reader(tmp_path):
    bundle = build_ohrc_bundle(tmp_path, lines=16, samples=12)
    scan = find_ohrc_products(root=tmp_path)
    entry = select_product(scan, bundle["bundle"].name)
    layout = discover_product(entry.path)
    assert layout.scientific_image.name.endswith(".img")


def test_parent_with_plain_data_dir_is_not_a_product(tmp_path):
    # Regression: the project root has its own data/ folder that is NOT part
    # of a product bundle -- it must never be treated as a product.
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "history").mkdir()
    (tmp_path / "data" / "history" / "notes.txt").write_text("x")
    scan = find_ohrc_products(root=tmp_path)
    assert not any(p.path == tmp_path for p in scan.products)


def test_root_with_data_dir_still_discovers_real_bundles_only(tmp_path):
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "unrelated.bin").write_bytes(b"\x00\x01")
    bundle = build_ohrc_bundle(tmp_path, lines=8, samples=6)
    scan = find_ohrc_products(root=tmp_path)
    assert [p.path for p in scan.products] == [bundle["bundle"]]
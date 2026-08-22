"""Phase 5UI — regression tests for product-free deployment environments.

Streamlit Community Cloud has no local OHRC dataset, so every code path must
degrade gracefully when discovery finds no products (GitHub issue: app crashed
with AttributeError in _render_source_selector when the product list was
empty).
"""

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def empty_environment(monkeypatch, tmp_path):
    """Point OHRC discovery at an empty directory (cloud-like)."""
    import configs.defaults as defaults
    import core.ohrc.discovery as discovery

    monkeypatch.setattr(defaults, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(discovery, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_app_boots_without_ohrc_product(empty_environment):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    assert not at.error


def test_lab_shows_guidance_and_no_crash_without_product(empty_environment):
    """The lab tab must render guidance instead of raising AttributeError."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    infos = [i.value for i in at.info]
    assert any("No valid OHRC product" in v for v in infos)


def test_overview_handles_missing_product(empty_environment):
    from app.pages import overview

    overview._cached_scan.clear()
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    mets = [m.label for m in at.metric]
    assert "Product" in mets and "Status" in mets
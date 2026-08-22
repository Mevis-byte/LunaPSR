"""Regression tests for the OHRC Streamlit tab session-state cache.

The app keeps its per-product derived data (exact histogram, block-averaged
display views, size-reduced samples, statistics) in ``st.session_state``.
_Loss_ of that cache shape while the product is still resident must trigger a
rebuild, not a ``KeyError`` (regression: loading overwrote the session dict
with only ``path``/``product``/``revision``, so ``cache["blocks"]`` vanished).
"""

from __future__ import annotations

import numpy as np
import pytest

import app.pages.ohrc as tab
from conftest import build_ohrc_bundle

_SESSION_KEY = tab._SESSION_KEY


class _FakeSession(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def app_session(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(tab.st, "session_state", session)
    return session


def test_load_product_preserves_full_cache(tmp_path, app_session):
    bundle = build_ohrc_bundle(tmp_path, lines=10, samples=8)
    product = tab._load_product(str(bundle["bundle"]))
    cache = app_session[_SESSION_KEY]
    assert cache["path"] == str(bundle["bundle"].resolve())
    assert cache["product"] is product
    assert "revision" in cache
    assert "hist" in cache and "hist_path" in cache
    assert "blocks" in cache and isinstance(cache["blocks"], dict)
    assert "sample" in cache
    assert "stats" in cache


def test_display_view_builds_block_view(tmp_path, app_session):
    bundle = build_ohrc_bundle(tmp_path, lines=16, samples=16)
    product = tab._load_product(str(bundle["bundle"]))
    cache = tab._setup(str(bundle["bundle"].resolve()), product)
    lo, hi = 0, 255
    view = tab._display_view(cache, product, factor=4, lo_val=lo, hi_val=hi)
    assert isinstance(view, np.ndarray)
    assert view.dtype == np.uint8
    assert view.shape == (4, 4)
    assert cache["blocks"].get(4) is not None


def test_stale_cache_without_blocks_self_heals(tmp_path, app_session):
    bundle = build_ohrc_bundle(tmp_path, lines=16, samples=16)
    app_session[_SESSION_KEY] = {
        "path": str(bundle["bundle"].resolve()),
        "product": tab._load_product(str(bundle["bundle"])),
    }
    cache = tab._setup(str(bundle["bundle"].resolve()), app_session[_SESSION_KEY]["product"])
    assert "blocks" in cache and "stats" in cache
    view = tab._display_view(cache, app_session[_SESSION_KEY]["product"], factor=2, lo_val=0, hi_val=255)
    assert view.shape == (8, 8)
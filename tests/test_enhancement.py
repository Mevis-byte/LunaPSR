import numpy as np
import pytest

from core.enhancement.clahe import clahe
from core.enhancement.detail import unsharp_mask
from core.enhancement.gamma import gamma_correction
from core.enhancement.retinex import single_scale_retinex
from core.preprocessing.normalization import robust_normalize


def test_gamma_brightens_dark():
    dark = np.full((32, 32), 0.2, dtype=np.float32)
    out = gamma_correction(dark, gamma=0.5)
    assert out.mean() > 0.2


def test_gamma_darkens_bright():
    bright = np.full((32, 32), 0.8, dtype=np.float32)
    out = gamma_correction(bright, gamma=2.0)
    assert out.mean() < 0.8


def test_gamma_one_is_identity():
    arr = np.random.rand(64, 64).astype(np.float32)
    out = gamma_correction(arr, gamma=1.0)
    assert np.allclose(out, arr, atol=1e-4)


def test_gamma_invalid():
    with pytest.raises(ValueError):
        gamma_correction(np.zeros((8, 8)), gamma=0)
    with pytest.raises(ValueError):
        gamma_correction(np.zeros((8, 8)), gamma=-1)


def test_clahe_improves_local_contrast():
    # Low-contrast dark scene (narrow, low DN values) should gain spread.
    base = np.linspace(0.10, 0.22, 128 * 128).reshape(128, 128).astype(np.float32)
    out = clahe(base, clip_limit=3.0, tile_size=8)
    lo_hi = np.percentile(out, [2, 98])
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert (lo_hi[1] - lo_hi[0]) > 0.3  # clearly wider than the 0.12 input spread
    assert np.isfinite(out).all()


def test_clahe_invalid():
    with pytest.raises(ValueError):
        clahe(np.zeros((16, 16)), clip_limit=0)
    with pytest.raises(ValueError):
        clahe(np.zeros((16, 16)), tile_size=0)


def test_clahe_tiny_image_does_not_crash():
    out = clahe(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    assert np.isfinite(out).all()


def test_retinex_range():
    arr = robust_normalize(np.random.rand(100, 100).astype(np.float32) * 255)
    out = single_scale_retinex(arr, sigma=10.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_retinex_uniform_image_is_flat():
    out = single_scale_retinex(np.full((64, 64), 0.5, dtype=np.float32), sigma=8.0)
    # log-uniform = log-uniform -> response near zero -> after normalisation ~constant
    assert out.std() < 1e-4


def test_unsharp_preserves_range():
    arr = np.random.rand(64, 64).astype(np.float32)
    out = unsharp_mask(arr, strength=0.5, sigma=1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_unsharp_zero_strength_identity():
    arr = np.random.rand(64, 64).astype(np.float32)
    out = unsharp_mask(arr, strength=0.0)
    assert np.allclose(out, arr, atol=1e-5)


def test_unsharp_invalid():
    with pytest.raises(ValueError):
        unsharp_mask(np.zeros((8, 8)), strength=-1)


def test_all_outputs_finite():
    rng = np.random.default_rng(0)
    arr = rng.random((96, 96)).astype(np.float32)
    for fn, kw in [
        (gamma_correction, {"gamma": 0.7}),
        (clahe, {"clip_limit": 2.0, "tile_size": 8}),
        (single_scale_retinex, {"sigma": 20.0}),
        (unsharp_mask, {"strength": 0.5, "sigma": 1.0}),
    ]:
        out = fn(arr, **kw)
        assert np.isfinite(out).all(), fn.__name__
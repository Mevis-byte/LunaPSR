import numpy as np

import pytest

from core.metrics.core import (compute_metrics, edge_preservation_index,
                               entropy_8bit, robust_contrast,
                               sharpness_laplacian)
from core.metrics.noise import (estimate_noise_sigma, noise_map, roi_mask,
                                snr_proxy)


def test_uniform_image_zero_metrics(uniform_image):
    m = compute_metrics(uniform_image)
    assert m.contrast == pytest.approx(0, abs=1e-6)
    assert m.entropy == pytest.approx(0, abs=1e-6)
    assert m.sharpness == pytest.approx(0, abs=1e-6)


def test_contrast_scale():
    low = np.full((50, 50), 0.1, dtype=np.float32)
    low[20, 20] = 0.12  # tiny spread
    wide = np.linspace(0.0, 0.6, 50 * 50).reshape(50, 50).astype(np.float32)
    assert robust_contrast(wide) > robust_contrast(low)


def test_entropy_full_split_is_one_bit():
    half = np.vstack([np.zeros((64, 64)), np.ones((64, 64))]).astype(np.float32)
    assert entropy_8bit(half) == pytest.approx(1.0, abs=0.15)


def test_sharpness_uniform_zero(black_image):
    assert sharpness_laplacian(black_image) == pytest.approx(0, abs=1e-6)


def test_noise_sigma_accuracy(noisy_image):
    sigma = estimate_noise_sigma(noisy_image)
    assert 0.01 < sigma < 0.04


def test_noise_sigma_black_zero(black_image):
    assert estimate_noise_sigma(black_image) == 0.0


def test_snr_proxy_degrades_with_more_noise():
    """On a textured scene, adding more noise must lower the SNR proxy."""
    ramp = np.broadcast_to(
        np.linspace(0.1, 0.9, 200, dtype=np.float32)[None, :], (200, 200)
    ).copy()
    # add gentle texture so the signal is non-trivial
    rng = np.random.default_rng(5)
    texture = rng.uniform(-0.05, 0.05, ramp.shape).astype(np.float32)
    clean = np.clip(ramp + texture, 0, 1)
    low_noise = np.clip(clean + rng.normal(0, 0.01, clean.shape), 0, 1).astype(np.float32)
    high_noise = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1).astype(np.float32)
    assert snr_proxy(low_noise) > snr_proxy(high_noise) + 4.0


def test_noise_map_masks_edges():
    img = np.zeros((100, 100), dtype=np.float32)
    img[:, 50:] = 0.8
    nm = noise_map(img)
    # Edge column area should have NaN (masked)
    assert np.isnan(nm[:, 50]).any() or np.isnan(nm[:, 51]).any()
    # Interior of the flat regions should be finite
    assert np.isfinite(nm[10, 10])


def test_roi_mask_shape():
    m = roi_mask((100, 100), 10, 20, 5, 15)
    assert m.shape == (100, 100)
    assert m.dtype == np.bool_


def test_edge_preservation_identical():
    img = np.random.rand(100, 100).astype(np.float32)
    assert edge_preservation_index(img, img) > 0.99


def test_metrics_no_nan(noisy_image):
    m = compute_metrics(noisy_image).to_dict()
    assert all(np.isfinite(v) for v in m.values())
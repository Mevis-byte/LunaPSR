import numpy as np

from core.denoising.filters import (denoise_bilateral, denoise_gaussian,
                                    denoise_nlm)


def test_denoisers_reduce_noise_on_flat_region(noisy_image):
    flat = np.full_like(noisy_image, 0.5)
    rng = np.random.default_rng(0)
    noisy = np.clip(flat + rng.normal(0, 0.02, flat.shape), 0, 1).astype(np.float32)

    err_before = np.abs(noisy - 0.5).mean()
    for fn in (denoise_nlm, denoise_gaussian, denoise_bilateral):
        out = fn(noisy, strength=4.0)
        err_after = np.abs(out - 0.5).mean()
        assert err_after * 1.05 < err_before, f"{fn.__name__} did not reduce noise"


def test_denoisers_preserve_shape(noisy_image):
    for fn in (denoise_nlm, denoise_gaussian, denoise_bilateral):
        out = fn(noisy_image, strength=3.0)
        assert out.shape == noisy_image.shape
        assert np.isfinite(out).all()


def test_denoisers_range(noisy_image):
    for fn in (denoise_nlm, denoise_gaussian, denoise_bilateral):
        out = fn(noisy_image, strength=3.0)
        assert out.min() >= 0.0 and out.max() <= 1.0, fn.__name__


def test_strong_denoise_does_not_erase_edges():
    """A hard step edge must survive moderate bilateral smoothing."""
    img = np.zeros((128, 128), dtype=np.float32)
    img[:, 64:] = 0.9
    rng = np.random.default_rng(2)
    noisy = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
    out = denoise_bilateral(noisy, strength=4.0)
    # edge column zone should stay high-ish on the bright side
    assert out[:, 90].mean() > 0.7
    assert out[:, 30].mean() < 0.3
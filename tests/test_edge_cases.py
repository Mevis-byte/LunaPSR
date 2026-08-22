import numpy as np
import pytest

from core.denoising.filters import denoise_nlm
from core.enhancement.clahe import clahe
from core.enhancement.gamma import gamma_correction
from core.enhancement.retinex import single_scale_retinex
from core.metrics.core import compute_metrics
from core.metrics.noise import estimate_noise_sigma
from core.pipeline import EnhancementPipeline
from core.preprocessing.normalization import robust_normalize
from core.psr.detector import ManualPSRDetector, PSRRegion, PSRDetector


def assert_no_nan_inf(arr):
    assert np.isfinite(arr).all(), "NaN/Inf present"


def test_black_image_pipeline(black_image):
    p = EnhancementPipeline()
    p.add("normalize")
    p.add("gamma")
    p.add("clahe")
    res = p.run(black_image)
    assert_no_nan_inf(res.image)
    m = compute_metrics(res.image).to_dict()
    assert all(np.isfinite(v) for v in m.values())


def test_bright_image(gradient_image):
    assert_no_nan_inf(gamma_correction(gradient_image, gamma=0.5))
    assert_no_nan_inf(clahe(gradient_image))
    assert_no_nan_inf(single_scale_retinex(gradient_image, sigma=5.0))


def test_uniform_pipeline(uniform_image):
    p = EnhancementPipeline()
    p.add("normalize")
    p.add("gamma", gamma=0.8)
    res = p.run(uniform_image)
    assert_no_nan_inf(res.image)


def test_tiny_image_everything(tiny_image):
    assert_no_nan_inf(gamma_correction(tiny_image))
    assert_no_nan_inf(clahe(tiny_image))
    assert_no_nan_inf(single_scale_retinex(tiny_image, sigma=2.0))
    assert_no_nan_inf(denoise_nlm(tiny_image))
    m = compute_metrics(tiny_image).to_dict()
    assert all(np.isfinite(v) for v in m.values())


def test_large_image(big_image):
    norm = robust_normalize(big_image)
    assert_no_nan_inf(norm)
    assert estimate_noise_sigma(norm) >= 0.0


def test_16bit_through_pipeline(image16):
    arr = image16.astype(np.float32)
    p = EnhancementPipeline()
    p.add("normalize")
    p.add("gamma", gamma=0.8)
    p.add("clahe", clip_limit=3.0, tile_size=8)
    res = p.run(arr)
    assert_no_nan_inf(res.image)
    assert 0.0 <= res.image.min() <= res.image.max() <= 1.0


def test_metrics_16bit(image16):
    norm = robust_normalize(image16.astype(np.float32))
    m = compute_metrics(norm).to_dict()
    assert all(np.isfinite(v) for v in m.values())


def test_psr_detector_interface():
    class Dummy(PSRDetector):
        def detect(self, image, metadata=None):
            return [PSRRegion(0, 0, 10, 10, label="dummy")]

    assert len(Dummy("x").detect(np.zeros((20, 20))))

    # PSRDetector is abstract: cannot instantiate it directly.
    with pytest.raises(TypeError):
        PSRDetector("abstract")


def test_manual_psr_clamped_to_bounds():
    det = ManualPSRDetector(PSRRegion(x0=-5, y0=-5, x1=999, y1=999))
    regs = det.detect(np.zeros((100, 80)))
    assert len(regs) == 1
    r = regs[0]
    assert r.x0 >= 0 and r.y0 >= 0
    assert r.x1 <= 80 and r.y1 <= 100
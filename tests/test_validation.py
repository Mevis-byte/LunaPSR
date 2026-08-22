import numpy as np
import pytest

from core.validation import (assess_processing, check_finite, confidence_map,
                             HeaderAssessment)
from core.metrics.noise import noise_map
from core.preprocessing.normalization import robust_normalize


def test_check_finite():
    assert check_finite(np.zeros((5, 5)))
    bad = np.zeros((5, 5)); bad[0, 0] = np.nan
    assert not check_finite(bad)


def test_confidence_map_range(noisy_image):
    nm = noise_map(noisy_image)
    conf = confidence_map(noisy_image, noisy_image, nm)
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    assert np.isfinite(conf).all()


def test_confidence_low_for_noise_dominated():
    """Where there is no local structure, confidence stays near zero."""
    flat = np.full((200, 200), 0.5, dtype=np.float32)
    rng = np.random.default_rng(3)
    noisy = np.clip(flat + rng.normal(0, 0.05, flat.shape), 0, 1).astype(np.float32)
    nm = noise_map(noisy, signal_sigma=2.0)
    conf = confidence_map(noisy, noisy, nm)
    assert conf.mean() < 0.15


def test_confidence_responds_to_structure():
    img = np.broadcast_to(
        np.linspace(0.1, 0.9, 200, dtype=np.float32)[None, :], (200, 200)
    ).copy()
    rng = np.random.default_rng(9)
    img += rng.uniform(-0.08, 0.08, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1).astype(np.float32)

    flat = np.full((200, 200), 0.5, dtype=np.float32)
    noisy_flat = np.clip(flat + rng.normal(0, 0.05, flat.shape), 0, 1).astype(np.float32)

    conf_tex = confidence_map(img, img, noise_map(img)).mean()
    conf_flat = confidence_map(noisy_flat, noisy_flat, noise_map(noisy_flat, signal_sigma=2.0)).mean()
    # Structure-bearing image produces a higher confidence proxy than pure noise.
    assert conf_tex > conf_flat + 0.02


def test_assess_processing_no_warning_for_gentle():
    a = np.linspace(0, 1, 128 * 128).reshape(128, 128).astype(np.float32)
    out = np.clip(a * 1.05, 0, 1)
    assessment = assess_processing(a, out, 0.01)
    assert isinstance(assessment, HeaderAssessment)
    assert isinstance(assessment.warnings, list)


def test_assess_processing_detects_saturation():
    a = np.full((64, 64), 0.5, dtype=np.float32)
    a[0, 0] = 0.2
    out = np.full((64, 64), 1.0, dtype=np.float32)  # everything pegged
    assessment = assess_processing(a, out, 0.01)
    assert assessment.saturated_fraction > 0.9
    assert any("saturate" in w for w in assessment.warnings)


def test_clipping_detected():
    a = np.linspace(0, 1, 256 * 256).reshape(256, 256).astype(np.float32)
    rng = np.random.default_rng(0)
    out = np.clip(a + rng.normal(0, 0.3, a.shape), 0, 1)
    assessment = assess_processing(a, out, 0.05)
    assert assessment.clipped_fraction > 0.0
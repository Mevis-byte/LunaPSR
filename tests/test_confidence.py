"""Phase 5 tests: model-based local confidence map (LunaPSR).

The confidence map is a *research diagnostic* built on the Phase 4B noise
model.  These tests verify its numerical stability, boundedness, monotonicity,
band classification and NaN handling -- not any physical claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
from core.enhancement.confidence import (
    ConfidenceMap,
    confidence_categories,
    local_model_confidence,
    model_based_local_snr,
    sigma_map_dn,
)

cfg = OHRCNoiseModelConfig.default()


def test_sigma_map_positive_and_has_floor():
    sig = np.array([[0.0, 3.0], [50.0, 200.0]], dtype=np.float32)
    sigma = sigma_map_dn(sig, cfg)
    assert np.isfinite(sigma).all()
    assert (sigma > 0.0).all()  # read + quantization floor guarantees sigma > 0
    floor = float(np.sqrt((cfg.read_noise_e / cfg.gain_e_per_dn) ** 2
                          + cfg.quantization_variance_dn2))
    assert sigma[0, 0] == pytest.approx(floor)


def test_snr_is_signal_over_noise():
    sig = np.array([10.0, 104.0, 0.0], dtype=np.float32)
    snr = model_based_local_snr(sig, cfg)
    sigma = sigma_map_dn(sig, cfg)
    expected = sig / sigma
    assert np.allclose(snr, expected, rtol=1e-4)
    assert snr[2] == pytest.approx(0.0)  # zero signal -> zero SNR (finite)


def test_confidence_bounded_and_monotone():
    sig = np.array([0.0, 1.0, 5.0, 30.0, 120.0, 255.0], dtype=np.float32)
    conf = local_model_confidence(sig, cfg)
    assert np.isfinite(conf).all()
    assert (conf >= 0.0).all() and (conf < 1.0).all()
    assert conf[0] == pytest.approx(0.0)
    # strictly increasing with signal
    assert np.all(np.diff(conf) > 0)


def test_confidence_equals_snr_over_1_plus_snr():
    sig = np.array([3.0, 50.0], dtype=np.float32)
    snr = model_based_local_snr(sig, cfg)
    conf = local_model_confidence(sig, cfg)
    assert np.allclose(conf, snr / (1.0 + snr))


def test_noise_floor_increases_guard():
    sig = np.array([5.0], dtype=np.float32)
    c0 = local_model_confidence(sig, cfg, noise_floor_dn=0.0)
    c1 = local_model_confidence(sig, cfg, noise_floor_dn=1.0)
    assert c1[0] < c0[0]  # a guard floor lowers confidence


def test_categories_and_bands():
    conf = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    cat = confidence_categories(conf, bands=(0.3, 0.7))
    assert cat.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        confidence_categories(conf, bands=(0.7, 0.3))


def test_categories_invalid_pixels_marked():
    conf = np.array([[0.2, np.nan], [0.8, 0.5]], dtype=np.float32)
    cat = confidence_categories(conf)
    assert cat[0, 1] == 255
    assert int(cat[cat < 255].max()) == 2


def test_confidence_map_bundle():
    sig = np.full((16, 16), 30.0, dtype=np.float32)
    cm = ConfidenceMap(
        confidence=local_model_confidence(sig, cfg),
        model_based_local_snr=model_based_local_snr(sig, cfg),
        noise_sigma_dn=sigma_map_dn(sig, cfg),
        categories=confidence_categories(local_model_confidence(sig, cfg)),
    )
    d = cm.to_dict()
    assert d["status"] == "model_based"
    assert 0.0 < d["confidence_mean"] < 1.0
    f = cm.fractions()
    assert abs(sum(f.values()) - 1.0) < 1e-6


def test_snr_nan_preserved():
    sig = np.array([[np.nan, 5.0]], dtype=np.float32)
    conf = local_model_confidence(sig, cfg)
    snr = model_based_local_snr(sig, cfg)
    assert np.isnan(conf[0, 0]) and np.isnan(snr[0, 0])
    assert np.isfinite(conf[0, 1])
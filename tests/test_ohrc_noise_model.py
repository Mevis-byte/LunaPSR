"""Phase 4B tests: physics-aware OHRC noise model (Poisson + read + 8-bit
quantization), model-based SNR, flat-window measured-vs-model validation, and
the real-product reproduction of the Phase 4A dark-patch result (DN ~ 3,
sigma ~ 0.50 DN).

All model figures are MODEL-BASED ESTIMATES; the gain (104 e-/DN) is
INFERRED, not an official OHRC calibration coefficient.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.ohrc_noise_model import (
    OHRCNoiseModelConfig,
    estimate_flat_window_noise,
    model_snr,
    noise_breakdown,
    quant_sigma_dn,
    quantization_variance_dn2,
    read_sigma_dn,
    read_variance_e,
    sigma_total_dn,
    shot_sigma_dn,
    shot_variance_e,
    variance_total_dn,
)

# ---------------------------------------------------------------------------
# Configuration / assumptions
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = OHRCNoiseModelConfig.default()
    assert cfg.type == "poisson_read_quantization"
    assert cfg.gain_e_per_dn == pytest.approx(104.0)
    assert cfg.gain_status == "inferred"
    assert cfg.read_noise_e == pytest.approx(40.0)
    assert cfg.quantization_variance_dn2 == pytest.approx(1.0 / 12.0)
    assert cfg.status == "model_based"
    assert cfg.tdi_stages == 64
    # the inferred status must be machine-readable
    d = cfg.to_dict()
    assert d["gain_e_per_dn"] == pytest.approx(104.0)
    assert d["gain_status"] == "inferred"
    assert d["status"] == "model_based"


# ---------------------------------------------------------------------------
# Per-component variance / sigma with explicit units
# ---------------------------------------------------------------------------


def test_shot_variance_is_signal():
    assert shot_variance_e(500.0) == pytest.approx(500.0)
    assert shot_variance_e(0.0) == pytest.approx(0.0)
    assert shot_variance_e(-3.0) == pytest.approx(0.0)  # clamped


def test_read_variance_is_noise_squared():
    assert read_variance_e(40.0) == pytest.approx(1600.0)


def test_quantization_variance_is_one_twelfth():
    assert quantization_variance_dn2() == pytest.approx(1.0 / 12.0)
    assert quant_sigma_dn() == pytest.approx(np.sqrt(1.0 / 12.0))


def test_shot_sigma_dn_units():
    # shot sigma in DN = sqrt(dn / gain)
    assert shot_sigma_dn(104.0, 104.0) == pytest.approx(1.0)
    assert shot_sigma_dn(0.0, 104.0) == pytest.approx(0.0)


def test_read_sigma_dn_units():
    assert read_sigma_dn(40.0, 104.0) == pytest.approx(40.0 / 104.0)


# ---------------------------------------------------------------------------
# Total variance / sigma / SNR
# ---------------------------------------------------------------------------


def test_total_variance_is_sum_of_components():
    cfg = OHRCNoiseModelConfig.default()
    dn = 50.0
    shot = shot_sigma_dn(dn, cfg.gain_e_per_dn) ** 2
    read = read_sigma_dn(cfg.read_noise_e, cfg.gain_e_per_dn) ** 2
    quant = quantization_variance_dn2()
    assert variance_total_dn(dn, cfg) == pytest.approx(shot + read + quant)
    assert sigma_total_dn(dn, cfg) == pytest.approx(np.sqrt(shot + read + quant))


def test_model_snr_equals_dn_over_sigma():
    cfg = OHRCNoiseModelConfig.default()
    for dn in (1.0, 3.06, 50.0, 255.0):
        sigma = sigma_total_dn(dn, cfg)
        assert model_snr(dn, cfg) == pytest.approx(dn / sigma)
        # electron definition must agree (signal/noise)
        signal_e = dn * cfg.gain_e_per_dn
        noise_e = cfg.gain_e_per_dn * sigma
        assert model_snr(dn, cfg) == pytest.approx(signal_e / noise_e)


def test_snr_increases_with_signal_but_is_never_std_mean():
    cfg = OHRCNoiseModelConfig.default()
    # model-based SNR is deterministic and increases with signal level;
    # it is NOT a data-derived std/mean figure.
    assert model_snr(3.0, cfg) < model_snr(50.0, cfg) < model_snr(200.0, cfg)
    assert np.isfinite(model_snr(3.0, cfg))


def test_breakdown_dict_and_status():
    cfg = OHRCNoiseModelConfig.default()
    rep = noise_breakdown(3.06, cfg)
    d = rep.to_dict()
    assert d["status"] == "model_based"
    assert d["total_sigma_dn"] == pytest.approx(sigma_total_dn(3.06, cfg), rel=1e-4)
    assert d["model_snr"] == pytest.approx(model_snr(3.06, cfg), rel=1e-3)
    # component sigmas combine in quadrature to the total
    total = (d["shot_sigma_dn"] ** 2 + d["read_sigma_dn"] ** 2 + d["quant_sigma_dn"] ** 2)
    assert np.sqrt(total) == pytest.approx(d["total_sigma_dn"], rel=1e-4)


def test_dark_dn_model_floor():
    cfg = OHRCNoiseModelConfig.default()
    # at DN 0 the shot term vanishes; the floor is read + quantization only.
    floor = sigma_total_dn(0.0, cfg)
    assert floor == pytest.approx(
        np.sqrt((40.0 / 104.0) ** 2 + 1.0 / 12.0)
    )
    assert floor > 0.0


# ---------------------------------------------------------------------------
# Flat-window measured-vs-model estimator (Phase 4A §10.5 methodology)
# ---------------------------------------------------------------------------


def _synthetic_dark_region(mean_dn: float = 3.0, size: int = 220, seed: int = 1):
    """Simulate Poisson shot + Gaussian read + 8-bit quantization at low DN."""
    cfg = OHRCNoiseModelConfig.default()
    rng = np.random.default_rng(seed)
    signal_e = rng.poisson(mean_dn * cfg.gain_e_per_dn, size=(size, size))
    electrons = signal_e + rng.normal(0.0, cfg.read_noise_e, size=(size, size))
    dn = np.round(electrons / cfg.gain_e_per_dn)
    return np.clip(dn, 0, 255).astype(np.uint8)


def test_flat_window_matches_model_on_synthetic_dark_region():
    cfg = OHRCNoiseModelConfig.default()
    region = _synthetic_dark_region(mean_dn=3.0)
    res = estimate_flat_window_noise(region, cfg, window=11, mean_max=5.0)
    assert res.n_windows > 100
    assert res.mean_dn == pytest.approx(3.0, abs=0.3)
    # measured spread of the simulation reproduces the model prediction
    assert res.measured_sigma_dn == pytest.approx(res.model_sigma_dn, rel=0.15)
    assert res.model_sigma_dn == pytest.approx(sigma_total_dn(res.mean_dn, cfg))


def test_flat_window_never_mutates_input():
    cfg = OHRCNoiseModelConfig.default()
    region = _synthetic_dark_region(mean_dn=2.0, seed=2)
    before = region.copy()
    estimate_flat_window_noise(region, cfg, window=11, mean_max=5.0)
    assert np.array_equal(region, before)


def test_flat_window_deterministic():
    cfg = OHRCNoiseModelConfig.default()
    region = _synthetic_dark_region(mean_dn=3.0, seed=3)
    a = estimate_flat_window_noise(region, cfg, window=11, mean_max=5.0)
    b = estimate_flat_window_noise(region, cfg, window=11, mean_max=5.0)
    assert a.measured_sigma_dn == pytest.approx(b.measured_sigma_dn)
    assert a.to_dict() == b.to_dict()


def test_flat_window_validates_inputs():
    cfg = OHRCNoiseModelConfig.default()
    with pytest.raises(ValueError):
        estimate_flat_window_noise(np.ones((5, 5)), cfg, window=11)
    with pytest.raises(ValueError):
        estimate_flat_window_noise(np.ones((40, 40)), cfg, mean_max=0.01)  # too bright
    with pytest.raises(ValueError):
        estimate_flat_window_noise(np.ones((40, 40), dtype=np.float64) * np.nan, cfg)


# ---------------------------------------------------------------------------
# Real-product reproduction of the Phase 4A result (DN ~ 3, sigma ~ 0.50 DN)
# ---------------------------------------------------------------------------


def test_real_product_dark_patch_validates_model(real_ohrc_product):
    """Reproduce Phase 4A §10.3/§10.5: dark patch mean ~3 DN, sigma ~0.50 DN,
    model ~0.51 DN (relative error ~3%)."""
    if real_ohrc_product is None:
        pytest.skip("real OHRC product not present")
    cfg = OHRCNoiseModelConfig.default()
    sci = real_ohrc_product.image.scientific_array()
    patch = np.asarray(sci[61750:61950, 6000:6200])
    assert patch.shape == (200, 200)
    res = estimate_flat_window_noise(patch, cfg, window=11, mean_max=5.0)
    # Phase 4A values
    assert res.mean_dn == pytest.approx(3.06, abs=0.3)
    assert res.measured_sigma_dn == pytest.approx(0.50, abs=0.15)
    assert res.model_sigma_dn == pytest.approx(0.51, abs=0.05)
    # relative error reproduces the ~3% agreement (tolerance for terrain spread)
    assert abs(res.relative_error) < 0.2
    assert isinstance(res.to_dict(), dict)


def test_real_product_model_scope_matches_label(real_ohrc_product):
    if real_ohrc_product is None:
        pytest.skip("real OHRC product not present")
    cfg = OHRCNoiseModelConfig.default()
    assert cfg.tdi_stages == 64
    tdi = real_ohrc_product.metadata.acquisition.tdi_stages
    assert tdi is not None and "64" in str(tdi)
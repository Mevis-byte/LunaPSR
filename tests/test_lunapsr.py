"""Phase 5 tests: LunaPSR adaptive enhancement.

Covers source immutability, bounded/monotonic output, adaptive strength
behaviour in low/high-signal regions, NaN handling, determinism, parameter
recording, artifact/clipping diagnostics, the ablation switch, the benchmark
framework, and an end-to-end run on the real OHRC product.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
from core.enhancement.lunapsr import (
    LunaPSRParams,
    adaptive_strength_map,
    benchmark_lunapsr,
    enhance_lunapsr,
    smoothstep,
)
from core.preprocessing.normalization import normalize_scientific

cfg = OHRCNoiseModelConfig.default()


def _scene() -> tuple:
    rng = np.random.default_rng(11)
    base = np.full((160, 160), 30.0)
    base[40:60, 40:60] = 120.0          # high-signal structure
    base[110:130, 110:130] = 0.0        # truly dark, low-confidence region
    sci = np.clip(base + rng.normal(0.0, 0.6, base.shape), 0, 255).astype(np.float32)
    norm, params = normalize_scientific(sci, percentiles=(1, 99))
    return sci, norm, params


def _default_params(**kw) -> LunaPSRParams:
    return LunaPSRParams(**kw)


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------


def test_source_immutability():
    sci, norm, _ = _scene()
    sci_b, norm_b = sci.copy(), norm.copy()
    enhance_lunapsr(sci, norm, noise_model=cfg, bounds=(norm_b.min(), norm_b.max()))
    assert np.array_equal(sci, sci_b)
    assert np.array_equal(norm, norm_b)


def test_bounded_output_and_nan_mask():
    sci, norm, _ = _scene()
    mask = np.ones(sci.shape, dtype=bool)
    mask[0:8, 0:8] = False
    norm_m = norm.copy()
    norm_m[~mask] = np.nan
    res = enhance_lunapsr(sci, norm_m, mask=mask, noise_model=cfg)
    valid = np.isfinite(res.enhanced)
    assert (res.enhanced[valid] >= 0.0).all() and (res.enhanced[valid] <= 1.0).all()
    assert np.isnan(res.enhanced[~mask]).all()
    assert res.enhanced.shape == sci.shape


def test_masked_pixels_stay_nan_after_enhancement():
    sci, norm, _ = _scene()
    mask = np.ones(sci.shape, dtype=bool)
    mask[50:54, 50:54] = False
    norm_m = norm.copy()
    norm_m[~mask] = np.nan
    res = enhance_lunapsr(sci, norm_m, mask=mask, noise_model=cfg)
    assert np.isnan(res.enhanced[~mask]).all()


def test_deterministic():
    sci, norm, _ = _scene()
    a = enhance_lunapsr(sci, norm, noise_model=cfg)
    b = enhance_lunapsr(sci, norm, noise_model=cfg)
    assert np.array_equal(a.enhanced, b.enhanced)
    assert np.array_equal(a.confidence, b.confidence)


def test_parameter_recording():
    sci, norm, _ = _scene()
    p = _default_params(components="full", local_window=7)
    res = enhance_lunapsr(sci, norm, noise_model=cfg, params=p)
    d = res.parameters
    assert d["version"] == p.version
    assert d["components"] == "full"
    assert d["local_window"] == 7
    assert res.metadata["method"] == "lunapsr"
    assert res.status == "model_based"


# ---------------------------------------------------------------------------
# Tone curve: monotonic, bounded
# ---------------------------------------------------------------------------


def test_tone_only_is_monotonic():
    sci, norm, _ = _scene()
    p = _default_params(components="tone_only")
    res = enhance_lunapsr(sci, norm, noise_model=cfg, params=p)
    v = np.isfinite(res.enhanced)
    order = np.argsort(norm[v])
    # ordering of pixel intensities is preserved
    assert np.all(np.diff(res.enhanced[v][order]) >= -1e-6)


def test_gamma_bounded():
    sci, norm, _ = _scene()
    p = _default_params()
    res = enhance_lunapsr(sci, norm, noise_model=cfg, params=p)
    g = res.gamma_map[np.isfinite(res.gamma_map)]
    assert g.min() >= p.gamma_min - 1e-6
    assert g.max() <= 1.0


def test_no_unexpected_clipping():
    sci, norm, _ = _scene()
    p = _default_params()
    res = enhance_lunapsr(sci, norm, noise_model=cfg, params=p)
    clip = res.diagnostics.artifacts
    assert clip["new_clipping"] < 0.05


# ---------------------------------------------------------------------------
# Adaptive strength behaviour
# ---------------------------------------------------------------------------


def test_low_signal_conservative_high_signal_strong():
    sci, norm, _ = _scene()
    res = enhance_lunapsr(sci, norm, noise_model=cfg)
    low = res.strength_map[110:130, 110:130]
    high = res.strength_map[40:60, 40:60]
    assert float(np.mean(low)) < float(np.mean(high))
    # low-confidence region is conservative (strength well below the max)
    assert float(np.mean(low)) < 0.7


def test_strength_map_bounded():
    sci, norm, _ = _scene()
    res = enhance_lunapsr(sci, norm, noise_model=cfg)
    s = res.strength_map[np.isfinite(res.strength_map)]
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_adaptive_strength_monotone_and_bounded():
    p = _default_params()
    conf = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    s = adaptive_strength_map(conf, p)
    assert np.all(np.diff(s) >= -1e-6)  # non-decreasing
    assert s.min() >= p.min_strength - 1e-6
    assert s.max() <= p.max_strength + 1e-6


def test_smoothstep_continuous_bounded():
    t = np.linspace(-1, 2, 100)
    s = smoothstep(t)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[0] == 0.0 and s[-1] == 1.0
    assert np.all(np.diff(s) >= 0.0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_noise_amplification_ratio_works():
    sci, norm, _ = _scene()
    res = enhance_lunapsr(sci, norm, noise_model=cfg)
    na = res.diagnostics.noise_amplification
    assert na["laplacian_ratio"] > 0.0
    assert isinstance(na["flat_window"], dict)


def test_flat_window_note_when_no_bounds():
    sci, norm, _ = _scene()
    res = enhance_lunapsr(sci, norm, noise_model=cfg)  # no bounds
    fw = res.diagnostics.noise_amplification["flat_window"]
    assert fw["note"].startswith("Not computed")


def test_structure_preservation_metrics_shapes():
    sci, norm, _ = _scene()
    res = enhance_lunapsr(sci, norm, noise_model=cfg)
    sp = res.diagnostics.structure_preservation
    for k in ("edge_preservation_index", "gradient_orientation_similarity",
              "edge_displacement_px", "structure_correlation"):
        assert k in sp


def test_artifact_diagnostics_keys():
    sci, norm, _ = _scene()
    res = enhance_lunapsr(sci, norm, noise_model=cfg)
    art = res.diagnostics.artifacts
    for k in ("clipping_fraction_before", "clipping_fraction_after", "new_clipping",
              "isolated_pixel_amplification", "high_frequency_energy_ratio",
              "halo_score", "ringing_proxy", "warnings"):
        assert k in art


def test_score_is_research_diagnostic():
    sci, norm, _ = _scene()
    res = enhance_lunapsr(sci, norm, noise_model=cfg)
    assert res.diagnostics.score["status"] == "research_diagnostic"
    assert "score" in res.diagnostics.score


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def test_ablation_tone_only_has_no_contrast():
    sci, norm, _ = _scene()
    p = _default_params(components="tone_only")
    res = enhance_lunapsr(sci, norm, noise_model=cfg, params=p)
    # tone_only: enhanced equals pure tone curve at constant gamma
    gamma = float(np.nanmean(res.gamma_map))
    expected = np.power(norm, gamma)
    assert np.allclose(res.enhanced, expected, atol=1e-6)


def test_ablation_full_differs_from_adaptive_contrast():
    sci, norm, _ = _scene()
    a = enhance_lunapsr(sci, norm, noise_model=cfg,
                       params=_default_params(components="adaptive_contrast"))
    b = enhance_lunapsr(sci, norm, noise_model=cfg,
                        params=_default_params(components="full"))
    # full applies the structure-weight (w_s) to contrast gain; it must differ
    assert not np.allclose(a.enhanced, b.enhanced, atol=1e-4)


def test_ablation_invalid_component_raises():
    with pytest.raises(ValueError):
        LunaPSRParams(components="diffusion")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_returns_all_methods():
    sci, norm, params = _scene()
    lo, hi = params["lower_value"], params["upper_value"]
    bench = benchmark_lunapsr(norm, sci, None, (lo, hi),
                              _default_params(), cfg)
    assert "lunapsr" in bench
    assert "clahe" in bench and "gamma" in bench and "retinex" in bench
    lun = bench["lunapsr"]
    for k in ("method", "contrast", "entropy", "gradient_magnitude", "edge_density",
              "noise_sigma", "model_snr_median", "noise_amplification_ratio",
              "structure_preservation", "clipping_fraction", "score"):
        assert k in lun


def test_benchmark_zero_dce_error_recorded():
    from core.enhancement.zero_dce import is_available

    sci, norm, params = _scene()
    lo, hi = params["lower_value"], params["upper_value"]
    bench = benchmark_lunapsr(norm, sci, None, (lo, hi),
                              _default_params(), cfg)
    if not is_available():
        assert "zero_dce" in bench
        assert "error" in bench["zero_dce"]


def test_benchmark_no_source_mutation():
    sci, norm, params = _scene()
    lo, hi = params["lower_value"], params["upper_value"]
    sci_b, norm_b = sci.copy(), norm.copy()
    benchmark_lunapsr(norm, sci, None, (lo, hi), _default_params(), cfg)
    assert np.array_equal(sci, sci_b) and np.array_equal(norm, norm_b)


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------


def test_param_validation():
    with pytest.raises(ValueError):
        LunaPSRParams(min_strength=0.9, max_strength=0.5)
    with pytest.raises(ValueError):
        LunaPSRParams(gamma_min=0.0)
    with pytest.raises(ValueError):
        LunaPSRParams(confidence_low=0.8, confidence_high=0.6)
    with pytest.raises(ValueError):
        LunaPSRParams(local_window=1)


# ---------------------------------------------------------------------------
# Real-product end-to-end (skipped when the product is absent)
# ---------------------------------------------------------------------------


def test_real_product_lunapsr_end_to_end(real_ohrc_product):
    if real_ohrc_product is None:
        pytest.skip("real OHRC product not present")
    sample = real_ohrc_product.decimate_average(factor=100)  # ~121 kpx display-level
    norm, params = normalize_scientific(sample, percentiles=(1, 99))
    lo, hi = params["lower_value"], params["upper_value"]
    res = enhance_lunapsr(sample, norm, noise_model=cfg,
                          params=_default_params(), bounds=(lo, hi))
    assert res.enhanced.shape == sample.shape
    v = np.isfinite(res.enhanced)
    assert (res.enhanced[v] >= 0.0).all() and (res.enhanced[v] <= 1.0).all()
    assert float(np.nanmean(res.confidence)) > 0.0
    # source untouched
    assert np.array_equal(sample, real_ohrc_product.decimate_average(factor=100))
    # deterministic
    res2 = enhance_lunapsr(sample, norm, noise_model=cfg,
                           params=_default_params(), bounds=(lo, hi))
    assert np.array_equal(res.enhanced, res2.enhanced)
    assert res.parameters["version"] == LunaPSRParams().version


def test_real_product_benchmark(real_ohrc_product):
    if real_ohrc_product is None:
        pytest.skip("real OHRC product not present")
    sample = real_ohrc_product.decimate_average(factor=100)
    norm, params = normalize_scientific(sample, percentiles=(1, 99))
    lo, hi = params["lower_value"], params["upper_value"]
    bench = benchmark_lunapsr(norm, sample, None, (lo, hi), _default_params(), cfg)
    assert bench["lunapsr"]["model_snr_median"] > 0.0
    assert bench["clahe"]["score"] is not None
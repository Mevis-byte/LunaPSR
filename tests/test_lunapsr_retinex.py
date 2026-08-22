"""Phase 5B -- tests for the LunaPSR Adaptive Retinex prototype.

Covers: source immutability, output bounds, determinism, alpha/confidence
bounds, low-confidence protection, high-confidence behavior, invalid pixel
handling, NaN/Inf prevention, parameter serialization, artifact diagnostics,
and the MANDATORY Phase 5A regression benchmark on the real OHRC ROI.
"""

import numpy as np
import pytest

from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
from core.enhancement.lunapsr_retinex import (
    LunaPSRRetinexConfig,
    enhance_lunapsr_retinex,
    required_halo,
    smoothstep,
)
from core.enhancement.retinex import single_scale_retinex
from core.preprocessing.normalization import normalize_scientific

cfg_nm = OHRCNoiseModelConfig.default()


# ---------------------------------------------------------------------------
# synthetic scenes
# ---------------------------------------------------------------------------


def _dark_noise_scene(size=120):
    """Phase 5A-like scene: ~3 DN level, variation consistent with the model."""
    rng = np.random.default_rng(3)
    dn = np.clip(3.0 + rng.normal(0, 0.47, (size, size)), 0, 255).astype(np.float32)
    norm = ((dn - 1.0) / 4.0).astype(np.float32)  # ROI-local style bounds (1,5)
    return dn, np.clip(norm, 0.0, 1.0)


def _structured_scene(size=120):
    """High-signal scene with strong medium-scale structure."""
    rng = np.random.default_rng(7)
    dn = np.full((size, size), 80.0)
    dn[20:60, 20:60] = 150.0
    dn[70:100, 30:90] = 30.0
    dn += rng.normal(0, 0.8, dn.shape)
    dn = dn.astype(np.float32)
    lo, hi = float(dn.min()), float(dn.max())
    return dn, ((dn - lo) / (hi - lo)).astype(np.float32)


def _run(dn, norm, config=None, mask=None):
    return enhance_lunapsr_retinex(
        dn, norm, valid_mask=mask, noise_model=cfg_nm, config=config)


# ---------------------------------------------------------------------------
# basic contract
# ---------------------------------------------------------------------------


def test_source_immutability():
    dn, norm = _dark_noise_scene()
    dn_before, norm_before = dn.copy(), norm.copy()
    _run(dn, norm)
    assert np.array_equal(dn, dn_before)
    assert np.array_equal(norm, norm_before)


def test_output_bounds_and_nan_preserved():
    dn, norm = _dark_noise_scene()
    res = _run(dn, norm)
    v = np.isfinite(res.enhanced)
    assert v.all()
    assert (res.enhanced[v] >= 0.0).all() and (res.enhanced[v] <= 1.0).all()
    for map_arr in (res.confidence, res.alpha_map, res.retinex_response,
                    res.illumination, res.noise_sigma_dn, res.structure_dn):
        assert map_arr.shape == dn.shape
        assert np.all(np.isfinite(map_arr[v]))


def test_deterministic():
    dn, norm = _dark_noise_scene()
    a = _run(dn, norm)
    b = _run(dn, norm)
    assert np.array_equal(a.enhanced, b.enhanced)
    assert np.array_equal(a.alpha_map, b.alpha_map)


def test_invalid_pixel_handling():
    dn, norm = _dark_noise_scene()
    mask = np.ones(dn.shape, dtype=bool)
    mask[:10, :] = False
    res = _run(dn, norm, mask=mask)
    assert np.isnan(res.enhanced[:10, :]).all()
    assert np.isfinite(res.enhanced[10:, :]).all()


def test_nan_inf_prevention():
    dn, norm = _dark_noise_scene()
    dn_bad = dn.copy()
    dn_bad[0, 0] = np.nan
    dn_bad[5, 5] = np.inf
    norm_bad = norm.copy()
    norm_bad[9, 9] = np.nan
    res = _run(dn_bad, norm_bad)
    v = np.isfinite(res.enhanced)
    assert not v[0, 0] and not v[5, 5] and not v[9, 9]
    finite = res.enhanced[np.isfinite(res.enhanced)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_shape_mismatch_rejected():
    dn, norm = _dark_noise_scene(40)
    with pytest.raises(ValueError):
        enhance_lunapsr_retinex(dn, norm[:, :20], noise_model=cfg_nm)


def test_all_invalid_rejected():
    dn, norm = _dark_noise_scene(40)
    mask = np.zeros(dn.shape, dtype=bool)
    with pytest.raises(ValueError):
        _run(dn, norm, mask=mask)


# ---------------------------------------------------------------------------
# confidence / alpha behaviour
# ---------------------------------------------------------------------------


def test_confidence_and_alpha_bounds():
    for scene in (_dark_noise_scene(), _structured_scene()):
        res = _run(*scene)
        c = res.confidence[np.isfinite(res.confidence)]
        a = res.alpha_map[np.isfinite(res.alpha_map)]
        assert (c >= 0.0).all() and (c <= 1.0).all()
        assert (a >= 0.0).all() and (a <= 1.0).all()


def test_low_confidence_protection_conservative_alpha():
    """Noise-dominated dark region must stay near alpha_min (Phase 5A rule)."""
    _, norm = _dark_noise_scene()
    cfg = LunaPSRRetinexConfig()
    res = _run(_dark_noise_scene()[0], norm, config=cfg)
    mean_alpha = float(np.nanmean(res.alpha_map))
    # conservative: closer to alpha_min than to the midpoint of the range
    assert mean_alpha < cfg.alpha_min + 0.35 * (cfg.alpha_max - cfg.alpha_min)


def test_high_confidence_behavior_strong_alpha():
    """Structured, high-signal regions justify strong enhancement."""
    dn, norm = _structured_scene()
    cfg = LunaPSRRetinexConfig()
    res = _run(dn, norm, config=cfg)
    assert float(np.nanmean(res.alpha_map)) > 0.6


def test_protection_gate_raises_conservatism():
    """Increasing low_signal_protection must not increase alpha in noise."""
    dn, norm = _dark_noise_scene()
    loose = _run(dn, norm, config=LunaPSRRetinexConfig(low_signal_protection=0.0))
    tight = _run(dn, norm, config=LunaPSRRetinexConfig(low_signal_protection=1.0))
    assert (float(np.nanmean(tight.alpha_map))
            < float(np.nanmean(loose.alpha_map)))


def test_lunapsr_beats_ssr_on_new_edges_dark_scene():
    """On a noise-dominated scene LunaPSR must create fewer new edges than SSR."""
    dn, norm = _dark_noise_scene(160)
    from core.analysis.ssr_investigation import feature_consistency

    ssr_out = single_scale_retinex(norm, sigma=30.0)
    lp = _run(dn, norm)
    f_ssr = feature_consistency(norm, ssr_out)
    f_lp = feature_consistency(norm, lp.enhanced)
    assert (f_lp["new_edge_fraction_of_ssr_edges"]
            < f_ssr["new_edge_fraction_of_ssr_edges"])
    assert (f_lp["isolated_new_components"]
            < f_ssr["isolated_new_components"])


def test_smoothstep_bounded_continuous():
    t = np.linspace(-0.5, 1.5, 101, dtype=np.float32)  # step 0.02
    s = smoothstep(t)
    assert (s >= 0.0).all() and (s <= 1.0).all()
    assert s[0] == 0.0 and s[-1] == 1.0
    # zero slope at the ends: values just inside [0,1] stay tiny/near-1
    assert smoothstep(np.float32(0.02)) < 2e-3
    assert smoothstep(np.float32(0.98)) > 1.0 - 2e-3


def test_required_halo_covers_sigma():
    assert required_halo(LunaPSRRetinexConfig(sigma=30.0)) >= 4 * 30
    assert required_halo(LunaPSRRetinexConfig(sigma=10.0)) >= 4 * 10


# ---------------------------------------------------------------------------
# configuration serialization / validation
# ---------------------------------------------------------------------------


def test_parameter_serialization_roundtrip():
    cfg = LunaPSRRetinexConfig()
    d = cfg.to_dict()
    assert d["algorithm"] == "LunaPSR Adaptive Retinex"
    assert d["version"] == "0.1"
    for key in ("sigma", "epsilon", "alpha_min", "alpha_max",
                "confidence_transition", "detail_strength",
                "low_signal_protection"):
        assert key in d
    assert d["alpha_max"] == cfg.alpha_max


@pytest.mark.parametrize("field,value", [
    ("sigma", 0.0), ("sigma", 500.0), ("epsilon", 0.0),
    ("alpha_min", 1.5), ("alpha_max", -0.1), ("confidence_transition", 0.0),
    ("detail_strength", 3.0), ("low_signal_protection", 1.5),
])
def test_parameter_validation(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        LunaPSRRetinexConfig(**kwargs)


def test_alpha_min_above_alpha_max_rejected():
    with pytest.raises(ValueError):
        LunaPSRRetinexConfig(alpha_min=0.9, alpha_max=0.2)


# ---------------------------------------------------------------------------
# diagnostics / artifacts
# ---------------------------------------------------------------------------


def test_artifact_diagnostics_present():
    dn, norm = _structured_scene()
    res = _run(dn, norm)
    for key in ("clipping_fraction_after", "new_edge_fraction",
                "isolated_new_components", "halo_like_profile",
                "noise_amplification_ratio", "hf_energy_ratio"):
        assert key in res.artifacts
    assert isinstance(res.warnings, list)
    d = res.diagnostics
    for key in ("metrics", "feature_consistency", "hf_energy_ratio",
                "confidence_mean", "alpha_mean"):
        assert key in d
    # every warning uses the honest wording
    assert all(w.startswith("Potential enhancement artifact detected")
               for w in res.warnings)


def test_result_to_dict_serializable():
    dn, norm = _dark_noise_scene(60)
    res = _run(dn, norm)
    payload = res.to_dict()
    import json

    json.dumps(payload, default=str)  # must not raise


# ---------------------------------------------------------------------------
# MANDATORY Phase 5A regression benchmark (real product, exact ROI)
# ---------------------------------------------------------------------------

ROI_ROW0, ROI_COL0, ROI_SIZE = 61750, 6000, 200
P5A_SSR_NEW_EDGE = 0.314
P5A_SSR_ISOLATED = 391
P5A_SSR_EPI = 0.827
P5A_SSR_HF_INCREASE = 0.18


def test_phase5a_regression_real_roi(real_ohrc_product):
    if real_ohrc_product is None:
        pytest.skip("real OHRC product not present")
    sci = real_ohrc_product.image.scientific_array()
    roi = np.asarray(sci[ROI_ROW0:ROI_ROW0 + ROI_SIZE,
                         ROI_COL0:ROI_COL0 + ROI_SIZE], dtype=np.float32)
    before = roi.copy()
    norm, params = normalize_scientific(roi, percentiles=(1, 99))

    ssr_out = single_scale_retinex(norm, sigma=30.0)
    res = enhance_lunapsr_retinex(roi, norm, noise_model=cfg_nm)

    # source untouched
    assert np.array_equal(roi, before)

    from core.analysis.ssr_investigation import feature_consistency, hf_energy

    f_ssr = feature_consistency(norm, ssr_out)
    f_lp = feature_consistency(norm, res.enhanced)

    # SSR reproduces the Phase 5A references (loose sanity bounds)
    assert f_ssr["new_edge_fraction_of_ssr_edges"] == pytest.approx(
        P5A_SSR_NEW_EDGE, abs=0.02)
    assert f_ssr["isolated_new_components"] == pytest.approx(
        P5A_SSR_ISOLATED, rel=0.05)
    assert f_ssr["edge_preservation_index"] == pytest.approx(P5A_SSR_EPI, abs=0.02)

    # REGRESSION: LunaPSR must not worsen the Phase 5A observations
    assert (f_lp["new_edge_fraction_of_ssr_edges"] < P5A_SSR_NEW_EDGE)
    assert (f_lp["isolated_new_components"] < P5A_SSR_ISOLATED)
    assert f_lp["edge_preservation_index"] > P5A_SSR_EPI

    hf_orig = hf_energy(norm)
    hf_lp = hf_energy(res.enhanced)
    assert hf_lp < hf_orig * (1.0 + P5A_SSR_HF_INCREASE)

    # confidence/alpha stay bounded on real data
    a = res.alpha_map[np.isfinite(res.alpha_map)]
    assert (a >= 0).all() and (a <= 1).all()
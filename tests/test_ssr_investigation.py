"""Phase 5A -- tests for the SSR investigation analysis functions."""

import numpy as np
import pytest

from core.analysis.ohrc_noise_model import OHRCNoiseModelConfig
from core.analysis.ssr_investigation import (
    ABOVE_MODEL_FACTOR,
    dn_statistics,
    feature_consistency,
    gradient_energy,
    hf_energy,
    local_variance_map,
    model_noise_analysis,
    ssr_metrics,
    structure_maps,
    window_analysis_table,
)

cfg = OHRCNoiseModelConfig.default()


# ---------------------------------------------------------------------------
# dn_statistics
# ---------------------------------------------------------------------------


def test_dn_statistics_basic():
    dn = np.array([[0, 2, 4], [2, 255, 2]], dtype=np.float32)
    s = dn_statistics(dn)
    assert s["n_pixels"] == 6
    assert s["min"] == 0.0 and s["max"] == 255.0
    assert s["mean"] == pytest.approx((0 + 2 + 4 + 2 + 255 + 2) / 6)
    assert s["median"] == 2.0
    assert s["zero_fraction"] == pytest.approx(1 / 6)
    assert s["saturation_fraction"] == pytest.approx(1 / 6)
    assert set(s["percentiles"]) == {1, 5, 25, 50, 75, 95, 99}


def test_dn_statistics_does_not_modify_input():
    dn = np.full((8, 8), 3.0)
    before = dn.copy()
    dn_statistics(dn)
    assert np.array_equal(dn, before)


def test_dn_statistics_rejects_empty():
    with pytest.raises(ValueError):
        dn_statistics(np.array([], dtype=np.float32))


# ---------------------------------------------------------------------------
# model_noise_analysis (Phase 4B model on ORIGINAL DN)
# ---------------------------------------------------------------------------


def test_model_noise_analysis_fields_and_consistency():
    dn = np.full((16, 16), 3.0)
    r = model_noise_analysis(dn, cfg)
    for key in ("mean_dn", "signal_e", "shot_sigma_dn", "read_sigma_dn",
                "quant_sigma_dn", "total_sigma_dn", "model_snr"):
        assert key in r
    var = r["shot_sigma_dn"] ** 2 + r["read_sigma_dn"] ** 2 + r["quant_sigma_dn"] ** 2
    assert r["total_sigma_dn"] == pytest.approx(np.sqrt(var), rel=1e-6)
    # model_snr is rounded to 4 decimals in the report dict
    assert r["model_snr"] == pytest.approx(r["mean_dn"] / r["total_sigma_dn"], abs=1e-3)
    assert r["signal_e"] == pytest.approx(r["mean_dn"] * cfg.gain_e_per_dn)


def test_model_noise_analysis_monotone_in_dn():
    lo = model_noise_analysis(np.full((8, 8), 3.0), cfg)
    hi = model_noise_analysis(np.full((8, 8), 120.0), cfg)
    assert hi["total_sigma_dn"] > lo["total_sigma_dn"]
    assert hi["model_snr"] > lo["model_snr"]
    assert hi["shot_sigma_dn"] > lo["shot_sigma_dn"]


# ---------------------------------------------------------------------------
# window_analysis_table
# ---------------------------------------------------------------------------


def test_window_table_flat_scene_is_noise_consistent():
    rng = np.random.default_rng(3)
    model_sigma = float(np.sqrt(3.0 / cfg.gain_e_per_dn
                                + (cfg.read_noise_e / cfg.gain_e_per_dn) ** 2
                                + cfg.quantization_variance_dn2))
    dn = np.clip(3.0 + rng.normal(0, model_sigma, (80, 80)), 0, 255).astype(np.float32)
    rows = window_analysis_table(dn, cfg, window=20)
    assert len(rows) == 16
    ratios = np.array([r["ratio"] for r in rows])
    assert np.median(ratios) == pytest.approx(1.0, abs=0.35)
    assert all(r["category"] == "consistent_with_modeled_noise" for r in rows)


def test_window_table_structured_windows_flagged_cautiously():
    rng = np.random.default_rng(5)
    dn = np.full((80, 80), 30.0) + rng.normal(0, 0.7, (80, 80))
    dn[:, ::2] += 20.0  # vertical stripes: structure INSIDE every window
    dn = dn.astype(np.float32)
    rows = window_analysis_table(dn, cfg, window=20)
    # every window now contains both stripe levels -> variation far above model
    assert all(r["category"] == "spatial_variation_above_the_current_noise_model"
               for r in rows)
    assert all(r["ratio"] >= ABOVE_MODEL_FACTOR for r in rows)
    # cautious wording: no geology claims anywhere in the labels
    assert all("geolog" not in r["category"] for r in rows)


def test_window_table_columns():
    dn = np.full((40, 40), 3.0)
    rows = window_analysis_table(dn, cfg, window=20)
    assert {"row0", "col0", "mean_dn", "measured_sigma", "model_sigma",
            "ratio", "model_based_snr", "gradient_energy", "edge_density",
            "local_variance", "category"} <= set(rows[0].keys())


# ---------------------------------------------------------------------------
# maps and energies
# ---------------------------------------------------------------------------


def test_structure_maps_keys_shapes_finite():
    rng = np.random.default_rng(7)
    dn = np.clip(3.0 + rng.normal(0, 0.5, (48, 48)), 0, 255).astype(np.float32)
    maps = structure_maps(dn, cfg, window=5)
    assert {"gradient_magnitude", "local_variance", "local_mean",
            "model_noise_sigma_dn", "model_confidence"} <= set(maps.keys())
    for m in maps.values():
        assert m.shape == dn.shape
        assert np.all(np.isfinite(m))
    assert maps["model_confidence"].min() >= 0.0
    assert maps["model_confidence"].max() <= 1.0
    assert np.all(maps["local_variance"] >= 0.0)


def test_energies_zero_for_constant_array():
    const = np.full((24, 24), 0.5, dtype=np.float32)
    assert gradient_energy(const) == pytest.approx(0.0, abs=1e-12)
    assert hf_energy(const) == pytest.approx(0.0, abs=1e-12)
    assert float(local_variance_map(const).max()) == pytest.approx(0.0, abs=1e-9)


def test_hf_energy_positive_for_texture():
    rng = np.random.default_rng(9)
    a = rng.normal(0.5, 0.05, (48, 48)).astype(np.float32)
    assert hf_energy(a) > 0.0
    assert gradient_energy(a) > 0.0


# ---------------------------------------------------------------------------
# ssr_metrics
# ---------------------------------------------------------------------------


def test_ssr_metrics_keys_and_ranges():
    rng = np.random.default_rng(11)
    o = np.clip(rng.normal(0.5, 0.1, (64, 64)), 0, 1).astype(np.float32)
    e = np.clip(o * 1.2, 0, 1).astype(np.float32)
    m = ssr_metrics(o, e)
    for side in ("original", "ssr"):
        assert {"gradient_energy", "edge_density", "local_variance",
                "hf_energy", "clipping_fraction"} <= set(m[side].keys())
        for v in m[side].values():
            assert np.isfinite(v)
        assert 0.0 <= m[side]["clipping_fraction"] <= 1.0
    assert m["noise_amplification_ratio"] > 0.0


def test_ssr_metrics_keeps_2d_for_noise_estimate():
    # regression: the noise estimator must run on the 2-D array, not a
    # flattened copy (a flattened array corrupts the spatial kernel)
    rng = np.random.default_rng(13)
    o = np.clip(rng.normal(0.5, 0.02, (60, 60)), 0, 1).astype(np.float32)
    e = np.clip(rng.normal(0.5, 0.06, (60, 60)), 0, 1).astype(np.float32)
    m = ssr_metrics(o, e)
    assert m["noise_amplification_ratio"] != 1.0
    assert np.isfinite(m["noise_amplification_ratio"])


# ---------------------------------------------------------------------------
# feature_consistency
# ---------------------------------------------------------------------------


def test_feature_consistency_pure_amplification():
    y, x = np.mgrid[0:64, 0:64]
    o = np.where(x < 32, 0.3, 0.7).astype(np.float32)
    e = np.clip(o * 1.5 - 0.05, 0, 1).astype(np.float32)
    f = feature_consistency(o, e)
    assert f["strengthened_fraction_of_original_edges"] == pytest.approx(1.0)
    assert f["new_edge_fraction_of_ssr_edges"] == pytest.approx(0.0, abs=1e-6)
    assert f["isolated_new_components"] == 0
    assert f["orientation_agreement_doubled_angle"] > 0.99
    assert f["halo_like_profile"] is False


def test_feature_consistency_noise_only_creates_new_edges():
    rng = np.random.default_rng(17)
    o = np.clip(0.5 + rng.normal(0, 0.004, (96, 96)), 0, 1).astype(np.float32)
    e = np.clip(0.5 + rng.normal(0, 0.05, (96, 96)), 0, 1).astype(np.float32)
    f = feature_consistency(o, e)
    assert f["new_edge_fraction_of_ssr_edges"] > 0.5
    assert f["orientation_agreement_doubled_angle"] < 0.5
    assert f["isolated_new_components"] > 0


def test_feature_consistency_reports_distance_profile():
    rng = np.random.default_rng(19)
    o = np.clip(rng.normal(0.5, 0.01, (64, 64)), 0, 1).astype(np.float32)
    e = np.clip(rng.normal(0.5, 0.03, (64, 64)), 0, 1).astype(np.float32)
    f = feature_consistency(o, e)
    prof = f["abs_difference_by_distance_from_original_edges"]
    assert {"1", "2", "3", "6+"} <= set(prof.keys())
    assert isinstance(f["change_concentrated_at_original_edges"], bool)
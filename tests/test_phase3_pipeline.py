"""Phase 3 pipeline tests: normalization, baselines, Zero-DCE isolation,
noise diagnostics, no-reference metrics, experiment recording, and the
critical guarantee that the original scientific array is never modified.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.experiment import ExperimentRecord, ExperimentStore
from core.analysis.metrics import quality_metrics, quality_report
from core.analysis.noise import analyze_noise
from core.enhancement.baseline import (
    apply_baseline,
    baseline_metadata,
)
from core.enhancement.zero_dce import (
    ZeroDCEUnavailableError,
    is_available,
    zero_dce_enhance,
)
from core.preprocessing.mask import build_valid_mask
from core.preprocessing.normalization import normalize_scientific


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_scientific_returns_params_and_copy(gradient_image):
    before = gradient_image.copy()
    norm, params = normalize_scientific(gradient_image, percentiles=(1, 99))
    assert norm is not gradient_image
    assert norm.dtype == np.float32
    assert norm.shape == gradient_image.shape
    assert np.all(np.isfinite(norm))
    assert norm.min() >= 0.0 and norm.max() <= 1.0
    # reproducibility parameters
    assert params["method"] == "percentile"
    assert params["lower_percentile"] == 1.0
    assert params["upper_percentile"] == 99.0
    assert "lower_value" in params and "upper_value" in params
    assert params["bounds_source"] == "array"
    # never mutates
    assert np.array_equal(gradient_image, before)


def test_normalize_scientific_uses_precomputed_bounds():
    data = np.arange(0, 1000, dtype=np.float32).reshape(10, 100)
    base, _ = normalize_scientific(data, percentiles=(1, 99))
    bound, params = normalize_scientific(data, precomputed_bounds=(10.0, 900.0), percentiles=(1, 99))
    assert params["bounds_source"] == "precomputed"
    # precomputed range is wider: more low pixels clip to 0 than with the
    # percentile-derived bounds of the same data.
    assert int((bound == 0.0).sum()) == 11
    assert int((base == 0.0).sum()) == 10
    assert bound.max() == 1.0


def test_normalize_scientific_masked_invalid_become_nan():
    data = np.ones((8, 8), dtype=np.float32) * 100.0
    data[0, 0] = 0.0
    mask, _ = build_valid_mask(data, special_constants={"missing_constant": 0.0})
    norm, _ = normalize_scientific(data, mask=mask, percentiles=(1, 99))
    assert np.isnan(norm[0, 0])
    assert np.isfinite(norm[1, 1])
    assert data[0, 0] == 0.0  # untouched


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,params", [
    ("gamma", {"gamma": 0.75}),
    ("histogram_equalization", {}),
    ("clahe", {"clip_limit": 2.0, "tile_size": 8}),
    ("unsharp_mask", {"strength": 0.5, "sigma": 1.0}),
    ("denoising", {"strength": 3.0}),
    ("retinex", {"sigma": 30.0}),
])
def test_classical_baselines_finite_and_bounded(gradient_image, method, params):
    input_copy = gradient_image.copy()
    out = apply_baseline(gradient_image, method, **params)
    assert out.dtype == np.float32
    assert out.shape == gradient_image.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    # source never mutated
    assert np.array_equal(gradient_image, input_copy)


def test_none_baseline_returns_copy():
    a = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    out = apply_baseline(a, "none")
    assert out is not a
    assert np.array_equal(out, a)


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        apply_baseline(np.zeros((4, 4), dtype=np.float32), "not_a_method")


def test_baseline_metadata_records_reproducibly():
    meta = baseline_metadata("clahe", {"clip_limit": 2.0, "tile_size": 8},
                             normalization={"lower_percentile": 1.0})
    assert meta["method"] == "clahe"
    assert meta["parameters"] == {"clip_limit": 2.0, "tile_size": 8}
    assert meta["normalization"]["lower_percentile"] == 1.0


def test_baseline_on_black_image_never_crashes(black_image):
    for method in ("gamma", "histogram_equalization", "clahe", "unsharp_mask",
                   "denoising", "retinex"):
        out = apply_baseline(black_image, method, **({"sigma": 8.0} if method == "retinex" else {}))
        assert out.shape == black_image.shape


# ---------------------------------------------------------------------------
# Zero-DCE isolation
# ---------------------------------------------------------------------------


def test_zero_dce_unavailable_degrades_gracefully():
    if is_available():
        pytest.skip("deep-learning backend present in this environment")
    with pytest.raises(ZeroDCEUnavailableError):
        zero_dce_enhance(np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10))
    try:
        zero_dce_enhance(np.zeros((8, 8), dtype=np.float32))
    except ZeroDCEUnavailableError as exc:
        assert "dependency not installed" in str(exc)


def test_zero_dce_module_imports_without_backend():
    # Importing the module must never fail just because torch is absent.
    import core.enhancement.zero_dce as zd

    assert isinstance(zd.is_available(), bool)


# ---------------------------------------------------------------------------
# Noise diagnostics
# ---------------------------------------------------------------------------


def test_noise_diagnostics_report_snr_not_estimated(noisy_image):
    d = analyze_noise(noisy_image, factor=2)
    assert np.isfinite(d.mean)
    assert d.snr is None
    assert d.snr_status == "Not yet estimated"
    assert d.std >= 0.0
    assert d.coefficient_of_variation is not None
    rep = d.to_dict()
    assert rep["snr"] is None
    assert rep["snr_status"] == "Not yet estimated"


def test_noise_diagnostics_masked_pixels_excluded():
    a = np.ones((64, 64), dtype=np.float32)
    a[0, 0] = np.nan
    d = analyze_noise(a, factor=2)
    assert d.masked_fraction == pytest.approx(1 / 4096)
    assert d.n_pixels == 4095


# ---------------------------------------------------------------------------
# No-reference metrics
# ---------------------------------------------------------------------------


def test_quality_metrics_no_reference(gradient_image):
    m = quality_metrics(gradient_image)
    for key in ("entropy", "contrast", "gradient_magnitude", "edge_density",
                "local_variance", "noise_sigma", "histogram"):
        assert key in m
    assert m["n_pixels"] == gradient_image.size
    report = quality_report(gradient_image)
    assert report["noreference"] is True
    assert report["ground_truth_required"] is False
    assert "PSNR" not in report  # never pretend PSNR/SSIM vs original is quality


def test_quality_metrics_require_valid_pixels():
    a = np.full((8, 8), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        quality_metrics(a)
    with pytest.raises(ValueError):
        quality_metrics(np.zeros((0, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Experiment records
# ---------------------------------------------------------------------------


def test_experiment_record_written_to_derived_outputs(tmp_path, gradient_image):
    store = ExperimentStore(root=tmp_path / "outputs")
    rec = ExperimentRecord(
        product="ch2_test_product",
        method="clahe",
        parameters={"clip_limit": 2.0, "tile_size": 8},
        normalization={
            "lower_percentile": 1.0, "upper_percentile": 99.0,
            "lower_value": 3.0, "upper_value": 245.0,
        },
        metrics={"enhanced": {"entropy": 7.2}, "original": {"entropy": 5.1}},
    )
    path = store.save_record(rec)
    assert path.exists()
    assert "ch2_test_product" in path.name
    import json

    loaded = json.loads(path.read_text())
    assert loaded["method"] == "clahe"
    assert loaded["parameters"] == {"clip_limit": 2.0, "tile_size": 8}
    assert loaded["normalization"]["lower_percentile"] == 1.0
    assert loaded["metrics"]["enhanced"]["entropy"] == 7.2

    preview = store.save_preview("test_view", (gradient_image * 255).astype(np.uint8))
    assert preview.exists()
    metrics = store.save_metrics("test_m", {"entropy": 7.2})
    assert metrics.exists()
    assert (tmp_path / "outputs" / "experiments").is_dir()
    assert (tmp_path / "outputs" / "previews").is_dir()
    assert (tmp_path / "outputs" / "metrics").is_dir()


def test_experiment_record_never_touches_product(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    original = (bundle / "data.img")
    original.write_bytes(b"\x00\x01\x02\x03")
    store = ExperimentStore(root=tmp_path / "outputs")
    rec = ExperimentRecord(product=bundle.name, method="gamma", parameters={"gamma": 0.75})
    store.save_record(rec)
    assert original.read_bytes() == b"\x00\x01\x02\x03"


# ---------------------------------------------------------------------------
# Real Chandrayaan-2 OHRC product end-to-end (Phase 3 requirement 21)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_hist(real_ohrc_product):
    """Exact histogram of the real 1.2 Gpx scientific array (computed once)."""
    from core.ohrc.statistics import exact_histogram

    sci = real_ohrc_product.image.scientific_array()
    counts, edges, _, _ = exact_histogram(sci)
    return counts, edges


@pytest.mark.parametrize("method", [
    "gamma", "histogram_equalization", "clahe", "unsharp_mask", "denoising", "retinex",
])
def test_real_product_baseline_never_mutates_source(real_ohrc_product, real_hist, method):
    from app.pages.ohrc_lab import apply_enhancement, build_analysis

    sci = real_ohrc_product.image.scientific_array()
    sample_before = np.asarray(sci[::1000, ::1000]).copy()

    entry = build_analysis(real_ohrc_product, factor=40, count_hist=real_hist)
    assert entry["base"].shape[1] == sci.shape[1] // 40
    assert entry["bounds"][0] is not None and entry["bounds"][1] is not None

    res = apply_enhancement(entry["normalized"], method,
                            {"gamma": 0.75, "clip_limit": 2.0, "tile_size": 8,
                             "strength": 0.5, "sigma": 1.0, "iterations": 4})
    enhanced = res["enhanced"]
    assert enhanced.shape == entry["normalized"].shape
    assert np.isfinite(enhanced).all()

    # THE guarantee: source array unchanged after the full enhancement effort.
    assert np.array_equal(np.asarray(sci[::1000, ::1000]), sample_before)
    # product files unchanged
    assert real_ohrc_product.layout.scientific_image.stat().st_size == \
        real_ohrc_product.image.dimensions[0] * real_ohrc_product.image.dimensions[1]


def test_real_product_mask_and_normalization(real_ohrc_product, real_hist):
    sci = real_ohrc_product.image.scientific_array()
    sample = np.asarray(sci[::500, ::500]).copy()
    from app.pages.ohrc_lab import build_analysis

    entry = build_analysis(real_ohrc_product, factor=100, count_hist=real_hist)
    norm = entry["normalized"]
    assert norm.dtype == np.float32
    assert np.isfinite(norm[~np.isnan(norm)]).all()
    assert norm.min() >= 0.0 - 1e-6 and norm.max() <= 1.0 + 1e-6
    assert entry["mask_details"]["declared_sentinels"] == {}
    assert np.array_equal(np.asarray(sci[::500, ::500]), sample)
    assert entry["scientific_stats"]["valid_pixels"] == real_ohrc_product.image.dimensions[0] * \
        real_ohrc_product.image.dimensions[1]
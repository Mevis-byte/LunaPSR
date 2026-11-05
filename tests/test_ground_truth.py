"""Phase 5C -- unit tests for the synthetic ground-truth primitives.

These tests pin the properties the benchmark relies on:

* analytic profiles have the documented peak/extent behaviour,
* injection is linear, deterministic and never modifies the input,
* placement is reproducible and non-overlapping,
* the generic detector responds to injected structure above the control
  distribution,
* threshold calibration delivers the requested false-positive rate,
* recovery readouts are scale invariant.

They do NOT test detection performance of any enhancement method (that is the
experiment's job).  No OHRC product is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.ground_truth import (
    InjectedFeature,
    auc,
    boulder_profile,
    calibrate_threshold,
    crater_profile,
    dog_response,
    false_positive_rate,
    feature_readout,
    feature_score,
    inject_features,
    place_features,
    random_location_scores,
    truth_signal,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_crater_profile_has_negative_bowl_and_positive_rim():
    prof = crater_profile((64, 64), 32.0, 32.0, 20.0)
    assert prof[32, 32] == pytest.approx(-1.0, abs=1e-5)      # bowl centre
    assert prof[32, 42] == pytest.approx(0.8, abs=0.05)       # rim at r=10 px
    assert np.max(np.abs(prof)) == pytest.approx(1.0, abs=1e-6)
    assert abs(float(prof[0, 0])) < 1e-3                      # far field decays


def test_boulder_profile_is_a_unit_peak_dome():
    prof = boulder_profile((64, 64), 32.0, 32.0, 20.0)
    assert prof[32, 32] == pytest.approx(1.0, abs=1e-6)
    assert np.all(prof >= 0.0)
    # 2.5 sigma = half the nominal diameter -> ~exp(-3.125)
    assert prof[32, 42] == pytest.approx(np.exp(-3.125), abs=0.01)
    assert float(prof[0, 0]) < 1e-3


def test_profile_rejects_bad_parameters():
    with pytest.raises(ValueError):
        InjectedFeature("meteorite", 5, 5, 10, 1.0)
    with pytest.raises(ValueError):
        InjectedFeature("crater", 5, 5, 0.0, 1.0)
    with pytest.raises(ValueError):
        InjectedFeature("crater", 5, 5, 10.0, float("nan"))


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def test_injection_is_linear_and_does_not_modify_input():
    rng = np.random.default_rng(7)
    base = rng.normal(3.0, 0.5, size=(64, 64)).astype(np.float32)
    before = base.copy()
    feats = [InjectedFeature("crater", 20.0, 22.0, 16.0, 2.0),
             InjectedFeature("boulder", 45.0, 40.0, 10.0, 1.5)]

    inj1, truth1 = inject_features(base, feats, floor=None)
    inj2, truth2 = inject_features(base, feats, floor=None)
    np.testing.assert_array_equal(base, before)          # read-only contract
    np.testing.assert_allclose(inj1, inj2)               # deterministic
    np.testing.assert_allclose(inj1 - base, truth1, atol=1e-6)
    np.testing.assert_allclose(truth1, truth2)

    doubled = [InjectedFeature(f.kind, f.row, f.col, f.diameter_px,
                               2.0 * f.amplitude_dn) for f in feats]
    inj3, truth3 = inject_features(base, doubled, floor=None)
    np.testing.assert_allclose(truth3, 2.0 * truth1, atol=1e-5)
    np.testing.assert_allclose(inj3 - base, 2.0 * truth1, atol=1e-5)


def test_injection_never_produces_non_physical_values():
    zeros = np.zeros((64, 64), dtype=np.float32)
    feats = [InjectedFeature("crater", 32.0, 32.0, 30.0, 3.0)]
    inj, truth = inject_features(zeros, feats, floor=0.0)
    assert np.isfinite(inj).all()
    assert inj.min() >= 0.0
    assert truth.min() < 0.0 < truth.max()


def test_truth_signal_sums_individual_profiles():
    feats = [InjectedFeature("crater", 20.0, 20.0, 12.0, 1.0),
             InjectedFeature("boulder", 45.0, 45.0, 8.0, 2.0)]
    total = truth_signal((64, 64), feats)
    expected = (1.0 * crater_profile((64, 64), 20.0, 20.0, 12.0)
                + 2.0 * boulder_profile((64, 64), 45.0, 45.0, 8.0))
    np.testing.assert_allclose(total, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_placement_is_deterministic_and_non_overlapping():
    kw = dict(shape=(128, 128), kind="crater", diameter_px=16.0,
              amplitude_dn=1.0, count=4)
    a = place_features(rng=np.random.default_rng(123), **kw)
    b = place_features(rng=np.random.default_rng(123), **kw)
    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]
    assert len(a) == 4
    for i, fi in enumerate(a):
        assert 0.0 <= fi.row < 128.0 and 0.0 <= fi.col < 128.0
        for fj in a[i + 1:]:
            sep = float(np.hypot(fi.row - fj.row, fi.col - fj.col))
            assert sep >= 2.0 * 1.25 * 8.0  # 2 * footprint radius


def test_placement_respects_existing_features():
    fixed = [InjectedFeature("crater", 64.0, 64.0, 24.0, 1.0)]
    placed = place_features((128, 128), "boulder", 12.0, 1.0, 3,
                            np.random.default_rng(3), existing=fixed)
    assert len(placed) == 3
    for f in placed:
        sep = float(np.hypot(f.row - 64.0, f.col - 64.0))
        assert sep >= 2.0 * (1.25 * 12.0 + 2.0)


def test_placement_refuses_features_larger_than_tile():
    with pytest.raises(ValueError):
        place_features((16, 16), "crater", 40.0, 1.0, 1,
                       np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detector_finds_an_injected_feature_on_noise():
    rng = np.random.default_rng(11)
    base = rng.normal(3.0, 0.5, size=(96, 96)).astype(np.float32)
    feat = InjectedFeature("crater", 48.0, 48.0, 24.0, 4.0)
    injected, _ = inject_features(base, [feat], floor=None)
    resp = dog_response(injected, 0.18 * feat.diameter_px)
    control = random_location_scores(
        dog_response(base, 0.18 * feat.diameter_px),
        radius=15.0, count=400, rng=np.random.default_rng(5),
    )
    threshold = calibrate_threshold(control, fp_rate=0.05)
    assert feature_score(resp, feat) > threshold


def test_dog_response_shape_and_finiteness():
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(40, 40))
    resp = dog_response(arr, 1.5)
    assert resp.shape == arr.shape
    assert np.isfinite(resp).all()
    # a band-pass removes the DC level
    assert abs(float(resp.mean())) < 1e-3


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def test_calibration_delivers_requested_false_positive_rate():
    rng = np.random.default_rng(42)
    cal = np.abs(rng.normal(size=20000))
    val = np.abs(rng.normal(size=20000))
    thr = calibrate_threshold(cal, fp_rate=0.05)
    assert false_positive_rate(cal, thr) == pytest.approx(0.05, abs=0.005)
    assert false_positive_rate(val, thr) == pytest.approx(0.05, abs=0.01)


def test_calibration_rejects_bad_rate():
    with pytest.raises(ValueError):
        calibrate_threshold(np.arange(10.0), fp_rate=0.0)


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def test_readout_correlation_is_scale_invariant():
    feat = InjectedFeature("crater", 32.0, 32.0, 20.0, 1.0)
    truth = truth_signal((64, 64), [feat])
    scaled = 7.0 * truth + 0.25
    r = feature_readout(scaled, feat, truth=truth)
    assert r["corr_truth"] == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(r["cnr_ptp"]) and r["cnr_ptp"] > 0


def test_readout_detects_boulder_positive_mean_shift():
    feat = InjectedFeature("boulder", 32.0, 32.0, 12.0, 1.0)
    truth = truth_signal((64, 64), [feat])
    r = feature_readout(truth, feat, truth=truth)
    assert r["mean_shift"] > 0.0
    assert r["corr_truth"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_auc_extremes():
    assert auc([2.0, 3.0, 4.0], [0.0, 1.0, 0.5]) == pytest.approx(1.0)
    assert auc([0.0, 1.0, 0.5], [2.0, 3.0, 4.0]) == pytest.approx(0.0)
    assert auc([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert np.isnan(auc([], [1.0]))


def test_wilson_interval_is_bounded_and_centred():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    assert lo < hi
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == pytest.approx(0.0, abs=1e-9) and 0.0 < hi0 < 0.4
    assert np.isnan(wilson_interval(0, 0)[0])

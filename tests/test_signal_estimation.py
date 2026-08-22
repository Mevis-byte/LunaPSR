"""Phase 5 tests: local signal estimation (LunaPSR).

Covers the three-scale separation (illumination / structure / residual), the
robust local signal estimate, NaN/mask handling, input validation, parameter
recording and source immutability.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.enhancement.signal_estimation import (
    SignalEstimationParams,
    estimate_local_signal,
)


def _synthetic() -> np.ndarray:
    rng = np.random.default_rng(3)
    base = np.full((128, 128), 40.0)
    base[40:60, 40:60] = 90.0          # medium-scale structure blob
    base[20:24, 80:96] = 160.0         # sharp bright stripe
    a = base + rng.normal(0.0, 1.0, base.shape)   # shot/read-like noise
    return np.clip(a, 0, 255).astype(np.float32)


def test_components_have_input_shape():
    a = _synthetic()
    c = estimate_local_signal(a)
    for name in ("signal", "illumination", "structure", "residual"):
        assert getattr(c, name).shape == a.shape
    assert c.valid.shape == a.shape
    assert c.valid.all()


def test_illumination_is_smooth():
    a = _synthetic()
    c = estimate_local_signal(a)
    # illumination should vary slowly: local spread much smaller than input
    assert float(np.std(c.illumination)) < float(np.std(a))


def test_structure_captures_blob():
    a = _synthetic()
    c = estimate_local_signal(a, SignalEstimationParams(local_window=7))
    blob = c.structure[45:55, 45:55]
    outside = np.concatenate([c.structure[:15, :15].ravel(), c.structure[100:115, 100:115].ravel()])
    assert float(np.mean(blob)) > 0.0
    assert float(np.mean(np.abs(blob))) > float(np.mean(np.abs(outside)))


def test_residual_is_noise_dominated_and_small():
    a = _synthetic()
    c = estimate_local_signal(a, SignalEstimationParams(local_window=5))
    assert float(np.std(c.residual)) < float(np.std(a))
    assert float(np.mean(c.residual)) == pytest.approx(0.0, abs=1.0)


def test_signal_matches_local_median_in_flat_region():
    a = _synthetic()
    p = SignalEstimationParams(local_window=11, illumination_sigma=11.0)
    c = estimate_local_signal(a, p)
    # in a flat region the local median approximates the constant level
    flat = a[90:110, 90:110].ravel()
    med = float(np.median(flat))
    sig = float(np.mean(c.signal[90:110, 90:110]))
    assert sig == pytest.approx(med, abs=2.0)


def test_use_median_false_uses_local_mean():
    a = _synthetic()
    c_med = estimate_local_signal(a, SignalEstimationParams(use_median=True, local_window=7))
    c_mean = estimate_local_signal(a, SignalEstimationParams(use_median=False, local_window=7))
    # in the flat (structure-free) region both local estimates approximate the
    # constant level; they differ only near sharp edges/blobs
    med = c_med.signal[90:110, 90:110]
    mean = c_mean.signal[90:110, 90:110]
    assert np.allclose(med, mean, atol=1.0)
    assert float(np.mean(np.abs(c_med.signal - c_mean.signal))) < 2.0


def test_nan_and_mask_handling():
    a = _synthetic()
    a[0:5, 0:5] = np.nan
    mask = np.ones(a.shape, dtype=bool)
    mask[60:70, 60:70] = False
    c = estimate_local_signal(a, mask=mask)
    assert np.isnan(c.signal[0:5, 0:5]).all()
    assert np.isnan(c.signal[60:70, 60:70]).all()
    assert not c.valid[0:5, 0:5].any()
    assert np.isfinite(c.signal[10:20, 10:20]).all()
    assert np.isnan(c.residual[60:70, 60:70]).all()


def test_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        estimate_local_signal(_synthetic(), mask=np.ones((10, 10), dtype=bool))


def test_all_invalid_raises():
    with pytest.raises(ValueError):
        estimate_local_signal(np.full((8, 8), np.nan))


def test_non_2d_raises():
    with pytest.raises(ValueError):
        estimate_local_signal(np.ones((4, 4, 4)))


def test_param_validation():
    with pytest.raises(ValueError):
        SignalEstimationParams(local_window=1)
    with pytest.raises(ValueError):
        SignalEstimationParams(illumination_sigma=0.0)


def test_source_immutability():
    a = _synthetic()
    before = a.copy()
    estimate_local_signal(a)
    assert np.array_equal(a, before)


def test_to_dict_records_params():
    a = _synthetic()
    c = estimate_local_signal(a, SignalEstimationParams(local_window=7))
    d = c.to_dict()
    assert d["local_window"] == 7
    assert d["use_median"] is True
    assert d["valid_fraction"] == pytest.approx(1.0)
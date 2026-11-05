"""Phase 5C -- synthetic ground-truth benchmark primitives.

WHY THIS MODULE EXISTS
----------------------
Phases 5A/5B can show that enhancement methods *add* structure in a dark PSR
region, and the forensics quantify how much.  They cannot show whether real
structure is *recovered*, because no ground truth exists in a single OHRC
frame.  This module supplies a controlled ground truth:

    real OHRC dark tile (real noise, real quantization)
        + analytically known crater / boulder profile in DN
        = input image with exact known signal

Every injected feature is therefore a truth-tagged signal whose amplitude,
size and position are known to the benchmark, while the noise, background
statistics and sensor quantisation remain those of the real product.

HONESTY BOUNDARIES (read before using any output)
-------------------------------------------------
* Injected profiles are **idealised analytic shapes**, not real lunar
  photometry.  They model the *information content* that a lander-relevant
  feature would produce, not the true radiance field of a crater.
* Injection is performed in floating-point DN on top of real quantised data.
  Amplitudes below 1 DN are a continuous-signal proxy: the delivered product
  quantises to integer counts, so sub-DN contrast could not be stored
  literally.  This is the standard synthetic-injection methodology; it is not
  a claim that such a signal exists in the product.
* The background tiles are noise-dominated regions only.  A transition from
  "noise floor" to "terrain with structure" is *not* measured here.
* Nothing in this module proves geological reality.  It measures detection
  limits under known contrast, which is exactly what a benchmark should do.

All functions are pure, deterministic (caller supplies RNGs) and never modify
their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

__all__ = [
    "KINDS",
    "InjectedFeature",
    "feature_profile",
    "truth_signal",
    "inject_features",
    "footprint_radius",
    "place_features",
    "dog_response",
    "detector_sigma",
    "max_abs_in_disc",
    "feature_score",
    "random_location_scores",
    "calibrate_threshold",
    "false_positive_rate",
    "feature_readout",
    "auc",
    "wilson_interval",
]

KINDS = ("crater", "boulder")


# ---------------------------------------------------------------------------
# Feature description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectedFeature:
    """One synthetic feature in pixel coordinates (read-only description).

    ``amplitude_dn`` is the peak signal amplitude of the profile in DN
    (positive part).  For a crater the bowl depth equals ``amplitude_dn`` and
    the rim rises to ``rim_ratio * amplitude_dn`` above the local background.
    """

    kind: str
    row: float
    col: float
    diameter_px: float
    amplitude_dn: float

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not np.isfinite(self.diameter_px) or self.diameter_px <= 0:
            raise ValueError(f"diameter_px must be positive, got {self.diameter_px}")
        if not np.isfinite(self.amplitude_dn):
            raise ValueError(f"amplitude_dn must be finite, got {self.amplitude_dn}")

    @property
    def radius_px(self) -> float:
        return 0.5 * float(self.diameter_px)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "row": float(self.row),
            "col": float(self.col),
            "diameter_px": float(self.diameter_px),
            "amplitude_dn": float(self.amplitude_dn),
        }


# ---------------------------------------------------------------------------
# Analytic profiles
# ---------------------------------------------------------------------------


def _distance_grid(shape: tuple[int, int], row: float, col: float) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    rr, cc = np.mgrid[0:h, 0:w]
    return np.hypot(rr - float(row), cc - float(col)).astype(np.float32)


def crater_profile(
    shape: tuple[int, int],
    row: float,
    col: float,
    diameter_px: float,
    rim_ratio: float = 0.8,
    rim_width_fraction: float = 0.22,
) -> np.ndarray:
    """Idealised crater: parabolic bowl plus a raised rim ring.

    The profile is normalised so ``max(|profile|) == 1``.  It is radially
    symmetric (no illumination direction), which is a deliberate
    simplification: the benchmark measures contrast detection, not photometric
    shape recovery.
    """
    radius = max(0.5 * float(diameter_px), 1.5)
    d = _distance_grid(shape, row, col)
    inner = np.clip(1.0 - (d / radius) ** 2, 0.0, 1.0)
    bowl = -inner.astype(np.float32)
    width = max(0.75, float(rim_width_fraction) * radius)
    rim = float(rim_ratio) * np.exp(-0.5 * ((d - radius) / width) ** 2)
    prof = (bowl + rim).astype(np.float32)
    peak = float(np.max(np.abs(prof)))
    if peak > 0.0:
        prof /= peak
    return prof


def boulder_profile(
    shape: tuple[int, int],
    row: float,
    col: float,
    diameter_px: float,
    shadow_ratio: float = 0.0,
) -> np.ndarray:
    """Idealised boulder: Gaussian dome (optionally with an offset shadow).

    ``diameter_px`` covers ~5 sigma of the dome; the visible footprint radius
    is about ``0.5 * diameter_px``.
    """
    sigma = max(float(diameter_px) / 5.0, 0.8)
    d = _distance_grid(shape, row, col)
    dome = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
    prof = dome
    if shadow_ratio > 0.0:
        shift = 0.9 * sigma
        d_shadow = _distance_grid(shape, row + shift, col + shift)
        shadow = np.exp(-0.5 * (d_shadow / (0.7 * sigma)) ** 2)
        prof = dome - float(shadow_ratio) * shadow.astype(np.float32)
    peak = float(np.max(np.abs(prof)))
    if peak > 0.0:
        prof /= peak
    return prof.astype(np.float32)


def feature_profile(feature: InjectedFeature, shape: tuple[int, int]) -> np.ndarray:
    """Unit-peak profile for ``feature`` on a ``shape`` grid."""
    if feature.kind == "crater":
        return crater_profile(shape, feature.row, feature.col,
                              feature.diameter_px)
    if feature.kind == "boulder":
        return boulder_profile(shape, feature.row, feature.col,
                               feature.diameter_px)
    raise ValueError(f"unknown feature kind {feature.kind!r}")


def truth_signal(
    shape: tuple[int, int],
    features: Sequence[InjectedFeature],
) -> np.ndarray:
    """Signed ground-truth signal in DN (sum of all injected profiles)."""
    truth = np.zeros(shape, dtype=np.float32)
    for f in features:
        truth += float(f.amplitude_dn) * feature_profile(f, shape)
    return truth


def inject_features(
    dn: np.ndarray,
    features: Sequence[InjectedFeature],
    floor: Optional[float] = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(injected_dn, truth_dn)`` without modifying ``dn``.

    ``floor`` clips the injected image at a physical floor (default 0 DN, the
    sensor's valid minimum).  Set ``floor=None`` to keep the raw sum.
    """
    arr = np.asarray(dn, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"inject_features expects a 2-D array, got {arr.shape}")
    truth = truth_signal(tuple(arr.shape), features)
    out = arr + truth
    if floor is not None:
        out = np.maximum(out, float(floor))
    return out.astype(np.float32), truth


# ---------------------------------------------------------------------------
# Deterministic placement
# ---------------------------------------------------------------------------


def footprint_radius(feature: InjectedFeature) -> float:
    """Radius of the visible feature footprint used by detector/readout."""
    if feature.kind == "crater":
        return 1.25 * feature.radius_px
    return 0.6 * float(feature.diameter_px)


def place_features(
    shape: tuple[int, int],
    kind: str,
    diameter_px: float,
    amplitude_dn: float,
    count: int,
    rng: np.random.Generator,
    existing: Sequence[InjectedFeature] = (),
    margin_px: float = 2.0,
    max_tries: int = 4000,
) -> list[InjectedFeature]:
    """Place ``count`` non-overlapping features with a deterministic RNG."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    h, w = int(shape[0]), int(shape[1])
    probe = InjectedFeature(kind, 0.0, 0.0, diameter_px, amplitude_dn)
    radius = footprint_radius(probe)
    lo_r, hi_r = radius + margin_px, h - radius - margin_px
    lo_c, hi_c = radius + margin_px, w - radius - margin_px
    if lo_r >= hi_r or lo_c >= hi_c:
        raise ValueError(
            f"{kind} of diameter {diameter_px} does not fit in {shape}"
        )
    min_sep = 2.0 * radius + margin_px
    placed = list(existing)
    out: list[InjectedFeature] = []
    tries = 0
    while len(out) < int(count) and tries < max_tries:
        tries += 1
        row = float(rng.uniform(lo_r, hi_r))
        col = float(rng.uniform(lo_c, hi_c))
        ok = True
        for other in placed:
            sep = np.hypot(row - other.row, col - other.col)
            if sep < min_sep:
                ok = False
                break
        if not ok:
            continue
        feature = InjectedFeature(kind, row, col, diameter_px, amplitude_dn)
        out.append(feature)
        placed.append(feature)
    return out


# ---------------------------------------------------------------------------
# Generic (shape-agnostic) detector
# ---------------------------------------------------------------------------


def dog_response(img: np.ndarray, sigma: float) -> np.ndarray:
    """Difference-of-Gaussians band-pass response.

    Deliberately generic: the detector is not matched to the injected profile,
    so a high detection rate means the structure is recoverable by an ordinary
    blob/ring detector, not that the template was correlated with itself.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"dog_response expects a 2-D array, got {a.shape}")
    s = float(max(sigma, 0.8))
    lo = gaussian_filter(a, s, mode="reflect")
    hi = gaussian_filter(a, 2.0 * s, mode="reflect")
    return (lo - hi).astype(np.float32)


def detector_sigma(feature: InjectedFeature) -> float:
    """Band-pass scale appropriate to a feature's angular size."""
    if feature.kind == "crater":
        return max(1.0, 0.18 * float(feature.diameter_px))
    return max(0.8, 0.20 * float(feature.diameter_px))


def _disc_mask(shape: tuple[int, int], row: float, col: float,
               radius: float) -> np.ndarray:
    return _distance_grid(shape, row, col) <= float(radius)


def max_abs_in_disc(
    response: np.ndarray,
    row: float,
    col: float,
    radius: float,
) -> float:
    """Largest ``|response|`` inside a disc (0.0 if the disc is empty)."""
    r = float(radius)
    h, w = response.shape
    r0 = max(0, int(np.floor(row - r)))
    r1 = min(h, int(np.ceil(row + r)) + 1)
    c0 = max(0, int(np.floor(col - r)))
    c1 = min(w, int(np.ceil(col + r)) + 1)
    if r1 <= r0 or c1 <= c0:
        return 0.0
    patch = np.abs(response[r0:r1, c0:c1])
    mask = _disc_mask((r1 - r0, c1 - c0), row - r0, col - c0, r)
    if not mask.any():
        return 0.0
    return float(patch[mask].max())


def feature_score(response: np.ndarray, feature: InjectedFeature) -> float:
    """Detector statistic for one feature (max |DoG| in its footprint)."""
    return max_abs_in_disc(response, feature.row, feature.col,
                           footprint_radius(feature))


def random_location_scores(
    response: np.ndarray,
    radius: float,
    count: int,
    rng: np.random.Generator,
    margin_px: float = 2.0,
) -> np.ndarray:
    """Detector statistics at ``count`` random feature-free locations.

    These samples form the control distribution from which the detection
    threshold is calibrated (fixed false-positive rate).
    """
    h, w = response.shape
    r = float(radius)
    lo_r, hi_r = r + margin_px, h - r - margin_px
    lo_c, hi_c = r + margin_px, w - r - margin_px
    if lo_r >= hi_r or lo_c >= hi_c:
        raise ValueError("search radius leaves no valid control locations")
    rows = rng.uniform(lo_r, hi_r, size=int(count))
    cols = rng.uniform(lo_c, hi_c, size=int(count))
    return np.array(
        [max_abs_in_disc(response, y, x, r) for y, x in zip(rows, cols)],
        dtype=np.float64,
    )


def calibrate_threshold(control_scores: np.ndarray, fp_rate: float = 0.05) -> float:
    """Score threshold whose control false-positive rate is ``fp_rate``."""
    s = np.asarray(control_scores, dtype=np.float64)
    if s.size == 0:
        raise ValueError("control_scores is empty")
    if not (0.0 < float(fp_rate) < 1.0):
        raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")
    return float(np.quantile(s, 1.0 - float(fp_rate)))


def false_positive_rate(scores: np.ndarray, threshold: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return float("nan")
    return float(np.mean(s > float(threshold)))


# ---------------------------------------------------------------------------
# Recovery readouts (scale invariant -> comparable across methods)
# ---------------------------------------------------------------------------


def feature_readout(
    output: np.ndarray,
    feature: InjectedFeature,
    truth: Optional[np.ndarray] = None,
) -> dict:
    """Local structure readout around one feature.

    Returns scale-invariant quantities only, because different enhancement
    methods apply different global tone curves:

    ``cnr_ptp``
        (max - min) of the output inside the footprint divided by the annulus
        standard deviation -- a local contrast-to-noise ratio.
    ``corr_truth``
        Pearson correlation between the background-subtracted output and the
        injected signal inside the footprint (NaN when ``truth`` is None or
        the truth is locally flat).
    ``mean_shift``
        footprint mean minus annulus mean (dominant for boulder-like features,
        near zero for balanced crater rim+bowl footprints).
    """
    out = np.asarray(output, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"feature_readout expects a 2-D array, got {out.shape}")
    r = footprint_radius(feature)
    d = _distance_grid(out.shape, feature.row, feature.col)
    disc = d <= r
    annulus = (d > r) & (d <= 1.8 * r)
    if not disc.any() or annulus.sum() < 4:
        return {"cnr_ptp": float("nan"), "corr_truth": float("nan"),
                "mean_shift": float("nan")}

    patch = out[disc]
    ring = out[annulus]
    std_ring = float(np.std(ring))
    ptp = float(patch.max() - patch.min())
    cnr = ptp / std_ring if std_ring > 1e-12 else float("nan")
    mean_shift = float(patch.mean() - ring.mean())

    corr = float("nan")
    if truth is not None:
        t = np.asarray(truth, dtype=np.float32)[disc]
        if t.size >= 8 and float(np.std(t)) > 1e-12 and ptp > 1e-12:
            o = patch - patch.mean()
            tc = t - t.mean()
            denom = float(np.linalg.norm(o) * np.linalg.norm(tc))
            if denom > 1e-12:
                corr = float(np.dot(o, tc) / denom)
    return {"cnr_ptp": cnr, "corr_truth": corr, "mean_shift": mean_shift}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties handled by average ranks."""
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    u = float(ranks[:pos.size].sum()) - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    n = int(trials)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(successes) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(centre - half), float(centre + half))

"""Unit tests for matte_core.despill's V6 speed/quality experiment fields
(band_only, est_scale) -- see the plan. Pure numpy/synthetic, no real
clip or GPU needed: these test the write-back SCOPE and that the function
still runs, not matting quality (that's tool/tests/test_regression_gates.py
on real clips).
"""
import numpy as np

from tool.matte_core import despill


def _synthetic_frame(h=40, w=60):
    """A soft-edged synthetic subject: opaque core, a genuinely
    semi-transparent ring around it, fully transparent background."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    a = np.zeros((h, w), dtype=np.float32)
    a[8:32, 12:48] = 1.0        # opaque core
    a[6:34, 10:50] = np.where(a[6:34, 10:50] == 0, 0.4, a[6:34, 10:50])  # semi-transparent ring
    return rgb, a


def test_band_only_leaves_fully_opaque_pixels_byte_identical():
    rgb, a = _synthetic_frame()
    baseline = despill(rgb, a, band_only=False)
    banded = despill(rgb, a, band_only=True)
    opaque = a >= 0.98
    assert opaque.any(), "fixture must contain some fully-opaque pixels to test the claim"
    np.testing.assert_array_equal(banded[opaque], rgb[opaque],
        "band_only=True must leave fully-opaque pixels untouched (no spill to remove there)")
    # The two variants may legitimately differ inside the semi-transparent
    # band (that's real ML-solve inputs, not a hardcoded value) -- just
    # confirm both actually ran without raising and returned the right shape.
    assert baseline.shape == banded.shape == rgb.shape


def test_band_only_skips_the_solve_entirely_when_the_band_is_empty():
    """A frame with alpha only ever 0 or 1 (no semi-transparent pixels at
    all) has nothing for despill to do under band_only -- must return
    early rather than running estimate_foreground_ml for no reason."""
    rgb, _ = _synthetic_frame()
    a = np.zeros(rgb.shape[:2], dtype=np.float32)
    a[8:32, 12:48] = 1.0  # hard-edged, no ring
    out = despill(rgb, a, band_only=True)
    np.testing.assert_array_equal(out, rgb)


def test_est_scale_runs_and_stays_close_to_full_resolution():
    # A smooth gradient, not random noise -- real footage (what est_scale is
    # actually measured against, in the plan's V6 harness) has spatial
    # coherence, so downsample-solve-upsample only makes sense to compare
    # against a fixture that does too; pure per-pixel noise has none, and
    # a solve over it isn't expected to survive resampling at all.
    h, w = 40, 60
    gy, gx = np.mgrid[0:h, 0:w]
    rgb = np.stack([gx * 255 // w, gy * 255 // h, np.full((h, w), 128)], axis=-1).astype(np.uint8)
    a = np.zeros((h, w), dtype=np.float32)
    a[8:32, 12:48] = 1.0
    a[6:34, 10:50] = np.where(a[6:34, 10:50] == 0, 0.4, a[6:34, 10:50])

    full = despill(rgb, a, est_scale=1.0)
    half = despill(rgb, a, est_scale=0.5)
    assert full.shape == half.shape == rgb.shape
    # Downsample-solve-upsample is an approximation, not identical output --
    # just confirm it's in the same ballpark on the pixels that changed.
    diff = np.abs(full[a > 0.02].astype(int) - half[a > 0.02].astype(int))
    assert diff.mean() < 40, f"est_scale=0.5 diverged too far from full resolution (mean abs diff {diff.mean():.1f})"


def test_est_scale_and_band_only_combine():
    rgb, a = _synthetic_frame()
    out = despill(rgb, a, band_only=True, est_scale=0.5)
    assert out.shape == rgb.shape
    opaque = a >= 0.98
    np.testing.assert_array_equal(out[opaque], rgb[opaque])

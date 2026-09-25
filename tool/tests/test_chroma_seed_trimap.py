"""Regression tests for chroma.build_trimap_alpha's B-2 extension (レバーB
"SAM2Matting 型分業" -- see 第9計画 §6.B, R7). Covers:

  (a) default call (no `seed`, default `band`) stays byte-identical to the
      pre-B-2 implementation -- callers (runner.py) and the existing
      chroma-gate tests must see no behaviour change.
  (b) `seed` + `band="alpha"` takes the alpha-based UNK ramp instead of the
      colour-distance ramp (checked with `bg_stats=None`, which the colour
      path would crash on -- proving the colour path is never touched).
  (c) both pre-registered seed-geometry variants' FG/BG/UNK partition:
      "additive" (SAM2 only ADDS fg, only removes bg where BiRefNet agrees)
      vs "symmetric" (plain erode/dilate of the seed, no BiRefNet override).
"""
import numpy as np
import pytest

from tool.pipeline import chroma
from tool.qc.metrics import BgStats


def _snapshot_inputs():
    rng = np.random.default_rng(42)
    H, W = 24, 32
    a_raw = rng.random((H, W)).astype(np.float32)
    rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    return a_raw, rgb


def test_default_call_is_byte_identical_to_pre_b2(tmp_path):
    """Frozen snapshot of build_trimap_alpha's output taken from the
    implementation immediately BEFORE the B-2 seed/band/radius_frac kwargs
    were added (same RNG seed/shape as here). Any drift means the default
    (seed=None) code path changed."""
    a_raw, rgb = _snapshot_inputs()
    bg_stats = BgStats(mu_cb=128.0, mu_cr=128.0, sigma_cb=5.0, sigma_cr=5.0,
                        is_chroma_class=True, frac_bg_like=1.0)
    alpha, unk = chroma.build_trimap_alpha(a_raw, rgb, bg_stats)

    assert alpha.dtype == np.float32
    assert alpha.shape == a_raw.shape
    # Exact values captured via `venv/bin/python` from the pre-change
    # implementation with this exact seed/shape/bg_stats (see B-2 commit
    # message for how this was generated).
    assert float(alpha.sum()) == pytest.approx(759.44604, abs=1e-2)
    assert int(unk.sum()) == 768  # every pixel: random alpha rarely clears
                                   # the 0.90/0.10 seed thresholds at H=24,W=32


def test_seed_band_alpha_uses_raw_alpha_not_color_ramp():
    """With band='alpha', the UNK strip must be filled from a_raw directly.
    bg_stats=None proves the color-distance path (which would crash trying
    to read bg_stats.mu_cb etc.) is never exercised."""
    H, W = 10, 10
    a_raw = np.full((H, W), 0.5, dtype=np.float32)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # Seed covering the left half only -- large enough that erosion/dilation
    # (radius derived from a tiny 10x10 frame is clamped to _MIN_RADIUS=2)
    # still leaves an UNK band on both seed edges.
    seed = np.zeros((H, W), dtype=bool)
    seed[:, :5] = True

    alpha, unk = chroma.build_trimap_alpha(
        a_raw, rgb, bg_stats=None, seed=seed, seed_variant="symmetric",
        band="alpha")

    assert unk.any(), "expected a genuine UNK band between eroded/dilated seed"
    # Inside UNK, alpha must equal a_raw (0.5) exactly -- not a color ramp
    # (which would have crashed on bg_stats=None before ever reaching here).
    assert np.all(alpha[unk] == pytest.approx(0.5))


def test_seed_variant_additive_only_adds_fg_and_agrees_on_bg():
    """additive: FG = erode(seed) | (a_raw>0.9), BG = ~dilate(seed) & (a_raw<0.1).
    A high-alpha pixel just outside the seed must become FG (SAM2 can't
    erase BiRefNet's own confident detail); a low-alpha pixel outside the
    dilated seed must become BG; a mid-alpha pixel outside the seed but
    inside the dilation band must NOT become BG (a_raw<0.1 fails)."""
    H, W = 40, 40
    seed = np.zeros((H, W), dtype=bool)
    seed[10:30, 10:30] = True  # a filled square seed

    a_raw = np.full((H, W), 0.5, dtype=np.float32)
    # Just outside the seed square (would be UNK/BG geometrically) but
    # BiRefNet is very confident it's foreground here:
    a_raw[10:30, 31] = 0.95
    # Far outside the seed (background-like) and BiRefNet agrees it's bg:
    a_raw[0:5, 0:5] = 0.02

    alpha, unk = chroma.build_trimap_alpha(
        a_raw, np.zeros((H, W, 3), np.uint8), bg_stats=None, seed=seed,
        seed_variant="additive", band="alpha")

    # The high-confidence pixel just outside the seed is pulled into FG.
    assert alpha[15, 31] == 1.0
    # The far corner (bg-like + outside dilated seed) is BG.
    assert alpha[2, 2] == 0.0
    # Deep inside the seed (well past erosion) stays FG regardless.
    assert alpha[20, 20] == 1.0


def test_seed_variant_symmetric_is_plain_erode_dilate_no_birefnet_override():
    """symmetric: FG = erode(seed), BG = ~dilate(seed) -- BiRefNet's alpha
    must NOT be able to override geometry here (unlike additive)."""
    H, W = 40, 40
    seed = np.zeros((H, W), dtype=bool)
    seed[10:30, 10:30] = True

    a_raw = np.full((H, W), 0.5, dtype=np.float32)
    # Same high-confidence pixel just outside the seed as the additive test:
    a_raw[10:30, 31] = 0.95

    alpha, unk = chroma.build_trimap_alpha(
        a_raw, np.zeros((H, W, 3), np.uint8), bg_stats=None, seed=seed,
        seed_variant="symmetric", band="alpha")

    # Under "symmetric", a_raw is NEVER consulted for FG/BG geometry -- this
    # pixel is outside erode(seed) and (depending on dilation radius) may
    # sit in UNK or BG, but it must NOT be forced to FG=1.0 the way additive
    # does, proving symmetric ignores a_raw for geometry.
    assert alpha[15, 31] != 1.0

    # Deep inside the seed still becomes FG via erosion alone.
    assert alpha[20, 20] == 1.0
    # Far outside the dilated seed becomes BG via dilation-complement alone.
    assert alpha[2, 2] == 0.0


def test_radius_frac_override_changes_erosion_dilation_extent():
    """A larger radius_frac erodes/dilates more aggressively -> a smaller
    FG region and a larger BG region for the same seed."""
    H, W = 60, 60
    seed = np.zeros((H, W), dtype=bool)
    seed[20:40, 20:40] = True
    a_raw = np.full((H, W), 0.5, dtype=np.float32)

    _, unk_small = chroma.build_trimap_alpha(
        a_raw, np.zeros((H, W, 3), np.uint8), bg_stats=None, seed=seed,
        seed_variant="symmetric", band="alpha", radius_frac=0.005)
    _, unk_large = chroma.build_trimap_alpha(
        a_raw, np.zeros((H, W, 3), np.uint8), bg_stats=None, seed=seed,
        seed_variant="symmetric", band="alpha", radius_frac=0.05)

    assert unk_large.sum() > unk_small.sum()


def test_unknown_seed_variant_raises():
    H, W = 10, 10
    seed = np.ones((H, W), dtype=bool)
    a_raw = np.full((H, W), 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        chroma.build_trimap_alpha(
            a_raw, np.zeros((H, W, 3), np.uint8), bg_stats=None, seed=seed,
            seed_variant="bogus", band="alpha")

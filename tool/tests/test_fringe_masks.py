"""Unit tests for the fringe-candidate-mask sharing between
fringe_strip_is_safe and strip_bg_fringe (a postprocess speed lever --
see stages._fringe_candidate_masks's own docstring). The optimisation is
purely "compute once, reuse" -- these tests pin the equivalence: with or
without a precomputed `candidate_masks` argument, both functions must
produce IDENTICAL results.
"""
import numpy as np

from tool.pipeline.config import PipelineConfig
from tool.pipeline.stages import (
    _fringe_candidate_masks,
    fringe_strip_is_safe,
    postprocess,
    strip_bg_fringe,
)

BG_RGB = np.array([60, 180, 60], np.float32)  # a flat green backdrop


def _frames_with_fringe(n=3, h=40, w=40, fringe_px=2):
    """A subject square with a `fringe_px`-wide ring of backdrop-ish
    colour surviving right at its opaque edge (surviving 1-bit alpha
    binarisation) -- exactly what strip_bg_fringe exists to remove."""
    frames = []
    for _ in range(n):
        f = np.zeros((h, w, 4), np.uint8)
        f[:, :, :3] = BG_RGB.astype(np.uint8)  # background colour everywhere in RGB
        f[10:30, 10:30, 3] = 255  # opaque subject region
        # a thin fringe strip just inside the opaque edge keeps the
        # BACKGROUND colour (simulating leaked backdrop at the silhouette
        # boundary), the rest of the interior is a clearly different colour
        f[10 + fringe_px:30 - fringe_px, 10 + fringe_px:30 - fringe_px, :3] = (220, 40, 40)
        frames.append(f)
    return frames


def _config(**kw):
    # mc_median_half=0: postprocess() runs mc_median before strip_bg_fringe,
    # and these tests pass src_gray=[] (no optical-flow input prepared) --
    # irrelevant to what's under test here (the fringe-mask sharing).
    return PipelineConfig(strip_bg_fringe=True, fringe_safety_max_frac=1.0,
                          mc_median_half=0, **kw)


def test_candidate_masks_matches_per_frame_computation():
    frames = _frames_with_fringe()
    cfg = _config()
    bg_hue, masks = _fringe_candidate_masks(frames, BG_RGB, cfg)
    assert bg_hue is not None
    assert len(masks) == len(frames)
    assert any(m.any() for m in masks), "the synthetic fringe should be detected as a candidate"


def test_candidate_masks_is_none_when_bg_ref_is_none():
    bg_hue, masks = _fringe_candidate_masks(_frames_with_fringe(), None, _config())
    assert bg_hue is None and masks is None


def test_fringe_strip_is_safe_gives_the_same_verdict_with_or_without_precomputed_masks():
    frames = _frames_with_fringe()
    cfg = _config()
    safe_a, frac_a = fringe_strip_is_safe(frames, BG_RGB, cfg)
    _, masks = _fringe_candidate_masks(frames, BG_RGB, cfg)
    safe_b, frac_b = fringe_strip_is_safe(frames, BG_RGB, cfg, candidate_masks=masks)
    assert safe_a == safe_b
    assert frac_a == frac_b


def test_strip_bg_fringe_gives_the_same_output_with_or_without_precomputed_masks():
    frames_a = _frames_with_fringe()
    frames_b = [f.copy() for f in frames_a]
    cfg = _config()

    strip_bg_fringe(frames_a, BG_RGB, cfg)  # computes its own masks internally
    _, masks = _fringe_candidate_masks(frames_b, BG_RGB, cfg)
    strip_bg_fringe(frames_b, BG_RGB, cfg, candidate_masks=masks)

    for a, b in zip(frames_a, frames_b):
        assert np.array_equal(a, b)


def test_strip_bg_fringe_actually_removes_the_synthetic_fringe():
    frames = _frames_with_fringe(fringe_px=2)
    cfg = _config()
    before_opaque = int((frames[0][:, :, 3] > 0).sum())
    strip_bg_fringe(frames, BG_RGB, cfg)
    after_opaque = int((frames[0][:, :, 3] > 0).sum())
    assert after_opaque < before_opaque


def test_postprocess_with_strip_bg_fringe_matches_calling_the_two_stages_manually():
    """Regression test for the mask-sharing refactor: postprocess()'s own
    strip_bg_fringe branch (which now threads a shared candidate_masks
    through both calls) must produce byte-identical output to the
    equivalent manual two-call sequence."""
    frames_a = _frames_with_fringe()
    frames_b = [f.copy() for f in frames_a]
    cfg = _config()

    postprocess(frames_a, {}, [], None, BG_RGB, cfg)

    safe, _ = fringe_strip_is_safe(frames_b, BG_RGB, cfg)
    assert safe
    strip_bg_fringe(frames_b, BG_RGB, cfg)

    for a, b in zip(frames_a, frames_b):
        assert np.array_equal(a, b)

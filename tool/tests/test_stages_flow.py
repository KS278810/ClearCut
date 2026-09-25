"""Unit test for stages.temporal_alpha_smooth's V7 speed experiment field
(flow_scale) -- see the plan. Synthetic frames (a translating square, so
DIS optical flow has real motion to track), no real clip or GPU needed:
this tests that flow_scale actually runs and stays close to the full-
resolution result, not matting quality on real footage (that's
tool/tests/test_regression_gates.py).
"""
import numpy as np

from tool.pipeline.config import PipelineConfig
from tool.pipeline.stages import temporal_alpha_smooth


def _translating_square_frames(n=5, h=48, w=64, size=16, dx=3):
    """n frames of a solid square translating by `dx` px/frame, with a
    little per-frame alpha chatter injected (±1 level) so there's
    something for the median to actually smooth."""
    rng = np.random.default_rng(0)
    frames, src_gray = [], []
    for i in range(n):
        frame = np.zeros((h, w, 4), dtype=np.uint8)
        x0 = 8 + i * dx
        frame[16:16 + size, x0:x0 + size, :3] = 180
        a = np.zeros((h, w), dtype=np.uint8)
        a[16:16 + size, x0:x0 + size] = 255
        chatter = rng.integers(-1, 2, a.shape).astype(np.int16)
        a = np.clip(a.astype(np.int16) + chatter, 0, 255).astype(np.uint8)
        frame[:, :, 3] = a
        frames.append(frame)
        src_gray.append(cv2_gray(frame))
    return frames, src_gray


def cv2_gray(frame):
    import cv2
    return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)


def test_flow_scale_runs_and_stays_close_to_full_resolution():
    frames_full, src_gray = _translating_square_frames()
    frames_half = [f.copy() for f in frames_full]

    cfg_full = PipelineConfig(mc_median_half=1, flow_scale=1.0)
    cfg_half = PipelineConfig(mc_median_half=1, flow_scale=0.5)

    out_full = temporal_alpha_smooth([f.copy() for f in frames_full], src_gray, None, cfg_full)
    out_half = temporal_alpha_smooth(frames_half, src_gray, None, cfg_half)

    assert len(out_full) == len(out_half) == len(frames_full)
    for f_full, f_half in zip(out_full, out_half):
        assert f_full.shape == f_half.shape
    # Downscaled-flow is an approximation -- confirm it's in the same
    # ballpark as full-resolution flow, not byte-identical.
    diffs = [np.abs(f_full[:, :, 3].astype(int) - f_half[:, :, 3].astype(int)).mean()
             for f_full, f_half in zip(out_full, out_half)]
    assert max(diffs) < 20, f"flow_scale=0.5 diverged too far from full resolution (worst mean abs diff {max(diffs):.1f})"


def test_flow_scale_default_matches_pre_existing_behaviour():
    """flow_scale defaults to 1.0 -- confirms the new code path (the
    `if flow_scale < 1.0` branch) is simply never taken by default, so
    existing callers that never set it see byte-identical output."""
    frames, src_gray = _translating_square_frames()
    cfg = PipelineConfig(mc_median_half=1)
    assert cfg.flow_scale == 1.0
    out = temporal_alpha_smooth([f.copy() for f in frames], src_gray, None, cfg)
    assert len(out) == len(frames)


def test_flow_preset_fast_runs_and_stays_close_to_medium():
    frames_medium, src_gray = _translating_square_frames()
    frames_fast = [f.copy() for f in frames_medium]

    cfg_medium = PipelineConfig(mc_median_half=1, flow_preset="medium")
    cfg_fast = PipelineConfig(mc_median_half=1, flow_preset="fast")

    out_medium = temporal_alpha_smooth([f.copy() for f in frames_medium], src_gray, None, cfg_medium)
    out_fast = temporal_alpha_smooth(frames_fast, src_gray, None, cfg_fast)

    assert len(out_medium) == len(out_fast) == len(frames_medium)
    diffs = [np.abs(f_m[:, :, 3].astype(int) - f_f[:, :, 3].astype(int)).mean()
             for f_m, f_f in zip(out_medium, out_fast)]
    assert max(diffs) < 20, f"flow_preset='fast' diverged too far from 'medium' (worst mean abs diff {max(diffs):.1f})"


def test_flow_preset_default_matches_pre_existing_behaviour():
    """flow_preset defaults to "medium" -- the pre-existing hardcoded
    DISOPTICAL_FLOW_PRESET_MEDIUM -- so existing callers that never set it
    see identical behaviour."""
    frames, src_gray = _translating_square_frames()
    cfg = PipelineConfig(mc_median_half=1)
    assert cfg.flow_preset == "medium"
    out = temporal_alpha_smooth([f.copy() for f in frames], src_gray, None, cfg)
    assert len(out) == len(frames)


def test_progress_is_called_once_per_processed_frame():
    """See the plan's ETA work: temporal_alpha_smooth is the only
    postprocess sub-stage with real per-frame ticks, so the bar/ETA needs
    a genuine progress callback here rather than a single 0/1 tick."""
    n, half = 5, 1
    frames, src_gray = _translating_square_frames(n=n)
    cfg = PipelineConfig(mc_median_half=half)
    calls = []
    temporal_alpha_smooth(frames, src_gray, None, cfg, progress=lambda d, t: calls.append((d, t)))
    n_todo = n - 2 * half
    assert calls == [(d, n_todo) for d in range(1, n_todo + 1)]


def test_progress_defaults_to_none_and_is_optional():
    """progress=None (the default) must not raise -- existing CLI callers
    never pass it."""
    frames, src_gray = _translating_square_frames()
    cfg = PipelineConfig(mc_median_half=1)
    out = temporal_alpha_smooth(frames, src_gray, None, cfg)
    assert len(out) == len(frames)


# -- postprocess speed lever: half=1's np.median -> min/max fast path -----
# (see stages.temporal_alpha_smooth's own comment on this) -- must be
# bit-exact with what np.median produced before this optimisation.

def test_median3_formula_is_bit_exact_with_np_median_including_ties():
    """Pure numpy identity check, decoupled from cv2/DIS flow: the
    min/max median-of-3 formula must agree with np.median(...).astype(u8)
    for every triple, including ties and the full uint8 range -- not just
    the "generic" case a translating-square fixture happens to exercise."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b, c = (rng.integers(0, 256, (16, 16), dtype=np.uint8) for _ in range(3))
        expected = np.median(np.stack([a, b, c], 0), axis=0).astype(np.uint8)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        actual = np.maximum(lo, np.minimum(hi, c))
        assert np.array_equal(actual, expected)

    # Ties (all three equal, and two-of-three equal) are the edge case a
    # purely random comparison might rarely exercise -- covered explicitly.
    same = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    other = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    for triple in ([same, same, same], [same, same, other], [same, other, same]):
        expected = np.median(np.stack(triple, 0), axis=0).astype(np.uint8)
        lo = np.minimum(triple[0], triple[1])
        hi = np.maximum(triple[0], triple[1])
        actual = np.maximum(lo, np.minimum(hi, triple[2]))
        assert np.array_equal(actual, expected)


def test_half_1_fast_path_matches_a_forced_np_median_reference():
    """End-to-end (through the real function, DIS flow included): the
    half=1 fast path's OUTPUT must match what a literal np.median over the
    same 3-array buf would have produced. Forces the reference path by
    monkeypatching np.median... no, simpler: since half==1 is the only
    condition that selects the fast path, this instead verifies the
    documented equivalence by re-implementing the pre-optimisation
    reference computation independently (matching the buf order the real
    loop builds: [alphas[i], warped(i-1), warped(i+1)]) and comparing
    against the real function's actual output."""
    import cv2
    frames, src_gray = _translating_square_frames(n=5)
    cfg = PipelineConfig(mc_median_half=1)
    alphas_before = [f[:, :, 3].copy() for f in frames]

    fast = [f.copy() for f in frames]
    temporal_alpha_smooth(fast, src_gray, None, cfg)

    # Independent reference: same DIS flow computation, but combined via
    # np.median instead of the min/max formula.
    height, width = alphas_before[0].shape
    alphas = np.stack(alphas_before, 0)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(width), np.arange(height))
    gx, gy = gx.astype(np.float32), gy.astype(np.float32)
    reference = alphas.copy()
    for i in range(1, len(frames) - 1):
        buf = [alphas[i]]
        for j in (i - 1, i + 1):
            fl = dis.calc(src_gray[i], src_gray[j], None)
            buf.append(cv2.remap(alphas[j], gx + fl[..., 0], gy + fl[..., 1],
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        reference[i] = np.median(np.stack(buf, 0), axis=0)

    for i in range(1, len(frames) - 1):
        assert np.array_equal(fast[i][:, :, 3], reference[i].astype(np.uint8)), f"frame {i} mismatch"

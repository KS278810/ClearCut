"""stages.keep_main_subject (第11計画 Part 3-1): drop opaque connected
components that are not the main subject, keeping every kept component's
soft edge intact."""
import numpy as np
import pytest

from tool.pipeline.config import PipelineConfig
from tool.pipeline.stages import keep_main_subject, postprocess


def _frame(h=120, w=160):
    return np.zeros((h, w, 4), np.uint8)


def _blob(f, y0, y1, x0, x1, rim=True):
    f[y0:y1, x0:x1, 3] = 255
    if rim:  # a 2px soft rim (alpha 60) around the opaque core
        f[y0 - 2:y0, x0:x1, 3] = 60
        f[y1:y1 + 2, x0:x1, 3] = 60
        f[y0:y1, x0 - 2:x0, 3] = 60
        f[y0:y1, x1:x1 + 2, 3] = 60


def test_small_detached_prop_is_removed_with_its_rim_and_subject_rim_survives():
    f = _frame()
    _blob(f, 10, 100, 10, 60)      # subject: 90x50 = 4500 px
    _blob(f, 90, 105, 120, 140)    # prop: 15x20 = 300 px (6.7% of subject)
    before = f.copy()
    keep_main_subject([f], None, PipelineConfig())
    assert (f[90:105, 120:140, 3] == 0).all()
    assert (f[88:90, 120:140, 3] == 0).all(), "the dropped prop's soft rim goes with it"
    np.testing.assert_array_equal(f[:, :80], before[:, :80])  # subject incl. its soft rim untouched


def test_comparably_sized_second_component_is_kept():
    f = _frame()
    _blob(f, 10, 100, 10, 60)     # 4500 px
    _blob(f, 10, 60, 100, 130)    # 1500 px = 33% of the largest (>= 25%)
    before = f.copy()
    keep_main_subject([f], None, PipelineConfig())
    np.testing.assert_array_equal(f, before)


def test_component_touching_the_subject_box_is_kept_even_if_small():
    """(iii): e.g. a hand separated from the torso by a thin gap, still inside
    the person's own YOLOX box."""
    f = _frame()
    _blob(f, 10, 100, 10, 60)
    _blob(f, 20, 30, 70, 80, rim=False)     # 100 px, inside the box below
    _blob(f, 90, 105, 120, 140, rim=False)  # outside the box
    keep_main_subject([f], [np.array([5.0, 5.0, 90.0, 110.0])], PipelineConfig())
    assert (f[20:30, 70:80, 3] == 255).all()
    assert (f[90:105, 120:140, 3] == 0).all()


def test_frames_without_a_box_fall_back_to_the_area_rules():
    f1, f2 = _frame(), _frame()
    for f in (f1, f2):
        _blob(f, 10, 100, 10, 60)
        _blob(f, 20, 30, 70, 80, rim=False)
    keep_main_subject([f1, f2], [np.array([5.0, 5.0, 90.0, 110.0]), np.full(4, np.nan)], PipelineConfig())
    assert (f1[20:30, 70:80, 3] == 255).all()
    assert (f2[20:30, 70:80, 3] == 0).all()


def test_soft_alpha_far_from_any_dropped_component_is_left_alone():
    f = _frame()
    _blob(f, 10, 100, 10, 60)
    f[110:118, 100:150, 3] = 40  # faint haze, no opaque core -> not a component
    before = f.copy()
    keep_main_subject([f], None, PipelineConfig())
    np.testing.assert_array_equal(f, before)


def test_rgb_is_untouched_and_single_component_frames_are_a_no_op():
    f = _frame()
    f[..., :3] = 77
    _blob(f, 10, 100, 10, 60)
    before = f.copy()
    _, removed, sam2_used, sam2_fallback = keep_main_subject([f], None, PipelineConfig())
    assert removed == [0]
    assert (sam2_used, sam2_fallback) == (0, 0)
    np.testing.assert_array_equal(f, before)


@pytest.mark.parametrize("flag", [True, False])
def test_postprocess_honours_the_config_flag(flag):
    f = _frame()
    _blob(f, 10, 100, 10, 60)
    _blob(f, 90, 105, 120, 140, rim=False)
    cfg = PipelineConfig(keep_main_subject=flag, mc_median_half=0)
    postprocess([f], {}, [], None, None, cfg)
    assert (f[95, 130, 3] == 0) == flag


def test_default_is_off_after_the_failed_gate():
    """Pre-registered gate failed on the wide-pose real clip fixture (see config.py) -> opt-in only."""
    assert PipelineConfig().keep_main_subject is False


# --------------------------------------------------------------------------
# 第12計画: rule (iii) refined with a SAM2 pixel mask
# --------------------------------------------------------------------------

def _person_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def test_sam2_drops_a_prop_fully_inside_the_box_that_the_box_rule_could_not():
    """A wide-pose real clip's bottle prop: fully inside the wide-pose person box,
    so no box-rectangle rule (however tuned) can separate it -- only a pixel
    mask can. This is the exact case the box-only rule (iii) failed on."""
    f = _frame()
    _blob(f, 10, 100, 10, 60, rim=False)      # person: 90x50 = 4500 px
    _blob(f, 70, 90, 65, 90, rim=False)       # prop fully inside the box below: 20x25 = 500 px
    box = np.array([5.0, 5.0, 95.0, 110.0])
    sam2_mask = _person_mask(120, 160, 10, 100, 10, 60)  # matches the person blob exactly
    _, removed, sam2_used, sam2_fallback = keep_main_subject(
        [f], [box], PipelineConfig(), sam2_fn=lambda rgb, b: sam2_mask)
    assert (f[70:90, 65:90, 3] == 0).all(), "prop inside the box but outside the SAM2 mask is dropped"
    assert (f[10:100, 10:60, 3] == 255).all(), "person untouched"
    assert (sam2_used, sam2_fallback) == (1, 0)
    assert removed == [500]


def test_sam2_keeps_a_component_that_overlaps_the_dilated_mask():
    """A hand separated from the torso by a thin gap: still overlaps the
    (dilated) SAM2 person mask, so it survives -- same intent as the old
    box-touch test, now via the mask."""
    f = _frame()
    _blob(f, 10, 100, 10, 60, rim=False)
    _blob(f, 20, 30, 62, 72, rim=False)  # just outside the person blob, inside the box
    box = np.array([5.0, 5.0, 95.0, 110.0])
    sam2_mask = _person_mask(120, 160, 10, 100, 10, 60)
    keep_main_subject([f], [box], PipelineConfig(), sam2_fn=lambda rgb, b: sam2_mask)
    assert (f[20:30, 62:72, 3] == 255).all(), "close enough to the dilated mask to be kept"


def test_sam2_small_but_nonempty_mask_is_used_not_rejected():
    """第13計画: a box-area-ratio health check used to reject a mask far
    smaller than the box (第12計画's original design) -- REMOVED after
    calibration showed it rejected CORRECT masks on the widepose clip's own
    frames (a spread-limbs box is mostly empty space, so a small
    mask/box ratio there is expected, not a sign SAM2 failed -- see
    DECISIONS.md). Any non-empty mask is now used as-is, however small
    relative to the box."""
    f = _frame()
    _blob(f, 10, 100, 10, 60, rim=False)
    _blob(f, 70, 90, 65, 90, rim=False)  # inside the box, outside a small-but-real mask
    box = np.array([5.0, 5.0, 95.0, 110.0])
    tiny_but_real_mask = _person_mask(120, 160, 50, 55, 30, 35)  # far smaller than the box area
    _, removed, sam2_used, sam2_fallback = keep_main_subject(
        [f], [box], PipelineConfig(), sam2_fn=lambda rgb, b: tiny_but_real_mask)
    assert (f[70:90, 65:90, 3] == 0).all(), "no overlap with the (small but real) mask -> dropped"
    assert (sam2_used, sam2_fallback) == (1, 0)
    assert removed == [500]


def test_sam2_empty_mask_falls_back_to_the_box_rule():
    """The only remaining fallback condition (第13計画): SAM2 returned
    literally nothing (all-zero mask) -- a genuine SAM2 failure, not a
    box-area-ratio judgement call."""
    f = _frame()
    _blob(f, 10, 100, 10, 60, rim=False)
    _blob(f, 70, 90, 65, 90, rim=False)  # inside the box
    box = np.array([5.0, 5.0, 95.0, 110.0])
    empty_mask = np.zeros((120, 160), np.float32)
    _, removed, sam2_used, sam2_fallback = keep_main_subject(
        [f], [box], PipelineConfig(), sam2_fn=lambda rgb, b: empty_mask)
    assert (f[70:90, 65:90, 3] == 255).all(), "box-rule fallback keeps it, same as the pre-第12計画 behaviour"
    assert (sam2_used, sam2_fallback) == (0, 1)
    assert removed == [0]


def test_sam2_fn_none_is_byte_identical_to_pre_lever_behaviour():
    """sam2_fn omitted -> box-rectangle rule for every frame, unchanged."""
    f = _frame()
    _blob(f, 10, 100, 10, 60, rim=False)
    _blob(f, 20, 30, 70, 80, rim=False)
    box = np.array([5.0, 5.0, 90.0, 110.0])
    keep_main_subject([f], [box], PipelineConfig())
    assert (f[20:30, 70:80, 3] == 255).all()


def test_sam2_call_raising_falls_back_to_the_box_rule():
    """A raising sam2_fn (e.g. the SAM2 subprocess timed out) must not kill
    the run -- fall back to the box rule for that frame."""
    f = _frame()
    _blob(f, 10, 100, 10, 60, rim=False)
    _blob(f, 70, 90, 65, 90, rim=False)  # inside the box
    box = np.array([5.0, 5.0, 95.0, 110.0])

    def _boom(rgb, b):
        raise TimeoutError("subprocess hung")

    _, removed, sam2_used, sam2_fallback = keep_main_subject([f], [box], PipelineConfig(), sam2_fn=_boom)
    assert (f[70:90, 65:90, 3] == 255).all(), "box-rule fallback on exception"
    assert (sam2_used, sam2_fallback) == (0, 0), "not counted as an attempted SAM2 call once it raised"


def test_sam2_not_called_when_area_rules_already_keep_everything():
    """No point calling SAM2 if rule (iii) can't change anything -- both
    components already comparable in size."""
    f = _frame()
    _blob(f, 10, 100, 10, 60, rim=False)   # 4500 px
    _blob(f, 10, 60, 100, 130, rim=False)  # 1500 px, >= 25% of the largest -> already kept by (ii)
    box = np.array([5.0, 5.0, 95.0, 110.0])
    calls = []

    def _spy(rgb, b):
        calls.append(1)
        return _person_mask(120, 160, 10, 100, 10, 60)

    keep_main_subject([f], [box], PipelineConfig(), sam2_fn=_spy)
    assert calls == [], "SAM2 must not be called when (i)/(ii) already keep every component"

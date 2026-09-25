"""Unit tests for Models.birefnet's crop-truncation fallback (found via a
real production defect: a confident-but-partial YOLOX box amputated a
mascot's head/frill along a dead-straight line, F1i=505,397 on a
the flatchroma2 clips -- see tool/docs/DECISIONS.md's
known-defects entry).

No real ONNX/BiRefNet involved: Models.birefnet() only depends on
self._birefnet_crop_pass and self.box_reinfer_frames, both duck-typeable,
so a lightweight fake stands in for the whole network. This tests the
CONTROL FLOW (detect truncation -> fall back to a full-frame pass -> count
it), not matting quality on real footage -- that needs a real render
(tracked separately, GPU/CPU-bound).
"""
import numpy as np

from tool.matte_core import Models, _alpha_touches_a_crop_edge


class _FakeModelsBackend:
    """Stands in for Models: only what birefnet() actually touches."""

    def __init__(self, crop_alpha, full_alpha=None):
        self.box_reinfer_frames = 0
        self._crop_alpha = crop_alpha    # (out, x0, y0, x1, y1) for the box-based call
        self._full_alpha = full_alpha    # (out, x0, y0, x1, y1) for the box=None fallback call
        self.calls = []                  # records which pass(es) ran, in order

    def _birefnet_crop_pass(self, rgb, box, margin):
        if box is None:
            self.calls.append("full")
            return self._full_alpha
        self.calls.append("crop")
        return self._crop_alpha


def _touches_frame_border_only(H=100, W=100, x0=20, y0=0, x1=80, y1=80, edge_val=0.9):
    """A crop clamped to the top frame edge (y0=0) with alpha touching that
    row -- the LEGITIMATE case (subject genuinely exits frame there)."""
    out = np.zeros((H, W), np.float32)
    out[y0, 40:50] = edge_val
    return out, x0, y0, x1, y1


def _touches_an_interior_crop_edge(H=100, W=100, x0=20, y0=20, x1=80, y1=80, edge_val=0.9):
    """A crop NOT touching any frame boundary, with alpha right on its own
    top edge -- the AMPUTATION signature this fallback exists to catch."""
    out = np.zeros((H, W), np.float32)
    out[y0, 40:50] = edge_val
    return out, x0, y0, x1, y1


def test_no_fallback_when_the_crop_alpha_stays_clear_of_every_edge():
    out, x0, y0, x1, y1 = np.zeros((100, 100), np.float32), 20, 20, 80, 80
    out[40:60, 40:60] = 0.9  # well inside the crop
    fake = _FakeModelsBackend(crop_alpha=(out, x0, y0, x1, y1))
    result = Models.birefnet(fake, np.zeros((100, 100, 3), np.uint8), box=np.array([10, 10, 90, 90]))
    assert fake.calls == ["crop"]
    assert fake.box_reinfer_frames == 0
    assert np.array_equal(result, out)


def test_no_fallback_when_alpha_touches_only_the_frame_boundary():
    fake = _FakeModelsBackend(crop_alpha=_touches_frame_border_only())
    Models.birefnet(fake, np.zeros((100, 100, 3), np.uint8), box=np.array([20, 0, 80, 80]))
    assert fake.calls == ["crop"]
    assert fake.box_reinfer_frames == 0


def test_falls_back_to_a_full_frame_pass_when_the_crop_edge_is_amputated():
    crop_out, cx0, cy0, cx1, cy1 = _touches_an_interior_crop_edge()
    full_out = np.zeros((100, 100), np.float32)
    full_out[5:95, 30:90] = 0.9  # the full-frame pass recovers the whole subject
    fake = _FakeModelsBackend(crop_alpha=(crop_out, cx0, cy0, cx1, cy1),
                              full_alpha=(full_out, 0, 0, 100, 100))
    result = Models.birefnet(fake, np.zeros((100, 100, 3), np.uint8), box=np.array([25, 25, 75, 75]))
    assert fake.calls == ["crop", "full"], "must fall back exactly once, in order"
    assert fake.box_reinfer_frames == 1
    assert np.array_equal(result, full_out), "the recovered alpha must be the full-frame pass's result"


def test_does_not_recurse_even_if_the_full_frame_pass_also_looks_edge_touched():
    """The fallback pass uses box=None -- its crop rect IS the whole frame,
    so every one of its edges is a frame boundary by construition and
    _alpha_touches_a_crop_edge can never fire on it. This pins that
    "at most one extra call" guarantee directly, in case that invariant is
    ever accidentally broken."""
    crop_out, cx0, cy0, cx1, cy1 = _touches_an_interior_crop_edge()
    full_out = np.zeros((100, 100), np.float32)
    full_out[0, :] = 0.9  # alpha right on the frame's own top row
    fake = _FakeModelsBackend(crop_alpha=(crop_out, cx0, cy0, cx1, cy1),
                              full_alpha=(full_out, 0, 0, 100, 100))
    Models.birefnet(fake, np.zeros((100, 100, 3), np.uint8), box=np.array([25, 25, 75, 75]))
    assert fake.calls == ["crop", "full"]
    assert fake.box_reinfer_frames == 1


def test_box_none_never_triggers_the_fallback_itself():
    """birefnet(box=None) (the general/full-frame call, e.g. no detection at
    all) must never re-enter the fallback -- there is no "original box" to
    have been truncated by."""
    out = np.zeros((100, 100), np.float32)
    out[0, :] = 0.9
    # box=None means the top-level call itself is the "full-frame" pass, so
    # the fake needs an answer under _both_ dispatch keys -- box=None is
    # indistinguishable, at the _birefnet_crop_pass level, from birefnet's
    # own internal fallback call.
    fake = _FakeModelsBackend(crop_alpha=(out, 0, 0, 100, 100), full_alpha=(out, 0, 0, 100, 100))
    Models.birefnet(fake, np.zeros((100, 100, 3), np.uint8), box=None)
    assert fake.calls == ["full"]
    assert fake.box_reinfer_frames == 0


# -- _alpha_touches_a_crop_edge itself (the detection primitive) ----------

def test_touches_each_of_the_four_interior_edges_independently():
    H, W, x0, y0, x1, y1 = 100, 100, 20, 20, 80, 80
    for edge, coords in {
        "top": (y0, slice(40, 50)),
        "bottom": (y1 - 1, slice(40, 50)),
    }.items():
        out = np.zeros((H, W), np.float32)
        out[coords] = 0.9
        assert _alpha_touches_a_crop_edge(out, x0, y0, x1, y1, W, H), edge

    for edge, coords in {
        "left": (slice(40, 50), x0),
        "right": (slice(40, 50), x1 - 1),
    }.items():
        out = np.zeros((H, W), np.float32)
        out[coords] = 0.9
        assert _alpha_touches_a_crop_edge(out, x0, y0, x1, y1, W, H), edge


def test_a_frame_clamped_edge_never_counts_as_truncated():
    """Every one of the 4 edges, when clamped to the frame boundary, must
    be exempt -- not just the one this module's docstring example uses."""
    H, W = 100, 100
    for x0, y0, x1, y1, touch in [
        (0, 20, 80, 80, (slice(30, 40), 0)),     # left == frame edge
        (20, 0, 80, 80, (0, slice(30, 40))),     # top == frame edge
        (20, 20, 100, 80, (slice(30, 40), 99)),  # right == frame edge
        (20, 20, 80, 100, (99, slice(30, 40))),  # bottom == frame edge
    ]:
        out = np.zeros((H, W), np.float32)
        out[touch] = 0.9
        assert not _alpha_touches_a_crop_edge(out, x0, y0, x1, y1, W, H)


def test_alpha_below_threshold_does_not_count():
    H, W, x0, y0, x1, y1 = 100, 100, 20, 20, 80, 80
    out = np.zeros((H, W), np.float32)
    out[y0, 40:50] = 0.3  # below the default thr=0.5
    assert not _alpha_touches_a_crop_edge(out, x0, y0, x1, y1, W, H)

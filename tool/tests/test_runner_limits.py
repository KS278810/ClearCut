"""tool/pipeline/runner.py's max_frames guard (ClipTooLong) -- defense in
depth behind server/probe.py's own upload-time frame-count validation.
See the plan's B1 finding: a webm (or fragmented mov) container can report
CAP_PROP_FRAME_COUNT as 0 or -1, which would bypass that check entirely;
this catches an over-long clip from the inside too, regardless of how
infer_clip ends up being called.

Uses a stub Models (no real GPU/CPU matting work) so this stays fast and
independent of device availability -- it's testing the frame-count guard,
not matting quality.
"""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

import tool.ffmpeg_encoders as ffenc
from tool.pipeline.config import PipelineConfig
from tool.pipeline.runner import ClipTooLong, infer_clip, run_clip

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLIP = ROOT.parent / "sample" / "dinosaur" / "Triceratops.mp4"

pytestmark = pytest.mark.skipif(not SAMPLE_CLIP.is_file(),
                                 reason="sample/dinosaur/Triceratops.mp4 not present on this machine")


class _StubModels:
    """Trivial stand-in for matte_core.Models: a fixed full-frame box and a
    constant alpha. Real matting quality is irrelevant to this test."""

    def subject_box(self, rgb, prefer_person=False, min_box_frac=0.15):
        h, w = rgb.shape[:2]
        return (0, 0, w, h), 1.0

    def birefnet(self, rgb, box=None, margin=0.12, auto_full_frame_fallback=True):
        h, w = rgb.shape[:2]
        return np.ones((h, w), dtype=np.float32)


def _config():
    # apply_despill=False keeps this test pure numpy/logic, no pymatting work.
    return PipelineConfig(apply_despill=False)


def test_max_frames_raises_clip_too_long_before_reading_the_whole_clip():
    with pytest.raises(ClipTooLong, match="max_frames=3"):
        infer_clip(SAMPLE_CLIP, _config(), models=_StubModels(), max_frames=3)


def test_no_max_frames_means_unlimited_as_before():
    raw = infer_clip(SAMPLE_CLIP, _config(), models=_StubModels(), max_frames=None)
    assert len(raw["frames"]) == 121  # sample/dinosaur/Triceratops.mp4's known frame count


def test_run_clip_raises_a_clear_error_for_a_zero_frame_raw(tmp_path):
    """Audit L2: run_clip used to index `frames[0]` unconditionally to read
    the output resolution, which raised a bare IndexError (no hint what
    went wrong) for a corrupt/0-frame source or a pre-existing cached raw
    result built before this guard existed. A synthetic empty `raw` is
    enough to exercise this -- no real decode needed."""
    empty_raw = {"frames": [], "clear_masks": {}, "src_gray": [], "motion_dmax": None,
                 "bg_ref": None, "fps": 24.0}
    with pytest.raises(RuntimeError, match="no frames to encode"):
        run_clip(SAMPLE_CLIP, tmp_path / "out.gif", _config(), raw=empty_raw)


def test_preview_callback_receives_one_call_per_frame_with_the_frame_itself():
    """Regression test for the live-preview feature: infer_clip must call
    preview(done, frame) once per frame with the just-produced RGBA frame,
    not just a count -- the server needs the actual pixels to encode and
    serve a preview image.

    RE-UPDATED for 第11計画 Part 1-2 (2026-09-23): the default box_mode
    ("per_frame") is a single streaming loop again, so -- as before レバーC's
    two-pass layout -- a clip that exceeds max_frames has already previewed
    every frame up to the cutoff (the two-pass layout gave ZERO calls,
    because nothing reached BiRefNet until the whole clip was decoded)."""
    calls = []

    def on_preview(done, frame):
        calls.append((done, frame.shape, frame.dtype))

    with pytest.raises(ClipTooLong):
        infer_clip(SAMPLE_CLIP, _config(), models=_StubModels(), max_frames=3, preview=on_preview)

    assert [c[0] for c in calls] == [1, 2, 3]

    calls2 = []

    def on_preview2(done, frame):
        calls2.append((done, frame.shape, frame.dtype))

    infer_clip(SAMPLE_CLIP, _config(), models=_StubModels(), max_frames=200, preview=on_preview2)

    assert [c[0] for c in calls2] == list(range(1, 122))  # known frame count
    for _done, shape, dtype in calls + calls2:
        assert len(shape) == 3 and shape[2] == 4  # RGBA
        assert dtype == np.uint8


class _OrderRecordingModels(_StubModels):
    def __init__(self, log):
        self.log = log

    def subject_box(self, rgb, prefer_person=False, min_box_frac=0.15):
        self.log.append("detect")
        return super().subject_box(rgb, prefer_person, min_box_frac)

    def birefnet(self, rgb, box=None, margin=0.12, auto_full_frame_fallback=True):
        self.log.append(("birefnet", auto_full_frame_fallback))
        return super().birefnet(rgb, box, margin, auto_full_frame_fallback)


def test_per_frame_mode_streams_one_frame_at_a_time_with_prepare_first():
    """第11計画 Part 1-2/1-3: the first preview/progress must follow the
    FIRST frame's detect+BiRefNet, not the whole clip's (the two-pass
    layout gave ~40-60s GPU / >1h CPU of silence), BiRefNet must use its
    own single-frame full-frame fallback, and "prepare" must be the very
    first progress call so the server heartbeat can publish from t=0."""
    log = []
    infer_clip(SAMPLE_CLIP, _config(), models=_OrderRecordingModels(log), max_frames=200,
               progress=lambda st, d, t: log.append(("progress", st, d)),
               preview=lambda d, f: log.append(("preview", d)))
    assert log[0] == ("progress", "prepare", 0)
    assert log[1] == ("progress", "prepare", 1)
    assert log[2:7] == ["detect", ("birefnet", True), ("preview", 1), ("progress", "infer", 1), "detect"]
    assert sum(1 for e in log if e == "detect") == 121


@pytest.mark.parametrize("box_mode", ["clip_union", "smoothed"])
def test_opt_in_box_modes_keep_the_two_pass_path(box_mode):
    log = []
    cfg = dataclasses.replace(_config(), box_mode=box_mode)
    raw = infer_clip(SAMPLE_CLIP, cfg, models=_OrderRecordingModels(log), max_frames=200,
                     preview=lambda d, f: log.append(("preview", d)))
    assert len(raw["frames"]) == 121
    first_preview = log.index(("preview", 1))
    assert log[:121] == ["detect"] * 121  # whole-clip detection before any BiRefNet
    assert all(e == ("birefnet", False) for e in log[121:first_preview])


def test_infer_clip_writes_stage_timings_into_the_caller_dict():
    """Regression test for the plan's B5 finding: per-stage timing only
    ever went to stdout, so a server job's job.json (which holds this same
    dict as clip["timings"]) recorded nothing at all -- a failure couldn't
    be told apart from a fast success."""
    timings = {}
    infer_clip(SAMPLE_CLIP, _config(), models=_StubModels(), timings=timings)
    assert set(timings) == {"detect_s", "birefnet_s", "despill_s"}
    assert all(isinstance(v, float) and v >= 0 for v in timings.values())


def test_run_clip_keeps_partial_timings_when_encode_fails(tmp_path, monkeypatch):
    """Regression test for the plan's B4/B5 findings: an encode failure
    (timeout, in practice) used to leave clip["timings"] at `{}` (no
    breakdown at all) AND left a 0-byte output file behind, unreferenced
    by clip["outputs"] and invisible to the UI/zip until the 14-day
    retention sweep. run_clip must now (a) record postprocess_s/encode_s
    even though encode raised, (b) NOT record total_s (the clip never
    finished), and (c) remove the partial output file _encode's ffmpeg
    call created the instant it launched."""
    def _boom(*a, **kw):
        raise RuntimeError("synthetic encode failure")

    # encoder="supersampled_gif" pinned explicitly (not PipelineConfig's own
    # default, which stopped being supersampled_gif on 2026-09-22 -- 第9計画
    # レバー4) so the patched function is actually the one run_clip calls,
    # independent of whichever encoder is currently the default.
    monkeypatch.setattr(ffenc, "encode_gif_supersampled", _boom)
    config = dataclasses.replace(_config(), encoder="supersampled_gif")
    out_path = tmp_path / "out.gif"
    out_path.write_bytes(b"")  # simulates ffmpeg having already created the file

    timings = {}
    with pytest.raises(RuntimeError, match="synthetic encode failure"):
        run_clip(SAMPLE_CLIP, out_path, config, models=_StubModels(), timings=timings)

    assert set(timings) == {"detect_s", "birefnet_s", "despill_s", "postprocess_s", "encode_s"}
    assert "total_s" not in timings
    assert not out_path.exists()


def test_infer_clip_returns_one_raw_subject_box_per_frame():
    """第11計画 Part 3-1: keep_main_subject (postprocess) reads raw["boxes"]."""
    raw = infer_clip(SAMPLE_CLIP, _config(), models=_StubModels(), max_frames=200)
    assert len(raw["boxes"]) == len(raw["frames"]) == 121
    assert tuple(raw["boxes"][0]) == (0, 0, raw["frames"][0].shape[1], raw["frames"][0].shape[0])

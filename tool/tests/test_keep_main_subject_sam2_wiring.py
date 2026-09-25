"""runner.run_clip's SAM2-worker wiring for keep_main_subject's rule (iii)
(第12計画 Commit 2). A synthetic `raw` (no real decode/inference needed --
same pattern as test_runner_limits.py) exercises just the wiring: does a
worker get created exactly when it should, does postprocess actually
receive a callable sam2_fn, and is the worker always stopped."""
from pathlib import Path

import numpy as np
import pytest

import tool.pipeline.runner as runner_mod
from tool.pipeline.config import PipelineConfig
from tool.pipeline.runner import run_clip

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLIP = ROOT.parent / "sample" / "dinosaur" / "Triceratops.mp4"


def _raw(n=2, keyed=False):
    h, w = 8, 8
    frames = [np.zeros((h, w, 4), np.uint8) for _ in range(n)]
    return {"frames": frames, "clear_masks": {}, "src_gray": [], "motion_dmax": None,
            "bg_ref": None, "fps": 24.0, "boxes": None, "keyed": keyed}


class _SpyWorker:
    """Stands in for matte_core._Sam2Worker: records construction and stop()
    calls without touching a real subprocess/GPU."""
    instances = []

    def __init__(self, device):
        self.device = device
        self.stopped = False
        _SpyWorker.instances.append(self)

    def mask(self, rgb, box):  # pragma: no cover -- never called, no boxes in this raw
        raise AssertionError("mask() should not be called when there are no boxes")

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def _reset_spy():
    _SpyWorker.instances.clear()
    yield
    _SpyWorker.instances.clear()


def _config(**kw):
    return PipelineConfig(apply_despill=False, mc_median_half=0, **kw)


def test_worker_created_and_stopped_when_flag_on_and_device_resolves_to_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=True, device="auto")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw())
    assert len(_SpyWorker.instances) == 1
    assert _SpyWorker.instances[0].device == "cuda"
    assert _SpyWorker.instances[0].stopped is True


def test_no_worker_when_device_resolves_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cpu")
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=True, device="cpu")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw())
    assert _SpyWorker.instances == []


def test_no_worker_on_the_keyer_route_even_with_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=True, device="cuda")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw(keyed=True))
    assert _SpyWorker.instances == [], "keyer route has no boxes for a SAM2 prompt"


def test_no_worker_when_keep_main_subject_sam2_flag_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=False, device="cuda")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw())
    assert _SpyWorker.instances == []


def test_no_worker_when_keep_main_subject_itself_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    cfg = _config(keep_main_subject=False, keep_main_subject_sam2=True, device="cuda")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw())
    assert _SpyWorker.instances == []


def test_worker_stopped_even_if_postprocess_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_mod, "postprocess", _boom)
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=True, device="cuda")
    with pytest.raises(RuntimeError, match="boom"):
        run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw())
    assert len(_SpyWorker.instances) == 1
    assert _SpyWorker.instances[0].stopped is True


def test_sam2_fn_is_forwarded_to_postprocess_as_a_callable(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    received = {}

    def _spy_postprocess(frames, clear_masks, src_gray, motion_dmax, bg_ref, config,
                          progress=None, boxes=None, sam2_fn=None):
        received["sam2_fn"] = sam2_fn
        return frames

    monkeypatch.setattr(runner_mod, "postprocess", _spy_postprocess)
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=True, device="cuda")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw())
    assert callable(received["sam2_fn"])


def test_sam2_fn_is_none_when_lever_off(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    received = {}

    def _spy_postprocess(frames, clear_masks, src_gray, motion_dmax, bg_ref, config,
                          progress=None, boxes=None, sam2_fn=None):
        received["sam2_fn"] = sam2_fn
        return frames

    monkeypatch.setattr(runner_mod, "postprocess", _spy_postprocess)
    cfg = _config(keep_main_subject=False, device="cuda")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw())
    assert received["sam2_fn"] is None


def test_keep_main_subject_itself_forced_off_on_the_keyer_route(tmp_path, monkeypatch):
    """第13計画: not just the SAM2 refinement (sam2_fn) -- keep_main_subject
    ITSELF must be forced off for a keyed clip, since rules (i)/(ii) alone
    (no boxes on the keyer route) were measured to cost F1i on a real-footage
    fixtures (see DECISIONS.md's 第12計画 entry)."""
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    received = {}

    def _spy_postprocess(frames, clear_masks, src_gray, motion_dmax, bg_ref, config,
                          progress=None, boxes=None, sam2_fn=None):
        received["config"] = config
        return frames

    monkeypatch.setattr(runner_mod, "postprocess", _spy_postprocess)
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=True, device="cuda")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw(keyed=True))
    assert received["config"].keep_main_subject is False


def test_keep_main_subject_unaffected_on_the_neural_route(tmp_path, monkeypatch):
    """Sanity check for the test above: the neural route's config is NOT
    force-replaced, so keep_main_subject=True actually reaches postprocess
    there."""
    monkeypatch.setattr(runner_mod, "_Sam2Worker", _SpyWorker)
    monkeypatch.setattr(runner_mod, "resolve_device", lambda pref: "cuda")
    received = {}

    def _spy_postprocess(frames, clear_masks, src_gray, motion_dmax, bg_ref, config,
                          progress=None, boxes=None, sam2_fn=None):
        received["config"] = config
        return frames

    monkeypatch.setattr(runner_mod, "postprocess", _spy_postprocess)
    cfg = _config(keep_main_subject=True, keep_main_subject_sam2=True, device="cuda")
    run_clip(SAMPLE_CLIP, tmp_path / "out.gif", cfg, raw=_raw(keyed=False))
    assert received["config"].keep_main_subject is True

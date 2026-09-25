"""tool/pipeline/cache.py's _key() must include every PipelineConfig field
that infer_clip() actually reads before/while building the cached frames --
see the docstring on _key() itself and audit A1: `device` was missing here,
so after matte_core.Models started honouring config.device for the YOLOX
detector (instead of hardcoding "cpu"), a cache entry built under one
device could be silently reused under the other, even though the
detector's box (and therefore the matte) can differ between them.
"""
import dataclasses

import numpy as np
import pytest

from tool.pipeline import cache
from tool.pipeline.cache import _key, load_raw, save_raw
from tool.pipeline.config import PipelineConfig


def test_key_differs_between_cpu_and_cuda_device():
    cpu_config = PipelineConfig(device="cpu")
    cuda_config = PipelineConfig(device="cuda")
    assert _key("clip", cpu_config) != _key("clip", cuda_config)


def test_key_is_stable_for_the_same_config():
    config = PipelineConfig(device="cuda")
    assert _key("clip", config) == _key("clip", dataclasses.replace(config))


# -- audit H1/H2: the keyer path's raw dict round-trip through the disk cache --
# tool/pipeline/keyer.py's infer_clip branch returns src_gray=[], motion_dmax=
# None and clear_masks={} (it has no optical-flow/clears input to offer), and
# sets raw["keyed"]=True. save_raw/load_raw must round-trip that shape without
# crashing (np.stack([]) raises) and without losing the "keyed" flag (losing it
# would make run_clip try to mc_median an empty src_gray on a cache hit).

def _fake_keyed_raw(n=3, h=8, w=8):
    frames = [np.zeros((h, w, 4), np.uint8) for _ in range(n)]
    return {"frames": frames, "clear_masks": {}, "src_gray": [],
            "motion_dmax": None, "bg_ref": np.array([10.0, 20.0, 30.0], np.float32),
            "fps": 24.0, "keyed": True}


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


def test_save_raw_does_not_crash_on_a_keyed_result_with_empty_src_gray(cache_dir):
    save_raw("clip", PipelineConfig(), _fake_keyed_raw())  # must not raise


def test_keyed_raw_round_trips_through_the_cache(cache_dir):
    raw = _fake_keyed_raw()
    save_raw("clip", PipelineConfig(), raw)
    loaded = load_raw("clip", PipelineConfig())
    assert loaded is not None
    assert loaded["keyed"] is True
    assert loaded["src_gray"] == []
    assert loaded["motion_dmax"] is None
    assert len(loaded["frames"]) == 3
    assert np.allclose(loaded["bg_ref"], raw["bg_ref"])


def test_a_non_keyed_raw_round_trips_without_the_keyed_flag(cache_dir):
    """The ordinary BiRefNet-route shape (non-empty src_gray, no "keyed" key
    at all) must still round-trip exactly as before this fix."""
    raw = _fake_keyed_raw()
    raw["keyed"] = False
    raw["src_gray"] = [np.zeros((8, 8), np.uint8) for _ in range(3)]
    del raw["keyed"]
    save_raw("clip", PipelineConfig(), raw)
    loaded = load_raw("clip", PipelineConfig())
    assert "keyed" not in loaded or not loaded["keyed"]
    assert len(loaded["src_gray"]) == 3


def test_boxes_round_trip_including_frames_without_a_box(cache_dir):
    """第11計画 Part 3-1: keep_main_subject (postprocess) needs infer_clip's
    per-frame raw YOLOX boxes on a cache hit too."""
    raw = _fake_keyed_raw()
    del raw["keyed"]
    raw["src_gray"] = [np.zeros((8, 8), np.uint8) for _ in range(3)]
    raw["boxes"] = [np.array([1.0, 2.0, 5.0, 7.0]), None, (0, 0, 8, 8)]
    save_raw("clip", PipelineConfig(), raw)
    loaded = load_raw("clip", PipelineConfig())
    assert np.allclose(loaded["boxes"][0], [1, 2, 5, 7])
    assert loaded["boxes"][1] is None
    assert np.allclose(loaded["boxes"][2], [0, 0, 8, 8])


def test_keyed_raw_round_trips_boxes_as_none(cache_dir):
    save_raw("clip", PipelineConfig(), dict(_fake_keyed_raw(), boxes=None))
    assert load_raw("clip", PipelineConfig())["boxes"] is None


def test_keep_main_subject_is_postprocess_only_and_not_in_the_key():
    import dataclasses
    from tool.pipeline.cache import _key
    on = PipelineConfig()
    assert _key("c", on) == _key("c", dataclasses.replace(on, keep_main_subject=not on.keep_main_subject))

import datetime
import json
import queue
import subprocess
import threading
from pathlib import Path

import pytest

from server import errors
from server.jobs import (
    GpuWorker, JobsError, JobStore, EventBus, bootstrap,
    _classify_exception, _job_needs_gpu, _sanitize_detail, _stage_weights, _stage_bases,
    STAGE_WEIGHTS, RETENTION_DAYS, RETENTION_INPUTS_DAYS,
)
from tool.pipeline.config import PipelineConfig
from server.presets import PresetError
from tool.pipeline.runner import JobCancelled


def _touch_upload(tmp_path, name="clip.mp4", content=b"fake mp4 bytes"):
    p = tmp_path / f"upload-{name}"
    p.write_bytes(content)
    return name, p


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def store(tmp_path, monkeypatch, bus):
    # validate_upload would reject our fake, non-decodable mp4 bytes -- these
    # tests are about JobStore/GpuWorker state machine, not input probing
    # (that's test_probe.py's job), so make every upload look valid.
    import server.jobs as jobs_mod
    from server.probe import ProbeInfo
    monkeypatch.setattr(jobs_mod, "validate_upload",
                         lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001))
    # Wired to the SAME `bus` fixture a test's own GpuWorker(store, bus, ...)
    # uses, so JobStore.cancel()'s publish (see its docstring) and a test's
    # own bus.subscribe() calls see the same events.
    return JobStore(tmp_path / "jobs", bus=bus)


def _fake_run_clip_factory(behavior="succeed"):
    """behavior: "succeed" | "fail" | "cancel_after_1" | "oom" """

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress("infer", 1, 1)
        if behavior == "cancel_after_1" and cancel is not None:
            cancel.set()
        if cancel is not None and cancel.is_set():
            raise JobCancelled("cancelled by fake test hook")
        if behavior == "fail":
            raise RuntimeError("synthetic pipeline failure")
        if behavior == "oom":
            raise RuntimeError("Failed to allocate memory for requested buffer of size 999")
        if progress is not None:
            progress("postprocess", 0, 1)
            progress("encode", 1, 1)
            progress("done", 1, 1)
        Path(out_path).write_bytes(b"GIF89afake")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    return fake_run_clip


@pytest.fixture(autouse=True)
def _no_real_models(monkeypatch):
    monkeypatch.setattr("server.jobs._load_models", lambda config: object())


@pytest.fixture(autouse=True)
def _no_gpu_preflight_by_default(monkeypatch):
    """L1's GPU pre-flight check (server/jobs.py's MIN_FREE_GPU_MB,
    default 8000) calls the REAL probe.gpu_status() (real nvidia-smi) when
    a job's device is "cuda" -- PipelineConfig's own default. Without this
    fixture, every pre-existing test here that creates a job with no
    explicit device override and runs it through GpuWorker would silently
    depend on this shared machine's actual free VRAM at test time, and
    could block for real GPU_WAIT_POLL_S-second intervals if it's ever
    reported busy. Tests that specifically exercise the preflight wait
    (test_gpu_preflight_*) override this back to a real threshold
    themselves, after this fixture has already run."""
    monkeypatch.setattr("server.jobs.MIN_FREE_GPU_MB", 0.0)


@pytest.fixture(autouse=True)
def _no_load_aware_encode_by_default(monkeypatch):
    """Same reasoning as _no_gpu_preflight_by_default above, for
    LOAD_AWARE_ENCODE's real os.getloadavg() call -- without this, tests
    would depend on this shared machine's actual CPU load average at test
    time, which can (and did, confirmed by testing) silently swap the
    encoder to webp and break any test assuming a .gif output. Tests that
    specifically exercise the fallback (test_load_aware_encoder_*) turn
    this back on themselves."""
    monkeypatch.setattr("server.jobs.LOAD_AWARE_ENCODE", False)


def test_create_job_with_all_valid_clips_is_queued(store):
    job = store.create([_touch_upload(store.root)], "quick", None)
    assert job["status"] == "queued"
    assert job["clips"][0]["status"] == "pending"
    assert job["mode"] == "quick"
    assert job["qc"] is False


def test_create_job_bad_mode_raises_and_creates_nothing(store, tmp_path):
    # The upload's temp file must live OUTSIDE store.root: it's an input to
    # create(), not something create() itself is expected to produce, and
    # writing it under store.root would contaminate the before/after glob
    # below with an artifact of test setup rather than of create()'s own
    # behaviour.
    upload = _touch_upload(tmp_path)
    before = list(store.root.glob("*"))
    with pytest.raises(PresetError):
        store.create([upload], "ultra-mode", None)
    after = list(store.root.glob("*"))
    assert before == after


def test_create_job_with_invalid_clip_marks_it_failed_but_job_still_queued(store, monkeypatch):
    import server.jobs as jobs_mod
    from server.probe import ProbeError

    def picky(path):
        if "bad" in path.name:
            raise ProbeError(errors.E_INPUT_UNREADABLE, "nope")
        from server.probe import ProbeInfo
        return ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001)

    monkeypatch.setattr(jobs_mod, "validate_upload", picky)
    good = _touch_upload(store.root, "good.mp4")
    bad = _touch_upload(store.root, "bad.mp4")
    job = store.create([good, bad], "quick", None)
    assert job["status"] == "queued"
    statuses = {c["name"]: c["status"] for c in job["clips"]}
    assert statuses["good.mp4"] == "pending"
    assert statuses["bad.mp4"] == "failed"
    assert job["clips"][1]["error"]["code"] == errors.E_INPUT_UNREADABLE


def test_create_job_stores_the_probe_result_on_the_clip(store, monkeypatch):
    """Regression test for the plan's B6 finding: validate_upload()'s
    return value (resolution/frame-count/backdrop-class) was computed and
    then discarded -- nothing in the job record ever saw it."""
    import dataclasses

    import server.jobs as jobs_mod
    from server.probe import ProbeInfo

    fake_info = ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001,
                          bg_is_chroma_class=True, bg_frac_bg_like=0.995)
    monkeypatch.setattr(jobs_mod, "validate_upload", lambda path: fake_info)
    job = store.create([_touch_upload(store.root)], "quick", None)
    assert job["clips"][0]["probe"] == dataclasses.asdict(fake_info)


def test_worker_disables_colour_dependent_stages_for_a_non_chroma_clip(store, bus, monkeypatch):
    """Regression test for the plan's G2 finding: apply_clears/
    strip_bg_fringe/use_trimap all assume a flat chroma backdrop, but
    nothing turned them off for a clip whose probe said the backdrop
    ISN'T flat -- this is what let a natural-background clip run the full
    colour-dependent pipeline with no gate ever catching it."""
    import server.jobs as jobs_mod
    from server.probe import ProbeInfo

    monkeypatch.setattr(jobs_mod, "validate_upload",
                         lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001,
                                                bg_is_chroma_class=False, bg_frac_bg_like=0.4))
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        captured["config"] = config
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"GIF89afake")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    # apply_clears isn't one of ALLOWED_OVERRIDES (server/presets.py) -- only
    # use_trimap/strip_bg_fringe are reachable through the API's override
    # path, so those two are what this test can actually turn on and expect
    # the worker to turn back off.
    job = store.create([_touch_upload(store.root)], "quick",
                        {"strip_bg_fringe": True, "use_trimap": True})
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert captured["config"].strip_bg_fringe is False
    assert captured["config"].use_trimap is False

    updated = store.get(job["id"])
    assert set(updated["clips"][0]["auto_disabled"]) == {"strip_bg_fringe", "use_trimap"}
    assert updated["clips"][0]["status"] == "done"


def test_worker_leaves_config_untouched_for_a_chroma_class_clip(store, bus, monkeypatch):
    """The auto-disable must be a no-op for the common (flat-backdrop)
    case -- confirms the gate doesn't fire when it shouldn't."""
    import server.jobs as jobs_mod
    from server.probe import ProbeInfo

    monkeypatch.setattr(jobs_mod, "validate_upload",
                         lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001,
                                                bg_is_chroma_class=True, bg_frac_bg_like=1.0))
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        captured["config"] = config
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"GIF89afake")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", {"use_trimap": True})
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert captured["config"].use_trimap is True
    updated = store.get(job["id"])
    assert updated["clips"][0]["auto_disabled"] == []


def test_load_aware_encoder_falls_back_under_heavy_cpu_load(store, bus, monkeypatch):
    """Regression test for the plan's incident: supersampled_gif measured
    1080-3605s under heavy contention from OTHER processes on this shared
    machine, hard-failing with E_ENCODE_TIMEOUT once (16x its usual
    ~220s). LOAD_AWARE_ENCODE should swap in a lighter encoder for a clip
    whose (untouched-default) encoder is still supersampled_gif and that
    starts under heavy load.

    supersampled_gif stopped being PipelineConfig's own default on
    2026-09-22 (第9計画/第8回監査レバー4 adopted ss_alpha_gif instead,
    which measured fast even under this session's own heavy load -- see
    tool/pipeline/config.py's docstring), so "quick"/None overrides no
    longer produce a config this fallback needs to touch. This test
    forces exactly the scenario the mechanism exists for -- an untouched
    default (encoder_explicit=False) that happens to be supersampled_gif
    -- directly on the persisted job, independent of which encoder
    currently ships as the preset default."""
    import server.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LOAD_AWARE_ENCODE", True)
    monkeypatch.setattr(jobs_mod, "LOAD_THRESHOLD", 1.5)
    monkeypatch.setattr(jobs_mod.probe, "cpu_load_ratio", lambda: 2.0)  # "heavy load"
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        captured["config"] = config
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"RIFF fake webp")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    job["config"]["encoder"] = "supersampled_gif"
    job["config"]["encoder_explicit"] = False
    store.save(job)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert captured["config"].encoder == "webp"
    updated = store.get(job["id"])
    clip = updated["clips"][0]
    assert clip["auto_encoder_fallback"] == "webp"
    assert clip["outputs"]["primary"].endswith(".webp"), "output filename must match the actually-used encoder"
    assert clip["status"] == "done"


def test_load_aware_encoder_does_not_fall_back_under_light_load(store, bus, monkeypatch):
    import server.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LOAD_AWARE_ENCODE", True)
    monkeypatch.setattr(jobs_mod, "LOAD_THRESHOLD", 1.5)
    monkeypatch.setattr(jobs_mod.probe, "cpu_load_ratio", lambda: 0.1)  # "light load"
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        captured["config"] = config
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"GIF89afake")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    # "quick"'s untouched default is ss_alpha_gif since 2026-09-22 (see the
    # sibling heavy-load test's docstring) -- LOAD_AWARE_ENCODE only ever
    # substitutes for supersampled_gif specifically, so a config that was
    # never supersampled_gif to begin with is untouched under any load.
    assert captured["config"].encoder == "ss_alpha_gif"
    updated = store.get(job["id"])
    assert updated["clips"][0]["auto_encoder_fallback"] is None


def test_load_aware_encoder_never_overrides_an_explicit_choice(store, bus, monkeypatch):
    """The fallback must only ever substitute for the DEFAULT encoder --
    a caller who explicitly asked for a specific encoder (even the heavy
    one, by name) gets exactly that regardless of load."""
    import server.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LOAD_AWARE_ENCODE", True)
    monkeypatch.setattr(jobs_mod, "LOAD_THRESHOLD", 1.5)
    monkeypatch.setattr(jobs_mod.probe, "cpu_load_ratio", lambda: 5.0)  # extreme load
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        captured["config"] = config
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"II* fake mov")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", {"encoder": "mov"})
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert captured["config"].encoder == "mov"
    updated = store.get(job["id"])
    assert updated["clips"][0]["auto_encoder_fallback"] is None


def test_load_aware_encoder_does_not_override_an_explicit_supersampled_gif(store, bus, monkeypatch):
    """Regression test for audit M5: the previous version of this gate
    compared the encoder's current VALUE against "supersampled_gif", which
    cannot tell "still at the untouched default" apart from "the caller
    explicitly asked for supersampled_gif by name" -- both look identical
    as a plain string. The sibling test above (encoder="mov") could never
    have caught that: "mov" never equals "supersampled_gif" regardless of
    provenance, so it passed even when the provenance check was missing
    entirely. This uses the one encoder value where the distinction
    actually matters."""
    import server.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LOAD_AWARE_ENCODE", True)
    monkeypatch.setattr(jobs_mod, "LOAD_THRESHOLD", 1.5)
    monkeypatch.setattr(jobs_mod.probe, "cpu_load_ratio", lambda: 5.0)  # extreme load
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        captured["config"] = config
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"GIF89a fake")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", {"encoder": "supersampled_gif"})
    assert job["config"]["encoder_explicit"] is True  # sanity: the override was recorded as explicit
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert captured["config"].encoder == "supersampled_gif"
    updated = store.get(job["id"])
    assert updated["clips"][0]["auto_encoder_fallback"] is None


def test_load_aware_encoder_still_substitutes_for_the_untouched_default(store, bus, monkeypatch):
    """The positive case for the same provenance flag: a config with
    encoder="supersampled_gif" but encoder_explicit=False (i.e. it got
    there without a caller naming it -- the shape a future default
    reversion, or any other non-preset construction path, could produce)
    must still fall back under load exactly as before. "quick"'s own
    untouched default stopped being supersampled_gif on 2026-09-22 (see
    the heavy-load sibling test's docstring), so this forces the shape
    directly on the persisted job rather than relying on the preset."""
    import server.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LOAD_AWARE_ENCODE", True)
    monkeypatch.setattr(jobs_mod, "LOAD_THRESHOLD", 1.5)
    monkeypatch.setattr(jobs_mod.probe, "cpu_load_ratio", lambda: 5.0)
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        captured["config"] = config
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"RIFF fake webp")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    job["config"]["encoder"] = "supersampled_gif"
    job["config"]["encoder_explicit"] = False
    store.save(job)
    assert job["config"]["encoder_explicit"] is False
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert captured["config"].encoder == jobs_mod.LOAD_FALLBACK_ENCODER
    updated = store.get(job["id"])
    assert updated["clips"][0]["auto_encoder_fallback"] == jobs_mod.LOAD_FALLBACK_ENCODER


def test_auto_keyer_is_recorded_even_when_the_clip_fails_after_keying(store, bus, monkeypatch):
    """Regression test for audit M6: auto_keyer used to be set only on the
    success path, right after run_clip returned -- so a clip that keyed
    fine and then died LATER (e.g. an encode timeout) kept auto_keyer=None,
    the schema's documented meaning for "not yet processed", even though it
    genuinely had been processed by the fast path. Moved into the per-clip
    `finally` so it reflects "key_s is in timings", independent of whether
    the clip ultimately succeeded."""
    import server.jobs as jobs_mod

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                       progress=None, cancel=None, timings=None, **_kw):
        if timings is not None:
            timings["key_s"] = 6.5  # keying itself succeeded...
        raise RuntimeError("ffmpeg timeout (600s) during frame write")  # ...but encode then failed

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["clips"][0]["status"] == "failed"
    assert updated["clips"][0]["auto_keyer"] is True


def test_a_mispredicted_keyer_clip_still_uses_the_cached_model_factory(store, bus, monkeypatch):
    """Regression test for audit M3: when the job-level prediction thinks a
    clip will take the colour-only keyer path (so the VRAM pre-flight wait
    is skipped) but the real per-frame check in infer_clip declines it, the
    `models` argument run_clip receives must be a FACTORY that goes through
    GpuWorker._get_models' cache -- not a bare `None`, which made infer_clip
    load its own throwaway Models instance every time this happened,
    bypassing the cache entirely (a multi-clip batch that keeps
    mispredicting would reload the network from scratch for every clip)."""
    import server.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "_clip_takes_keyer", lambda clip, config: True)
    captured = {}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        # Simulate infer_clip's own real behaviour when it needs the network:
        # models is a zero-arg callable, and calling it is what's supposed to
        # go through the worker's cache.
        assert callable(models), "expected a lazy factory, not a bare instance/None"
        captured["resolved"] = models()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"GIF89a fake")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", {"device": "cuda"})
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert "resolved" in captured
    # _get_models populates _models_cache["cuda"] and returns that same
    # object on a second call -- the cache-bypass bug this guards against
    # would instead hand back a fresh, uncached object each time.
    assert worker._models_cache.get("cuda") is captured["resolved"]


def test_create_job_all_clips_invalid_marks_job_failed_immediately(store, monkeypatch):
    import server.jobs as jobs_mod
    from server.probe import ProbeError
    monkeypatch.setattr(jobs_mod, "validate_upload",
                         lambda path: (_ for _ in ()).throw(ProbeError(errors.E_INPUT_EXT, "nope")))
    job = store.create([_touch_upload(store.root)], "quick", None)
    assert job["status"] == "failed"
    assert job["finished_at"] is not None


def test_gpu_worker_processes_job_to_done(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["status"] == "done"
    assert updated["clips"][0]["status"] == "done"
    assert updated["clips"][0]["outputs"]["primary"]
    assert updated["finished_at"] is not None


def test_gpu_worker_classifies_oom(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("oom"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["clips"][0]["status"] == "failed"
    assert updated["clips"][0]["error"]["code"] == errors.E_CUDA_OOM
    # one clip failing still lets the job finish (not "cancelled"/stuck);
    # with the only clip failed and none succeeding, the job itself failed.
    assert updated["status"] == "failed"
    # B8 regression: a failed job must carry job["error"] too, not just its
    # clip's error -- previously job["error"] stayed None on this path, so
    # the UI's `if (job.error)` banner never fired.
    assert updated["error"]["code"] == errors.E_CUDA_OOM
    assert "1 of 1" in updated["error"]["detail"]


def test_gpu_worker_partial_success_job_is_done_not_failed(store, bus, monkeypatch):
    calls = {"n": 0}

    def flaky(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"ok")
        return out_path

    monkeypatch.setattr("server.jobs.run_clip", flaky)
    job = store.create([_touch_upload(store.root, "a.mp4"), _touch_upload(store.root, "b.mp4")],
                        "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["status"] == "done"
    statuses = [c["status"] for c in updated["clips"]]
    assert statuses == ["failed", "done"]


def test_cancel_queued_job_is_immediate(store, bus):
    job = store.create([_touch_upload(store.root)], "quick", None)
    result = store.cancel(job["id"])
    assert result["status"] == "cancelled"
    assert result["clips"][0]["status"] == "cancelled"
    # the worker must be a no-op on a job it later dequeues that's already cancelled
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])
    assert store.get(job["id"])["status"] == "cancelled"


def test_cancel_mid_run_raises_job_cancelled_and_is_recorded(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("cancel_after_1"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["clips"][0]["status"] == "cancelled"
    assert updated["status"] == "cancelled"


def test_events_are_published_during_processing(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    types = [e["type"] for e in events]
    assert "progress" in types
    assert "clip_status" in types
    assert types[-1] == "job_status"
    assert events[-1]["status"] == "done"


def test_stage_weights_renormalize_without_qc():
    with_qc = _stage_weights(True)
    assert with_qc == STAGE_WEIGHTS

    without_qc = _stage_weights(False)
    assert "qc" not in without_qc
    assert sum(without_qc.values()) == pytest.approx(1.0)

    bases = _stage_bases(without_qc)
    assert bases["prepare"] == 0.0
    assert bases["infer"] == pytest.approx(without_qc["prepare"])
    assert bases["encode"] + without_qc["encode"] == pytest.approx(1.0)


def test_progress_fraction_is_monotonic_reaches_one_and_is_throttled(store, bus, monkeypatch):
    """Regression test for the plan's FE5/L4 findings. FE5: the front end
    used to compute overall progress from the CURRENT clip's raw
    frames_done/frames_total, which went backwards on every stage change
    (e.g. "infer" 100/100 -> "postprocess" 0/1) -- fraction is now computed
    server-side from STAGE_WEIGHTS so it can only go up. It must also never
    surface runner.py's own "done" stage marker (that used to reset the
    front end's stepper to all-pending). L4: publishing every one of a
    clip's ~150 raw progress() calls to the bus (fanned out to every
    connected client) was needless churn at browser refresh rates; most
    should be swallowed by the 5-frame/200ms throttle."""
    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress("prepare", 0, 1)  # runner.infer_clip's own first call
            progress("prepare", 1, 1)
            for i in range(1, 101):
                progress("infer", i, 100)
            progress("postprocess", 0, 1)
            for i in range(1, 51):
                progress("encode", i, 50)
            progress("done", 1, 1)  # runner's own terminal marker -- must never reach the bus
        Path(out_path).write_bytes(b"ok")
        return out_path

    monkeypatch.setattr("server.jobs.run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)  # qc=False
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    progress_events = [e for e in events if e["type"] == "progress"]
    assert progress_events, "expected at least one progress event"
    assert all(e["stage"] != "done" for e in progress_events)
    # 第11計画 Part 1-3: a stage exists from the very start (published by
    # _process_job before the model load), so the heartbeat can run from t=0.
    assert progress_events[0]["stage"] == "prepare"
    assert progress_events[0]["fraction"] == pytest.approx(0.0)

    fractions = [e["fraction"] for e in progress_events]
    for a, b in zip(fractions, fractions[1:]):
        assert b >= a - 1e-9, f"fraction went backwards: {a} -> {b}"
    assert fractions[-1] == pytest.approx(1.0, abs=1e-6)

    # 154 raw progress() calls were made (incl. _process_job's own "prepare"); throttling must cut that down a lot.
    assert len(progress_events) <= 50, f"expected throttling, got {len(progress_events)} events"


def test_preview_file_is_written_and_published_then_cleaned_up_on_success(store, bus, monkeypatch):
    """Regression test for the live-preview feature: run_clip's `preview`
    callback must be wired all the way through to a real PNG file on
    disk, a "preview" SSE event, and clip["preview"] pointing at it --
    then, once the clip is done, the file must be removed and
    clip["preview"] cleared, since it's only meaningful while in flight."""
    import numpy as np

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                       progress=None, cancel=None, preview=None, **_kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if preview is not None:
            frame = np.zeros((10, 10, 4), dtype=np.uint8)
            for i in range(1, 4):
                preview(i, frame)
        Path(out_path).write_bytes(b"ok")
        return out_path

    monkeypatch.setattr("server.jobs.PREVIEW_MIN_INTERVAL_S", 0.0)  # publish every call, not just every 0.5s
    monkeypatch.setattr("server.jobs.run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())

    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["status"] == "done"
    assert updated["clips"][0]["preview"] is None  # cleared once the clip finished
    job_dir = store.job_dir(job["id"])
    assert not (job_dir / "outputs" / "clip_preview.png").exists()

    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    preview_events = [e for e in events if e["type"] == "preview"]
    assert preview_events, "expected at least one preview event"
    assert all(e["clip_id"] == "00" for e in preview_events)
    assert [e["seq"] for e in preview_events] == [1, 2, 3]


def test_preview_write_failure_does_not_leave_a_tmp_file_behind(store, bus, monkeypatch):
    """Regression test for audit A2: if the atomic tmp->final swap fails
    AFTER the .png.tmp file was actually created on disk (write_bytes
    succeeded, replace() didn't), that .tmp must not outlive the clip --
    it used to be unlinked nowhere (only the final .png path was cleaned
    up in the per-clip finally)."""
    import pathlib

    import numpy as np

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                       progress=None, cancel=None, preview=None, **_kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if preview is not None:
            preview(1, np.zeros((10, 10, 4), dtype=np.uint8))
        Path(out_path).write_bytes(b"ok")
        return out_path

    real_replace = pathlib.Path.replace

    def failing_replace(self, target):
        if str(self).endswith(".png.tmp"):
            raise OSError("synthetic replace failure -- tmp file already exists on disk")
        return real_replace(self, target)

    monkeypatch.setattr("server.jobs.PREVIEW_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr("server.jobs.run_clip", fake_run_clip)
    monkeypatch.setattr(pathlib.Path, "replace", failing_replace)
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())

    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["status"] == "done"
    assert updated["clips"][0]["preview"] is None
    job_dir = store.job_dir(job["id"])
    assert not (job_dir / "outputs" / "clip_preview.png").exists()
    assert not (job_dir / "outputs" / "clip_preview.png.tmp").exists()


def test_preview_file_is_cleaned_up_on_failure_too(store, bus, monkeypatch):
    import numpy as np

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                       progress=None, cancel=None, preview=None, **_kw):
        if preview is not None:
            preview(1, np.zeros((10, 10, 4), dtype=np.uint8))
        raise RuntimeError("synthetic failure after emitting one preview frame")

    monkeypatch.setattr("server.jobs.PREVIEW_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr("server.jobs.run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["clips"][0]["status"] == "failed"
    assert updated["clips"][0]["preview"] is None
    job_dir = store.job_dir(job["id"])
    assert not (job_dir / "outputs" / "clip_preview.png").exists()


def test_preview_is_throttled_independently_of_progress(store, bus, monkeypatch):
    """PREVIEW_MIN_INTERVAL_S (2fps) must gate preview publishing even
    when it's left at its real default -- encoding a PNG on every one of
    a clip's ~100+ frames would be needless CPU work for a live glance."""
    import numpy as np

    calls = {"n": 0}

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                       progress=None, cancel=None, preview=None, **_kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if preview is not None:
            frame = np.zeros((10, 10, 4), dtype=np.uint8)
            for i in range(1, 21):
                calls["n"] += 1
                preview(i, frame)  # all 20 calls happen essentially instantly
        Path(out_path).write_bytes(b"ok")
        return out_path

    monkeypatch.setattr("server.jobs.run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    preview_events = [e for e in events if e["type"] == "preview"]
    assert calls["n"] == 20
    # 20 calls within well under one second must NOT all publish at the
    # real 0.5s throttle -- expect 1 (the first call always publishes).
    assert len(preview_events) <= 2, f"expected throttling, got {len(preview_events)} events"


def test_restart_recovery_marks_running_job_failed(tmp_path, monkeypatch):
    from server.probe import ProbeInfo
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "validate_upload", lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001))

    root = tmp_path / "jobs"
    store1 = JobStore(root)
    job = store1.create([_touch_upload(root)], "quick", None)
    job["status"] = "running"
    job["clips"][0]["status"] = "running"
    store1.save(job)

    store2 = JobStore(root)  # simulates a fresh process starting up
    reloaded = store2.get(job["id"])
    assert reloaded["status"] == "failed"
    assert reloaded["error"]["code"] == errors.E_SERVER_RESTART
    assert reloaded["clips"][0]["status"] == "failed"


def test_delete_refuses_running_job(store):
    job = store.create([_touch_upload(store.root)], "quick", None)
    job["status"] = "running"
    store.save(job)
    with pytest.raises(JobsError) as exc:
        store.delete(job["id"])
    assert exc.value.code == errors.E_JOB_RUNNING


def test_delete_removes_job_dir(store):
    job = store.create([_touch_upload(store.root)], "quick", None)
    assert store.job_dir(job["id"]).is_dir()
    store.delete(job["id"])
    assert store.get(job["id"]) is None
    assert not store.job_dir(job["id"]).is_dir()


def test_zip_path_contains_clip_outputs(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    import zipfile
    zpath = store.zip_path(job["id"])
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert any(n.endswith(".gif") for n in names)


def test_clip_ids_are_stable_index_based_and_distinct_from_name(store):
    job = store.create([_touch_upload(store.root, "a.mp4"), _touch_upload(store.root, "b.mp4")],
                        "quick", None)
    assert [c["id"] for c in job["clips"]] == ["00", "01"]
    assert [c["name"] for c in job["clips"]] == ["a.mp4", "b.mp4"]


def test_duplicate_clip_names_get_disambiguated_on_disk_and_by_name(store):
    """Regression test for the plan's FE2 finding: two uploads with the
    SAME original filename used to collide -- the second's shutil.move
    silently overwrote the first's input file (and later its output would
    overwrite the first's output too), both under a name the front end
    matched clips by, so the second clip sat "pending" forever. Names must
    now be distinct on disk; clip_id (tested above) is what the front end
    actually matches on."""
    # Two distinct temp files (as server/app.py's _save_uploads always
    # produces, one uuid-prefixed dest per part) sharing the SAME original
    # filename -- _touch_upload can't model this directly since it derives
    # its own tmp path from `name`.
    p1 = store.root / "tmp-upload-1"
    p1.write_bytes(b"first")
    p2 = store.root / "tmp-upload-2"
    p2.write_bytes(b"second")
    job = store.create([("a.mp4", p1), ("a.mp4", p2)], "quick", None)
    names = [c["name"] for c in job["clips"]]
    assert names == ["a.mp4", "a (2).mp4"]
    assert (store.job_dir(job["id"]) / "inputs" / "a.mp4").read_bytes() == b"first"
    assert (store.job_dir(job["id"]) / "inputs" / "a (2).mp4").read_bytes() == b"second"


def test_zip_path_is_rebuilt_while_running_but_reused_once_terminal(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    first = store.zip_path(job["id"])
    mtime1 = first.stat().st_mtime_ns
    second = store.zip_path(job["id"])
    assert second == first
    assert second.stat().st_mtime_ns == mtime1  # reused, not rebuilt, for a terminal job


def test_zip_path_concurrent_downloads_produce_a_valid_zip(store, bus, monkeypatch):
    """Regression test for the plan's B9 finding: zip_path() used to
    reopen the SAME path with mode "w" (truncating it) on every call with
    no locking, so two threads racing to download a finished job's zip
    could observe a torn/corrupt file mid-write."""
    import zipfile
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    results = []
    errors_seen = []

    def download():
        try:
            results.append(store.zip_path(job["id"]))
        except Exception as e:  # pragma: no cover -- the bug this guards against
            errors_seen.append(e)

    threads = [threading.Thread(target=download) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert errors_seen == []
    for zpath in results:
        with zipfile.ZipFile(zpath) as zf:
            assert zf.testzip() is None


def test_retry_reuses_inputs_and_mode(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    original = store.create([_touch_upload(store.root, "x.mp4")], "quick", {"device": "cpu"})
    retried = store.retry(original["id"])
    assert retried["id"] != original["id"]
    assert retried["mode"] == "quick"
    assert retried["overrides"] == {"device": "cpu"}
    assert retried["clips"][0]["name"] == "x.mp4"
    assert (store.job_dir(retried["id"]) / "inputs" / "x.mp4").is_file()


def test_retry_with_different_mode(store):
    original = store.create([_touch_upload(store.root)], "quick", None)
    retried = store.retry(original["id"], mode="thorough", overrides={})
    assert retried["mode"] == "thorough"
    assert retried["qc"] is True


def test_create_marks_rejected_uploads_as_failed_clips_with_no_input(store):
    """Regression test for the plan's B5 finding: an upload already refused
    by server/app.py's _save_uploads (streamed past probe.MAX_FILE_MB)
    must become a failed clip with input=None -- no bytes of it should
    ever have to touch this job's directory."""
    job = store.create([_touch_upload(store.root, "good.mp4")], "quick", None,
                        rejected=[("huge.mp4", "huge.mp4: exceeds the 500 MB limit")])
    statuses = {c["name"]: c for c in job["clips"]}
    assert statuses["good.mp4"]["status"] == "pending"
    assert statuses["huge.mp4"]["status"] == "failed"
    assert statuses["huge.mp4"]["input"] is None
    assert statuses["huge.mp4"]["error"]["code"] == errors.E_UPLOAD_TOO_LARGE
    assert not (store.job_dir(job["id"]) / "inputs" / "huge.mp4").exists()


def test_create_invalid_clip_does_not_leave_the_rejected_file_on_disk(store, monkeypatch):
    """Regression test for the other half of B5: a clip that fails
    validate_upload AFTER being moved into inputs/ (as opposed to a
    _save_uploads-level rejection, tested above) used to be kept on disk
    forever -- create() must unlink it and clear clip["input"]."""
    import server.jobs as jobs_mod
    from server.probe import ProbeError
    monkeypatch.setattr(jobs_mod, "validate_upload",
                         lambda path: (_ for _ in ()).throw(ProbeError(errors.E_INPUT_TOO_LARGE, "nope")))
    job = store.create([_touch_upload(store.root, "bad.mp4")], "quick", None)
    clip = job["clips"][0]
    assert clip["status"] == "failed"
    assert clip["input"] is None
    assert not (store.job_dir(job["id"]) / "inputs" / "bad.mp4").exists()


def test_retry_bad_mode_leaves_no_orphaned_retry_files(store):
    """Regression test for the plan's B10 finding: retry() used to
    hardlink every input BEFORE validating mode/overrides, so a
    PresetError left `.retry-*` files under store.root with nothing to
    ever clean them up."""
    original = store.create([_touch_upload(store.root)], "quick", None)
    with pytest.raises(PresetError):
        store.retry(original["id"], mode="not-a-real-mode")
    assert list(store.root.glob(".retry-*")) == []


def test_retry_raises_inputs_expired_when_no_input_files_remain(store):
    """Regression test for the plan's B10 finding: retrying a job whose
    inputs/ was already cleaned up (RETENTION_INPUTS_DAYS) used to
    silently create an empty job carrying the wrong error code
    (E_INPUT_UNREADABLE); it must instead refuse the retry itself."""
    original = store.create([_touch_upload(store.root)], "quick", None)
    import shutil as shutil_mod
    shutil_mod.rmtree(store.job_dir(original["id"]) / "inputs")
    with pytest.raises(JobsError) as exc_info:
        store.retry(original["id"])
    assert exc_info.value.code == errors.E_INPUTS_EXPIRED
    assert list(store.root.glob(".retry-*")) == []


def test_get_returns_a_deep_copy_not_a_live_reference(store):
    job = store.create([_touch_upload(store.root)], "quick", None)
    fetched = store.get(job["id"])
    fetched["status"] = "corrupted-by-caller"
    fetched["clips"][0]["name"] = "corrupted-by-caller"
    refetched = store.get(job["id"])
    assert refetched["status"] == "queued"
    assert refetched["clips"][0]["name"] != "corrupted-by-caller"


def test_list_returns_deep_copies_too(store):
    store.create([_touch_upload(store.root)], "quick", None)
    jobs = store.list()
    jobs[0]["clips"][0]["name"] = "corrupted-by-caller"
    assert store.list()[0]["clips"][0]["name"] != "corrupted-by-caller"


def test_list_summary_strips_config_and_per_clip_details(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    full = store.list()[0]
    assert "config" in full
    assert "outputs" in full["clips"][0]

    summary = store.list(summary=True)[0]
    assert "config" not in summary
    assert "metrics" not in summary["clips"][0]
    assert "timings" not in summary["clips"][0]
    assert "outputs" not in summary["clips"][0]
    # identity fields a history list still needs must survive summarising
    assert summary["clips"][0]["name"] == full["clips"][0]["name"]
    assert summary["clips"][0]["status"] == full["clips"][0]["status"]


def test_get_and_save_do_not_race_under_concurrent_mutation(store, bus, monkeypatch):
    """Regression test for the plan's B6 finding: get()/list() used to
    return a live reference into JobStore._jobs, which GpuWorker mutates
    in place (adding keys to clip["outputs"]/["timings"] etc.) -- a reader
    iterating/json.dumps-ing that same object while the worker mutated it
    could raise "dictionary changed size during iteration" or see torn
    data. Simulates the worker's mutate-then-save pattern from a second
    thread while this thread hammers get()/list() + json.dumps."""
    import json as json_mod

    job = store.create([_touch_upload(store.root)], "quick", None)
    job_id = job["id"]
    stop = threading.Event()
    errors_seen = []

    def mutator():
        n = 0
        while not stop.is_set():
            current = store.get(job_id)
            # Mirror GpuWorker._process_job's own pattern: mutate a nested
            # clip dict in place, then persist via save() (never touching
            # store._jobs directly in between).
            current["clips"][0]["timings"][f"stage_{n}"] = n
            current["clips"][0]["outputs"][f"file_{n}"] = f"outputs/f{n}.gif"
            store.save(current)
            n += 1

    def reader():
        while not stop.is_set():
            try:
                json_mod.dumps(store.get(job_id))
                json_mod.dumps(store.list())
            except Exception as e:  # pragma: no cover -- the bug this guards against
                errors_seen.append(e)

    threads = [threading.Thread(target=mutator), threading.Thread(target=reader),
               threading.Thread(target=reader)]
    for t in threads:
        t.start()
    threading.Event().wait(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert errors_seen == []


def test_cancel_queued_job_publishes_events_on_the_bus(store, bus):
    """Regression test for the plan's FE1 finding: cancelling a job that's
    still "queued" (the common case whenever the GPU worker is busy with
    someone else's job) used to flip its status on disk with NO event
    published anywhere -- a browser watching SSE would sit on "running"
    forever since GpuWorker._process_job's early-return for an
    already-non-queued job was silent too. cancel() must publish it
    itself, and a subscriber attached before the cancel() call must
    receive it."""
    job = store.create([_touch_upload(store.root)], "quick", None)
    sub = bus.subscribe(job["id"])

    store.cancel(job["id"])

    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    types_and_statuses = [(e["type"], e.get("status")) for e in events]
    assert ("clip_status", "cancelled") in types_and_statuses
    assert ("job_status", "cancelled") in types_and_statuses


def test_process_job_publishes_current_status_when_already_handled(store, bus):
    """The early-return branch in GpuWorker._process_job (job no longer
    "queued" by the time this thread's claim_queued() runs) must still
    publish the job's current status -- covers the race window between
    another thread's state change and this thread's own lock acquisition,
    distinct from cancel()'s own publish (test above) which only fires
    for the "found it still queued" case."""
    job = store.create([_touch_upload(store.root)], "quick", None)
    store.cancel(job["id"])  # already publishes once; drain that first
    sub = bus.subscribe(job["id"])

    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])  # claim_queued() returns None: already "cancelled"

    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    assert any(e["type"] == "job_status" and e["status"] == "cancelled" for e in events)


@pytest.mark.parametrize("exc, expected_code", [
    (JobCancelled("cancelled by test"), errors.E_CANCELLED),
    (subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600), errors.E_ENCODE_TIMEOUT),
    (RuntimeError("ffmpeg timeout (600s) during frame write. cmd=ffmpeg ... -> /tmp/x/out.gif"),
     errors.E_ENCODE_TIMEOUT),
    (RuntimeError("Failed to allocate memory for requested buffer of size 999"), errors.E_CUDA_OOM),
    (RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: out of memory"), errors.E_CUDA_OOM),
    (RuntimeError("ffmpeg failed (exit=1). cmd=ffmpeg ...\nstderr:\nCannot allocate memory"),
     errors.E_PIPELINE),
    (RuntimeError("ffmpeg failed (exit=1). cmd=ffmpeg ...\nstderr:\nout of memory"), errors.E_PIPELINE),
    (RuntimeError("something else entirely broke"), errors.E_PIPELINE),
])
def test_classify_exception(exc, expected_code):
    """Regression test for the plan's B7 finding: subprocess.TimeoutExpired
    (raised directly by ffmpeg_encoders.pipe_rgba_to_ffmpeg's own final
    proc.wait(timeout=...), as opposed to the explicit "ffmpeg timeout"
    RuntimeError raised earlier in its write loop) used to fall through to
    the generic E_PIPELINE bucket despite genuinely being a timeout. A bare
    "out of memory" with no cuda/cudnn/cublas/onnxruntime/gpu marker (as
    ffmpeg's own stderr can say for an ordinary host-memory exhaustion) must
    NOT be classified as E_CUDA_OOM -- only "failed to allocate" (the
    onnxruntime arena wording) or "out of memory" WITH one of those markers
    should be."""
    code, _detail = _classify_exception(exc)
    assert code == expected_code


def test_classify_exception_clip_too_long():
    from tool.pipeline.runner import ClipTooLong
    code, detail = _classify_exception(ClipTooLong("clip exceeds max_frames=1800"))
    assert code == errors.E_INPUT_TOO_LONG
    assert "1800" in detail


def test_sanitize_detail_strips_absolute_paths_and_caps_length():
    msg = ("ffmpeg failed (exit=1). cmd=ffmpeg ... -> "
           "/home/someuser/some/project/data/jobs/xyz/outputs/foo.gif")
    cleaned = _sanitize_detail(msg)
    assert "/home/" not in cleaned
    assert cleaned.endswith("foo.gif")

    long_msg = "x" * 1000
    assert len(_sanitize_detail(long_msg)) <= 501  # 500 chars + the ellipsis marker


def test_failed_job_error_falls_back_to_pipeline_when_clip_codes_disagree(store, bus, monkeypatch):
    """Regression test for the plan's B8 finding, mixed-cause case: if the
    job's failed clips don't all agree on one error code, job["error"]
    must fall back to the generic E_PIPELINE code rather than arbitrarily
    picking one clip's specific code as if it applied to the whole job."""
    calls = {"n": 0}

    def mixed_failures(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Failed to allocate memory for requested buffer of size 999")
        raise RuntimeError("ffmpeg timeout (600s) during frame write")

    monkeypatch.setattr("server.jobs.run_clip", mixed_failures)
    job = store.create([_touch_upload(store.root, "a.mp4"), _touch_upload(store.root, "b.mp4")],
                        "quick", None)
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["status"] == "failed"
    clip_codes = {c["error"]["code"] for c in updated["clips"]}
    assert clip_codes == {errors.E_CUDA_OOM, errors.E_ENCODE_TIMEOUT}
    assert updated["error"]["code"] == errors.E_PIPELINE
    assert "2 of 2" in updated["error"]["detail"]


def test_worker_periodic_cleanup_deletes_old_jobs_and_prunes_inputs(store, bus, monkeypatch):
    """Regression test for the plan's B4 finding: JobStore.cleanup() had no
    caller anywhere in the running server -- RETENTION_DAYS/
    RETENTION_INPUTS_DAYS were dead constants and data/jobs/ grew without
    bound. GpuWorker._maybe_cleanup() (wired into run()'s idle branch) is
    what's supposed to call it periodically."""
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    old_job = store.create([_touch_upload(store.root, "old.mp4")], "quick", None)
    recent_job = store.create([_touch_upload(store.root, "recent.mp4")], "quick", None)
    worker0 = GpuWorker(store, bus, queue.Queue())
    worker0._process_job(old_job["id"])
    worker0._process_job(recent_job["id"])

    old = store.get(old_job["id"])
    old["finished_at"] = (datetime.datetime.now(datetime.timezone.utc)
                           - datetime.timedelta(days=RETENTION_DAYS + 1)).isoformat()
    store.save(old)
    mid = store.get(recent_job["id"])
    mid["finished_at"] = (datetime.datetime.now(datetime.timezone.utc)
                           - datetime.timedelta(days=RETENTION_INPUTS_DAYS + 1)).isoformat()
    store.save(mid)

    worker = GpuWorker(store, bus, queue.Queue(), cleanup_interval_s=0)
    worker._maybe_cleanup()

    assert store.get(old_job["id"]) is None
    assert not store.job_dir(old_job["id"]).exists()
    assert store.get(recent_job["id"]) is not None
    assert not (store.job_dir(recent_job["id"]) / "inputs").exists()


def test_bootstrap_runs_a_cleanup_pass_at_startup(tmp_path, monkeypatch):
    from server.probe import ProbeInfo
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "validate_upload", lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001))

    root = tmp_path / "jobs"
    pre_store = JobStore(root)
    job = pre_store.create([_touch_upload(root)], "quick", None)
    job["status"] = "done"
    job["finished_at"] = (datetime.datetime.now(datetime.timezone.utc)
                           - datetime.timedelta(days=jobs_mod.RETENTION_DAYS + 1)).isoformat()
    pre_store.save(job)

    store, bus, worker = bootstrap(root)
    try:
        assert store.get(job["id"]) is None
    finally:
        worker.stop()


def test_gpu_preflight_waits_then_runs_once_the_gpu_frees_up(store, bus, monkeypatch):
    """Regression test for the plan's L1 finding: a cuda job used to be
    claimed and run immediately regardless of how little VRAM the shared
    GPU actually had free, discovering the OOM only after real (possibly
    lengthy) work. A job whose device is cuda must sit in "waiting_gpu"
    -- polling probe.gpu_status() -- until enough is free, then proceed
    to "running" exactly as before."""
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "GPU_WAIT_POLL_S", 0.02)  # real default is 15s
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB", 8000.0)
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))

    calls = {"n": 0}

    def fake_gpu_status():
        calls["n"] += 1
        if calls["n"] < 3:
            return {"used_mb": 9500, "total_mb": 10000, "name": "Fake GPU"}  # 500 MB free
        return {"used_mb": 1000, "total_mb": 10000, "name": "Fake GPU"}  # 9000 MB free

    monkeypatch.setattr("server.probe.gpu_status", fake_gpu_status)

    job = store.create([_touch_upload(store.root)], "quick", {"device": "cuda"})
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    assert store.get(job["id"])["status"] == "done"
    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    statuses = [e["status"] for e in events if e["type"] == "job_status"]
    assert statuses == ["waiting_gpu", "running", "done"]
    assert calls["n"] >= 3


def test_gpu_preflight_device_cpu_never_waits(store, bus, monkeypatch):
    """A cpu job must skip the GPU pre-flight check entirely, even if the
    GPU is reported as completely full -- CPU mode doesn't touch it."""
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB", 8000.0)
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    monkeypatch.setattr("server.probe.gpu_status",
                         lambda: {"used_mb": 10000, "total_mb": 10000, "name": "Fake GPU"})  # 0 free

    job = store.create([_touch_upload(store.root)], "quick", {"device": "cpu"})
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    events = [e for e in _drain(sub) if e["type"] == "job_status"]
    assert [e["status"] for e in events] == ["running", "done"]


def test_gpu_preflight_disabled_when_threshold_is_zero(store, bus, monkeypatch):
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB", 0.0)
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    monkeypatch.setattr("server.probe.gpu_status",
                         lambda: (_ for _ in ()).throw(AssertionError("gpu_status should not be called")))

    job = store.create([_touch_upload(store.root)], "quick", {"device": "cuda"})
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])  # must not raise
    assert store.get(job["id"])["status"] == "done"


def test_gpu_preflight_cancel_while_waiting_reaches_cancelled(store, bus, monkeypatch):
    """Regression test for the plan's L1 finding: cancelling a job that's
    stuck waiting for GPU headroom must behave like cancelling any other
    still-queued job (FE1) -- reach "cancelled" and publish it, not hang
    forever waiting for VRAM that may never free up."""
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "GPU_WAIT_POLL_S", 0.02)
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB", 8000.0)
    monkeypatch.setattr("server.probe.gpu_status",
                         lambda: {"used_mb": 9999, "total_mb": 10000, "name": "Fake GPU"})  # never frees

    job = store.create([_touch_upload(store.root)], "quick", {"device": "cuda"})
    worker = GpuWorker(store, bus, queue.Queue())
    t = threading.Thread(target=worker._process_job, args=(job["id"],))
    t.start()
    try:
        for _ in range(200):
            if store.get(job["id"])["status"] == "waiting_gpu":
                break
            threading.Event().wait(0.02)
        else:
            raise AssertionError("job never reached waiting_gpu")
        store.cancel(job["id"])
        t.join(timeout=5)
        assert not t.is_alive()
        assert store.get(job["id"])["status"] == "cancelled"
    finally:
        t.join(timeout=5)


def test_gpu_preflight_uses_warm_threshold_when_models_already_cached(store, bus, monkeypatch):
    """Regression test: gpu_status() reports TOTAL VRAM used regardless of
    who holds it, so a worker that had already loaded its own models for
    this device (observed ~7-8GB resident) mistook that as "no room" and
    waited on a GPU that was, from this job's point of view, already
    ready -- confirmed to cause a real ~10 minute wait in production.
    Once `device`'s models are already in _models_cache, the much lower
    MIN_FREE_GPU_MB_WARM threshold applies instead of MIN_FREE_GPU_MB."""
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB", 8000.0)
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB_WARM", 500.0)
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    # 1000 MB free: below the cold threshold (8000) but above the warm one (500).
    monkeypatch.setattr("server.probe.gpu_status",
                         lambda: {"used_mb": 9000, "total_mb": 10000, "name": "Fake GPU"})

    job = store.create([_touch_upload(store.root)], "quick", {"device": "cuda"})
    worker = GpuWorker(store, bus, queue.Queue())
    worker._get_models("cuda")  # pre-warm the cache, as a prior job on this worker would have
    worker._process_job(job["id"])  # must proceed straight to running, no waiting_gpu

    assert store.get(job["id"])["status"] == "done"


def test_gpu_preflight_still_uses_cold_threshold_when_device_not_cached(store, bus, monkeypatch):
    """The other half of the warm-threshold test: a device NOT already in
    _models_cache must still use the full MIN_FREE_GPU_MB, since a fresh
    Models() load needs real headroom of its own."""
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "GPU_WAIT_POLL_S", 0.02)
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB", 8000.0)
    monkeypatch.setattr(jobs_mod, "MIN_FREE_GPU_MB_WARM", 500.0)
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    # 1000 MB free: below the cold threshold this job must be held to,
    # since "cuda" was never warmed on this worker.
    monkeypatch.setattr("server.probe.gpu_status",
                         lambda: {"used_mb": 9000, "total_mb": 10000, "name": "Fake GPU"})

    job = store.create([_touch_upload(store.root)], "quick", {"device": "cuda"})
    worker = GpuWorker(store, bus, queue.Queue())
    t = threading.Thread(target=worker._process_job, args=(job["id"],))
    t.start()
    try:
        for _ in range(200):
            if store.get(job["id"])["status"] == "waiting_gpu":
                break
            threading.Event().wait(0.02)
        else:
            raise AssertionError("job never reached waiting_gpu despite the cold threshold")
    finally:
        store.cancel(job["id"])
        t.join(timeout=5)


def test_worker_idle_unload_clears_models_cache_after_idle_period(store, bus):
    """Regression test: the worker used to hold every model it ever
    loaded resident in VRAM forever, with no way to release it short of
    restarting the whole server -- a real, avoidable cost on a GPU shared
    with other services."""
    worker = GpuWorker(store, bus, queue.Queue(), idle_unload_s=1.0)
    worker._models_cache["cuda"] = object()
    worker._last_activity_t = 0.0  # arbitrarily long ago (monotonic clock only grows)
    worker._maybe_unload_idle_models()
    assert worker._models_cache == {}


def test_worker_does_not_unload_before_idle_threshold(store, bus):
    worker = GpuWorker(store, bus, queue.Queue(), idle_unload_s=3600.0)
    worker._models_cache["cuda"] = object()
    worker._maybe_unload_idle_models()  # _last_activity_t was just set at construction
    assert "cuda" in worker._models_cache


def test_worker_idle_unload_disabled_when_threshold_is_zero(store, bus):
    worker = GpuWorker(store, bus, queue.Queue(), idle_unload_s=0)
    worker._models_cache["cuda"] = object()
    worker._last_activity_t = 0.0
    worker._maybe_unload_idle_models()
    assert "cuda" in worker._models_cache


def test_unload_on_queue_empty_releases_models_immediately_after_a_job(store, bus, monkeypatch):
    """A solo-operator convenience (see run.sh): with this flag on, VRAM
    is released the instant nothing else is queued, rather than waiting
    up to idle_unload_s (which defaults to 600s -- a long time to sit on
    a shared GPU for no reason once the browser tab has its result)."""
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue(), idle_unload_s=3600.0, unload_on_queue_empty=True)
    worker._models_cache["cuda"] = object()  # simulate an already-loaded model
    worker._process_job(job["id"])
    assert worker._models_cache == {}, "models should be released once the queue is empty, not left resident"


def test_unload_on_queue_empty_off_by_default_keeps_models_resident(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_factory("succeed"))
    job = store.create([_touch_upload(store.root)], "quick", None)
    worker = GpuWorker(store, bus, queue.Queue(), idle_unload_s=3600.0)  # unload_on_queue_empty defaults False
    worker._models_cache["cuda"] = object()
    worker._process_job(job["id"])
    assert "cuda" in worker._models_cache


def _drain(sub):
    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    return events


def test_bootstrap_reenqueues_queued_jobs(tmp_path, monkeypatch):
    from server.probe import ProbeInfo
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "validate_upload", lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001))
    monkeypatch.setattr(jobs_mod, "run_clip", _fake_run_clip_factory("succeed"))

    root = tmp_path / "jobs"
    pre_store = JobStore(root)
    job = pre_store.create([_touch_upload(root)], "quick", None)
    assert job["status"] == "queued"

    store, bus, worker = bootstrap(root)
    try:
        # give the daemon worker thread a moment to drain the one re-enqueued job
        for _ in range(100):
            if store.get(job["id"])["status"] in ("done", "failed", "cancelled"):
                break
            threading.Event().wait(0.05)
        assert store.get(job["id"])["status"] == "done"
    finally:
        worker.stop()


# -- flat-chroma clips must not wait for, or occupy, the GPU ------------------

def _chroma_clip(saturation=61.0, is_chroma=True):
    return {"status": "pending", "probe": {"bg_is_chroma_class": is_chroma,
                                            "bg_key_saturation": saturation}}


def test_flat_chroma_job_does_not_need_the_gpu():
    """The colour-only keyer never touches the GPU, so a job made entirely of
    flat-chroma clips must skip the free-VRAM wait -- otherwise exactly the
    content this path exists to speed up would still sit in waiting_gpu behind
    whatever else is using the shared card (the 10-minute stalls that
    motivated the work)."""
    job = {"clips": [_chroma_clip(), _chroma_clip()]}
    assert _job_needs_gpu(job, PipelineConfig()) is False


@pytest.mark.parametrize("clip,why", [
    (_chroma_clip(saturation=2.0), "white cyclorama: flat but unkeyable"),
    (_chroma_clip(is_chroma=False), "not a flat backdrop"),
    ({"status": "pending", "probe": None}, "never probed"),
    ({"status": "pending"}, "no probe key at all"),
])
def test_job_needs_the_gpu_unless_every_clip_is_keyable(clip, why):
    assert _job_needs_gpu({"clips": [clip]}, PipelineConfig()) is True, why


def test_a_mixed_batch_still_needs_the_gpu():
    job = {"clips": [_chroma_clip(), _chroma_clip(is_chroma=False)]}
    assert _job_needs_gpu(job, PipelineConfig()) is True


def test_use_keyer_off_forces_the_gpu_path():
    job = {"clips": [_chroma_clip()]}
    assert _job_needs_gpu(job, PipelineConfig(use_keyer="off")) is True


def test_finished_clips_are_ignored_when_deciding():
    """Only work still to do can require the GPU."""
    job = {"clips": [{"status": "done", "probe": {"bg_is_chroma_class": False}}, _chroma_clip()]}
    assert _job_needs_gpu(job, PipelineConfig()) is False

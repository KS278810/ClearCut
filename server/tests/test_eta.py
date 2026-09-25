"""Unit tests for server/eta.py -- the ETA/elapsed-time prediction model
(see the plan's "待ち時間の可視化" section). Pure logic, no GpuWorker/SSE
involved; integration with jobs.py's _process_job is covered separately
in test_jobs.py."""
import json
import queue
from pathlib import Path

import pytest

from server import eta
from server.jobs import GpuWorker, JobStore, EventBus
from tool.pipeline.config import PipelineConfig
from tool.pipeline.runner import JobCancelled


def _probe(frames=121, w=1656, h=1248):
    return {"width": w, "height": h, "fps": 24.0, "frames": frames, "duration_s": 5.0, "size_mb": 8.0}


def _touch_upload(tmp_path, name="clip.mp4", content=b"fake mp4 bytes"):
    p = tmp_path / f"upload-{name}"
    p.write_bytes(content)
    return name, p


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def store(tmp_path, monkeypatch, bus):
    import server.jobs as jobs_mod
    from server.probe import ProbeInfo
    monkeypatch.setattr(jobs_mod, "validate_upload",
                         lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001))
    return JobStore(tmp_path / "jobs", bus=bus)


@pytest.fixture(autouse=True)
def _no_real_models(monkeypatch):
    monkeypatch.setattr("server.jobs._load_models", lambda config: object())


@pytest.fixture(autouse=True)
def _no_gpu_preflight_by_default(monkeypatch):
    monkeypatch.setattr("server.jobs.MIN_FREE_GPU_MB", 0.0)


def _fake_run_clip_with_ticks(video_path, out_path, config, models=None, raw=None, *,
                              progress=None, cancel=None, timings=None, **_kw):
    """A fake run_clip that emits enough real ticks for ClipEta to have
    something to blend against (unlike test_jobs.py's own one-shot fake,
    which is fine for state-machine tests but gives ETA nothing to chew
    on)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        for i in range(1, 6):
            progress("infer", i, 5)
        progress("postprocess", 1, 1)
        progress("encode", 1, 1)
        progress("done", 1, 1)
    if timings is not None:
        timings.update(detect_s=0.01, birefnet_s=0.02, despill_s=0.01, postprocess_s=0.01, encode_s=0.01)
    Path(out_path).write_bytes(b"GIF89afake")
    Path(str(out_path) + ".pipeline_config.json").write_text("{}")
    return out_path


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# --------------------------------------------------------------------------
# TimingStats
# --------------------------------------------------------------------------

def test_predict_uses_seed_coefficients_for_an_unknown_group():
    stats = eta.TimingStats(path=None)
    cfg = PipelineConfig(device="cuda")
    pred = stats.predict(_probe(), cfg, qc=False)
    assert pred["infer"] > 0
    assert pred["encode"] > 0
    assert pred["total"] == pytest.approx(sum(pred[s] for s in ("infer", "postprocess", "encode", "qc")))


def test_predict_scales_linearly_with_frames_times_megapixels():
    stats = eta.TimingStats(path=None)
    cfg = PipelineConfig(device="cuda")
    small = stats.predict(_probe(frames=60, w=800, h=600), cfg, qc=False)
    big = stats.predict(_probe(frames=120, w=800, h=600), cfg, qc=False)
    assert big["infer"] == pytest.approx(2 * small["infer"], rel=1e-6)


def test_predict_zeroes_stages_disabled_by_config():
    stats = eta.TimingStats(path=None)
    cfg_nodespill = PipelineConfig(device="cuda", apply_despill=False)
    cfg_despill = PipelineConfig(device="cuda", apply_despill=True)
    p_no = stats.predict(_probe(), cfg_nodespill, qc=False)
    p_yes = stats.predict(_probe(), cfg_despill, qc=False)
    assert p_no["infer"] < p_yes["infer"]

    cfg_nomedian = PipelineConfig(mc_median_half=0)
    assert stats.predict(_probe(), cfg_nomedian, qc=False)["postprocess"] == 0.0

    assert stats.predict(_probe(), cfg_despill, qc=False)["qc"] == 0.0
    assert stats.predict(_probe(), cfg_despill, qc=True)["qc"] > 0.0


def test_predict_keyed_skips_the_neural_stages_and_uses_ss_alpha_gif_encode():
    """Audit M2: without `keyed`, a flat-chroma clip's ETA added
    detect+birefnet+despill+postprocess+the heavy two-pass encoder, none of
    which tool/pipeline/keyer.py's fast path actually runs -- a ~30s clip
    showed "残り 約20分". keyed=True must route entirely differently: the
    "key" stage only, zero postprocess (mc_median is forced off on this
    route), and the encode coefficient priced as ss_alpha_gif regardless of
    what config.encoder itself says (run_clip swaps it automatically)."""
    stats = eta.TimingStats(path=None)
    # encoder="supersampled_gif" explicit (not PipelineConfig's own default,
    # which stopped being supersampled_gif on 2026-09-22): keeps this test
    # meaningful as "keyed pricing ignores config.encoder" -- if cfg already
    # said ss_alpha_gif, the assertion below would pass even if that
    # override logic were silently removed.
    cfg = PipelineConfig(device="cuda", encoder="supersampled_gif")  # mc_median_half=1 (default)
    probe = _probe(frames=124, w=768, h=768)

    keyed_pred = stats.predict(probe, cfg, qc=False, keyed=True)
    neural_pred = stats.predict(probe, cfg, qc=False, keyed=False)

    assert keyed_pred["postprocess"] == 0.0
    assert keyed_pred["infer"] > 0.0
    assert keyed_pred["infer"] < neural_pred["infer"], "keyed infer must not add detect+birefnet+despill"
    assert keyed_pred["encode"] == pytest.approx(
        stats.predict(probe, PipelineConfig(encoder="ss_alpha_gif"), qc=False, keyed=False)["encode"])
    assert keyed_pred["total"] < neural_pred["total"]


def test_update_keyed_writes_to_its_own_bucket_not_the_neural_ones(tmp_path):
    """Audit M1: a keyed clip's encode_s (~19s, ss_alpha_gif) used to be
    EMA'd straight into "encode:supersampled_gif@N" (true cost ~600s) and
    its postprocess_s (~0s) into "postprocess:cpu" -- silently dragging the
    NEURAL route's own coefficients down every time a flat-chroma clip ran,
    persisted to disk so it never self-corrected. keyed=True must isolate
    all of that into the "key" bucket (and skip postprocess_s entirely --
    it's always ~0 by construction on this route, nothing to learn)."""
    path = tmp_path / "timing_stats.json"
    stats = eta.TimingStats(path=path)
    # encoder="supersampled_gif" explicit: PipelineConfig's own default
    # stopped being supersampled_gif on 2026-09-22 (第9計画/第8回監査レバー
    # 4), so this pins the exact historical bug shape (a config nominally
    # requesting the heavy encoder that turns out to be keyed) rather than
    # relying on whatever the current default happens to be.
    cfg = PipelineConfig(device="cuda", encoder="supersampled_gif")
    probe = _probe(frames=124, w=768, h=768)

    neural_encode_before = stats.predict(probe, cfg, qc=False, keyed=False)["encode"]
    neural_postprocess_before = stats.predict(probe, cfg, qc=False, keyed=False)["postprocess"]

    stats.update(cfg, probe, {"key_s": 7.4, "postprocess_s": 0.0, "encode_s": 19.0}, ok=True, keyed=True)

    saved = json.loads(path.read_text())
    assert "key" in saved["coef"] and "key" in saved["coef"]["key"]
    assert "supersampled_gif" not in json.dumps(saved["coef"]), (
        "a keyed clip's encode_s must not touch any supersampled_gif@N bucket")
    assert saved["coef"].get("postprocess:cpu", {}).get("postprocess", {}).get("n", 0) == 0, (
        "a keyed clip's ~0 postprocess_s must not be recorded at all")

    # The neural route's own predictions must be completely unaffected.
    assert stats.predict(probe, cfg, qc=False, keyed=False)["encode"] == pytest.approx(neural_encode_before)
    assert stats.predict(probe, cfg, qc=False, keyed=False)["postprocess"] == pytest.approx(neural_postprocess_before)

    # The keyed prediction itself DOES move toward the observation.
    keyed_before = eta.SEED_COEF["key"]["key"]
    keyed_after_infer = stats.predict(probe, cfg, qc=False, keyed=True)["infer"]
    assert keyed_after_infer != pytest.approx(keyed_before * probe["frames"] * (probe["width"] * probe["height"] / 1e6))


def test_update_moves_coefficient_toward_observation_and_persists(tmp_path):
    path = tmp_path / "timing_stats.json"
    stats = eta.TimingStats(path=path)
    cfg = PipelineConfig(device="cuda")
    probe = _probe(frames=100, w=1000, h=1000)  # 100 frame*MPix
    before = stats.predict(probe, cfg, qc=False)["infer"]

    # Observation implies a MUCH larger coefficient than the seed -- one
    # EMA step should move toward it without jumping all the way there.
    stats.update(cfg, probe, {"detect_s": 1000.0, "birefnet_s": 0.0, "despill_s": 0.0}, ok=True)
    after = stats.predict(probe, cfg, qc=False)["infer"]
    assert after > before
    assert after < 1000.0  # not a full jump to the single observation

    assert path.is_file()
    reloaded = eta.TimingStats(path=path)
    assert reloaded.predict(probe, cfg, qc=False)["infer"] == pytest.approx(after)


def test_network_stages_are_priced_per_frame_not_per_megapixel():
    """第11計画 Part 1-4: YOLOX and BiRefNet run at a fixed input size, so a
    clip's resolution must not change their predicted cost (the per-MPix
    model over-predicted dinosaur GPU as 417.7s vs 332s actual)."""
    stats = eta.TimingStats(path=None)
    cfg = PipelineConfig(device="cuda", apply_despill=False)
    lo = stats.predict(_probe(frames=121, w=960, h=540), cfg, qc=False)["infer"]
    hi = stats.predict(_probe(frames=121, w=1656, h=1248), cfg, qc=False)["infer"]
    assert hi == pytest.approx(lo)
    assert lo == pytest.approx(121 * (eta.SEED_COEF["cuda"]["detect_frame"]
                                      + eta.SEED_COEF["cuda"]["birefnet_frame"]))


def test_old_per_mpix_network_entries_in_the_stats_file_are_ignored(tmp_path):
    """Inflated EMA values persisted under the old "detect"/"birefnet" names
    (per frame*MPix) must not leak into the new per-frame prediction."""
    path = tmp_path / "timing_stats.json"
    path.write_text(json.dumps({"version": 1, "coef": {"cuda": {
        "detect": {"s_per_frame_mpix": 50.0, "n": 6},
        "birefnet": {"s_per_frame_mpix": 50.0, "n": 6}}}}))
    stats = eta.TimingStats(path=path)
    cfg = PipelineConfig(device="cuda", apply_despill=False)
    assert stats.predict(_probe(), cfg, qc=False)["infer"] == pytest.approx(
        eta.TimingStats(path=None).predict(_probe(), cfg, qc=False)["infer"])


def test_update_learns_network_stages_per_frame(tmp_path):
    path = tmp_path / "timing_stats.json"
    stats = eta.TimingStats(path=path)
    cfg = PipelineConfig(device="cuda")
    probe = _probe(frames=100, w=1656, h=1248)
    stats.update(cfg, probe, {"birefnet_s": 100 * 1.31}, ok=True)  # 1.31 s/frame observed
    bucket = json.loads(path.read_text())["coef"]["cuda"]["birefnet_frame"]
    seed = eta.SEED_COEF["cuda"]["birefnet_frame"]
    assert bucket["s_per_frame"] == pytest.approx(0.7 * seed + 0.3 * 1.31)
    assert "s_per_frame_mpix" not in bucket


def test_cpu_neural_prediction_reflects_measured_cost():
    """CPU BiRefNet measured ~30-70 s/frame on this machine; the old seed
    predicted a 121-frame clip's CPU infer in minutes, 8-17x too low."""
    stats = eta.TimingStats(path=None)
    pred = stats.predict(_probe(frames=121), PipelineConfig(device="cpu"), qc=False)
    assert pred["infer"] >= 121 * 30


def test_update_ignores_encode_s_when_clip_did_not_succeed(tmp_path):
    stats = eta.TimingStats(path=None)
    cfg = PipelineConfig(device="cuda")
    probe = _probe(frames=100, w=1000, h=1000)
    before = stats.predict(probe, cfg, qc=False)["encode"]
    stats.update(cfg, probe, {"encode_s": 99999.0}, ok=False)
    after = stats.predict(probe, cfg, qc=False)["encode"]
    assert after == before


def test_corrupt_stats_file_is_ignored_without_raising(tmp_path):
    path = tmp_path / "timing_stats.json"
    path.write_text("{not valid json")
    stats = eta.TimingStats(path=path)  # must not raise
    cfg = PipelineConfig(device="cuda")
    pred = stats.predict(_probe(), cfg, qc=False)
    assert pred["total"] > 0  # fell back to seeds


def test_wrong_schema_stats_file_is_ignored_without_raising(tmp_path):
    path = tmp_path / "timing_stats.json"
    path.write_text(json.dumps({"coef": "not-a-dict"}))
    stats = eta.TimingStats(path=path)
    cfg = PipelineConfig(device="cuda")
    assert stats.predict(_probe(), cfg, qc=False)["total"] > 0


# --------------------------------------------------------------------------
# ClipEta
# --------------------------------------------------------------------------

def _pred():
    return {"infer": 100.0, "postprocess": 40.0, "encode": 200.0, "qc": 0.0, "total": 340.0}


def test_remaining_starts_at_the_full_prediction_before_any_ticks():
    clock = _FakeClock()
    ce = eta.ClipEta(_pred(), now=clock)
    assert ce.remaining() == pytest.approx(340.0, rel=1e-6)


def test_remaining_uses_seed_prediction_below_10pct_progress():
    clock = _FakeClock()
    ce = eta.ClipEta(_pred(), now=clock)
    ce.on_tick("infer", 1, 121)  # < 10%
    clock.advance(1.0)
    r = ce.remaining()
    # Should be close to the full prediction minus a little elapsed time,
    # not an observed-rate extrapolation from one tick.
    assert r == pytest.approx(340.0, abs=5.0)


def test_remaining_takes_over_with_observed_rate_past_10pct():
    clock = _FakeClock()
    ce = eta.ClipEta({"infer": 1000.0, "postprocess": 0.0, "encode": 0.0, "qc": 0.0, "total": 1000.0}, now=clock)
    ce.on_tick("infer", 1, 100)
    clock.advance(100.0)
    ce.on_tick("infer", 50, 100)  # 50% done in 100s -- observed rate says ~100s left
    r = ce.remaining()
    # Seed predicted 1000s total (900s remaining at 50% elapsed=100s); the
    # observed rate says ~100s remaining. Blended value must sit far closer
    # to the observed rate, not the stale seed.
    assert r < 400.0


def test_remaining_is_never_negative_and_reaches_zero_on_finish():
    clock = _FakeClock()
    ce = eta.ClipEta(_pred(), now=clock)
    ce.on_tick("infer", 121, 121)
    ce.on_tick("postprocess", 10, 10)
    ce.on_tick("encode", 121, 121)
    clock.advance(1000.0)  # long past every prediction
    assert ce.remaining() >= 0.0
    ce.finish()
    assert ce.remaining() == 0.0


def test_remaining_is_monotonically_non_increasing_under_constant_rate_progress():
    clock = _FakeClock()
    ce = eta.ClipEta({"infer": 100.0, "postprocess": 0.0, "encode": 0.0, "qc": 0.0, "total": 100.0}, now=clock)
    prev = None
    for done in range(1, 101, 5):
        clock.advance(1.0)
        ce.on_tick("infer", done, 100)
        r = ce.remaining()
        if prev is not None:
            assert r <= prev + 1e-6
        prev = r


def test_remaining_upward_jump_is_capped():
    clock = _FakeClock()
    ce = eta.ClipEta({"infer": 100.0, "postprocess": 0.0, "encode": 0.0, "qc": 0.0, "total": 100.0}, now=clock)
    ce.on_tick("infer", 50, 100)
    clock.advance(1.0)
    first = ce.remaining()
    # A later tick reporting LESS progress than expected (rate suddenly
    # slower) must not make the displayed remaining time jump arbitrarily
    # far upward in one step.
    ce.on_tick("infer", 51, 1000000)  # contrived: total exploded
    clock.advance(1.0)
    second = ce.remaining()
    assert second <= first * 1.5 + 1e-6


def test_encode_tail_has_a_real_estimate_once_all_frames_are_written():
    clock = _FakeClock()
    ce = eta.ClipEta({"infer": 0.0, "postprocess": 0.0, "encode": 200.0, "qc": 0.0, "total": 200.0}, now=clock)
    ce.on_tick("encode", 0, 121)
    clock.advance(70.0)  # ~write share (0.35*200=70s) elapsed
    ce.on_tick("encode", 121, 121)  # all frames written -- ffmpeg tail begins
    clock.advance(1.0)
    r = ce.remaining()
    # Tail share is 0.65*200=130s; only ~1s of it has elapsed.
    assert 100.0 < r < 130.0


# --------------------------------------------------------------------------
# Integration: server/jobs.py's GpuWorker actually wires eta.py in
# --------------------------------------------------------------------------

def test_first_progress_event_carries_a_positive_eta_and_job_eta(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_with_ticks)
    job = store.create([_touch_upload(store.root)], "quick", None)
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())  # stats_path=None -> in-memory only
    worker._process_job(job["id"])

    events = []
    while True:
        try:
            events.append(sub.get_nowait())
        except queue.Empty:
            break
    progress_events = [e for e in events if e["type"] == "progress"]
    assert progress_events, "expected at least one progress event"
    first = progress_events[0]
    assert first["eta_s"] is not None and first["eta_s"] > 0
    assert first["job_eta_s"] is not None
    assert first["job_eta_s"] >= first["eta_s"]
    assert first["elapsed_s"] is not None and first["elapsed_s"] >= 0


def _first_progress_job_eta(store, bus, monkeypatch, n_clips):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_with_ticks)
    uploads = [_touch_upload(store.root, f"clip{i}.mp4") for i in range(n_clips)]
    job = store.create(uploads, "quick", None)
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])
    while True:
        e = sub.get_nowait()
        if e["type"] == "progress":
            return e["job_eta_s"]


def test_job_eta_accounts_for_a_still_pending_second_clip(store, bus, monkeypatch, tmp_path):
    """job_eta_s at clip 0's very first tick must be larger when there's a
    second, still-pending clip behind it than when clip 0 is the only
    clip -- checked by comparing two separate runs (rather than reading
    clip 1's predicted_s back out after the job finishes) because
    TimingStats.update() mutates the shared coefficients as each clip
    completes, so a value read post-hoc no longer matches what was live
    at the moment of that first event."""
    job_eta_one_clip = _first_progress_job_eta(store, bus, monkeypatch, n_clips=1)

    bus2 = EventBus()
    store2 = JobStore(tmp_path / "jobs2", bus=bus2)
    import server.jobs as jobs_mod
    from server.probe import ProbeInfo
    monkeypatch.setattr(jobs_mod, "validate_upload", lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001))
    job_eta_two_clips = _first_progress_job_eta(store2, bus2, monkeypatch, n_clips=2)

    assert job_eta_two_clips > job_eta_one_clip


def test_done_clip_has_started_and_finished_timestamps_and_stats_are_updated(store, bus, monkeypatch, tmp_path):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_with_ticks)
    job = store.create([_touch_upload(store.root)], "quick", None)
    stats_path = tmp_path / "timing_stats.json"
    worker = GpuWorker(store, bus, queue.Queue(), stats_path=stats_path)
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    clip = updated["clips"][0]
    assert clip["status"] == "done"
    assert clip["started_at"] is not None
    assert clip["finished_at"] is not None
    assert clip["eta_s"] == 0.0
    assert clip["predicted_s"] is not None

    assert stats_path.is_file()
    saved = json.loads(stats_path.read_text())
    # At least one (group, stage) bucket should now have n>=1 after a
    # single completed clip fed its real timings into the EMA.
    ns = [bucket["n"] for group in saved["coef"].values() for bucket in group.values()]
    assert ns and max(ns) >= 1


def test_a_keyed_clips_predicted_and_saved_eta_use_the_key_pricing(store, bus, monkeypatch, tmp_path):
    """End-to-end (via GpuWorker._process_job, not just the unit-level
    predict()/update() calls above): a clip whose probe says it's a real,
    saturated flat-chroma backdrop must get a "key"-priced prediction (not
    the inflated neural-route one) BEFORE it runs, and its finished timings
    must land in the "key" bucket, not "encode:supersampled_gif@N"."""
    import server.jobs as jobs_mod
    from server.probe import ProbeInfo

    monkeypatch.setattr(jobs_mod, "validate_upload",
                         lambda path: ProbeInfo(768, 768, 24.0, 124, 5.0, 0.8,
                                                bg_is_chroma_class=True, bg_frac_bg_like=1.0,
                                                bg_key_saturation=61.0))

    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                       progress=None, cancel=None, timings=None, **_kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if timings is not None:
            timings.update(key_s=7.4, postprocess_s=0.0, encode_s=19.0)
        Path(out_path).write_bytes(b"GIF89afake")
        Path(str(out_path) + ".pipeline_config.json").write_text("{}")
        return out_path

    monkeypatch.setattr(jobs_mod, "run_clip", fake_run_clip)
    stats_path = tmp_path / "timing_stats.json"
    # encoder="supersampled_gif" explicit: "quick" itself defaults to
    # ss_alpha_gif since 2026-09-22 (第9計画レバー4), which would make the
    # "supersampled_gif" assertion below trivially true regardless of
    # whether the keyed-bucket isolation actually works. Pin the historical
    # bug shape explicitly instead.
    job = store.create([_touch_upload(store.root)], "quick", {"encoder": "supersampled_gif"})

    worker = GpuWorker(store, bus, queue.Queue(), stats_path=stats_path)
    worker._process_job(job["id"])

    updated = store.get(job["id"])
    assert updated["clips"][0]["auto_keyer"] is True
    # predicted_s is written twice (the job-start bulk pass over every
    # pending clip, then recomputed per-clip right before it runs -- see
    # _process_job) -- both must use the keyed pricing, and what's left on
    # the finished record is the second one.
    assert updated["clips"][0]["predicted_s"]["postprocess"] == 0.0

    saved = json.loads(stats_path.read_text())
    assert "key" in saved["coef"]
    assert "supersampled_gif" not in json.dumps(saved["coef"])


def test_clip_fraction_sequence_is_monotonically_non_decreasing(store, bus, monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip_with_ticks)
    job = store.create([_touch_upload(store.root)], "quick", None)
    sub = bus.subscribe(job["id"])
    worker = GpuWorker(store, bus, queue.Queue())
    worker._process_job(job["id"])

    fractions = []
    while True:
        try:
            e = sub.get_nowait()
        except queue.Empty:
            break
        if e["type"] == "progress":
            fractions.append(e["fraction"])
    assert fractions == sorted(fractions)


def test_cancelled_clip_does_not_feed_a_partial_encode_s_into_stats(store, bus, monkeypatch, tmp_path):
    def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                       progress=None, cancel=None, timings=None, **_kw):
        if timings is not None:
            timings["encode_s"] = 0.001  # a partial, misleadingly-tiny duration
        raise JobCancelled("cancelled mid-encode")

    monkeypatch.setattr("server.jobs.run_clip", fake_run_clip)
    job = store.create([_touch_upload(store.root)], "quick", None)
    stats_path = tmp_path / "timing_stats.json"
    worker = GpuWorker(store, bus, queue.Queue(), stats_path=stats_path)
    worker._process_job(job["id"])

    saved = json.loads(stats_path.read_text()) if stats_path.is_file() else {"coef": {}}
    encode_buckets = [bucket for group_name, group in saved["coef"].items()
                      if group_name.startswith("encode:") for stage, bucket in group.items()]
    assert not encode_buckets, "a cancelled clip's encode_s must not update the encode coefficient"

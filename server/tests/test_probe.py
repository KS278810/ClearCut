from pathlib import Path

import pytest

from server import errors
from server.probe import ProbeError, cpu_load_ratio, disk_free_mb, gpu_status, validate_upload

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLIP = REPO_ROOT.parent / "sample" / "dinosaur" / "Triceratops.mp4"
# A REAL webm reproducing the exact bug this guards against: piped/streamed
# through ffmpeg with no seekable cues index, so cv2's CAP_PROP_FRAME_COUNT
# comes back as a bogus huge-negative float (confirmed: -9.223372036854776e+16,
# i.e. int64-min misrepresented) instead of 0 or -1 as first suspected --
# `int(that)` is still comfortably <= 0, so the same guard catches it. 10
# frames @ 10fps, generated via:
#   ffmpeg -f lavfi -i testsrc=duration=1:size=64x48:rate=10 \
#     -c:v libvpx-vp9 -f webm - > frameless.webm
FRAMELESS_WEBM = Path(__file__).resolve().parent / "fixtures" / "frameless.webm"


@pytest.fixture(autouse=True)
def _reset_gpu_status_cache():
    """gpu_status() now memoizes for GPU_STATUS_CACHE_S (L5) in a
    module-level dict shared by every caller -- without resetting it here,
    whichever test happens to run first within that window populates the
    cache for every test after it, e.g. test_gpu_status_returns_none_or_
    well_shaped_dict()'s real nvidia-smi result would still be cached (and
    returned instead of None) when test_gpu_status_survives_missing_binary
    runs moments later with subprocess.run monkeypatched to always raise."""
    import server.probe as probe_mod
    probe_mod._gpu_status_cache["t"] = float("-inf")
    probe_mod._gpu_status_cache["value"] = None
    yield


@pytest.mark.skipif(not SAMPLE_CLIP.exists(), reason="sample/dinosaur/Triceratops.mp4 not present on this machine")
def test_real_sample_clip_validates():
    info = validate_upload(SAMPLE_CLIP)
    assert info.width == 1656
    assert info.height == 1248
    assert info.frames == 121
    assert 5.0 < info.duration_s < 5.2
    assert info.size_mb > 0
    # Triceratops.mp4 (the sample/dinosaur/ clip since the 2026-09-13 reorg)
    # is a natural/live-action-style backdrop, NOT the flat chroma colour
    # the original four (思考/喜び/感謝/挨拶, now deleted) were -- this is a
    # real, honest measurement of this specific clip, not a stand-in for
    # "the pipeline's design target must gate as chroma" (that assertion,
    # true of the old clips, would be FALSE here and was removed rather
    # than left silently wrong).
    assert info.bg_is_chroma_class is False
    assert info.bg_frac_bg_like < 0.98


def test_rejects_bad_extension(tmp_path):
    bad = tmp_path / "clip.avi"
    bad.write_bytes(b"not a real video")
    with pytest.raises(ProbeError) as exc:
        validate_upload(bad)
    assert exc.value.code == errors.E_INPUT_EXT


def test_rejects_unreadable_file(tmp_path):
    bad = tmp_path / "clip.mp4"
    bad.write_bytes(b"this is not actually an mp4 container")
    with pytest.raises(ProbeError) as exc:
        validate_upload(bad)
    assert exc.value.code == errors.E_INPUT_UNREADABLE


def test_rejects_oversized_file(tmp_path, monkeypatch):
    import server.probe as probe_mod
    monkeypatch.setattr(probe_mod, "MAX_FILE_MB", 0.001)
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"\x00" * 4096)
    with pytest.raises(ProbeError) as exc:
        validate_upload(fake)
    assert exc.value.code == errors.E_INPUT_TOO_LARGE


@pytest.mark.skipif(not SAMPLE_CLIP.exists(), reason="sample/dinosaur/Triceratops.mp4 not present on this machine")
def test_rejects_too_long(monkeypatch):
    import server.probe as probe_mod
    monkeypatch.setattr(probe_mod, "MAX_DURATION_S", 0.1)
    monkeypatch.setattr(probe_mod, "MAX_FRAMES", 1)
    with pytest.raises(ProbeError) as exc:
        validate_upload(SAMPLE_CLIP)
    assert exc.value.code == errors.E_INPUT_TOO_LONG


@pytest.mark.skipif(not SAMPLE_CLIP.exists(), reason="sample/dinosaur/Triceratops.mp4 not present on this machine")
def test_rejects_too_large_resolution(monkeypatch):
    import server.probe as probe_mod
    monkeypatch.setattr(probe_mod, "MAX_LONG_SIDE_PX", 100)
    with pytest.raises(ProbeError) as exc:
        validate_upload(SAMPLE_CLIP)
    assert exc.value.code == errors.E_INPUT_TOO_LARGE


def test_cv2_reports_a_bogus_frame_count_for_the_frameless_fixture():
    """Documents the actual failure mode this guards against, independent
    of validate_upload -- if a future OpenCV/ffmpeg version starts
    reporting this container's frame count sanely, this test (not the
    ffprobe-fallback ones below) is what will tell us so."""
    import cv2
    cap = cv2.VideoCapture(str(FRAMELESS_WEBM))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0
    finally:
        cap.release()


@pytest.mark.skipif(not FRAMELESS_WEBM.is_file(), reason="server/tests/fixtures/frameless.webm missing")
def test_ffprobe_fallback_recovers_the_real_frame_count():
    info = validate_upload(FRAMELESS_WEBM)
    assert info.frames == 10
    assert 0.9 < info.duration_s < 1.1


@pytest.mark.skipif(not FRAMELESS_WEBM.is_file(), reason="server/tests/fixtures/frameless.webm missing")
def test_unreadable_when_both_cv2_and_ffprobe_fail_to_report_frames(monkeypatch):
    import server.probe as probe_mod
    monkeypatch.setattr(probe_mod, "_ffprobe_frames", lambda path: None)
    with pytest.raises(ProbeError) as exc:
        validate_upload(FRAMELESS_WEBM)
    assert exc.value.code == errors.E_INPUT_UNREADABLE
    assert "frame count unknown" in exc.value.detail


@pytest.mark.skipif(not SAMPLE_CLIP.exists(), reason="sample/dinosaur/Triceratops.mp4 not present on this machine")
def test_rejects_when_megapixel_frame_budget_exceeded(monkeypatch):
    import server.probe as probe_mod
    monkeypatch.setattr(probe_mod, "MAX_MEGAPIXEL_FRAMES", 1.0)
    with pytest.raises(ProbeError) as exc:
        validate_upload(SAMPLE_CLIP)
    assert exc.value.code == errors.E_INPUT_TOO_LARGE
    assert "MPix-frame" in exc.value.detail


def test_gpu_status_returns_none_or_well_shaped_dict():
    status = gpu_status()
    assert status is None or ({"used_mb", "total_mb", "name"} <= status.keys()
                               and isinstance(status["used_mb"], int)
                               and isinstance(status["total_mb"], int))


def test_gpu_status_survives_missing_binary(monkeypatch):
    import server.probe as probe_mod

    def _boom(*a, **kw):
        raise FileNotFoundError("no nvidia-smi here")

    monkeypatch.setattr(probe_mod.subprocess, "run", _boom)
    assert gpu_status() is None


def test_gpu_status_is_memoized_within_the_cache_window(monkeypatch):
    """Regression test for the plan's L5 finding: gpu_status() forked
    nvidia-smi on every single call -- the system strip polls it every few
    seconds per connected browser, and commit 9's GPU pre-flight wait polls
    it every 15s from the worker thread on top of that."""
    import server.probe as probe_mod

    calls = {"n": 0}

    def fake_run(*a, **kw):
        calls["n"] += 1
        import subprocess as real_subprocess
        return real_subprocess.CompletedProcess(a[0], 0, stdout="1000, 8000, Fake GPU\n", stderr="")

    monkeypatch.setattr(probe_mod.subprocess, "run", fake_run)
    first = gpu_status()
    for _ in range(5):
        assert gpu_status() == first
    assert calls["n"] == 1


@pytest.mark.skipif(not SAMPLE_CLIP.exists(), reason="sample/dinosaur/Triceratops.mp4 not present on this machine")
def test_probe_bg_class_propagates_a_non_chroma_verdict(monkeypatch):
    """_probe_bg_class's own sampling/downscale plumbing is exercised
    against a real clip; the classification VERDICT itself is controlled
    here via estimate_bg_stats (rather than relying on SAMPLE_CLIP's own
    real classification, which happens to also be non-chroma since the
    2026-09-13 reorg -- see test_real_sample_clip_validates -- so this
    test still isolates "does the verdict propagate correctly" from
    "what does this one clip's own backdrop happen to look like")."""
    import server.probe as probe_mod
    from tool.qc.metrics import BgStats

    fake_stats = BgStats(mu_cb=100.0, mu_cr=100.0, sigma_cb=40.0, sigma_cr=40.0,
                          is_chroma_class=False, frac_bg_like=0.42)
    monkeypatch.setattr("tool.qc.metrics.estimate_bg_stats", lambda *a, **kw: fake_stats)
    is_chroma, frac, saturation = probe_mod._probe_bg_class(SAMPLE_CLIP, frames=121)
    assert is_chroma is False
    assert frac == pytest.approx(0.42)
    # hypot(100-128, 100-128) -- the backdrop-saturation figure the worker
    # uses to predict whether the keyer will handle a clip without the GPU.
    assert saturation == pytest.approx(39.598, abs=1e-2)


def test_probe_bg_class_returns_all_none_for_an_unopenable_path():
    import server.probe as probe_mod
    assert probe_mod._probe_bg_class(Path("/no/such/file.mp4"), frames=10) == (None, None, None)


def test_probe_bg_class_never_raises_even_if_estimate_bg_stats_does(monkeypatch):
    """Regression guard for the plan's G1 design constraint: a probe
    failure here must mean "unknown", never "reject the upload" -- the
    actual pipeline run has its own independent fallback regardless."""
    import server.probe as probe_mod

    def _boom(*a, **kw):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("tool.qc.metrics.estimate_bg_stats", _boom)
    result = probe_mod._probe_bg_class(SAMPLE_CLIP if SAMPLE_CLIP.exists() else Path("/no/such/file.mp4"),
                                        frames=10)
    assert result == (None, None, None)


def test_disk_free_mb_creates_dir_and_reports_positive(tmp_path):
    target = tmp_path / "nested" / "data"
    free = disk_free_mb(target)
    assert target.is_dir()
    assert free is not None and free > 0


def test_cpu_load_ratio_divides_load_average_by_core_count(monkeypatch):
    import server.probe as probe_mod
    monkeypatch.setattr(probe_mod.os, "getloadavg", lambda: (12.0, 10.0, 8.0))
    monkeypatch.setattr(probe_mod.os, "cpu_count", lambda: 24)
    assert cpu_load_ratio() == pytest.approx(0.5)


def test_cpu_load_ratio_returns_none_if_getloadavg_unavailable(monkeypatch):
    import server.probe as probe_mod

    def _boom():
        raise OSError("not supported on this platform")

    monkeypatch.setattr(probe_mod.os, "getloadavg", _boom)
    assert cpu_load_ratio() is None

import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from server import errors
from server.app import create_app


def _fake_run_clip(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress("infer", 1, 1)
        progress("postprocess", 0, 1)
        progress("encode", 1, 1)
        progress("done", 1, 1)
    Path(out_path).write_bytes(b"GIF89afake-gif-bytes")
    Path(str(out_path) + ".pipeline_config.json").write_text("{}")
    return out_path


@pytest.fixture(autouse=True)
def _fast_fake_pipeline(monkeypatch):
    monkeypatch.setattr("server.jobs.run_clip", _fake_run_clip)
    monkeypatch.setattr("server.jobs._load_models", lambda config: object())
    from server.probe import ProbeInfo
    monkeypatch.setattr("server.jobs.validate_upload", lambda path: ProbeInfo(64, 48, 24.0, 5, 0.2, 0.001))
    # L1's GPU pre-flight check calls the REAL probe.gpu_status() (real
    # nvidia-smi) for a "cuda"-device job (PipelineConfig's own default,
    # and every _upload() call here). Without this, these tests would
    # depend on this shared machine's actual free VRAM at test time --
    # see test_jobs.py's identically-purposed fixture for the full reasoning.
    monkeypatch.setattr("server.jobs.MIN_FREE_GPU_MB", 0.0)
    # Same reasoning for LOAD_AWARE_ENCODE's real os.getloadavg() call --
    # this shared machine's actual load average at test time would
    # otherwise silently swap the encoder to webp and break every test
    # here that assumes a .gif output (confirmed: this is exactly what
    # happened the first time this feature landed).
    monkeypatch.setattr("server.jobs.LOAD_AWARE_ENCODE", False)


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as c:
        yield c


def _upload(client, mode="quick", overrides=None, filename="clip.mp4"):
    files = [("files", (filename, io.BytesIO(b"fake bytes"), "video/mp4"))]
    data = {"mode": mode}
    if overrides is not None:
        data["overrides"] = json.dumps(overrides)
    return client.post("/api/jobs", files=files, data=data)


def _wait_until_terminal(client, job_id, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal status within {timeout_s}s")


def test_create_and_get_job(client):
    resp = _upload(client)
    assert resp.status_code == 201
    job = resp.json()
    assert job["status"] == "queued"

    got = client.get(f"/api/jobs/{job['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == job["id"]


def test_job_processes_to_done(client):
    job = _upload(client).json()
    final = _wait_until_terminal(client, job["id"])
    assert final["status"] == "done"
    assert final["clips"][0]["outputs"]["primary"]


def test_list_jobs_includes_created_job(client):
    job = _upload(client).json()
    listing = client.get("/api/jobs").json()
    ids = [j["id"] for j in listing["jobs"]]
    assert job["id"] in ids
    assert listing["total"] >= 1


def test_bad_mode_returns_400_with_code(client):
    resp = _upload(client, mode="ultra-fast")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == errors.E_BAD_MODE


def test_bad_overrides_json_returns_400(client):
    files = [("files", ("clip.mp4", io.BytesIO(b"x"), "video/mp4"))]
    resp = client.post("/api/jobs", files=files, data={"mode": "quick", "overrides": "{not json"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == errors.E_BAD_OVERRIDE


def test_get_missing_job_returns_404(client):
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == errors.E_JOB_NOT_FOUND


def test_cancel_endpoint_stops_a_running_job(client, monkeypatch):
    # A fake that blocks until the API's cancel() call has actually fired
    # (via the real cancel Event GpuWorker passes it), so this test proves
    # the whole path -- POST /cancel -> JobStore.cancel() -> the Event
    # run_clip receives -- without racing the background worker thread for
    # who gets to the job first (test_jobs.py already covers the
    # deterministic "still queued" transition at the store level directly).
    from pathlib import Path
    from tool.pipeline.runner import JobCancelled

    def slow_fake(video_path, out_path, config, models=None, raw=None, *, progress=None, cancel=None, **_kw):
        for _ in range(200):  # up to ~2s
            if cancel is not None and cancel.is_set():
                raise JobCancelled("cancelled by test")
            time.sleep(0.01)
        raise AssertionError("cancel was never observed by the fake pipeline")

    monkeypatch.setattr("server.jobs.run_clip", slow_fake)
    job = _upload(client).json()
    # give the worker a moment to actually start the job (so this exercises
    # the "running" cancel path, not the "still queued" one)
    for _ in range(100):
        if client.get(f"/api/jobs/{job['id']}").json()["status"] == "running":
            break
        time.sleep(0.01)

    resp = client.post(f"/api/jobs/{job['id']}/cancel")
    assert resp.status_code == 202

    final = _wait_until_terminal(client, job["id"])
    assert final["status"] == "cancelled"
    assert final["clips"][0]["status"] == "cancelled"


def test_preview_endpoint_returns_the_live_png_while_running_then_404s_after(client, monkeypatch):
    """Regression test for the live-preview feature: GET .../preview must
    serve the actual PNG bytes server/jobs.py's `_preview` hook wrote
    while a clip is running, with no-store caching (the same path is
    rewritten in place every tick) -- and must stop being available once
    the clip reaches a terminal state, since the file itself is deleted
    then."""
    import numpy as np
    from tool.pipeline.runner import JobCancelled

    def slow_fake_with_preview(video_path, out_path, config, models=None, raw=None, *,
                                progress=None, cancel=None, preview=None, **_kw):
        if preview is not None:
            preview(1, np.zeros((10, 10, 4), dtype=np.uint8))
        for _ in range(200):  # up to ~2s
            if cancel is not None and cancel.is_set():
                raise JobCancelled("cancelled by test")
            time.sleep(0.01)
        raise AssertionError("cancel was never observed by the fake pipeline")

    monkeypatch.setattr("server.jobs.run_clip", slow_fake_with_preview)
    job = _upload(client).json()
    for _ in range(100):
        if client.get(f"/api/jobs/{job['id']}").json()["status"] == "running":
            break
        time.sleep(0.01)

    png_bytes = None
    for _ in range(100):
        resp = client.get(f"/api/jobs/{job['id']}/preview")
        if resp.status_code == 200:
            png_bytes = resp.content
            assert resp.headers["cache-control"] == "no-store"
            break
        time.sleep(0.02)
    assert png_bytes is not None, "preview endpoint never returned 200 while the clip was running"
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    client.post(f"/api/jobs/{job['id']}/cancel")
    final = _wait_until_terminal(client, job["id"])
    assert final["status"] == "cancelled"

    resp = client.get(f"/api/jobs/{job['id']}/preview")
    assert resp.status_code == 404  # cleaned up once the clip stopped


def test_preview_endpoint_404_when_no_clip_is_running(client):
    job = _upload(client).json()
    _wait_until_terminal(client, job["id"])
    resp = client.get(f"/api/jobs/{job['id']}/preview")
    assert resp.status_code == 404


def test_delete_job_after_completion(client):
    job = _upload(client).json()
    _wait_until_terminal(client, job["id"])
    resp = client.delete(f"/api/jobs/{job['id']}")
    assert resp.status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_retry_job(client):
    job = _upload(client).json()
    _wait_until_terminal(client, job["id"])
    resp = client.post(f"/api/jobs/{job['id']}/retry")
    assert resp.status_code == 201
    retried = resp.json()
    assert retried["id"] != job["id"]
    _wait_until_terminal(client, retried["id"])


def test_download_output_file(client):
    job = _upload(client).json()
    final = _wait_until_terminal(client, job["id"])
    gif_rel = final["clips"][0]["outputs"]["primary"]
    resp = client.get(f"/api/jobs/{job['id']}/files/{gif_rel}")
    assert resp.status_code == 200
    assert resp.content == b"GIF89afake-gif-bytes"


def test_download_output_file_rejects_path_traversal(client):
    job = _upload(client).json()
    _wait_until_terminal(client, job["id"])
    resp = client.get(f"/api/jobs/{job['id']}/files/../../../etc/passwd")
    assert resp.status_code in (404, 400)


def test_download_zip(client):
    job = _upload(client).json()
    _wait_until_terminal(client, job["id"])
    resp = client.get(f"/api/jobs/{job['id']}/zip")
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert any(n.endswith(".gif") for n in zf.namelist())


def test_system_status_shape(client):
    resp = client.get("/api/system")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gpu"] is None or "used_mb" in body["gpu"]
    assert body["license"]["id"] == "CC-BY-NC-4.0"
    assert "instant" in body["presets"]["modes"]
    assert "queue_length" in body
    # Regression test for audit's UI step: the status-strip GPU/CPU toggle
    # needs to know which side device="auto" actually resolves to on this
    # host in order to show that state (see app.mjs's syncDeviceToggle).
    assert body["resolved_device"] in ("cuda", "cpu")


def test_system_status_version_matches_the_repo_root_VERSION_file():
    """Regression test for the plan's L9 finding: /api/system.version was
    a hardcoded "0.1.0" string literal in app.py, disconnected from
    whatever the repo actually was -- it now reads a real VERSION file."""
    import server.app as app_mod
    assert app_mod.VERSION == (app_mod.REPO_ROOT / "VERSION").read_text().strip()


def test_sse_events_stream_snapshot_then_terminal(client):
    """The fake pipeline finishes near-instantly, so there's a real race
    between the job completing and this test's SSE GET reaching the
    server: if the job is ALREADY terminal by the time _event_stream
    reads its snapshot, the server correctly yields just that one
    snapshot (already carrying the terminal status) and returns, with no
    separate "job_status" event -- see app.py's _event_stream and the
    frontend's onSnapshot handler, which already treats a terminal
    snapshot as sufficient on its own. So the terminal status can
    legitimately arrive via EITHER a snapshot or a job_status event;
    asserting the last event must specifically be "job_status" is what
    made this test flaky (reproduced failing ~50% of the time even
    without any pipeline changes)."""
    job = _upload(client).json()
    events = []
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line[len("data: "):]))
            ev = events[-1]
            terminal_status = ev.get("status") or (ev.get("job") or {}).get("status")
            if terminal_status in ("done", "failed", "cancelled"):
                break
    assert events[0]["type"] == "snapshot"
    last = events[-1]
    last_status = last.get("status") or (last.get("job") or {}).get("status")
    assert last_status == "done"
    # Not asserting a "progress" event ever appears: whether one does is
    # its own race (the job can finish before the fake pipeline's few
    # progress() calls get published/read at all), and is already covered
    # properly by test_jobs.py's
    # test_progress_fraction_is_monotonic_reaches_one_and_is_throttled.


def test_index_and_static_are_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<title>ClearCut</title>" in resp.content
    # Regression test for the plan's B1 finding: index.html used to be a
    # bare FileResponse with no Cache-Control, so it was left to browser
    # heuristic freshness while /static/* (served through
    # _NoCacheStaticFiles) forced revalidation -- a page reloaded after a
    # deploy could fetch a fresh index.html but an old cached app.css or
    # locales/*.json, mixing resources from different commits.
    assert resp.headers["cache-control"] == "no-cache"

    static_resp = client.get("/static/css/app.css")
    assert static_resp.status_code == 200
    assert static_resp.headers["cache-control"] == "no-cache"


def test_retry_with_bad_overrides_json_returns_400_not_500(client):
    """Regression test for the plan's B3 finding: /retry used to
    `json.loads()` its overrides directly with no try/except, so a
    malformed body crashed with a 500 instead of the same 400
    E_BAD_OVERRIDE that /api/jobs already returns for the same mistake."""
    job = _upload(client).json()
    _wait_until_terminal(client, job["id"])
    resp = client.post(f"/api/jobs/{job['id']}/retry", data={"overrides": "{not json"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == errors.E_BAD_OVERRIDE


def test_oversized_upload_body_rejected_before_being_saved(client, monkeypatch):
    """Regression test for the plan's B5 finding: the Content-Length
    middleware must reject a too-large POST /api/jobs body with 413
    before any of it is spooled to this job's directory."""
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "MAX_UPLOAD_BODY_MB", 1e-7)  # smaller than any real request body
    resp = _upload(client)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == errors.E_UPLOAD_TOO_LARGE


def test_two_uploads_with_the_same_filename_both_complete_independently(client):
    """Regression test for the plan's FE2 finding: two files uploaded in
    the same batch under the SAME original filename used to collide on
    disk (second overwrites first) and in SSE clip matching (front end
    matched by name); both must now finish with distinct outputs."""
    files = [("files", ("clip.mp4", io.BytesIO(b"first bytes"), "video/mp4")),
             ("files", ("clip.mp4", io.BytesIO(b"second bytes"), "video/mp4"))]
    resp = client.post("/api/jobs", files=files, data={"mode": "quick"})
    job = resp.json()
    final = _wait_until_terminal(client, job["id"])
    assert [c["status"] for c in final["clips"]] == ["done", "done"]
    names = [c["name"] for c in final["clips"]]
    assert names == ["clip.mp4", "clip (2).mp4"]
    outputs = [c["outputs"]["primary"] for c in final["clips"]]
    assert len(set(outputs)) == 2  # distinct output files, not one overwriting the other


def test_oversized_single_clip_becomes_a_failed_clip_with_no_bytes_kept(client, tmp_path, monkeypatch):
    """Regression test for the plan's B5 finding: a single clip streamed
    past probe.MAX_FILE_MB must become a failed clip (E_UPLOAD_TOO_LARGE)
    with no input file persisted, rather than being written to disk in
    full and only rejected afterwards deep inside JobStore.create()."""
    import server.probe as probe_mod
    monkeypatch.setattr(probe_mod, "MAX_FILE_MB", 0.000001)  # a handful of bytes
    job = _upload(client).json()
    assert job["clips"][0]["status"] == "failed"
    assert job["clips"][0]["error"]["code"] == errors.E_UPLOAD_TOO_LARGE
    assert job["clips"][0]["input"] is None
    job_dir = tmp_path / "data" / "jobs" / job["id"]
    assert not (job_dir / "inputs" / "clip.mp4").exists()

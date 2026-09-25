"""HeroExtractor FastAPI service: the browser-operated front end for
tool/pipeline's GPU chroma-background matting, running as one uvicorn
worker with one background GpuWorker thread (see jobs.py's module
docstring for why). Tailscale-internal only -- see serve.sh for how HOST
is set; this module never binds 0.0.0.0 on its own.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import signal
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import errors, presets, probe
from .jobs import JobsError, bootstrap
from .presets import ALLOWED_OVERRIDES, PRESETS, PresetError

logger = logging.getLogger("heroextractor")

class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles, but every response forces revalidation instead of
    letting the browser decide freshness on its own. Starlette's default
    sends an ETag/Last-Modified but NO Cache-Control -- browsers then fall
    back to a heuristic freshness lifetime (commonly ~10% of file age),
    so a page left open across a deploy can end up mixing resources from
    different commits (fresh index.html fetched on navigation, but a
    still-"fresh"-by-heuristic app.css or locale JSON served straight from
    disk cache with no request to this server at all) -- confirmed by
    testing: a user saw a brand-new index.html's markup rendered with an
    old app.css missing a CSS rule, and old locales/en.json missing keys
    added in the very same commit. `no-cache` (not `no-store`) still lets
    the browser cache the bytes, it just forces an If-None-Match check on
    every load, so an unchanged file is still a cheap 304 -- this trades a
    small conditional request for "resources can never silently drift out
    of sync with each other again."
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
DATA_ROOT = Path(os.environ.get("HEROEXTRACTOR_DATA_DIR", str(REPO_ROOT / "data")))
JOBS_ROOT = DATA_ROOT / "jobs"

SSE_KEEPALIVE_S = 15.0
_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})

#: Coarse cap on the WHOLE multipart body of a POST /api/jobs, checked from
#: its Content-Length header before Starlette spools any of it to disk --
#: probe.MAX_FILE_MB (500) caps a single clip, but a batch of several
#: clips (plus multipart overhead) can add up well past that, and by the
#: time handler code could count bytes itself Starlette has already
#: buffered the whole body. Larger than a single clip's own limit since a
#: multi-file batch is expected traffic, not just headroom.
MAX_UPLOAD_BODY_MB = float(os.environ.get("HEROEXTRACTOR_MAX_UPLOAD_MB", "2100"))

#: Auto-shutdown after this many seconds of uptime, once the queue is
#: idle -- 0 (default) disables it. Aimed at a single-operator deployment
#: left running from a terminal (see run.sh at this repo's parent
#: directory): a forgotten server shouldn't sit holding GPU/CPU/disk
#: indefinitely. Deliberately does NOT cut off an in-progress or queued
#: job -- see `_uptime_watchdog` below.
MAX_UPTIME_S = float(os.environ.get("HEROEXTRACTOR_MAX_UPTIME_S", "0"))

#: How often the watchdog rechecks "is the queue idle yet" once the
#: uptime deadline has passed.
_UPTIME_WATCHDOG_POLL_S = 15.0


async def _uptime_watchdog(deadline_s: float, store, worker) -> None:
    """Sleeps until `deadline_s` seconds of uptime have passed, then waits
    (polling, since GpuWorker has no "idle" event to await) until nothing
    is queued/running/waiting-for-GPU before actually shutting down --
    a 6-hour cap that killed someone's in-flight conversion would just
    trade one annoyance for a worse one. Shuts down via SIGTERM to this
    same process, which uvicorn's own signal handling turns into the
    normal graceful-shutdown path (this lifespan's `finally: worker.stop()`
    included) -- no different from an operator pressing Ctrl-C."""
    await asyncio.sleep(deadline_s)
    # This runs as a fire-and-forget asyncio.create_task (see
    # _make_lifespan) whose result nothing ever awaits -- an unhandled
    # exception here is silently swallowed by asyncio (surfaced only as a
    # "Task exception was never retrieved" warning at garbage-collection
    # time, easy to miss), which would disable auto-shutdown for the rest
    # of the process's life with no trace in the log (audit L13/L7).
    try:
        while True:
            idle = worker.job_queue.empty() and all(
                job["status"] in _TERMINAL_STATUSES for job in store.list(summary=True))
            if idle:
                break
            await asyncio.sleep(_UPTIME_WATCHDOG_POLL_S)
        logger.info("HEROEXTRACTOR_MAX_UPTIME_S=%.0f reached and queue is idle -- shutting down", deadline_s)
        os.kill(os.getpid(), signal.SIGTERM)
    except asyncio.CancelledError:
        raise  # normal shutdown path (_lifespan's finally cancels this task) -- not an error
    except Exception:
        logger.exception("uptime watchdog crashed -- HEROEXTRACTOR_MAX_UPTIME_S will not fire "
                          "again for the rest of this process's life")

#: Read once at import time rather than hardcoded (L9): a repo-root
#: VERSION file is the one place this needs bumping, instead of a stale
#: string literal baked into this module.
try:
    VERSION = (REPO_ROOT / "VERSION").read_text().strip()
except OSError:
    VERSION = "0.0.0-unknown"


#: Ceiling on the one-time torch/CUDA probe below. uvicorn runs lifespan
#: startup BEFORE binding the socket, so an unbounded await here means a
#: wedged probe (a known failure mode with a hung/reset NVIDIA driver --
#: resolve_device's own try/except catches an outright exception, but
#: cannot interrupt a hang) leaves the process never listening at all, with
#: no log line and the 6h uptime watchdog never even created (audit M7).
#: 30s is generous against the measured ~7s cold case.
_DEVICE_PROBE_TIMEOUT_S = 30.0


def _make_lifespan(jobs_root: Path):
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Pay the one-time ~7s torch/CUDA probe HERE rather than inside the
        # first POST /api/jobs (see presets.resolved_auto_device) -- run off
        # the event loop so a slow probe doesn't block startup's other work.
        # Bounded: if it hangs, log and continue rather than never binding
        # the socket -- the per-request call in build_config() will just
        # pay the same (possibly still-slow) probe again on demand, memoized
        # once it does succeed.
        try:
            await asyncio.wait_for(asyncio.to_thread(presets.resolved_auto_device),
                                    timeout=_DEVICE_PROBE_TIMEOUT_S)
        except TimeoutError:
            logger.warning("device auto-detection did not finish within %.0fs at startup -- "
                            "continuing; the first request needing it will retry",
                            _DEVICE_PROBE_TIMEOUT_S)
        store, bus, worker = bootstrap(jobs_root)
        app.state.store = store
        app.state.bus = bus
        app.state.worker = worker
        watchdog_task = None
        if MAX_UPTIME_S > 0:
            watchdog_task = asyncio.create_task(_uptime_watchdog(MAX_UPTIME_S, store, worker))
        try:
            yield
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
            worker.stop()
    return _lifespan


def create_app(data_root: Path | None = None) -> FastAPI:
    """`data_root` is exposed as a parameter (rather than only the
    HEROEXTRACTOR_DATA_DIR env var) so tests can point each app instance at
    its own tmp_path without any process-wide env-var juggling. Front-end
    static assets are NOT per-instance data -- always server/static/,
    regardless of data_root."""
    data_root = Path(data_root) if data_root is not None else DATA_ROOT
    app = FastAPI(title="ClearCut", lifespan=_make_lifespan(data_root / "jobs"))
    app.state.data_root = data_root

    # -- error handling: PresetError/probe.ProbeError/JobsError all carry
    # the same (code, detail) shape; map each to an HTTP status and the
    # {"error": {"code", "detail"}} body every front-end error path expects. --

    def _status_for(code: str) -> int:
        return {errors.E_JOB_NOT_FOUND: 404, errors.E_JOB_RUNNING: 409,
                 errors.E_UPLOAD_TOO_LARGE: 413}.get(code, 400)

    def _error_response(code: str, detail: str) -> JSONResponse:
        return JSONResponse(status_code=_status_for(code),
                             content={"error": {"code": code, "detail": detail}})

    @app.exception_handler(PresetError)
    async def _preset_error(request: Request, exc: PresetError):
        return _error_response(exc.code, exc.detail)

    @app.exception_handler(probe.ProbeError)
    async def _probe_error(request: Request, exc: probe.ProbeError):
        return _error_response(exc.code, exc.detail)

    @app.exception_handler(JobsError)
    async def _jobs_error(request: Request, exc: JobsError):
        return _error_response(exc.code, exc.detail)

    # -- upload size guard: Content-Length is checked BEFORE Starlette spools
    # the multipart body to disk/memory -- by the time handler code (or even
    # a File(...) parameter) sees anything, the whole body has already been
    # received, so counting bytes in the handler can only ever reject AFTER
    # paying that cost. Only guards POST /api/jobs; every other route here
    # has no body worth bounding this way. --

    @app.middleware("http")
    async def _limit_upload_body(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/api/jobs":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    length_mb = int(content_length) / 1e6
                except ValueError:
                    length_mb = None
                if length_mb is not None and length_mb > MAX_UPLOAD_BODY_MB:
                    return _error_response(
                        errors.E_UPLOAD_TOO_LARGE,
                        f"request body {length_mb:.0f} MB exceeds the {MAX_UPLOAD_BODY_MB:.0f} MB limit")
        return await call_next(request)

    # -- jobs --

    def _parse_overrides(overrides: str | None) -> dict | None:
        if not overrides:
            return None
        try:
            return json.loads(overrides)
        except json.JSONDecodeError as e:
            raise PresetError(errors.E_BAD_OVERRIDE, f"overrides must be valid JSON: {e}")

    async def _save_uploads(files: list[UploadFile], tmp_dir: Path
                             ) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
        """Streams each part to `tmp_dir` in chunks, capping at
        probe.MAX_FILE_MB -- previously an oversized file was written to
        disk in FULL and only rejected afterwards by validate_upload
        (inside JobStore.create(), after the move into inputs/), so the
        500MB limit was enforced only after paying for the write (and, per
        create()'s own fix, the rejected file used to be kept forever).
        Returns (uploads, rejected); `rejected` is [(name, detail)] for
        parts that hit the cap -- those never reach JobStore.create() as
        real uploads, just as failed-clip records with input=None."""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = probe.MAX_FILE_MB * 1e6
        uploads: list[tuple[str, Path]] = []
        rejected: list[tuple[str, str]] = []
        for f in files:
            name = Path(f.filename or "clip.mp4").name  # strip any path components
            dest = tmp_dir / f"{uuid.uuid4().hex}_{name}"
            written = 0
            too_large = False
            with dest.open("wb") as out:
                while chunk := await f.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        too_large = True
                        break
                    out.write(chunk)
            if too_large:
                dest.unlink(missing_ok=True)
                rejected.append((name, f"{name}: exceeds the {probe.MAX_FILE_MB:.0f} MB limit"))
            else:
                uploads.append((name, dest))
        return uploads, rejected

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request, files: list[UploadFile] = File(...),
                          mode: str = Form(...), overrides: str | None = Form(None)):
        overrides_dict = _parse_overrides(overrides)

        tmp_dir = request.app.state.data_root / "tmp" / uuid.uuid4().hex
        try:
            uploads, rejected = await _save_uploads(files, tmp_dir)
            job = request.app.state.store.create(uploads, mode, overrides_dict, rejected=rejected)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if job["status"] == "queued":
            request.app.state.worker.job_queue.put(job["id"])
        return job

    @app.get("/api/jobs")
    async def list_jobs(request: Request, limit: int = 50, offset: int = 0):
        # summary=True: the listing view never renders per-clip metrics/
        # timings/outputs or the full PipelineConfig -- a job carrying QC
        # metrics (per-frame arrays included) is not small, and returning
        # 50 of them in full on every history refresh (which the front end
        # does on every job event and language switch) was needless payload.
        jobs = request.app.state.store.list(summary=True)
        return {"jobs": jobs[offset:offset + limit], "total": len(jobs)}

    @app.get("/api/jobs/{job_id}")
    async def get_job(request: Request, job_id: str):
        job = request.app.state.store.get(job_id)
        if job is None:
            raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
        return job

    @app.post("/api/jobs/{job_id}/cancel", status_code=202)
    async def cancel_job(request: Request, job_id: str):
        return request.app.state.store.cancel(job_id)

    @app.post("/api/jobs/{job_id}/retry", status_code=201)
    async def retry_job(request: Request, job_id: str, mode: str | None = Form(None),
                         overrides: str | None = Form(None)):
        overrides_dict = _parse_overrides(overrides)
        job = request.app.state.store.retry(job_id, mode=mode, overrides=overrides_dict)
        if job["status"] == "queued":
            request.app.state.worker.job_queue.put(job["id"])
        return job

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def delete_job(request: Request, job_id: str):
        request.app.state.store.delete(job_id)
        return None

    def _sse_format(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _event_stream(request: Request, job_id: str):
        store, bus = request.app.state.store, request.app.state.bus
        # Subscribe BEFORE reading the snapshot: if the job finished in the
        # gap between a get()-first ordering and subscribing, its one-shot
        # terminal event (published exactly once, to whoever's subscribed
        # at that instant) would never reach this connection, and it would
        # sit on keepalive pings forever instead of ever reaching a
        # terminal state. Subscribing first means any event from here on
        # is queued for us even before the snapshot below is read.
        sub = bus.subscribe(job_id)
        try:
            job = store.get(job_id)
            if job is None:
                yield _sse_format({"type": "job_status", "job_id": job_id, "status": "failed",
                                    "error": {"code": errors.E_JOB_NOT_FOUND, "detail": job_id}})
                return
            yield _sse_format({"type": "snapshot", "job": job})
            if job["status"] in _TERMINAL_STATUSES:
                return
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await run_in_threadpool(sub.get, True, SSE_KEEPALIVE_S)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield _sse_format(event)
                if event.get("type") == "job_status" and event.get("status") in _TERMINAL_STATUSES:
                    break
        finally:
            bus.unsubscribe(job_id, sub)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str):
        return StreamingResponse(
            _event_stream(request, job_id), media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/files/{name:path}")
    async def job_file(request: Request, job_id: str, name: str, download: bool = False):
        job = request.app.state.store.get(job_id)
        if job is None:
            raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
        allowed = {rel for clip in job["clips"] for rel in clip["outputs"].values()}
        if name not in allowed:
            raise JobsError(errors.E_JOB_NOT_FOUND, f"{name!r} is not one of this job's output files")
        path = request.app.state.store.job_dir(job_id) / name
        if not path.is_file():
            raise JobsError(errors.E_JOB_NOT_FOUND, f"{name!r} is missing on disk")
        response = FileResponse(path)
        if download:
            # RFC 5987 percent-encoding: output filenames are routinely
            # non-ASCII (挨拶_matte.gif etc.), and a bare `filename=` would
            # mangle or reject those in several browsers.
            response.headers["Content-Disposition"] = (
                f"attachment; filename*=UTF-8''{quote(path.name)}")
        return response

    @app.get("/api/jobs/{job_id}/preview")
    async def job_preview(request: Request, job_id: str, clip: str | None = None):
        """The rolling, in-progress-only preview PNG a running clip writes
        via server/jobs.py's `_preview` hook -- NOT one of `outputs` (see
        clip["preview"]'s own docstring in jobs.py), so this is a
        dedicated endpoint rather than going through job_file's outputs
        allow-list above. `clip` (a clip id) is optional; omitted, this
        picks whichever clip is currently "running" -- the only one a
        preview is meaningful for."""
        job = request.app.state.store.get(job_id)
        if job is None:
            raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
        target = None
        for c in job["clips"]:
            if clip is not None:
                if c["id"] == clip:
                    target = c
                    break
            elif c["status"] == "running":
                target = c
                break
        if target is None or not target.get("preview"):
            raise JobsError(errors.E_JOB_NOT_FOUND, "no live preview available for this job/clip")
        path = request.app.state.store.job_dir(job_id) / target["preview"]
        if not path.is_file():
            raise JobsError(errors.E_JOB_NOT_FOUND, "preview file is missing on disk")
        # The same path is rewritten in place every tick (see _preview's
        # atomic replace) -- a cached/conditional response here would show
        # a stale frame indefinitely instead of the current one.
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/zip")
    async def job_zip(request: Request, job_id: str):
        store = request.app.state.store
        if store.get(job_id) is None:
            raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
        zpath = store.zip_path(job_id)
        response = FileResponse(zpath)
        response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(zpath.name)}"
        return response

    @app.get("/api/system")
    async def system_status(request: Request):
        store, worker = request.app.state.store, request.app.state.worker
        running = next((j["id"] for j in store.list(summary=True) if j["status"] == "running"), None)
        return {
            "gpu": probe.gpu_status(),
            "disk_free_mb": probe.disk_free_mb(request.app.state.data_root),
            "queue_length": worker.job_queue.qsize(),
            "running_job_id": running,
            # What device="auto" actually resolves to on THIS host (see
            # presets.resolved_auto_device) -- surfaced so the status-strip
            # GPU/CPU toggle can show which side "auto" is really landing
            # on when neither button is explicitly pressed, rather than
            # leaving the viewer to guess.
            "resolved_device": presets.resolved_auto_device(),
            "version": VERSION,
            "license": {"id": "CC-BY-NC-4.0",
                        "url": "https://creativecommons.org/licenses/by-nc/4.0/"},
            "presets": {"modes": sorted(PRESETS), "allowed_overrides": sorted(ALLOWED_OVERRIDES)},
            "limits": {
                "max_duration_s": probe.MAX_DURATION_S, "max_frames": probe.MAX_FRAMES,
                "max_file_mb": probe.MAX_FILE_MB, "max_long_side_px": probe.MAX_LONG_SIDE_PX,
            },
        }

    # -- static front-end (server/static/) --
    app.mount("/static", _NoCacheStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        # Same no-cache reasoning as _NoCacheStaticFiles above -- this route
        # bypasses that class entirely (it's a plain FileResponse, not
        # served through the /static mount), so it needs its own header or
        # index.html itself is exactly the resource left to heuristic
        # freshness that the class's docstring assumed was already covered.
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()

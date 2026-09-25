"""Job persistence (JobStore), the single-GPU sequential worker (GpuWorker),
and a thread-safe pub/sub for progress events (EventBus).

Design: ONE uvicorn worker process, ONE daemon GpuWorker thread pulling job
ids off a plain `queue.Queue` -- the GPU pipeline is sync Python/ffmpeg (cv2/
onnxruntime/pymatting all release the GIL for their actual work), so a
second process would only buy crash isolation for the price of IPC for
progress/cancel, and systemd's Restart=on-failure already gives crash
recovery. Models are loaded once per device and reused across jobs (loading
BiRefNet+YOLOX is the expensive part -- see runner._load_models).

Every job lives on disk as data/jobs/<job_id>/job.json (+ inputs/, outputs/)
so a server restart doesn't lose history; JobStore keeps an in-memory mirror
for fast reads (single process, so no cross-process consistency concern) but
every mutation is written to disk before it's considered to have happened.

job.json schema (kept here since this module is the only writer):

    {
      "id": str,                     # "<YYYYMMDD_HHMMSS>_<6 hex chars>"
      "created_at": str,             # ISO 8601 UTC
      "started_at": str | None,
      "finished_at": str | None,
      "status": "queued" | "waiting_gpu" | "running" | "done" | "failed" | "cancelled",
      "mode": str,                   # "instant" | "quick" | "thorough"
      "overrides": dict,             # raw per-field overrides as submitted
      "qc": bool,
      "config": dict,                # dataclasses.asdict(PipelineConfig) actually used
      "error": {"code": str, "detail": str} | None,   # set whenever status == "failed"
      "clips": [
        {
          "id": str,                 # "00", "01", ... -- stable, index-based, NOT the name
          "name": str,               # de-duplicated display name ("a.mp4", "a (2).mp4", ...)
          "input": str | None,       # path relative to the job dir, or None if never saved
          "status": "pending" | "running" | "done" | "failed" | "cancelled",
          "stage": "prepare" | "infer" | "postprocess" | "encode" | "qc" | None,
          "fraction": float,         # 0..1, monotonically non-decreasing (see STAGE_WEIGHTS)
          "frames_done": int, "frames_total": int | None,
          "timings": dict, "outputs": dict,   # outputs["primary"] is the main render
          "metrics": dict | None,    # tool.qc.metrics output, only when qc was requested
          "error": {"code": str, "detail": str} | None,
          "preview": str | None,     # rolling in-progress preview PNG, relative path; None once done.
                                     # NOTE: only ever non-None in the in-memory mirror (and therefore
                                     # in an SSE `snapshot`) while the clip is running -- save() is not
                                     # called between _preview() setting this and the per-clip `finally`
                                     # clearing it again, so job.json on disk is always "preview": null.
          "started_at": str | None, "finished_at": str | None,   # per-CLIP timestamps (see server/eta.py)
          "predicted_s": dict | None,   # eta.TimingStats.predict() output; set even for pending clips
          "eta_s": float | None,     # smoothed seconds remaining for THIS clip; None until it starts
          "elapsed_s": float | None, # seconds since this clip started; None until it starts
        },
        ...
      ],
    }

_load_from_disk() backfills clip.id/outputs.primary onto records written
before those fields existed, so an old job.json from before that schema
change still loads and matches correctly rather than crashing or silently
mismatching by name.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime
import gc
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path

from tool.pipeline import keyer
from tool.pipeline.config import PipelineConfig
from tool.pipeline.preview import encode_preview_png
from tool.pipeline.runner import ClipTooLong, JobCancelled, run_clip, _load_models

from . import errors
from . import eta
from . import probe
from .presets import PresetError, build_config, config_to_dict
from .probe import ProbeError, validate_upload

#: L6: server/ previously had no logging at all -- a GpuWorker crash (the
#: `except Exception` in GpuWorker.run below) or a periodic cleanup() run
#: left no trace anywhere. A dedicated handler is attached here rather
#: than relying on uvicorn's own logging config, since uvicorn does not
#: configure the root logger by default -- an unconfigured "heroextractor"
#: logger would otherwise silently drop every INFO/WARNING record.
logger = logging.getLogger("heroextractor")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

#: Output filename extension per encoder -- mirrors __main__.py's own
#: `ext = {...}[config.encoder]` table (the one true set of encoder
#: choices is presets.ENCODERS; this just maps each to its file suffix).
EXT_BY_ENCODER = {"supersampled_gif": ".gif", "ss_alpha_gif": ".gif",
                  "mov": ".mov", "webp": ".webp"}

#: A clip's overall progress ("fraction", 0..1) is a weighted sum across
#: these stages, in this order -- weights are rough wall-clock shares
#: measured on this repo's own dinosaur fixtures (see the plan's V0
#: measurement: infer ~25s, postprocess+encode dominated by encode's 4x
#: supersample pass). "qc" is dropped and the rest renormalized to sum to
#: 1.0 for a clip whose job has qc=False (see _stage_weights) -- runner.py
#: itself never emits a "qc" stage; jobs.py brackets tool.pipeline.
#: __main__._run_qc with synthetic ("qc", 0, 1)/("qc", 1, 1) calls through
#: the same _progress hook so QC gets its own slice of the bar instead of
#: pretending the clip finished when the render did.
#: "prepare" (第11計画 Part 1-3) is everything before the first real
#: frame -- model load, the keyer decision's frame sampling, bg stats --
#: emitted by runner.infer_clip on entry (and by _process_job itself just
#: before the model load it does ahead of run_clip), so the heartbeat has
#: a stage to publish from t=0 instead of staying silent until the first
#: "infer" tick. Its weight is small: it is a few seconds of a clip.
STAGES = ("prepare", "infer", "postprocess", "encode", "qc")
STAGE_WEIGHTS = {"prepare": 0.02, "infer": 0.53, "postprocess": 0.10, "encode": 0.30, "qc": 0.05}

#: Progress events are only actually published to the bus at most this
#: often (any earlier tick just updates the in-memory clip record) -- a
#: 900-frame clip emitting one SSE event per frame at ~30fps would fan out
#: to (frames x connected clients) JSON encodes/sec for no visible benefit
#: at browser refresh rates (L4). A stage change or a stage reaching 100%
#: is always published regardless, so the stepper/final-tick never lags.
PROGRESS_MIN_INTERVAL_S = 0.2
PROGRESS_MIN_FRAMES = 5

#: The live preview PNG is only actually re-encoded/rewritten this often
#: (2fps) -- independent of PROGRESS_MIN_INTERVAL_S/PROGRESS_MIN_FRAMES,
#: since encoding a PNG is far more expensive per tick than a JSON event
#: and this is a "what's happening right now" glance, not something that
#: benefits from every frame.
PREVIEW_MIN_INTERVAL_S = 0.5

#: How long a finished job's record (and, after RETENTION_INPUTS_DAYS, its
#: raw inputs/) is kept before automatic cleanup.
RETENTION_DAYS = 14
RETENTION_INPUTS_DAYS = 1

#: L1 GPU pre-flight: a cuda job whose PipelineConfig.device == "cuda"
#: won't be claimed off the queue while probe.gpu_status() reports less
#: than this many MB free -- this machine's GPU is routinely shared with
#: other services (vLLM/ComfyUI have been observed holding 20+ GB), and
#: starting a job anyway used to mean discovering the OOM only after
#: several minutes of real work. 0 disables the check entirely (e.g. a
#: CPU-only deployment, or a dedicated GPU where this would just be
#: needless waiting). gpu_status() returning None (nvidia-smi unavailable)
#: is treated as "can't verify" and does NOT block -- there's no VRAM
#: number to compare against, and refusing to ever run would be worse than
#: occasionally hitting the OOM this check exists to avoid.
MIN_FREE_GPU_MB = float(os.environ.get("HEROEXTRACTOR_MIN_FREE_MB", "8000"))

#: Lower threshold used INSTEAD OF MIN_FREE_GPU_MB when this worker's own
#: _models_cache already holds a Models instance for the job's device --
#: gpu_status() reports TOTAL used VRAM regardless of who holds it, so
#: without this a worker that already loaded ~7-8GB of its own models
#: mistook its own resident memory for "no room" and waited on a GPU that
#: was, from this job's point of view, already ready (confirmed: this
#: caused a real ~10 minute wait on a job that could have started
#: immediately). A freshly loading Models instance needs headroom for the
#: load itself; an already-resident one only needs inference-time working
#: memory, which is far smaller.
MIN_FREE_GPU_MB_WARM = float(os.environ.get("HEROEXTRACTOR_MIN_FREE_GPU_MB_WARM", "2000"))

#: How often a job sitting in "waiting_gpu" rechecks probe.gpu_status().
GPU_WAIT_POLL_S = 15.0

#: How long the worker sits idle (no job processed) before it drops its
#: _models_cache and frees the VRAM those sessions hold -- this server
#: shares its GPU with other services, and previously never released
#: model memory once loaded, no matter how long it sat unused. 0 disables
#: idle unloading (the model stays resident forever, as before).
IDLE_UNLOAD_S = float(os.environ.get("HEROEXTRACTOR_IDLE_UNLOAD_S", "600"))

#: Release VRAM the moment the queue goes empty after a job, rather than
#: waiting up to IDLE_UNLOAD_S -- see GpuWorker's own docstring on this
#: field. Off by default; the single-operator launcher (run.sh, at this
#: repo's parent directory) turns it on.
UNLOAD_ON_QUEUE_EMPTY = os.environ.get("HEROEXTRACTOR_UNLOAD_ON_QUEUE_EMPTY", "0") not in ("0", "", "false", "False")

#: Reliability fallback for a genuinely observed failure mode: the default
#: encoder (supersampled_gif's 4x two-pass palette quantization) measured
#: 1080-3605s under heavy contention from OTHER processes on this shared
#: machine (load average 60-97 on a 24-core box), 5-16x its usual ~220s,
#: hard-failing with E_ENCODE_TIMEOUT once (see the plan's incident record).
#: Every OTHER pipeline stage only slows 2-3x under the same contention --
#: this encoder's cost is uniquely, badly non-linear in CPU/memory-bandwidth
#: contention (palettegen's global histogram pass over 6624x4992px frames).
#: When enabled and this clip would otherwise use that encoder, a per-clip
#: check (server/probe.py's cpu_load_ratio) falls back to LOAD_FALLBACK_
#: ENCODER instead -- chosen as "webp", NOT the cheaper "ss_alpha_gif",
#: because webp keeps true 8-bit alpha (no 1-bit threshold snap), which is
#: what made ss_alpha_gif's own past rejection (a visible defect, later
#: traced to BiRefNet's OWN run-to-run nondeterminism landing exactly on
#: its hard alpha threshold -- see tool/docs/DECISIONS.md) structurally
#: impossible here: there's no threshold for that nondeterminism to flip
#: across. An explicit --encoder override is never touched -- this only
#: substitutes for the DEFAULT choice, never for one the caller asked for.
LOAD_AWARE_ENCODE = os.environ.get("HEROEXTRACTOR_LOAD_AWARE_ENCODE", "1") not in ("0", "", "false", "False")
LOAD_THRESHOLD = float(os.environ.get("HEROEXTRACTOR_LOAD_THRESHOLD", "1.5"))
LOAD_FALLBACK_ENCODER = "webp"

#: Mirrors server/app.py's own _TERMINAL_STATUSES (kept as a separate
#: constant here rather than imported, since app.py imports FROM this
#: module and a reverse import would be circular) -- used by zip_path() to
#: decide whether a job's outputs are still changing (always rebuild) or
#: settled (an existing zip can be reused as-is).
_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _clip_takes_keyer(clip: dict, config: PipelineConfig) -> bool:
    """Whether this clip will be matted by the colour-only path, decided from
    the upload-time probe. Deliberately conservative: anything unprobed or
    uncertain counts as needing the network, so a wrong guess can only cost
    time, never correctness. `runner._resolve_keyer` makes the real decision
    from the actual frames; this only predicts it well enough to skip work
    that would otherwise be wasted."""
    if config.use_keyer == "off":
        return False
    probe_info = clip.get("probe") or {}
    return keyer.probe_says_keyable(probe_info.get("bg_is_chroma_class"),
                                     probe_info.get("bg_key_saturation"))


def _job_needs_gpu(job: dict, config: PipelineConfig) -> bool:
    """False only when every not-yet-finished clip will take the keyer path."""
    pending = [c for c in job.get("clips", []) if c.get("status") in ("pending", "running")]
    return not pending or not all(_clip_takes_keyer(c, config) for c in pending)


def _new_clip_record(clip_id: str, name: str, input_rel: str | None) -> dict:
    return {
        "id": clip_id, "name": name, "input": input_rel,
        "status": "pending", "stage": None, "fraction": 0.0,
        "frames_done": 0, "frames_total": None,
        "timings": {}, "outputs": {}, "metrics": None, "error": None,
        # Live preview PNG path, relative to the job dir -- deliberately
        # NOT one of `outputs` (which get zipped and shown as final
        # results, see zip_path()): this is a rolling, in-progress-only
        # file that's deleted once the clip finishes.
        "preview": None,
        # Filled in by create() from validate_upload()'s return value
        # (dataclasses.asdict(ProbeInfo)) -- width/height/fps/frames/
        # duration_s/size_mb/bg_is_chroma_class/bg_frac_bg_like/
        # bg_key_saturation. Stays None for a clip that never validated
        # (ProbeError / rejected-oversize). bg_key_saturation is what
        # _clip_takes_keyer's prediction (and therefore the GPU-preflight
        # skip and the ETA's `keyed` pricing) hinges on.
        "probe": None,
        # Names of PipelineConfig fields the worker actually forced off for
        # THIS clip because its backdrop didn't gate as flat chroma (see
        # _wait_for_gpu_ready's neighbour, the auto-disable step in
        # _process_job) -- empty list until the worker runs, distinct from
        # None-probe (never checked) vs empty-list (checked, nothing to disable).
        "auto_disabled": [],
        # Set to the encoder actually used ONLY when LOAD_AWARE_ENCODE
        # substituted it in for the job's requested (or default) one due
        # to heavy CPU contention at the moment this clip started (see
        # LOAD_AWARE_ENCODE's own docstring) -- None otherwise, including
        # when the requested encoder was already something other than
        # supersampled_gif (this never overrides an explicit choice).
        "auto_encoder_fallback": None,
        # True when tool/pipeline/keyer.py's colour-only fast path actually
        # matted this clip (the backdrop measured flat AND saturated enough
        # to key on), False when the BiRefNet route ran, None until the clip
        # has been processed. Decided inside runner.infer_clip, which is the
        # only place that has seen the actual frames -- reported back via the
        # presence of a "key_s" entry in the timings dict this worker already
        # passes in.
        #
        # NOTE when reading a job record: this being True also means the
        # encoder was switched to ss_alpha_gif inside run_clip (the heavy
        # two-pass chain exists to mask BiRefNet's non-determinism, which a
        # keyed matte doesn't have). job["config"]["encoder"] still shows what
        # was REQUESTED; the effective config is written next to the output
        # as <name>_matte.gif.pipeline_config.json (outputs["config"]).
        "auto_keyer": None,
        # ETA fields (see server/eta.py and the plan's "待ち時間の可視化"):
        # started_at/finished_at are per-CLIP timestamps (the job already
        # has its own, but a multi-clip job's clips start/finish at
        # different times). predicted_s is eta.TimingStats.predict()'s
        # output, written for EVERY clip (including still-pending ones)
        # before the clip loop starts, so a pending clip contributes a
        # real number to job_eta_s rather than nothing. eta_s/elapsed_s
        # are the live, smoothed per-clip numbers from the last progress
        # tick; both None until the clip starts running.
        "started_at": None, "finished_at": None,
        "predicted_s": None, "eta_s": None, "elapsed_s": None,
    }


def _stage_weights(qc: bool) -> dict[str, float]:
    """STAGE_WEIGHTS as-is for a QC clip; with "qc" dropped and the rest
    renormalized to sum back to 1.0 for one that won't run QC at all --
    otherwise a non-QC clip's fraction would always cap out at 0.95."""
    if qc:
        return STAGE_WEIGHTS
    active = {s: w for s, w in STAGE_WEIGHTS.items() if s != "qc"}
    total = sum(active.values())
    return {s: w / total for s, w in active.items()}


def _stage_bases(weights: dict[str, float]) -> dict[str, float]:
    """Cumulative weight of every stage BEFORE each stage, in STAGES
    order -- e.g. {"infer": 0.0, "postprocess": 0.55, "encode": 0.65, ...}
    -- so a stage's own (done/total) only needs scaling by its own slice,
    not the stages before it."""
    bases = {}
    acc = 0.0
    for stage in STAGES:
        if stage not in weights:
            continue
        bases[stage] = acc
        acc += weights[stage]
    return bases


def _dedupe_name(seen: dict[str, int], name: str) -> str:
    """Disambiguate `name` against every name already seen in this job
    (tracked in `seen`, shared across all of a job's clips including
    rejected ones) -- e.g. two separate "clip.mp4" uploads in the same
    batch become "clip.mp4" and "clip (2).mp4". Previously a duplicate
    name's shutil.move silently overwrote the first upload's input file
    (and its output silently overwrote the first's output too), both
    under a name the front end used to match clips by -- so the second
    upload sat at "pending" forever since nothing with its now-collided
    name ever finished (FE2). Clips are now matched by `id`, not name,
    but names must still be unique for distinct files on disk."""
    count = seen.get(name, 0) + 1
    seen[name] = count
    if count == 1:
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    return f"{stem} ({count}){suffix}"


class JobsError(Exception):
    """Same (code, detail) shape as presets.PresetError/probe.ProbeError."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


class EventBus:
    """Thread-safe pub/sub keyed by job_id. Deliberately built on plain
    `queue.Queue` rather than `asyncio.Queue` + `loop.call_soon_threadsafe`:
    the publisher is a background thread (GpuWorker), and a stdlib Queue's
    blocking `get(timeout=...)` is trivially awaitable from an async SSE
    generator via a thread-pool (`anyio`/`starlette`'s `run_in_threadpool`),
    with `queue.Empty` doubling as the "send a keepalive" signal -- no
    event-loop coupling needed in this module at all, so it stays testable
    with zero FastAPI/asyncio imports."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subs: dict[str, list[queue.Queue]] = {}

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subs.get(job_id)
            if subs and q in subs:
                subs.remove(q)
                if not subs:
                    del self._subs[job_id]

    def publish(self, job_id: str, event: dict) -> None:
        with self._lock:
            subs = list(self._subs.get(job_id, ()))
        for q in subs:
            q.put(event)


class JobStore:
    def __init__(self, root: Path, bus: "EventBus | None" = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bus = bus  # only used by cancel() -- see its docstring for why
        self._jobs: dict[str, dict] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._zip_lock = threading.Lock()  # serializes zip_path()'s build step -- see its docstring
        self._load_from_disk()

    # -- paths --

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _job_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    # -- startup --

    def _load_from_disk(self) -> None:
        """Restart-recovery: any job caught mid-"running" didn't actually
        survive the restart (the GpuWorker thread that was processing it is
        gone) -- mark it failed rather than silently pretending it's still
        in flight forever. A job caught "waiting_gpu" lost nothing (no GPU
        work had actually started) -- it's demoted back to "queued" so
        bootstrap()'s own re-enqueue picks it up and it re-attempts the
        wait fresh, rather than being stuck displaying "waiting_gpu"
        forever with no worker thread polling on its behalf. "queued" jobs
        are left as-is; the caller (bootstrap()) re-enqueues them in
        created_at order once the worker exists.

        Also backfills two schema fields onto job.json records written by
        an older version of this module, so a job created before clip.id/
        outputs.primary existed can still be read/matched/downloaded
        correctly rather than crashing or silently mismatching: each clip
        gets an `id` derived from its position (matching how create() has
        always assigned them), and an `outputs.gif` key is renamed to
        `outputs.primary`."""
        for job_json in sorted(self.root.glob("*/job.json")):
            try:
                job = json.loads(job_json.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            changed = False
            for idx, clip in enumerate(job.get("clips", [])):
                if "id" not in clip:
                    clip["id"] = f"{idx:02d}"
                    changed = True
                outputs = clip.get("outputs") or {}
                if "gif" in outputs and "primary" not in outputs:
                    outputs["primary"] = outputs.pop("gif")
                    changed = True
                # Same reasoning as id/outputs.primary above: a clip record
                # from before these fields existed is missing the KEY
                # entirely (not just holding None), which raised KeyError
                # the first time something read clip["preview"] directly.
                for key in ("preview", "probe", "auto_disabled", "auto_encoder_fallback",
                            "auto_keyer",
                            "started_at", "finished_at", "predicted_s", "eta_s", "elapsed_s"):
                    if key not in clip:
                        clip[key] = [] if key == "auto_disabled" else None
                        changed = True
            if job.get("status") == "running":
                job["status"] = "failed"
                job["error"] = {"code": errors.E_SERVER_RESTART,
                                 "detail": "server restarted while this job was running"}
                job["finished_at"] = _now()
                for clip in job.get("clips", []):
                    if clip["status"] == "running":
                        clip["status"] = "failed"
                        clip["error"] = {"code": errors.E_SERVER_RESTART, "detail": ""}
                changed = True
            elif job.get("status") == "waiting_gpu":
                job["status"] = "queued"
                changed = True
            if changed:
                self._write(job)
            self._jobs[job["id"]] = job

    # -- persistence --

    def _write(self, job: dict) -> None:
        path = self._job_json_path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False))
        tmp.replace(path)

    def save(self, job: dict) -> None:
        with self._lock:
            self._jobs[job["id"]] = job
            self._write(job)

    # -- reads --
    #
    # get()/list() always return a copy.deepcopy taken under `_lock`, never
    # a live reference into self._jobs -- GpuWorker mutates a job's nested
    # clip dicts in place while processing it (see claim_queued()'s own
    # docstring for the other half of this: the worker's copy is a
    # SEPARATE object from whatever's in self._jobs until its next save()
    # call). Without this, a request thread iterating/json.dumps-ing the
    # dict this method returned could race the worker thread's in-place
    # mutation of that same object ("dictionary changed size during
    # iteration" or worse, silently wrong data).

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def list(self, *, summary: bool = False) -> list[dict]:
        """`summary=True` (used by the API's job-listing endpoint) strips
        each job's `config` and each clip's `metrics`/`timings`/`outputs`
        -- a job carrying full QC metrics (per-frame arrays included) is
        not small, and the listing view never renders any of that; only
        `GET /api/jobs/{id}` needs the full record."""
        with self._lock:
            jobs = [copy.deepcopy(j) for j in self._jobs.values()]
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        if summary:
            for job in jobs:
                job.pop("config", None)
                for clip in job.get("clips", ()):
                    clip.pop("metrics", None)
                    clip.pop("timings", None)
                    clip.pop("outputs", None)
        return jobs

    def queued_job_ids_in_created_order(self) -> list[str]:
        with self._lock:
            queued = [(j["created_at"], j["id"]) for j in self._jobs.values() if j["status"] == "queued"]
        queued.sort()
        return [job_id for _, job_id in queued]

    def cancel_event(self, job_id: str) -> threading.Event:
        with self._lock:
            ev = self._cancel_events.get(job_id)
            if ev is None:
                ev = threading.Event()
                self._cancel_events[job_id] = ev
            return ev

    def claim_queued(self, job_id: str) -> dict | None:
        """Atomically transition a job from "queued" OR "waiting_gpu" to
        "running" (a compare-and-swap under `_lock`) and hand the caller
        (GpuWorker) a private copy to mutate freely from here on -- returns
        None if it wasn't in either of those states anymore (deleted, or
        already flipped to "cancelled" by cancel() below while still
        queued/waiting).

        This copy is deliberately NOT the object stored in self._jobs:
        GpuWorker mutates its clips' status/timings/outputs in place as it
        works and only calls store.save() (which swaps the shared
        reference under the lock) at clip/stage boundaries, not on every
        micro-mutation -- so between saves, self._jobs still points at
        whatever was last written, and any concurrent get()/list() sees
        either the fully-pre-mutation or fully-post-mutation state, never
        a torn one."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] not in ("queued", "waiting_gpu"):
                return None
            job["status"] = "running"
            job["started_at"] = _now()
            self._write(job)
            return copy.deepcopy(job)

    def mark_waiting_gpu(self, job_id: str) -> dict | None:
        """CAS "queued" -> "waiting_gpu" (L1) -- called by GpuWorker the
        first time a cuda job actually has to wait for VRAM to free up (a
        job that finds the GPU already free never passes through this
        state at all), so the UI can show something more informative than
        an indefinitely "queued" job. Returns None if the job wasn't
        "queued" anymore (deleted, or cancel() already flipped it straight
        to "cancelled")."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "queued":
                return None
            job["status"] = "waiting_gpu"
            self._write(job)
            return copy.deepcopy(job)

    def cancel(self, job_id: str) -> dict:
        """Signal cancellation. A job still "queued" or "waiting_gpu" (the
        GpuWorker hasn't started actually running it yet) transitions to
        "cancelled" immediately here -- including publishing the
        clip_status/job_status events itself, since nothing else will: the
        worker's claim_queued() will simply find the job no longer in
        either of those states and return None without ever entering
        _process_job's normal event-publishing code path (see the plan's
        FE1 finding -- this used to leave a browser watching SSE stuck on
        "running" forever for exactly this case). A "running" job's clips/
        job status only flip once GpuWorker's own per-frame/per-stage
        `cancel.is_set()` checks notice and publish it themselves; this
        method only sets the event for that case, nothing else."""
        self.cancel_event(job_id).set()
        transitioned = False
        flipped_clips = []
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
            if job["status"] in ("queued", "waiting_gpu"):
                job["status"] = "cancelled"
                job["started_at"] = job["started_at"] or _now()
                job["finished_at"] = _now()
                for clip in job["clips"]:
                    if clip["status"] == "pending":
                        clip["status"] = "cancelled"
                        flipped_clips.append((clip["id"], clip["name"]))
                self._write(job)
                transitioned = True
            result = copy.deepcopy(job)
        if transitioned and self.bus is not None:
            for clip_id, name in flipped_clips:
                self.bus.publish(job_id, {"type": "clip_status", "job_id": job_id,
                                           "clip_id": clip_id, "clip": name, "status": "cancelled"})
            self.bus.publish(job_id, {"type": "job_status", "job_id": job_id, "status": "cancelled"})
        return result

    # -- writes --

    def create(self, uploads: list[tuple[str, Path]], mode: str, overrides: dict | None,
               *, rejected: list[tuple[str, str]] = ()) -> dict:
        """`uploads` is [(original_filename, tmp_path_holding_its_bytes)].
        Raises PresetError for a bad mode/overrides (no job is created --
        that's a malformed request, not a job that failed). A per-clip
        input that fails validate_upload does NOT abort job creation: that
        clip's record starts life already "failed" with its ProbeError
        code, and the job proceeds with whichever clips validated (mirrors
        __main__.py's own "one bad clip doesn't sink the batch" policy).

        `rejected` is [(original_filename, detail)] for uploads already
        refused before reaching this method at all (server/app.py's
        _save_uploads streams each part to disk and stops once it exceeds
        probe.MAX_FILE_MB, rather than writing the whole oversized file
        and only rejecting it after move()+validate_upload -- previously
        an oversized upload was fully persisted into inputs/ and then kept
        there forever, since a ProbeError there didn't delete `dest`
        either). Each becomes a failed clip with input=None; no bytes of
        it ever touch this job's directory."""
        config, qc = build_config(mode, overrides)  # raises PresetError

        job_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job_dir = self.job_dir(job_id)
        (job_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (job_dir / "outputs").mkdir(parents=True, exist_ok=True)

        seen_names: dict[str, int] = {}
        clips = []
        for original_name, tmp_path in uploads:
            safe_name = _dedupe_name(seen_names, Path(original_name).name)
            dest = job_dir / "inputs" / safe_name
            shutil.move(str(tmp_path), str(dest))
            clip = _new_clip_record(f"{len(clips):02d}", safe_name, str(dest.relative_to(job_dir)))
            try:
                # Previously discarded (bug B6) -- resolution/frame-count/
                # backdrop-class never reached the job record at all.
                clip["probe"] = dataclasses.asdict(validate_upload(dest))
            except ProbeError as e:
                clip["status"] = "failed"
                clip["error"] = {"code": e.code, "detail": e.detail}
                dest.unlink(missing_ok=True)  # don't keep a rejected upload around forever
                clip["input"] = None
            clips.append(clip)

        for original_name, detail in rejected:
            safe_name = _dedupe_name(seen_names, Path(original_name).name)
            clip = _new_clip_record(f"{len(clips):02d}", safe_name, None)
            clip["status"] = "failed"
            clip["error"] = {"code": errors.E_UPLOAD_TOO_LARGE, "detail": detail}
            clips.append(clip)

        job = {
            "id": job_id, "created_at": _now(), "started_at": None, "finished_at": None,
            "status": "queued" if any(c["status"] == "pending" for c in clips) else "failed",
            "mode": mode, "overrides": overrides or {}, "qc": qc,
            "config": config_to_dict(config),
            "clips": clips, "error": None,
        }
        if job["status"] == "failed" and job["error"] is None:
            job["error"] = {"code": errors.E_INPUT_UNREADABLE,
                             "detail": "no clip in this job passed input validation"}
            job["finished_at"] = _now()
        self.save(job)
        return job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
        if job["status"] == "running":
            raise JobsError(errors.E_JOB_RUNNING, "cancel it first")
        with self._lock:
            self._jobs.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def zip_path(self, job_id: str) -> Path:
        """Build (or reuse) a zip of every output file across this job's
        clips. A job still in flight always rebuilds -- its outputs are
        still changing, so "always correct" beats caching. A job in a
        TERMINAL status has settled outputs, so an existing zip is reused
        as-is rather than rebuilt on every download.

        Building writes to a `.zip.tmp` sibling and atomically replaces
        `out_zip` with it under `_zip_lock` -- previously this opened
        `out_zip` itself with mode "w" (truncating it in place) on every
        single call with no locking at all, so two concurrent downloads of
        the same finished job's zip could observe a torn/corrupt file
        (B9)."""
        job = self.get(job_id)
        if job is None:
            raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
        job_dir = self.job_dir(job_id)
        out_zip = job_dir / "outputs" / f"{job_id}.zip"
        if job["status"] in _TERMINAL_STATUSES and out_zip.is_file():
            return out_zip
        with self._zip_lock:
            # Re-check: another thread may have just finished building this
            # exact zip for a now-terminal job while we were waiting on the lock.
            if job["status"] in _TERMINAL_STATUSES and out_zip.is_file():
                return out_zip
            tmp_zip = out_zip.with_suffix(".zip.tmp")
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as zf:
                for clip in job["clips"]:
                    for rel in clip["outputs"].values():
                        p = job_dir / rel
                        if p.is_file():
                            zf.write(p, arcname=f"{Path(clip['name']).stem}/{p.name}")
            tmp_zip.replace(out_zip)
        return out_zip

    def retry(self, job_id: str, mode: str | None = None, overrides: dict | None = None) -> dict:
        """New job over the SAME input files (hardlinked, not copied) as
        `job_id`, defaulting to that job's own mode/overrides -- pass
        `mode`/`overrides` to retry under a different preset instead (the
        UI's "再実行(別モード)" button).

        Validates mode/overrides BEFORE touching any files: this used to
        hardlink every input first and only call build_config() (inside
        create()) afterwards, so a bad mode/override left orphaned
        `.retry-*` files in self.root with nothing to ever clean them up.
        The try/finally below is a second backstop for the rarer case of a
        mid-loop hardlink/copy failure."""
        src = self.get(job_id)
        if src is None:
            raise JobsError(errors.E_JOB_NOT_FOUND, job_id)
        effective_mode = mode or src["mode"]
        effective_overrides = overrides if overrides is not None else src["overrides"]
        build_config(effective_mode, effective_overrides)  # raises PresetError; no files touched yet

        uploads = []
        tmp_paths = []
        try:
            for clip in src["clips"]:
                if clip["input"] is None:
                    continue  # never saved (rejected upload) or already input=None from a prior failure
                src_path = self.job_dir(job_id) / clip["input"]
                if not src_path.is_file():
                    continue  # cleaned up by retention (RETENTION_INPUTS_DAYS) since the source job finished
                tmp = self.root / f".retry-{uuid.uuid4().hex}"
                try:
                    tmp.hardlink_to(src_path)
                except OSError:
                    shutil.copy2(src_path, tmp)
                tmp_paths.append(tmp)
                uploads.append((clip["name"], tmp))
            if not uploads:
                raise JobsError(errors.E_INPUTS_EXPIRED,
                                 "none of this job's input files are still available to retry")
            return self.create(uploads, effective_mode, effective_overrides)
        finally:
            # create() moves every tmp file it's handed into the new job's
            # inputs/ before returning, so this is a no-op for the common
            # (successful) case -- it only cleans up files left behind by
            # an exception raised partway through the loop above or inside
            # create() itself.
            for tmp in tmp_paths:
                tmp.unlink(missing_ok=True)

    # -- cleanup --

    def cleanup(self, now: datetime.datetime | None = None) -> tuple[int, int]:
        """Delete job records older than RETENTION_DAYS and prune the raw
        inputs/ of ones older than RETENTION_INPUTS_DAYS. Returns
        (jobs_deleted, inputs_pruned) so a caller (GpuWorker's periodic
        timer, bootstrap()'s startup pass) can log something only when
        this actually did work (B4: nothing ever called this at all before,
        so data/jobs/ grew without bound)."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        jobs_deleted = 0
        inputs_pruned = 0
        for job in list(self._jobs.values()):
            finished_at = job.get("finished_at")
            if not finished_at:
                continue
            age_days = (now - datetime.datetime.fromisoformat(finished_at)).total_seconds() / 86400
            if age_days > RETENTION_DAYS:
                try:
                    self.delete(job["id"])
                    jobs_deleted += 1
                except JobsError:
                    pass
            elif age_days > RETENTION_INPUTS_DAYS:
                inputs_dir = self.job_dir(job["id"]) / "inputs"
                if inputs_dir.is_dir():
                    shutil.rmtree(inputs_dir, ignore_errors=True)
                    inputs_pruned += 1
        return jobs_deleted, inputs_pruned


#: Matches one or more absolute-path-looking segments ending in a final
#: component, e.g. "/home/user/theory/data/jobs/xxx/outputs/foo.gif" ->
#: keeps only "foo.gif". Used by _sanitize_detail to keep server-internal
#: directory layout (job ids, usernames, repo location) out of error text
#: a browser ends up rendering.
_PATH_RE = re.compile(r"(?:/[\w.\-]+)+/([\w.\-]+)")


def _sanitize_detail(detail: str, *, max_len: int = 500) -> str:
    """Strip absolute filesystem paths down to their basename and cap
    length -- ffmpeg stderr and exception messages embed full paths
    (cmd=... -> /home/.../data/jobs/<id>/outputs/foo.gif) and can be
    arbitrarily long, and this text is rendered directly in the UI."""
    detail = _PATH_RE.sub(lambda m: m.group(1), detail)
    if len(detail) > max_len:
        detail = detail[:max_len] + "…"
    return detail


def _classify_exception(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, JobCancelled):
        return errors.E_CANCELLED, str(exc)
    if isinstance(exc, ClipTooLong):
        # Should be rare in practice -- server/probe.py's own validate_upload
        # rejects an over-long clip before a job is even created -- but this
        # is exactly the defense-in-depth path for a container (webm/
        # fragmented mov) whose frame count probe.py couldn't determine
        # up front either.
        return errors.E_INPUT_TOO_LONG, str(exc)
    if isinstance(exc, subprocess.TimeoutExpired):
        # ffmpeg_encoders.pipe_rgba_to_ffmpeg's own proc.wait(timeout=...)
        # (the final reap, AFTER the write loop's own explicit "ffmpeg
        # timeout (...)" RuntimeError below) raises this directly -- its
        # str() doesn't contain the phrase "ffmpeg timeout" the substring
        # check below looks for, so without this isinstance check it fell
        # through to the generic E_PIPELINE bucket despite genuinely being
        # a timeout.
        return errors.E_ENCODE_TIMEOUT, str(exc)
    msg = str(exc)
    lowered = msg.lower()
    if "ffmpeg timeout" in lowered:
        return errors.E_ENCODE_TIMEOUT, msg
    if ("failed to allocate" in lowered or "cuda_error_out_of_memory" in lowered
            or "cudamalloc" in lowered):
        return errors.E_CUDA_OOM, msg
    if "out of memory" in lowered and any(
            k in lowered for k in ("cuda", "cudnn", "cublas", "onnxruntime", "gpu")):
        # A bare "out of memory" with none of these markers is more likely
        # ffmpeg's own process exhausting HOST memory (its stderr, wrapped
        # in the "ffmpeg failed (exit=...)" RuntimeError, sometimes says
        # exactly this) than a GPU/CUDA allocation failure -- misclassifying
        # it as E_CUDA_OOM used to point someone at "retry on CPU" for a
        # problem CPU mode has just as much.
        return errors.E_CUDA_OOM, msg
    return errors.E_PIPELINE, msg


class GpuWorker(threading.Thread):
    """The one and only consumer of `job_queue`; processes jobs strictly
    one at a time (one GPU). Progress/clip/job events are pushed to `bus`
    as they happen so any subscribed SSE connection sees them live; job
    state is always saved to `store` first so a `GET /api/jobs/{id}` made
    right after an SSE event reflects the same thing the event just said."""

    def __init__(self, store: JobStore, bus: EventBus, job_queue: "queue.Queue[str | None]",
                 *, cleanup_interval_s: float = 3600.0, idle_unload_s: float = IDLE_UNLOAD_S,
                 stats_path: Path | None = None, unload_on_queue_empty: bool = False):
        super().__init__(daemon=True, name="GpuWorker")
        self.store = store
        self.bus = bus
        self.job_queue = job_queue
        self._models_cache: dict[str, object] = {}
        # ETA predictions (server/eta.py) -- stats_path=None (the default,
        # used by every existing test) keeps everything in memory only, so
        # no test gains a dependency on a real data/ directory.
        self.stats = eta.TimingStats(path=stats_path)
        self._stop = threading.Event()
        # B4: cleanup() previously had no caller anywhere -- data/jobs/ (and
        # RETENTION_DAYS/RETENTION_INPUTS_DAYS) were dead in practice. Timed
        # from construction, not from the first idle tick, so a worker that
        # starts already busy doesn't run its first cleanup any later than
        # one that starts idle.
        self._cleanup_interval_s = cleanup_interval_s
        self._last_cleanup_t = time.monotonic()
        # Idle GPU-memory release: this server previously held every model
        # it ever loaded resident in VRAM forever (confirmed: 7.6GB still
        # held after being idle for days) -- on a GPU shared with other
        # services that's a real, avoidable cost. 0 disables unloading.
        self._idle_unload_s = idle_unload_s
        self._last_activity_t = time.monotonic()
        # Single-operator convenience (see run.sh at the repo's parent
        # directory): unload right after a job finishes if nothing else is
        # queued, instead of waiting up to idle_unload_s for the next
        # periodic check -- a solo user watching the browser tab wants the
        # GPU back the moment their conversion is done, not up to 10
        # minutes later. Off by default (a shared/always-on deployment
        # would rather absorb the reload cost of back-to-back jobs than
        # pay a cold-start on every single one).
        self._unload_on_queue_empty = unload_on_queue_empty

    def _get_models(self, device: str):
        if device not in self._models_cache:
            self._models_cache[device] = _load_models(PipelineConfig(device=device))
        return self._models_cache[device]

    def stop(self) -> None:
        self._stop.set()
        self.job_queue.put(None)  # unblock a pending .get()

    def _maybe_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup_t < self._cleanup_interval_s:
            return
        self._last_cleanup_t = now
        try:
            jobs_deleted, inputs_pruned = self.store.cleanup()
        except Exception:
            logger.exception("periodic cleanup() failed")
            return
        if jobs_deleted or inputs_pruned:
            logger.info("cleanup: deleted %d job(s), pruned inputs/ on %d job(s)",
                        jobs_deleted, inputs_pruned)

    def _unload_models_now(self, reason: str) -> None:
        if not self._models_cache:
            return
        devices = sorted(self._models_cache)
        self._models_cache.clear()
        # Dropping the last reference triggers Models.__del__ (stops the
        # SAM2 subprocess) and releases each onnxruntime InferenceSession,
        # which frees its CUDA memory arena -- gc.collect() makes that
        # deterministic rather than "whenever CPython gets around to it".
        gc.collect()
        logger.info("released model(s) for: %s (%s)", ", ".join(devices), reason)

    def _maybe_unload_idle_models(self) -> None:
        if self._idle_unload_s <= 0 or not self._models_cache:
            return
        now = time.monotonic()
        if now - self._last_activity_t < self._idle_unload_s:
            return
        self._unload_models_now(f"idle for {now - self._last_activity_t:.0f}s")
        # Reset the clock so an idle server doesn't log this every 1s tick
        # forever after the first unload.
        self._last_activity_t = now

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.job_queue.get(timeout=1.0)
            except queue.Empty:
                self._maybe_cleanup()
                self._maybe_unload_idle_models()
                continue
            if job_id is None:
                continue
            try:
                self._process_job(job_id)
            except Exception as e:  # pragma: no cover -- defensive: a bug here must not kill the worker thread
                logger.exception("GpuWorker crashed processing job %s", job_id)
                job = self.store.get(job_id)
                if job is not None:
                    job["status"] = "failed"
                    job["error"] = {"code": errors.E_PIPELINE, "detail": f"worker crashed: {e}"}
                    job["finished_at"] = _now()
                    self.store.save(job)
                    self.bus.publish(job_id, {"type": "job_status", "job_id": job_id,
                                               "status": "failed", "error": job["error"]})
            finally:
                self._last_activity_t = time.monotonic()

    def _publish_current_status(self, job_id: str) -> None:
        """Cheap backstop: publish whatever the job's status actually is
        right now, for a caller that just found it not runnable (deleted,
        or already flipped to "cancelled" by JobStore.cancel() -- which
        publishes its own events, so this is a no-op in the common case
        and only closes a narrow race window otherwise). A no-op if
        there's no SSE subscriber."""
        current = self.store.get(job_id)
        if current is not None:
            self.bus.publish(job_id, {"type": "job_status", "job_id": job_id,
                                       "status": current["status"], "error": current.get("error")})

    def _wait_for_gpu_ready(self, job_id: str, cancel_event: threading.Event, device: str) -> bool:
        """L1 GPU pre-flight: block (nothing else can run anyway -- one
        GPU, one worker thread) until probe.gpu_status() reports at least
        the required amount free, cancellation is requested, or the job is
        cancelled/deleted out from under us. Transitions the job to
        "waiting_gpu" the first time it actually has to wait -- a job that
        finds the GPU already free never enters that state at all, going
        straight to "running" exactly as before this existed. Returns True
        once the caller's claim_queued() should proceed, False if the wait
        ended any other way (in which case JobStore.cancel() has already
        published whatever needed publishing, per its own docstring).

        Uses MIN_FREE_GPU_MB_WARM instead of MIN_FREE_GPU_MB when `device`'s
        models are already resident in _models_cache -- otherwise this
        worker's OWN already-loaded models (observed ~7-8GB) counted
        against itself as "no room", producing real multi-minute waits for
        jobs that could have started immediately (see MIN_FREE_GPU_MB_WARM's
        own docstring)."""
        threshold = MIN_FREE_GPU_MB_WARM if device in self._models_cache else MIN_FREE_GPU_MB
        announced = False
        while True:
            if cancel_event.is_set():
                return False
            status = probe.gpu_status()
            free_mb = (status["total_mb"] - status["used_mb"]) if status is not None else None
            if free_mb is None or free_mb >= threshold:
                return True
            if not announced:
                waiting = self.store.mark_waiting_gpu(job_id)
                if waiting is None:
                    return False  # deleted, or cancel() already flipped it to cancelled
                self.bus.publish(job_id, {"type": "job_status", "job_id": job_id, "status": "waiting_gpu"})
                announced = True
            if cancel_event.wait(GPU_WAIT_POLL_S):
                return False  # cancelled during the wait

    def _process_job(self, job_id: str) -> None:
        peek = self.store.get(job_id)
        if peek is None or peek["status"] != "queued":
            # Deleted, or already handled elsewhere -- most commonly
            # JobStore.cancel() flipping a still-"queued" job straight to
            # "cancelled" (and publishing that itself, see its docstring).
            self._publish_current_status(job_id)
            return

        cancel_event = self.store.cancel_event(job_id)
        peek_config = PipelineConfig(**peek["config"])
        # A clip the flat-chroma keyer will matte never touches the GPU, so
        # waiting for free VRAM would be pure delay -- and this is exactly the
        # content most likely to hit that wait, since a busy shared GPU is
        # what produced the 10-minute waiting_gpu stalls in the first place.
        # Requires EVERY clip to qualify: a mixed batch still needs the GPU
        # for its other clips.
        config_device = peek_config.device if _job_needs_gpu(peek, peek_config) else "cpu"
        if MIN_FREE_GPU_MB > 0 and config_device == "cuda":
            if not self._wait_for_gpu_ready(job_id, cancel_event, config_device):
                self._publish_current_status(job_id)
                return

        job = self.store.claim_queued(job_id)
        if job is None:
            # Closes the narrow race where some other state change landed
            # between this thread's checks above and claim_queued's own
            # lock acquisition -- cheap and a no-op if there's no SSE
            # subscriber.
            self._publish_current_status(job_id)
            return

        self.bus.publish(job_id, {"type": "job_status", "job_id": job_id, "status": "running"})

        job_dir = self.store.job_dir(job_id)
        config = PipelineConfig(**job["config"])
        any_succeeded = False

        # ETA (server/eta.py): give every still-pending clip a prediction
        # BEFORE any of them run, using the job-level config (auto-disable
        # below only turns off colour-dependent stages that aren't part of
        # the timing model, so this doesn't need to wait for that decision)
        # -- this is what lets job_eta_s already account for clips 2..N
        # while clip 1 is still running.
        for c in job["clips"]:
            if c["status"] == "pending":
                c["predicted_s"] = self.stats.predict(c.get("probe"), config, job["qc"],
                                                        keyed=_clip_takes_keyer(c, config))
        self.store.save(job)

        def _pending_predicted_total(after_clip_id: str) -> float:
            total = 0.0
            for c in job["clips"]:
                if c["id"] != after_clip_id and c["status"] == "pending" and c.get("predicted_s"):
                    total += c["predicted_s"].get("total", 0.0)
            return total

        for clip in job["clips"]:
            if clip["status"] != "pending":
                continue
            if cancel_event.is_set():
                clip["status"] = "cancelled"
                self.store.save(job)
                self.bus.publish(job_id, {"type": "clip_status", "job_id": job_id,
                                           "clip_id": clip["id"], "clip": clip["name"], "status": "cancelled"})
                continue

            clip["status"] = "running"
            self.store.save(job)
            self.bus.publish(job_id, {"type": "clip_status", "job_id": job_id,
                                       "clip_id": clip["id"], "clip": clip["name"], "status": "running"})

            input_path = job_dir / clip["input"]

            # Every colour-dependent optional stage (apply_clears,
            # strip_bg_fringe, use_trimap) assumes the backdrop is a flat
            # chroma colour -- silently wrong on a clip whose probe (upload
            # time, server/probe.py) says otherwise. Per-CLIP, not per-job:
            # a batch can mix chroma and non-chroma clips.
            clip_config = config
            probe_info = clip.get("probe") or {}
            if probe_info.get("bg_is_chroma_class") is False:
                to_disable = [f for f in ("apply_clears", "strip_bg_fringe", "use_trimap")
                              if getattr(config, f)]
                if to_disable:
                    clip_config = dataclasses.replace(clip_config, **{f: False for f in to_disable})
                    clip["auto_disabled"] = to_disable
                    logger.info("job %s clip %s: backdrop is not flat chroma "
                                "(frac_bg_like=%.3f) -- disabled %s",
                                job_id, clip["id"], probe_info.get("bg_frac_bg_like") or 0.0, to_disable)

            # LOAD_AWARE_ENCODE (see its own docstring): only ever
            # substitutes for the DEFAULT heavy encoder, never for an
            # explicit override -- a caller who asked for supersampled_gif
            # by name gets exactly that, contention or not. Gated on
            # PROVENANCE ("encoder" not requested in this job's overrides),
            # not on the encoder's current VALUE (audit M5): a value-only
            # check can't tell "still at the default" from "the caller
            # explicitly asked for supersampled_gif by name", so an explicit
            # request used to get silently swapped anyway -- confirmed the
            # existing regression test only caught this because it happened
            # to use "mov" rather than "supersampled_gif" as its explicit
            # example.
            #
            # Also skipped for a clip the keyer will matte (audit M4): the
            # keyer never uses the heavy two-pass encoder this fallback
            # exists to avoid in the first place (run_clip swaps it to
            # ss_alpha_gif on its own, see runner.py), so falling back to
            # webp here only means the SAME clip's output format flips
            # between .gif and .webp depending on unrelated machine load.
            if (LOAD_AWARE_ENCODE and not clip_config.encoder_explicit
                    and clip_config.encoder == "supersampled_gif"
                    and not _clip_takes_keyer(clip, clip_config)):
                load_ratio = probe.cpu_load_ratio()
                if load_ratio is not None and load_ratio > LOAD_THRESHOLD:
                    clip_config = dataclasses.replace(clip_config, encoder=LOAD_FALLBACK_ENCODER)
                    clip["auto_encoder_fallback"] = LOAD_FALLBACK_ENCODER
                    logger.info("job %s clip %s: CPU load ratio %.2f > %.2f -- "
                                "falling back to encoder=%s instead of supersampled_gif",
                                job_id, clip["id"], load_ratio, LOAD_THRESHOLD, LOAD_FALLBACK_ENCODER)

            output_path = job_dir / "outputs" / (
                Path(clip["name"]).stem + "_matte" + EXT_BY_ENCODER[clip_config.encoder])

            clip["started_at"] = _now()
            # Recompute with clip_config now that auto-disable is known --
            # slightly more accurate than the job-level prediction written
            # before the loop, though the two rarely differ (see above).
            clip["predicted_s"] = self.stats.predict(clip.get("probe"), clip_config, job["qc"],
                                                       keyed=_clip_takes_keyer(clip, clip_config))
            clip_eta = eta.ClipEta(clip["predicted_s"])
            heartbeat_stop = threading.Event()

            weights = _stage_weights(job["qc"])
            bases = _stage_bases(weights)
            throttle = {"last_pub_t": 0.0, "last_done": -1, "last_stage": None}
            preview_path = job_dir / "outputs" / f"{Path(clip['name']).stem}_preview.png"
            preview_state = {"last_pub_t": 0.0, "seq": 0}

            def _preview(done, frame, _clip=clip, _job_id=job_id, _path=preview_path,
                         _state=preview_state):
                now = time.monotonic()
                if now - _state["last_pub_t"] < PREVIEW_MIN_INTERVAL_S:
                    return
                _state["last_pub_t"] = now
                _state["seq"] += 1
                tmp = _path.with_suffix(".png.tmp")
                try:
                    png_bytes = encode_preview_png(frame)
                    tmp.write_bytes(png_bytes)
                    tmp.replace(_path)  # atomic -- a GET never sees a half-written file
                except Exception:
                    logger.exception("preview encode/write failed for job %s clip %s", _job_id, _clip["id"])
                    # write_bytes() can fail after creating the file (e.g.
                    # disk full mid-write) -- clean up the partial .tmp so
                    # it doesn't outlive this clip (audit A2; the per-clip
                    # finally below only unlinks the final .png path).
                    tmp.unlink(missing_ok=True)
                    return
                _clip["preview"] = str(_path.relative_to(job_dir))
                self.bus.publish(_job_id, {"type": "preview", "job_id": _job_id,
                                            "clip_id": _clip["id"], "seq": _state["seq"]})

            def _eta_fields(_clip=clip):
                elapsed = clip_eta.elapsed()
                remaining = clip_eta.remaining()
                _clip["elapsed_s"] = elapsed
                _clip["eta_s"] = remaining
                return elapsed, remaining, remaining + _pending_predicted_total(_clip["id"])

            def _progress(stage, done, total, _clip=clip, _job_id=job_id):
                if stage == "done":
                    # runner.py's own terminal marker -- not one of STAGES,
                    # and this clip's own clip_status event (published once
                    # run_clip returns/raises, below) already covers "it
                    # finished"; publishing it as a "stage" here is exactly
                    # what used to reset the front end's stepper to all-
                    # pending, since it didn't recognize "done" as a stage.
                    return
                clip_eta.on_tick(stage, done, total)
                weight = weights.get(stage, 0.0)
                frac_in_stage = min(1.0, max(0.0, (done / total) if total else 0.0))
                candidate = bases.get(stage, 0.0) + weight * frac_in_stage
                _clip["fraction"] = max(_clip.get("fraction", 0.0), candidate)
                _clip["stage"] = stage
                _clip["frames_done"] = done
                _clip["frames_total"] = total

                now = time.monotonic()
                is_final_tick = bool(total) and done >= total
                should_publish = (
                    stage != throttle["last_stage"] or is_final_tick
                    or now - throttle["last_pub_t"] >= PROGRESS_MIN_INTERVAL_S
                    or done - throttle["last_done"] >= PROGRESS_MIN_FRAMES
                )
                if not should_publish:
                    return
                throttle.update(last_pub_t=now, last_done=done, last_stage=stage)
                elapsed_s, eta_s, job_eta_s = _eta_fields()
                self.bus.publish(_job_id, {"type": "progress", "job_id": _job_id,
                                            "clip_id": _clip["id"], "clip": _clip["name"],
                                            "stage": stage, "done": done, "total": total,
                                            "fraction": _clip["fraction"],
                                            "elapsed_s": elapsed_s, "eta_s": eta_s,
                                            "job_eta_s": job_eta_s, "heartbeat": False})

            def _heartbeat_loop(_clip=clip, _job_id=job_id, _stop=heartbeat_stop):
                # Ticks stop entirely once ffmpeg has every frame and is
                # off doing its own two-pass palette quantization (see
                # ClipEta's encode-tail handling) -- without this, elapsed_s
                # would visibly freeze on the front end for however long
                # that tail takes (measured: several minutes), even though
                # real time is passing. 2s period, independent of
                # PROGRESS_MIN_INTERVAL_S (that throttle is about not
                # over-publishing genuine ticks, not about pacing a clock).
                while not _stop.wait(2.0):
                    if throttle["last_stage"] is None:
                        continue
                    elapsed_s, eta_s, job_eta_s = _eta_fields()
                    self.bus.publish(_job_id, {"type": "progress", "job_id": _job_id,
                                                "clip_id": _clip["id"], "clip": _clip["name"],
                                                "stage": _clip["stage"], "done": _clip["frames_done"],
                                                "total": _clip["frames_total"], "fraction": _clip["fraction"],
                                                "elapsed_s": elapsed_s, "eta_s": eta_s,
                                                "job_eta_s": job_eta_s, "heartbeat": True})

            heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True,
                                                 name=f"eta-heartbeat-{clip['id']}")
            heartbeat_thread.start()

            try:
                # `timings=clip["timings"]` -- run_clip writes each stage's
                # elapsed time into this dict AS THAT STAGE FINISHES (see
                # its own docstring), so a clip that fails partway through
                # (e.g. an encode timeout after a perfectly normal infer)
                # keeps the earlier stages' numbers in job.json instead of
                # the previous all-or-nothing behaviour, where a failed
                # clip's timings stayed `{}` and a 983s failure was
                # indistinguishable from a 5s one.
                # For a predicted-keyer clip, pass a zero-arg FACTORY
                # rather than models=None (audit M3): infer_clip only calls
                # it if the real frame-level check unexpectedly declines the
                # fast path, so BiRefNet/YOLOX still never touch GPU memory
                # for the common case. The earlier models=None form was
                # itself correct on a misprediction (infer_clip loads its
                # own Models when passed None) but bypassed self._get_models'
                # cache entirely -- a batch where several clips mispredict
                # in a row reloaded the network from scratch (3-5s each)
                # every time instead of loading it once and reusing it, and
                # never populated _models_cache for a later job to warm-start
                # from either.
                # Stage exists from here on, so the heartbeat starts
                # publishing elapsed time even while _get_models loads the
                # network (first job after a restart) -- see STAGES.
                _progress("prepare", 0, 1)
                if _clip_takes_keyer(clip, clip_config):
                    models = lambda _dev=clip_config.device: self._get_models(_dev)
                else:
                    models = self._get_models(clip_config.device)
                run_clip(input_path, output_path, clip_config, models=models,
                         progress=_progress, cancel=cancel_event, max_frames=probe.MAX_FRAMES,
                         preview=_preview, timings=clip["timings"])
                clip["outputs"]["primary"] = str(output_path.relative_to(job_dir))
                cfg_json = output_path.with_suffix(output_path.suffix + ".pipeline_config.json")
                if cfg_json.is_file():
                    clip["outputs"]["config"] = str(cfg_json.relative_to(job_dir))
                if job["qc"]:
                    from tool.pipeline.__main__ import _run_qc
                    _progress("qc", 0, 1)
                    t_qc0 = time.monotonic()
                    clip["metrics"] = _run_qc(input_path, output_path)
                    clip["timings"]["qc_s"] = time.monotonic() - t_qc0
                    _progress("qc", 1, 1)
                    sheet = output_path.with_suffix(output_path.suffix + ".qc_contact_sheet.png")
                    if sheet.is_file():
                        clip["outputs"]["contact_sheet"] = str(sheet.relative_to(job_dir))
                    metrics_json = output_path.with_suffix(".qc.json")
                    if metrics_json.is_file():
                        clip["outputs"]["metrics"] = str(metrics_json.relative_to(job_dir))
                clip["status"] = "done"
                any_succeeded = True
            except JobCancelled as e:
                clip["status"] = "cancelled"
                # _sanitize_detail's own docstring says this text "is
                # rendered directly in the UI" -- the generic Exception
                # branch below already goes through it; this one didn't,
                # so a cancel message embedding an absolute path (e.g. from
                # ffmpeg_encoders' cancel-during-frame-write text) could
                # leak server-internal paths to the browser.
                clip["error"] = {"code": errors.E_CANCELLED, "detail": _sanitize_detail(str(e))}
            except Exception as e:
                code, detail = _classify_exception(e)
                clip["status"] = "failed"
                clip["error"] = {"code": code, "detail": _sanitize_detail(detail)}
            finally:
                # infer_clip records key_s only on the colour-only path, so
                # its presence is the authoritative answer to "which matting
                # route actually ran" -- the server can't decide that itself
                # (half the gate, keyer.is_safe, needs frames the worker
                # never decodes). Set in `finally`, not just the success
                # path (audit M6): a clip that keyed fine and then died
                # later (e.g. an encode timeout) still has "key_s" in its
                # timings dict, so leaving auto_keyer at None for it -- the
                # schema's own documented meaning for "not yet processed" --
                # would misreport a clip that WAS processed, just
                # unsuccessfully.
                clip["auto_keyer"] = "key_s" in clip["timings"]
                # The live preview is only meaningful WHILE this clip is
                # in flight -- once it's done/failed/cancelled, either the
                # real output (outputs["primary"]) or nothing at all is
                # what should be shown, so the rolling preview file is
                # removed rather than left as a stale, orphaned PNG.
                clip["preview"] = None
                preview_path.unlink(missing_ok=True)
                preview_path.with_suffix(".png.tmp").unlink(missing_ok=True)
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=5.0)
                clip["finished_at"] = _now()
                clip["eta_s"] = 0.0
                # ok=False for encode_s specifically: run_clip writes a
                # PARTIAL encode_s even when the clip failed/cancelled
                # mid-encode (its own docstring), which would otherwise
                # teach the coefficient "encode took 3 seconds" from a
                # clip that never actually finished encoding.
                self.stats.update(clip_config, clip.get("probe"), clip["timings"],
                                  ok=clip["status"] == "done", keyed="key_s" in clip["timings"])
            # clip["stage"] is deliberately left at whatever the last real
            # progress stage was (FE5): overwriting it to "done" here (as
            # this used to) isn't one of STAGE_ORDER on the front end
            # either, and had the same stepper-reset effect as runner's own
            # "done" marker above.
            self.store.save(job)
            self.bus.publish(job_id, {
                "type": "clip_status", "job_id": job_id, "clip_id": clip["id"], "clip": clip["name"],
                "status": clip["status"], "timings": clip["timings"],
                "outputs": clip["outputs"], "metrics": clip.get("metrics"), "error": clip.get("error"),
                "started_at": clip["started_at"], "finished_at": clip["finished_at"],
                "predicted_s": clip["predicted_s"],
            })

        if cancel_event.is_set() and not any_succeeded:
            job["status"] = "cancelled"
        elif any_succeeded or not any(c["status"] != "pending" for c in job["clips"]):
            # (the `or` arm only fires for an empty-clips edge case; kept
            # explicit rather than defaulting to "failed" for a job with
            # no clips to run at all)
            job["status"] = "done" if job["clips"] else "failed"
        else:
            job["status"] = "failed"

        if job["status"] == "failed":
            # B8: this used to leave job["error"] as None on a failed job --
            # each clip carried its own error, but nothing summarized it at
            # the job level, so the UI's `if (job.error)` banner never fired
            # and a failure reason never reached the screen at all. If every
            # failed clip agrees on one code, surface that; otherwise fall
            # back to the generic pipeline code rather than picking one
            # arbitrarily.
            failed_clips = [c for c in job["clips"] if c["status"] == "failed"]
            codes = {c["error"]["code"] for c in failed_clips if c.get("error")}
            job["error"] = {
                "code": codes.pop() if len(codes) == 1 else errors.E_PIPELINE,
                "detail": (f"{len(failed_clips)} of {len(job['clips'])} clip(s) failed"
                           if job["clips"] else "job has no clips to process"),
            }
        job["finished_at"] = _now()
        self.store.save(job)
        if job["status"] == "failed":
            logger.warning("job %s failed: %s", job_id, job.get("error"))
        else:
            logger.info("job %s finished: %s", job_id, job["status"])
        self.bus.publish(job_id, {"type": "job_status", "job_id": job_id, "status": job["status"],
                                   "error": job.get("error")})
        if self._unload_on_queue_empty and self.job_queue.empty():
            self._unload_models_now("queue is empty after finishing a job")


def bootstrap(data_root: Path) -> tuple[JobStore, EventBus, GpuWorker]:
    """Wire up a JobStore + EventBus + started GpuWorker, re-enqueuing any
    job that was "queued" (not yet started) when the server last stopped,
    in the order it was originally created. Also runs one cleanup() pass
    immediately (B4) rather than waiting up to GpuWorker's own
    cleanup_interval_s for the first one -- a server that's restarted
    often (e.g. during development) would otherwise rarely go long enough
    between restarts to ever trigger the worker's own periodic check."""
    bus = EventBus()
    store = JobStore(data_root, bus=bus)  # cancel() needs the bus to publish; see its docstring
    jobs_deleted, inputs_pruned = store.cleanup()
    if jobs_deleted or inputs_pruned:
        logger.info("startup cleanup: deleted %d job(s), pruned inputs/ on %d job(s)",
                    jobs_deleted, inputs_pruned)
    job_queue: "queue.Queue[str | None]" = queue.Queue()
    worker = GpuWorker(store, bus, job_queue, stats_path=store.root.parent / "timing_stats.json",
                       unload_on_queue_empty=UNLOAD_ON_QUEUE_EMPTY)
    worker.start()
    for job_id in store.queued_job_ids_in_created_order():
        job_queue.put(job_id)
    return store, bus, worker

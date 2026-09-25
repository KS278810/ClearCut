"""Upload validation (cv2-based clip probing) and system status (GPU/disk)
for the server. Kept separate from presets.py: that module validates
PROCESSING options (mode/overrides), this one validates the INPUT file
itself and reports on shared resources the UI needs to show regardless of
any particular job (GPU free memory, queue length's disk footprint).
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import errors

#: Long edge each sampled frame is downscaled to before the backdrop-class
#: check -- estimate_bg_stats only reads a border RING, so full resolution
#: buys nothing here; this keeps the whole check (a handful of decodes +
#: resizes) comfortably under the sub-second budget an upload-time check
#: needs (it must finish before the UI can warn, well before the 10+
#: minute job it's warning about even starts).
_BG_CLASS_SAMPLE_LONG_EDGE = 480
_BG_CLASS_SAMPLE_FRAMES = 5
#: estimate_bg_stats' own default ring width (24px) was tuned at that
#: resolution -- scaling it down with the frame keeps "24px of border" a
#: consistent FRACTION of the frame either way, not a suddenly-thicker
#: relative ring at the smaller sample size (which would let more interior
#: subject pixels leak into the ring and skew the classification).
_BG_CLASS_DEFAULT_RING = 24

#: Accepted upload extensions (matches what ffmpeg_encoders'/OpenCV's
#: decode path is actually exercised against in this repo's fixtures).
ALLOWED_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})

#: A clip beyond either limit would blow well past what any of this
#: pipeline's encoders can realistically finish inside their own timeouts
#: (see ffmpeg_encoders._FFMPEG_TIMEOUT_S) -- reject it up front with a
#: clear reason instead of letting it fail confusingly deep in a job.
#: The two are meant to agree (60s @ 30fps = 1800 frames) -- they used to
#: be 60.0/900 (i.e. 900 frames = 30s, silently halving the advertised
#: duration limit for any 30fps+ clip); MAX_FRAMES alone was always the
#: tighter bound in practice, which is exactly the inconsistency this
#: fixes.
MAX_DURATION_S = 60.0
MAX_FRAMES = 1800
MAX_FILE_MB = 500.0
MAX_LONG_SIDE_PX = 4096

#: frames * megapixels -- an independent guard on the thing that actually
#: bounds infer_clip's peak memory (RGBA + source-gray held per frame, plus
#: run_clip's own `[f.copy() for f in raw["frames"]]`, roughly 3x a single
#: frame's bytes), rather than frame COUNT or duration alone. 1800 frames
#: at a small resolution is cheap; 1800 frames at 4096x4096 is not, and the
#: existing MAX_FRAMES/MAX_LONG_SIDE_PX checks don't catch that combination.
#: 3800 MPix-frames -- e.g. 1800 frames at 1656x1248 (~2.07 MPix) = ~3730 --
#: keeps this repo's own dinosaur and real-footage fixtures well within budget
#: while still bounding worst-case peak RAM on this machine (125 GB) to a
#: few tens of GB rather than unbounded.
MAX_MEGAPIXEL_FRAMES = 3800.0


class ProbeError(Exception):
    """Same shape as presets.PresetError (`code` + `detail`) -- kept as its
    own class rather than importing that one across modules, since input
    validation and processing-option validation are independent concerns
    that happen to need the same two attributes."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclasses.dataclass(frozen=True)
class ProbeInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    size_mb: float
    # Whether the clip's own backdrop looks like a flat chroma colour (see
    # _probe_bg_class below) -- None when the check itself couldn't run
    # (never raises on its own; a probe failure just means "unknown", not
    # "reject the upload"). Appended fields with defaults so existing
    # positional constructions (fake_server.py, test fixtures) still work.
    bg_is_chroma_class: bool | None = None
    bg_frac_bg_like: float | None = None
    # Backdrop chroma saturation (raw Cb/Cr levels from neutral). Recorded so
    # the worker can tell, BEFORE starting a job, whether the flat-chroma
    # keyer will handle a clip -- and therefore whether it needs the GPU at
    # all (see tool/pipeline/keyer.probe_says_keyable).
    bg_key_saturation: float | None = None


def _ffprobe_frames(path: Path) -> int | None:
    """Fallback frame count via ffprobe's packet counter, for containers
    (webm and some fragmented mov files, confirmed by testing against real
    VP9 exports) where cv2's CAP_PROP_FRAME_COUNT comes back 0 or -1 --
    without this, `duration_s = frames/fps` computes to 0.0 and BOTH length
    checks below silently pass regardless of the clip's real length. Slower
    than the cv2 property read (this makes ffprobe actually decode/count
    packets), so it's only ever used as a fallback, not the primary path."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    text = result.stdout.strip()
    return int(text) if text.isdigit() else None


def _probe_bg_class(path: Path, frames: int) -> tuple[bool | None, float | None, float | None]:
    """Best-effort check of whether this clip's backdrop looks like a flat
    chroma colour -- this pipeline was designed and validated against
    exactly that (see tool/pipeline/'s "chroma-background pipeline"
    docstrings), and every colour-dependent optional stage (apply_clears,
    strip_bg_fringe, use_trimap) silently assumes it. A natural-background
    clip slipped through with no gate anywhere catching it (confirmed by
    testing: a forest-background clip ran the full ~15 minute pipeline
    with despill/temporal-smoothing tuned for a backdrop that isn't
    flat). Runs at UPLOAD time specifically so a warning can reach the UI
    before that run ever starts, not after.

    Never raises -- a probe failure here means "unknown", not "reject the
    upload"; the actual pipeline run has its own fallback (runner.py's
    use_trimap gate) regardless of what this reports."""
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, None, None
        try:
            sample_idxs = sorted(set(
                min(frames - 1, round(k * frames / _BG_CLASS_SAMPLE_FRAMES))
                for k in range(_BG_CLASS_SAMPLE_FRAMES)
            )) if frames > 0 else [0]
            sampled = []
            downscale = 1.0  # the ring width must shrink by the SAME factor as the frame
            for idx in sample_idxs:
                if idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = cap.read()
                if not ok:
                    continue  # a seek/read miss just means fewer samples, not a failure
                h, w = bgr.shape[:2]
                long_edge = max(h, w)
                if long_edge > _BG_CLASS_SAMPLE_LONG_EDGE:
                    downscale = _BG_CLASS_SAMPLE_LONG_EDGE / long_edge
                    bgr = cv2.resize(bgr, (max(1, round(w * downscale)), max(1, round(h * downscale))),
                                      interpolation=cv2.INTER_AREA)
                sampled.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

        if not sampled:
            return None, None, None

        from tool.qc.metrics import estimate_bg_stats
        h0, w0 = sampled[0].shape[:2]
        sampled = [f for f in sampled if f.shape[:2] == (h0, w0)]  # a mid-stream seek glitch shouldn't crash the stack()
        if not sampled:
            return None, None, None
        ring = max(4, min(round(_BG_CLASS_DEFAULT_RING * downscale), min(h0, w0) // 2 or 1))
        stats = estimate_bg_stats(np.stack(sampled, 0), ring=ring)
        saturation = float(np.hypot(stats.mu_cb - 128.0, stats.mu_cr - 128.0))
        return stats.is_chroma_class, stats.frac_bg_like, saturation
    except Exception:
        return None, None, None


def validate_upload(path: str | Path) -> ProbeInfo:
    """Raise ProbeError if `path` isn't a clip this pipeline should attempt;
    otherwise return its basic properties for the job record."""
    path = Path(path)

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ProbeError(errors.E_INPUT_EXT,
                          f"{path.name}: extension must be one of "
                          f"{sorted(ALLOWED_EXTENSIONS)}, got {path.suffix!r}")

    size_mb = path.stat().st_size / 1e6
    if size_mb > MAX_FILE_MB:
        raise ProbeError(errors.E_INPUT_TOO_LARGE,
                          f"{path.name}: {size_mb:.0f} MB exceeds the {MAX_FILE_MB:.0f} MB limit")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ProbeError(errors.E_INPUT_UNREADABLE, f"{path.name}: cv2.VideoCapture couldn't open it")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
    finally:
        cap.release()

    if not ok or width <= 0 or height <= 0:
        raise ProbeError(errors.E_INPUT_UNREADABLE,
                          f"{path.name}: opened but no readable video frame (width={width}, height={height})")

    if frames <= 0:
        # cv2 couldn't report a frame count (webm/VP9 and some fragmented
        # mov containers do this) -- without a real count, duration_s below
        # computes to 0.0 and both length checks would silently pass no
        # matter how long the clip actually is.
        frames = _ffprobe_frames(path) or 0
        if frames <= 0:
            raise ProbeError(errors.E_INPUT_UNREADABLE,
                              f"{path.name}: frame count unknown (cv2 and ffprobe both failed to report it)")

    duration_s = (frames / fps) if fps > 0 else 0.0
    if duration_s > MAX_DURATION_S or frames > MAX_FRAMES:
        raise ProbeError(errors.E_INPUT_TOO_LONG,
                          f"{path.name}: {duration_s:.1f}s / {frames} frames exceeds the "
                          f"{MAX_DURATION_S:.0f}s / {MAX_FRAMES}-frame limit")

    if max(width, height) > MAX_LONG_SIDE_PX:
        raise ProbeError(errors.E_INPUT_TOO_LARGE,
                          f"{path.name}: {width}x{height} exceeds the {MAX_LONG_SIDE_PX}px long-side limit")

    megapixel_frames = frames * width * height / 1e6
    if megapixel_frames > MAX_MEGAPIXEL_FRAMES:
        raise ProbeError(errors.E_INPUT_TOO_LARGE,
                          f"{path.name}: {frames} frames at {width}x{height} "
                          f"(~{megapixel_frames:.0f} MPix-frames) exceeds the "
                          f"{MAX_MEGAPIXEL_FRAMES:.0f} MPix-frame memory budget")

    bg_is_chroma_class, bg_frac_bg_like, bg_key_saturation = _probe_bg_class(path, frames)
    return ProbeInfo(width=width, height=height, fps=fps, frames=frames,
                      duration_s=duration_s, size_mb=size_mb,
                      bg_is_chroma_class=bg_is_chroma_class, bg_frac_bg_like=bg_frac_bg_like,
                      bg_key_saturation=bg_key_saturation)


#: How long a gpu_status() result is reused before forking nvidia-smi
#: again -- the system strip polls this every few seconds per connected
#: browser, and commit 9's GPU pre-flight wait polls it every 15s from the
#: worker thread on top of that; none of those callers need sub-second
#: freshness (L5).
GPU_STATUS_CACHE_S = 2.0

_gpu_status_cache: dict = {"t": float("-inf"), "value": None}


def gpu_status() -> dict | None:
    """{"used_mb", "total_mb", "name"} for the first GPU, or None if
    nvidia-smi isn't available/fails (e.g. a CPU-only deployment) -- the
    system strip in the UI treats None as "unknown", not an error.

    Memoized for GPU_STATUS_CACHE_S: every caller shares one cache, keyed
    on wall-clock time only (there's exactly one GPU status worth knowing
    on this machine, not one per caller)."""
    now = time.monotonic()
    if now - _gpu_status_cache["t"] < GPU_STATUS_CACHE_S:
        return _gpu_status_cache["value"]

    value = _query_gpu_status()
    _gpu_status_cache["t"] = now
    _gpu_status_cache["value"] = value
    return value


def _query_gpu_status() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if not out:
        return None
    used, total, name = (p.strip() for p in out[0].split(",", 2))
    try:
        return {"used_mb": int(used), "total_mb": int(total), "name": name}
    except ValueError:
        return None


def cpu_load_ratio() -> float | None:
    """1-minute load average divided by CPU core count -- 1.0 means "as
    many runnable processes as cores, on average, over the last minute".
    Used by jobs.py to fall back to a lighter encoder under heavy
    contention from OTHER processes on this shared machine (this project's
    own GPU-bound stages don't drive load average up the way ffmpeg's
    CPU-bound palette quantization does -- see the plan's incident record
    of a 3605s encode timeout at a load average of 60-97 on this 24-core
    box, ~16x its usual ~220s). None if unavailable (os.getloadavg() is
    POSIX-only; this project targets Linux, but a graceful None here keeps
    a hypothetical other platform from crashing rather than just skipping
    the fallback)."""
    try:
        load1, _, _ = os.getloadavg()
    except OSError:
        return None
    cores = os.cpu_count() or 1
    return load1 / cores


def disk_free_mb(path: str | Path) -> float | None:
    """Free space (MB) on the filesystem holding `path` (created if
    missing, so a not-yet-created data dir can still be checked)."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(path).free / 1e6
    except OSError:
        return None

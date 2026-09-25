"""ffmpeg-based encoders for RGBA frame streams (MOV with alpha, GIF).

Encoders consume an iterator of raw ``bytes`` — one RGBA frame each, row-major
``H×W×4`` uint8.

This tool ships no ffmpeg binary and does not depend on imageio-ffmpeg, whose
auto-downloaded build is GPL (bundles libx264/libx265, etc.). Use a system
ffmpeg on PATH — ideally a LGPL build such as BtbN's
``ffmpeg-master-latest-*-lgpl-shared`` (github.com/BtbN/FFmpeg-Builds/releases).
This pipeline only encodes ProRes 4444 (MOV) and GIF, neither of which needs a
GPL-only codec, so an LGPL build loses no functionality here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


class EncodeCancelled(Exception):
    """Raised by pipe_rgba_to_ffmpeg when `cancel` is set mid-stream (server
    job cancellation) -- distinct from a genuine ffmpeg failure so callers
    (runner.py) can tell "the user cancelled" from "ffmpeg errored"."""


def _resolve_ffmpeg() -> str:
    env = os.environ.get("BGREMOVE_FFMPEG")
    if env and shutil.which(env):
        return env
    return shutil.which("ffmpeg") or "ffmpeg"


#: The ffmpeg binary used by every encoder in this module.
FFMPEG: str = _resolve_ffmpeg()

#: Transparent-GIF paletteuse settings.
DEFAULT_GIF_ALPHA_THRESHOLD = 96
DEFAULT_GIF_DITHER = "none"

#: ffmpeg stdin-stream timeout (seconds) for encoding one clip. On a
#: dedicated machine the default supersampled_gif encoder normally finishes
#: in ~200-230s, but this pipeline's actual host is a multi-user machine
#: shared with other GPU/CPU jobs -- under real contention the same encode
#: has repeatedly taken 10+ minutes, and was previously raised to 1800s for
#: exactly that reason. That, too, turned out not to be enough: a real job
#: on this host was killed by the 1800s timeout after running 1804s, while
#: another user's CPU-bound batch job (8 processes at ~100% CPU each) was
#: running at the same time -- 4s over is still "the timeout was too tight
#: for this host's real contention", not "something hung". Raised again,
#: with more margin, and still overridable per the HEROEXTRACTOR_* env var
#: convention used elsewhere (server/jobs.py) for the same reason: a fixed
#: constant can't know how busy this shared box will be on any given run.
_FFMPEG_TIMEOUT_S = float(os.environ.get("HEROEXTRACTOR_FFMPEG_TIMEOUT_S", "3600"))


def pipe_rgba_to_ffmpeg(cmd: list[str], rgba_chunks, *, cancel=None,
                        timeout_s: float = _FFMPEG_TIMEOUT_S) -> None:
    """Launch ffmpeg, stream raw RGBA frame bytes to stdin, and reap robustly.

    `cancel` (anything with `.is_set()`, e.g. a threading.Event) is checked
    once per frame *during the stdin-write loop* -- checking it only around
    the final `proc.wait()` (as an earlier version of this function did)
    would never see it: for a clip whose write loop itself is the slow part
    (confirmed during the V5 parallel-encode experiments, where CPU/memory
    contention could stall writes for 20+ minutes with nothing past this
    point ever running), a job requested to cancel would keep running to
    completion regardless.

    `timeout_s` bounds the ENTIRE call (write loop + final wait), not just
    the wait -- the previous version's deadline only ever covered the wait,
    so a pathologically slow write loop had no timeout at all (the same gap
    `cancel` fixes, just for hangs instead of cancellation)."""
    deadline = time.monotonic() + timeout_s
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    stream_err: Exception | None = None
    try:
        try:
            for frame_bytes in rgba_chunks:
                if cancel is not None and cancel.is_set():
                    raise EncodeCancelled("cancelled during frame write")
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"ffmpeg timeout ({timeout_s}s) during frame write. "
                        f"cmd={cmd[0]} ... -> {cmd[-1]}")
                try:
                    proc.stdin.write(frame_bytes)
                except BrokenPipeError:
                    break
        except Exception as e:
            stream_err = e
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if stream_err is None:
            wait_s = max(1.0, deadline - time.monotonic())
            ret = proc.wait(timeout=wait_s)
            if ret != 0:
                err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                raise RuntimeError(
                    f"ffmpeg failed (exit={ret}). cmd={cmd[0]} ... -> {cmd[-1]}\n"
                    f"stderr:\n{err.strip() or '(empty)'}"
                )
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except OSError:
                pass
    if stream_err is not None:
        raise stream_err


def _rawvideo_input(width: int, height: int, fps: float) -> list[str]:
    return [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}", "-framerate", str(fps),
        "-i", "pipe:0",
    ]


def encode_gif(rgba_chunks, width: int, height: int, fps: float, out_path: Path,
               dither: str = DEFAULT_GIF_DITHER,
               alpha_threshold: int = DEFAULT_GIF_ALPHA_THRESHOLD, *, cancel=None) -> None:
    """Transparent GIF via a global palette (palettegen + transparent paletteuse)."""
    cmd = _rawvideo_input(width, height, fps) + [
        "-filter_complex",
        "[0:v]split[a][b];"
        "[a]palettegen=reserve_transparent=1[p];"
        f"[b][p]paletteuse=alpha_threshold={alpha_threshold}:dither={dither}",
        "-loop", "0", str(out_path),
    ]
    pipe_rgba_to_ffmpeg(cmd, rgba_chunks, cancel=cancel)


def encode_gif_supersampled(rgba_chunks, width: int, height: int, fps: float, out_path: Path,
                             ss: int = 4, alpha_threshold: int = 128,
                             dither: str = DEFAULT_GIF_DITHER, *, cancel=None) -> None:
    """Transparent GIF via the two-quantisation-pass chain scripts/run_gpu_
    birefnet_v8.py through v25.py all used: 4x lanczos upscale -> palette
    pass 1 -> area downscale -> palette pass 2. Measured (Phase 2 of the
    18-bg-remove plan) to NOT be the colour-flicker source that motivated
    it, but this IS still the validated default: encode_gif_ss_alpha
    (single quantisation pass, numpy handles the supersampled alpha edge
    instead) looked like a simpler equally-faithful replacement, but the
    plan's L1/V2/V3 experiments found it -- and every other faster
    variant tried (ss=2/3, a pre-smoothed single pass) -- measurably worse
    on the primary fidelity gate (F1i), later traced to BiRefNet's own
    alpha non-determinism near the threshold rather than the encoder
    itself. supersampled_gif stays the default until that root cause is
    addressed, not because a replacement was never tried."""
    cmd = _rawvideo_input(width, height, fps) + [
        "-filter_complex",
        f"[0:v]scale={ss}*iw:{ss}*ih:flags=lanczos,split[a1][b1];"
        "[a1]palettegen=reserve_transparent=1[p1];"
        f"[b1][p1]paletteuse=alpha_threshold={alpha_threshold}:dither=none[hi];"
        f"[hi]scale=iw/{ss}:ih/{ss}:flags=area,split[a2][b2];"
        "[a2]palettegen=reserve_transparent=1[p2];"
        f"[b2][p2]paletteuse=alpha_threshold=128:dither={dither}",
        "-loop", "0", str(out_path),
    ]
    pipe_rgba_to_ffmpeg(cmd, rgba_chunks, cancel=cancel)


def alpha_supersample_threshold(alpha: np.ndarray, ss: int = 4,
                                 alpha_threshold: int = 128) -> np.ndarray:
    """Binarise a soft (0-255) alpha channel with sub-pixel edge accuracy,
    WITHOUT touching colour -- upscale ss x via linear interpolation, hard
    threshold, box-average back down, threshold again. This reproduces what
    the old two-quantisation GIF chain got "for free" on the alpha edge, but
    as a pure numpy op decoupled from the palette, so colour is never
    quantised twice (see encode_gif_ss_alpha for why that mattered: identical
    RGB fed through two palettegen/paletteuse passes still produced ~1000
    px/frame of >8-level colour jumps on a perfectly static surface)."""
    h, w = alpha.shape
    up = cv2.resize(alpha, (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
    up_bin = (up >= alpha_threshold).astype(np.float32) * 255.0
    down = cv2.resize(up_bin, (w, h), interpolation=cv2.INTER_AREA)
    return (down >= alpha_threshold).astype(np.uint8) * 255


def encode_gif_ss_alpha(rgba_chunks, width: int, height: int, fps: float, out_path: Path,
                         ss: int = 4, alpha_threshold: int = 128,
                         dither: str = DEFAULT_GIF_DITHER, *, cancel=None) -> None:
    """Transparent GIF with a supersampled alpha edge but a SINGLE colour
    quantisation pass -- the Phase 2 fix for the 18-bg-remove pipeline's
    colour-flicker regression. RGB is passed through byte-for-byte; only the
    alpha channel is supersampled (see alpha_supersample_threshold). A static
    pixel therefore gets bit-identical RGB on every frame, and one global
    palette maps identical input to identical output deterministically, so
    GIF-palette flicker cannot occur structurally -- no colour-freeze /
    "still pixel" heuristic is needed to suppress it."""
    def _resample(chunks):
        for c in chunks:
            f = np.frombuffer(c, np.uint8).reshape(height, width, 4).copy()
            f[:, :, 3] = alpha_supersample_threshold(f[:, :, 3], ss=ss, alpha_threshold=alpha_threshold)
            yield f.tobytes()

    encode_gif(_resample(rgba_chunks), width, height, fps, out_path,
               dither=dither, alpha_threshold=alpha_threshold, cancel=cancel)


def encode_webp(rgba_chunks, width: int, height: int, fps: float, out_path: Path, *,
                 lossless: bool = False, quality: int = 90,
                 cancel=None) -> None:
    """Animated WebP with real 8-bit alpha -- diagnostic + delivery format.
    Because it carries soft alpha (no 1-bit snap, no palette quantisation),
    comparing this against the GIF of the SAME frames isolates encoder-caused
    defects (only in the GIF) from matting-caused defects (in both).

    `lossless`/`quality` exist for the 第9計画 A2 WebP-variant measurement:
    the lossy default internally subsamples chroma to 4:2:0, which can
    reintroduce colour bleed at the semi-transparent edge -- exactly what
    despill spends real time removing. `lossless=True` removes that step
    entirely (at a size/speed cost). Note: this ffmpeg build's libwebp/
    libwebp_anim encoder does NOT expose libwebp's own `-exact` flag
    (RGB-under-alpha=0 preservation) -- `ffmpeg -h encoder=libwebp_anim`
    lists only lossless/preset/quality/cr_threshold/cr_size, confirmed on
    ffmpeg 6.1.1 here; passing `-exact` fails with "Unrecognized option".
    Production call sites (tool/pipeline/runner.py) don't pass these, so
    behaviour is unchanged until a caller opts in."""
    cmd = _rawvideo_input(width, height, fps) + [
        "-c:v", "libwebp_anim", "-lossless", "1" if lossless else "0",
        "-quality", str(quality),
        "-loop", "0", "-an", "-vsync", "0", str(out_path),
    ]
    pipe_rgba_to_ffmpeg(cmd, rgba_chunks, cancel=cancel)


def encode_mov(rgba_chunks, width: int, height: int, fps: float, out_path: Path, *,
                cancel=None) -> None:
    """Transparent MOV (ProRes 4444) with an alpha channel — the default, high-quality output."""
    cmd = _rawvideo_input(width, height, fps) + [
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        str(out_path),
    ]
    pipe_rgba_to_ffmpeg(cmd, rgba_chunks, cancel=cancel)

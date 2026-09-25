"""B-1: SAM2 video-tracking VRAM measurement (see the plan's Part B, 第9計画
2026-09-22, "### B. SAM2 動画追跡シード", step 1).

MEASUREMENT ONLY. This script does NOT implement the trimap/seeding feature
(B-2/B-3) -- it exists solely to answer the pre-registered decision-gate
question: does SAM2's video predictor fit the existing 8GB cold-start VRAM
budget (`HEROEXTRACTOR_MIN_FREE_MB`, server/jobs.py) at realistic clip
lengths? If not, Lever B is abandoned before any trimap code is written.

Design:
  1. Decode sample/dinosaur/Triceratops.mp4 (1656x1248, 121 frames @ 24fps --
     only ~5s of native footage) and write frames resized to 1024x1024
     (SAM2's native resolution, INTER_AREA) as `NNNNN.jpg` into a scratch
     directory. `init_state()` only accepts a JPEG-directory or MP4 path
     (decord, needed for direct MP4 loading, isn't installed in this venv --
     confirmed by prior investigation, see DECISIONS.md), so this JPEG
     pre-pass is mandatory either way.
     Triceratops itself is far short of the 300/900/1800-frame (10/30/60s)
     targets, so frames are CYCLED (looped) to reach the target count. This
     is a proxy for VRAM behavior at a given FRAME COUNT (which is what
     drives SAM2's memory bank / offloaded-frame-tensor growth), not a claim
     about tracking quality over a real 60s continuous shot -- quality is
     out of scope for B-1 (that's B-2).
  2. For each duration, run the SAM2 video predictor in a spawned
     subprocess -- mirroring `_Sam2Worker`/`_sam2_worker_main` in
     matte_core.py (matte_core.py:513-556), which keeps SAM2's torch/CUDA
     context out of the main process because sharing one CUDA context with
     onnxruntime's CUDA session was confirmed by testing to make BOTH
     collapse 10-30x. The child: builds the video predictor, calls
     `init_state(jpg_dir, offload_video_to_cpu=True, offload_state_to_cpu=True,
     async_loading_frames=True)`, adds a placeholder box prompt on frame 0
     (precision doesn't matter for a VRAM measurement -- only that tracking
     actually runs end to end), then `propagate_in_video()`s across every
     frame, recording `torch.cuda.max_memory_allocated/reserved`.
  3. The PARENT process independently polls
     `nvidia-smi --query-compute-apps=pid,used_memory` for the child's PID
     (and, in the co-resident scenario below, its own PID too) throughout
     the run, as an OS-level cross-check against the torch-reported numbers
     -- torch's counters only see its own process's allocator, not e.g.
     cuDNN workspace/driver overhead outside it.
  4. Each duration is measured twice: "alone" (SAM2 subprocess only) and
     "co-resident" (the parent ALSO loads BiRefNet+YOLOX via
     `tool.pipeline.runner._load_models`, matching the real production
     scenario where the matting models and SAM2 would both be GPU-resident
     during a clip's `track` stage). The co-resident scenario is the
     realistic worst case the decision gate is judged against.

Decision gate (pre-registered in the plan, § do not deviate): if co-resident
peak VRAM at 60s exceeds 8000MB (HEROEXTRACTOR_MIN_FREE_MB's default), Lever
B is abandoned -- see tool/docs/DECISIONS.md for the recorded verdict.

Usage:
    venv/bin/python -m tool.scripts.sam2_seed_experiment --measure-only
    venv/bin/python -m tool.scripts.sam2_seed_experiment --measure-only \\
        --durations 10,30 --out data/sam2_seed_experiment/quick

GPU etiquette: checks `nvidia-smi` for pre-existing VRAM usage before doing
any heavy GPU work and refuses to run if another process already holds more
than ~3-4GB (this GPU is shared with other users/services on this machine).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_ld_library_path() -> None:
    """Mirror run.sh/serve.sh's LD_LIBRARY_PATH export (this project's
    onnxruntime-gpu needs the venv's pip-installed nvidia-*-cu12 cuDNN/cuBLAS
    .so's on LD_LIBRARY_PATH on Linux -- unlike the Windows DLL-search-path
    fix in matte_core.py's _ensure_nvidia_dlls_on_path, which only touches
    PATH and is a no-op here). Without this, onnxruntime's CUDAExecutionProvider
    silently fails to load (libcudnn.so.9 not found) and BOTH BiRefNet and
    YOLOX fall back to CPU -- which would make the "co-resident" scenario
    below measure nothing (no matting models actually on the GPU).

    Setting os.environ['LD_LIBRARY_PATH'] from *within* an already-running
    process does NOT work here (confirmed by testing): glibc's dynamic
    loader parses LD_LIBRARY_PATH once at process startup, not per dlopen()
    call, so onnxruntime's later dlopen of its CUDA provider library still
    fails to find cuDNN even though os.environ looks correct. The only fix
    is to have the *exec'd* process start with it already set -- so if it's
    missing, this re-execs the current interpreter with the variable set
    (once, guarded by _SAM2_EXPERIMENT_REEXEC so it can't loop)."""
    if os.environ.get("_SAM2_EXPERIMENT_REEXEC") == "1":
        return
    venv_nvidia = REPO_ROOT / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    if not venv_nvidia.is_dir():
        return
    pkg_dirs = [str(venv_nvidia / pkg / "lib") for pkg in
                ("cudnn", "cublas", "cufft", "curand", "cuda_runtime", "nvjitlink", "cuda_nvrtc")]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld_path = os.pathsep.join(pkg_dirs) + (os.pathsep + existing if existing else "")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = new_ld_path
    env["_SAM2_EXPERIMENT_REEXEC"] = "1"
    print("[env] re-execing with LD_LIBRARY_PATH set for onnxruntime-gpu's cuDNN "
          "(see run.sh/serve.sh)...", flush=True)
    argv = [sys.executable, "-m", "tool.scripts.sam2_seed_experiment"] + sys.argv[1:]
    os.execve(sys.executable, argv, env)
SRC_CLIP = REPO_ROOT.parent / "sample" / "dinosaur" / "Triceratops.mp4"
CKPT_DIR = REPO_ROOT / "tool" / "checkpoints"
SAM2_CKPT = CKPT_DIR / "sam2.1_hiera_tiny.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # hydra config bundled in the sam2 package
SIZE = 1024  # SAM2's native resolution
FPS = 24  # Triceratops.mp4's native fps -- duration(s) * FPS = target frame count

# Existing cold-start VRAM budget this decision is judged against (server/jobs.py
# MIN_FREE_GPU_MB, env var HEROEXTRACTOR_MIN_FREE_MB, default "8000"). Not
# imported from server/jobs.py on purpose -- this script has no server
# dependency and the number is a pre-registered constant from the plan, not
# something that should silently drift if the server's default ever changes.
BUDGET_MB = 8000.0

# Pre-existing GPU usage above this is treated as "someone else is using this
# shared GPU" -- wait/poll rather than barrel ahead (see module docstring).
OTHER_USER_HEADROOM_MB = 3500.0


# ---------- frame preparation ----------

def _prepare_frames(out_dir: Path, n_frames: int) -> int:
    """Decode SRC_CLIP once, cycle its frames to reach n_frames, resize each
    to SIZExSIZE with INTER_AREA, and write `NNNNN.jpg` into out_dir (the
    layout SAM2's init_state expects). Returns the number of *native* frames
    the source clip had (for logging how much looping was needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SRC_CLIP))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {SRC_CLIP}")
    src_frames = []
    ok, frame = cap.read()
    while ok:
        src_frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if not src_frames:
        raise RuntimeError(f"no frames decoded from {SRC_CLIP}")
    for i in range(n_frames):
        frame = src_frames[i % len(src_frames)]
        resized = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), resized)
        if not ok:
            raise RuntimeError(f"failed to write frame {i} to {out_dir}")
    return len(src_frames)


# ---------- SAM2 video-predictor subprocess (mirrors _Sam2Worker) ----------
#
# Run in a spawned child process for the same reason matte_core.py's
# _sam2_worker_main is (matte_core.py:504-511): sharing one process's CUDA
# context between torch (SAM2) and onnxruntime's CUDA session (BiRefNet/
# YOLOX) was confirmed by testing to make both collapse 10-30x once they
# alternate GPU calls. Isolating them in separate OS processes (each with
# its own CUDA context) is the only way to measure BOTH the "SAM2 alone" and
# "co-resident" scenarios without that pathology confounding the numbers.

def _sam2_video_worker_main(conn, ckpt: str, cfg: str, device: str, jpg_dir: str) -> None:
    """Subprocess entry point for one measurement run: builds the SAM2 video
    predictor, inits state on the JPEG frame directory, adds a placeholder
    box prompt on frame 0, propagates across the whole clip, and reports
    peak torch CUDA memory. Sends a single result dict back over `conn`."""
    import traceback

    result: dict = {"ok": False}
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
        state = predictor.init_state(
            video_path=jpg_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )
        # Placeholder box covering the central region of frame 0. Precision
        # is irrelevant for a VRAM measurement -- only that a real
        # init->prompt->propagate cycle actually executes.
        box = np.array([SIZE * 0.25, SIZE * 0.25, SIZE * 0.75, SIZE * 0.75], dtype=np.float32)
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)

        n_tracked = 0
        for _frame_idx, _obj_ids, _mask_logits in predictor.propagate_in_video(state):
            n_tracked += 1

        result["ok"] = True
        result["frames_tracked"] = n_tracked
        result["elapsed_s"] = time.time() - t0
        result["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1e6
        result["max_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
    except Exception as e:  # noqa: BLE001 -- an OOM/crash here IS a result, not a bug to hide
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    try:
        conn.send(result)
    except (BrokenPipeError, OSError):
        pass
    conn.close()


def _spawn_sam2_child(jpg_dir: Path, device: str = "cuda"):
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_sam2_video_worker_main,
        args=(child_conn, str(SAM2_CKPT), SAM2_CFG, device, str(jpg_dir)),
        daemon=True,
    )
    proc.start()
    return proc, parent_conn


# ---------- OS-level VRAM cross-check ----------

def _nvidia_smi_used_mb(pids: set[int]) -> float | None:
    """Sum `nvidia-smi --query-compute-apps` used_memory (MiB) for the given
    PIDs. Returns None if the query itself failed (nvidia-smi unavailable,
    permissions, etc.) rather than pretending 0."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except Exception:
        return None
    total = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, mem = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid in pids:
            total += mem
    return total


def _poll_nvidia_smi(pids: set[int], samples: list[float], stop_event: threading.Event,
                      interval: float = 0.5) -> None:
    while not stop_event.is_set():
        used = _nvidia_smi_used_mb(pids)
        if used is not None:
            samples.append(used)
        stop_event.wait(interval)


def check_gpu_is_free() -> None:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    used_mb, total_mb = (float(x) for x in out.split(","))
    if used_mb > OTHER_USER_HEADROOM_MB:
        raise RuntimeError(
            f"GPU already has {used_mb:.0f}MB in use (> {OTHER_USER_HEADROOM_MB:.0f}MB "
            f"headroom) out of {total_mb:.0f}MB total -- this GPU is shared, refusing to "
            f"start heavy SAM2 VRAM measurement while someone else may be using it. "
            f"Wait and retry.")
    print(f"[gpu] {used_mb:.0f}MB / {total_mb:.0f}MB in use before starting -- OK", flush=True)


# ---------- measurement runs ----------

def measure_one(jpg_dir: Path, label: str, extra_pids: set[int] | None = None,
                 timeout_s: float = 900.0) -> dict:
    """Run one SAM2 video-tracking pass in a subprocess, polling nvidia-smi
    throughout. `extra_pids` (e.g. {os.getpid()}) are included in the
    nvidia-smi sum for the co-resident scenario, where the parent process
    itself also holds BiRefNet/YOLOX VRAM."""
    proc, conn = _spawn_sam2_child(jpg_dir)
    pids = {proc.pid} | (extra_pids or set())
    samples: list[float] = []
    stop_event = threading.Event()
    poll_thread = threading.Thread(target=_poll_nvidia_smi, args=(pids, samples, stop_event),
                                    daemon=True)
    poll_thread.start()

    try:
        if conn.poll(timeout_s):
            child_result = conn.recv()
        else:
            child_result = {"ok": False, "error": f"child did not respond within {timeout_s}s "
                                                    f"(hang or OOM without a raised exception?)"}
    except (EOFError, OSError) as e:
        child_result = {"ok": False, "error": f"connection error reading child result: {e}"}

    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    stop_event.set()
    poll_thread.join(timeout=2)

    return {
        "label": label,
        "n_frames": None,  # filled by caller
        "child_ok": child_result.get("ok", False),
        "child_result": child_result,
        "exitcode": proc.exitcode,
        "nvidia_smi_peak_mb": max(samples) if samples else None,
        "nvidia_smi_n_samples": len(samples),
    }


def run_duration(duration_s: int, frames_root: Path) -> dict:
    """Run both the "alone" and "co-resident" scenarios at one clip duration
    (in seconds, converted to frame count via FPS)."""
    import os

    n_frames = duration_s * FPS
    jpg_dir = frames_root / f"{duration_s}s_{n_frames}f"
    print(f"\n=== {duration_s}s ({n_frames} frames) ===", flush=True)
    if not jpg_dir.exists() or len(list(jpg_dir.glob("*.jpg"))) != n_frames:
        print(f"[frames] writing {n_frames} JPEGs to {jpg_dir} (cycling Triceratops.mp4)...",
              flush=True)
        t0 = time.time()
        n_native = _prepare_frames(jpg_dir, n_frames)
        print(f"[frames] done in {time.time() - t0:.1f}s ({n_native} native frames, "
              f"looped {n_frames / n_native:.2f}x)", flush=True)
    else:
        print(f"[frames] reusing existing {jpg_dir}", flush=True)

    print(f"[alone] measuring SAM2 video tracking alone...", flush=True)
    alone = measure_one(jpg_dir, f"{duration_s}s_alone")
    alone["n_frames"] = n_frames
    _print_result(alone)

    print(f"[co-resident] loading BiRefNet+YOLOX in parent process...", flush=True)
    from tool.pipeline.config import PipelineConfig
    from tool.pipeline.runner import _load_models
    models = _load_models(PipelineConfig(device="cuda"))
    try:
        print(f"[co-resident] measuring SAM2 video tracking alongside BiRefNet+YOLOX...",
              flush=True)
        co_resident = measure_one(jpg_dir, f"{duration_s}s_co_resident",
                                   extra_pids={os.getpid()})
    finally:
        del models  # drop the ORT sessions / release GPU memory before the next duration
    co_resident["n_frames"] = n_frames
    _print_result(co_resident)

    return {"duration_s": duration_s, "n_frames": n_frames, "alone": alone,
            "co_resident": co_resident}


def _print_result(r: dict) -> None:
    cr = r["child_result"]
    if r["child_ok"]:
        print(f"  [{r['label']}] OK: {cr['frames_tracked']} frames tracked in "
              f"{cr['elapsed_s']:.1f}s -- torch max_allocated={cr['max_allocated_mb']:.0f}MB "
              f"max_reserved={cr['max_reserved_mb']:.0f}MB | nvidia-smi peak="
              f"{r['nvidia_smi_peak_mb']}MB ({r['nvidia_smi_n_samples']} samples)", flush=True)
    else:
        print(f"  [{r['label']}] FAILED (exitcode={r['exitcode']}): "
              f"{cr.get('error')}", flush=True)
        if cr.get("traceback"):
            print(cr["traceback"], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure-only", action="store_true",
                     help="required flag: this script only measures VRAM, it never seeds/"
                          "renders a matte (see module docstring)")
    ap.add_argument("--durations", default="10,30,60",
                     help="comma-separated clip durations in seconds (default: 10,30,60)")
    ap.add_argument("--out", default=None,
                     help="output dir for results.json + frame scratch dirs "
                          "(default: data/sam2_seed_experiment/<timestamp>)")
    args = ap.parse_args()

    if not args.measure_only:
        print("This script is measurement-only for B-1 -- pass --measure-only to run it "
              "(B-2/B-3 trimap/seeding are separate, not-yet-implemented future work).",
              file=sys.stderr)
        return 2

    _ensure_ld_library_path()
    check_gpu_is_free()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "sam2_seed_experiment" / \
        time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = out_dir / "frames"

    durations = [int(x) for x in args.durations.split(",")]
    results = [run_duration(d, frames_root) for d in durations]

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[out] results written to {results_path}", flush=True)

    print("\n=== summary ===")
    header = f"{'dur(s)':>7} {'frames':>7} {'scenario':>14} {'torch_alloc_MB':>15} " \
             f"{'torch_reserved_MB':>18} {'nvidia_smi_MB':>14} {'status':>8}"
    print(header)
    verdict_rows = []
    for r in results:
        for scenario in ("alone", "co_resident"):
            row = r[scenario]
            cr = row["child_result"]
            status = "OK" if row["child_ok"] else "FAILED"
            alloc = f"{cr['max_allocated_mb']:.0f}" if row["child_ok"] else "-"
            reserved = f"{cr['max_reserved_mb']:.0f}" if row["child_ok"] else "-"
            smi = f"{row['nvidia_smi_peak_mb']:.0f}" if row["nvidia_smi_peak_mb"] is not None else "-"
            print(f"{r['duration_s']:>7} {r['n_frames']:>7} {scenario:>14} {alloc:>15} "
                  f"{reserved:>18} {smi:>14} {status:>8}")
            verdict_rows.append((r["duration_s"], scenario, row))

    # Decision gate: co-resident peak at the LONGEST duration tested vs BUDGET_MB.
    longest = max(results, key=lambda r: r["duration_s"])
    co = longest["co_resident"]
    if not co["child_ok"]:
        print(f"\n[VERDICT] co-resident run at {longest['duration_s']}s CRASHED/FAILED -- "
              f"treated as budget EXCEEDED (a crash/OOM is itself evidence the budget doesn't "
              f"fit). Lever B should be abandoned; see tool/docs/DECISIONS.md.")
    else:
        peak = co["nvidia_smi_peak_mb"] if co["nvidia_smi_peak_mb"] is not None else \
            co["child_result"]["max_reserved_mb"]
        verdict = "WITHIN budget" if peak <= BUDGET_MB else "EXCEEDS budget"
        print(f"\n[VERDICT] co-resident peak at {longest['duration_s']}s = {peak:.0f}MB vs "
              f"{BUDGET_MB:.0f}MB budget -> {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

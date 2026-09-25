"""Fake-pipeline FastAPI app for the frontend E2E tests (e2e.test.mjs).

Monkeypatches server.jobs.run_clip / _load_models / validate_upload and
tool.pipeline.__main__._run_qc exactly like server/tests/test_api.py does,
so every job.json this produces has the identical shape a real completed
job would -- only the pixels (a tiny real GIF/PNG) and metrics (real
numbers copied from an actual results_dinosaur/ run) are canned. This lets
the frontend suite exercise the full "done" rendering path (GIF preview,
metrics table, contact sheet) in milliseconds instead of the minutes a
real GPU render (or, worse, a real CPU one under this machine's often-
contended shared GPU) takes.

Usage: uvicorn fake_server:app --app-dir <this dir> --host 127.0.0.1 --port <port>
(see run.sh in this directory, or e2e.test.mjs which spawns it directly).
"""
import json
import os
import sys
from pathlib import Path

THEORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(THEORY_ROOT))

import server.jobs as jobs_mod
from server.probe import ProbeInfo

# Real numbers from an actual results_dinosaur/思考_matte.gif harness run
# (see the plan's Phase 0 entries) -- realistic values, not just zeros, so
# a test asserting "the metrics table shows real-looking numbers" means
# something.
FAKE_METRICS = {
    "F1_false_erase": {"total": 1166834, "per_frame": [], "worst_frame": 41, "worst_value": 12000},
    "F1i_false_erase_interior": {"total": 89809, "per_frame": [], "worst_frame": 41, "worst_value": 3000},
    "F2_false_keep": {"total": 47, "per_frame": [], "worst_frame": 10, "worst_value": 5},
    "F3_interior_holes": {"total": 4058, "per_frame": [], "worst_frame": 20, "worst_value": 300},
    "S1_mc_chatter": {"total": 172096, "per_frame": [], "worst_frame": 30, "worst_value": 3000,
                      "mean_per_frame": 1410.628},
    "S2_color_flicker": {"total": 120, "per_frame": [], "worst_frame": 5, "worst_value": 20,
                         "mean_per_frame": 0.992},
    "S3_frozen_px": {"total": 0, "bbox": None},
    "S4_area_jump": {"worst_frame": 12, "worst_value": 0.085, "mean": 0.02, "areas": []},
    "E1_perimeter_ratio": {"median": 7.968, "worst_frame": 8, "worst_value": 8.5},
    "E2_fringe_quality": {"mean": 70.964, "worst_frame": 15, "worst_value": 55.0},
}

# A minimal but valid 2x2 transparent GIF89a and a 1x1 PNG -- real enough
# that an <img> actually decodes and paints (naturalWidth > 0) instead of
# showing a broken-image icon.
_TINY_GIF = bytes.fromhex(
    "47494638396102000200800000000000ffffff21f9040100000000"
    "2c00000000020002000002024401003b"
)
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600"
    "00001f15c4890000000a49444154789c6360000002000155a2415a"
    "0000000049454e44ae426082"
)

# A clip whose UPLOADED FILENAME contains "slow" blocks on `cancel`
# instead of finishing instantly -- needed for a test that clicks Cancel
# mid-run and expects to observe "running" before it flips to
# "cancelled". Scoped per-clip (L7) rather than a process-wide
# FAKE_SLOW_CANCEL_CHECK env var (the previous design): that made EVERY
# job in EVERY test pay a real ~2s penalty regardless of whether that
# test cared about the cancel window at all.
import time


def fake_run_clip(video_path, out_path, config, models=None, raw=None, *,
                   progress=None, cancel=None, preview=None, timings=None, **_kw):
    from tool.pipeline.runner import JobCancelled

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The real run_clip opens with a "prepare" tick (model load / key
    # sampling happen before any frame-level stage) so the front end can
    # show "Preparing…" immediately. A filename containing "preparing"
    # then LINGERS in that stage for ~1.5s (cancellable), long enough for
    # an E2E test to observe the preparing status line + indeterminate bar.
    if progress is not None:
        progress("prepare", 0, 1)
    if "preparing" in Path(video_path).name.lower():
        for _ in range(150):
            if cancel is not None and cancel.is_set():
                raise JobCancelled("cancelled by fake_server")
            time.sleep(0.01)
    # A filename containing "keyable" simulates a clip fake_validate_upload
    # (below) reports as flat-chroma AND saturated enough to key on --
    # writing key_s into timings is what a real run_clip does on that path
    # (see runner.py's infer_clip), and clip["auto_keyer"] is derived from
    # its presence (server/jobs.py), so this is what lets an E2E test
    # assert the fast path's outcome actually reaches the job record.
    if timings is not None and "keyable" in Path(video_path).name.lower():
        timings["key_s"] = 0.01
    if preview is not None:
        import numpy as np
        frame = np.zeros((48, 64, 4), dtype=np.uint8)
        frame[:, :, 1] = 180  # a visibly non-black square, easy to eyeball in a screenshot
        frame[:, :, 3] = 255
        preview(1, frame)
    if progress is not None:
        progress("infer", 1, 4)
    if "slow" in Path(video_path).name.lower():
        for _ in range(200):  # up to ~2s of "still running" for a cancel test to land in
            if cancel is not None and cancel.is_set():
                raise JobCancelled("cancelled by fake_server")
            time.sleep(0.01)
    # A filename containing "fail" raises with a distinctive detail string,
    # for the one E2E test that checks the error banner's SECOND line
    # (job.error.detail / clip.error.detail) actually renders -- the plan's
    # B3 finding was that this reached the browser's memory but nothing on
    # screen ever displayed it.
    if "fail" in Path(video_path).name.lower():
        raise RuntimeError("ffmpeg timeout (600s) during frame write -- fake_server test failure")
    if progress is not None:
        for i in range(2, 5):
            progress("infer", i, 4)
        progress("postprocess", 0, 1)
        progress("encode", 1, 1)
        progress("done", 1, 1)
    out_path.write_bytes(_TINY_GIF)
    Path(str(out_path) + ".pipeline_config.json").write_text("{}")
    return out_path


def fake_run_qc(source_path, out_path):
    out_path = Path(out_path)
    out_path.with_suffix(out_path.suffix + ".qc_contact_sheet.png").write_bytes(_TINY_PNG)
    out_path.with_suffix(".qc.json").write_text(json.dumps(FAKE_METRICS))
    return FAKE_METRICS


jobs_mod.run_clip = fake_run_clip
jobs_mod._load_models = lambda config: object()


def fake_validate_upload(path):
    # A filename containing "keyable" reports a real, saturated flat-chroma
    # backdrop (matching flatchroma2's own measured
    # saturation) -- everything else keeps the prior default (flat but
    # bg_key_saturation=None, i.e. NOT reported as keyable: keyer.
    # probe_says_keyable treats an unprobed/unknown saturation as
    # conservatively "needs the GPU", same as a real never-probed clip).
    is_keyable = "keyable" in Path(path).name.lower()
    return ProbeInfo(1656, 1248, 24.0, 122, 5.09, 4.6,
                      bg_is_chroma_class=True, bg_frac_bg_like=1.0,
                      bg_key_saturation=61.0 if is_keyable else None)


jobs_mod.validate_upload = fake_validate_upload
# L1's GPU pre-flight check would otherwise call the REAL probe.gpu_status()
# (real nvidia-smi) for every job here -- device defaults to "auto"
# (server/presets.py), which resolves to whatever this host's real
# onnxruntime/torch CUDA probe finds, and this fake server's whole point is
# running in milliseconds regardless of this machine's actual (often
# contended) GPU state.
jobs_mod.MIN_FREE_GPU_MB = 0.0

import tool.pipeline.__main__ as main_mod

main_mod._run_qc = fake_run_qc

from server.app import create_app

data_root = Path(os.environ.get("FAKE_SERVER_DATA_DIR", "/tmp/heroextractor_fake_server_data"))
app = create_app(data_root)

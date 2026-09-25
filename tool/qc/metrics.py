"""Fidelity + stability + edge-quality metrics for background-removal output.

Deliberately takes ONLY (source video path, output file path) as input -- no
dependency on pipeline internals -- so it can grade any config, any encoder
(GIF/WebP/MOV), and any historical output (out_v23b, out_v24, ...) the same
way. See the 18-bg-remove pipeline overhaul plan for why fidelity metrics
(F1/F2/F3) are the primary gate and stability metrics (S1-S4) are secondary:
on run_gpu_birefnet_v23.py's output, every stability metric was green while
130,340 px of the subject's own white body had been erased.

Chroma-distance model (shared with the planned trimap stage in
tool/pipeline/chroma.py): background colour is estimated from the border ring
of the SOURCE frames only (no matting internals available here), robustly via
sigma-clipped median/MAD in the YCrCb chroma plane (Cb, Cr) -- luma is
excluded because the subject's own shading moves luma a lot while true
background chroma stays flat. Distance is expressed in sigma units, so a
single pair of thresholds (T_LO, T_HI) works unscaled across clips.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

# Distance thresholds, in units of the background's own chroma sigma.
T_LO = 3.0   # d < T_LO  => pixel colour is background-like
T_HI = 8.0   # d > T_HI  => pixel colour is clearly NOT background (subject)

STILL_THRESH = 10.0     # source per-pixel max-channel delta below which a
                         # pixel counts as "temporally static" (for S2)
COLOR_JUMP = 8           # RGB level jump counted as a flicker event (for S2)
ALPHA_THR = 127          # opaque/transparent split for all metrics here


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def read_rgba_frames(path):
    """Decode any ffmpeg-readable file (GIF/WebP/MOV/...) to (T,H,W,4) uint8,
    RGB-ordered with a real alpha channel.

    WebP is special-cased to Pillow instead of ffmpeg: this machine's ffmpeg
    (6.1.1) cannot decode animated WebP at all -- it skips the ANIM/ANMF
    chunks entirely and raises "image data not found", so read_rgba_frames
    used to throw RuntimeError("no frames decoded") for every .webp input.
    That silently turned into a hard job failure wherever QC runs on a webp
    output (server/jobs.py's `_run_qc` after a successful render, or the
    LOAD_AWARE_ENCODE fallback's own QC pass) -- a render that succeeded
    was reported as failed because grading it crashed. Animated-WebP
    support only lands in ffmpeg 7.1+; Pillow (already a dependency, WebP
    support compiled in) decodes it correctly today, confirmed against a
    synthetic 3-frame WebP that ffmpeg returns 0 frames for."""
    if Path(path).suffix.lower() == ".webp":
        im = Image.open(path)
        frames = [np.array(frame.convert("RGBA"))
                  for frame in ImageSequence.Iterator(im)]
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        h, w = frames[0].shape[:2]
        return np.stack(frames, 0), h, w
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h * 4
    frames = []
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 4).copy())
    proc.wait()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, 0), h, w


def read_source_rgb(path):
    """Decode the source MP4 to (T,H,W,3) uint8, RGB-ordered, via OpenCV."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, 0)


# --------------------------------------------------------------------------
# Background chroma model
# --------------------------------------------------------------------------

@dataclass
class BgStats:
    mu_cb: float
    mu_cr: float
    sigma_cb: float
    sigma_cr: float
    is_chroma_class: bool
    frac_bg_like: float = 1.0


def estimate_bg_stats(source_rgb, ring=24, clip_rounds=2, clip_k=5.0):
    """Robust background chroma stats from the border ring of every frame,
    via iterative sigma-clipping (no matting internals available here, so a
    handful of subject pixels touching the border are tolerated as outliers)."""
    T, H, W, _ = source_rgb.shape
    cb_all, cr_all = [], []
    for i in range(T):
        ycc = cv2.cvtColor(source_rgb[i], cv2.COLOR_RGB2YCrCb)
        cr = ycc[:, :, 1]
        cb = ycc[:, :, 2]
        ring_mask = np.zeros((H, W), bool)
        ring_mask[:ring, :] = True
        ring_mask[-ring:, :] = True
        ring_mask[:, :ring] = True
        ring_mask[:, -ring:] = True
        cb_all.append(cb[ring_mask])
        cr_all.append(cr[ring_mask])
    cb_all = np.concatenate(cb_all).astype(np.float32)
    cr_all = np.concatenate(cr_all).astype(np.float32)

    keep = np.ones(cb_all.shape[0], bool)
    for _ in range(clip_rounds):
        mu_cb, mu_cr = np.median(cb_all[keep]), np.median(cr_all[keep])
        # Floor at 1.0 level, not an epsilon: a clean synthetic-render background
        # (e.g. a flat rendered yellow) can have EXACTLY zero measured spread in
        # one chroma channel, which would divide distances by ~0 and misclassify
        # nearly every pixel (background included) as "far from background".
        mad_cb = max(np.median(np.abs(cb_all[keep] - mu_cb)) * 1.4826, 1.0)
        mad_cr = max(np.median(np.abs(cr_all[keep] - mu_cr)) * 1.4826, 1.0)
        d = np.sqrt(((cb_all - mu_cb) / mad_cb) ** 2 + ((cr_all - mu_cr) / mad_cr) ** 2)
        keep = d < clip_k

    # Classifier question: is nearly the ENTIRE border clearly background-
    # coloured (d < T_HI), not "how many points are within one background
    # sigma" (T_LO) -- a mascot clip's gestures occasionally reach the frame
    # edge, so demanding tight clustering at T_LO undercounts a genuinely
    # clean chroma background. T_HI is the same "clearly not background"
    # line false_erase() uses, so this reuses one threshold, not two.
    frac_bg_like = float((np.sqrt(((cb_all - mu_cb) / mad_cb) ** 2 +
                                   ((cr_all - mu_cr) / mad_cr) ** 2) < T_HI).mean())
    return BgStats(mu_cb=float(mu_cb), mu_cr=float(mu_cr),
                   sigma_cb=float(mad_cb), sigma_cr=float(mad_cr),
                   is_chroma_class=frac_bg_like > 0.98, frac_bg_like=frac_bg_like)


def chroma_distance(frame_rgb, bg: BgStats):
    """Per-pixel background distance in sigma units (YCbCr chroma plane)."""
    ycc = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycc[:, :, 1].astype(np.float32)
    cb = ycc[:, :, 2].astype(np.float32)
    return np.sqrt(((cb - bg.mu_cb) / bg.sigma_cb) ** 2 +
                    ((cr - bg.mu_cr) / bg.sigma_cr) ** 2)


# --------------------------------------------------------------------------
# Fidelity metrics (primary gate)
# --------------------------------------------------------------------------

def false_erase(source_rgb, out_rgba, bg: BgStats):
    """F1: source pixel colour is clearly NOT background (d>T_HI) but the
    output made it transparent -- i.e. the subject's own body got erased."""
    T = min(len(source_rgb), len(out_rgba))
    per_frame = np.zeros(T, dtype=np.int64)
    for i in range(T):
        d = chroma_distance(source_rgb[i], bg)
        transparent = out_rgba[i, :, :, 3] < ALPHA_THR
        per_frame[i] = int(((d > T_HI) & transparent).sum())
    return {"total": int(per_frame.sum()), "per_frame": per_frame.tolist(),
            "worst_frame": int(np.argmax(per_frame)), "worst_value": int(per_frame.max())}


def false_erase_interior(source_rgb, out_rgba, bg: BgStats, margin=10):
    """F1i: like false_erase, but excludes pixels near a REAL background
    region -- i.e. ordinary 1-bit hard-threshold snapping at a legitimate
    soft edge (which sits right next to true background) doesn't count.
    Only a false-erase that eats `margin`+ px into where no real background
    exists nearby counts. This isolates bugs like _strip_tail_shadow (which
    gouged ~15px into the body, far from any real background pixel) from
    routine edge-antialiasing loss, which false_erase() cannot distinguish
    and which dominates its raw total (see qc harness Phase 0 notes)."""
    T = min(len(source_rgb), len(out_rgba))
    per_frame = np.zeros(T, dtype=np.int64)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(T):
        d = chroma_distance(source_rgb[i], bg)
        real_bg = (d < T_LO).astype(np.uint8)
        # distance (px) from every pixel to the nearest REAL background pixel
        dist_from_bg = cv2.distanceTransform(1 - real_bg, cv2.DIST_L2, 5)
        transparent = out_rgba[i, :, :, 3] < ALPHA_THR
        far_from_bg = dist_from_bg > margin
        per_frame[i] = int(((d > T_HI) & transparent & far_from_bg).sum())
    return {"total": int(per_frame.sum()), "per_frame": per_frame.tolist(),
            "worst_frame": int(np.argmax(per_frame)), "worst_value": int(per_frame.max())}


def false_keep(source_rgb, out_rgba, bg: BgStats):
    """F2: source pixel colour IS background-like (d<T_LO) but the output
    kept it opaque -- i.e. background leaked through."""
    T = min(len(source_rgb), len(out_rgba))
    per_frame = np.zeros(T, dtype=np.int64)
    for i in range(T):
        d = chroma_distance(source_rgb[i], bg)
        opaque = out_rgba[i, :, :, 3] >= ALPHA_THR
        per_frame[i] = int(((d < T_LO) & opaque).sum())
    return {"total": int(per_frame.sum()), "per_frame": per_frame.tolist(),
            "worst_frame": int(np.argmax(per_frame)), "worst_value": int(per_frame.max())}


def interior_holes(out_rgba):
    """F3: connected background-alpha regions fully enclosed by the subject's
    silhouette (ported from scripts/qc_strict.py's floodFill approach)."""
    T, H, W = out_rgba.shape[:3]
    per_frame = np.zeros(T, dtype=np.int64)
    for i in range(T):
        fg = (out_rgba[i, :, :, 3] > ALPHA_THR).astype(np.uint8)
        inv = (1 - fg) * 255
        # Pad with a guaranteed-transparent (255) 1px border before seeding
        # floodFill at (0,0) on the padded canvas: seeding directly on `inv`
        # silently fills nothing if pixel (0,0) itself happens to be opaque
        # (confirmed to happen -- a single stray opaque corner pixel made
        # this metric misreport ~83% of the whole frame as "interior holes"
        # on an otherwise-clean image). The 1px pad is always 255, so the
        # seed always matches and the real border ring's connectivity to
        # frame corners no longer depends on one literal pixel's alpha.
        padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(padded, ff_mask, (0, 0), 128)
        interior_bg = (padded[1:-1, 1:-1] == 255)
        per_frame[i] = int(interior_bg.sum())
    return {"total": int(per_frame.sum()), "per_frame": per_frame.tolist(),
            "worst_frame": int(np.argmax(per_frame)), "worst_value": int(per_frame.max())}


# --------------------------------------------------------------------------
# Stability metrics (secondary gate)
# --------------------------------------------------------------------------

def mc_chatter(source_rgb, out_rgba):
    """S1: motion-compensated alpha chatter -- warp each frame's alpha into
    its neighbour via DIS optical flow (computed on SOURCE luma) before
    XOR-ing, so genuine motion isn't counted as instability."""
    T, H, W = out_rgba.shape[:3]
    if T < 2:
        return {"total": 0, "per_frame": [], "worst_frame": 0, "worst_value": 0}
    grays = [cv2.cvtColor(source_rgb[i], cv2.COLOR_RGB2GRAY) for i in range(min(T, len(source_rgb)))]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    gx, gy = gx.astype(np.float32), gy.astype(np.float32)
    alphas = (out_rgba[:, :, :, 3] > ALPHA_THR)
    per_frame = np.zeros(T - 1, dtype=np.int64)
    for i in range(T - 1):
        fl = dis.calc(grays[i], grays[i + 1], None)
        warped = cv2.remap((alphas[i + 1] * 255).astype(np.uint8), gx + fl[..., 0], gy + fl[..., 1],
                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 127
        per_frame[i] = int((alphas[i] != warped).sum())
    return {"total": int(per_frame.sum()), "per_frame": per_frame.tolist(),
            "worst_frame": int(np.argmax(per_frame)), "worst_value": int(per_frame.max()),
            "mean_per_frame": float(per_frame.mean())}


def color_flicker(source_rgb, out_rgba):
    """S2: RGB level jumps (>COLOR_JUMP) between consecutive frames, on
    pixels opaque in both AND whose source content is temporally static --
    isolates GIF-palette-style flicker from legitimate motion."""
    T = min(len(source_rgb), len(out_rgba))
    src = source_rgb[:T].astype(np.int16)
    static = np.abs(np.diff(src, axis=0)).max(3).mean(0) < 1.0 if T > 1 else np.zeros(src.shape[1:3], bool)
    per_frame = np.zeros(max(T - 1, 0), dtype=np.int64)
    for i in range(T - 1):
        op = (out_rgba[i, :, :, 3] > ALPHA_THR) & (out_rgba[i + 1, :, :, 3] > ALPHA_THR)
        jump = np.abs(out_rgba[i + 1, :, :, :3].astype(np.int16) -
                      out_rgba[i, :, :, :3].astype(np.int16)).max(2) > COLOR_JUMP
        per_frame[i] = int((op & jump & static).sum())
    return {"total": int(per_frame.sum()), "per_frame": per_frame.tolist(),
            "worst_frame": int(np.argmax(per_frame)) if len(per_frame) else 0,
            "worst_value": int(per_frame.max()) if len(per_frame) else 0,
            "mean_per_frame": float(per_frame.mean()) if len(per_frame) else 0.0}


def frozen_px(out_rgba):
    """S3: pixels identical across every single frame AND always opaque --
    a constant-colour patch "pasted" onto the whole clip."""
    identical = np.all(out_rgba[:, :, :, :3] == out_rgba[0:1, :, :, :3], axis=0).all(axis=2)
    opaque_always = (out_rgba[:, :, :, 3] > ALPHA_THR).all(axis=0)
    frozen = identical & opaque_always
    ys, xs = np.nonzero(frozen)
    bbox = None
    if ys.size:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {"total": int(frozen.sum()), "bbox": bbox}


def area_jump(out_rgba):
    """S4: frame-to-frame relative change in foreground area."""
    areas = (out_rgba[:, :, :, 3] > ALPHA_THR).reshape(len(out_rgba), -1).sum(1)
    if len(areas) < 2:
        return {"worst_frame": 0, "worst_value": 0.0, "areas": areas.tolist()}
    diffs = np.abs(np.diff(areas)) / np.maximum(areas[:-1], 1)
    return {"worst_frame": int(np.argmax(diffs)), "worst_value": float(diffs.max()),
            "mean": float(diffs.mean()), "areas": areas.tolist()}


# --------------------------------------------------------------------------
# Edge-quality metric (guards against trading edge fidelity for stability)
# --------------------------------------------------------------------------

def perimeter_ratio(out_rgba):
    """E1: silhouette perimeter / sqrt(area) -- rises when 1-bit alpha or an
    encoder change makes the boundary jagged without changing area."""
    T = len(out_rgba)
    per_frame = np.zeros(T, dtype=np.float64)
    for i in range(T):
        fg = (out_rgba[i, :, :, 3] > ALPHA_THR).astype(np.uint8)
        area = int(fg.sum())
        if area == 0:
            continue
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in contours)
        per_frame[i] = perim / np.sqrt(area)
    valid = per_frame[per_frame > 0]
    return {"median": float(np.median(valid)) if valid.size else 0.0,
            "worst_frame": int(np.argmax(per_frame)), "worst_value": float(per_frame.max())}


def fringe_chroma_distance(out_rgba, bg: BgStats, ring=2):
    """E2: mean chroma distance (sigma units) from the background colour,
    measured on OUTPUT pixels in a `ring`-px band just inside the opaque
    silhouette's edge. This is despill's whole job -- pushing edge-adjacent
    colour away from the backdrop colour instead of letting the (1-alpha)
    contribution from the original background show through as a fringe.
    A LOWER value means MORE background contamination (worse); worst_frame
    is the frame with the lowest (most contaminated) value, unlike every
    other metric here where "worst" means highest. Added because prior to
    this the harness had no way to tell despill's effect from zero -- S2/E1
    don't touch colour-vs-background-colour at all, and F1/F1i/F2 only see
    binarised alpha, not the semi-transparent edge colour despill acts on."""
    T = len(out_rgba)
    per_frame = np.full(T, np.nan, dtype=np.float64)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(T):
        opaque = (out_rgba[i, :, :, 3] > ALPHA_THR).astype(np.uint8)
        eroded = cv2.erode(opaque, kernel, iterations=ring)
        edge_band = (opaque - eroded).astype(bool)
        if not edge_band.any():
            continue
        d = chroma_distance(out_rgba[i, :, :, :3], bg)
        per_frame[i] = float(d[edge_band].mean())
    valid = per_frame[~np.isnan(per_frame)]
    if not valid.size:
        return {"mean": 0.0, "worst_frame": 0, "worst_value": 0.0}
    worst_idx = int(np.nanargmin(per_frame))
    return {"mean": float(valid.mean()), "worst_frame": worst_idx,
            "worst_value": float(per_frame[worst_idx])}


def soft_alpha_band_frac(out_rgba, ring=2):
    """A2 (第9計画 A0): fraction of boundary-ring pixels whose alpha is
    strictly between 0 and 255. Every other metric here binarises alpha at
    ALPHA_THR, which is exactly why the past V1-V4/第8回 investigations
    (1-bit GIF threshold flips from BiRefNet's own run-to-run
    nondeterminism) had no metric that could SHOW the thing an 8-bit
    encoder (WebP/MOV) is supposed to fix: whether the boundary actually
    still carries fractional alpha, or got collapsed to 0/255 by the
    encoder regardless of what BiRefNet produced. A 1-bit GIF always
    scores ~0 here by construction; an 8-bit encoder should score > 0
    wherever BiRefNet's own alpha was genuinely soft at the edge. This is
    NOT a fidelity gate on its own (a higher value isn't "better" in
    isolation -- it just proves soft alpha survived encoding); pair it
    with A1_soft_alpha_sad when raw pre-encode alpha is available."""
    T = len(out_rgba)
    per_frame = np.zeros(T, dtype=np.float64)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(T):
        opaque = (out_rgba[i, :, :, 3] > ALPHA_THR).astype(np.uint8)
        dilated = cv2.dilate(opaque, kernel, iterations=ring)
        eroded = cv2.erode(opaque, kernel, iterations=ring)
        band = (dilated - eroded).astype(bool)
        if not band.any():
            continue
        a = out_rgba[i, :, :, 3][band]
        per_frame[i] = float(((a > 0) & (a < 255)).mean())
    return {"mean": float(per_frame.mean()), "worst_frame": int(np.argmax(per_frame)),
            "worst_value": float(per_frame.max())}


def soft_alpha_sad(out_rgba, raw_alpha):
    """A1 (第9計画 A0): mean absolute difference (0-255 scale) between the
    OUTPUT's alpha channel and the RAW pre-encode alpha BiRefNet actually
    produced (raw_alpha: (T,H,W) uint8 or float in [0,1]/[0,255], same
    resolution as out_rgba). Requires the caller to have the raw alpha on
    hand (e.g. from tool.pipeline.cache.infer_clip_cached) -- there is no
    way to recover it from an already-encoded file. A 1-bit encoder will
    show a large SAD wherever BiRefNet's raw alpha was genuinely
    fractional; an 8-bit encoder that preserves soft alpha faithfully
    (not just "some soft alpha survived", which is what
    soft_alpha_band_frac measures) should score near zero."""
    raw = np.asarray(raw_alpha)
    if raw.dtype != np.uint8:
        raw = np.clip(raw, 0, 1) * 255.0 if raw.max() <= 1.0 + 1e-6 else raw
    raw = raw.astype(np.float64)
    out_a = out_rgba[:, :, :, 3].astype(np.float64)
    if raw.shape != out_a.shape:
        raise ValueError(f"raw_alpha shape {raw.shape} != output alpha shape {out_a.shape}")
    per_frame = np.abs(out_a - raw).mean(axis=(1, 2))
    return {"mean": float(per_frame.mean()), "worst_frame": int(np.argmax(per_frame)),
            "worst_value": float(per_frame.max())}


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

def evaluate(clip_name, source_path, output_path, bg: BgStats | None = None, raw_alpha=None):
    source_rgb = read_source_rgb(source_path)
    out_rgba, H, W = read_rgba_frames(output_path)
    if source_rgb.shape[1:3] != (H, W):
        # The pipeline's own --scale/--max-side (see tool/pipeline/config.py)
        # legitimately delivers a smaller frame than its source -- resize
        # the SOURCE down to the output's resolution before any per-pixel
        # comparison. Every fidelity/stability metric here is pixel-paired,
        # so this must happen before anything else touches source_rgb.
        source_rgb = np.stack(
            [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in source_rgb], 0)
    if bg is None:
        bg = estimate_bg_stats(source_rgb)
    # F1/F1i/F2/E2 all measure distance-to-the-estimated-backdrop-colour --
    # meaningless (and actively misleading) on a clip whose backdrop isn't
    # a flat colour to begin with, since `bg` is then just the mean of
    # whatever varied colours happened to be in the border ring, not an
    # actual backdrop. Report None rather than a number that LOOKS like a
    # real measurement (see the plan's G4 finding: this pipeline was
    # applied to a natural-background clip with no gate catching it).
    is_chroma = bg.is_chroma_class
    return {
        "clip": clip_name,
        "bg_is_chroma_class": is_chroma,
        "F1_false_erase": false_erase(source_rgb, out_rgba, bg) if is_chroma else None,
        "F1i_false_erase_interior": false_erase_interior(source_rgb, out_rgba, bg) if is_chroma else None,
        "F2_false_keep": false_keep(source_rgb, out_rgba, bg) if is_chroma else None,
        "F3_interior_holes": interior_holes(out_rgba),
        "S1_mc_chatter": mc_chatter(source_rgb, out_rgba),
        "S2_color_flicker": color_flicker(source_rgb, out_rgba),
        "S3_frozen_px": frozen_px(out_rgba),
        "S4_area_jump": area_jump(out_rgba),
        "E1_perimeter_ratio": perimeter_ratio(out_rgba),
        "E2_fringe_quality": fringe_chroma_distance(out_rgba, bg) if is_chroma else None,
        "A2_band_soft_frac": soft_alpha_band_frac(out_rgba),
        "A1_soft_alpha_sad": soft_alpha_sad(out_rgba, raw_alpha) if raw_alpha is not None else None,
    }

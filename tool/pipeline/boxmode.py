"""Crop-box time stabilization (レバーC, 第9計画 section 6.C).

Two independent pieces, both driven by `config.box_mode`:

1. `resolve_boxes`: turns a clip's per-frame YOLOX detections (pass 1, run
   over every frame up front -- see tool/pipeline/runner.py's infer_clip)
   into the boxes BiRefNet's pass 2 actually crops to. "per_frame" (the
   default) reproduces the old single-pass behaviour exactly -- each frame
   keeps its own independently-detected box. "clip_union" and "smoothed"
   are the two variants registered for the box_mode experiment (plan
   section 6.C); see their docstrings below.

2. `repair_truncated_regions`: replaces the old per-frame "one truncated
   frame jumps straight to a full-frame BiRefNet pass" behaviour
   (matte_core.Models.birefnet's own auto_full_frame_fallback=True path,
   still used by every OTHER caller, e.g. the still-image pipeline) with a
   clip-level fix: group truncated frames into contiguous regions, grow the
   box for the whole region, and re-run BiRefNet over just that region with
   the grown box. A single full-screen-box frame in the middle of an
   otherwise tightly-cropped clip was a visible S1 (chatter) spike -- the
   subject's effective resolution changes abruptly for one frame and snaps
   back the next; a grown-but-still-cropped box across the whole affected
   region is a gentler, less visible correction.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# box_mode: per_frame / clip_union / smoothed
# ---------------------------------------------------------------------

def resolve_boxes(raw_boxes, box_mode: str):
    """raw_boxes: one entry per frame, each either an xyxy box (anything
    array-like) or None (subject_box found nothing / rejected it via
    min_box_frac). Returns a same-length list in the same format."""
    if box_mode == "per_frame":
        return list(raw_boxes)
    if not raw_boxes:
        return list(raw_boxes)
    if box_mode == "clip_union":
        return _clip_union(raw_boxes)
    if box_mode == "smoothed":
        return _smoothed(raw_boxes)
    raise ValueError(f"unknown box_mode: {box_mode!r}")


def _clip_union(raw_boxes):
    """One fixed box for the whole clip: the union (min x0/y0, max x1/y1)
    of every frame's detected box. Simple and immune to jitter -- but a
    subject that travels far across the clip inflates this toward the full
    frame, losing the effective super-resolution a tight per-frame crop
    gives BiRefNet's fixed 1024^2 input."""
    valid = [b for b in raw_boxes if b is not None]
    if not valid:
        return [None] * len(raw_boxes)
    arr = np.stack([np.asarray(b, dtype=np.float64) for b in valid])
    union = np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()])
    return [union] * len(raw_boxes)


def _fill_gaps(raw_boxes):
    """Forward-fill then back-fill None entries from the nearest valid
    neighbour so every frame has SOME box to feed the median filter. A clip
    with zero detections on every frame stays all-None (full-frame
    everywhere), matching per_frame's own "nothing detected" behaviour."""
    filled = list(raw_boxes)
    last = None
    for i in range(len(filled)):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    last = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    return filled


def _median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


def _smoothed(raw_boxes, window: int = 15, shrink_hold: int = 5):
    """Per-frame box, median-filtered on centre/size to kill jitter, with
    size hysteresis.

    window=15: at typical 24-30fps this spans roughly half a second --
    enough to smooth frame-to-frame detector jitter (a box wobbling by a
    few percent of frame size between adjacent frames with no real subject
    motion) while still tracking genuine subject movement within about
    that same half second.

    shrink_hold=5 (N in the plan): box GROWTH is applied immediately --
    never delay recovering from an under-sized box, since a crop that's too
    tight risks amputating the subject (the same concern min_box_frac
    exists for). Box SHRINKAGE only takes effect once 5 consecutive frames
    have all requested a smaller size than the currently-applied box, so a
    single-frame (or two-frame) misdetection or brief pose change that
    transiently shrinks the raw/median box doesn't cause a premature
    tightening that then clips the subject the next frame. 5 is short
    enough (well under a quarter-second at typical frame rates) that a
    genuine, sustained shrink still catches up quickly rather than leaving
    a stale, oversized box for long."""
    n = len(raw_boxes)
    filled = _fill_gaps(raw_boxes)
    if all(b is None for b in filled):
        return [None] * n

    arr = np.array([np.asarray(b, dtype=np.float64) for b in filled])
    cx = (arr[:, 0] + arr[:, 2]) / 2.0
    cy = (arr[:, 1] + arr[:, 3]) / 2.0
    w = arr[:, 2] - arr[:, 0]
    h = arr[:, 3] - arr[:, 1]

    cx_s = _median_filter_1d(cx, window)
    cy_s = _median_filter_1d(cy, window)
    w_s = _median_filter_1d(w, window)
    h_s = _median_filter_1d(h, window)

    applied_w, applied_h = w_s[0], h_s[0]
    shrink_run = 0
    out_boxes = []
    for i in range(n):
        cand_w, cand_h = w_s[i], h_s[i]
        if cand_w * cand_h >= applied_w * applied_h:
            applied_w, applied_h = cand_w, cand_h
            shrink_run = 0
        else:
            shrink_run += 1
            if shrink_run >= shrink_hold:
                applied_w, applied_h = cand_w, cand_h
                shrink_run = 0
        out_boxes.append(np.array([
            cx_s[i] - applied_w / 2.0, cy_s[i] - applied_h / 2.0,
            cx_s[i] + applied_w / 2.0, cy_s[i] + applied_h / 2.0,
        ]))
    return out_boxes


# ---------------------------------------------------------------------
# clip-level truncation repair (replaces the single-frame full-screen jump)
# ---------------------------------------------------------------------

def _group_regions(truncated, gap_merge: int = 3):
    """Contiguous runs of True in `truncated`, merging two runs separated by
    a gap of `gap_merge` frames or fewer into one region -- a short gap
    between two truncated runs is cheaper to bridge with a single grown-box
    region than to leave as two separately-regrown regions with a few
    untouched (and then abruptly differently-sized) frames between them.
    Returns a list of inclusive (lo, hi) index pairs."""
    idxs = [i for i, t in enumerate(truncated) if t]
    if not idxs:
        return []
    regions = []
    lo = hi = idxs[0]
    for i in idxs[1:]:
        if i - hi <= gap_merge + 1:
            hi = i
        else:
            regions.append((lo, hi))
            lo = hi = i
    regions.append((lo, hi))
    return regions


def _grow_box(box, shape, grow: float = 0.5):
    """Box grown by `grow` (50%) on each side, centred on the same point,
    clamped to the frame -- the region-level analogue of birefnet()'s own
    `margin`, just larger, since this is specifically trying to stop
    re-cutting off the subject that just got truncated."""
    H, W = shape[:2]
    bx = np.asarray(box, dtype=np.float64)
    cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
    w, h = (bx[2] - bx[0]) * (1.0 + grow), (bx[3] - bx[1]) * (1.0 + grow)
    return np.array([
        max(0.0, cx - w / 2.0), max(0.0, cy - h / 2.0),
        min(float(W), cx + w / 2.0), min(float(H), cy + h / 2.0),
    ])


def repair_truncated_regions(models, rgb_frames, boxes, alphas, truncated,
                              margin: float, matte_refine: bool, gap_merge: int = 3, grow: float = 0.5):
    """Mutates and returns `alphas` in place: for every contiguous region of
    truncated frames, re-run BiRefNet's crop pass with a box grown from the
    union of that region's own per-frame boxes; if the grown box STILL
    looks truncated (rare), fall back to a full-frame pass for just that
    frame (the same ultimate safety net birefnet() itself uses, just scoped
    to the frames that actually need it instead of firing independently per
    frame). Returns (alphas, reinfer_count) -- reinfer_count is added to
    `models.box_reinfer_frames` so the existing timings/gate plumbing
    (runner.infer_clip, the box_mode experiment's harness comparison) is
    unchanged in shape."""
    from ..matte_core import _alpha_touches_a_crop_edge, _matte_refine, _normalize_alpha

    regions = _group_regions(truncated, gap_merge)
    reinfer = 0
    for lo, hi in regions:
        region_idxs = range(lo, hi + 1)
        region_boxes = [boxes[i] for i in region_idxs if boxes[i] is not None]
        union = None
        if region_boxes:
            arr = np.stack([np.asarray(b, dtype=np.float64) for b in region_boxes])
            union = np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()])
        for i in region_idxs:
            rgb = rgb_frames[i]
            H, W = rgb.shape[:2]
            grown = _grow_box(union, rgb.shape, grow=grow) if union is not None else None
            out, x0, y0, x1, y1 = models._birefnet_crop_pass(rgb, grown, margin)
            if grown is not None and _alpha_touches_a_crop_edge(out, x0, y0, x1, y1, W, H):
                out, x0, y0, x1, y1 = models._birefnet_crop_pass(rgb, None, margin)
            a = _normalize_alpha(out, H, W)
            if matte_refine:
                a = _matte_refine(rgb, a)
            alphas[i] = a
            reinfer += 1
    if reinfer:
        models.box_reinfer_frames = getattr(models, "box_reinfer_frames", 0) + reinfer
    return alphas, reinfer

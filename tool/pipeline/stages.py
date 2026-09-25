"""Post-processing stages for the chroma-background pipeline, as pure
functions: (frames, config, ...) -> frames, mutated in place and also
returned. No shared-module state, no monkey-patching (see config.py's
docstring and the plan's B5 finding).

Ported from scripts/run_gpu_birefnet_v24.py's _temporal_postprocess and its
helpers. `_strip_tail_shadow` is NOT ported: measured on 感謝.mp4 it erased
130,340px of the character's own white body (legs/belly/tail) vs only
10,553px of baseline mismatch without it, to remove a contact-shadow
artifact that turned out to be imperceptible on this asset (verified
visually with no strip applied at all). See run_gpu_birefnet_v24.py's own
history for the full measurement trail; it stays there as a record, not
here.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig


# --------------------------------------------------------------------------
# bg-leak-fix: candidate detection (per frame, during inference) + resolution
# (across frames, after inference)
# --------------------------------------------------------------------------

def zoom_bg_fraction(models, rgb, comp, stats_row, config: PipelineConfig):
    """Re-run BiRefNet on a window around this island only, so it occupies a
    large share of the fixed 1024^2 input, and report how much of the
    island the zoomed pass calls background."""
    H, W = rgb.shape[:2]
    x, y = int(stats_row[cv2.CC_STAT_LEFT]), int(stats_row[cv2.CC_STAT_TOP])
    w, h = int(stats_row[cv2.CC_STAT_WIDTH]), int(stats_row[cv2.CC_STAT_HEIGHT])
    side = max(config.clear_zoom_min_side, config.clear_zoom_window_mult * max(w, h))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(W, x0 + side), min(H, y0 + side)
    x0, y0 = max(0, x1 - side), max(0, y1 - side)
    za = models.birefnet(rgb[y0:y1, x0:x1])
    sub = comp[y0:y1, x0:x1]
    if not sub.any():
        return 0.0
    return float((za[sub] < 0.5).mean())


_ERODE_KERNEL = np.ones((5, 5), np.uint8)


def find_clear_candidates(models, rgb, a, bg_ref, config: PipelineConfig):
    """Prefilter with cheap gates (desaturated, big enough, compact, near the
    measured backdrop colour), ask a zoomed BiRefNet pass about each
    surviving island, and return them individually as (mask, verdict, area)
    so the caller can link them across frames."""
    fg = (a > 0.5).astype(np.uint8)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    low_sat_fg = (fg & (sat < config.clear_sat_thresh)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low_sat_fg, connectivity=8)
    if n <= 1:
        return []
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < config.clear_min_area_link:
            continue
        comp = (labels == i)
        core = cv2.erode(comp.astype(np.uint8), _ERODE_KERNEL).astype(bool)
        if not core.any():
            continue
        if bg_ref is not None and np.linalg.norm(rgb[core].mean(axis=0) - bg_ref) > config.clear_dist_thresh:
            continue
        out.append((comp.astype(np.uint8),
                    zoom_bg_fraction(models, rgb, comp, stats[i], config),
                    int(stats[i, cv2.CC_STAT_AREA])))
    return out


def apply_clears(frames, clear_masks, config: PipelineConfig):
    """Resolve bg-leak-fix candidates across the whole clip: link islands
    into tracks (allowing a short gap so a momentary dip below the area
    floor doesn't split one gap into two tracks), then clear every frame of
    a track that's mostly zoom-confirmed AND whose confirmed islands
    include one at or above the area floor (the floor applies to the
    CONFIRMED members specifically -- see config.py / the v24 history for
    why "any member is big" is unsafe)."""
    if not clear_masks:
        return frames
    n = len(frames)
    nodes = []
    for i in sorted(clear_masks):
        if i < n:
            for mask, verdict, area in clear_masks[i]:
                nodes.append([i, mask, verdict, area])
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for u in range(len(nodes)):
        for v in range(u + 1, len(nodes)):
            dt = nodes[v][0] - nodes[u][0]
            if dt < 1 or dt > config.clear_track_max_gap:
                continue
            a_m, b_m = nodes[u][1].astype(bool), nodes[v][1].astype(bool)
            inter = int((a_m & b_m).sum())
            if inter and inter >= config.clear_track_overlap * min(a_m.sum(), b_m.sum()):
                union(u, v)

    tracks = {}
    for idx in range(len(nodes)):
        tracks.setdefault(find(idx), []).append(idx)

    cleared_px, fired, rescued = 0, [], 0
    for members in tracks.values():
        strong = [m for m in members if nodes[m][2] >= config.clear_zoom_bg_frac]
        if not strong or len(strong) < config.clear_track_strong_frac * len(members):
            continue
        if max(nodes[m][3] for m in strong) < config.clear_min_area:
            continue
        for m in members:
            i, mask, verdict, _area = nodes[m]
            frames[i][:, :, 3][mask.astype(bool)] = 0
            cleared_px += int(mask.sum())
            fired.append(i)
            if verdict < config.clear_zoom_bg_frac:
                rescued += 1
    fired = sorted(set(fired))
    print(f"    [bg-leak-fix] cleared {cleared_px} px on frames {fired}"
          + (f" ({rescued} island(s) carried by their track -- zoom was unsure)"
             if rescued else ""), flush=True)
    return frames


# --------------------------------------------------------------------------
# single-frame area-collapse repair
# --------------------------------------------------------------------------

def collapse_fix(frames, config: PipelineConfig):
    """Replace a frame whose opaque area collapses to <60% of BOTH neighbours
    with a copy of the previous frame -- catches a single bad keyframe
    (e.g. a transient detector misfire) without needing to know why it
    happened."""
    areas = [int((f[:, :, 3] > 127).sum()) for f in frames]
    n_fixed = 0
    for i in range(1, len(frames) - 1):
        if (areas[i] < config.collapse_area_ratio * areas[i - 1] and
                areas[i] < config.collapse_area_ratio * areas[i + 1]):
            frames[i] = frames[i - 1].copy()
            n_fixed += 1
            print(f"    [collapse-fix] frame {i}: area {areas[i]} -> replaced with frame {i-1} "
                  f"(neighbours: {areas[i-1]}, {areas[i+1]})", flush=True)
    if n_fixed:
        print(f"    total collapse-corrected frames: {n_fixed}", flush=True)
    return frames


# --------------------------------------------------------------------------
# temporal alpha smoothing (+ optional colour-freeze on still pixels)
# --------------------------------------------------------------------------

def temporal_alpha_smooth(frames, src_gray, motion_dmax, config: PipelineConfig, progress=None):
    """Motion-compensated median (warp each frame's alpha neighbours in via
    DIS optical flow before taking the median, so real motion isn't treated
    as chatter) plus an optional freeze-on-never-changing-pixels for alpha.

    If config.colour_freeze is True, ALSO freezes the RGB channels on those
    same still pixels -- OFF by default; see config.py's docstring for why
    (it visibly flattened a real shading region on one asset).

    `progress(done, total)`, if given, is called once per processed frame
    (this is the only sub-stage of postprocess() with real per-frame ticks
    -- the others are seconds-scale; see the plan's ETA work)."""
    n_frames = len(frames)
    if n_frames < 3 or config.mc_median_half <= 0:
        return frames
    height, width = frames[0].shape[:2]
    half = config.mc_median_half
    alphas = np.stack([f[:, :, 3] for f in frames], 0)
    dis_preset = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }[config.flow_preset]
    dis = cv2.DISOpticalFlow_create(dis_preset)
    gx, gy = np.meshgrid(np.arange(width), np.arange(height))
    gx, gy = gx.astype(np.float32), gy.astype(np.float32)
    smoothed = alphas.copy()

    # V7 speed experiment (see the plan): DIS flow at full resolution
    # dominates this function's cost (V0 profiling). flow_scale<1 computes
    # flow on a downscaled gray pair (each frame pre-downscaled ONCE here,
    # not per-pair, since it participates in up to 2*half pairs) and
    # upsamples+rescales the flow field before the still-full-res remap --
    # the remap itself (and its accuracy) is unaffected.
    flow_scale = config.flow_scale
    if flow_scale < 1.0:
        sh, sw = max(8, round(height * flow_scale)), max(8, round(width * flow_scale))
        rx, ry = width / sw, height / sh
        flow_gray = [cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA) for g in src_gray]
    else:
        flow_gray = src_gray

    n_todo = max(0, n_frames - 2 * half)
    for done, i in enumerate(range(half, n_frames - half), start=1):
        buf = [alphas[i]]
        for k in range(1, half + 1):
            for j in (i - k, i + k):
                fl = dis.calc(flow_gray[i], flow_gray[j], None)
                if flow_scale < 1.0:
                    fl = cv2.resize(fl, (width, height), interpolation=cv2.INTER_LINEAR)
                    fl[..., 0] *= rx
                    fl[..., 1] *= ry
                buf.append(cv2.remap(alphas[j], gx + fl[..., 0], gy + fl[..., 1],
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        if half == 1:
            # Audit finding (postprocess speed lever): half=1 (the only
            # value this pipeline's presets/config ever actually use --
            # PipelineConfig's own default, and results_dinosaur/ was
            # produced under it) makes `buf` exactly 3 arrays, so the
            # median is exactly the sorted middle element -- computable via
            # min/max without np.median's np.stack + full sort + float64
            # promotion + cast-back-to-uint8 round trip. Bit-exact with the
            # np.median path below (same value, same eventual uint8 cast);
            # see test_stages_flow.py's direct comparison.
            lo = np.minimum(buf[0], buf[1])
            hi = np.maximum(buf[0], buf[1])
            smoothed[i] = np.maximum(lo, np.minimum(hi, buf[2]))
        else:
            smoothed[i] = np.median(np.stack(buf, 0), axis=0)
        if progress is not None:
            progress(done, n_todo)

    still = None
    if motion_dmax is not None and config.still_thresh:
        still = motion_dmax < config.still_thresh
        frozen = np.median(alphas, axis=0).astype(alphas.dtype)
        smoothed[:, still] = frozen[still][None, :]

    changed = int((smoothed != alphas).sum())
    for i, f in enumerate(frames):
        f[:, :, 3] = smoothed[i]
    n_still = int(still.sum()) if still is not None else 0

    if config.colour_freeze and still is not None:
        rgb_changed = 0
        for c in range(3):
            chan = np.stack([f[:, :, c] for f in frames], 0)
            med = np.median(chan, axis=0).astype(chan.dtype)
            col = med[still]
            for f in frames:
                view = f[:, :, c]
                rgb_changed += int((view[still] != col).sum())
                view[still] = col
        print(f"    [temporal] colour frozen on {n_still} px (rgb px rewritten: {rgb_changed})",
              flush=True)

    print(f"    [temporal] motion-compensated median +/-{half} on moving px, "
          f"frozen on {n_still} never-changing px; alpha px rewritten: {changed}", flush=True)
    return frames


# --------------------------------------------------------------------------
# background-colour fringe strip (narrow, hue+distance gated)
# --------------------------------------------------------------------------

def _fringe_candidate_mask(f, bg_ref, bg_hue, config: PipelineConfig):
    a = f[:, :, 3] > 127
    rgb = f[:, :, :3]
    dist = np.linalg.norm(rgb.astype(np.float32) - bg_ref[None, None, :], axis=2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    hue_close = np.abs(((hue - bg_hue + 90) % 180) - 90) <= config.fringe_hue_thresh
    return a & (dist < config.fringe_dist_thresh) & hue_close


def _fringe_candidate_masks(frames, bg_ref, config: PipelineConfig):
    """Per-frame fringe candidate masks, computed once so
    fringe_strip_is_safe and strip_bg_fringe (called back-to-back by
    postprocess(), the ONLY real caller of either) don't each loop over
    every frame independently re-deriving the identical hue-convert +
    distance + hue-match masks (audit finding: this was a full duplicate
    pass over the clip whenever strip_bg_fringe was enabled)."""
    if bg_ref is None:
        return None, None
    bg_hue = int(cv2.cvtColor(np.uint8([[bg_ref]]), cv2.COLOR_RGB2HSV)[0, 0, 0])
    return bg_hue, [_fringe_candidate_mask(f, bg_ref, bg_hue, config) for f in frames]


def fringe_strip_is_safe(frames, bg_ref, config: PipelineConfig, *, candidate_masks=None):
    """Self-check run BEFORE strip_bg_fringe actually zeroes anything:
    measure what fraction of its candidate pixels sit deep inside
    confidently-opaque subject interior (eroded ~1% of the frame diagonal
    in from the alpha boundary) rather than near the true edge.

    Exists because this rule's hue+distance signature, tuned safe on one
    asset, is not guaranteed safe on another -- measured directly: on the
    dinosaur clips (yellow backdrop / teal subject) this fraction is
    ~0.1-0.5%; on the purplebg batch (purple backdrop / navy suit) it exceeded
    10% because the dark suit fabric's hue happened to fall within
    fringe_hue_thresh of the backdrop. Every colour heuristic this pipeline
    has ever shipped with a static ON default eventually erased real
    subject content on SOME asset (see config.py's strip_bg_fringe
    docstring) -- this makes that failure self-detecting instead of
    something a human has to notice by running the harness.

    `candidate_masks`, if given (see _fringe_candidate_masks), skips
    recomputing the per-frame candidate mask -- an optional optimisation
    for postprocess()'s own back-to-back call into strip_bg_fringe right
    after this; a standalone caller passing frames+bg_ref alone still works
    exactly as before."""
    if bg_ref is None:
        return True, 0.0
    if candidate_masks is None:
        _, candidate_masks = _fringe_candidate_masks(frames, bg_ref, config)
    h, w = frames[0].shape[:2]
    r = max(1, int(round(float(np.hypot(h, w)) * 0.01)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    interior_hits = 0
    total_opaque = 0
    for f, candidate in zip(frames, candidate_masks):
        a_bin = (f[:, :, 3] > 127).astype(np.uint8)
        interior = cv2.erode(a_bin, kernel) > 0
        interior_hits += int((candidate & interior).sum())
        total_opaque += int(a_bin.sum())
    frac = interior_hits / max(total_opaque, 1)
    return frac <= config.fringe_safety_max_frac, frac


def strip_bg_fringe(frames, bg_ref, config: PipelineConfig, *, candidate_masks=None):
    """Remove opaque pixels along the silhouette edge that are actually a
    thin fringe of the backdrop colour itself surviving 1-bit alpha
    binarisation. Deliberately narrow (tight distance AND hue match) --
    see the plan's chroma-key experiment for why a loose colour-only rule
    over-reaches; this one only ever touches genuine near-backdrop pixels
    -- ASSUMING fringe_strip_is_safe() passed for this clip; postprocess()
    checks that before calling this.

    `candidate_masks`: see fringe_strip_is_safe's docstring -- the same
    optional precomputed-mask reuse."""
    if bg_ref is None:
        return frames
    if candidate_masks is None:
        _, candidate_masks = _fringe_candidate_masks(frames, bg_ref, config)
    for f, fringe in zip(frames, candidate_masks):
        f[:, :, 3][fringe] = 0
    return frames


# --------------------------------------------------------------------------
# top-level: run every enabled stage in the fixed, safe order
# --------------------------------------------------------------------------

def _component_keep_labels(lab, stats, box, min_frac, sam2_mask=None, config: PipelineConfig | None = None):
    """Labels (1..n-1) of the connected components keep_main_subject keeps.

    `sam2_mask`, if given, is a {0,1}/float [0,1] (H,W) SAM2 person mask for
    this frame (第12計画): rule (iii) becomes "component overlaps the dilated
    SAM2 mask by >= keep_main_subject_sam2_overlap", replacing the old
    "touches the subject's raw YOLOX box" rule, which could not separate a
    prop fully inside a box from the person (the widepose clip's frame 110's
    bottle -- see DECISIONS.md). Falls back to the box-rectangle rule (iii)
    only when sam2_mask is None or empty (SAM2 genuinely found nothing) --
    both return a second value: whether the SAM2 path was actually used for
    this frame's rule (iii), for the caller to count fallback frames.

    第13計画 (2026-09-24): the box-area-ratio health check (a lower/upper
    bound on mask_area/box_area) that used to gate this was REMOVED after
    calibration data showed it was the actual bug, not a safety net: on
    the widepose clip's own frames (105-110, 177-182 -- the exact
    frames this whole lever exists to fix), the SAM2 mask area legitimately
    sits below any box-area-normalized floor (a spread-limbs bounding box
    is mostly empty space), even though the mask itself was measured to be
    CORRECT there (its area matched the person's own connected component,
    and its overlap with every genuine prop component was ALWAYS 0.000
    across 62 real prop components -- see DECISIONS.md's calibration table).
    The floor rejected a correct mask on exactly the frames it needed to
    accept one. Across all 90 calibration decision-points (6 clips), SAM2
    never returned a literally empty mask, so "empty -> fallback" is a
    real (if rarely-firing) backstop, not a disguised re-introduction of
    the removed floor."""
    config = config or PipelineConfig()
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= min_frac * areas.max()  # (i) largest + (ii) comparable ones
    used_sam2 = False
    if box is not None:
        h, w = lab.shape
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        mask_ok = sam2_mask is not None and np.any(sam2_mask > 0.5)
        if mask_ok:
            used_sam2 = True
            d = config.keep_main_subject_sam2_dilate_px
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
            dilated = cv2.dilate((sam2_mask > 0.5).astype(np.uint8), kernel).astype(bool)
            overlap_thr = config.keep_main_subject_sam2_overlap
            comp_areas = stats[1:, cv2.CC_STAT_AREA]
            for lbl in range(1, lab.max() + 1):
                if keep[lbl - 1]:
                    continue
                comp_px = comp_areas[lbl - 1]
                if comp_px <= 0:
                    continue
                overlap_px = int(np.count_nonzero(dilated[lab == lbl]))
                if overlap_px / comp_px >= overlap_thr:
                    keep[lbl - 1] = True
        elif x1 > x0 and y1 > y0:
            inside = np.unique(lab[y0:y1, x0:x1])
            inside = inside[inside > 0]
            keep[inside - 1] = True  # (iii) fallback: touches the subject's own box
    return np.flatnonzero(keep) + 1, used_sam2


def keep_main_subject(frames, boxes=None, config: PipelineConfig | None = None, sam2_fn=None):
    """Zero the alpha of every opaque component that isn't the main subject
    (see config.keep_main_subject's comment for the rule). `boxes`: one
    entry per frame, an (x0, y0, x1, y1) YOLOX box before crop margin, or
    None / NaN row when that frame had none; None altogether on the keyer
    route. `sam2_fn(rgb, box) -> (H,W) mask or None`, if given (第12計画),
    is called ONLY for frames where rule (iii) can actually change the
    outcome (a box exists and rules (i)/(ii) don't already keep every
    component) -- on frames where every component is already kept, calling
    SAM2 would be pure cost for no effect. Returns (frames, removed_px_per_frame,
    sam2_used_count, sam2_fallback_count)."""
    config = config or PipelineConfig()
    min_frac = config.keep_main_subject_min_frac
    r = int(config.keep_main_subject_rim_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) if r > 0 else None
    removed = []
    sam2_used = 0
    sam2_fallback = 0
    for i, f in enumerate(frames):
        a = f[:, :, 3]
        n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 127).astype(np.uint8), connectivity=8)
        if n <= 2:
            removed.append(0)
            continue
        box = None
        if boxes is not None and i < len(boxes) and boxes[i] is not None:
            b = np.asarray(boxes[i], dtype=np.float64)
            if b.shape == (4,) and np.all(np.isfinite(b)):
                box = b
        areas = stats[1:, cv2.CC_STAT_AREA]
        already_all_kept = bool(np.all(areas >= min_frac * areas.max()))
        sam2_mask = None
        if sam2_fn is not None and box is not None and not already_all_kept:
            try:
                sam2_mask = sam2_fn(f[:, :, :3], box)
            except Exception as exc:  # noqa: BLE001 -- fall back to the box rule, don't kill the run
                print(f"    [keep_main_subject] SAM2 call failed on frame {i}, "
                      f"falling back to the box rule: {exc}", flush=True)
                sam2_mask = None
        keep_labels, used_sam2 = _component_keep_labels(lab, stats, box, min_frac, sam2_mask, config)
        if sam2_mask is not None:
            if used_sam2:
                sam2_used += 1
            else:
                sam2_fallback += 1
        if len(keep_labels) == n - 1:
            removed.append(0)
            continue
        keep_mask = np.isin(lab, keep_labels)
        drop_mask = (lab > 0) & ~keep_mask
        if kernel is not None:
            near_drop = cv2.dilate(drop_mask.astype(np.uint8), kernel).astype(bool)
            near_keep = cv2.dilate(keep_mask.astype(np.uint8), kernel).astype(bool)
            zero = drop_mask | (near_drop & ~near_keep)
        else:
            zero = drop_mask
        removed.append(int(np.count_nonzero(drop_mask)))
        a[zero] = 0
    return frames, removed, sam2_used, sam2_fallback


def postprocess(frames, clear_masks, src_gray, motion_dmax, bg_ref, config: PipelineConfig, progress=None,
                boxes=None, sam2_fn=None):
    """Fixed stage order. Colour-fix/strip stages run LAST, after temporal
    smoothing: the motion-compensated median mixes in neighbouring frames'
    alpha, so anything stripped earlier gets partially pulled back in from
    frames where it hadn't been stripped yet (see v19's fix for this exact
    bug, in the run_gpu_birefnet_v*.py history).

    V0 (speed audit, 2026-09-02): per-stage timing, gated behind a module-
    level flag rather than always printing -- this is a debug/profiling
    aid, not something every run's stdout should carry.

    `progress(done, total)`, if given, is forwarded ONLY to mc_median (the
    only sub-stage with real per-frame ticks; the others are seconds-scale
    -- see the plan's ETA work). `boxes`: infer_clip's per-frame raw YOLOX
    boxes (raw["boxes"]), read only by keep_main_subject. `sam2_fn`
    (第12計画): forwarded to keep_main_subject's rule (iii) refinement,
    None means the box-rectangle fallback is used for every frame."""
    import time
    t = {}

    def _timed(name, fn, *a):
        t0 = time.perf_counter()
        fn(*a)
        t[name] = time.perf_counter() - t0

    if config.apply_clears:
        _timed("apply_clears", apply_clears, frames, clear_masks, config)
    if config.collapse_fix:
        _timed("collapse_fix", collapse_fix, frames, config)
    if config.mc_median_half > 0:
        t0 = time.perf_counter()
        temporal_alpha_smooth(frames, src_gray, motion_dmax, config, progress=progress)
        t["mc_median"] = time.perf_counter() - t0
    if config.keep_main_subject:
        # After the temporal median (which mixes neighbours' alpha back in,
        # so anything dropped earlier could partially return -- same reason
        # strip_bg_fringe runs late), before the fringe strip.
        t0 = time.perf_counter()
        _, removed, sam2_used, sam2_fallback = keep_main_subject(frames, boxes, config, sam2_fn)
        t["keep_main_subject"] = time.perf_counter() - t0
        if any(removed):
            nz = [x for x in removed if x]
            print(f"    [keep_main_subject] dropped non-subject components on {len(nz)}/{len(frames)} "
                  f"frame(s), {sum(nz)} opaque px total", flush=True)
        if sam2_used or sam2_fallback:
            print(f"    [keep_main_subject] SAM2 rule (iii): used on {sam2_used} frame(s), "
                  f"fell back to the box rectangle on {sam2_fallback} frame(s)", flush=True)
    if config.strip_bg_fringe:
        t0 = time.perf_counter()
        _, candidate_masks = _fringe_candidate_masks(frames, bg_ref, config)
        safe, frac = fringe_strip_is_safe(frames, bg_ref, config, candidate_masks=candidate_masks)
        if safe:
            strip_bg_fringe(frames, bg_ref, config, candidate_masks=candidate_masks)
        else:
            print(f"    [safety] strip_bg_fringe auto-disabled for this clip: "
                  f"{frac:.2%} of its candidate pixels sit deep inside confident "
                  f"subject interior (threshold {config.fringe_safety_max_frac:.2%}) "
                  f"-- its hue+distance rule is not safe for this clip's backdrop/"
                  f"subject colour pairing", flush=True)
        t["strip_bg_fringe"] = time.perf_counter() - t0
    if t:
        print(f"    [timing] postprocess stages: "
              + ", ".join(f"{k} {v:.1f}s" for k, v in t.items()), flush=True)
    return frames

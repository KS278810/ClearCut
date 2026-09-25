"""End-to-end runner for the chroma-background video pipeline: one function,
run_clip(), that owns the whole per-clip flow. No shared-module monkey-
patching (see config.py's docstring for why that matters) -- everything is
threaded through explicit arguments and a single PipelineConfig.

This intentionally does NOT reuse bg_remove_video.py's process_video(): that
function's value is routing between the person/SAM2/flow-propagation paths
for GENERAL video, none of which this pipeline's content ever takes (no
person route, no SAM2 region gate -- see the plan's simplification lever
S5). Reusing it here would mean carrying that whole surface area just to
immediately disable most of it, as every scripts/run_gpu_birefnet_v*.py
version had to.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np

from .. import ffmpeg_encoders as ffenc
from ..matte_core import Models, despill, _matte_refine, _Sam2Worker, resolve_device
from . import boxmode
from . import cache
from . import chroma
from . import keyer
from .config import PipelineConfig
from .stages import find_clear_candidates, postprocess


class JobCancelled(Exception):
    """Raised by infer_clip/run_clip when `cancel` (anything with
    `.is_set()`, e.g. a threading.Event) is set mid-run -- the server's way
    of stopping a job between frames/stages without killing the whole
    process. ffmpeg_encoders.EncodeCancelled (raised inside the encode
    stage, which needs its own check since it's a separate subprocess) is
    caught and re-raised as this type in _encode so every caller only ever
    needs to catch ONE cancellation type regardless of which stage it
    happened in."""


class ClipTooLong(Exception):
    """Raised by infer_clip when `max_frames` is given and the clip exceeds
    it. This is defense IN DEPTH behind server/probe.py's own upload-time
    frame-count check, not a replacement for it: a webm (or fragmented mov)
    container can report CAP_PROP_FRAME_COUNT as 0 or -1, which probe.py
    falls back to ffprobe for -- but if that ever comes back wrong too (or
    this function is called some other way that skipped probing entirely),
    this catches the actual frame count from the inside, before infer_clip's
    frame list has grown large enough to matter."""


def resolve_scale(config: PipelineConfig, native_w: int, native_h: int) -> float:
    """`config.scale` wins if explicitly set (!= 1.0); otherwise derive a
    scale from `config.max_side` for THIS clip's native resolution (a
    1440x1440 clip and a 1656x1248 clip need different multipliers to both
    land at the same delivered long edge). Never upscales (capped at 1.0)."""
    if config.scale != 1.0:
        return config.scale
    if config.max_side:
        long_side = max(native_w, native_h)
        return min(1.0, config.max_side / long_side)
    return 1.0


def sample_bg_ref(video_path):
    """Robust-ish single-frame corner sample of the flat backdrop colour.
    Phase 4's trimap work supersedes this with a full-clip, alpha-excluded,
    sigma-clipped estimate (tool/qc/metrics.py's estimate_bg_stats); this
    stays as the simple version the rest of this pipeline's stages (which
    predate Phase 4) were tuned against."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        return None
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    H0, W0 = frame0.shape[:2]
    p = 40
    corners = [frame0[0:p, 0:p], frame0[0:p, W0 - p:W0],
               frame0[H0 - p:H0, 0:p], frame0[H0 - p:H0, W0 - p:W0]]
    return np.mean([c.reshape(-1, 3).mean(axis=0) for c in corners], axis=0)


def _track_source_motion(rgb, motion_state):
    """Running max of the blurred per-frame max-channel delta, used to
    classify a pixel as 'never really changes' for the alpha freeze."""
    prev = motion_state["prev"]
    if prev is not None:
        d = np.abs(rgb.astype(np.int16) - prev).max(axis=2).astype(np.float32)
        d = cv2.GaussianBlur(d, (0, 0), 3.0)
        if motion_state["dmax"] is None:
            motion_state["dmax"] = d
        else:
            np.maximum(motion_state["dmax"], d, out=motion_state["dmax"])
    motion_state["prev"] = rgb.astype(np.int16)


#: How many frames keyer.estimate_key gets to build its model from. It only
#: reads each frame's border ring, and the backdrop of a clip this path
#: accepts is flat by definition, so the estimate converges almost
#: immediately -- 24 spread across the clip is plenty while bounding peak
#: memory, which matters because a 1800-frame 4096px clip cannot be held in
#: RAM twice (the same reason infer_clip's trimap path discards its own
#: first-pass frame list, audit N3).
_KEY_SAMPLE_FRAMES = 24


def _sample_frames_for_key(video_path, eff_scale, total_frames, cancel=None):
    """Read up to _KEY_SAMPLE_FRAMES evenly-spaced frames for key estimation.

    `total_frames` is trusted only when it is a sane positive count above the
    sample size -- audit H3: a webm/fragmented-mov container can make
    CAP_PROP_FRAME_COUNT come back 0 or a huge negative value (confirmed:
    -9223372036854775808, an int64-min misread -- see server/probe.py's own
    comment on the same bug), and `int(that) or None` upstream turns 0 into
    None but leaves a negative count as a truthy, non-positive number. The
    OLD version's `total_frames and total_frames > _KEY_SAMPLE_FRAMES` check
    was False for such a value (a huge negative is never > 24), so it fell
    into the "read every frame, unresized, into a Python list" branch --
    for a 1800-frame 4096px clip that is tens of GB before a single frame
    gets matted, and it ran BEFORE infer_clip's own max_frames/ClipTooLong
    guard even sees a frame, so the server's upload-time frame-count limit
    provided no protection either. This version never reads more than
    _KEY_SAMPLE_FRAMES frames regardless of what total_frames claims, and
    resizes/converts each frame immediately after decoding it rather than
    holding a native-resolution list around to convert later."""
    cap = cv2.VideoCapture(str(video_path))
    sampled = []
    try:
        trustworthy = total_frames is not None and total_frames > 0
        if trustworthy and total_frames > _KEY_SAMPLE_FRAMES:
            idxs = sorted({round(k * (total_frames - 1) / (_KEY_SAMPLE_FRAMES - 1))
                           for k in range(_KEY_SAMPLE_FRAMES)})
            # Walk the stream sequentially -- grab() every frame, decode
            # (retrieve()) only the chosen ones -- instead of seeking with
            # cap.set(CAP_PROP_POS_FRAMES). A seek lands on the nearest
            # PRECEDING keyframe and decodes forward from there; the sample
            # clips (and most phone/editor exports of a short clip) have a
            # single keyframe at frame 0, so each of the 24 seeks re-decoded
            # the clip from the start -- measured 55-70s of dead time per
            # clip before any progress was reported (第11計画 Part 1-1).
            # grab() still demuxes+decodes (H.264 has no skip-decode), but
            # it does so once, in order, and skips the colour conversion of
            # frames nobody reads. Same chosen indices as before, so the
            # keyer's model (and its output) is unchanged.
            wanted = set(idxs)
            last = idxs[-1]
            idx = 0
            while idx <= last:
                if cancel is not None and cancel.is_set():
                    return None
                if not cap.grab():
                    break
                if idx in wanted:
                    ok, bgr = cap.retrieve()
                    if ok:
                        sampled.append(_scale_and_convert(bgr, eff_scale))
                idx += 1
        else:
            # total_frames is either a small reliable count (read it all --
            # already <= _KEY_SAMPLE_FRAMES) or untrustworthy (cap the read
            # at _KEY_SAMPLE_FRAMES so an unreliable count can't blow memory;
            # the clip's own frame-count/size limits, checked once real
            # inference starts, are what actually reject an oversized clip).
            while len(sampled) < _KEY_SAMPLE_FRAMES:
                if cancel is not None and cancel.is_set():
                    return None
                ok, bgr = cap.read()
                if not ok:
                    break
                sampled.append(_scale_and_convert(bgr, eff_scale))
    finally:
        cap.release()
    if not sampled:
        return None
    h0, w0 = sampled[0].shape[:2]
    sampled = [f for f in sampled if f.shape[:2] == (h0, w0)]  # a mid-stream size change shouldn't crash the stack()
    return np.stack(sampled, 0) if sampled else None


def _scale_and_convert(bgr, eff_scale):
    if eff_scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=eff_scale, fy=eff_scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resolve_keyer(video_path, config: PipelineConfig, eff_scale, total_frames, cancel=None):
    """Decide whether this clip takes the colour-only fast path, and return
    its KeyModel (or None for the BiRefNet route).

    Refusal is never silent: `use_keyer="on"` prints why it was overruled, so
    "I asked for the fast path and got the slow one" is always diagnosable.
    is_safe() is a CORRECTNESS gate, not a preference -- "on" cannot bypass
    it, because a backdrop that isn't a saturated flat colour cannot be keyed
    at all, and the failure mode is silent erasure of subject content (the
    defect class this repo has shipped three times under other names)."""
    if config.use_keyer == "off":
        return None
    sampled = _sample_frames_for_key(video_path, eff_scale, total_frames, cancel=cancel)
    if sampled is None:
        return None
    model = keyer.estimate_key(sampled, ramp_k=config.keyer_ramp_k,
                               unmix_coverage=config.keyer_unmix_coverage,
                               measured_t_lo=config.keyer_measured_t_lo)
    ok, reason = keyer.is_safe(model)
    if not ok:
        if config.use_keyer == "on":
            print(f"  [keyer] requested but refused: {reason}", flush=True)
        return None
    print(f"  [keyer] colour-only fast path: {reason}; ramp {model.t_lo:.1f}..{model.t_hi:.1f} sigma",
          flush=True)
    return model


def _load_models(config: PipelineConfig) -> Models:
    M = Models(device=config.device)
    if M._sam2_worker is not None:
        M._sam2_worker.stop()
    M._sam2_worker = None
    M.use_sam2 = False
    return M


def infer_clip(video_path, config: PipelineConfig, models: Models | None = None, *,
                progress=None, cancel=None, max_frames=None, preview=None, timings=None):
    """The expensive, upstream half of run_clip: per-frame detection +
    BiRefNet + despill (+ bg-leak-fix candidate detection, which needs a
    zoomed BiRefNet re-inference so it belongs here, not in postprocess).

    `models` may be a loaded Models, None (load one here), or a zero-arg
    callable returning one (loaded only if this clip actually needs the
    network -- see the keyer branch below).

    Split out from run_clip so its result (a RawInference) can be cached to
    disk once per clip and reused across many postprocess-stage
    combinations -- see Phase 3's ablation, which needs several configs per
    clip and would otherwise re-run GPU inference once per config for no
    reason, since collapse_fix/temporal_alpha_smooth/strip_bg_fringe (all
    in stages.postprocess) touch nothing upstream of this function.

    NOTE: apply_clears (config.apply_clears) is NOT one of those safely-
    cacheable-across fields -- candidate detection (find_clear_candidates)
    runs INSIDE this function's per-frame loop and needs its own zoomed
    BiRefNet re-inference, so it's part of the cached result, not a
    postprocess stage. See cache.py's _key, which includes it for exactly
    this reason (audit N1).

    `progress`, if given, is called as `progress("prepare", 0, 1)` on
    entry and `progress("prepare", 1, 1)` once the route (keyer or
    network) is decided and ready, then as `progress("infer", done,
    total)` once per frame (server job UI); `cancel`, if given, is checked once per
    frame (anything with `.is_set()`, e.g. threading.Event) and raises
    JobCancelled; `max_frames`, if given, raises ClipTooLong once the clip
    exceeds it (see that class's docstring); `preview`, if given, is called
    as `preview(done, frame)` once per frame with the just-produced RGBA
    frame (R/G/B/A order, uint8, (H, W, 4)) -- a live "what's being worked
    on right now" glance for the server UI, NOT the final matte: this is
    the raw per-frame BiRefNet(+trimap) alpha before postprocess's
    whole-clip stages (temporal smoothing, bg-leak-fix island removal,
    fringe stripping) ever see it, so it can show chatter/leaked islands/
    an occasional collapsed frame that the real output won't have. All
    four are optional and CLI behaviour is unchanged when none are passed
    (every existing `print` stays, for the same reason).

    `timings`, if given a dict, gets `detect_s`/`birefnet_s`/`despill_s`
    (and `trimap_s`/`clears_s` when those stages ran) written into it once
    the frame loop completes normally -- a mutation of the CALLER's dict,
    so a server job record holding it keeps these numbers even if a LATER
    stage (postprocess/encode, in run_clip) goes on to fail. A cancel or
    ClipTooLong raised mid-loop skips this write (the loop never reached
    its end), matching the existing behaviour where those exceptions carry
    no timing breakdown either."""
    video_path = Path(video_path)
    t0 = time.perf_counter()
    if progress is not None:
        # "prepare" covers everything before the first real frame: the
        # keyer decision's frame sampling, bg_ref/trimap stats, model load.
        # Emitted first thing so a server driving this has a stage from
        # t=0 (its heartbeat only publishes once a stage exists -- without
        # this the UI showed nothing at all, not even elapsed time, until
        # the first "infer" tick; 第11計画 Part 1-3).
        progress("prepare", 0, 1)
    print(f"\n{video_path.name}", flush=True)

    _probe = cv2.VideoCapture(str(video_path))
    fps = _probe.get(cv2.CAP_PROP_FPS) or 30.0
    native_w = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    _probe.release()
    eff_scale = resolve_scale(config, native_w, native_h)
    if eff_scale != 1.0:
        print(f"  scale {eff_scale:.3f} ({native_w}x{native_h} -> "
              f"{round(native_w*eff_scale)}x{round(native_h*eff_scale)})", flush=True)

    # Decided BEFORE loading Models: on the keyer path no network is used at
    # all, so loading BiRefNet/YOLOX would be pure cost (and on the server it
    # would pin GPU memory for a job that never touches the GPU).
    key_model = _resolve_keyer(video_path, config, eff_scale, total_frames, cancel=cancel)
    # Only the BiRefNet route's stages consume bg_ref (strip_bg_fringe /
    # apply_clears); the keyer reports its own, better-estimated backdrop
    # colour instead, so this extra decode is skipped on that path.
    bg_ref = key_model.bg_rgb if key_model is not None else sample_bg_ref(video_path)
    own_models = models is None
    if key_model is None:
        if own_models:
            models = _load_models(config)
        elif callable(models):
            # A caller processing several clips can pass a zero-arg factory
            # instead of an instance, so the network is loaded lazily on the
            # first clip that actually needs it and then reused -- a batch of
            # flat-chroma clips never loads it at all. Passing an instance
            # (the long-standing form) still works unchanged.
            models = models()

    bg_stats = None
    if config.use_trimap and key_model is None:
        # Full-clip, sigma-clipped background estimate needs every frame's
        # border ring -- a SEPARATE read pass, immediately discarded, rather
        # than holding the whole clip's frames in memory for the rest of
        # this function too (audit N3: holding both the full RGB clip AND
        # the RGBA output frames roughly doubles this function's peak
        # memory for every clip, including the common use_trimap=False
        # case, for a stat only the trimap path needs).
        all_rgb_for_bg_stats = []
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if eff_scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=eff_scale, fy=eff_scale,
                                  interpolation=cv2.INTER_AREA)
            all_rgb_for_bg_stats.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        bg_stats = chroma.bg_stats_from_frames(all_rgb_for_bg_stats)
        del all_rgb_for_bg_stats
        if not bg_stats.is_chroma_class:
            print("  [trimap] background doesn't gate as a flat chroma backdrop "
                  "-- falling back to plain BiRefNet alpha for this clip", flush=True)

    if progress is not None:
        progress("prepare", 1, 1)

    motion_state = {"prev": None, "dmax": None}
    src_gray = []
    subject_boxes = []  # per frame, raw YOLOX box before crop margin (or None) -- keep_main_subject
    clear_masks = {}
    frames = []
    t_detect = t_birefnet = t_trimap = t_despill = t_clears = t_key = 0.0

    if key_model is not None:
        # The whole neural route in one numpy op. No detection box, no
        # network, no temporal state -- key_frame is a pure function of
        # this frame's pixels, which is what makes the downstream
        # smoothing and the heavyweight encoder unnecessary (see
        # keyer.py's docstring and run_clip's mc_median note). Unlike the
        # BiRefNet route below, this never touches subject_box, so it stays
        # a single streaming pass -- box_mode's two-pass restructuring
        # (レバーC) doesn't apply here.
        cap = cv2.VideoCapture(str(video_path))
        idx = -1
        while True:
            if cancel is not None and cancel.is_set():
                cap.release()
                raise JobCancelled(f"cancelled during inference at frame {idx + 1}")
            ok, bgr = cap.read()
            if not ok:
                break
            idx += 1
            if max_frames is not None and idx >= max_frames:
                cap.release()
                raise ClipTooLong(f"clip exceeds max_frames={max_frames} (stopped at frame {idx + 1})")
            if eff_scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=eff_scale, fy=eff_scale,
                                  interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _t0 = time.perf_counter()
            frames.append(keyer.key_frame(rgb, key_model))
            t_key += time.perf_counter() - _t0
            if preview is not None:
                preview(idx + 1, frames[-1])
            if progress is not None:
                progress("infer", idx + 1, total_frames)
            if (idx + 1) % 30 == 0:
                el = time.perf_counter() - t0
                print(f"    {idx + 1} frames ({el:.0f}s, {(idx + 1) / el:.1f} fps)", flush=True)
        cap.release()
    else:
        tm = {"trimap": 0.0, "clears": 0.0, "despill": 0.0}

        def _finish_frame(idx, rgb, a):
            """Per-frame downstream of the BiRefNet alpha (trimap / clears /
            despill / encode-ready RGBA + preview/progress) -- shared by the
            streaming per_frame loop and the opt-in two-pass box_mode path
            so the two can never drift apart."""
            if config.still_thresh:
                # Only cost this GaussianBlur+max per frame when something
                # will actually read motion_state["dmax"] afterwards (audit
                # N4): stages.temporal_alpha_smooth ignores it entirely
                # when config.still_thresh is falsy, which is this
                # pipeline's default (Phase 3 ablation: negligible effect
                # either way).
                _track_source_motion(rgb, motion_state)
            src_gray.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

            unk = None
            if bg_stats is not None and bg_stats.is_chroma_class:
                _t0 = time.perf_counter()
                a, unk = chroma.build_trimap_alpha(
                    a, rgb, bg_stats,
                    fg_seed_thresh=config.trimap_fg_seed_thresh,
                    bg_seed_thresh=config.trimap_bg_seed_thresh)
                tm["trimap"] += time.perf_counter() - _t0

            if config.apply_clears:
                _t0 = time.perf_counter()
                islands = find_clear_candidates(models, rgb, a, bg_ref, config)
                if islands:
                    clear_masks[idx] = islands
                tm["clears"] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            rgb_out = despill(rgb, a, band_only=config.despill_band_only,
                              est_scale=config.despill_est_scale) if config.apply_despill else rgb
            tm["despill"] += time.perf_counter() - _t0
            if unk is not None:
                # Colour correction stays inside the uncertain band
                # (config.py's principle 2): a confidently-FG pixel needs
                # no spill removal (nothing to remove it from), and a
                # confidently-BG pixel is about to be fully transparent
                # anyway.
                rgb_out = np.where(unk[:, :, None], rgb_out, rgb)
            frames.append(np.dstack([rgb_out, a8]))

            if preview is not None:
                preview(idx + 1, frames[-1])
            if progress is not None:
                progress("infer", idx + 1, total_frames)

            if (idx + 1) % 30 == 0:
                el = time.perf_counter() - t0
                print(f"    {idx + 1} frames ({el:.0f}s, {(idx + 1) / el:.1f} fps)", flush=True)

        if config.box_mode == "per_frame":
            # The default: ONE streaming pass -- decode -> YOLOX -> BiRefNet
            # -> despill -> preview/progress for each frame before the next
            # is even decoded (第11計画 Part 1-2). レバーC's two-pass layout
            # (below) was rejected, and for per_frame it bought nothing: the
            # box is this frame's own detection either way, while it held
            # every decoded frame in RAM and delayed the first preview/
            # progress until detection AND BiRefNet had run over the whole
            # clip (measured ~40-60s on GPU, >1h on CPU, of total silence).
            # A truncated crop is repaired right here by BiRefNet's own
            # single-frame full-frame re-inference (auto_full_frame_fallback=
            # True, the pre-レバーC behaviour, counted in box_reinfer_frames).
            cap = cv2.VideoCapture(str(video_path))
            idx = -1
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise JobCancelled(f"cancelled during inference at frame {idx + 1}")
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    idx += 1
                    if max_frames is not None and idx >= max_frames:
                        raise ClipTooLong(f"clip exceeds max_frames={max_frames} (stopped at frame {idx + 1})")
                    if eff_scale != 1.0:
                        bgr = cv2.resize(bgr, None, fx=eff_scale, fy=eff_scale,
                                          interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    _t0 = time.perf_counter()
                    box, _ = models.subject_box(rgb, prefer_person=False, min_box_frac=config.min_box_frac)
                    t_detect += time.perf_counter() - _t0
                    subject_boxes.append(box)
                    _t0 = time.perf_counter()
                    a = models.birefnet(rgb, box=box)
                    if config.matte_refine:
                        a = _matte_refine(rgb, a)
                    t_birefnet += time.perf_counter() - _t0
                    _finish_frame(idx, rgb, a)
            finally:
                cap.release()
        else:
            # ---- opt-in box_mode ("clip_union"/"smoothed", レバーC --
            # measured and NOT adopted, see DECISIONS.md): needs every
            # frame's detection before any box can be resolved, hence the
            # two passes. Pass 0: decode the whole clip up front (still the
            # one place max_frames/cancel are enforced against the raw
            # frame count) ----
            rgb_frames = []
            cap = cv2.VideoCapture(str(video_path))
            idx = -1
            while True:
                if cancel is not None and cancel.is_set():
                    cap.release()
                    raise JobCancelled(f"cancelled during inference at frame {idx + 1}")
                ok, bgr = cap.read()
                if not ok:
                    break
                idx += 1
                if max_frames is not None and idx >= max_frames:
                    cap.release()
                    raise ClipTooLong(f"clip exceeds max_frames={max_frames} (stopped at frame {idx + 1})")
                if eff_scale != 1.0:
                    bgr = cv2.resize(bgr, None, fx=eff_scale, fy=eff_scale,
                                      interpolation=cv2.INTER_AREA)
                rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()

            # ---- Pass 1: full-clip YOLOX detection, resolved per box_mode ----
            _t0 = time.perf_counter()
            raw_boxes = [models.subject_box(rgb, prefer_person=False,
                                             min_box_frac=config.min_box_frac)[0]
                         for rgb in rgb_frames]
            boxes = boxmode.resolve_boxes(raw_boxes, config.box_mode)
            t_detect += time.perf_counter() - _t0
            subject_boxes = list(raw_boxes)

            # ---- Pass 2a: BiRefNet crop pass per frame, no per-frame
            # full-frame fallback (repaired at clip level in 2b instead) ----
            alphas = []
            truncated = []
            for idx, rgb in enumerate(rgb_frames):
                if cancel is not None and cancel.is_set():
                    raise JobCancelled(f"cancelled during inference at frame {idx + 1}")
                _t0 = time.perf_counter()
                a = models.birefnet(rgb, box=boxes[idx], auto_full_frame_fallback=False)
                t_birefnet += time.perf_counter() - _t0
                if config.matte_refine:
                    a = _matte_refine(rgb, a)
                alphas.append(a)
                truncated.append(bool(getattr(models, "_last_truncated", False)))

            # ---- Pass 2b: clip-level truncation repair over contiguous
            # truncated regions ----
            if any(truncated):
                _t0 = time.perf_counter()
                alphas, _ = boxmode.repair_truncated_regions(
                    models, rgb_frames, boxes, alphas, truncated,
                    margin=0.12, matte_refine=config.matte_refine)
                t_birefnet += time.perf_counter() - _t0

            # ---- Pass 2c: per-frame downstream ----
            for idx, rgb in enumerate(rgb_frames):
                if cancel is not None and cancel.is_set():
                    raise JobCancelled(f"cancelled during inference at frame {idx + 1}")
                _finish_frame(idx, rgb, alphas[idx])
            del rgb_frames, alphas
        t_trimap, t_clears, t_despill = tm["trimap"], tm["clears"], tm["despill"]

    # getattr, not a bare attribute access: test doubles (_StubModels in
    # tool/tests/test_runner_limits.py) duck-type Models without this
    # counter, and treating "no counter" as "never fired" is correct there.
    box_reinfer_frames = getattr(models, "box_reinfer_frames", 0) if models is not None else 0
    if own_models and models is not None:
        del models
    el = time.perf_counter() - t0
    print(f"  inferred {len(frames)} frames ({el:.0f}s)", flush=True)
    if key_model is not None:
        print(f"    [timing] keyer {t_key:.1f}s", flush=True)
        if timings is not None:
            timings["key_s"] = t_key
        # Same dict shape as the BiRefNet route so run_clip/cache.py/the
        # server contract are unchanged; the empty/None members are stages
        # this path genuinely has no input for (no clear candidates, no
        # source-gray for optical flow, no motion map). "keyed" tells
        # run_clip to skip the stages that would consume them.
        return {"frames": frames, "clear_masks": {}, "src_gray": [],
                "motion_dmax": None, "bg_ref": bg_ref, "fps": fps,
                "keyed": True, "boxes": None}
    # detect (YOLOX subject_box) and birefnet are reported SEPARATELY --
    # they used to be folded into one "[timing] birefnet" number, which
    # misattributed several minutes/clip of YOLOX running on CPU (a
    # settings bug, since fixed -- see matte_core.Models.__init__) to
    # BiRefNet, sending prior optimization effort at the wrong model.
    print(f"    [timing] detect {t_detect:.1f}s, birefnet {t_birefnet:.1f}s, despill {t_despill:.1f}s"
          + (f", trimap {t_trimap:.1f}s" if t_trimap else "")
          + (f", apply_clears-candidates {t_clears:.1f}s" if t_clears else ""), flush=True)
    if box_reinfer_frames:
        # matte_core.Models.birefnet's crop-truncation fallback fired --
        # worth surfacing even outside a full [timing] line, since a high
        # count on a clip that's supposedly working fine is itself a signal
        # (either genuinely difficult content, or a detector systematically
        # under-covering the subject on this asset).
        print(f"    [box] {box_reinfer_frames}/{len(frames)} frame(s) needed a full-frame "
              f"re-inference after the crop-based alpha looked truncated", flush=True)
    if timings is not None:
        timings["detect_s"] = t_detect
        timings["birefnet_s"] = t_birefnet
        timings["despill_s"] = t_despill
        if t_trimap:
            timings["trimap_s"] = t_trimap
        if t_clears:
            timings["clears_s"] = t_clears
        if box_reinfer_frames:
            timings["box_reinfer_frames"] = box_reinfer_frames
    return {"frames": frames, "clear_masks": clear_masks, "src_gray": src_gray,
            "motion_dmax": motion_state["dmax"], "bg_ref": bg_ref, "fps": fps,
            "boxes": subject_boxes}


def run_clip(video_path, out_path, config: PipelineConfig, models: Models | None = None,
             raw=None, *, progress=None, cancel=None, max_frames=None, preview=None, timings=None):
    """Process one clip end-to-end: inference -> postprocess stages -> encode.
    Pass `models` to reuse a loaded Models instance across clips (loading it
    is the expensive part); omit it to load one for just this call. Pass
    `raw` (an infer_clip() result, or a cache.load_raw() one) to skip
    inference entirely and only re-run the postprocess+encode half -- this
    is what makes ablation over postprocess stages fast.

    `progress`/`cancel`/`max_frames`: see infer_clip's docstring --
    forwarded there (when `raw` is None; `max_frames` has no effect if
    `raw` is already available, since inference already happened) and
    `progress`/`cancel` are also used to bracket the postprocess/encode
    stages here, so a server driving this function gets one consistent
    stream of ("prepare"|"infer"|"postprocess"|"encode"|"done", done, total) calls
    across the whole clip regardless of which stage is currently running.

    `preview`, if given, is called as preview(done, frame) from BOTH the
    infer loop (see infer_clip's docstring for what that frame actually
    is) and _encode's own per-frame chunks() generator -- the latter walks
    the fully post-processed frames one at a time already, so a preview
    taken there is the ACTUAL final-quality output, letting a live preview
    converge from "rough in-progress glance" to "what you'll actually get"
    over the course of one clip.

    `timings`, if given a dict, is forwarded to infer_clip (when `raw` is
    None) and additionally gets `postprocess_s`/`encode_s`/`total_s`
    written into it as each stage actually finishes -- so a caller (the
    server job record) that passed its own dict in still has the earlier
    stages' numbers even when a LATER one raises (e.g. an encode timeout
    after a perfectly normal infer+postprocess): previously this
    breakdown existed only as a stdout print, and a failed clip's job.json
    recorded no timing at all, making a 983s failure indistinguishable
    from a 5s one."""
    video_path = Path(video_path)
    out_path = Path(out_path)
    t0 = time.perf_counter()

    if raw is None:
        raw = infer_clip(video_path, config, models=models, progress=progress, cancel=cancel,
                          max_frames=max_frames, preview=preview, timings=timings)
    elif cancel is not None and cancel.is_set():
        raise JobCancelled("cancelled before postprocess (raw already available)")

    if raw.get("keyed"):
        # The motion-compensated median exists to damp BiRefNet's frame-to-
        # frame chatter, and it is the single most expensive postprocess
        # stage (57.1s of a 737s clip, all optical flow). The keyer's alpha
        # is a deterministic per-pixel function of the source, so identical
        # pixels give identical alpha on every frame and there is no chatter
        # to damp -- measured S1 was 2.2x BETTER than the neural path WITH
        # smoothing. It also has no src_gray to run flow on. Forced off here
        # rather than left to the caller so no config can ask for a stage
        # whose inputs don't exist.
        config = dataclasses.replace(config, mc_median_half=0, still_thresh=0)

        # supersampled_gif's two-pass 4x palette chain (604.1s of that same
        # 737s clip -- 82% of the whole run) exists to smooth over BiRefNet's
        # run-to-run alpha non-determinism near the 1-bit threshold, which the
        # plan's V4 investigation identified as the real reason every lighter
        # encoder measured worse. A keyed matte is a deterministic function of
        # the source pixels, so that justification does not apply to it and
        # the single-pass encoder becomes sound here (measured 604.1s -> 19s).
        # Substituted ONLY for the untouched default, never for an encoder the
        # caller named explicitly (config.encoder_explicit -- see its own
        # docstring for why a value-only check couldn't tell these apart) --
        # same rule as the server's LOAD_AWARE_ENCODE. Both produce .gif, so
        # no caller's output path or extension changes.
        if not config.encoder_explicit and config.encoder == "supersampled_gif":
            config = dataclasses.replace(config, encoder="ss_alpha_gif")

        # 第13計画: keep_main_subject forced off on the keyer route, not just
        # its SAM2 refinement (that part was already gated below on
        # `not raw.get("keyed")`). The keyer route has no boxes, so rule
        # (iii) can never fire there -- keep_main_subject would run rules
        # (i)/(ii) alone (largest component + comparably-sized ones), which
        # WAS measured to cost F1i (+1.2-9.7% on real-footage fixtures, see
        # DECISIONS.md's 第12計画 entry) by dropping small islands that were
        # actually part of the subject. Forced off here (same pattern as
        # mc_median_half/still_thresh above) so no caller's config can ask
        # for a stage whose own measured cost on this route is negative.
        config = dataclasses.replace(config, keep_main_subject=False)

    frames = [f.copy() for f in raw["frames"]]  # never mutate a cached/shared raw result
    if not frames:
        # infer_clip should never produce this for a clip that passed
        # server/probe.py's upload-time validation -- but a corrupt/0-frame
        # source, or a cached raw result built before that validation
        # existed, would otherwise die on `frames[0]` below with a bare
        # IndexError that gives no hint what actually went wrong (audit L2).
        raise RuntimeError(f"{video_path.name}: no frames to encode (0-frame or corrupt source)")
    H, W = frames[0].shape[:2]
    n_post = max(0, len(frames) - 2 * config.mc_median_half) if config.mc_median_half > 0 else 1
    if progress is not None:
        progress("postprocess", 0, n_post)
    post_progress = (lambda d, t: progress("postprocess", d, n_post)) if progress is not None else None

    # 第12計画: a SAM2 worker for keep_main_subject's rule (iii), lazily
    # started (its subprocess only actually spawns on the first frame that
    # needs it -- see keep_main_subject's docstring) and always stopped
    # before this function returns. GPU-only (matte_core's Models docstring
    # measured ~100s/frame for SAM2 on CPU -- not usable in a job); the
    # keyer route has no boxes for SAM2 to be prompted with, so it's skipped
    # there too (same as the box-rectangle rule it replaces).
    sam2_worker = None
    sam2_fn = None
    if (config.keep_main_subject and config.keep_main_subject_sam2 and not raw.get("keyed")
            and resolve_device(config.device) == "cuda"):
        sam2_worker = _Sam2Worker("cuda")
        sam2_fn = lambda rgb, box: sam2_worker.mask(rgb, np.asarray(box, dtype=np.float32))

    t_post0 = time.perf_counter()
    try:
        postprocess(frames, raw["clear_masks"], raw["src_gray"], raw["motion_dmax"], raw["bg_ref"], config,
                    progress=post_progress, boxes=raw.get("boxes"), sam2_fn=sam2_fn)
    finally:
        if sam2_worker is not None:
            sam2_worker.stop()
    t_post = time.perf_counter() - t_post0
    if progress is not None:
        progress("postprocess", n_post, n_post)
    if timings is not None:
        timings["postprocess_s"] = t_post

    if cancel is not None and cancel.is_set():
        raise JobCancelled("cancelled before encode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress("encode", 0, 1)
    t_enc0 = time.perf_counter()
    try:
        _encode(frames, W, H, raw["fps"], out_path, config, progress=progress, cancel=cancel, preview=preview)
    finally:
        # Recorded even on failure (timeout/cancel/anything else _encode
        # raises) -- "how long did encode run before it died" is exactly
        # the number this dict exists to preserve; see the docstring above.
        if timings is not None:
            timings["encode_s"] = time.perf_counter() - t_enc0
    t_enc = time.perf_counter() - t_enc0
    print(f"    [timing] postprocess {t_post:.1f}s, encode {t_enc:.1f}s", flush=True)
    if progress is not None:
        progress("done", 1, 1)

    # Provenance: which PipelineConfig actually produced this file, next to
    # it -- so a defect found later doesn't require guessing which knobs
    # were on (this session repeatedly had to re-derive that from stdout
    # logs or by re-running). One JSON per output file, not per clip name,
    # since --use-trimap / --encoder can differ between calls to the same
    # video_path in one process (e.g. the Phase 3/4 ablation scripts).
    _write_config_provenance(out_path, config)

    el = time.perf_counter() - t0
    print(f"  done: {out_path}  ({len(frames)} frames, {out_path.stat().st_size / 1e6:.1f} MB, "
          f"{el:.0f}s)", flush=True)
    if timings is not None:
        timings["total_s"] = el
    return out_path


def encode_worker(video_path_str: str, out_path_str: str, config: PipelineConfig):
    """(V5) Top-level, picklable worker body for --parallel-encode's process
    pool: load THIS clip's cached raw inference result and run only the
    postprocess+encode half. Must live in a real importable module, not
    __main__.py -- a `python -m tool.pipeline` invocation's __main__ isn't
    reimportable by dotted name, which is what the pool's spawn start
    method needs to hand this function to each fresh worker interpreter
    (confirmed by testing: defining this in __main__.py instead raises
    "AttributeError: Can't get attribute '_encode_worker' on <module
    '__main__'>" in every worker, failing all clips)."""
    video_path = Path(video_path_str)
    raw = cache.load_raw(video_path.stem, config)
    if raw is None:
        raise RuntimeError(f"no cached inference for {video_path.stem} "
                            f"(cache dir or config mismatch)")
    run_clip(video_path, Path(out_path_str), config, raw=raw)


def _write_config_provenance(out_path: Path, config: PipelineConfig):
    import dataclasses
    import json
    provenance_path = out_path.with_suffix(out_path.suffix + ".pipeline_config.json")
    provenance_path.write_text(json.dumps(dataclasses.asdict(config), indent=2, ensure_ascii=False))


def _encode(frames, W, H, fps, out_path, config: PipelineConfig, *, progress=None, cancel=None, preview=None):
    n = len(frames)

    def chunks():
        # ffmpeg reports no per-frame progress of its own, so the count of
        # frames handed to its stdin so far is used as the encode stage's
        # progress proxy (this generator runs on the main thread, driven by
        # pipe_rgba_to_ffmpeg's write loop). `preview` here sees the fully
        # post-processed frame -- the real final-quality output, not the
        # rough infer-time glance (see run_clip's docstring).
        for i, f in enumerate(frames):
            if preview is not None:
                preview(i + 1, f)
            if progress is not None:
                progress("encode", i + 1, n)
            yield f.tobytes()

    try:
        if config.encoder == "supersampled_gif":
            ffenc.encode_gif_supersampled(chunks(), W, H, fps, out_path,
                                           ss=config.supersample, alpha_threshold=config.alpha_threshold,
                                           cancel=cancel)
        elif config.encoder == "ss_alpha_gif":
            ffenc.encode_gif_ss_alpha(chunks(), W, H, fps, out_path,
                                       ss=config.supersample, alpha_threshold=config.alpha_threshold,
                                       cancel=cancel)
        elif config.encoder == "mov":
            ffenc.encode_mov(chunks(), W, H, fps, out_path, cancel=cancel)
        elif config.encoder == "webp":
            ffenc.encode_webp(chunks(), W, H, fps, out_path, cancel=cancel)
        else:
            raise ValueError(f"unknown encoder {config.encoder!r}")
    except ffenc.EncodeCancelled as e:
        _unlink_partial_output(out_path)
        raise JobCancelled(str(e)) from e
    except Exception:
        # ffmpeg is launched with out_path as ITS OWN output argument, so it
        # creates/truncates the file the moment it starts -- a timeout or
        # any other mid-encode failure (not just a cancel) leaves a 0-byte
        # (or partial) file behind with no clip["outputs"] entry ever
        # pointing at it, invisible to the UI/zip and orphaned until the
        # 14-day job-retention sweep (confirmed: 4 such files accumulated
        # in one evening of contended-machine timeouts).
        _unlink_partial_output(out_path)
        raise


def _unlink_partial_output(out_path) -> None:
    try:
        out_path.unlink(missing_ok=True)
    except OSError:
        pass  # best-effort cleanup; the real error is already propagating

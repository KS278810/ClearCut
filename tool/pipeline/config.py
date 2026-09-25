"""Config for the chroma-background (flat-colour-backdrop) video pipeline.

Ported from scripts/run_gpu_birefnet_v24.py (the last hand-validated version)
as part of the Phase 1 refactor: v8-v25 grew by monkey-patching shared module
attributes (`brv.analyze = ...`, `ffenc.encode_gif = ...`) per bug report,
never removing or re-validating a prior patch -- which is what let a stale
`encode_gif` binding silently corrupt a later measurement (see the plan's B1
finding). This package instead threads one immutable config object through
plain functions; nothing here reassigns another module's attribute.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    device: str = "cuda"

    # -- flat-chroma fast path (tool/pipeline/keyer.py) --
    # "auto": use the colour-only keyer when the clip's backdrop actually
    # measures as flat chroma AND the subject's own palette stays clear of
    # the key colour (keyer.is_safe); fall back to the BiRefNet route
    # otherwise. "on" forces it (still refuses if is_safe fails -- it is a
    # correctness gate, not a preference), "off" never uses it.
    #
    # Why this exists: on a flat backdrop the neural route's four expensive
    # stages are all compensating for BiRefNet weaknesses that don't arise
    # (see keyer.py's module docstring). Measured 737s -> ~30-46s per clip
    # with F1i 42x better on the flatchroma2 clips.
    use_keyer: str = "auto"  # "auto" | "on" | "off"
    # Ramp top as a fraction of the measured distance from the key colour to
    # the nearest confident-subject pixel. Scales a per-clip MEASUREMENT, so
    # it adapts to any key-colour/subject-palette pairing rather than being a
    # colour threshold in disguise. See keyer.estimate_key for why the QC
    # metrics' own T_HI (8 sigma) is unusable for generating alpha.
    # (A full-resolution luma term was implemented and MEASURED INEFFECTIVE --
    # the 4:2:0 chroma subsampling hypothesis for the edge numbers did not hold
    # up: weights 0.05-0.2 moved E1 by <0.02 and made E2 slightly worse. Deleted
    # rather than shipped as a dead knob; see tool/docs/DECISIONS.md.)
    keyer_ramp_k: float = 0.5
    # 第11計画 Part 3-3 (ADOPTED 2026-09-23): unmix the keyer's edge colour
    # with a separate coverage estimate (ramp widened to neutral_dist) over
    # a 2px edge band that includes display-opaque pixels -- the display
    # alpha (and so the matte's shape, F1i/F2) is unchanged by construction.
    # Measured: backdrop-coloured outline px -> ~0, E2 up on all 8 keyer
    # fixtures. See keyer.key_frame and DECISIONS.md.
    keyer_unmix_coverage: bool = True
    # 第11計画 Part 3-4 (REJECTED, opt-in only): floor the ramp bottom
    # (t_lo) at this clip's own far-backdrop distance p99.9 instead of a
    # fixed 3 sigma. It does cut webp/mov backdrop haze where it engages,
    # but raising t_lo also shifts the 1-bit edge: F1i +2.7%..+28% on 3 of
    # 4 real-footage clips, failing the pre-registered "F1i unchanged" gate.
    # See keyer.estimate_key and DECISIONS.md.
    keyer_measured_t_lo: bool = False

    # -- input resolution --
    # Downscale the source frame once, right after decode, before anything
    # else touches it -- so matting, despill, alpha and the final output
    # size all inherit the smaller resolution (same convention as
    # bg_remove_video.py's own --scale). Added after a real-footage batch's clips
    # (1440x1440 square, vs the dinosaur clips' 1656x1248) blew the
    # encoder's 600s ffmpeg timeout at scale=1.0: BiRefNet always resizes
    # its crop to a fixed 1024^2 for inference regardless of source
    # resolution, so a full-frame box gets ~the same matting quality at
    # scale=1.0 or scale=0.6 -- the extra native pixels above ~1024 on the
    # long side buy very little there. What scale=1.0 actually costs is the
    # encoder: encode_gif_supersampled's 4x pass scales with resolution
    # SQUARED, and a delivery GIF rarely needs source-video resolution
    # anyway. See the plan's G4 note for the measured effect on that batch.
    #
    # `scale` is the low-level knob (explicit multiplier); `max_side` is
    # the practical default (cap the long edge, auto-deriving whatever
    # scale that implies for THIS clip's resolution) -- per the user's
    # 2026-09-02 decision ("実際の用途ではそんなに解像度いらない、長辺を
    # 上限で揃える"). `scale` wins if set to anything but 1.0.
    #
    # max_side=1000 was TESTED and FAILED the pre-registered speed-review
    # criteria (see the plan's V1 entry): F2 false-keep worsened 400-1300%
    # across all 4 dinosaur clips, and visually confirmed as a real defect
    # -- background colour visibly left un-removed in the gaps between
    # teeth on 挨拶.mp4 (absent at native resolution, same frame). Cause:
    # cv2.INTER_AREA downscaling erases fine structure like tooth gaps
    # before BiRefNet ever sees it. Default kept at None (native
    # resolution, old behaviour) until a gentler cap or a different
    # downscale method is found and re-verified against the same criteria
    # -- do NOT re-enable a max_side default without re-running that check.
    scale: float = 1.0
    max_side: int | None = None

    # -- subject detection (fixes B2: this used to only exist as a
    # per-script monkey-patch on Models.subject_box, so every OTHER caller
    # of subject_box silently carried the same YOLOX-misfire collapse risk) --
    min_box_frac: float = 0.15
    # -- box_mode: how the per-frame YOLOX detections become the boxes
    # BiRefNet's own pass crops to. "per_frame" (default) is each frame's
    # own independently-detected box, run as ONE streaming loop with the
    # single-frame full-frame fallback on a truncated crop (第11計画 Part
    # 1-2 restored this after レバーC's two-pass layout delayed the first
    # preview by the whole clip's detect+BiRefNet time). "clip_union" and "smoothed" are the two variants registered for
    # the box-stabilization experiment -- these alone use runner.infer_clip's
    # two-pass path (full up-front YOLOX pass, clip-level truncation repair;
    # see tool/pipeline/boxmode.py and
    # tool/docs/DECISIONS.md for the gate result) -- kept default here
    # unless that experiment's pre-registered gate passed and this default
    # was deliberately flipped afterwards (see DECISIONS.md for whichever
    # is currently true).
    box_mode: str = "per_frame"  # "per_frame" | "clip_union" | "smoothed"

    # -- Phase 4: trimap compositing (chroma-background asset class only) --
    # BiRefNet decides the FG/BG regions; colour distance only ramps alpha
    # inside the narrow uncertain band between them. See chroma.py's
    # docstring for why this is structurally immune to the fringe/shadow/
    # over-erosion tradeoff a single global colour rule cannot escape.
    use_trimap: bool = False
    trimap_fg_seed_thresh: float = 0.90
    trimap_bg_seed_thresh: float = 0.10

    # -- despill (pymatting estimate_foreground_ml, runs over the WHOLE
    # subject bbox+padding every frame -- not just the semi-transparent
    # edge it exists to fix). V0 profiling (2026-09-02, see the plan's V0
    # entry) found despill costs MORE wall time per clip (128.6s) than
    # BiRefNet inference itself (116.3s) -- an unexpected result that
    # motivated V4: a fringe-colour-quality metric (tool/qc/metrics.py's
    # E2_fringe_quality) plus this flag, to measure whether that cost is
    # earning its keep before considering scoping it down to the
    # alpha<1 band only. Default ON (current shipped behaviour).
    apply_despill: bool = True
    # V6 speed/quality experiment (see the plan): despill's write-back used
    # to overwrite the WHOLE subject crop every frame, including the fully-
    # opaque interior where there is no spill to remove -- band_only=True
    # limits both the write-back AND the ML solve itself to the genuinely
    # semi-transparent band (0.02 < alpha < 0.98), skipping the solve
    # entirely on a frame with no such pixels. est_scale<1 additionally
    # runs the solve on a downscaled crop and upsamples the result.
    # ADOPTED 2026-09-22 (第9計画/第8回監査レバー3): tool/scripts/
    # a speed-lever experiment on the 4 flatchroma2 clips,
    # forced neural via use_keyer=off, shared BiRefNet inference across
    # variants to remove its own run-to-run nondeterminism as a confound)
    # -- band_only+est_scale=0.5 gave F1i/F2 BIT-IDENTICAL to the old
    # defaults on all 4 clips (despill never touches alpha, as expected)
    # and E2_fringe_quality (fringe colour contamination) 5-6% BETTER, not
    # just non-worse, while cutting despill wall time ~40% (31-56s ->
    # 18-29s). band_only alone (est_scale=1.0) also passed but with far
    # smaller speed gain (~5-20%) and slightly worse E2 than the combined
    # variant, so it wasn't worth keeping as a separate default.
    despill_band_only: bool = True
    despill_est_scale: float = 0.5

    # -- matting --
    matte_refine: bool = False  # closed-form refine: OFF (v24 docstring
    # DEFECT 4 -- it was the single biggest chatter source, +52%, and a 1-bit
    # GIF alpha throws away the soft edge it exists to recover)

    # -- bg-leak-fix (zoomed re-inference on desaturated near-backdrop
    # islands, tracked across frames) --
    # Phase 3 ablation (2026-08-31): on all 4 dinosaur clips, with per-frame
    # BiRefNet, find_clear_candidates' prefilter found ZERO islands on EVERY
    # frame (clear_masks == {} for all 4 clips) -- confirmed not a wiring
    # bug (config flag genuinely differs; output was still byte-identical
    # with the stage on vs off because there was nothing for it to act on).
    # This stage exists for a REAL-FOOTAGE class (a different
    # failure mode: real background gaps left opaque by matting error) --
    # kept available, default OFF here since this package is chroma-only
    # (see lever S5; bg_remove_video.py's general route is untouched).
    apply_clears: bool = False
    clear_sat_thresh: float = 12
    clear_min_area: int = 2000
    clear_min_area_link: int = 600
    clear_dist_thresh: float = 60
    clear_zoom_window_mult: int = 5
    clear_zoom_min_side: int = 512
    clear_zoom_bg_frac: float = 0.85
    clear_track_overlap: float = 0.30
    clear_track_strong_frac: float = 0.30
    clear_track_max_gap: int = 3

    # -- single-frame area-collapse repair --
    # Phase 3 ablation: 0 triggers on all 4 clips (byte-identical with/
    # without) -- the collapse mode it repairs (a YOLOX misfire producing a
    # tiny wrong crop box for several consecutive frames) doesn't happen
    # here because min_box_frac already rejects that box before BiRefNet
    # ever sees it. Kept available for content where that's not guaranteed.
    collapse_fix: bool = False
    collapse_area_ratio: float = 0.6

    # -- temporal alpha smoothing --
    # mc_median: KEEP -- ablation showed a real, large effect (S1 chatter
    # 2306 -> 1634, -29%, averaged over 4 clips) undoing it.
    mc_median_half: int = 1       # 0 disables the motion-compensated median
    # V7 speed/quality experiment (see the plan): DIS optical flow is
    # computed at full resolution for every neighbour pair -- V0 profiling
    # found this dominates mc_median's cost (linear in 2*mc_median_half
    # flow calls). flow_scale<1 computes flow on a downscaled gray pair and
    # upsamples+rescales the flow field before the (still full-res) remap.
    # Defaults to 1.0 (pre-existing behaviour) until the V7 harness
    # comparison decides whether to flip it.
    flow_scale: float = 1.0
    # V7 companion speed experiment (see the plan's Part C): DIS's own
    # PRESET_MEDIUM (hardcoded until now) trades accuracy for a faster
    # coarse-to-fine search under "fast"/"ultrafast". Postprocess-only,
    # like flow_scale -- excluded from cache._key. Defaults to the
    # pre-existing behaviour until the harness comparison decides.
    flow_preset: str = "medium"  # "medium" | "fast" | "ultrafast"
    # alpha freeze-on-still: ablation showed a negligible effect either way
    # (F1i/S1/S3 all within noise with vs without) on this per-frame
    # pipeline -- default OFF to keep the pipeline minimal; the code stays
    # available since a slower/keyint-based route could behave differently.
    still_thresh: float = 0

    # -- colour-freeze on still pixels: OFF. v24 removed it after it froze a
    # 41,290px shading region on 感謝.mp4 into a visibly "stuck" patch
    # (STILL_THRESH only bounds the per-STEP delta, not the full-clip range,
    # so a slowly-drifting-but-visually-alive pixel still passed it). Kept
    # as a config knob (not deleted) so Phase 3's ablation can re-measure it
    # against the real S2 colour-flicker regression it was trying to fix,
    # rather than re-deriving this from scratch if a better still-detection
    # rule is found later.
    colour_freeze: bool = False

    # -- background-colour fringe strip (narrow, hue+distance gated; see
    # tool/qc's discovery that a global colour threshold alone can't do
    # this safely -- this stays intentionally narrow) --
    #
    # Default OFF, opt-in only (2026-09-01): this was the last colour
    # heuristic still defaulting ON, tuned and validated ONLY against the
    # dinosaur asset (yellow backdrop / teal subject). On a second client's
    # footage (purplebg: purple backdrop / navy suit) it fired throughout the
    # dark clothing -- F1i 574k-1,028k px across the 4 clips -- because its
    # hue+distance rule, safe for one backdrop/subject colour pairing, does
    # not generalise to another. Every colour heuristic in this pipeline's
    # history that defaulted ON eventually erased real subject content on
    # SOME asset (_strip_tail_shadow, colour-freeze, now this one) --
    # see the plan's "3案件の経緯" note. Explicitly opt in per-clip-class
    # once verified safe (see stages._fringe_strip_is_safe, which also
    # self-disables with a warning if enabled somewhere it measures unsafe).
    strip_bg_fringe: bool = False
    fringe_dist_thresh: float = 100
    fringe_hue_thresh: int = 6
    fringe_safety_max_frac: float = 0.005  # see stages._fringe_strip_is_safe

    # -- keep_main_subject (第11計画 Part 3-1): per frame, drop opaque
    # (alpha>127) connected components that are not the main subject --
    # BiRefNet foregrounds any salient prop inside its crop (the widepose clip's
    # dumbbell/bottle/laptop on the floor, separate components at 1-3k px vs
    # the person's 25-28k), and the keyer keeps small detached islands. A
    # component survives if it is (i) the frame's largest, (ii) at least
    # keep_main_subject_min_frac of the largest's area, or (iii) touches that
    # frame's own YOLOX subject box (pre-margin; neural route only). Soft
    # rim pixels within keep_main_subject_rim_px of a dropped component (and
    # not also that close to a kept one) go with it; every kept component's
    # soft edge is untouched. Thresholds pre-registered, not tuned on the
    # gate clips. LIMIT: a prop held in the hand touches the person and is
    # one component with them -- it stays.
    # DEFAULT OFF (2026-09-23): the pre-registered gate FAILED -- on
    # the widepose clip's frames 110/180 the laptop, bottle and dumbbell (3 extra
    # components each) all touch the person's own wide-pose YOLOX box, so
    # rule (iii) keeps them; the filter only dropped 1,339 px on 4/245
    # frames. Not re-tuned after seeing that (see DECISIONS.md). Opt in per
    # clip; on the keyer route (no boxes, rules (i)(ii) only) it does remove
    # motion-blur ghost islands, at a measured F1i cost -- also DECISIONS.md.
    #
    # 第12計画 (2026-09-24): rule (iii) can be refined with a SAM2 pixel
    # mask instead of the raw box rectangle -- a prop fully INSIDE the
    # person's box (the widepose clip's frame 110's bottle) cannot be
    # separated by any box-rectangle rule, no matter how it's tuned; only a
    # pixel-level person mask can. keep_main_subject_sam2 controls whether
    # runner.run_clip wires a SAM2 worker in as keep_main_subject's sam2_fn
    # (GPU + keep_main_subject both required; CPU and the keyer route always
    # use the box-rectangle fallback).
    #
    # 第12計画's first attempt ALSO gated rule (iii') behind a box-area-ratio
    # health check (mask_area/box_area in [0.20, 2.0]) -- that gate FAILED
    # (see DECISIONS.md): the check rejected a CORRECT SAM2 mask on that
    # widepose clip's own frames (105-110/177-182), because a spread-limbs
    # bounding box is mostly empty space, so a legitimately-correct person
    # mask naturally sits below any box-area-normalized floor there. 第13計画
    # (2026-09-24) removed that check after calibrating it: across 90
    # decision-point frames over 6 clips, SAM2's overlap with every one of
    # that clip's 62 real prop components was EXACTLY 0.000, vs 1.000 (or
    # otherwise high) for every legitimate person-fragment component, and
    # SAM2 never returned a literally empty mask -- see DECISIONS.md's
    # calibration table. The overlap rule alone, unblocked by the floor,
    # separates props from the person with maximum margin; "mask is empty"
    # is the only fallback condition left (a real but rarely-firing
    # backstop for a genuine SAM2 failure, not the disguised floor).
    #
    # Thresholds below are pre-registered from that calibration data (see
    # DECISIONS.md), not tuned on the 3 known gate frames themselves.
    keep_main_subject: bool = False
    keep_main_subject_min_frac: float = 0.25
    keep_main_subject_rim_px: int = 5
    keep_main_subject_sam2: bool = True
    keep_main_subject_sam2_overlap: float = 0.30
    keep_main_subject_sam2_dilate_px: int = 5

    # -- encoder --
    # ADOPTED 2026-09-22 (第9計画/第8回監査レバー4): the historical concern
    # blocking ss_alpha_gif as the neural-route default (2026-09-02/03's
    # V2/V3, and every earlier attempt at a lighter encoder) was a visible
    # F2 defect -- LATER TRACED to BiRefNet's own run-to-run nondeterminism
    # near the alpha=128 threshold, not the encoder. tool/scripts/
    # speed_experiment.py's decisive re-test shares ONE cached BiRefNet
    # inference across all encoder variants (eliminating that confound) on
    # the 4 flatchroma2 clips forced through the neural route: F1i within
    # +2-3%, E1/S1 within noise, F2 improved (fewer false-keep px) on all
    # 4, encode time 422-556s -> 27-45s (~15x). Worst-F2-frame crops
    # inspected visually per clip -- no fringe/patch defects (the mandatory
    # human check this repo's history says a passing number alone can't
    # replace). ss_alpha_gif was already the keyer path's runtime swap
    # target (see the encoder_explicit note below); this makes it the
    # single default for both routes instead of two different ones.
    encoder: str = "ss_alpha_gif"   # "supersampled_gif" | "ss_alpha_gif" | "mov" | "webp"
    # Whether `encoder` was set BY THE CALLER (an explicit --encoder / API
    # override), as opposed to sitting at this field's own default. Two
    # automatic substitutions -- server/jobs.py's LOAD_AWARE_ENCODE fallback
    # and run_clip's keyer-path swap to ss_alpha_gif (runner.py) -- both
    # claimed in their comments to "never override an explicit choice" but
    # could only test the encoder's current VALUE, which cannot distinguish
    # "still at the untouched default" from "the caller asked for
    # supersampled_gif by name" (audit M9: confirmed the existing regression
    # test for this only passed because it happened to use "mov" as its
    # explicit example, not "supersampled_gif"). Always False for direct CLI/
    # library construction unless the caller sets it; server/presets.py sets
    # it from whether "encoder" is a key in the request's own overrides dict.
    # runner.py's keyer-path swap (config.encoder == "supersampled_gif" and
    # not explicit -> ss_alpha_gif) is now a no-op in the common case since
    # ss_alpha_gif IS the default -- left in place rather than removed,
    # since it still does the right thing if a future default ever reverts.
    encoder_explicit: bool = False
    supersample: int = 4
    alpha_threshold: int = 128

"""Phase 4: trimap refinement for chroma-background (flat-colour-backdrop)
clips. Reverses the roles from the rest of this pipeline: BiRefNet decides
REGION (confident foreground / confident background), colour decides
alpha ONLY inside the narrow band between them.

Why: a single global colour threshold cannot simultaneously (a) remove the
boundary's colour fringe, (b) remove a floor shadow, and (c) avoid eating
into the character -- confirmed by direct experiment (see the plan's
Phase-4-preceding chroma-key investigation): the true background/foreground
blend-zone width varies spatially (compression, render softening, motion
blur), so no single threshold value sits correctly everywhere. Restricting
colour logic to a band BiRefNet has already narrowed down removes that
failure mode structurally: colour never has to decide anything deep inside
the body or deep inside the background, only in the strip where BiRefNet
itself is unsure.

Literature backing (see the plan's web-research addendum): Smith & Blinn
(1996) prove single-known-background compositing is under-determined from
colour alone; professional keyers (Primatte's nested RGB polyhedra, Vlahos
two-threshold keying) have used exactly this inner/outer-band structure
since the 1990s.

STATUS (2026-08-31, see the plan's Phase 4 section): measured on 感謝 --
F1i/F3 improve sharply (-33%/-26%) confirming the design's core idea works,
but S1 (chatter, +39%) and E1 (edge jaggedness, +9%) regress and the
completion condition was NOT met. A 3px median blur on the distance map
(below) recovers only a small fraction of that (S1 -2%, E1 -0.4%) -- the
likelier cause is the FG/BG seed masks' own hard-threshold boundary being
jagged before the ramp ever runs (untested next step, see the plan). Kept
available behind PipelineConfig.use_trimap (default False); NOT the
pipeline's default until that regression is resolved.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..qc import metrics as qc_metrics

# FG/BG erosion-dilation radius, as a fraction of the frame diagonal (plan
# 4-2: "r_in, r_out はフレーム対角の0.5%など、素材解像度から導出").
_RADIUS_FRAC = 0.005
_MIN_RADIUS = 2

# Median-blur kernel on the UNK-band distance map (plan 4's "3pxメディアン
# ブラー" experiment) -- small, measured net-positive (if marginal) so kept
# on unconditionally rather than adding a second flag for a fractional gain.
_DIST_MEDIAN_KSIZE = 3


def bg_stats_from_frames(rgb_frames):
    """Full-clip, sigma-clipped background chroma model from a plain list
    of (H,W,3) uint8 RGB frames (as read straight off cv2.VideoCapture) --
    reuses tool.qc.metrics's validated implementation (same sigma-floor and
    classification-gate fixes found while building the QC harness) so the
    trimap's notion of "background-like" matches the harness's F1/F2
    fidelity metrics exactly."""
    return qc_metrics.estimate_bg_stats(np.stack(rgb_frames, 0))


def _radius(h, w, radius_frac=None):
    frac = _RADIUS_FRAC if radius_frac is None else radius_frac
    diag = float(np.hypot(h, w))
    return max(_MIN_RADIUS, int(round(diag * frac)))


def build_trimap_alpha(a_raw, rgb, bg_stats, fg_seed_thresh=0.90, bg_seed_thresh=0.10,
                        seed=None, seed_variant="additive", band="color", radius_frac=None):
    """One frame. `a_raw`: BiRefNet's alpha ([0,1] float, any shape (H,W)).
    `rgb`: that frame's SOURCE colour (H,W,3) uint8 -- pass the true
    captured frame, not a despilled/processed one: despill can overwrite
    colour deep inside its crop bbox wherever alpha is near 0, which would
    corrupt the distance computation if that ever overlapped the boundary.
    `bg_stats`: from bg_stats_from_frames(). Only read when `band=="color"`
    (may be None when `band=="alpha"`, e.g. natural-background clips with
    no chroma model at all -- see レバーB).

    `seed` (B-2, レバーB "SAM2Matting 型分業"): an optional (H,W) mask
    (bool or {0,1}) giving a temporally-tracked FG/BG geometry (e.g. SAM2's
    per-frame tracked object mask) to use INSTEAD of the colour-threshold-
    derived FG/BG geometry below. When given, `seed_variant` selects one of
    the two pre-registered geometries (plan 第9計画 R7):
      - "additive" (default, primary per R7): SAM2 only ADDS foreground on
        top of BiRefNet's own high-confidence pixels and only removes
        background where BOTH the dilated-seed-complement AND BiRefNet's
        own low-alpha agree -- SAM2 can never erase detail BiRefNet already
        found (frills/horns/tail-tips, this pipeline's most-watched defect
        class): `FG = erode(seed) | (a_raw > fg_seed_thresh)`,
        `BG = ~dilate(seed) & (a_raw < bg_seed_thresh)`.
      - "symmetric" (comparison-only, plain SAM2Matting-style):
        `FG = erode(seed)`, `BG = ~dilate(seed)`.
    When `seed` is None (default), FG/BG geometry is unchanged from the
    original colour-threshold design (byte-identical to the pre-B-2
    behaviour) and `seed_variant` is ignored.

    `band` selects what fills the UNK strip between FG and BG:
      - "color" (default, unchanged behaviour): the colour-distance ramp
        against `bg_stats` (flat-backdrop chroma-key logic).
      - "alpha" (B-2): BiRefNet's own raw alpha (`a_raw`) directly, clipped
        to [0,1] -- for seed-driven trimaps on natural (non-chroma)
        backgrounds, where there is no bg_stats colour model to ramp
        against.

    `radius_frac` overrides the module-level `_RADIUS_FRAC` erosion/
    dilation radius fraction for this call only (default: None, meaning use
    `_RADIUS_FRAC`, unchanged behaviour).

    Returns (alpha, unk_mask): alpha is [0,1] float32 (H,W); unk_mask is
    the boolean band the ramp actually touched, so callers can restrict
    colour-correction (despill) to it (config.py's principle 2 -- confident
    FG/BG never gets colour-processed)."""
    H, W = a_raw.shape
    r_in = r_out = _radius(H, W, radius_frac)
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1, 2 * r_in + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1))

    if seed is None:
        fg_mask = cv2.erode((a_raw > fg_seed_thresh).astype(np.uint8), k_in) > 0
        bg_core = cv2.dilate((a_raw > bg_seed_thresh).astype(np.uint8), k_out) > 0
        bg_mask = ~bg_core
    else:
        seed_bool = np.asarray(seed).astype(np.uint8) > 0
        eroded_seed = cv2.erode(seed_bool.astype(np.uint8), k_in) > 0
        dilated_seed = cv2.dilate(seed_bool.astype(np.uint8), k_out) > 0
        if seed_variant == "additive":
            fg_mask = eroded_seed | (a_raw > fg_seed_thresh)
            bg_mask = (~dilated_seed) & (a_raw < bg_seed_thresh)
        elif seed_variant == "symmetric":
            fg_mask = eroded_seed
            bg_mask = ~dilated_seed
        else:
            raise ValueError(f"unknown seed_variant: {seed_variant!r} (expected "
                              f"'additive' or 'symmetric')")
    unk_mask = ~fg_mask & ~bg_mask

    if band == "alpha":
        ramp = np.clip(a_raw.astype(np.float32), 0.0, 1.0)
    else:
        d = qc_metrics.chroma_distance(rgb, bg_stats)
        d = cv2.medianBlur(d.astype(np.float32), _DIST_MEDIAN_KSIZE)
        ramp = np.clip((d - qc_metrics.T_LO) / (qc_metrics.T_HI - qc_metrics.T_LO), 0.0, 1.0)

    alpha = np.where(fg_mask, 1.0, np.where(bg_mask, 0.0, ramp)).astype(np.float32)
    return alpha, unk_mask

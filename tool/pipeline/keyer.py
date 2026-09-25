"""Pure-colour matte for flat-chroma-backdrop clips -- no neural network.

This is a SECOND matting path, not a replacement: it only runs for clips whose
backdrop actually gates as a flat chroma colour (see estimate_key/is_safe
below), and every other clip class keeps taking the BiRefNet route in
runner.infer_clip exactly as before.

Why a second path exists at all
-------------------------------
Every expensive stage in the neural path is there to compensate for a
BiRefNet weakness that simply does not arise when the backdrop is one flat
colour:

  YOLOX detect   gives BiRefNet a crop box        -> no box needed here
  mc_median      suppresses BiRefNet's frame-to-   -> this matte is a
                 frame chatter (optical flow,         deterministic per-pixel
                 the single most expensive            function of the source,
                 postprocess stage)                   so chatter has no source
  despill (ML)   estimates foreground colour      -> the background colour is
                 because the background is             KNOWN, so Smith & Blinn's
                 unknown                               unpremultiply is exact
  encode ss=4    hides BiRefNet's run-to-run      -> this matte is bit-exactly
                 alpha non-determinism near the       reproducible, so the
                 1-bit threshold (the plan's V4       cheap encoder is sound
                 root-cause finding)

So this is one structural change that removes four justifications at once,
rather than four independent micro-optimisations. Measured on a flat-chroma
mascot fixture (768x768, 124 frames) under an equally
loaded machine: 737s end-to-end on the neural path vs ~30-46s here, with F1i
(the primary fidelity gate) 42x better -- see the plan's 第7計画 section.

What this deliberately does NOT do
----------------------------------
* No interior-hole fill. A flood-fill "background region enclosed by subject
  must be opaque" guard was prototyped and MEASURED HARMFUL: it painted over
  the genuine see-through gap between the mascot's raised arm, body and tail.
  A real gap and a real hole are not distinguishable by topology alone, so
  this does not guess.
* No attempt to detect a subject that genuinely contains the key colour --
  that is not decidable from colour alone (Smith & Blinn 1996) and an earlier
  version's check for it turned out to be circular. See is_safe.
* No temporal filtering of any kind. Same input pixels always produce the
  same alpha, so there is nothing to stabilise.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..qc import metrics as qc_metrics

#: Ring width (px) sampled from the frame border to estimate the backdrop's
#: RGB. Matches qc_metrics.estimate_bg_stats' own default so both halves of
#: the model describe the same pixels.
_BG_RING = 24

#: Lower ramp end, in units of the backdrop's own chroma sigma -- reused from
#: the QC metrics so "background-like" means the same thing here as it does
#: in the fidelity metrics that grade this module's output.
_T_LO = qc_metrics.T_LO

#: Minimum chroma saturation (raw Cb/Cr levels away from neutral 128,128) the
#: backdrop must have for this path to be usable at all.
#:
#: This is THE gate, and it encodes why chroma keying works in the first
#: place: the backdrop must be a saturated colour the subject doesn't contain.
#: A neutral backdrop -- a white/grey cyclorama, a black stage -- has no
#: chroma to key on, so any desaturated part of the SUBJECT (a white belly, a
#: grey marking, a shadow) is chromatically identical to it and would be cut
#: out. Measured across this repo's fixture sets:
#:
#:   flatchroma2 mascot batch (blue key)      61.0   <- real key
#:   purplebg real footage (purple key)        42.1   <- real key
#:   dinosaur/Triceratops (forest)            10.0
#:   white-cyclorama real footage (white)      2.0   <- NOT a key, despite
#:   office photo                              1.4      gating as "flat"
#:
#: 20 sits in the middle of a 4x gap with 2x margin either side. Note what
#: this catches that the flatness check alone does not: the white-cyclorama set reports
#: frac_bg_like=1.0000 (its backdrop genuinely IS uniform) and would otherwise
#: have taken this path.
MIN_KEY_SATURATION = 20.0


@dataclass(frozen=True)
class KeyModel:
    """Everything needed to matte one clip, estimated once from that clip.

    Every field is MEASURED from the source, not configured: the plan's
    design principle 1 ("only use signals you can measure") is what keeps
    this from becoming another hand-tuned colour heuristic of the kind that
    misfired on three separate client assets (see config.py's strip_bg_fringe
    note)."""
    mu_cb: float
    mu_cr: float
    sigma_cb: float
    sigma_cr: float
    #: Backdrop colour in RGB, for the unpremultiply. estimate_bg_stats works
    #: in the chroma plane only, which is the right space for DECIDING alpha
    #: (luma varies with shading, chroma doesn't) but cannot reconstruct the
    #: colour to subtract.
    bg_rgb: np.ndarray
    t_lo: float
    t_hi: float
    #: Distance (in sigma) of a perfectly NEUTRAL pixel from the backdrop --
    #: the natural scale of this key. Any subject colour that isn't itself
    #: tinted toward the key sits at least this far away, so it is the right
    #: quantity to express the ramp as a fraction of. Derived from the
    #: backdrop alone, which is what keeps it non-circular (an earlier version
    #: tried to measure the nearest SUBJECT pixel, but the subject could only
    #: be located by the same distance being thresholded, so the answer was
    #: pinned to that threshold on every clip -- see DECISIONS.md).
    neutral_dist: float
    #: Backdrop chroma saturation in raw Cb/Cr levels (see MIN_KEY_SATURATION).
    key_saturation: float
    is_chroma_class: bool
    frac_bg_like: float
    #: 第11計画 Part 3-3: unmix edge colour with a separate COVERAGE estimate
    #: (the ramp widened to neutral_dist) instead of the display alpha -- see
    #: key_frame. False reproduces the pre-2026-09-23 keyer exactly.
    unmix_coverage: bool = False
    #: 第11計画 Part 3-4: p99.9 of this clip's own far-backdrop distance, when
    #: estimate_key measured it (None otherwise) -- the source of t_lo's floor.
    backdrop_p999: float | None = None


def _chroma_distance(rgb, model_or_stats):
    """Per-pixel distance from the backdrop chroma, in sigma units.

    Same definition as qc_metrics.chroma_distance, but computed in float32
    in place. The metrics version lets uint8 channels promote to float64,
    which costs ~19ms/frame at 768x768 -- fine for grading a handful of
    outputs, far too slow for a per-frame matting stage.
    """
    m = model_or_stats
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    cb -= np.float32(m.mu_cb)
    cb *= np.float32(1.0 / m.sigma_cb)
    cr -= np.float32(m.mu_cr)
    cr *= np.float32(1.0 / m.sigma_cr)
    cb *= cb
    cr *= cr
    cb += cr
    return cv2.sqrt(cb)


#: Part 3-3's edge band: opaque pixels within this many px of a non-opaque
#: one also get their colour unmixed (their display alpha is 1 but their
#: true coverage need not be). Pre-registered, not tuned.
_UNMIX_EDGE_PX = 2
_UNMIX_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _UNMIX_EDGE_PX + 1,) * 2)

#: Part 3-4: a pixel counts as "far backdrop" for the t_lo floor when it is
#: more than this fraction of the frame's long side from every pixel the key
#: calls opaque -- far enough to exclude the subject's edge blend, chroma-
#: subsampling bleed and contact shadow, which is what makes the percentile
#: describe the BACKDROP rather than the subject touching the frame edge
#: (the plain border ring's p99.9 read 24-29 sigma on clips where it does).
_FAR_BACKDROP_FRAC = 0.05
_BACKDROP_PCT = 99.9


def _far_backdrop_p999(frames, stats, t_hi):
    """p99.9 of the chroma distance over far-backdrop pixels of `frames`, or
    None if there are none (a subject filling the frame)."""
    H, W = frames.shape[1:3]
    far_px = _FAR_BACKDROP_FRAC * max(H, W)
    vals = []
    for f in frames:
        d = _chroma_distance(f, stats)
        opaque = (d > t_hi).astype(np.uint8)
        if opaque.any():
            dist = cv2.distanceTransform(1 - opaque, cv2.DIST_L2, 5)
            vals.append(d[dist > far_px])
        else:
            vals.append(d.ravel())
    vals = np.concatenate(vals) if vals else np.empty(0, np.float32)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, _BACKDROP_PCT))


def estimate_key(rgb_frames, ramp_k: float = 0.5, *, unmix_coverage: bool = False,
                 measured_t_lo: bool = False) -> KeyModel:
    """Build a KeyModel from a clip's frames (list/array of (H,W,3) uint8 RGB).

    `ramp_k` sets the ramp top as a fraction of `neutral_dist` -- the distance
    a perfectly grey pixel sits from this backdrop, i.e. the natural scale of
    this particular key. Scaling THAT (rather than a fixed sigma count) is
    what makes one k work across key colours: a strongly saturated key gets a
    proportionally wider ramp than a weak one, automatically.

    The ramp cannot reuse the QC metrics' own T_LO/T_HI (3 and 8 sigma),
    which are calibrated for CLASSIFYING pixels rather than generating alpha:
    with sigma pinned at its 1.0 floor on a clean render and a neutral pixel
    sitting ~61 sigma out, T_HI=8 makes a pixel only 13% of the way from
    backdrop to subject fully opaque, stranding the boundary's blended pixels
    at alpha=1 with their key colour intact (measured: E2 fringe quality 24 at
    T_HI=8 vs 55 at T_HI=35, while opaque area moved only 17.17% -> 16.40% --
    i.e. widening eats the contaminated ring, not the subject).
    """
    frames = np.asarray(rgb_frames) if not isinstance(rgb_frames, np.ndarray) else rgb_frames
    if frames.ndim == 3:
        frames = frames[None]
    stats = qc_metrics.estimate_bg_stats(frames, ring=_BG_RING)

    ring_px = np.concatenate([
        frames[:, :_BG_RING].reshape(-1, 3), frames[:, -_BG_RING:].reshape(-1, 3),
        frames[:, :, :_BG_RING].reshape(-1, 3), frames[:, :, -_BG_RING:].reshape(-1, 3),
    ])
    # audit M10: `stats` (mu_cb/mu_cr) is sigma-clipped by estimate_bg_stats,
    # but a plain median over ALL of ring_px is not -- a mascot resting
    # against the frame edge for most of the clip contaminates the ring with
    # subject-coloured pixels, and the chroma stats stay correct (clipping
    # handles that) while bg_rgb quietly drifts toward the subject. Since
    # key_frame's unpremultiply subtracts bg_rgb from a blended edge pixel's
    # colour, a wrong bg_rgb pushes that correction in the wrong direction --
    # a visible colour halo, not merely a missed correction. Restrict the
    # median to ring pixels that are themselves confidently background-like
    # by the SAME distance/threshold this module uses everywhere else (T_LO),
    # matching the chroma stats' own clipping intent rather than introducing
    # a second, unrelated one.
    ring_ycc = cv2.cvtColor(ring_px.reshape(-1, 1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    ring_d = np.hypot((ring_ycc[:, 2].astype(np.float32) - stats.mu_cb) / stats.sigma_cb,
                      (ring_ycc[:, 1].astype(np.float32) - stats.mu_cr) / stats.sigma_cr)
    bg_like = ring_d < _T_LO
    bg_rgb = np.median(ring_px[bg_like] if bg_like.any() else ring_px, axis=0).astype(np.float32)

    key_saturation = float(np.hypot(stats.mu_cb - 128.0, stats.mu_cr - 128.0))
    # Distance of a neutral grey pixel from this backdrop, in the very same
    # sigma units everything downstream uses -- computed by running the real
    # distance function on a grey pixel rather than re-deriving the formula.
    neutral_dist = float(_chroma_distance(np.full((1, 1, 3), 128, np.uint8), stats)[0, 0])

    t_lo = _T_LO
    t_hi = max(_T_LO + 1.0, ramp_k * neutral_dist)
    backdrop_p999 = None
    if measured_t_lo:
        # 第11計画 Part 3-4: 3 sigma is the QC metrics' CLASSIFICATION
        # threshold, but a real backdrop (compression noise, faint shading)
        # can sit further out than that, and every such pixel became a faint
        # soft-alpha haze -- invisible in a 1-bit GIF, visible in webp/mov.
        # Floor t_lo at the clip's own far-backdrop p99.9 instead, never
        # below 3 sigma and never above half the ramp top.
        backdrop_p999 = _far_backdrop_p999(frames, stats, t_hi)
        if backdrop_p999 is not None:
            t_lo = min(max(_T_LO, backdrop_p999), max(_T_LO, 0.5 * t_hi))
        t_hi = max(t_lo + 1.0, t_hi)

    return KeyModel(mu_cb=stats.mu_cb, mu_cr=stats.mu_cr, sigma_cb=stats.sigma_cb,
                    sigma_cr=stats.sigma_cr, bg_rgb=bg_rgb,
                    t_lo=t_lo, t_hi=t_hi,
                    neutral_dist=neutral_dist, key_saturation=key_saturation,
                    is_chroma_class=stats.is_chroma_class, frac_bg_like=stats.frac_bg_like,
                    unmix_coverage=unmix_coverage, backdrop_p999=backdrop_p999)


def probe_says_keyable(bg_is_chroma_class, bg_key_saturation) -> bool:
    """is_safe()'s rule, evaluated from server/probe.py's upload-time numbers
    instead of a KeyModel.

    Exists so the server can know BEFORE a job starts whether it will need the
    GPU at all: a clip this returns True for is matted without loading
    BiRefNet/YOLOX, so waiting for free VRAM (and occupying it afterwards)
    would be pure delay. Kept next to is_safe so the two rules cannot drift.
    Falsy/None inputs -> False, since "not probed" must never read as
    "safe to skip the network"."""
    if not bg_is_chroma_class or bg_key_saturation is None:
        return False
    return bg_key_saturation >= MIN_KEY_SATURATION


def is_safe(model: KeyModel) -> tuple[bool, str]:
    """Whether this clip may take the colour-only path.

    Returns (ok, reason); `reason` is always populated so a refusal can be
    logged and recorded on the job rather than silently changing behaviour.

    Two conditions, both measured from the backdrop alone:

    1. the backdrop is FLAT (one colour, not a scene), and
    2. that colour is SATURATED enough to key on (MIN_KEY_SATURATION).

    Condition 2 is not redundant: a white cyclorama is perfectly flat and
    passes condition 1 outright (the white-cyclorama fixtures report frac_bg_like=1.0000)
    while being unkeyable, because a neutral backdrop cannot be told apart
    from the subject's own neutral colours.

    What this deliberately does NOT claim to check is the remaining risk --
    a subject that genuinely contains the key colour. That is not decidable
    from colour alone (Smith & Blinn 1996), and an earlier attempt to check it
    by finding "the nearest subject pixel" was circular: the subject could
    only be located by thresholding the same distance, so the answer came back
    pinned to that threshold on every clip, making the gate vacuous while
    looking like it worked. It is the operator's job at shoot/render time --
    for these generated assets the key colour is chosen explicitly (the render
    batch records it in report.md) -- and the output is visibly wrong if it is
    ever violated.
    """
    if not model.is_chroma_class:
        return False, f"backdrop is not flat chroma (frac_bg_like={model.frac_bg_like:.4f})"
    if model.key_saturation < MIN_KEY_SATURATION:
        return False, (f"backdrop is flat but near-neutral (chroma saturation "
                       f"{model.key_saturation:.1f} < {MIN_KEY_SATURATION:.0f}) -- "
                       f"a white/grey/black backdrop has no chroma to key on")
    return True, (f"flat chroma backdrop (frac_bg_like={model.frac_bg_like:.4f}, "
                  f"saturation {model.key_saturation:.0f})")


def key_frame(rgb, model: KeyModel) -> np.ndarray:
    """Matte one frame -> (H,W,4) uint8 RGBA (R,G,B,A order).

    Alpha is a ramp on the measured backdrop distance; colour is then
    corrected by C = alpha*F + (1-alpha)*B solved for F, which is computable
    at all only because B is known and constant here (Smith & Blinn 1996 prove
    the system is under-determined for an unknown background -- precisely why
    the neural path needs pymatting's iterative estimator and this one does
    not).

    Note what this is NOT: the solve is exact given the TRUE coverage, but the
    ramp supplies an ESTIMATE of it (alpha = d/t_hi), and with the shipped
    ramp_k=0.5 a pixel that is genuinely ~50% covered already reads as fully
    opaque, so its share of backdrop colour is not removed. That is the
    residual edge tint the E2 measurements track, and it is the standard
    ramp-keyer compromise: pushing ramp_k toward 1.0 would make the estimate
    truer but leaves no margin for a subject colour dimmer than the sampled
    frames happened to show (see is_safe). Measured across ramp_k 0.3-0.65,
    E2 improves monotonically with ramp_k while F1i stays flat; 0.5 is the
    chosen point, with a 2.0x safety margin on this fixture set.

    With model.unmix_coverage (PipelineConfig.keyer_unmix_coverage, the
    pipeline default since 2026-09-23) that residual is addressed without
    touching the alpha: see the branch below. Without it (the legacy path,
    still what estimate_key builds by default for direct callers):
    the correction is applied ONLY where 0 < alpha < 1 (measured: ~1.5% of
    pixels). At alpha=1 it is the identity, and at alpha=0 the pixel is fully
    transparent, so touching either is wasted work.
    """
    d = _chroma_distance(rgb, model)
    d_raw = d.copy() if model.unmix_coverage else None
    d -= np.float32(model.t_lo)
    d *= np.float32(1.0 / (model.t_hi - model.t_lo))
    alpha = np.clip(d, 0.0, 1.0, out=d)

    # np.rint, not a bare cast: casting truncates, so a nominally-opaque
    # ramp top (alpha=0.999...) would land at 254 rather than 255 -- never
    # visible on a 1-bit GIF (alpha_threshold snaps it back to opaque either
    # way) but a real, visible 1-level dimming on mov/webp output, which
    # keep the soft alpha channel as-is (audit L1).
    out = np.dstack([rgb, np.rint(alpha * 255).astype(np.uint8)])
    if model.unmix_coverage:
        # 第11計画 Part 3-3: the display alpha stays exactly as above (the
        # matte's SHAPE does not change), but the colour is unmixed with a
        # truer coverage estimate -- the same ramp widened to neutral_dist,
        # where a pixel is only called fully covered once it is as far from
        # the key as a neutral grey -- and over an edge band that includes
        # opaque pixels within _UNMIX_EDGE_PX of the boundary. With ramp_k
        # =0.5 a ~50%-covered edge pixel displays as opaque, so the display-
        # alpha unmix (below) never touched it and its key-colour share
        # stayed in: the purple outline on the purplebg fixture (6.6k of ~7.5k
        # outer-ring px near the backdrop colour). Restricted to the edge
        # band so a genuinely key-tinted subject INTERIOR is never recoloured.
        soft = (alpha < 0.996).astype(np.uint8)
        near_edge = cv2.dilate(soft, _UNMIX_KERNEL).astype(bool)
        span = max(model.neutral_dist - model.t_lo, model.t_hi - model.t_lo)
        cov = np.clip((d_raw - np.float32(model.t_lo)) * np.float32(1.0 / span), 0.0, 1.0)
        band = near_edge & (alpha > 0.004) & (cov < 0.996)
        if band.any():
            c_band = np.maximum(cov[band], 0.004)[:, None]
            fg = (rgb[band].astype(np.float32) - (1.0 - c_band) * model.bg_rgb) / c_band
            out[band, :3] = np.clip(fg, 0, 255).astype(np.uint8)
        return out
    band = (alpha > 0.004) & (alpha < 0.996)
    if band.any():
        a_band = alpha[band][:, None]
        fg = (rgb[band].astype(np.float32) - (1.0 - a_band) * model.bg_rgb) / a_band
        out[band, :3] = np.clip(fg, 0, 255).astype(np.uint8)
    return out

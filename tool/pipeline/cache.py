"""Disk cache for infer_clip()'s result, so Phase 3's ablation over
postprocess stages (apply_clears / collapse_fix / temporal_alpha_smooth /
strip_bg_fringe) doesn't re-run GPU inference once per config -- none of
those stages touch anything upstream of infer_clip.

Cache lives in a scratch dir, not the repo (frames alone run ~1GB/clip).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

# Bump whenever infer_clip/chroma.py's ALGORITHM changes in a way that
# would silently change a cached result's meaning -- e.g. adding the
# median-blur step to build_trimap_alpha did exactly this, and without a
# version in the key an old cache entry would keep matching (found in
# audit N2; worked around that one time by deleting the file by hand).
# Bumped to 3 for the YOLOX device fix (matte_core.Models no longer
# hardcodes the detector to CPU): the detector's box can now differ
# between "cpu" and "cuda" runs (float differences straddle the conf
# threshold, min_box_frac, and the int() crop-rect truncation), which
# was invisible to this key before device was added below (audit A1).
CACHE_VERSION = 7  # 6: box_mode (レバーC) -- the boxes BiRefNet crops to
# depend on config.box_mode, which a v5 entry could not reflect. 7 (第11計画):
# subject_box no longer rejects a small confident person box (a different
# crop -> different frames for the same config), per_frame's truncation
# repair is the single-frame full-frame fallback again, and the result now
# carries "boxes" (read by the postprocess-only keep_main_subject, so that
# flag itself is deliberately NOT in _key).

# Was hardcoded to a Claude Code session's own /tmp scratchpad path -- gone
# the moment that session ended, silently forcing a full re-inference on
# every following run. PIPELINE_CACHE_DIR lets any long-lived caller (the
# server, an ablation script) point this somewhere stable of their own;
# the default lives under this repo's data/ (gitignored, not the repo
# itself -- frames alone run ~1GB/clip).
CACHE_DIR = Path(os.environ.get(
    "PIPELINE_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / "data" / "pipeline_cache")))


def _key(clip_name, config):
    """Every config field `infer_clip` actually reads before/while building
    the cached `frames` must be in this key -- a postprocess-only field
    (e.g. strip_bg_fringe) must NOT be, or that just bloats the cache with
    duplicates. Getting this wrong is silent: a stale cache entry looks
    like a successful hit and returns a result computed under a DIFFERENT
    config (audit N1 -- apply_clears/clear_* were missing here, so turning
    apply_clears on after caching with it off would silently keep returning
    the empty-candidate result)."""
    return (f"{clip_name}__v{CACHE_VERSION}"
            f"__device{config.device}"
            f"__scale{config.scale}_maxside{config.max_side}"
            f"__box{config.min_box_frac}_{config.box_mode}__refine{config.matte_refine}"
            f"__despill{config.apply_despill}_{config.despill_band_only}_{config.despill_est_scale}"
            f"__clears{config.apply_clears}_{config.clear_sat_thresh}_{config.clear_min_area}"
            f"_{config.clear_min_area_link}_{config.clear_dist_thresh}_{config.clear_zoom_window_mult}"
            f"_{config.clear_zoom_min_side}_{config.clear_zoom_bg_frac}"
            f"__trimap{config.use_trimap}_{config.trimap_fg_seed_thresh}_{config.trimap_bg_seed_thresh}"
            f"__keyer{config.use_keyer}_{config.keyer_ramp_k}"
            f"_{config.keyer_unmix_coverage}_{config.keyer_measured_t_lo}")


def save_raw(clip_name, config, raw):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key(clip_name, config)
    frames = np.stack(raw["frames"], 0)
    # The keyer path (tool/pipeline/keyer.py) returns src_gray=[] -- it has
    # no optical-flow input to offer, unlike the BiRefNet route -- and
    # np.stack([]) raises ValueError rather than yielding a usable empty
    # array (audit H1: this made save_raw crash on every flat-chroma clip,
    # taking --parallel-encode and speed_experiment.py down with it).
    src_gray = np.stack(raw["src_gray"], 0) if raw["src_gray"] else np.empty((0,), np.uint8)
    # savez_compressed: measured 1008MB -> 192MB (5.3x) on a 122-frame clip,
    # for ~12s extra write time and, surprisingly, FASTER reads (1.3s vs
    # plain savez) -- decompression is cheaper than the larger disk read it
    # replaces. Strict win given this cache exists specifically to avoid
    # GPU re-inference, not to avoid a few seconds of (de)compression.
    np.savez_compressed(CACHE_DIR / f"{key}.npz",
              frames=frames, src_gray=src_gray,
              motion_dmax=raw["motion_dmax"] if raw["motion_dmax"] is not None else np.array([]),
              bg_ref=raw["bg_ref"] if raw["bg_ref"] is not None else np.array([]),
              fps=raw["fps"],
              # audit H2: load_raw used to drop this entirely, so a cached
              # keyed result came back indistinguishable from a BiRefNet one
              # -- run_clip would then try to run mc_median on an empty
              # src_gray (IndexError) and skip the ss_alpha_gif encoder swap
              # that makes the keyer path fast in the first place.
              keyed=np.array(bool(raw.get("keyed"))),
              # Per-frame raw YOLOX boxes for keep_main_subject: (N, 4) with a
              # NaN row where a frame had no box; (0,) on the keyer route.
              boxes=_boxes_to_array(raw.get("boxes")))
    with open(CACHE_DIR / f"{key}.clearmasks.pkl", "wb") as f:
        pickle.dump(raw["clear_masks"], f)
    print(f"  [cache] saved {key}", flush=True)


def load_raw(clip_name, config):
    key = _key(clip_name, config)
    npz_path = CACHE_DIR / f"{key}.npz"
    pkl_path = CACHE_DIR / f"{key}.clearmasks.pkl"
    if not (npz_path.exists() and pkl_path.exists()):
        return None
    d = np.load(npz_path)
    with open(pkl_path, "rb") as f:
        clear_masks = pickle.load(f)
    result = {
        "frames": [f for f in d["frames"]],
        "src_gray": [g for g in d["src_gray"]],
        "motion_dmax": d["motion_dmax"] if d["motion_dmax"].size else None,
        "bg_ref": d["bg_ref"] if d["bg_ref"].size else None,
        "fps": float(d["fps"]),
        "clear_masks": clear_masks,
    }
    # "keyed" was added after some caches were written -- an old npz simply
    # won't have the key, and a bool(False) round-trips as falsy either way,
    # so .get(...) with a default reproduces the pre-existing (BiRefNet-route)
    # behaviour for anything cached before this fix.
    if "keyed" in d.files and bool(d["keyed"]):
        result["keyed"] = True
    result["boxes"] = _boxes_from_array(d["boxes"]) if "boxes" in d.files else None
    return result


def _boxes_to_array(boxes):
    if boxes is None:
        return np.empty((0,), np.float64)
    out = np.full((len(boxes), 4), np.nan, np.float64)
    for i, b in enumerate(boxes):
        if b is not None:
            out[i] = np.asarray(b, np.float64)
    return out


def _boxes_from_array(arr):
    if arr.ndim != 2:
        return None
    return [None if np.isnan(row).any() else row for row in arr]


def infer_clip_cached(video_path, config, models=None, clip_name=None):
    """infer_clip(), but reusing a prior cached result keyed by (clip_name,
    upstream-relevant config) when available."""
    from .runner import infer_clip
    name = clip_name or Path(video_path).stem
    cached = load_raw(name, config)
    if cached is not None:
        print(f"  [cache] hit for {name}", flush=True)
        return cached
    raw = infer_clip(video_path, config, models=models)
    save_raw(name, config, raw)
    return raw

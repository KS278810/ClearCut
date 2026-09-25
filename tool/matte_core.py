"""Shared matte primitives for image and video background removal.

Permissive-only stack (Apache-2.0 / MIT / BSD only — no AGPL/GPL):
  - YOLOX-S (ONNX)   : subject/person box detection, onnxruntime-only (Apache-2.0)
  - SAM2 Hiera-Tiny  : box-prompt subject selection / region gate (Apache-2.0)
  - BiRefNet_lite    : high-resolution alpha matte (MIT)
  - pymatting        : closed-form matting edge refine (MIT)

CPU by default; pass device="cuda" (or leave "auto") to use a CUDA GPU when
available. Both bg_remove_image.py and bg_remove_video.py build on Models here
so the detect -> select -> matte -> refine chain is defined exactly once.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label as _cc_label

_NVIDIA_DLL_DIRS: list | None = None


def _nvidia_dll_dirs() -> list:
    """bin/ dirs of the nvidia-*-cu12 pip packages (cuBLAS/cuDNN/cudart/...),
    memoised. Empty on the CPU-only install (those packages aren't installed
    there) or on non-Windows."""
    global _NVIDIA_DLL_DIRS
    if _NVIDIA_DLL_DIRS is not None:
        return _NVIDIA_DLL_DIRS
    dirs: list = []
    if sys.platform == "win32":
        import importlib.util
        # find_spec("nvidia.<pkg>") imports the "nvidia" parent to resolve the
        # dotted name; on the CPU-only install (no nvidia-*-cu12 packages at
        # all) that raises ModuleNotFoundError instead of returning None.
        if importlib.util.find_spec("nvidia") is not None:
            for pkg in ("cublas", "cudnn", "cuda_nvrtc", "cufft", "curand",
                        "nvjitlink", "cuda_runtime"):
                try:
                    spec = importlib.util.find_spec(f"nvidia.{pkg}")
                except ModuleNotFoundError:
                    continue
                if spec and spec.submodule_search_locations:
                    bin_dir = Path(list(spec.submodule_search_locations)[0]) / "bin"
                    if bin_dir.is_dir():
                        dirs.append(str(bin_dir))
    _NVIDIA_DLL_DIRS = dirs
    return dirs


_NVIDIA_DLLS_ON_PATH = False


def _ensure_nvidia_dlls_on_path() -> None:
    """Permanently prepend the nvidia-*-cu12 pip packages' bin/ dirs to PATH so
    onnxruntime-gpu's CUDAExecutionProvider can find cuBLAS/cuDNN/cudart — those
    packages install into site-packages, not PATH, and os.add_dll_directory
    alone does NOT fix this (confirmed by testing: onnxruntime's internal load
    of its CUDA provider DLL doesn't use the LOAD_LIBRARY_SEARCH_* directories
    add_dll_directory registers; only the PATH env var actually works).

    Cannot be scoped to just session creation: cuDNN's Frontend is initialised
    lazily on the FIRST inference call, not at InferenceSession() time, so a
    context manager around session creation alone leaves PATH already reverted
    by the time a Conv node actually needs cuDNN (confirmed by testing).

    Safe to call without torch ever having been imported in this process: SAM2
    (the only thing that needs torch) runs in its own subprocess — see
    _Sam2Worker — so the main process's onnxruntime sessions are the only CUDA
    consumer here and there's no torch-bundled-cuDNN-vs-pip-cuDNN conflict to
    order around."""
    global _NVIDIA_DLLS_ON_PATH
    if _NVIDIA_DLLS_ON_PATH:
        return
    dirs = _nvidia_dll_dirs()
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
    _NVIDIA_DLLS_ON_PATH = True

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
BRF = CKPT_DIR / "birefnet_lite.onnx"
YOLOX_ONNX = CKPT_DIR / "yolox_s.onnx"
SAM2_CKPT = CKPT_DIR / "sam2.1_hiera_tiny.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # hydra config name bundled in the sam2 package

_YOLO_PERSON = 0  # COCO 'person' class index
#: subject_box's min_box_frac is not applied to a person box at or above
#: this confidence -- see subject_box's docstring (第11計画 Part 3-2).
_CONFIDENT_PERSON_CONF = 0.8
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------- device selection (CPU/GPU switch) ----------

def resolve_device(pref: str = "auto") -> str:
    """Resolve "auto"/"cpu"/"cuda" to the device actually used ("cpu" or "cuda").

    "auto" picks "cuda" only when both onnxruntime reports CUDAExecutionProvider
    and torch reports a usable CUDA device; otherwise falls back to "cpu".
    """
    if pref in ("cpu", "cuda"):
        return pref
    try:
        import onnxruntime as ort
        has_ort_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        has_ort_cuda = False
    try:
        import torch
        has_torch_cuda = torch.cuda.is_available()
    except Exception:
        has_torch_cuda = False
    return "cuda" if (has_ort_cuda and has_torch_cuda) else "cpu"


def _onnx_providers(device: str) -> list:
    if device == "cuda":
        # cudnn_conv_algo_search=HEURISTIC (default is EXHAUSTIVE): cuDNN 9's
        # default conv algo search tries its runtime-compiled JIT engine, which
        # NVIDIA restricts to Ampere+ (confirmed by testing: EXHAUSTIVE failed
        # with CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE on an OLDER
        # Turing/cc7.5 GPU this project ran on previously). HEURISTIC picks a
        # precompiled algo directly, sidestepping the JIT path entirely.
        #
        # Re-verified 2026-09-13 on this deployment's actual GPU (Blackwell-class,
        # cc12.0 -- Ampere+, so the restriction above no longer
        # applies): EXHAUSTIVE now creates a session fine here, and on a real
        # clip (挨拶.mp4, 122 frames, idle GPU) both EXHAUSTIVE and HEURISTIC
        # produced BYTE-IDENTICAL alpha (sha256 match) across separate process
        # launches -- no run-to-run nondeterminism reproduced under today's
        # (uncontended-GPU) conditions with either setting. This does NOT
        # necessarily explain away the F2 defect investigated in
        # tool/docs/DECISIONS.md (2026-09-03, on this same shared machine but
        # possibly under GPU contention from other users at the time) --
        # HEURISTIC's algo choice is a static lookup keyed on tensor shape, so
        # it should be UNAFFECTED by concurrent load; EXHAUSTIVE times
        # candidate kernels at runtime and could in principle pick a
        # DIFFERENT winner under different contention, i.e. no theoretical
        # reason to expect EXHAUSTIVE to be MORE stable under load than
        # HEURISTIC. Left on HEURISTIC (proven stable today, and cheaper to
        # initialize) rather than switching on inconclusive grounds -- flag
        # for whoever revisits this: a real repro would need reproducing the
        # original defect under deliberately reproduced GPU contention, which
        # this investigation did not attempt.
        #
        # arena_extend_strategy=kSameAsRequested (default is kNextPowerOfTwo):
        # confirmed by testing that kNextPowerOfTwo makes BiRefNet's 1024x1024
        # conv inference collapse from ~1s to ~20-30s starting on the SECOND
        # call (first call is fast, so this isn't a one-time warmup cost) —
        # repeated on this machine across both HEURISTIC and DEFAULT algo
        # search, so the memory-arena growth policy is the actual cause, not
        # algo search. kSameAsRequested keeps every call at ~0.8s in isolation.
        #
        # NOTE on SAM2 (torch): running onnxruntime (this session) and torch's
        # CUDA context in the SAME process was confirmed by testing to make
        # BOTH collapse by 10-30x once they alternate GPU calls (BiRefNet
        # ~0.8s->5-30s, SAM2 ~0.5s->13-20s) — a stream-sharing workaround
        # (user_compute_stream) only partially helped and didn't reproduce
        # outside a minimal repro. Running SAM2 in its own subprocess (see
        # _Sam2Worker) avoids the problem entirely rather than fighting it, so
        # this process's onnxruntime session never shares a CUDA context with
        # torch at all.
        opts = {"cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kSameAsRequested"}
        return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ---------- input sanity gates ----------
#
# Cheap pre-flight checks for a CLI tool run by an end user on their own
# machine (not a server processing untrusted uploads at scale): the goal is
# failing fast with a clear message before a multi-minute job, not surviving
# every conceivable input. See README's "hardening" note.

_MAX_MEGAPIXELS = 50.0  # ~an 8000x6000 photo; well above any of this project's
                        # sample content, high enough to not bother normal users


def check_frame_size(w: int, h: int, allow_large: bool) -> None:
    """Raise if a frame is unusually large, unless the caller opted in via
    --allow-large. Catches an unwitting user pointing this at e.g. a 100MP
    photo or 8K video -- which would otherwise silently turn into a very
    long, memory-hungry run with no upfront warning."""
    mp = (w * h) / 1e6
    if mp > _MAX_MEGAPIXELS and not allow_large:
        raise ValueError(
            f"input is {mp:.0f} megapixels ({w}x{h}), over the "
            f"{_MAX_MEGAPIXELS:.0f}MP sanity limit. Pass --allow-large to "
            f"process it anyway (may be slow and memory-heavy)."
        )


def check_disk_space(out_dir, estimated_bytes: float, margin: float = 1.5) -> None:
    """Raise if the output directory's free space looks insufficient for the
    estimated output size (with a safety margin for encoder overhead) --
    catches running out of disk space midway through a long encode, which
    otherwise fails late with a confusing partial-file error."""
    import shutil
    free = shutil.disk_usage(out_dir).free
    needed = estimated_bytes * margin
    if free < needed:
        raise RuntimeError(
            f"insufficient disk space at {out_dir}: need ~{needed / 1e6:.0f}MB "
            f"(estimated), only {free / 1e6:.0f}MB free."
        )


# ---------- post-processing primitives (model-agnostic) ----------

def _topology_clean(a, area_frac=0.003):
    """Enforce 'one subject, no internal holes': drop connected components smaller
    than area_frac of the frame (removes stray objects) and fill holes inside the
    subject. Use this for a single still image (no previous-frame state)."""
    binm = (a > 0.5).astype(np.uint8)
    H, W = a.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binm)
    keep = np.zeros_like(binm)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= area_frac * H * W:
            keep[lab == c] = 1
    m = a * keep
    return np.maximum(m, binary_fill_holes(m > 0.5).astype(np.float32))


def _topology_temporal(a, prev_final, gray_cur, gray_prev, dis, gx, gy, area_frac=0.003, flow=None):
    """Topology-change-aware cleanup for video. Drops small stray blobs, then fills
    ONLY holes that were FOREGROUND in the previous frame (genuine matte holes). A
    hole that was BACKGROUND last frame is a region newly enclosed by a moving limb
    -> kept transparent instead of filled white.

    Holes are filled with the flow-warped previous alpha's actual value, not a
    binary 0/1 -- preserves soft/semi-transparent structure (fur edges, motion
    blur) that a hard fill would flatten."""
    binm = (a > 0.5).astype(np.uint8)
    H, W = a.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binm)
    keep = np.zeros_like(binm)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] >= area_frac * H * W:
            keep[lab == c] = 1
    m = a * keep
    fg = m > 0.5
    holes = binary_fill_holes(fg) & ~fg
    if prev_final is not None:
        fl = flow if flow is not None else dis.calc(gray_cur, gray_prev, None)
        wprev = cv2.remap(prev_final, (gx + fl[..., 0]).astype(np.float32),
                          (gy + fl[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
        fillable = holes & (wprev > 0.5)
        fill_values = np.where(fillable, wprev, 0.0).astype(np.float32)
    else:
        fill_values = holes.astype(np.float32)
    return np.maximum(m, fill_values)


def _armback(a_gated, a_raw, reg, flow, move_tau=1.5):
    """Restore a per-frame matte's MOVING regions that a region gate dropped, IFF
    they are connected to the kept body. STATIC CAMERA ONLY (else camera motion
    makes everything 'move').

    The region gate (a = a_raw * reg) removes clutter (a static prop) but can also
    drop thin, motion-blurred, low-contrast limbs the gate misses — e.g. an arm
    extended over the floor. Distinguish limb from prop by MOTION: the limb is
    moving (that's why the gate blurred over it); a prop is static.

    a_raw: the per-frame matte model's raw alpha before the region gate (any
    matte model — BiRefNet, RVM, ...). reg: the region-gate mask (SAM2, MobileSAM,
    ...) that was multiplied into a_raw to make a_gated.
    """
    raw_fg = a_raw > 0.5
    kept = a_gated > 0.5
    lost = raw_fg & (reg < 0.5)
    if not lost.any() or not kept.any():
        return a_gated
    lbl, _ = _cc_label(raw_fg)
    body = set(np.unique(lbl[kept])) - {0}
    conn = np.isin(lbl, list(body))  # limb must hang off the body, not float free
    mag = np.hypot(flow[..., 0], flow[..., 1])
    moving = cv2.dilate((mag > move_tau).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    add = lost & conn & moving
    return np.maximum(a_gated, a_raw * add.astype(np.float32))


def _guided(I, p, r=8, eps=1e-3):
    """Edge-aware snap of matte p onto guide image I (gray, [0,1]). Fixes small flow drift."""
    m = (r, r)
    mI = cv2.boxFilter(I, -1, m); mp = cv2.boxFilter(p, -1, m)
    a = (cv2.boxFilter(I * p, -1, m) - mI * mp) / (cv2.boxFilter(I * I, -1, m) - mI * mI + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, m) * I + cv2.boxFilter(b, -1, m)


def _matte_refine(rgb, a, kernel_scale=1.0):
    """Closed-form matting inside the subject bbox (crisp hair, kept fast).

    kernel_scale multiplies the edge-band dilation / padding: video frames share a
    roughly fixed resolution (fixed pixel kernel is fine), but a single image can
    range from a few hundred px to several thousand, so callers with widely
    varying resolutions should scale this by bbox size (see bg_remove_image.py)."""
    from pymatting import estimate_alpha_cf
    ys, xs = np.where(a > 0.05)
    if len(xs) < 50:
        return a
    pad = max(1, int(round(8 * kernel_scale)))
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad)
    ac = a[y0:y1, x0:x1]
    img = rgb[y0:y1, x0:x1]
    ch, cw = ac.shape
    cap = 512
    sc = min(1.0, cap / max(ch, cw))
    if sc < 1.0:
        sw, sh = max(1, int(cw * sc)), max(1, int(ch * sc))
        acs = cv2.resize(ac, (sw, sh)); imgs = cv2.resize(img, (sw, sh))
    else:
        acs, imgs = ac, img
    k = max(1, int(round(7 * kernel_scale)))
    if k % 2 == 0:
        k += 1
    band = cv2.dilate((np.abs(acs - 0.5) < 0.45).astype(np.uint8), np.ones((k, k), np.uint8))
    tri = np.where(acs > 0.95, 1.0, np.where(acs < 0.05, 0.0, 0.5)).astype(np.float64)
    tri[band > 0] = 0.5
    try:
        am = estimate_alpha_cf(imgs.astype(np.float64) / 255.0, tri)
        if sc < 1.0:
            am = cv2.resize(am.astype(np.float32), (cw, ch))
        a = a.copy()
        a[y0:y1, x0:x1] = am
    except Exception as e:
        # Silent on purpose in the sense that a single frame's refine
        # failing shouldn't abort a whole clip -- but silent forever means a
        # systematic failure (e.g. a pymatting regression) could degrade
        # every frame with no signal at all. Warn once per process instead.
        warnings.warn(f"_matte_refine: closed-form refine failed, using "
                       f"un-refined alpha for this frame ({e})", stacklevel=2)
    return a


def despill(rgb, a, kernel_scale=1.0, *, band_only=False, est_scale=1.0):
    """Recover the true foreground color at semi-transparent edges (hair, fur,
    motion blur) via multi-level foreground estimation (Germer et al. 2020,
    pymatting's estimate_foreground_ml), instead of compositing the raw
    photographed pixel -- which still carries a (1-alpha) contribution from
    the ORIGINAL background color, visible as a color fringe once
    recomposited onto a NEW background.

    Runs inside the subject's bbox+padding (same crop strategy as
    _matte_refine): estimate_foreground_ml is local (small-neighbourhood
    pyramid solve), so cropping doesn't change its result at pixels away
    from the crop edge. NOTE this docstring used to claim the WRITE-BACK
    was also limited to where alpha isn't fully opaque -- it wasn't; the
    whole crop (including the fully-opaque interior) was overwritten every
    time. `band_only=True` (measured as the V6 speed/quality experiment,
    see the plan) makes the docstring's original claim actually true: only
    pixels in the genuinely semi-transparent band (0.02 < alpha < 0.98,
    where spill can exist at all) get the estimated colour, which also
    lets an empty band skip the ML solve entirely. `est_scale<1` runs the
    solve on a downscaled crop and upsamples the result -- a separate,
    independent lever (can combine with band_only or not)."""
    from pymatting import estimate_foreground_ml
    ys, xs = np.where(a > 0.02)
    if len(xs) < 50:
        return rgb
    pad = max(1, int(round(8 * kernel_scale)))
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad)
    ac = a[y0:y1, x0:x1]
    band = (ac > 0.02) & (ac < 0.98) if band_only else None
    if band is not None and not band.any():
        return rgb  # nothing in the uncertain band -- despill has no work to do here
    ac64 = ac.astype(np.float64)
    img = rgb[y0:y1, x0:x1].astype(np.float64) / 255.0
    ch, cw = ac64.shape
    if est_scale < 1.0:
        sh, sw = max(8, round(ch * est_scale)), max(8, round(cw * est_scale))
        ac_solve = cv2.resize(ac64.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float64)
        img_solve = cv2.resize(img.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float64)
    else:
        ac_solve, img_solve = ac64, img
    try:
        fg = estimate_foreground_ml(img_solve, ac_solve)
    except Exception as e:
        warnings.warn(f"despill: foreground estimation failed, keeping "
                       f"un-despilled colour for this frame ({e})", stacklevel=2)
        return rgb
    if est_scale < 1.0:
        fg = cv2.resize(fg.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    fg8 = np.clip(fg * 255.0, 0, 255).astype(np.uint8)
    out = rgb.copy()
    crop = out[y0:y1, x0:x1]
    if band is not None:
        crop[band] = fg8[band]
    else:
        crop[...] = fg8
    return out


def _salient_box(alpha, pad_frac=0.12, thr=0.5):
    """Padded, image-clamped xyxy bbox of the alpha>thr region, or None if empty.

    Used when the detector finds no box (e.g. a class outside COCO-80, or a
    stylised/illustrated subject): the bbox of a coarse full-frame matte anchors
    a SAM2 box-prompt so selection stays class-agnostic."""
    ys, xs = np.where(alpha > thr)
    if xs.size == 0:
        return None
    H, W = alpha.shape[:2]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pw = int(round((x1 - x0 + 1) * pad_frac))
    ph = int(round((y1 - y0 + 1) * pad_frac))
    x0 = max(0, x0 - pw); y0 = max(0, y0 - ph)
    x1 = min(W - 1, x1 + pw); y1 = min(H - 1, y1 + ph)
    return np.array([x0, y0, x1, y1], np.float32)


def _alpha_touches_a_crop_edge(out, x0, y0, x1, y1, W, H, thr=0.5):
    """Whether the crop-based alpha `out` (zero outside [y0:y1, x0:x1], see
    Models._birefnet_crop_pass) has any confidently-opaque pixel right on an
    edge of the crop rect that ISN'T also the frame boundary -- the
    signature of birefnet()'s crop having cut off real subject content
    (a straight-line amputation), as opposed to the subject genuinely
    exiting the frame there (expected and harmless)."""
    if y0 > 0 and np.any(out[y0, x0:x1] > thr):
        return True
    if y1 < H and np.any(out[y1 - 1, x0:x1] > thr):
        return True
    if x0 > 0 and np.any(out[y0:y1, x0] > thr):
        return True
    if x1 < W and np.any(out[y0:y1, x1 - 1] > thr):
        return True
    return False


def _normalize_alpha(a, h: int, w: int) -> np.ndarray:
    """Coerce any backend's alpha to a 2-D float32 array of shape (h, w), values
    in [0, 1], no NaN/Inf. Handles the common extra-dim shapes a model might
    emit: (1,1,h,w), (1,h,w), (h,w,1)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4 and a.shape[:2] == (1, 1):
        a = a[0, 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    elif a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim != 2:
        raise ValueError(f"matte must be 2-D after squeeze, got shape {a.shape}")
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(a, 0.0, 1.0)


# ---------- ONNX YOLOX detector ----------

class _OnnxYOLO:
    """onnxruntime-only YOLOX detector (no torch) for the subject/person box.

    YOLOX-S (Apache-2.0; Megvii COCO weights) exported to FP32 ONNX with the grid
    decode baked in (decode_in_inference) -> output [1, 8400, 85] = 4 box
    (cxcywh, 640-letterbox px) + 1 objectness + 80 COCO class scores. NOTE: BGR
    input, NO /255 normalisation, pad 114 (differs from YOLOv5/v8 preprocessing)."""

    def __init__(self, onnx_path, so, device):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=_onnx_providers(device))
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = 640

    def detect(self, rgb, conf_thr=0.25, iou_thr=0.45):
        """Return (xyxy[N,4] in ORIGINAL coords, cls[N] int, conf[N])."""
        h, w = rgb.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(rgb[..., ::-1], (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: x})[0][0]
        cls_scores = out[:, 5:]
        cls = cls_scores.argmax(1)
        conf = out[:, 4] * cls_scores.max(1)
        keep = conf > conf_thr
        empty = (np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32))
        if not keep.any():
            return empty
        b = out[keep, :4]; cls = cls[keep]; conf = conf[keep]
        xyxy = np.empty_like(b)
        xyxy[:, 0] = (b[:, 0] - b[:, 2] / 2) / r
        xyxy[:, 1] = (b[:, 1] - b[:, 3] / 2) / r
        xyxy[:, 2] = (b[:, 0] + b[:, 2] / 2) / r
        xyxy[:, 3] = (b[:, 1] + b[:, 3] / 2) / r
        idxs = cv2.dnn.NMSBoxes([[float(x0), float(y0), float(x2 - x0), float(y2 - y0)]
                                 for x0, y0, x2, y2 in xyxy], conf.tolist(), conf_thr, iou_thr)
        if len(idxs) == 0:
            return empty
        idxs = np.array(idxs).reshape(-1)
        return xyxy[idxs], cls[idxs], conf[idxs]


# ---------- SAM2 subprocess worker ----------
#
# SAM2 (torch/CUDA) is run in its own subprocess rather than the main process,
# because running it alongside onnxruntime's CUDA session in ONE process was
# confirmed by testing to make BOTH collapse 10-30x once they alternate GPU
# calls (see _onnx_providers' note). Isolating them in separate OS processes
# (each with its own CUDA context) restores both to near their standalone
# speed. Only imported/used when Models.use_sam2 is true.

def _sam2_worker_main(conn, ckpt: str, cfg: str, device: str) -> None:
    """Subprocess entry point: loads SAM2 once, then serves (rgb, box) requests
    over `conn` until it receives None. Runs in a spawned child process (see
    _Sam2Worker.start), so this needs its own torch/CUDA init — nothing here
    executes in the parent."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=device))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        rgb, box = msg
        predictor.set_image(rgb)
        with torch.inference_mode():
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
        m = masks[0] if masks.ndim == 3 else masks
        conn.send(np.ascontiguousarray(m, dtype=np.float32))
    conn.close()


class _Sam2Worker:
    """Lazily starts the SAM2 subprocess on first use and reuses it for the
    life of this Models instance. `stop()` is called from Models.__del__."""

    def __init__(self, device: str):
        self._device = device
        self._proc = None
        self._conn = None

    def _ensure_started(self):
        if self._proc is None:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe()
            self._proc = ctx.Process(
                target=_sam2_worker_main,
                args=(child_conn, str(SAM2_CKPT), SAM2_CFG, self._device),
                daemon=True,
            )
            self._proc.start()
            self._conn = parent_conn
        return self._conn

    def mask(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        conn = self._ensure_started()
        conn.send((rgb, box))
        # SAM2 calls measured at well under 20s even in the worst case
        # (see README's "GPU internals"); a much longer wait means the
        # subprocess likely crashed/hung (e.g. CUDA OOM) rather than that
        # this particular call is just slow, so fail fast instead of
        # blocking the whole run indefinitely.
        if not conn.poll(60):
            raise TimeoutError(
                "SAM2 subprocess did not respond within 60s -- it may have "
                "crashed or hung (e.g. CUDA out of memory). Not retried "
                "automatically."
            )
        return conn.recv()

    def stop(self):
        if self._proc is not None:
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None
            self._conn = None


# ---------- unified model bundle ----------

class Models:
    """Loads YOLOX + BiRefNet (+ SAM2 on GPU). Pass device="cpu"/"cuda"/"auto" (default).

    SAM2 (a ViT/Transformer region-selection model) is GPU-oriented: on CPU it
    runs at roughly 1-2 orders of magnitude below its GPU speed (measured: ~100s
    per frame on this machine), turning a single image into a 1-2 minute job and
    a video into a multi-hour one. So the SAM2 region gate / _armback are only
    used when device == "cuda" (use_sam2). On CPU, matting is BiRefNet (+ YOLOX
    box for crop super-resolution) + _topology_clean only — no region gate, so
    clutter next to the subject isn't removed, but it stays fast."""

    def __init__(self, device: str = "auto"):
        self.device = resolve_device(device)
        # Incremented by birefnet() each time a crop-based pass looked
        # truncated (subject alpha touching a crop edge that wasn't just
        # the frame boundary) and it had to fall back to a full-frame
        # re-inference -- see birefnet()'s own docstring. Read by
        # runner.infer_clip to report timings["box_reinfer_frames"].
        self.box_reinfer_frames = 0
        self.use_sam2 = self.device == "cuda"
        self._sam2_worker = _Sam2Worker(self.device) if self.use_sam2 else None
        if self.use_sam2:
            _ensure_nvidia_dlls_on_path()
        import onnxruntime as ort
        # YOLOX now follows self.device like every other model here. It used
        # to be hardcoded to "cpu" on the theory that it was "tiny (640x640,
        # ~0.3s on CPU)" -- that figure was measured on an old Windows dev
        # box; on this machine (a GPU host with 24 logical CPUs) it actually
        # ran at ~4.5s/frame on CPU, which for a while was misread as BiRefNet
        # being slow (both were folded into one timer in runner.py) and made
        # this the dominant cost of the whole pipeline. On CUDA it's ~0.02s.
        so_det = ort.SessionOptions()
        so_det.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Only matters for the CPU-fallback path (device=="cpu"): confirmed by
        # testing that ORT's own default thread count is faster than pinning
        # every logical CPU (24 threads was 2.1x SLOWER than 4 here -- the
        # thread pool thrashes past a point), so cap rather than maximize.
        so_det.intra_op_num_threads = max(1, min(4, os.cpu_count() or 4))
        self.det = _OnnxYOLO(YOLOX_ONNX, so_det, self.device)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4))
        if self.use_sam2:
            # See _onnx_providers: pairs with arena_extend_strategy=kSameAsRequested
            # to avoid the GPU memory-arena growth pattern that made repeated
            # BiRefNet calls collapse from ~1s to ~20-30s (confirmed by testing).
            so.enable_mem_pattern = False
        self.brf = ort.InferenceSession(str(BRF), sess_options=so, providers=_onnx_providers(self.device))
        self._brf_in = self.brf.get_inputs()[0].name  # export-agnostic (lite/full BiRefNet)
        if self.device == "cuda":
            # _onnx_providers always lists CUDAExecutionProvider first with a
            # CPU fallback, so a driver/library mismatch (e.g. missing
            # libcudnn.so.9, confirmed reproducible by testing with
            # LD_LIBRARY_PATH stripped) fails SILENTLY into ~50s/frame on
            # BiRefNet with no error anywhere -- warn loudly instead, since
            # "cuda" was explicitly requested.
            for label, sess in (("YOLOX", self.det.sess), ("BiRefNet", self.brf)):
                if "CUDAExecutionProvider" not in sess.get_providers():
                    warnings.warn(
                        f"device='cuda' was requested but {label}'s onnxruntime "
                        f"session is actually running on {sess.get_providers()!r} "
                        f"-- inference will be far slower than expected. Check "
                        f"LD_LIBRARY_PATH / cuDNN install (see run.sh/serve.sh).",
                        RuntimeWarning, stacklevel=2)

    def __del__(self):
        if getattr(self, "_sam2_worker", None) is not None:
            self._sam2_worker.stop()

    def sam2_mask(self, rgb, box):
        """SAM2 box-prompted subject mask, computed in the SAM2 subprocess (see
        _Sam2Worker). Returns alpha (H,W) float32 {0,1}. box=None -> all-ones
        (keep everything), so gate call-sites never blank a frame."""
        H, W = rgb.shape[:2]
        if box is None:
            return np.ones((H, W), np.float32)
        bx = box.astype(np.float32) if hasattr(box, "astype") else np.array(box, np.float32)
        m = self._sam2_worker.mask(rgb, bx)
        return _normalize_alpha(m, H, W)

    def _brf_1024(self, img1024):
        x = ((img1024.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        o = self.brf.run(None, {self._brf_in: x})[0][0, 0]
        return 1.0 / (1.0 + np.exp(-o)) if (o.min() < 0 or o.max() > 1) else o

    def _birefnet_crop_pass(self, rgb, box, margin):
        """One BiRefNet forward pass over `rgb`, cropped+letterboxed to `box`
        (padded by `margin`) or the full frame if `box` is None. Returns
        (out, x0, y0, x1, y1): `out` is (H,W) float32, zero outside the crop
        rect; x0/y0/x1/y1 are that rect in `rgb`'s own pixel coordinates
        (clamped to the frame). Split out of birefnet() so a truncation-
        recovery fallback pass can reuse the exact same crop/letterbox/paste
        logic instead of duplicating it."""
        H, W = rgb.shape[:2]
        if box is not None:
            bx = box.astype(int) if hasattr(box, "astype") else np.array(box, int)
            mw, mh = int((bx[2] - bx[0]) * margin), int((bx[3] - bx[1]) * margin)
            x0, y0 = max(0, bx[0] - mw), max(0, bx[1] - mh)
            x1, y1 = min(W, bx[2] + mw), min(H, bx[3] + mh)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        crop = rgb[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        s = 1024.0 / max(ch, cw)
        nw, nh = max(1, int(cw * s)), max(1, int(ch * s))
        lb = np.zeros((1024, 1024, 3), np.uint8)
        lb[:nh, :nw] = cv2.resize(crop, (nw, nh))
        ac = cv2.resize(self._brf_1024(lb)[:nh, :nw], (cw, ch))
        out = np.zeros((H, W), np.float32)
        out[y0:y1, x0:x1] = ac
        return out, x0, y0, x1, y1

    def birefnet(self, rgb, box=None, margin=0.12, auto_full_frame_fallback: bool = True):
        """High-res alpha. With a subject box, CROP+letterbox so the subject fills
        the fixed 1024^2 input (effective super-resolution) and out-of-box clutter
        is physically excluded. ONNX input is locked at 1024^2.

        Alpha outside the crop rect is hard zero (see _birefnet_crop_pass) --
        if `box` under-covers the real subject (found in production: a
        confident-but-partial YOLOX detection, e.g. a "sports ball"/"person"
        box that captures the torso but not the head), the crop amputates
        whatever falls outside it along a dead-straight edge. min_box_frac
        (subject_box) only rejects a box that's too SMALL in area -- a box
        that's normal-sized but positioned/shaped wrong passes it untouched.
        When the crop-based alpha (`ac > 0.5`) touches an edge of the crop
        rect that ISN'T also the frame boundary (touching the true frame
        edge is normal -- the subject legitimately leaves frame there), that
        edge is where the crop cut the subject off.

        auto_full_frame_fallback (default True): on a detected truncation,
        immediately retry as one full-frame (box=None) pass -- bounded to at
        most 2x the cost of a single call, and not recursive (a full-frame
        pass has no crop edge of its own to be truncated by, only the frame
        boundary itself). This is the right, simplest fix for a caller with
        no wider context to fall back on (e.g. keyframe_alpha / the
        still-image pipeline -- one frame, one decision). tool/pipeline/
        runner.py's clip inference pass sets this False instead: a single
        truncated frame there is better repaired by growing the box over
        its whole contiguous region and re-running that region (see
        tool.pipeline.boxmode.repair_truncated_regions) than by jumping
        straight to an un-cropped full frame for just that one frame, which
        was a visible S1 (chatter) spike -- one frame at a very different
        effective resolution from its neighbours, then snapping back.
        `self._last_truncated` is always set (read by callers that disable
        the auto fallback and need to know whether one happened); `self.
        box_reinfer_frames` is only incremented here when this method
        itself performs the fallback (auto_full_frame_fallback=True) --
        the clip-level repair path counts its own re-inferred frames
        instead, so the two mechanisms don't double-count."""
        H, W = rgb.shape[:2]
        out, x0, y0, x1, y1 = self._birefnet_crop_pass(rgb, box, margin)
        truncated = box is not None and _alpha_touches_a_crop_edge(out, x0, y0, x1, y1, W, H)
        self._last_truncated = truncated
        if truncated and auto_full_frame_fallback:
            self.box_reinfer_frames += 1
            out, x0, y0, x1, y1 = self._birefnet_crop_pass(rgb, None, margin)
        return _normalize_alpha(out, H, W)

    def subject_box(self, rgb, prefer_person: bool = True, min_box_frac: float = 0.0):
        """Largest detected box. If prefer_person, break ties toward a 'person'
        detection (used by the video router, which dispatches on person/non-person);
        the image pipeline passes prefer_person=False for unbiased class-agnostic
        selection. Returns (xyxy, is_person).

        min_box_frac: reject a chosen box below this fraction of the frame area,
        falling back to (None, False) as if nothing had been detected (the
        caller's existing "no box" path already handles that: BiRefNet runs on
        the full frame instead of a crop). Off by default (0.0) to keep every
        existing caller's behaviour unchanged.

        Exists because YOLOX can misclassify a small part of the real subject
        (e.g. a raised fist) as an unrelated COCO class at just-above-threshold
        confidence, and with nothing else detected that becomes "the subject
        box" -- BiRefNet then crops+letterboxes to that fragment and the rest
        of the subject falls outside the crop, collapsing the alpha area for
        possibly many consecutive frames (too long for a single-frame collapse
        repair to catch). Confidence alone can't separate this: legitimate
        detections score as low as misfires do. Measured across several video
        clips: every misfire topped out at 7.93% of the frame area, every
        legitimate full-subject detection started at 25.63% -- min_box_frac=0.15
        sits cleanly in that gap. Was previously patched in per-clip
        video-pipeline scripts only (never reached this shared method, or
        bg_remove_image.py, or the SAM2 route), so every other caller carried
        the same collapse risk silently.

        Exempt (第11計画 Part 3-2): a confident PERSON box (class 0, conf >=
        _CONFIDENT_PERSON_CONF) is never rejected for size. The misfires this
        gate exists for were a fragment of the subject labelled as some
        unrelated COCO class; a whole person seen from further away is not
        that, and rejecting its box sent BiRefNet onto the FULL frame, where
        it happily foregrounds every salient prop (the widepose clip's frames 34-47:
        person box 12.6-14.9% of the frame at conf 0.90-0.93 -> the dumbbell,
        bottle and laptop on the floor became opaque)."""
        xyxy, cls, conf = self.det.detect(rgb)
        if len(xyxy) == 0:
            return None, False
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        persons = [i for i in range(len(cls)) if int(cls[i]) == _YOLO_PERSON]
        if prefer_person and persons:
            idx = max(persons, key=lambda i: areas[i])
        else:
            idx = int(np.argmax(areas))
        is_person = int(cls[idx]) == _YOLO_PERSON
        confident_person = is_person and float(conf[idx]) >= _CONFIDENT_PERSON_CONF
        if min_box_frac > 0.0 and not confident_person:
            h, w = rgb.shape[:2]
            if areas[idx] / (w * h) < min_box_frac:
                return None, False
        return xyxy[idx], is_person


# ---------- one high-quality alpha for a single frame ----------

def keyframe_alpha(rgb, M, gate=None, do_matte=True, matte_kernel_scale=1.0, temporal=None,
                    min_box_frac: float = 0.15):
    """Compute one high-quality alpha: detect -> BiRefNet matte -> [SAM2 region
    gate, GPU only] -> hole-fill -> optional CF matting refine.

    temporal=None: static per-frame processing (the image pipeline, or any video
    route that doesn't need frame-to-frame state).

    temporal={"prev_alpha", "prev_gray", "dis", "gx", "gy", "is_static", "move_tau"}:
    on GPU (M.use_sam2), adds _armback (restore SAM-dropped moving limbs, static
    camera only) and _topology_temporal (don't fill background newly enclosed by
    a moving limb) — the video person-path's per-frame BiRefNet route. On CPU,
    only _topology_temporal runs (no SAM2 gate, so nothing for _armback to
    restore). Returns (alpha, gray) so the caller can carry state into the next
    frame.

    min_box_frac: forwarded to subject_box() -- 0.15 (PipelineConfig's own
    validated default) rather than subject_box's own 0.0, since this function
    had no caller anywhere in the repo carrying that gate (audit finding:
    config.py's B2 note claimed "every OTHER caller" of subject_box had it,
    which was true of tool/pipeline/runner.py but not this one -- this
    function currently has zero call sites, so the change is behaviour-
    neutral for anything that exists today, but a future caller inherits the
    same tiny-misdetected-box collapse protection runner.py already has).
    """
    box, _ = M.subject_box(rgb, prefer_person=temporal is not None, min_box_frac=min_box_frac)
    a_raw = M.birefnet(rgb, box=box)
    if M.use_sam2:
        smask = M.sam2_mask(rgb, box)
        smask = cv2.dilate(smask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        smask = cv2.GaussianBlur(smask, (21, 21), 0)
        a = a_raw * smask
    else:
        smask = None
        a = a_raw

    if temporal is None:
        if gate is not None:
            a = a * gate
        if smask is not None:
            a = np.maximum(a, binary_fill_holes(a > 0.5).astype(np.float32) * (smask > 0.3))
        else:
            a = _topology_clean(a)
        if do_matte:
            a = _matte_refine(rgb, a, kernel_scale=matte_kernel_scale)
        return np.clip(a, 0, 1)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    prev_gray = temporal.get("prev_gray")
    fl = temporal["dis"].calc(gray, prev_gray, None) if prev_gray is not None else None
    if smask is not None and temporal.get("is_static") and fl is not None:
        a = _armback(a, a_raw, smask, fl, move_tau=temporal.get("move_tau", 1.5))
    if gate is not None:
        a = a * gate
    a = _topology_temporal(a, temporal.get("prev_alpha"), gray, prev_gray,
                           temporal["dis"], temporal["gx"], temporal["gy"], flow=fl)
    if do_matte:
        a = _matte_refine(rgb, a, kernel_scale=matte_kernel_scale)
    return np.clip(a, 0, 1), gray

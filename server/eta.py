"""Per-stage timing predictions and remaining-time estimates for the GUI's
progress bar ("残り 約4分 · 経過 2:13") -- see the plan's "待ち時間の可視化"
section. Two responsibilities, kept in one small module since they always
travel together (a `ClipEta` is built FROM a `TimingStats.predict()` call):

- `TimingStats`: a tiny persistent table of "seconds per (frame x megapixel)"
  (or, for the fixed-input-size networks, "seconds per frame") coefficients, one set per pipeline stage x (device/encoder) group, updated
  with an EMA from every clip that finishes. This is what lets a clip that
  hasn't even started yet (still "pending") show a real ETA -- the probe
  (width/height/frames) is known at upload time, before any GPU work runs.
- `ClipEta`: turns one clip's live progress ticks into a smoothed "seconds
  remaining" value, blending the prediction with the observed rate once a
  stage is far enough along to trust it.

Neither class touches threads, SSE, or job.json directly -- that wiring
lives in jobs.py's `_process_job`/`_progress`, which is what makes this
module unit-testable without a GpuWorker.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("heroextractor")

#: Stage order matches jobs.py's STAGES -- kept as a separate tuple (not
#: imported) since jobs.py imports server-level state and importing this
#: module from jobs.py, not the reverse, keeps the dependency one-way.
STAGES = ("prepare", "infer", "postprocess", "encode", "qc")

#: Seed coefficients: seconds / (frames x megapixels), from this repo's own
#: measured profile (121 frames x 1656x1248 = 2.068 MPix -> 250 frame*MPix)
#: on the shared GPU -- see the plan's "速度・追加レバー計画" and the
#: "第6計画" measurements: detect ~15s, birefnet ~36s, postprocess
#: (mc_median) ~137s, encode(supersampled_gif@4) >=420s. These are only
#: ever the FALLBACK for a (group, stage) pair with no observed history
#: yet -- see TimingStats.predict().
#: despill: 0.55 -> ~0.25 on 2026-09-22 (第9計画/第8回監査レバー3: adopted
#: despill_band_only=True + despill_est_scale=0.5 as the new default --
#: see tool/pipeline/config.py's docstring). Re-measured directly from
#: this change: Triceratops.mp4 (2.068 MPix x 121f) 60.3s -> 0.241
#: s/frame*MPix; 4 flat-chroma clips (0.59 MPix x ~124f each) 17.7-18.8s
#: -> 0.242-0.257 -- consistent across very different resolution/content.
#:
#: detect_frame / birefnet_frame (第11計画 Part 1-4, 2026-09-23): seconds
#: per FRAME, not per frame*MPix. Both networks run at a fixed input size
#: (YOLOX letterboxes to 640, BiRefNet always sees a 1024^2 crop), so their
#: cost does not scale with the clip's resolution -- the old per-MPix
#: "detect"/"birefnet" coefficients, EMA-learned on 0.5-0.6 MPix clips,
#: over-predicted a 2.07 MPix clip by the resolution ratio (dinosaur GPU:
#: predicted 417.7s vs actual 332s). New stage names so the inflated
#: entries already persisted in timing_stats.json under the old names are
#: simply never read again. Seeds: cuda measured on Triceratops.mp4
#: 2026-09-23 (birefnet_s 37.8s / detect_s 14.0s over 121 frames, on a
#: shared GPU); cpu measured on this machine under its usual load (BiRefNet
#: ~30-70s/frame -> 35, YOLOX with 4 threads ~0.75s/frame) -- the old cpu
#: seeds were 8-17x too low.
SEED_COEF = {
    "cuda": {"detect_frame": 0.12, "birefnet_frame": 0.31, "despill": 0.25},
    "cpu": {"detect_frame": 0.75, "birefnet_frame": 35.0, "despill": 0.25},
    "postprocess:cpu": {"postprocess": 0.55},
    "qc:cpu": {"qc": 0.15},
    "encode:supersampled_gif@4": {"encode": 1.9},
    "encode:supersampled_gif@3": {"encode": 1.1},
    "encode:supersampled_gif@2": {"encode": 0.5},
    "encode:ss_alpha_gif@4": {"encode": 0.25},
    "encode:mov": {"encode": 0.1},
    "encode:webp": {"encode": 0.1},
    # tool/pipeline/keyer.py's colour-only fast path -- one stage covers
    # everything upstream of encode (no separate detect/birefnet/despill on
    # this route, see runner.py's keyer branch). Seeded from the measured
    # ①感謝.mp4 run (124 frames, 768x768 = 0.59 MPix -> 73.1 frame*MPix,
    # key_s=7.4s -> ~0.10 s/frame*MPix). A keyed clip also always encodes
    # via ss_alpha_gif (run_clip's own swap) and never runs postprocess
    # (mc_median forced off), which predict()/update() encode directly
    # rather than via a seed here -- see their own `keyed` branches.
    "key": {"key": 0.10},
}

#: EMA smoothing factor for TimingStats.update() -- how much weight a
#: single new observation gets against the running coefficient.
_STATS_ALPHA = 0.3

#: Stages whose coefficient is seconds per FRAME (bucket field "s_per_frame")
#: rather than seconds per frame*MPix (bucket field "s_per_frame_mpix") --
#: see SEED_COEF's detect_frame/birefnet_frame note.
PER_FRAME_STAGES = frozenset({"detect_frame", "birefnet_frame"})


def _field(stage: str) -> str:
    return "s_per_frame" if stage in PER_FRAME_STAGES else "s_per_frame_mpix"

#: Default probe assumptions when a clip has no probe yet (shouldn't happen
#: in practice -- validate_upload() always runs before a clip is queued --
#: but predict() must never raise just because a caller passes None).
_DEFAULT_FRAMES = 120
_DEFAULT_MPIX = 2.0


def _infer_group(device: str) -> str:
    return device if device in ("cuda", "cpu") else "cuda"


def _encode_group(encoder: str, supersample: int) -> str:
    if encoder == "supersampled_gif":
        return f"encode:supersampled_gif@{supersample}"
    if encoder == "ss_alpha_gif":
        return "encode:ss_alpha_gif@4"
    return f"encode:{encoder}"


def _mpix(probe: dict | None) -> float:
    if not probe or not probe.get("width") or not probe.get("height"):
        return _DEFAULT_MPIX
    return (probe["width"] * probe["height"]) / 1e6


def _frames(probe: dict | None) -> int:
    if not probe or not probe.get("frames"):
        return _DEFAULT_FRAMES
    return probe["frames"]


class TimingStats:
    """Persistent per-stage timing coefficients (seconds / frame*MPix),
    updated by an EMA as clips finish. `path=None` keeps everything in
    memory only (used by tests and any deployment that doesn't want a
    stats file) -- `load()`/`save()` become no-ops in that case."""

    def __init__(self, path: Path | None):
        self._path = path
        self._lock = threading.Lock()
        self._coef: dict[str, dict[str, dict[str, float]]] = {}
        self.load()

    def load(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
            coef = data["coef"]
            # Minimal shape check -- a corrupt or foreign-schema file must
            # never crash the worker; it just means "start from seeds".
            if not isinstance(coef, dict):
                raise ValueError("coef is not a dict")
            with self._lock:
                self._coef = coef
        except Exception:
            logger.warning("timing_stats.json unreadable/corrupt -- starting from seed coefficients", exc_info=True)
            with self._lock:
                self._coef = {}

    def _save_locked(self) -> None:
        if self._path is None:
            return
        tmp = self._path.with_suffix(".json.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"version": 1, "coef": self._coef}, indent=2))
            tmp.replace(self._path)
        except OSError:
            logger.warning("failed to persist timing_stats.json", exc_info=True)

    def _coef_for(self, group: str, stage: str) -> float:
        with self._lock:
            observed = self._coef.get(group, {}).get(stage)
        if observed is not None and _field(stage) in observed:
            return observed[_field(stage)]
        return SEED_COEF.get(group, {}).get(stage, 0.0)

    def predict(self, probe: dict | None, config, qc: bool, *, keyed: bool = False) -> dict[str, float]:
        """Predicted seconds per stage for a clip with this probe+config,
        BEFORE it has run at all -- this is what lets a still-pending
        clip contribute to job_eta_s. Returns the four STAGES keys plus
        "total".

        `keyed`: whether this clip is predicted to take tool/pipeline/
        keyer.py's colour-only fast path (server/jobs.py's
        _clip_takes_keyer, itself derived from the upload-time probe).
        Without this, a flat-chroma clip's ETA used to add detect+birefnet+
        despill+postprocess+the heavy two-pass encoder -- none of which the
        keyer path actually runs -- showing "残り 約20分" for a clip that
        finishes in well under a minute (audit M2). When keyed, this skips
        straight to the "key" stage coefficient, forces postprocess to 0
        (mc_median is forced off on this route -- see run_clip), and prices
        the encode stage as ss_alpha_gif (run_clip's own automatic swap for
        a keyed clip), regardless of what `config.encoder` itself says."""
        frames = _frames(probe)
        mpix = _mpix(probe)
        fm = frames * mpix

        if keyed:
            infer_s = self._coef_for("key", "key") * fm
            post_s = 0.0
            encode_group = _encode_group("ss_alpha_gif", config.supersample)
        else:
            infer_group = _infer_group(config.device)
            infer_s = (self._coef_for(infer_group, "detect_frame")
                       + self._coef_for(infer_group, "birefnet_frame")) * frames
            if config.apply_despill:
                infer_s += self._coef_for(infer_group, "despill") * fm
            post_s = self._coef_for("postprocess:cpu", "postprocess") * fm if config.mc_median_half > 0 else 0.0
            encode_group = _encode_group(config.encoder, config.supersample)

        encode_s = self._coef_for(encode_group, "encode") * fm
        qc_s = self._coef_for("qc:cpu", "qc") * fm if qc else 0.0

        pred = {"infer": infer_s, "postprocess": post_s, "encode": encode_s, "qc": qc_s}
        pred["total"] = sum(pred.values())
        return pred

    def update(self, config, probe: dict | None, timings: dict, *, ok: bool, keyed: bool = False) -> None:
        """EMA-update coefficients from one clip's completed `timings`
        dict (key_s | detect_s/birefnet_s/despill_s/postprocess_s/encode_s/
        qc_s). `ok=False` (clip failed/cancelled) still ingests infer-side
        timings (they're only written once that stage genuinely finished --
        see run_clip's own docstring) but skips encode_s, since run_clip
        writes a partial encode_s even on a timeout/cancel mid-encode,
        which would corrupt the coefficient with an incomplete duration.

        `keyed`: pass the GROUND-TRUTH outcome (whether "key_s" is actually
        in `timings`), not a prediction -- unlike predict()'s `keyed`, this
        one has the real result available and should use it. Audit M1: a
        keyed clip's encode_s (~19s, ss_alpha_gif) was previously EMA'd
        straight into the "encode:supersampled_gif@N" bucket (true cost
        ~600s), and its postprocess_s (~0s, mc_median forced off) into
        "postprocess:cpu" -- both persisted to timing_stats.json, so they
        kept dragging the NEURAL route's own coefficients down every time a
        flat-chroma clip ran, with no self-correction. A keyed clip's timing
        now goes into its own "key" bucket (see predict()) instead of
        anything the neural route reads, and its postprocess_s (always ~0
        by construction on this route -- predict() hardcodes post_s=0 for a
        keyed clip rather than looking up a coefficient) isn't recorded at
        all, since there's nothing to learn from a value that's always 0."""
        frames = _frames(probe)
        mpix = _mpix(probe)
        fm = frames * mpix
        if fm <= 0:
            return

        samples: list[tuple[str, str, float]] = []
        if keyed:
            if "key_s" in timings:
                samples.append(("key", "key", timings["key_s"] / fm))
            encode_group = _encode_group("ss_alpha_gif", config.supersample)
        else:
            infer_group = _infer_group(config.device)
            for key, (group, stage) in {
                "detect_s": (infer_group, "detect_frame"),
                "birefnet_s": (infer_group, "birefnet_frame"),
                "despill_s": (infer_group, "despill"),
                "postprocess_s": ("postprocess:cpu", "postprocess"),
            }.items():
                if key in timings:
                    denom = frames if stage in PER_FRAME_STAGES else fm
                    samples.append((group, stage, timings[key] / denom))
            encode_group = _encode_group(config.encoder, config.supersample)

        if "qc_s" in timings:
            samples.append(("qc:cpu", "qc", timings["qc_s"] / fm))
        if ok and "encode_s" in timings:
            samples.append((encode_group, "encode", timings["encode_s"] / fm))

        if not samples:
            return
        with self._lock:
            for group, stage, observed in samples:
                field = _field(stage)
                bucket = self._coef.setdefault(group, {}).setdefault(
                    stage, {field: SEED_COEF.get(group, {}).get(stage, observed), "n": 0})
                bucket[field] = (1 - _STATS_ALPHA) * bucket[field] + _STATS_ALPHA * observed
                bucket["n"] += 1
            self._save_locked()


class ClipEta:
    """Turns one clip's live progress ticks into a smoothed "seconds
    remaining" estimate. One instance per running clip; not shared/reused
    across clips (each has its own `prediction`, its own stage timers)."""

    #: Encode is the one stage with no ticks for its second half (ffmpeg's
    #: two-pass palettegen/paletteuse only emits after every frame is
    #: written -- see the plan's "-progress pipe: 却下" note). This splits
    #: the encode prediction into a ticked "write" share and a tick-less
    #: "tail" share so the tail still has a real time estimate instead of
    #: sitting at "prediction minus elapsed" forever.
    _ENCODE_WRITE_SHARE = 0.35

    def __init__(self, prediction: dict[str, float], *, now=time.monotonic):
        self._pred = prediction
        self._now = now
        self._clip_start = now()
        self._stage_start: dict[str, float] = {}
        self._last_stage: str | None = None
        self._last_tick: tuple[int, int] | None = None
        self._disp_remaining: float | None = None

    def elapsed(self) -> float:
        return self._now() - self._clip_start

    def on_tick(self, stage: str, done: int, total: int) -> None:
        if stage != self._last_stage:
            self._stage_start[stage] = self._now()
            self._last_stage = stage
        self._last_tick = (done, total)

    def _stage_elapsed(self, stage: str) -> float:
        started = self._stage_start.get(stage)
        return 0.0 if started is None else self._now() - started

    def _remaining_for_stage(self, stage: str) -> float:
        """Remaining seconds for `stage`, called ONLY for the currently
        active stage (`remaining()` uses the full prediction for stages
        that haven't started and 0 for stages already finished)."""
        pred = self._pred.get(stage, 0.0)
        elapsed = self._stage_elapsed(stage)
        if self._last_tick is None:
            return max(0.0, pred - elapsed)
        done, total = self._last_tick
        frac = (done / total) if total else 0.0

        if stage != "encode":
            if frac < 0.10:
                return max(0.0, pred - elapsed)
            observed_remaining = max(0.0, elapsed / frac - elapsed)
            predicted_remaining = max(0.0, pred - elapsed)
            w = min(1.0, (frac - 0.10) / 0.40)
            return w * observed_remaining + (1 - w) * predicted_remaining

        # Encode: ffmpeg's two-pass palettegen/paletteuse only emits after
        # every frame is written, so there are no ticks for its second
        # half -- split the prediction into a ticked "write" share and a
        # tick-less "tail" share (see the plan's "-progress pipe: 却下").
        write_pred = pred * self._ENCODE_WRITE_SHARE
        tail_pred = pred * (1 - self._ENCODE_WRITE_SHARE)
        if frac >= 1.0:
            tail_elapsed = max(0.0, elapsed - write_pred)
            return max(0.0, tail_pred - tail_elapsed)
        if frac < 0.10:
            return max(0.0, write_pred - elapsed) + tail_pred
        observed_remaining_write = max(0.0, elapsed / frac - elapsed)
        predicted_remaining_write = max(0.0, write_pred - elapsed)
        w = min(1.0, (frac - 0.10) / 0.40)
        remaining_write = w * observed_remaining_write + (1 - w) * predicted_remaining_write
        return remaining_write + tail_pred

    def remaining(self) -> float:
        """Smoothed seconds remaining for the WHOLE clip (all stages from
        here on), never negative. Applies two smoothing rules so the
        display doesn't visibly jump: (1) an EMA against the previous
        displayed value, (2) an upward jump is capped at +50% of the
        previous displayed value per call (a downward jump is never
        capped -- the clip finishing early should be reflected immediately)."""
        idx = STAGES.index(self._last_stage) if self._last_stage in STAGES else -1
        raw = 0.0
        for i, stage in enumerate(STAGES):
            if stage not in self._pred or i < idx:
                continue  # already finished (or not part of this clip's plan)
            raw += self._remaining_for_stage(stage) if i == idx else self._pred[stage]
        raw = max(0.0, raw)

        if self._disp_remaining is None:
            self._disp_remaining = raw
        else:
            capped = min(raw, self._disp_remaining * 1.5) if raw > self._disp_remaining else raw
            self._disp_remaining = 0.7 * self._disp_remaining + 0.3 * capped
        return max(0.0, self._disp_remaining)

    def finish(self) -> None:
        self._disp_remaining = 0.0

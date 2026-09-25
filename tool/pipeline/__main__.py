"""CLI entry point: python -m tool.pipeline <clip.mp4> [more.mp4 ...] <outdir>

Replaces the previous /tmp-scratch-script-only workflow (every render in
this session's history was a one-off script under /tmp) with a real,
reusable command. Loads Models once and reuses it across all clips.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .cache import infer_clip_cached
from .config import PipelineConfig
from .runner import run_clip, _load_models, encode_worker


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m tool.pipeline",
        description="Chroma-background video pipeline (flat-colour-backdrop matting).")
    ap.add_argument("clips", nargs="+", help="one or more source video paths")
    ap.add_argument("outdir", help="output directory (created if missing)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    # default=None (not the string "supersampled_gif") so main() can tell
    # "user passed --encoder supersampled_gif explicitly" apart from
    # "--encoder was never given" -- PipelineConfig.encoder_explicit (see its
    # docstring) needs that distinction to know whether the keyer path's
    # automatic ss_alpha_gif swap is allowed to touch this run.
    ap.add_argument("--encoder", default=None,
                     choices=["supersampled_gif", "ss_alpha_gif", "mov", "webp"],
                     help="ss_alpha_gif is the validated default (adopted 2026-09-22, "
                          "第9計画/第8回監査レバー4 -- L1's original 'measured worse' finding "
                          "was later traced to BiRefNet's own run-to-run nondeterminism, not "
                          "the encoder; the decisive re-test that shares one cached inference "
                          "across variants found no fidelity cost). supersampled_gif (the old "
                          "default, ~15x slower) remains available for comparison.")
    ap.add_argument("--use-keyer", default="auto", choices=["auto", "on", "off"],
                     help="colour-only fast path for flat-chroma backdrops (default auto: "
                          "used when the clip measures as flat chroma AND the subject's "
                          "palette stays clear of the key colour). 'on' still refuses an "
                          "unsafe clip -- that check is a correctness gate, not a preference")
    ap.add_argument("--use-trimap", action="store_true",
                     help="Phase 4 trimap compositing -- EXPERIMENTAL, default off "
                          "(measured S1/E1 regression not yet resolved, see the plan)")
    ap.add_argument("--no-despill", action="store_true",
                     help="disable pymatting spill removal (diagnostic/V4 ablation flag -- "
                          "see tool/qc/metrics.py's E2_fringe_quality, added specifically to "
                          "measure this stage's effect after V0 profiling found despill "
                          "costs MORE wall time per clip than BiRefNet inference itself).")
    ap.add_argument("--strip-bg-fringe", action="store_true",
                     help="narrow backdrop-fringe cleanup -- default OFF (opt-in only). "
                          "Verified safe on the dinosaur/yellow-backdrop asset; measured "
                          "UNSAFE on the purplebg real-footage asset (erased dark clothing). "
                          "Self-disables with a warning if this clip's own colours make it "
                          "unsafe (see stages.fringe_strip_is_safe), but starts opt-in so a "
                          "new asset's default run never depends on that check catching it.")
    ap.add_argument("--keep-main-subject", action="store_true",
                     help="drop opaque connected components that aren't the main subject "
                          "(a salient prop BiRefNet foregrounds alongside the person, e.g. "
                          "the widepose clip's dumbbell/bottle/laptop) -- default OFF (opt-in). "
                          "On GPU, a component fully inside the person's own YOLOX box is "
                          "still tested against a SAM2 person mask (第12計画) rather than "
                          "kept outright, so a prop like a bottle sitting inside the widepose "
                          "box can still be dropped; CPU and the keyer route fall back to "
                          "the plain box-rectangle rule. See config.py's docstring for the "
                          "full rule and its measured limits (a prop held in the hand still "
                          "survives -- it's one connected component with the person).")
    ap.add_argument("--min-box-frac", type=float, default=0.15)
    ap.add_argument("--scale", type=float, default=1.0,
                     help="downscale the source frame before processing (default 1.0 = "
                          "native resolution). BiRefNet always resizes its crop to a fixed "
                          "1024^2 for inference, so a full-frame box gets ~the same matting "
                          "quality regardless of this value; what scales quadratically with "
                          "resolution is the supersampled GIF encoder. A 1440x1440 clip at "
                          "1.0 blew the encoder's 600s timeout -- 0.6-0.7 is a reasonable "
                          "starting point for source video much larger than the delivery "
                          "size actually needs.")
    ap.add_argument("--max-side", type=int, default=0,
                     help="cap the delivered GIF's long edge at this many px (0 = native "
                          "resolution, the default). --max-side 1000 was TESTED and "
                          "REJECTED (see the plan's V1 entry): cv2.INTER_AREA downscaling "
                          "erased fine detail (tooth gaps on 挨拶.mp4) that BiRefNet then "
                          "couldn't recover, leaving background colour visibly un-removed "
                          "in that gap -- absent at native resolution, same frame. Opt in "
                          "explicitly and re-run the same F2/visual check before trusting "
                          "any value here for a new asset.")
    ap.add_argument("--qc", action="store_true",
                     help="after rendering, grade each clip against its source with "
                          "tool.qc.metrics and write a fidelity/stability table plus a "
                          "worst-frame contact sheet PNG next to the output -- makes the "
                          "'run the harness before delivering' step (this session's own "
                          "hard-learned lesson: a past over-erasure regression was only caught by "
                          "running the harness, not by the render finishing without error) "
                          "part of the tool itself instead of a separate manual step.")
    ap.add_argument("--parallel-encode", type=int, default=1, metavar="N",
                     help="(V5) after all clips finish GPU inference (still sequential -- "
                          "one GPU, no benefit to overlapping that part), run the "
                          "postprocess+encode half of each clip in N parallel worker "
                          "processes instead of one at a time. Encoding is CPU/ffmpeg-bound "
                          "(~80%% of a clip's wall time, see the plan's V0 profiling) and "
                          "each clip's encode is independent, so this is a straight wall-clock "
                          "win for a multi-clip batch. Default 1 = current sequential "
                          "behaviour, unchanged.")
    args = ap.parse_args(argv)

    encoder_explicit = args.encoder is not None
    config = PipelineConfig(device=args.device, encoder=(args.encoder or "ss_alpha_gif"),
                             encoder_explicit=encoder_explicit,
                             use_trimap=args.use_trimap, min_box_frac=args.min_box_frac,
                             strip_bg_fringe=args.strip_bg_fringe, scale=args.scale,
                             max_side=(args.max_side or None),
                             apply_despill=not args.no_despill,
                             use_keyer=args.use_keyer,
                             keep_main_subject=args.keep_main_subject)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ext = {"supersampled_gif": ".gif", "ss_alpha_gif": ".gif",
           "mov": ".mov", "webp": ".webp"}[config.encoder]
    clip_paths = []
    for clip_path in args.clips:
        clip_path = Path(clip_path)
        if not clip_path.exists():
            print(f"skip (not found): {clip_path}", file=sys.stderr)
            continue
        clip_paths.append(clip_path)

    if args.parallel_encode > 1:
        failed = _run_parallel(clip_paths, outdir, ext, config, args.parallel_encode, args.qc)
    else:
        failed = _run_sequential(clip_paths, outdir, ext, config, args.qc)

    if failed:
        print(f"\n{len(failed)} clip(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _run_sequential(clip_paths, outdir, ext, config, run_qc):
    # Loaded on first actual use rather than up front: a batch of flat-chroma
    # clips takes the colour-only keyer path and never needs the network at
    # all, so eagerly loading it here would pin ~700MB of VRAM (and require a
    # working CUDA stack) for a run that never touches the GPU. Still loaded
    # exactly once and shared across clips when anything does need it.
    _models = {}

    def models_factory():
        if "m" not in _models:
            _models["m"] = _load_models(config)
        return _models["m"]

    models = models_factory
    failed = []
    for clip_path in clip_paths:
        out_path = outdir / (clip_path.stem + "_matte" + ext)
        try:
            run_clip(clip_path, out_path, config, models=models)
        except Exception as e:
            # One clip's encoder timing out (etc.) shouldn't silently drop
            # every clip after it in the batch -- confirmed this actually
            # happened (one client-footage batch: clip 1 OK, then a 600s ffmpeg
            # timeout on clips 2-4's un-run turns an unrelated per-clip failure
            # into "the whole delivery batch didn't finish").
            print(f"  FAILED: {clip_path} ({e}) -- continuing with remaining clips",
                  file=sys.stderr)
            failed.append(str(clip_path))
            continue
        if run_qc:
            _run_qc(clip_path, out_path)
    return failed


def _run_parallel(clip_paths, outdir, ext, config, workers, run_qc):
    """(V5) GPU inference stays sequential in THIS process (one GPU -- no
    benefit to overlapping that part, and Models isn't fork-safe to share
    across processes anyway); only postprocess+encode (CPU/ffmpeg-bound,
    independent per clip) runs in a worker pool. Each clip's raw inference
    result is round-tripped through cache.py's disk cache rather than
    passed through the pool directly -- a large numpy array pickled
    through multiprocessing IPC costs roughly what writing/reading it from
    disk does anyway, and this way a worker can be retried/reasoned about
    without holding every clip's ~1GB raw result in the parent's memory at
    once."""
    # Lazy, same as _run_sequential's models_factory (audit M8: this used to
    # call _load_models(config) unconditionally, so a batch made entirely of
    # flat-chroma clips -- which never reach infer_clip's model-loading branch
    # at all -- still paid to load BiRefNet/YOLOX onto the GPU for a run that
    # never used them). infer_clip accepts a zero-arg callable and only
    # invokes it the first time a clip actually needs the network.
    _models = {}

    def models_factory():
        if "m" not in _models:
            _models["m"] = _load_models(config)
        return _models["m"]

    failed = []
    jobs = []
    for clip_path in clip_paths:
        try:
            infer_clip_cached(clip_path, config, models=models_factory, clip_name=clip_path.stem)
        except Exception as e:
            print(f"  FAILED (inference): {clip_path} ({e}) -- continuing with remaining clips",
                  file=sys.stderr)
            failed.append(str(clip_path))
            continue
        jobs.append(clip_path)
    _models.pop("m", None)  # free GPU/session state before forking workers

    out_paths = {}
    # spawn, not fork (the default on Linux): onnxruntime's CUDA provider
    # leaves a CUDA context initialised at the process level that survives
    # `del models` -- forking a process holding one into workers that never
    # touch the GPU is a known hazard (undefined behaviour up to a hang).
    # spawn starts each worker as a fresh interpreter with no inherited
    # CUDA state, at the one-time cost of re-importing this module per
    # worker (cheap next to a single clip's encode time).
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {}
        for clip_path in jobs:
            out_path = outdir / (clip_path.stem + "_matte" + ext)
            out_paths[clip_path] = out_path
            futures[pool.submit(encode_worker, str(clip_path), str(out_path), config)] = clip_path
        for fut in as_completed(futures):
            clip_path = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  FAILED (encode): {clip_path} ({e}) -- continuing with remaining clips",
                      file=sys.stderr)
                failed.append(str(clip_path))

    if run_qc:
        for clip_path in jobs:
            if str(clip_path) not in failed:
                _run_qc(clip_path, out_paths[clip_path])
    return failed


def _run_qc(source_path: Path, out_path: Path):
    """Grade one rendered clip and write <out_path>.qc_contact_sheet.png
    (the F1i and S1 worst frames, magenta-composited so transparency is
    obvious) and <stem>_matte.qc.json (the raw metrics dict) next to it.
    Metrics-only, no pipeline internals -- same tool.qc.metrics every
    harness run in this plan used. Returns the metrics dict so a caller
    (the server's job store) can attach it to a job record without
    re-parsing the JSON file it also writes."""
    import json

    import cv2
    import numpy as np

    from ..qc import metrics as qc

    print(f"  [qc] grading {out_path.name} ...", flush=True)
    result = qc.evaluate(source_path.stem, str(source_path), str(out_path))
    for name in ["F1_false_erase", "F1i_false_erase_interior", "F2_false_keep",
                 "F3_interior_holes", "S1_mc_chatter", "S2_color_flicker",
                 "S3_frozen_px", "S4_area_jump", "E1_perimeter_ratio"]:
        r = result[name]
        value = r.get("total", r.get("median"))
        print(f"    {name:<28} {value}", flush=True)

    out_frames, H, W = qc.read_rgba_frames(str(out_path))
    worst_indices = sorted({result["F1i_false_erase_interior"]["worst_frame"],
                            result["S1_mc_chatter"]["worst_frame"]})
    tiles = []
    for i in worst_indices:
        f = out_frames[i]
        rgb = f[:, :, :3][:, :, ::-1]
        a = f[:, :, 3].astype(np.float32) / 255.0
        bg = np.full_like(rgb, (255, 0, 255))
        comp = (rgb.astype(np.float32) * a[:, :, None] +
                bg.astype(np.float32) * (1 - a[:, :, None])).astype(np.uint8)
        cv2.putText(comp, f"frame {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(comp)
    sheet = np.hstack(tiles) if len(tiles) > 1 else tiles[0]
    sheet_path = out_path.with_suffix(out_path.suffix + ".qc_contact_sheet.png")
    cv2.imwrite(str(sheet_path), sheet)
    print(f"  [qc] wrote {sheet_path}", flush=True)

    metrics_path = out_path.with_suffix(".qc.json")
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  [qc] wrote {metrics_path}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

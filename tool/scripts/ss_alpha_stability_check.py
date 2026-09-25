"""Lever 4 validation (see the plan's 第7/8計画 speed-lever table): does
BiRefNet's run-to-run alpha non-determinism survive `temporal_alpha_smooth`
(mc_median_half=1, the default)?

Background: V2/V3 (2026-09-02/03) rejected lighter GIF encoders (ss=2/3,
ss_alpha_gif) on the neural route after they showed a visible F2 defect
(background colour left in fine gaps) -- later traced NOT to the encoder
itself but to BiRefNet producing slightly different alpha on independent
inference runs, with the difference straddling the 1-bit threshold in
exactly the fine-detail spots ss=4's two-pass palette chain happens to blur
over (see DECISIONS.md's V4 entry). That confound was measured on RAW
per-frame alpha, straight out of BiRefNet, with no temporal smoothing
applied. It has never been measured AFTER mc_median_half=1 -- the pipeline's
actual default -- which might already suppress it (mc_median exists
precisely to damp frame-to-frame BiRefNet variation).

This script: run infer_clip (device=cuda, apply_despill=False -- despill
never touches alpha, so it's a needless cost here) TWICE, independently
(NOT via cache.infer_clip_cached -- the whole point is two genuinely
separate inference passes), apply postprocess()'s temporal_alpha_smooth to
each run's alpha independently, then count how many pixels cross the
alpha_threshold=128 line between the two runs' smoothed alpha. Near-zero
means the non-determinism is suppressed by smoothing and ss_alpha_gif's
lack of a second quantization pass is no longer disqualifying; a real count
means the V2/V3 rejection still applies post-smoothing and lever 4 stays
unadopted.

Usage: venv/bin/python -m tool.scripts.ss_alpha_stability_check [--clip Triceratops]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tool.pipeline.config import PipelineConfig
from tool.pipeline.runner import _load_models, infer_clip
from tool.pipeline.stages import postprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT.parent / "sample" / "dinosaur"


def _smoothed_alpha(src, config, models):
    raw = infer_clip(src, config, models=models)
    frames = [f.copy() for f in raw["frames"]]
    postprocess(frames, raw["clear_masks"], raw["src_gray"], raw["motion_dmax"], raw["bg_ref"], config, boxes=raw.get("boxes"))
    return np.stack([f[:, :, 3] for f in frames], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="Triceratops")
    args = ap.parse_args()

    src = SRC_DIR / f"{args.clip}.mp4"
    config = PipelineConfig(device="cuda", apply_despill=False)  # despill never touches alpha
    models = _load_models(config)

    print(f"run 1: {src.name}", flush=True)
    t0 = time.perf_counter()
    a1 = _smoothed_alpha(src, config, models)
    print(f"  {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"run 2: {src.name}", flush=True)
    t0 = time.perf_counter()
    a2 = _smoothed_alpha(src, config, models)
    print(f"  {time.perf_counter() - t0:.1f}s", flush=True)

    thr = config.alpha_threshold
    bin1 = a1 >= thr
    bin2 = a2 >= thr
    flipped = bin1 != bin2
    total_px = flipped.size
    flipped_n = int(flipped.sum())
    per_frame = flipped.reshape(flipped.shape[0], -1).sum(axis=1)
    worst_frame = int(np.argmax(per_frame))

    print(f"\n=== {args.clip}: alpha threshold (128) crossings between two independent runs, "
          f"AFTER temporal_alpha_smooth ===")
    print(f"  total pixels flipped: {flipped_n} / {total_px} ({flipped_n / total_px * 100:.4f}%)")
    print(f"  worst frame: {worst_frame} ({int(per_frame[worst_frame])} px)")
    print(f"  max raw alpha delta: {np.abs(a1.astype(int) - a2.astype(int)).max()}")
    verdict = "near-zero -- smoothing suppresses it, lever 4 not disqualified on this evidence" \
        if flipped_n / total_px < 0.0001 else \
        "non-trivial -- non-determinism survives smoothing, lever 4 stays unadopted"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()

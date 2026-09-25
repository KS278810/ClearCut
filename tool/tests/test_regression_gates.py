"""Permanent regression gates for the chroma-background pipeline
(tool/pipeline/), grading results_dinosaur/ (the shipped deliverable)
against the known-bug thresholds established in the 18-bg-remove plan.

These exist because "the metrics were all green" is exactly what let
run_gpu_birefnet_v23.py ship with 130,340px of the subject's own body
erased -- every number here traces to a REAL, previously-shipped defect,
not an arbitrary tolerance. A gate failing here means a regression toward
one of those specific defects, not generic quality drift. The bounds are
generic (not tuned to any one clip's own numbers), so they remain valid
sanity checks for whatever clip(s) CLIPS below names.

The original four clips (思考/喜び/感謝/挨拶) were deleted from sample/
dinosaur/ in a 2026-09-13 reorganisation (no longer needed) and replaced
with a single clip, Triceratops.mp4 -- results_dinosaur/ was regenerated
against it at the same time (see tool/docs/DECISIONS.md).

Requires ffmpeg on PATH and the real results_dinosaur/*.gif + source
sample/dinosaur/*.mp4 files -- skipped (not failed) if either is missing,
so this doesn't block on machines without the dataset checked out.

2026-09-25: the flat-chroma second fixture set that used to restore the
F1i gate's coverage here (a client-adjacent batch, kept out of this
repo's published tree) was removed -- see tool/docs/DECISIONS.md's
"Fixture / dataset changes". F1i/F2 (chroma-distance metrics) are
therefore skipped again for now, same as before that set existed: the
Triceratops.mp4 fixture is natural-background (bg_is_chroma_class=False),
so those metrics are undefined for it. Re-add a flat-chroma fixture set
in tool/qc/fixtures.py and the chroma_* tests below to restore that
coverage.
"""
from pathlib import Path

import pytest

from tool.qc import metrics as qc

ROOT = Path(__file__).resolve().parents[2]  # .../theory
SRC_DIR = ROOT.parent / "sample" / "dinosaur"
OUT_DIR = ROOT / "results_dinosaur"
CLIPS = ["Triceratops"]

pytestmark = pytest.mark.skipif(
    not (SRC_DIR.is_dir() and OUT_DIR.is_dir()),
    reason="sample/dinosaur/ and/or results_dinosaur/ not present on this machine",
)


def _graded(src_dir, out_dir, clip):
    src = src_dir / f"{clip}.mp4"
    out = out_dir / f"{clip}_matte.gif"
    if not (src.exists() and out.exists()):
        pytest.skip(f"{clip}: missing {src} or {out}")
    return clip, qc.evaluate(clip, str(src), str(out))


@pytest.fixture(scope="module", params=CLIPS)
def result(request):
    return _graded(SRC_DIR, OUT_DIR, request.param)


def test_f1i_false_erase_interior_bounded(result):
    """The v23 bug this guards: _strip_tail_shadow erased 130,340px of the
    dinosaur's own white body on 感謝.mp4 -- F1i was 88,100 even AFTER that
    fix (v24). 200,000 keeps real headroom while still catching a return
    of that bug class.

    F1i/F2 are chroma-distance metrics -- meaningless (qc.evaluate reports
    None) for a clip whose backdrop doesn't gate as flat chroma, which is
    exactly what sample/dinosaur/Triceratops.mp4 is (see tool/tests/
    test_probe.py's test_real_sample_clip_validates) after the 2026-09-13
    reorg replaced the original four flat-chroma clips. Skipped rather
    than asserted for such a clip -- there is currently no flat-chroma
    clip left in this fixture set for this particular gate to check."""
    clip, r = result
    if r["F1i_false_erase_interior"] is None:
        pytest.skip(f"{clip}: not chroma-class (bg_is_chroma_class=False) -- F1i is undefined for this clip")
    assert r["F1i_false_erase_interior"]["total"] < 200_000, (
        f"{clip}: F1i={r['F1i_false_erase_interior']['total']} -- "
        f"subject body is being erased (the v23 _strip_tail_shadow bug class)")


def test_s3_frozen_px_bounded(result):
    """The v23 bug this guards: an RGB colour-freeze flattened 41,290px of
    real shading into a static "stuck" patch on 感謝.mp4. Current shipped
    worst case is 405px (no colour-freeze in this pipeline by design)."""
    clip, r = result
    assert r["S3_frozen_px"]["total"] < 2_000, (
        f"{clip}: S3={r['S3_frozen_px']['total']} -- a region is being "
        f"frozen to a constant colour (the v23 colour-freeze bug class)")


def test_s2_color_flicker_bounded(result):
    """The Phase 2 regression this guards: switching off colour-freeze (to
    fix S3 above) let GIF-palette flicker back in -- v24 measured 196.9
    px/frame on 感謝.mp4. Current shipped worst case (感謝) is ~85-115
    px/frame with the per-frame pipeline; 150 keeps headroom while still
    catching a return to v24-era levels."""
    clip, r = result
    assert r["S2_color_flicker"]["mean_per_frame"] < 150, (
        f"{clip}: S2={r['S2_color_flicker']['mean_per_frame']:.1f} px/frame "
        f"-- GIF-palette colour flicker regression")

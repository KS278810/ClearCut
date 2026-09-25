"""Named fixture sets for tool.qc.harness.

Each set maps a namespaced clip key -> source MP4 path.

- "chroma": the dinosaur mascot clip(s) (flat yellow background, 3D render)
  -- the class the trimap work in Phase 4 targets. Source video lives in the
  sibling sample/ directory (01_背景除去/sample/dinosaur/); the accepted-final
  GIF (results_dinosaur/) stays in this theory/ tree since it's this repo's
  own QC baseline, not raw source material. Reduced to a single clip
  (Triceratops.mp4) after a 2026-09-13 reorg deleted the original four
  (思考/喜び/感謝/挨拶) as no longer needed -- see tool/tests/
  test_regression_gates.py's own note.

2026-09-25: this repo's other fixture sets (a second flat-chroma mascot
batch, and two client-footage regression sets) were removed to keep only
this project's own dinosaur material in the published tree -- see
tool/docs/DECISIONS.md's "Fixture / dataset changes" for what coverage
that dropped and how to re-add a set here if a same-class replacement
material (this project's own, not a client deliverable) becomes available.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../01_背景除去/theory
SAMPLE_ROOT = ROOT.parent / "sample"  # .../01_背景除去/sample

_CHROMA_SRC_DIR = SAMPLE_ROOT / "dinosaur"

CHROMA = {"chroma_Triceratops": _CHROMA_SRC_DIR / "Triceratops.mp4"}

_SETS = {
    "chroma": CHROMA,
    "all": {**CHROMA},
}


def get_sources(set_name):
    """Return {clip_key: source_mp4_path} for a named set."""
    if set_name not in _SETS:
        raise KeyError(f"unknown fixture set {set_name!r}; choices: {sorted(_SETS)}")
    return dict(_SETS[set_name])


def clip_short_name(clip_key):
    """Strip the project prefix, e.g. 'chroma_Triceratops' -> 'Triceratops'
    (for building output filenames that follow the '{name}_matte.gif'
    convention)."""
    return clip_key.split("_", 1)[1]

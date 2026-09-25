"""Regression test for the plan's G4 finding: F1/F1i/F2/E2 all measure
distance-to-the-estimated-backdrop-colour, which is meaningless (and
actively misleading -- it LOOKS like a real number) on a clip whose
backdrop isn't actually a flat colour. evaluate() must report None for
those four instead of a number computed against a nonsense "backdrop"."""
import cv2
import numpy as np
import pytest

from tool.qc.metrics import BgStats, evaluate

CHROMA_METRIC_KEYS = ("F1_false_erase", "F1i_false_erase_interior", "F2_false_keep", "E2_fringe_quality")
COLOUR_FREE_METRIC_KEYS = ("F3_interior_holes", "S1_mc_chatter", "S2_color_flicker",
                           "S3_frozen_px", "S4_area_jump", "E1_perimeter_ratio")


def _write_tiny_clip(tmp_path, w=16, h=12, n=3):
    src_path = tmp_path / "src.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(src_path), fourcc, 10.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n):
        writer.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    writer.release()

    # A tiny real GIF with alpha (2x2, one fully opaque and one fully
    # transparent pixel column) via ffmpeg, so read_rgba_frames (which
    # shells out to ffprobe/ffmpeg) has something real to decode.
    out_path = tmp_path / "out.gif"
    raw_path = tmp_path / "out.rgba"
    frame = np.zeros((h, w, 4), dtype=np.uint8)
    frame[:, : w // 2, :3] = 200
    frame[:, : w // 2, 3] = 255
    with open(raw_path, "wb") as f:
        for _ in range(n):
            f.write(frame.tobytes())
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
         "-framerate", "10", "-i", str(raw_path), "-loop", "0", str(out_path)],
        check=True, capture_output=True,
    )
    return src_path, out_path


@pytest.mark.skipif(not cv2.VideoWriter_fourcc(*"mp4v"), reason="mp4v fourcc unavailable")
def test_chroma_only_metrics_are_none_when_backdrop_is_not_flat(tmp_path):
    src_path, out_path = _write_tiny_clip(tmp_path)
    non_chroma_bg = BgStats(mu_cb=100.0, mu_cr=100.0, sigma_cb=40.0, sigma_cr=40.0,
                             is_chroma_class=False, frac_bg_like=0.3)
    result = evaluate("synthetic", str(src_path), str(out_path), bg=non_chroma_bg)
    assert result["bg_is_chroma_class"] is False
    for key in CHROMA_METRIC_KEYS:
        assert result[key] is None, f"{key} should be None for a non-chroma-class backdrop"
    for key in COLOUR_FREE_METRIC_KEYS:
        assert result[key] is not None, f"{key} doesn't depend on backdrop colour and should still be computed"


@pytest.mark.skipif(not cv2.VideoWriter_fourcc(*"mp4v"), reason="mp4v fourcc unavailable")
def test_chroma_metrics_are_computed_when_backdrop_is_flat(tmp_path):
    src_path, out_path = _write_tiny_clip(tmp_path)
    chroma_bg = BgStats(mu_cb=128.0, mu_cr=128.0, sigma_cb=1.0, sigma_cr=1.0,
                        is_chroma_class=True, frac_bg_like=1.0)
    result = evaluate("synthetic", str(src_path), str(out_path), bg=chroma_bg)
    assert result["bg_is_chroma_class"] is True
    for key in CHROMA_METRIC_KEYS:
        assert result[key] is not None, f"{key} should be computed for a chroma-class backdrop"

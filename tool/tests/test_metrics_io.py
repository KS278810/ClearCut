"""Regression test for read_rgba_frames' WebP decoding (see the 第9計画's
A0 item): this machine's ffmpeg (6.1.1) cannot decode animated WebP at all
-- it skips the ANIM/ANMF chunks and raises "image data not found" --
which used to make read_rgba_frames throw RuntimeError("no frames
decoded") for every .webp input, silently turning a successful render
into a "failed" QC job (server/jobs.py's _run_qc runs right after a
successful run_clip)."""
import numpy as np
import pytest
from PIL import Image

from tool.qc.metrics import read_rgba_frames, soft_alpha_band_frac, soft_alpha_sad


def _write_animated_webp(path, n=4, w=6, h=5):
    frames = []
    for i in range(n):
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[:, :, 0] = i * 20  # distinct per-frame red level
        arr[:, :, 3] = 255
        frames.append(Image.fromarray(arr, "RGBA"))
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=50, lossless=True)


def test_read_rgba_frames_decodes_all_webp_frames(tmp_path):
    path = tmp_path / "out.webp"
    _write_animated_webp(path, n=4, w=6, h=5)

    frames, h, w = read_rgba_frames(path)

    assert frames.shape == (4, 5, 6, 4)
    assert h == 5 and w == 6


def test_read_rgba_frames_webp_frame_order_and_values_preserved(tmp_path):
    path = tmp_path / "out.webp"
    _write_animated_webp(path, n=3, w=4, h=4)

    frames, _, _ = read_rgba_frames(path)

    # lossless=True above so the per-frame red level should round-trip exactly.
    for i in range(3):
        assert frames[i, 0, 0, 0] == i * 20
        assert frames[i, 0, 0, 3] == 255


def test_read_rgba_frames_webp_single_frame(tmp_path):
    path = tmp_path / "out.webp"
    _write_animated_webp(path, n=1, w=4, h=4)

    frames, h, w = read_rgba_frames(path)

    assert frames.shape[0] == 1


def _square_silhouette(h=20, w=20, soft_edge=False):
    """A centred square: alpha=255 well inside, 0 well outside. With
    soft_edge, the boundary ring gets a linear ramp (simulating BiRefNet's
    genuinely fractional alpha); without it, alpha snaps to 0/255 right at
    the same boundary (simulating a 1-bit GIF)."""
    yy, xx = np.mgrid[0:h, 0:w]
    # Signed distance (in px) from the square's edge at [5,15)x[5,15).
    dist = np.minimum(np.minimum(xx - 5, 14 - xx), np.minimum(yy - 5, 14 - yy))
    if soft_edge:
        alpha = np.clip((dist + 2) * 51, 0, 255).astype(np.uint8)  # ramps over ~5px
    else:
        alpha = np.where(dist >= 0, 255, 0).astype(np.uint8)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, 3] = alpha
    return out


def test_soft_alpha_band_frac_is_zero_for_binary_alpha():
    frame = _square_silhouette(soft_edge=False)
    out_rgba = np.stack([frame, frame], 0)

    result = soft_alpha_band_frac(out_rgba)

    assert result["mean"] == 0.0


def test_soft_alpha_band_frac_is_positive_for_soft_edge():
    frame = _square_silhouette(soft_edge=True)
    out_rgba = np.stack([frame, frame], 0)

    result = soft_alpha_band_frac(out_rgba)

    assert result["mean"] > 0.0


def test_soft_alpha_sad_is_zero_when_output_matches_raw():
    frame = _square_silhouette(soft_edge=True)
    out_rgba = np.stack([frame, frame], 0)
    raw_alpha = out_rgba[:, :, :, 3].copy()

    result = soft_alpha_sad(out_rgba, raw_alpha)

    assert result["mean"] == 0.0


def test_soft_alpha_sad_detects_binarisation_loss():
    soft_frame = _square_silhouette(soft_edge=True)
    hard_frame = _square_silhouette(soft_edge=False)
    out_rgba = np.stack([hard_frame, hard_frame], 0)  # encoder collapsed to 0/255
    raw_alpha = np.stack([soft_frame[:, :, 3], soft_frame[:, :, 3]], 0)  # BiRefNet's true soft alpha

    result = soft_alpha_sad(out_rgba, raw_alpha)

    assert result["mean"] > 0.0


def test_soft_alpha_sad_rejects_shape_mismatch():
    frame = _square_silhouette(soft_edge=True)
    out_rgba = np.stack([frame, frame], 0)
    raw_alpha = np.zeros((3, 10, 10), dtype=np.uint8)  # wrong shape

    with pytest.raises(ValueError):
        soft_alpha_sad(out_rgba, raw_alpha)


def test_soft_alpha_sad_accepts_float_0_1_raw_alpha():
    frame = _square_silhouette(soft_edge=True)
    out_rgba = np.stack([frame, frame], 0)
    raw_alpha = (out_rgba[:, :, :, 3].astype(np.float64) / 255.0)

    result = soft_alpha_sad(out_rgba, raw_alpha)

    assert result["mean"] == pytest.approx(0.0, abs=1.0)

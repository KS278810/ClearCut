"""Coverage for encode_webp's lossless/quality kwargs (第9計画 A2 WebP-variant
measurement). No test file existed for tool/ffmpeg_encoders.py before this."""
import numpy as np
import pytest
from PIL import Image

from tool.ffmpeg_encoders import encode_webp


def _synthetic_frames(n=3, w=8, h=8):
    out = []
    for i in range(n):
        f = np.zeros((h, w, 4), dtype=np.uint8)
        f[:, :, 0] = i * 40  # distinct per-frame red level
        f[:, :, 3] = 200
        out.append(f)
    return out


def _chunks(frames):
    for f in frames:
        yield f.tobytes()


def test_encode_webp_default_kwargs_still_work(tmp_path):
    """Backward compatibility: no lossless/quality passed -- matches the
    only production call site's usage (tool/pipeline/runner.py)."""
    frames = _synthetic_frames()
    out = tmp_path / "out.webp"

    encode_webp(_chunks(frames), 8, 8, 10.0, out)

    assert out.is_file() and out.stat().st_size > 0
    im = Image.open(out)
    assert im.n_frames == 3


def test_encode_webp_lossless_round_trips_exactly(tmp_path):
    frames = _synthetic_frames()
    out = tmp_path / "out.webp"

    encode_webp(_chunks(frames), 8, 8, 10.0, out, lossless=True)

    im = Image.open(out)
    from PIL import ImageSequence
    decoded = [np.array(f.convert("RGBA")) for f in ImageSequence.Iterator(im)]
    assert len(decoded) == 3
    for i, frame in enumerate(decoded):
        assert (frame[:, :, 0] == i * 40).all()
        assert (frame[:, :, 3] == 200).all()


def test_encode_webp_rejects_unsupported_exact_flag_is_not_passed(tmp_path):
    """Regression guard: this ffmpeg build's libwebp_anim encoder has no
    -exact option (confirmed via `ffmpeg -h encoder=libwebp_anim` on
    ffmpeg 6.1.1) -- encode_webp must never construct a command line that
    includes it, or every call fails with "Unrecognized option 'exact'"."""
    frames = _synthetic_frames(n=1)
    out = tmp_path / "out.webp"

    # Passing an unexpected kwarg should be a TypeError (no such param),
    # not a silent no-op or a passthrough to ffmpeg.
    with pytest.raises(TypeError):
        encode_webp(_chunks(frames), 8, 8, 10.0, out, exact=True)

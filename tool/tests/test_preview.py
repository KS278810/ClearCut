"""tool/pipeline/preview.py's encode_preview_png -- pure numpy/cv2, no
real clip or model needed."""
import cv2
import numpy as np

from tool.pipeline.preview import encode_preview_png


def _decode_bgra(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def test_encode_preview_png_round_trips_with_alpha_and_correct_channel_order():
    """Regression guard: cv2's PNG encoder wants BGRA, but every frame in
    this pipeline is stored RGBA -- skipping the swap flips red and blue
    in the browser."""
    frame = np.zeros((40, 60, 4), dtype=np.uint8)
    frame[:, :, 0] = 200   # R
    frame[:, :, 1] = 10    # G
    frame[:, :, 2] = 5     # B
    frame[:, :, 3] = 128   # A

    png = encode_preview_png(frame, long_edge=360)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes, not junk

    decoded_bgra = _decode_bgra(png)  # cv2 always decodes back to BGRA order
    assert decoded_bgra.shape == (40, 60, 4)
    b, g, r, a = (decoded_bgra[0, 0, i] for i in range(4))
    assert (int(r), int(g), int(b), int(a)) == (200, 10, 5, 128)


def test_encode_preview_png_downscales_when_long_edge_exceeds_the_cap():
    frame = np.full((200, 800, 4), 255, dtype=np.uint8)
    png = encode_preview_png(frame, long_edge=360)
    decoded = _decode_bgra(png)
    assert max(decoded.shape[:2]) <= 360


def test_encode_preview_png_does_not_upscale_a_smaller_frame():
    frame = np.full((50, 80, 4), 255, dtype=np.uint8)
    png = encode_preview_png(frame, long_edge=360)
    decoded = _decode_bgra(png)
    assert decoded.shape[:2] == (50, 80)

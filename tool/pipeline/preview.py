"""Numpy RGBA frame -> small PNG bytes, for the server's live preview
(server/jobs.py's `_preview` hook -> GET /api/jobs/{id}/preview).

No such numpy->PNG helper existed before this: the pipeline's only other
PNG writer is __main__.py's QC contact sheet, which flattens alpha onto
opaque magenta and writes a 3-channel image -- the opposite of what a
transparent preview needs.
"""
from __future__ import annotations

import cv2
import numpy as np


def encode_preview_png(frame: np.ndarray, long_edge: int = 360,
                        compression: int = 1) -> bytes:
    """`frame` is one RGBA uint8 array, R/G/B/A channel order (as produced
    by tool.pipeline.runner.infer_clip -- NOT BGR), shape (H, W, 4).
    Downscales so the long edge is at most `long_edge` px (PNG encode cost
    is dominated by pixel count, and this is a "what's being worked on
    right now" glance, not a deliverable) and returns encoded PNG bytes
    with alpha preserved.

    `compression` is cv2's IMWRITE_PNG_COMPRESSION level (0-9); the
    default (1) trades a slightly larger file for single-digit-ms encode
    time at this size, which matters since this runs on the same thread
    as inference."""
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side > long_edge:
        scale = long_edge / long_side
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                            interpolation=cv2.INTER_AREA)
    # cv2's PNG encoder expects BGRA, not the RGBA order every frame in
    # this pipeline is actually stored in -- skipping this swaps red and
    # blue in the browser (confirmed by testing).
    bgra = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise RuntimeError("cv2.imencode(.png) failed for a preview frame")
    return buf.tobytes()

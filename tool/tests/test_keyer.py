"""Unit tests for the flat-chroma colour-only matting path.

Synthetic clips throughout: the properties under test (does a pixel this far
from the key colour become opaque, does the unpremultiply remove the key
colour's contribution, does the safety gate refuse a subject that shares the
key colour) are all exactly computable, so a real clip (e.g. flatchroma2/
purplebg) would only add decode
time and sampling noise. The real-clip behaviour is covered by
tool/tests/test_regression_gates.py against the shipped output.
"""
import cv2
import numpy as np
import pytest

from tool.pipeline import keyer
from tool.pipeline.runner import _KEY_SAMPLE_FRAMES, _sample_frames_for_key

KEY_RGB = (54, 73, 183)       # flatchroma2's blue backdrop
SUBJECT_RGB = (205, 133, 63)  # an orange mascot body, far from the key in chroma


def _clip(n_frames=4, size=128, subject_rgb=SUBJECT_RGB, subject_box=(40, 40, 88, 88),
          key_rgb=KEY_RGB):
    """A flat key-colour field with one solid rectangle standing in for the subject."""
    frames = np.empty((n_frames, size, size, 3), np.uint8)
    frames[:] = np.array(key_rgb, np.uint8)
    x0, y0, x1, y1 = subject_box
    frames[:, y0:y1, x0:x1] = np.array(subject_rgb, np.uint8)
    return frames


def _clip_with_key(key_rgb):
    return _clip(key_rgb=key_rgb)


def test_estimate_key_measures_a_flat_backdrop():
    model = keyer.estimate_key(_clip())
    assert model.is_chroma_class is True
    assert model.frac_bg_like > 0.98
    # bg_rgb is what the unpremultiply subtracts, so it must be the ACTUAL
    # backdrop colour, not merely something background-like.
    assert np.allclose(model.bg_rgb, KEY_RGB, atol=1)


def test_ramp_top_scales_with_the_key_colours_own_strength():
    """ramp_k scales `neutral_dist` -- the distance a grey pixel sits from
    THIS backdrop -- so one k adapts across key colours instead of being a
    fixed sigma count (see keyer.estimate_key). Checked behaviourally (a
    smaller k must narrow the ramp) rather than by re-deriving the formula
    `t_hi == ramp_k * neutral_dist`, which would just restate the
    implementation line back at itself and pass even if the multiplier
    changed to something else entirely."""
    wide = keyer.estimate_key(_clip(), ramp_k=0.5)
    narrow = keyer.estimate_key(_clip(), ramp_k=0.25)
    assert narrow.t_hi < wide.t_hi


def test_ramp_top_never_collapses_below_the_floor():
    """The floor (`max(_T_LO + 1.0, ramp_k * neutral_dist)`) is what keeps
    t_hi - t_lo >= 1.0 so key_frame's division by (t_hi - t_lo) can never
    be by zero. A backdrop with near-zero saturation (neutral_dist close to
    0) plus a tiny ramp_k is the case that would hit it if the floor were
    ever removed or miscoded."""
    model = keyer.estimate_key(_clip_with_key((131, 129, 127)), ramp_k=0.01)
    assert model.t_hi > model.t_lo
    assert model.t_hi - model.t_lo >= 1.0


def test_bg_rgb_stays_accurate_under_realistic_border_noise():
    """Audit M10: bg_rgb (what key_frame's unpremultiply subtracts) used to
    be a plain median over every ring pixel, independent of the sigma-clipped
    chroma stats estimate_bg_stats already computed for the SAME ring.
    Restricting the median to ring pixels within T_LO of that estimate
    guards against exactly the kind of light per-pixel border noise/
    compression artifact real video actually has (a hard block-colour
    synthetic clip can't demonstrate a numeric difference either way here --
    a median is inherently robust to any minority contamination below 50%,
    the same threshold estimate_bg_stats' own sigma-clipping needs to find
    the right cluster at all -- but this exercises the real code path with
    realistic per-pixel jitter and confirms it stays exact)."""
    rng = np.random.default_rng(0)
    size = 100
    key = np.array(KEY_RGB, np.int16)
    frames = np.empty((4, size, size, 3), np.uint8)
    noise = rng.integers(-2, 3, (4, size, size, 3))
    frames[:] = np.clip(key[None, None, None, :] + noise, 0, 255).astype(np.uint8)
    frames[:, 40:60, 40:60] = np.array(SUBJECT_RGB, np.uint8)  # inside the ring-free interior

    model = keyer.estimate_key(frames)
    assert model.is_chroma_class is True
    assert np.array_equal(model.bg_rgb, key.astype(np.float32))


def test_bg_rgb_falls_back_to_the_full_ring_median_if_nothing_passes_the_filter():
    """Defence in depth for the `ring_px[bg_like] if bg_like.any() else
    ring_px` fallback: if no ring pixel is confidently background-like
    (T_LO), estimate_key must still return SOME bg_rgb rather than indexing
    into an empty array (np.median([]) is nan, not a crash, but a NaN
    backdrop colour would silently poison every unpremultiply in
    key_frame)."""
    rng = np.random.default_rng(1)
    size = 60
    # every ring pixel is independent uniform noise, no coherent backdrop
    # colour at all -- essentially guarantees no pixel individually lands
    # within T_LO=3 sigma of whatever median the noise happens to average to.
    frames = rng.integers(0, 255, (3, size, size, 3), dtype=np.uint8)

    model = keyer.estimate_key(frames)
    assert np.all(np.isfinite(model.bg_rgb))


def test_a_weaker_key_colour_gets_a_proportionally_narrower_ramp():
    strong = keyer.estimate_key(_clip())                       # saturated blue
    weak = keyer.estimate_key(_clip_with_key((120, 128, 150)))  # barely-tinted grey
    assert weak.key_saturation < strong.key_saturation
    assert weak.t_hi < strong.t_hi


def test_alpha_channel_is_rounded_not_truncated():
    """Audit L1: a bare `.astype(np.uint8)` cast truncates, so an alpha
    that's numerically 0.999... (a ramp value just shy of 1.0) would land
    at 254 rather than 255. Invisible on a 1-bit GIF (alpha_threshold snaps
    it back to opaque either way) but a real, visible 1-level dimming on
    mov/webp output, which keep the soft channel as-is."""
    frames = _clip()
    model = keyer.estimate_key(frames)
    # A pixel well inside the ramp but a hair short of the top: d/t_hi just
    # under 1.0, so alpha should round UP to fully opaque (255), not
    # truncate down to 254.
    d = model.t_lo + (model.t_hi - model.t_lo) * (1 - 1e-4)
    # Reverse-engineer an RGB whose chroma distance from the key is `d`:
    # nudge Cb by d*sigma_cb (Cr unchanged) so the distance is exactly d.
    import cv2
    key_ycc = cv2.cvtColor(np.uint8([[KEY_RGB]]), cv2.COLOR_RGB2YCrCb)[0, 0].astype(np.int16)
    nudged_ycc = np.clip(key_ycc + [0, 0, round(d * model.sigma_cb)], 0, 255).astype(np.uint8)
    nudged_rgb = cv2.cvtColor(np.uint8([[nudged_ycc]]), cv2.COLOR_YCrCb2RGB)[0, 0]
    frame = frames[0].copy()
    frame[64, 30] = nudged_rgb  # outside the subject rect
    rgba = keyer.key_frame(frame, model)
    assert rgba[64, 30, 3] == 255


def test_background_is_transparent_and_subject_is_opaque():
    frames = _clip()
    model = keyer.estimate_key(frames)
    rgba = keyer.key_frame(frames[0], model)
    assert rgba.shape == (128, 128, 4)
    assert rgba[5, 5, 3] == 0        # far outside the subject rect
    assert rgba[64, 64, 3] == 255    # dead centre of it


def test_blended_pixel_gets_fractional_alpha_and_loses_its_key_colour():
    """A partly-covered boundary pixel must land INSIDE the ramp (not snap to
    0/1) and have its backdrop contribution pulled back out of its colour.

    Deliberately uses a lightly-covered pixel: with the shipped ramp_k=0.5 the
    ramp saturates at half the subject's distance, so anything from ~50%
    coverage upwards already reads as fully opaque and is left untouched (see
    key_frame's docstring -- the ramp estimates coverage rather than measuring
    it). 10% coverage sits well inside the ramp where the correction applies."""
    frames = _clip()
    model = keyer.estimate_key(frames)
    key = np.array(KEY_RGB, np.float32)
    blended = key * 0.9 + np.array(SUBJECT_RGB, np.float32) * 0.1
    frames[0, 64, 30] = blended.astype(np.uint8)  # outside the rect, so it can't be clamped to 1
    rgba = keyer.key_frame(frames[0], model)

    alpha = rgba[64, 30, 3] / 255.0
    assert 0.0 < alpha < 1.0
    # The corrected colour must have moved AWAY from the key colour -- that is
    # the despill working; without it the pixel would still read as backdrop.
    before = np.linalg.norm(blended - key)
    after = np.linalg.norm(rgba[64, 30, :3].astype(np.float32) - key)
    assert after > before * 2


def test_is_safe_refuses_a_flat_but_near_neutral_backdrop():
    """The case flatness alone does NOT catch, and the reason this second gate
    exists: a white/grey cyclorama is perfectly uniform (a real-footage
    fixture on a white cyclorama measured frac_bg_like=1.0000) but has no chroma to key on, so the
    subject's own neutral colours would be indistinguishable from it."""
    for neutral_backdrop in [(240, 240, 240), (128, 128, 128), (16, 16, 16)]:
        model = keyer.estimate_key(_clip_with_key(neutral_backdrop))
        ok, reason = keyer.is_safe(model)
        assert ok is False, f"{neutral_backdrop} should be refused"
        assert "neutral" in reason


def test_is_safe_does_not_claim_to_catch_a_key_coloured_subject():
    """Documents a KNOWN, deliberate limitation so a later reader doesn't
    assume it is covered.

    A subject genuinely containing the key colour is not detectable from
    colour alone (Smith & Blinn 1996), and the previous attempt to check it --
    "find the nearest subject pixel and require a margin" -- was circular: the
    subject could only be located by thresholding the very same distance, so
    the measurement came back pinned to that threshold on every real clip and
    the gate passed unconditionally while appearing to work. Removed rather
    than left in place looking protective; see keyer.is_safe's docstring."""
    model = keyer.estimate_key(_clip(subject_rgb=(58, 77, 179)))  # subject ~= the key colour
    ok, _ = keyer.is_safe(model)
    assert ok is True, "the gate is backdrop-only by design -- see this test's docstring"


def test_is_safe_accepts_a_well_separated_subject():
    ok, reason = keyer.is_safe(keyer.estimate_key(_clip()))
    assert ok is True
    assert "flat chroma" in reason


def test_is_safe_refuses_pure_noise():
    """Defence in depth. A uniformly random field actually PASSES the
    flat-chroma gate -- estimate_bg_stats measures its own huge sigma, so
    almost every border pixel falls within T_LO of the median and
    frac_bg_like reads ~1.0. The second gate catches it instead: with sigma
    that large nothing is far enough from the "backdrop" to register as a
    subject at all, so there is nothing to verify the key colour against and
    the fast path is refused. Either refusal is correct; asserting only that
    it IS refused keeps this from pinning down which gate fires."""
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (4, 128, 128, 3), dtype=np.uint8)
    ok, reason = keyer.is_safe(keyer.estimate_key(noisy))
    assert ok is False
    assert reason


def test_matte_is_deterministic_across_independently_built_frames():
    """The property every downstream simplification rests on: no temporal
    smoothing and the single-pass encoder are both justified only because
    identical pixels always produce identical alpha (unlike BiRefNet, whose
    run-to-run variation near the 1-bit threshold is what motivated the
    expensive two-pass encoder -- see the plan's V4 finding).

    Calling key_frame twice on the very same numpy array (as an earlier
    version of this test did) cannot exercise anything but pure-function
    determinism, which is trivially true of any side-effect-free numpy code
    -- it could never fail even if key_frame secretly depended on frame
    order or object identity. This instead builds two frames independently
    (different arrays, different estimate_key calls, same clip content) and
    checks they matte byte-identically -- the actual property S1's
    "structurally zero chatter on a static region" claim depends on."""
    a = _clip(n_frames=1)[0]
    b = _clip(n_frames=1)[0].copy()  # a distinct array with the same content
    assert a is not b
    model_a = keyer.estimate_key(_clip())
    model_b = keyer.estimate_key(_clip().copy())
    out_a = keyer.key_frame(a, model_a)
    out_b = keyer.key_frame(b, model_b)
    assert np.array_equal(out_a, out_b)


# -- audit H3: _sample_frames_for_key must never read more than
# -- _KEY_SAMPLE_FRAMES frames, regardless of what CAP_PROP_FRAME_COUNT
# -- reports (it can be 0, or a huge negative int64-min misread, for webm
# -- and some fragmented mov containers -- see server/probe.py's own note
# -- on the same underlying bug). The old version's `total_frames and
# -- total_frames > _KEY_SAMPLE_FRAMES` check let a negative count fall
# -- through into an unbounded "read every frame" loop.

def _write_synthetic_clip(tmp_path, n_frames, size=32, color_bgr=(183, 73, 54)):
    path = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (size, size))
    frame = np.full((size, size, 3), color_bgr, np.uint8)
    for _ in range(n_frames):
        vw.write(frame)
    vw.release()
    return path


@pytest.mark.parametrize("total_frames", [None, 0, -9223372036854775808], ids=["none", "zero", "huge_negative"])
def test_sampling_is_bounded_even_when_the_frame_count_is_unreliable(tmp_path, total_frames):
    path = _write_synthetic_clip(tmp_path, n_frames=60)
    out = _sample_frames_for_key(path, eff_scale=1.0, total_frames=total_frames)
    assert out is not None
    assert out.shape[0] <= _KEY_SAMPLE_FRAMES


def test_sampling_reads_the_whole_clip_when_it_is_shorter_than_the_cap(tmp_path):
    path = _write_synthetic_clip(tmp_path, n_frames=5)
    out = _sample_frames_for_key(path, eff_scale=1.0, total_frames=5)
    assert out.shape[0] == 5


def test_sampling_respects_eff_scale(tmp_path):
    path = _write_synthetic_clip(tmp_path, n_frames=10, size=32)
    out = _sample_frames_for_key(path, eff_scale=0.5, total_frames=10)
    assert out.shape[1:3] == (16, 16)


def test_sampling_stops_immediately_when_cancelled(tmp_path):
    path = _write_synthetic_clip(tmp_path, n_frames=60)

    class _AlreadySet:
        def is_set(self):
            return True

    assert _sample_frames_for_key(path, eff_scale=1.0, total_frames=None, cancel=_AlreadySet()) is None


# -- 第11計画 Part 1-1: the sampler walks the stream with grab()/retrieve()
# -- instead of 24 cap.set(CAP_PROP_POS_FRAMES) seeks (each of which re-
# -- decoded a single-keyframe clip from frame 0 -- 55-70s of dead time per
# -- clip). The chosen frames must be exactly the ones the seek version
# -- returned, or the keyer's model (and so every keyed output) would drift.

def _write_varying_clip(tmp_path, n_frames, size=48):
    """Every frame distinct (a moving bar + a per-frame grey level), so a
    sampler that picked the wrong index would return different pixels."""
    path = str(tmp_path / "varying.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (size, size))
    for i in range(n_frames):
        frame = np.full((size, size, 3), (40 + 3 * i) % 256, np.uint8)
        x = i % size
        frame[:, x:x + 4] = (0, 0, 255)
        vw.write(frame)
    vw.release()
    return path


def _seek_sampler_reference(video_path, eff_scale, total_frames):
    """The pre-第11計画 seek implementation, kept verbatim as the oracle."""
    from tool.pipeline.runner import _scale_and_convert
    cap = cv2.VideoCapture(str(video_path))
    idxs = sorted({round(k * (total_frames - 1) / (_KEY_SAMPLE_FRAMES - 1))
                   for k in range(_KEY_SAMPLE_FRAMES)})
    out = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            out.append(_scale_and_convert(bgr, eff_scale))
    cap.release()
    return np.stack(out, 0)


@pytest.mark.parametrize("n_frames,claimed", [(60, 60), (121, 121), (60, 75)],
                         ids=["60f", "121f", "count_overclaims"])
@pytest.mark.parametrize("eff_scale", [1.0, 0.5])
def test_sequential_sampling_matches_the_old_seek_sampling(tmp_path, n_frames, claimed, eff_scale):
    path = _write_varying_clip(tmp_path, n_frames)
    got = _sample_frames_for_key(path, eff_scale=eff_scale, total_frames=claimed)
    want = _seek_sampler_reference(path, eff_scale, claimed)
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_sequential_sampling_honours_a_cancel_set_mid_walk(tmp_path):
    path = _write_varying_clip(tmp_path, 60)

    class _SetAfter:
        def __init__(self, n):
            self.calls, self.n = 0, n

        def is_set(self):
            self.calls += 1
            return self.calls > self.n

    c = _SetAfter(10)
    assert _sample_frames_for_key(path, eff_scale=1.0, total_frames=60, cancel=c) is None
    assert c.calls == 11  # checked once per walked frame, stopped at the first set


def test_sequential_sampling_stops_at_the_last_chosen_index(tmp_path, monkeypatch):
    """Never walks past the last sampled frame (no pointless tail decode),
    and never seeks."""
    path = _write_varying_clip(tmp_path, 60)
    import tool.pipeline.runner as runner_mod
    real = cv2.VideoCapture
    counts = {"grab": 0, "set": 0}

    class _Counting:
        def __init__(self, *a):
            self._c = real(*a)

        def grab(self):
            counts["grab"] += 1
            return self._c.grab()

        def set(self, *a):
            counts["set"] += 1
            return self._c.set(*a)

        def __getattr__(self, name):
            return getattr(self._c, name)

    monkeypatch.setattr(runner_mod.cv2, "VideoCapture", _Counting)
    out = _sample_frames_for_key(path, eff_scale=1.0, total_frames=50)
    assert out.shape[0] == _KEY_SAMPLE_FRAMES
    assert counts["set"] == 0
    assert counts["grab"] == 50  # indices 0..49, last chosen index is 49


# -- 第11計画 Part 3-3: colour unmix with a separate coverage estimate --------

NEUTRALISH_RGB = (200, 200, 200)  # skin/white clothing-like: about neutral_dist from the key


def _edge_blended_clip(coverage, subject_rgb=NEUTRALISH_RGB):
    """Subject rect whose left boundary column is a `coverage` blend of
    subject over key -- a real edge pixel, adjacent to the backdrop.

    Near-neutral subject on purpose: the coverage estimate is a linear ramp
    up to neutral_dist, which is exact for a subject about as far from the
    key as grey is; a subject colour FURTHER out than that (e.g. saturated
    orange on blue) reaches full estimated coverage early and is only
    partially cleaned -- a documented limit, not what this test is about."""
    frames = _clip(subject_rgb=subject_rgb)
    key = np.array(KEY_RGB, np.float32)
    subj = np.array(subject_rgb, np.float32)
    frames[:, 40:88, 40] = (key * (1 - coverage) + subj * coverage).astype(np.uint8)
    return frames


def test_coverage_unmix_leaves_the_display_alpha_byte_identical():
    frames = _edge_blended_clip(0.6)
    old = keyer.key_frame(frames[0], keyer.estimate_key(frames))
    new = keyer.key_frame(frames[0], keyer.estimate_key(frames, unmix_coverage=True))
    np.testing.assert_array_equal(old[..., 3], new[..., 3])


def test_coverage_unmix_cleans_an_edge_pixel_that_displays_as_opaque():
    """With ramp_k=0.5 a 60%-covered edge pixel already reads as alpha=1, so
    the display-alpha unmix never touched it and it kept 40% key colour --
    the purple outline on the purplebg fixture."""
    frames = _edge_blended_clip(0.6)
    old = keyer.key_frame(frames[0], keyer.estimate_key(frames))
    new = keyer.key_frame(frames[0], keyer.estimate_key(frames, unmix_coverage=True))
    assert old[64, 40, 3] == 255 and new[64, 40, 3] == 255
    subj = np.array(NEUTRALISH_RGB, np.float32)
    err_old = np.linalg.norm(old[64, 40, :3].astype(np.float32) - subj)
    err_new = np.linalg.norm(new[64, 40, :3].astype(np.float32) - subj)
    assert err_new < 0.5 * err_old


def test_coverage_unmix_never_recolours_a_key_tinted_subject_interior():
    """A subject colour that is genuinely a bit key-tinted (d between t_hi and
    neutral_dist) but far from any edge must keep its own colour."""
    key = np.array(KEY_RGB, np.float32)
    tinted = (np.array((128, 128, 128), np.float32) * 0.8 + key * 0.2).astype(np.uint8)
    frames = _clip(subject_rgb=tuple(int(v) for v in tinted))
    model = keyer.estimate_key(frames, unmix_coverage=True)
    d = keyer._chroma_distance(tinted.reshape(1, 1, 3), model)[0, 0]
    assert model.t_hi < d < model.neutral_dist  # the case under test
    out = keyer.key_frame(frames[0], model)
    np.testing.assert_array_equal(out[64, 64, :3], tinted)
    np.testing.assert_array_equal(out[50:78, 50:78, :3], frames[0, 50:78, 50:78])


# -- 第11計画 Part 3-4: t_lo floor from the clip's own far-backdrop distance --

def _noisy_backdrop_clip(noise_sigma_levels, n_frames=4, seed=0):
    rng = np.random.default_rng(seed)
    frames = _clip(n_frames=n_frames).astype(np.int16)
    noise = rng.normal(0, noise_sigma_levels, frames.shape[:3] + (1,))
    frames = frames + np.round(noise * np.array([1.0, -1.0, 1.0])).astype(np.int16)
    frames[:, 40:88, 40:88] = np.array(SUBJECT_RGB, np.int16)
    return np.clip(frames, 0, 255).astype(np.uint8)


def test_measured_t_lo_stays_at_the_3_sigma_floor_on_a_clean_backdrop():
    model = keyer.estimate_key(_clip(), measured_t_lo=True)
    assert model.t_lo == pytest.approx(keyer._T_LO)


def test_measured_t_lo_rises_above_a_heavy_tailed_backdrop_and_removes_its_haze():
    """A backdrop whose own spread puts some pixels beyond 3 sigma (sigma is
    floored at 1.0 on a clean render) gets a haze of faint soft alpha under a
    fixed t_lo; the measured floor moves t_lo above that backdrop's p99.9."""
    frames = _noisy_backdrop_clip(2.0)
    fixed = keyer.estimate_key(frames)
    measured = keyer.estimate_key(frames, measured_t_lo=True)
    assert measured.backdrop_p999 is not None
    assert measured.t_lo > fixed.t_lo
    assert measured.t_lo <= 0.5 * measured.t_hi
    haze = lambda m: int(((keyer.key_frame(frames[0], m)[..., 3] > 0)[:30, :]).sum())  # rows far from the subject
    assert haze(measured) < haze(fixed)
    # the subject itself is unaffected
    assert (keyer.key_frame(frames[0], measured)[45:83, 45:83, 3] == 255).all()


def test_measured_t_lo_ignores_a_subject_touching_the_frame_edge():
    """The plain border ring's p99.9 reads 24-29 sigma on clips whose subject
    touches the edge; the far-backdrop measurement must not."""
    frames = _clip(subject_box=(0, 40, 60, 88))  # touches the left border
    model = keyer.estimate_key(frames, measured_t_lo=True)
    assert model.t_lo == pytest.approx(keyer._T_LO)


def test_pipeline_defaults_adopt_coverage_unmix_but_not_measured_t_lo():
    """Part 3-3 adopted (alpha untouched); Part 3-4 failed its F1i gate on
    the purplebg fixture and stays opt-in (see config.py / DECISIONS.md)."""
    from tool.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.keyer_unmix_coverage is True
    assert cfg.keyer_measured_t_lo is False

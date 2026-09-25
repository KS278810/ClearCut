"""Mode (瞬速/お急ぎ/じっくり) -> PipelineConfig mapping + override validation.

Mode ids are language-neutral ("instant"/"quick"/"thorough"); the front-end's
i18n dictionaries own the displayed labels/descriptions. This module's only
job is turning (mode, overrides) into a validated PipelineConfig + whether to
run QC, with every rejection carrying one of the language-neutral error codes
the front-end already knows how to render (see the plan's error-code list).
"""
from __future__ import annotations

import dataclasses
import functools

from tool.matte_core import resolve_device
from tool.pipeline.config import PipelineConfig

from . import errors


@functools.lru_cache(maxsize=1)
def resolved_auto_device() -> str:
    """`resolve_device("auto")`, computed at most once per process.

    It imports torch and calls torch.cuda.is_available(), which costs ~7s the
    first time and is what build_config's "auto" default triggers -- i.e. it
    landed inside the request handler for the first POST /api/jobs after a
    restart, stalling it for seven seconds (caught by the E2E suite's live-
    preview timing test, which started failing the moment "auto" became the
    default). The answer is fixed for the life of the process (a GPU does not
    appear or vanish mid-run), so caching it is safe as well as necessary.

    server/app.py's lifespan calls this at startup so even that first call is
    paid before the server begins accepting requests."""
    return resolve_device("auto")


class PresetError(Exception):
    """Carries a language-neutral `code` (see server/errors.py) plus a
    human-readable `detail` for logs -- the front-end renders code, not
    detail, so detail is free to be as specific as useful here."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


#: Field overrides applied on top of PipelineConfig's own defaults for each
#: mode. Empty dict = PipelineConfig's validated defaults untouched (this is
#: the exact configuration results_dinosaur/ was produced and accepted
#: under) -- "quick" and "thorough" deliberately do NOT touch matting
#: quality, only "instant" trades it away for speed.
PRESETS: dict[str, dict] = {
    # 瞬速: max_side=1000 (resolve_scale() derives ~0.6 for this repo's own
    # 1656px-long-side dinosaur fixtures, same ballpark as the old fixed
    # scale=0.6 -- see the plan's V1 measurement) + ss_alpha_gif (single-
    # pass encoder, ~8x faster than the default two-pass chain, see the
    # plan's L1 finding). Both were TESTED and REJECTED as the pipeline's
    # DEFAULT because of a visible defect (background colour left in fine
    # gaps, e.g. teeth) -- offered here anyway as an explicit, opt-in "fast
    # preview" trade the UI must label as such, not because the defect
    # stopped existing. Uses max_side rather than a fixed `scale` override
    # (the L2 finding) specifically so the UI's own max_side control (which
    # PipelineConfig.resolve_scale() gives `scale` precedence over) has any
    # effect at all in this mode -- a `scale` override here would silently
    # shadow whatever max_side the user picked.
    "instant": dict(max_side=1000, encoder="ss_alpha_gif"),
    # お急ぎ: PipelineConfig's own validated defaults -- the exact
    # configuration results_dinosaur/ (the accepted deliverable) was
    # rendered with.
    "quick": dict(),
    # じっくり: same defaults as "quick"; the only difference is qc=True
    # (see QC_BY_MODE) -- a metrics table + worst-frame contact sheet next
    # to the output, not a different render.
    "thorough": dict(),
}

QC_BY_MODE: dict[str, bool] = {"instant": False, "quick": False, "thorough": True}

#: "auto" isn't a real PipelineConfig/matte_core device -- build_config()
#: resolves it (via tool.matte_core.resolve_device) to a concrete "cuda"/
#: "cpu" before ever constructing a PipelineConfig, so nothing downstream
#: (Models, the GpuWorker's per-device model cache, job.json's persisted
#: config) ever sees the string "auto". This is what answers "GPU版・CPU版を
#: 端末の性能をみて自動判断できないか": rather than always attempting CUDA and
#: relying on onnxruntime's own silent same-process fallback (which only a
#: RuntimeWarning in the log distinguishes from a real GPU run), the server
#: checks CUDA usability up front -- cheaply, via get_available_providers()/
#: torch.cuda.is_available(), no session/inference needed -- and picks CPU
#: outright when it isn't.
DEVICES = frozenset({"cuda", "cpu", "auto"})
ENCODERS = frozenset({"supersampled_gif", "ss_alpha_gif", "mov", "webp"})
SUPERSAMPLE_CHOICES = frozenset({1, 2, 3, 4})

#: key -> (kind, spec). kind is one of "bool" / "enum" / "range" (int or
#: float) / "range_int" / "range_or_none_int". Anything not listed here is
#: not overridable from the API at all (e.g. the real-footage-only
#: apply_clears/collapse_fix/colour_freeze knobs, or the clear_*/fringe_*
#: sub-parameters of stages this chroma-only service never turns on) --
#: an unknown key is rejected as E_BAD_OVERRIDE rather than silently
#: ignored, so a front-end/back-end drift shows up immediately instead of
#: quietly not doing what the UI implied it would.
KEYER_MODES = frozenset({"auto", "on", "off"})

ALLOWED_OVERRIDES: dict[str, tuple[str, object]] = {
    "device": ("enum", DEVICES),
    # Exposed so a clip whose subject genuinely contains the key colour --
    # the one failure the backdrop-only safety gate cannot detect, see
    # tool/pipeline/keyer.py's is_safe -- can be forced back onto the
    # BiRefNet route without editing config defaults.
    "use_keyer": ("enum", KEYER_MODES),
    "encoder": ("enum", ENCODERS),
    "use_trimap": ("bool", None),
    "apply_despill": ("bool", None),
    "strip_bg_fringe": ("bool", None),
    # 第12計画: not surfaced in the UI (settings only exposes output format,
    # 第11計画 Part 4) -- API/CLI-only, same tier as use_trimap/strip_bg_fringe
    # above. Pre-registered gate result decides whether this becomes the
    # neural-route default; until then it stays reachable for a caller that
    # wants to opt in per-clip (see config.py's docstring for the rule).
    "keep_main_subject": ("bool", None),
    "min_box_frac": ("range", (0.01, 0.9)),
    "scale": ("range", (0.2, 1.0)),
    "max_side": ("range_or_none_int", (256, 4096)),
    "supersample": ("enum", SUPERSAMPLE_CHOICES),
    "alpha_threshold": ("range_int", (0, 255)),
    "mc_median_half": ("range_int", (0, 3)),
}


def _validate_override(key: str, kind: str, spec, value) -> None:
    if kind == "bool":
        if not isinstance(value, bool):
            raise PresetError(errors.E_BAD_OVERRIDE, f"{key} must be a bool, got {value!r}")
    elif kind == "enum":
        if value not in spec:
            raise PresetError(errors.E_BAD_OVERRIDE,
                               f"{key} must be one of {sorted(spec, key=str)}, got {value!r}")
    elif kind == "range":
        lo, hi = spec
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (lo <= value <= hi):
            raise PresetError(errors.E_BAD_OVERRIDE, f"{key} must be a number in [{lo}, {hi}], got {value!r}")
    elif kind == "range_int":
        lo, hi = spec
        if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
            raise PresetError(errors.E_BAD_OVERRIDE, f"{key} must be an int in [{lo}, {hi}], got {value!r}")
    elif kind == "range_or_none_int":
        if value is None:
            return
        lo, hi = spec
        if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
            raise PresetError(errors.E_BAD_OVERRIDE,
                               f"{key} must be null or an int in [{lo}, {hi}], got {value!r}")
    else:  # pragma: no cover -- defensive, ALLOWED_OVERRIDES is closed above
        raise AssertionError(f"unknown validator kind {kind!r} for {key!r}")


def build_config(mode: str, overrides: dict | None = None) -> tuple[PipelineConfig, bool]:
    """Turn (mode, overrides) into a (PipelineConfig, run_qc) pair, or raise
    PresetError. `overrides` may additionally carry the pseudo-field "qc"
    (bool) to force QC on/off regardless of the mode's default."""
    if mode not in PRESETS:
        raise PresetError(errors.E_BAD_MODE, f"unknown mode {mode!r}; choices: {sorted(PRESETS)}")

    fields = dict(PRESETS[mode])
    qc = QC_BY_MODE[mode]

    for key, value in (overrides or {}).items():
        if key == "qc":
            if not isinstance(value, bool):
                raise PresetError(errors.E_BAD_OVERRIDE, f"qc must be a bool, got {value!r}")
            qc = value
            continue
        if key not in ALLOWED_OVERRIDES:
            raise PresetError(errors.E_BAD_OVERRIDE, f"unknown/unsupported override field {key!r}")
        kind, spec = ALLOWED_OVERRIDES[key]
        _validate_override(key, kind, spec, value)
        fields[key] = value

    # None of PRESETS sets "device" (PipelineConfig's own hardcoded default,
    # "cuda", is a CLI-oriented default that assumes CUDA is always usable --
    # true for this repo's own fixtures/CI but not guaranteed for a server
    # that keeps running across host changes). The server's own default is
    # "auto" unless the client explicitly asked for "cuda"/"cpu".
    fields.setdefault("device", "auto")
    if fields["device"] == "auto":
        fields["device"] = resolved_auto_device()

    # See PipelineConfig.encoder_explicit's docstring: only a genuine
    # per-request override counts, not a preset (PRESETS["instant"] sets
    # encoder="ss_alpha_gif" itself, but that value never collides with the
    # two swaps this flag gates -- both only fire when encoder is still
    # "supersampled_gif").
    fields["encoder_explicit"] = "encoder" in (overrides or {})

    try:
        config = PipelineConfig(**fields)
    except TypeError as e:
        # A field name that IS in ALLOWED_OVERRIDES but somehow isn't a
        # PipelineConfig field would be a bug in this module's table, not a
        # bad request -- but fail closed as a bad request rather than a
        # 500, since the practical effect (config didn't build) is the same
        # either way from the caller's side.
        raise PresetError(errors.E_BAD_OVERRIDE, str(e))
    return config, qc


def config_to_dict(config: PipelineConfig) -> dict:
    return dataclasses.asdict(config)

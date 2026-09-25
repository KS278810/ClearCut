import pytest

from server import errors, presets
from server.presets import ALLOWED_OVERRIDES, PRESETS, PresetError, build_config, config_to_dict


@pytest.fixture(autouse=True)
def _resolve_device_to_cuda(monkeypatch):
    """build_config() now resolves an unset/"auto" device via
    tool.matte_core.resolve_device("auto"), which genuinely probes THIS
    host's onnxruntime/torch CUDA availability -- fine for the real server,
    but it would make every other test in this file depend on whether the
    machine running pytest happens to have a working CUDA stack. Pin it to
    "cuda" (the pre-existing assumption these tests were written under) so
    they stay deterministic regardless of host; the auto-detection logic
    itself is exercised directly below.

    The cache_clear calls are load-bearing: the result is memoized for the
    process (see resolved_auto_device -- it exists to keep a ~7s torch probe
    off the request path), so without clearing it either side, the first
    test's answer would leak into every later test and monkeypatching the
    probe would silently do nothing."""
    presets.resolved_auto_device.cache_clear()
    monkeypatch.setattr(presets, "resolve_device", lambda pref: "cuda")
    yield
    presets.resolved_auto_device.cache_clear()


@pytest.mark.parametrize("mode", sorted(PRESETS))
def test_each_mode_builds_a_valid_config(mode):
    config, qc = build_config(mode)
    assert config.device == "cuda"  # via the pinned resolve_device fixture above
    assert isinstance(qc, bool)


def test_unset_device_defaults_to_auto_and_gets_resolved(monkeypatch):
    """Answers "GPU版・CPU版を端末の性能をみて自動判断できないか": when the
    client doesn't specify a device at all, build_config asks
    tool.matte_core.resolve_device("auto") instead of always assuming CUDA
    is usable -- and the concrete result (never the string "auto" itself)
    is what ends up on the PipelineConfig / job.json."""
    monkeypatch.setattr(presets, "resolve_device", lambda pref: "cpu")
    config, _ = build_config("quick")
    assert config.device == "cpu"


def test_explicit_device_override_skips_auto_resolution(monkeypatch):
    def _boom(pref):
        raise AssertionError("resolve_device must not be called for an explicit device override")
    monkeypatch.setattr(presets, "resolve_device", _boom)
    config, _ = build_config("quick", {"device": "cpu"})
    assert config.device == "cpu"


def test_explicit_auto_override_is_also_resolved(monkeypatch):
    monkeypatch.setattr(presets, "resolve_device", lambda pref: "cpu")
    config, _ = build_config("quick", {"device": "auto"})
    assert config.device == "cpu"


def test_instant_trades_speed_for_quality_as_documented():
    """Regression test for the plan's L2 finding: "instant" used to set a
    fixed scale=0.6 override, which resolve_scale() gives precedence over
    max_side -- meaning the UI's own max_side control silently had no
    effect while this mode was selected. It now expresses the same speed
    trade via max_side instead, leaving scale at PipelineConfig's own
    default (1.0, i.e. "not explicitly set" from resolve_scale()'s point
    of view) so an explicit max_side override still works in this mode."""
    from tool.pipeline.runner import resolve_scale
    config, qc = build_config("instant")
    assert config.scale == 1.0
    assert config.max_side == 1000
    assert config.encoder == "ss_alpha_gif"
    assert qc is False
    # This repo's own dinosaur fixtures are 1656px on the long side -- same
    # ballpark speed trade as the old fixed scale=0.6 (the plan's V1 measurement).
    assert resolve_scale(config, 1656, 1248) == pytest.approx(1000 / 1656, abs=1e-6)


def test_quick_matches_results_dinosaur_defaults():
    from tool.pipeline.config import PipelineConfig
    config, qc = build_config("quick")
    assert config == PipelineConfig()
    assert qc is False


def test_thorough_is_quick_plus_qc():
    from tool.pipeline.config import PipelineConfig
    config, qc = build_config("thorough")
    assert config == PipelineConfig()
    assert qc is True


def test_unknown_mode_rejected():
    with pytest.raises(PresetError) as exc:
        build_config("ultra")
    assert exc.value.code == errors.E_BAD_MODE


def test_unknown_override_field_rejected():
    with pytest.raises(PresetError) as exc:
        build_config("quick", {"apply_clears": True})
    assert exc.value.code == errors.E_BAD_OVERRIDE


@pytest.mark.parametrize("field,bad_value", [
    ("device", "tpu"),
    ("encoder", "avif"),
    ("use_trimap", "yes"),
    ("min_box_frac", 5.0),
    ("scale", 0.05),
    ("max_side", "1000"),
    ("max_side", 100),
    ("supersample", 5),
    ("alpha_threshold", 300),
    ("mc_median_half", -1),
    ("keep_main_subject", "yes"),
])
def test_out_of_range_or_wrong_type_overrides_rejected(field, bad_value):
    with pytest.raises(PresetError) as exc:
        build_config("quick", {field: bad_value})
    assert exc.value.code == errors.E_BAD_OVERRIDE


def test_valid_overrides_are_applied():
    config, qc = build_config("quick", {"device": "cpu", "scale": 0.8, "max_side": None, "qc": True})
    assert config.device == "cpu"
    assert config.scale == 0.8
    assert config.max_side is None
    assert qc is True


def test_qc_pseudo_field_must_be_bool():
    with pytest.raises(PresetError) as exc:
        build_config("quick", {"qc": "true"})
    assert exc.value.code == errors.E_BAD_OVERRIDE


def test_every_allowed_override_is_a_real_pipelineconfig_field():
    from tool.pipeline.config import PipelineConfig
    field_names = {f.name for f in __import__("dataclasses").fields(PipelineConfig)}
    assert set(ALLOWED_OVERRIDES) <= field_names


def test_config_to_dict_round_trips_through_dataclasses_asdict():
    config, _ = build_config("quick")
    d = config_to_dict(config)
    assert d["device"] == "cuda"
    assert d["encoder"] == "ss_alpha_gif"


def test_auto_device_is_resolved_at_most_once_per_process(monkeypatch):
    """Regression test for a 7-second stall on the first POST /api/jobs.

    resolve_device("auto") imports torch and calls torch.cuda.is_available(),
    which takes ~7s the first time. When "auto" became build_config's default
    that cost landed inside the request handler, so the first job submitted
    after a server restart hung for seven seconds -- caught only because the
    E2E suite's live-preview timing test started failing. The answer can't
    change during a process's life, so it is memoized (and warmed at startup
    in server/app.py's lifespan)."""
    presets.resolved_auto_device.cache_clear()
    calls = []
    monkeypatch.setattr(presets, "resolve_device", lambda pref: calls.append(pref) or "cuda")
    try:
        for _ in range(5):
            build_config("quick")
        assert calls == ["auto"], f"expected exactly one probe, got {len(calls)}"
    finally:
        presets.resolved_auto_device.cache_clear()

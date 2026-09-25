"""Regression tests for tool/matte_core.py's Models.__init__ device wiring.

Mocks onnxruntime.InferenceSession and _OnnxYOLO so these run in
milliseconds without loading any real ONNX file or touching a real GPU --
what's being tested is the WIRING (which device each session is built
for, thread-count settings, and the provider-mismatch warning), not
inference correctness itself (that's tool/tests/test_runner_limits.py's
job, gated on a real sample clip).
"""
import warnings

import numpy as np

import onnxruntime
import pytest

from tool import matte_core


class _FakeSession:
    def __init__(self, path, sess_options=None, providers=None):
        self.path = path
        self.sess_options = sess_options
        self.providers = providers or []

    def get_providers(self):
        # Real onnxruntime's get_providers() always returns plain provider
        # NAME strings, even when a (name, options_dict) tuple was passed
        # into `providers=` at construction -- mirror that here, since
        # matte_core.py's own membership check ("CUDAExecutionProvider"
        # not in sess.get_providers()) depends on it.
        return [p[0] if isinstance(p, tuple) else p for p in self.providers]

    def get_inputs(self):
        class _Inp:
            name = "input"
        return [_Inp()]


class _FakeYOLO:
    """Stands in for matte_core._OnnxYOLO -- records what device/session
    options it was constructed with instead of loading the real ONNX file."""
    instances = []

    def __init__(self, onnx_path, so, device):
        self.onnx_path = onnx_path
        self.so = so
        self.device = device
        self.sess = _FakeSession(onnx_path, so, providers=_FakeYOLO.providers_for(device))
        _FakeYOLO.instances.append(self)

    @staticmethod
    def providers_for(device):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]


@pytest.fixture(autouse=True)
def _fake_onnx(monkeypatch):
    _FakeYOLO.instances = []
    monkeypatch.setattr(matte_core, "_OnnxYOLO", _FakeYOLO)
    monkeypatch.setattr(onnxruntime, "InferenceSession",
                         lambda path, sess_options=None, providers=None:
                             _FakeSession(path, sess_options, providers))
    yield


def test_yolox_session_follows_the_requested_device_not_hardcoded_cpu():
    """Regression test: YOLOX's own onnxruntime session used to be hardcoded
    to "cpu" regardless of Models(device=...) -- on this machine that ran at
    ~4.5s/frame (vs ~0.02s/frame on CUDA), and because runner.py folded its
    cost into the same timer as BiRefNet, this went undetected as the
    dominant cost of the whole pipeline (see runner.py's t_detect/t_birefnet
    split, added alongside this fix)."""
    matte_core.Models(device="cuda")
    assert _FakeYOLO.instances[-1].device == "cuda"

    matte_core.Models(device="cpu")
    assert _FakeYOLO.instances[-1].device == "cpu"


def test_yolox_thread_count_is_capped_not_maximized():
    """Regression test: intra_op_num_threads used to be os.cpu_count() (24
    on this machine) which measured 2.1x SLOWER than a small fixed cap --
    ORT's own thread pool thrashes past a point on this workload."""
    matte_core.Models(device="cpu")
    so = _FakeYOLO.instances[-1].so
    assert so.intra_op_num_threads <= 4


def test_cuda_requested_but_cpu_provider_actually_used_warns():
    """Regression test: a CUDA/cuDNN library mismatch makes onnxruntime
    silently fall back to CPUExecutionProvider with zero indication
    anywhere (confirmed reproducible by stripping LD_LIBRARY_PATH) --
    BiRefNet then runs at ~50s/frame instead of ~0.13s/frame with nothing
    in the logs to explain why. device="cuda" must now warn loudly."""
    monkey_providers = {"det": ["CPUExecutionProvider"], "brf": ["CPUExecutionProvider"]}

    class _FallenBackYOLO(_FakeYOLO):
        def __init__(self, onnx_path, so, device):
            super().__init__(onnx_path, so, device)
            self.sess.providers = monkey_providers["det"]

    import tool.matte_core as mc
    orig_yolo = mc._OnnxYOLO
    mc._OnnxYOLO = _FallenBackYOLO
    orig_sess = onnxruntime.InferenceSession
    onnxruntime.InferenceSession = (lambda path, sess_options=None, providers=None:
                                     _FakeSession(path, sess_options, monkey_providers["brf"]))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mc.Models(device="cuda")
        messages = [str(w.message) for w in caught]
        assert any("YOLOX" in m and "cuda" in m.lower() for m in messages)
        assert any("BiRefNet" in m and "cuda" in m.lower() for m in messages)
    finally:
        mc._OnnxYOLO = orig_yolo
        onnxruntime.InferenceSession = orig_sess


def test_cuda_requested_and_actually_used_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matte_core.Models(device="cuda")
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


# -- 第11計画 Part 3-2: min_box_frac never rejects a confident person box --

class _FakeDet:
    def __init__(self, xyxy, cls, conf):
        self._r = (np.asarray(xyxy, np.float32), np.asarray(cls), np.asarray(conf, np.float32))

    def detect(self, rgb):
        return self._r


def _subject_box(det, min_box_frac=0.15):
    from tool.matte_core import Models
    fake_self = type("M", (), {"det": det})()
    return Models.subject_box(fake_self, np.zeros((100, 100, 3), np.uint8),
                              prefer_person=False, min_box_frac=min_box_frac)


@pytest.mark.parametrize("cls,conf,kept", [
    (0, 0.91, True),    # widepose: low-confidence person at ~13% of the frame
    (0, 0.80, True),    # threshold is inclusive
    (0, 0.79, False),   # an unsure person still has to clear the size gate
    (58, 0.95, False),  # a confident NON-person fragment is exactly the misfire class
])
def test_min_box_frac_exempts_only_confident_person_boxes(cls, conf, kept):
    box, _ = _subject_box(_FakeDet([[0, 0, 36, 36]], [cls], [conf]))  # 12.96% of the frame
    assert (box is not None) == kept


def test_min_box_frac_still_passes_large_boxes_of_any_class():
    box, is_person = _subject_box(_FakeDet([[0, 0, 60, 60]], [58], [0.3]))
    assert box is not None and not is_person

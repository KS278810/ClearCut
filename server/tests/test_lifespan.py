"""Unit test for app.py's _make_lifespan device-probe timeout (audit M7).

uvicorn runs lifespan startup BEFORE binding the socket, so an unbounded
await on the device probe (which imports torch and calls
torch.cuda.is_available() -- see presets.resolved_auto_device) means a
wedged probe (a known failure mode with a hung/reset NVIDIA driver) would
leave the server never listening at all, with no log line and the uptime
watchdog never even created. This exercises the real async context manager
end to end (not a fake), with the probe monkeypatched to outlast a very
short timeout -- the lifespan must still complete and yield.
"""
import asyncio
import logging

import pytest

from server import app as app_mod
from server import presets


def test_lifespan_proceeds_past_a_wedged_device_probe(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(app_mod, "_DEVICE_PROBE_TIMEOUT_S", 0.05)

    def _wedged():
        import time
        time.sleep(0.3)  # outlasts the timeout above; real thread, not cancellable
        return "cuda"

    monkeypatch.setattr(presets, "resolved_auto_device", _wedged)

    lifespan = app_mod._make_lifespan(tmp_path)
    app = type("FakeApp", (), {"state": type("State", (), {})()})()

    async def run_once():
        async with lifespan(app):
            pass  # the point under test is reaching this line at all

    with caplog.at_level(logging.WARNING, logger="heroextractor"):
        asyncio.run(asyncio.wait_for(run_once(), timeout=5.0))

    assert any("did not finish within" in r.message for r in caplog.records)
    # Startup must have continued past the timeout to actually build the
    # store/worker -- not silently skip the rest of _lifespan.
    assert hasattr(app.state, "store")
    assert hasattr(app.state, "worker")

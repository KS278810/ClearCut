"""Unit tests for app.py's _uptime_watchdog (HEROEXTRACTOR_MAX_UPTIME_S --
see run.sh at this repo's parent directory). No real FastAPI app/lifespan
involved; a fake store/worker is enough to exercise the wait-for-idle
logic and the actual shutdown trigger in isolation. Plain `asyncio.run()`
rather than pytest-asyncio (not a dependency of this project) -- these
coroutines need nothing from a pytest-managed event loop."""
import asyncio
import queue

import pytest

from server import app as app_mod


class _FakeStore:
    def __init__(self, jobs):
        self._jobs = jobs

    def list(self, *, summary=False):
        return self._jobs


class _FakeWorker:
    def __init__(self, queue_empty=True):
        self.job_queue = queue.Queue()
        if not queue_empty:
            self.job_queue.put("some-job-id")


def test_watchdog_shuts_down_immediately_when_already_idle(monkeypatch):
    monkeypatch.setattr(app_mod, "_UPTIME_WATCHDOG_POLL_S", 0.001)
    killed = []
    monkeypatch.setattr(app_mod.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    store = _FakeStore([{"status": "done"}, {"status": "failed"}])
    worker = _FakeWorker(queue_empty=True)
    asyncio.run(app_mod._uptime_watchdog(0.0, store, worker))
    assert killed, "expected the watchdog to call os.kill once idle"


def test_watchdog_waits_for_a_running_job_before_shutting_down(monkeypatch):
    monkeypatch.setattr(app_mod, "_UPTIME_WATCHDOG_POLL_S", 0.001)
    killed = []
    monkeypatch.setattr(app_mod.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    jobs = [{"status": "running"}]
    store = _FakeStore(jobs)
    worker = _FakeWorker(queue_empty=True)

    async def _scenario():
        async def _finish_job_soon():
            await asyncio.sleep(0.01)
            jobs[0] = {"status": "done"}

        finisher = asyncio.create_task(_finish_job_soon())
        await app_mod._uptime_watchdog(0.0, store, worker)
        await finisher

    asyncio.run(_scenario())
    assert killed, "expected the watchdog to eventually shut down once the job finished"


def test_watchdog_waits_for_a_non_empty_queue_before_shutting_down(monkeypatch):
    monkeypatch.setattr(app_mod, "_UPTIME_WATCHDOG_POLL_S", 0.001)
    killed = []
    monkeypatch.setattr(app_mod.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    store = _FakeStore([{"status": "done"}])
    worker = _FakeWorker(queue_empty=False)  # a job is still queued

    async def _scenario():
        async def _drain_queue_soon():
            await asyncio.sleep(0.01)
            worker.job_queue.get_nowait()

        drainer = asyncio.create_task(_drain_queue_soon())
        await app_mod._uptime_watchdog(0.0, store, worker)
        await drainer

    asyncio.run(_scenario())
    assert killed


def test_watchdog_respects_the_initial_deadline_sleep(monkeypatch):
    """A non-zero deadline must actually be slept BEFORE the first idle
    check, even if the queue is already idle -- otherwise this is not an
    "after N seconds of uptime" cap at all."""
    monkeypatch.setattr(app_mod, "_UPTIME_WATCHDOG_POLL_S", 0.001)
    killed = []
    monkeypatch.setattr(app_mod.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    real_sleep = asyncio.sleep
    slept_for = []

    async def _fake_sleep(seconds):
        slept_for.append(seconds)
        await real_sleep(0)  # don't actually wait in the test

    monkeypatch.setattr(app_mod.asyncio, "sleep", _fake_sleep)
    store = _FakeStore([{"status": "done"}])
    worker = _FakeWorker(queue_empty=True)
    asyncio.run(app_mod._uptime_watchdog(21600.0, store, worker))
    assert slept_for[0] == 21600.0
    assert killed


def test_watchdog_logs_instead_of_dying_silently_on_an_unexpected_error(monkeypatch, caplog):
    """Regression test for audit L7/L13: this coroutine runs as a fire-and-
    forget asyncio.create_task in _make_lifespan, whose result nothing ever
    awaits -- before this fix, an exception inside the poll loop (e.g.
    store.list() raising) would be swallowed by asyncio entirely, silently
    disabling HEROEXTRACTOR_MAX_UPTIME_S for the rest of the process's life
    with nothing in the log to explain why. Must now be logged via
    logger.exception rather than propagating unseen."""
    import logging

    monkeypatch.setattr(app_mod, "_UPTIME_WATCHDOG_POLL_S", 0.001)

    class _BoomStore:
        def list(self, *, summary=False):
            raise RuntimeError("synthetic failure")

    worker = _FakeWorker(queue_empty=True)
    with caplog.at_level(logging.ERROR, logger="heroextractor"):
        asyncio.run(app_mod._uptime_watchdog(0.0, _BoomStore(), worker))
    assert any("watchdog crashed" in r.message for r in caplog.records)


def test_watchdog_cancellation_is_not_logged_as_a_crash(monkeypatch, caplog):
    """The normal shutdown path cancels this task (_lifespan's finally) --
    that must propagate as CancelledError, not get misreported as the
    watchdog having crashed."""
    import logging

    async def _scenario():
        task = asyncio.create_task(app_mod._uptime_watchdog(100.0, _FakeStore([]), _FakeWorker()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.ERROR, logger="heroextractor"):
        asyncio.run(_scenario())
    assert not any("crashed" in r.message for r in caplog.records)

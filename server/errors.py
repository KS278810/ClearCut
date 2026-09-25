"""Language-neutral error codes shared by the whole server package.

Every code the API/job store/GPU worker can produce is listed here, once,
so nothing downstream (front-end i18n, logs, tests) has to guess at the set
-- the front-end's error dictionary should have exactly one `err.<CODE>`
entry per name below, plus an `err.UNKNOWN` fallback for anything that
somehow isn't (a version-skew safety net, not something normal use should
ever hit).
"""
from __future__ import annotations

# -- input validation (server/probe.py, presets.build_config) --
E_INPUT_EXT = "E_INPUT_EXT"
E_INPUT_UNREADABLE = "E_INPUT_UNREADABLE"
E_INPUT_TOO_LONG = "E_INPUT_TOO_LONG"
E_INPUT_TOO_LARGE = "E_INPUT_TOO_LARGE"
E_BAD_MODE = "E_BAD_MODE"
E_BAD_OVERRIDE = "E_BAD_OVERRIDE"

# -- job lifecycle (server/jobs.py, server/app.py) --
E_JOB_NOT_FOUND = "E_JOB_NOT_FOUND"
E_JOB_RUNNING = "E_JOB_RUNNING"
E_CANCELLED = "E_CANCELLED"
E_SERVER_RESTART = "E_SERVER_RESTART"
E_INPUTS_EXPIRED = "E_INPUTS_EXPIRED"

# -- upload handling (server/app.py's Content-Length middleware + _save_uploads) --
E_UPLOAD_TOO_LARGE = "E_UPLOAD_TOO_LARGE"

# -- pipeline execution (server/jobs.py's GpuWorker exception classifier) --
E_CUDA_OOM = "E_CUDA_OOM"
E_ENCODE_TIMEOUT = "E_ENCODE_TIMEOUT"
E_PIPELINE = "E_PIPELINE"

ALL_CODES = frozenset({
    E_INPUT_EXT, E_INPUT_UNREADABLE, E_INPUT_TOO_LONG, E_INPUT_TOO_LARGE,
    E_BAD_MODE, E_BAD_OVERRIDE,
    E_JOB_NOT_FOUND, E_JOB_RUNNING, E_CANCELLED, E_SERVER_RESTART, E_INPUTS_EXPIRED,
    E_UPLOAD_TOO_LARGE,
    E_CUDA_OOM, E_ENCODE_TIMEOUT, E_PIPELINE,
})

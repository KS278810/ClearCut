"""serve.sh must never fall back to binding 0.0.0.0 when tailscale is
unavailable -- see the plan's B2 finding: `head -1`'s exit code used to
mask a failing `tailscale ip -4`, so `set -e` never caught it and HOST
silently became an empty string (which uvicorn/asyncio treats as "all
interfaces"). These tests exercise the real script with fakes on PATH,
not a re-implementation of its logic.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVE_SH = REPO_ROOT / "serve.sh"
RUN_SH = REPO_ROOT / "run.sh"


def test_serve_sh_is_syntactically_valid_bash():
    result = subprocess.run(["bash", "-n", str(SERVE_SH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_run_sh_is_syntactically_valid_bash():
    result = subprocess.run(["bash", "-n", str(RUN_SH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _write_fake_bin(tmp_path: Path, name: str, script_body: str) -> Path:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/bin/bash\n{script_body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def test_serve_sh_exits_nonzero_and_never_execs_python_when_tailscale_fails(tmp_path):
    """A failing `tailscale` must not silently become HOST="" -> 0.0.0.0.
    Points `python3`/`venv/bin/python3` at a sentinel-writing fake too, so
    if serve.sh ever DID reach the final `exec`, this test would catch it
    instead of actually trying to bind a port."""
    fake_tailscale_dir = _write_fake_bin(tmp_path, "tailscale", "exit 1")
    sentinel = tmp_path / "python_was_exec_d"
    fake_python_dir = _write_fake_bin(
        tmp_path, "python3", f"echo EXECUTED > {sentinel}\nexit 0",
    )

    env = {**os.environ, "PATH": f"{fake_tailscale_dir}:{fake_python_dir}:{os.environ['PATH']}"}
    env.pop("HOST", None)
    result = subprocess.run(["bash", str(SERVE_SH)], capture_output=True, text=True, env=env,
                             cwd=REPO_ROOT, timeout=15)

    assert result.returncode != 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    # (the error message's own explanatory text legitimately contains the
    # substring "0.0.0.0" -- e.g. "0.0.0.0では待ち受けません" -- so assert on
    # the sentinel never being created, which is what actually matters: no
    # uvicorn --host 0.0.0.0 was ever launched.)
    assert not sentinel.exists(), "serve.sh reached exec despite tailscale failing"


def test_serve_sh_respects_explicit_host_override(tmp_path):
    """HOST=... must skip the tailscale lookup entirely. serve.sh cd's to
    its own directory and execs the REAL repo venv's python3 -m uvicorn,
    so this test doesn't fake that part -- it binds a non-routable local
    IP (203.0.113.5, TEST-NET-3) which fails fast at the socket-bind step
    regardless, well after the point this test actually cares about
    (whether HOST resolution needed tailscale at all)."""
    fake_tailscale_dir = _write_fake_bin(tmp_path, "tailscale", "echo SHOULD_NOT_RUN; exit 1")
    env = {**os.environ, "HOST": "203.0.113.5", "PATH": f"{fake_tailscale_dir}:{os.environ['PATH']}"}
    proc = subprocess.Popen(["bash", str(SERVE_SH)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, cwd=REPO_ROOT, text=True)
    try:
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    assert "SHOULD_NOT_RUN" not in out + err
    assert "tailscale IP" not in err  # didn't hit the "unavailable" error path


@pytest.mark.skipif(shutil.which("tailscale") is None, reason="no system tailscale to compare against")
def test_serve_sh_and_run_sh_export_the_same_nvidia_lib_dirs():
    serve_text = SERVE_SH.read_text()
    run_text = RUN_SH.read_text()
    for lib in ("cudnn", "cublas", "cufft", "curand", "cuda_runtime", "nvjitlink", "cuda_nvrtc"):
        assert lib in serve_text, f"{lib} missing from serve.sh LD_LIBRARY_PATH"
        assert lib in run_text, f"{lib} missing from run.sh LD_LIBRARY_PATH"

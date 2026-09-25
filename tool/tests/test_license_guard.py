"""License-regression guard: this tool is offered for commercial/public
distribution, so copyleft (AGPL/GPL) or otherwise non-permissive code in the
shipped core is a blocker. Fails if any tracked module reintroduces:

  * AGPL: ``import ultralytics`` / ``from ultralytics`` — tracked set is EMPTY.
    (Ultralytics loads YOLO11n/FastSAM/MobileSAM under AGPL-3.0; this tool
    uses YOLOX (Apache-2.0, onnxruntime-only) and SAM2 (Apache-2.0, loaded
    directly via the `sam2` package) instead.)
  * GPL-RVM: the upstream model (RobustVideoMatting) or its GPL-derived ONNX
    weight (rvm_mobilenetv3). Bare "RVM" in prose is intentionally NOT a
    marker, so docs recording the exclusion don't trip it.
  * A core file importing `plugins_noncommercial/` — see
    docs/LICENSE_POLICY.md. That directory is where non-commercially-licensed
    experiments are meant to live, physically separate from the
    commercial-use-clean core; a core import of it would silently break that
    guarantee.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_TRACKED_AGPL_IMPORTERS: set[str] = set()

_AGPL_RE = re.compile(r"^\s*(?:import\s+ultralytics\b|from\s+ultralytics\b)", re.M)
# Markers are built from fragments so THIS guard file is not flagged by its own scan.
_GPL_RVM_RE = re.compile("Robust" "VideoMatting" "|" "rvm_" "mobilenetv3")
_PLUGINS_DIR = "plugins_noncommercial"
_PLUGIN_IMPORT_RE = re.compile(
    rf"^\s*(?:import\s+{_PLUGINS_DIR}\b|from\s+{_PLUGINS_DIR}\b)", re.M
)

_SELF_REL = Path(__file__).resolve().relative_to(REPO).as_posix()


def _tracked_py_files():
    """Enumerate .py files this guard should scan. Prefers `git ls-files`
    (only scans committed files, so an untracked scratch script can't
    false-positive a failure); falls back to a plain filesystem walk if
    there's no git repo yet (e.g. before this project's first commit)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.py"], cwd=REPO,
            capture_output=True, text=True, check=True,
        ).stdout
        rels = out.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        rels = [
            p.relative_to(REPO).as_posix() for p in REPO.rglob("*.py")
            if not (set(p.relative_to(REPO).parts) & {"__pycache__", ".venv", "venv"})
        ]
    for rel in rels:
        if rel == _SELF_REL:
            continue
        p = REPO / rel
        if p.is_file():
            yield rel, p


def test_ultralytics_import_set_is_locked():
    found = {
        rel
        for rel, p in _tracked_py_files()
        if _AGPL_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
    }
    assert found == _TRACKED_AGPL_IMPORTERS, (
        "ultralytics (AGPL-3.0) import set changed.\n"
        f"  found:    {sorted(found)}\n"
        f"  expected: {sorted(_TRACKED_AGPL_IMPORTERS)}\n"
        "ADDED an import? Stop — ultralytics is AGPL and a distribution blocker.\n"
        "REMOVED it? Keep _TRACKED_AGPL_IMPORTERS empty."
    )


def test_no_gpl_rvm_code_tracked():
    found = sorted(
        rel
        for rel, p in _tracked_py_files()
        if _GPL_RVM_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
    )
    assert not found, (
        "GPL-RVM markers (RobustVideoMatting / the rvm_mobilenetv3 weight) found "
        f"in tracked .py:\n  {found}\n"
        "RVM is GPL-3.0. Delete the code (do not park it in an archive/ dir) and "
        "drop any weight provisioning of it."
    )


def test_core_does_not_import_plugins_noncommercial():
    """Files under plugins_noncommercial/ may import each other freely (that's
    what the directory is for); everything else — the commercial-use-clean
    core — must never import from it. See plugins_noncommercial/README.md and
    docs/LICENSE_POLICY.md."""
    found = sorted(
        rel
        for rel, p in _tracked_py_files()
        if not rel.startswith(f"{_PLUGINS_DIR}/")
        and _PLUGIN_IMPORT_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
    )
    assert not found, (
        f"core file(s) importing {_PLUGINS_DIR}/ — this breaks the "
        "commercial-use-clean guarantee for the core (see docs/LICENSE_POLICY.md):\n"
        f"  {found}"
    )

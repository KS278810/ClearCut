# plugins_noncommercial/

This directory exists to hold anything under a non-commercial or otherwise
non-permissive license (research-only model weights, restricted-use
datasets, etc.) that gets tried out in the future — kept physically separate
from the core so the core's Apache-2.0/MIT/BSD-only guarantee is never at
risk of quietly picking up a dependency it shouldn't.

**Rules for anything placed here:**

1. Nothing in this directory may be imported by the core (`matte_core.py`,
   `ffmpeg_encoders.py`, `pipeline/`, `qc/`).
   `tests/test_license_guard.py::test_core_does_not_import_plugins_noncommercial`
   enforces this — it fails the build if a core file ever imports from here.
2. Every subdirectory must carry its own `LICENSE`/`NOTICE` file stating the
   exact license of whatever it wraps, and a one-line note in this README
   pointing at it.
3. Nothing here is installed or loaded by `requirements-cpu.txt` /
   `requirements-gpu.txt` — if a plugin needs extra packages, they go in a
   `requirements.txt` local to its own subdirectory, invoked manually.
4. Never redistribute restricted-use datasets/weights directly — document
   how to fetch them (a script or a link), same pattern as
   `docs/MODEL_WEIGHTS.md` uses for the core's own checkpoints.

**Currently empty.** See [`docs/LICENSE_POLICY.md`](../docs/LICENSE_POLICY.md)
for the full policy and [`docs/DECISIONS.md`](../docs/DECISIONS.md) for
research candidates that would land here if ever implemented (e.g. MatAnyone,
SAM2Matting, RMBG-2.0 — all found non-commercially licensed and not currently
implemented anywhere in this repo).

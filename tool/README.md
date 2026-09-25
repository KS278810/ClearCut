# 背景除去 (bg_remove)

Video matting: per-frame BiRefNet alpha (a general salient-object
segmenter, not tied to any particular subject type or content class),
motion-compensated temporal smoothing, and a supersampled GIF encoder, run
via `python -m tool.pipeline`. CPU-capable but GPU (`--device cuda`) is the
realistic mode for video-length clips. Most validated and highest-quality
on a flat-colour (chroma-key) backdrop -- the pipeline auto-detects this at
upload time and disables its colour-dependent optional stages otherwise --
but natural/live-action backgrounds work too, with a known limitation
documented in the top-level README's "対象素材と限界".

Permissive-only stack — no AGPL/GPL anywhere (see
[`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md) and
[`docs/LICENSE_POLICY.md`](docs/LICENSE_POLICY.md)):

- **YOLOX-S** (Apache-2.0) — subject box detection, always CPU (tiny)
- **BiRefNet_lite** (MIT) — high-resolution alpha matte
- **pymatting** (MIT) — closed-form matting edge refine + foreground color
  estimation (despill)

`matte_core.py` (the shared model wrapper both this pipeline and any future
tool build on) also carries SAM2 Hiera-Tiny (Apache-2.0) support for a
box-prompt region gate, but `tool/pipeline/` explicitly disables it
(`_load_models` in `runner.py` sets `use_sam2 = False`) — this pipeline's
content (flat chroma backdrop) never needs it. See "GPU internals" below if
a future general-purpose (real-footage) entry point re-enables that path.

## Install

```
pip install -r requirements-cpu.txt   # no GPU — no torch/sam2 needed at all
# or
pip install -r requirements-gpu.txt   # NVIDIA GPU (CUDA driver already installed)
```

`requirements-gpu.txt` also pulls in `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` —
on Windows, `onnxruntime-gpu`'s CUDAExecutionProvider cannot find these on the
default DLL search path otherwise (`matte_core.py` prepends their directories
to `PATH` at import time, so no manual PATH edit is needed). On Linux, set
`LD_LIBRARY_PATH` to include the venv's `nvidia/{cudnn,cublas}/lib` dirs
before running if the CUDA provider fails to load (`matte_core.py` only
handles the Windows DLL-search-path case automatically).

ffmpeg must be on `PATH` (for GIF/MOV output). Use an **LGPL build**, e.g.
BtbN's `ffmpeg-master-latest-*-lgpl-shared`
(github.com/BtbN/FFmpeg-Builds/releases) — see
[`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md) for why.

Model weights go in `checkpoints/` — see
[`docs/MODEL_WEIGHTS.md`](docs/MODEL_WEIGHTS.md) for what's needed and where to
get each file.

## Usage

```
python -m tool.pipeline clip1.mp4 clip2.mp4 outdir/
python -m tool.pipeline clip.mp4 outdir/ --device cuda --qc
```

Outputs `outdir/<name>_matte.gif` (transparent GIF, `ss_alpha_gif` encoder by
default — single-pass palette quantization with a numpy-supersampled alpha
channel; the older 4x-supersampled two-pass `supersampled_gif` encoder is
still available via `--encoder supersampled_gif` but is no longer the
default, adopted 2026-09-22 after a multi-clip speed/quality experiment, see
`docs/DECISIONS.md`) plus, with `--qc`, a fidelity/stability table and a
worst-frame contact-sheet PNG next to each output. On a flat-colour backdrop
the pipeline instead auto-selects a colour-only keyer (`tool/pipeline/
keyer.py`, no neural net) that is faster still and bypasses this encoder
choice's speed tradeoff entirely — see the top-level README's "単色背景の
高速パス". Run `python -m tool.pipeline --help`
for the full flag list (encoder choice, `--scale`/`--max-side`,
`--use-trimap`, `--no-despill`, `--parallel-encode`, etc.) — every
experimental flag's docstring explains what it changed and why it defaults
the way it does, including levers that were tested and rejected (kept as
opt-in for future re-verification against a different asset, not because
they're currently recommended).

### GPU internals: why SAM2 runs in its own subprocess

Running SAM2 (torch/CUDA) and BiRefNet (onnxruntime/CUDA) in the *same*
process was confirmed by testing to make both collapse 10-30x once they
alternate GPU calls — an onnxruntime/torch CUDA-context interaction, not a
bug in either library. `matte_core.py`'s `Models` therefore runs SAM2 in a
dedicated child process (`_Sam2Worker`) and talks to it over a pipe. This
only matters if something re-enables SAM2 (`tool/pipeline/` does not, see
above); it stays documented here because `matte_core.py` is shared
infrastructure a future general-purpose (person/real-footage) entry point
would build on.

Full research notes and every model/backend/optimization considered and
rejected (or adopted) along the way live in
[`docs/DECISIONS.md`](docs/DECISIONS.md) — not needed to use the tool, kept
so a future session doesn't re-derive the same dead ends.

## Quality / regression testing

```
pytest tool/tests/                              # regression gates + license guard
python -m tool.qc.harness --config <name> --fixture-set chroma \
    --output-dir results_dinosaur --out /tmp/qc.json   # ad-hoc metric run
```

`tool/qc/metrics.py` defines the fidelity (F1/F1i/F2/F3), stability
(S1-S4), and edge/fringe (E1/E2) metrics; `tool/tests/test_regression_gates.py`
locks in known-bug thresholds against the shipped `results_dinosaur/`
deliverable so a future change can't silently reintroduce them.

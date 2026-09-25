# Third-Party Licenses

This tool is built entirely on permissively-licensed components (Apache-2.0 /
MIT / BSD-3-Clause). No AGPL or GPL code or model weight is used — see
`tests/test_license_guard.py`, which fails CI if one is reintroduced.

## Model weights

| File | Model | License | Source |
|---|---|---|---|
| `checkpoints/yolox_s.onnx` | YOLOX-S (Megvii) | Apache-2.0 | github.com/Megvii-BaseDetection/YOLOX — exported from the official `yolox_s.pth` release with `tools/export_onnx.py --decode_in_inference`; see `docs/MODEL_WEIGHTS.md`. |
| `checkpoints/sam2.1_hiera_tiny.pt` | SAM2 Hiera-Tiny (Meta) | Apache-2.0 | github.com/facebookresearch/sam2 |
| `checkpoints/birefnet_lite.onnx` | BiRefNet (lite variant) | MIT | github.com/ZhengPeng7/BiRefNet |

## Python packages

| Package | License |
|---|---|
| onnxruntime / onnxruntime-gpu | MIT |
| torch / torchvision | BSD-3-Clause |
| sam2 | Apache-2.0 |
| opencv-python-headless | Apache-2.0 |
| numpy / scipy | BSD-3-Clause |
| pymatting | MIT |
| Pillow | HPND (MIT/BSD-style) |

## External binary

ffmpeg is not bundled or installed by this project. Use a system ffmpeg on
`PATH` — an **LGPL build** is required for redistribution/commercial use, e.g.
BtbN's `ffmpeg-master-latest-*-lgpl-shared`
(github.com/BtbN/FFmpeg-Builds/releases). This pipeline only encodes ProRes
4444 (MOV) and GIF, neither of which needs a GPL-only codec (libx264 etc.), so
an LGPL build loses no functionality here. Do not install `imageio-ffmpeg`
instead — its auto-downloaded build is GPL.

## GPU runtime (optional, `requirements-gpu.txt` only)

CUDA/cuDNN are provided by NVIDIA under the NVIDIA EULA and are **not bundled
or redistributed** by this project; `onnxruntime-gpu`/`torch` wheels merely
link against a CUDA toolkit/driver already present on the machine running this
tool.

## Explicitly excluded (do not reintroduce)

| Component | License | Why excluded |
|---|---|---|
| RVM (RobustVideoMatting) | GPL-3.0 | Copyleft; superseded by BiRefNet. |
| Ultralytics (YOLO11n / FastSAM / MobileSAM loader) | AGPL-3.0 | Copyleft that triggers on network/SaaS use; superseded by YOLOX (Apache-2.0) + SAM2 (Apache-2.0) direct-loaded, not via `ultralytics`. |

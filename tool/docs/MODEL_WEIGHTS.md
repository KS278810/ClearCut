# Model Weights

`checkpoints/` is gitignored — none of these files are committed. Fetch/build
each one as described below.

## `birefnet_lite.onnx` — BiRefNet (lite variant)

- **License**: MIT — github.com/ZhengPeng7/BiRefNet
- ~214 MB. Copied from the sibling project's working checkpoint on 2026-08-10.
- **Known provenance gap** (inherited, unresolved): the exact HuggingFace
  source repo/revision and export recipe that produced this exact binary are
  not recorded anywhere. If you need to reproduce it from scratch, export the
  lite BiRefNet checkpoint to ONNX yourself (1024x1024 input, sigmoid applied
  only if raw output falls outside [0,1] — see `matte_core.py:_brf_1024`).

## `sam2.1_hiera_tiny.pt` — SAM2 (Segment Anything Model 2), Hiera-Tiny

- **License**: Apache-2.0 — github.com/facebookresearch/sam2
- ~149 MB, official release `092824`.
- Fetch: `wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`
- Loaded via the `sam2` PyPI package directly (`sam2.build_sam.build_sam2`),
  never through `ultralytics` (which would pull in AGPL-3.0 code).

## `yolox_s.onnx` — YOLOX-S detector (COCO)

- **License**: Apache-2.0 — github.com/Megvii-BaseDetection/YOLOX
- ~35 MB, 640x640 input, decode-in-inference baked in (output `[1, 8400, 85]`).
- Megvii does **not** publish an ONNX release asset directly — only the
  PyTorch `.pth` checkpoints (see the README's benchmark table, e.g.
  `releases/download/0.1.1rc0/yolox_s.pth`). Build the ONNX yourself:

  ```
  git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git
  cd YOLOX && pip install -e . --no-deps
  pip install loguru onnx onnx-simplifier==0.4.10 tabulate torch torchvision opencv-python-headless
  wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
  python tools/export_onnx.py --output-name yolox_s.onnx -f exps/default/yolox_s.py \
      -c yolox_s.pth --decode_in_inference
  ```

  `--decode_in_inference` is required — without it the grid decode isn't baked
  into the graph and `matte_core.py`'s `_OnnxYOLO.detect()` (which expects the
  already-decoded `[1, 8400, 85]` output) will misinterpret the raw output.
- Do this in an isolated virtualenv — the YOLOX export tooling pulls in torch,
  torchvision, and onnx-simplifier, none of which the runtime otherwise needs
  (the shipped `_OnnxYOLO` detector is onnxruntime-only, no torch).
- The checked-in copy was built this way on 2026-08-10 (torch 1.12.1+cu116,
  opset 11, onnxsim-simplified): 35,974,565 bytes,
  sha256 `22962a61a5b43506de580bbbb4f2a2ae3db016a84ab8cea6f094c30a01f57b02`.
  Verified: input `images` `[1,3,640,640]` float, output `output` `[1,8400,85]`
  float (decode-in-inference baked in), matching `_OnnxYOLO`'s contract.

## Explicitly excluded (do not add back)

RVM (`rvm_mobilenetv3_fp32.onnx`, GPL-3.0) and any Ultralytics-loaded weight
(`yolo11n.*`, `FastSAM-s.pt`, `mobile_sam.pt` via `ultralytics.SAM()`,
AGPL-3.0) are not used by any code here — see
`tests/test_license_guard.py`.

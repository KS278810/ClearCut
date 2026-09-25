# ClearCut Web版

ブラウザ完結の背景除去(ONNX Runtime Web)。YOLOX-S検出 + BiRefNet_lite(512²、
自前再エクスポート)+ despill(色にじみ除去)。

端末にWebGPUがあれば自動利用し、UIのバッジでCPU/GPU切替も可能。BiRefNetは
ONNX Runtime Web側の制約(`microsoft/onnxruntime#21968`)で現状CPU実行に
なることが多く、GPUが安定して効くのは検出(YOLOX)のみ。CPU実行時の目安は
約3.5秒/フレーム。

## 使い方

1. `models/`に`birefnet_lite_512.onnx`・`yolox_s.onnx`を配置
   (変換手順は[`docs/MODEL_CONVERSION.md`](docs/MODEL_CONVERSION.md))
2. ローカルサーバーで`index.html`を開く(`file://`直開きはWorker/WASMの制約で不可。
   例: `python3 -m http.server` をこのディレクトリで実行)

`src/`はビルド済み配布物です(難読化ビルド、ソースは非公開)。`vendor/`は
ONNX Runtime Web本体で無加工同梱。ライセンスはリポジトリルートの[`LICENSE`](../LICENSE)を参照。

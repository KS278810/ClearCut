# ClearCut

動画・画像の背景除去ツール。単色背景はクロマキー高速パス、それ以外は
YOLOX-S検出+BiRefNet_liteによるAIマッティング(オプションでSAM2による
主被写体フィルタ、既定OFF)。

## GPU版(サーバー)

FastAPI製のバッチ処理サーバー。社内GPUマシン上でTailscale経由でのみ運用しており、
このリポジトリにあるのはソースコードのみで、動作するデプロイ先はありません。

起動: `./serve.sh`(要CUDA環境・モデル重み。入手方法は
[`tool/docs/MODEL_WEIGHTS.md`](tool/docs/MODEL_WEIGHTS.md))
CLI版: `./run.sh <clip.mp4> <outdir> [--device cuda]`(詳細は
[`tool/README.md`](tool/README.md))

## Web版(ブラウザ、`web/`)

ONNX Runtime Webによる、ブラウザ完結の軽量版。端末にWebGPUがあれば自動利用し、
UIのバッジでCPU/GPU切替も可能。BiRefNetはONNX Runtime Web側の制約
(`microsoft/onnxruntime#21968`)で現状CPU実行になることが多く、GPUが安定して
効くのは検出(YOLOX)のみ。CPU実行時の目安は約3.5秒/フレーム。

モデル重みは同梱していないため、`web/models/`に`birefnet_lite_512.onnx`・
`yolox_s.onnx`を配置してから`web/index.html`をローカルサーバーで開いて動かして
ください(`file://`直開きはWorker/WASMの制約で不可)。

## このリポジトリについて

ビルド済みの配布物のみを含みます(`web/`のJSは難読化ビルド)。ソースは非公開
(`ClearCut-dev`)で、履歴は持たず更新のたびに単一コミットで差し替えます。

## ライセンス

CC BY-NC 4.0(表示・非営利)。教育・研究での利用は自由です。商用利用をご希望の場合は
[Issues](https://github.com/KS278810/ClearCut/issues)からご相談ください。依存ライブラリの
ライセンス内訳は[`tool/docs/THIRD_PARTY_LICENSES.md`](tool/docs/THIRD_PARTY_LICENSES.md)を参照。

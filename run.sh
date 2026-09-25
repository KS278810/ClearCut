#!/bin/bash
# コマンドライン版の起動スクリプト: python -m tool.pipeline をこのvenvと
# CUDAライブラリパスを設定した状態で実行する。
#
#   ./run.sh <clip.mp4> [more.mp4 ...] <outdir> [--device cuda] [--qc] ...
#
# 全オプションは `./run.sh --help` を参照(tool/pipeline/__main__.pyの
# argparseがそのまま反映される)。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_NVIDIA="$PWD/venv/lib/python3.12/site-packages/nvidia"
# onnxruntimeのCUDAExecutionProviderが必要とする共有ライブラリをこの
# venvのnvidia-*-cu12パッケージから見つけられるようにする。Linuxではプロ
# セス起動前に設定しないと反映されない(matte_core.pyの自動DLLパス解決は
# Windows専用)。cudnn/cublas以外(cufft/curand/cuda_runtime/nvjitlink/
# cuda_nvrtc)もこのマシンではシステムのCUDA 12ツールキットが偶然解決して
# いるだけで、クリーンな環境では見つからない -- venv自身のものを明示する。
export LD_LIBRARY_PATH="$VENV_NVIDIA/cudnn/lib:$VENV_NVIDIA/cublas/lib:$VENV_NVIDIA/cufft/lib:$VENV_NVIDIA/curand/lib:$VENV_NVIDIA/cuda_runtime/lib:$VENV_NVIDIA/nvjitlink/lib:$VENV_NVIDIA/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

exec venv/bin/python3 -m tool.pipeline "$@"

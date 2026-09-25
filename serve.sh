#!/bin/bash
# HeroExtractor サーバー版の起動スクリプト: FastAPI(server/app.py)を
# uvicornで、このvenvとCUDAライブラリパスを設定した状態、かつ
# Tailscale IPで待ち受けて起動する(社内ネットワークのみ、0.0.0.0では
# 待ち受けない)。
#
# set -o pipefail が無いと `tailscale ip -4 | head -1` の終了コードは
# head のものになり、tailscale 自体が失敗(未インストール/未接続)しても
# $() は空文字列を返すだけで set -e に引っかからない。HOST="" のまま
# uvicorn --host "" を渡すと 0.0.0.0(全インターフェース)で待ち受けて
# しまう -- このコメントが「しない」と書いていた挙動そのもの。
# HOST 環境変数で明示的に上書きできるようにしておく(tailscale が
# 使えない環境や、複数IPを持つ機での固定用途)。
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

if [ -z "${HOST:-}" ]; then
    HOST="$(tailscale ip -4 2>/dev/null | head -1 || true)"
fi
if [ -z "$HOST" ]; then
    echo "serve.sh: tailscale IP が取得できません(tailscaleが未接続/未インストール)。" >&2
    echo "  HOST=<待ち受けIP> ./serve.sh のように明示してください。0.0.0.0では待ち受けません。" >&2
    exit 1
fi
PORT="${PORT:-7863}"

exec venv/bin/python3 -m uvicorn server.app:app --host "$HOST" --port "$PORT" --workers 1

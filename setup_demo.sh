#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/.demo"
VENV_DIR="$RUNTIME_DIR/venv"
MODELS_DIR="$RUNTIME_DIR/models"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"
WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-large-v3-turbo}"
WHISPER_MODEL_REPO="${WHISPER_MODEL_REPO:-mlx-community/whisper-large-v3-turbo}"
WHISPER_MODEL_DIR="${WHISPER_MODEL_PATH:-$MODELS_DIR/whisper-$WHISPER_MODEL_NAME}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "오류: MLX Whisper는 Apple Silicon Mac이 필요합니다."
  exit 1
fi

mkdir -p "$RUNTIME_DIR" "$MODELS_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/backend/requirements.txt"

if [[ "${ENABLE_DEEPFILTER:-true}" =~ ^(1|true|yes)$ ]]; then
  "$VENV_DIR/bin/python" <<'PY'
from df.enhance import init_df

init_df(
    model_base_dir="DeepFilterNet3",
    log_level="ERROR",
    log_file=None,
)
PY
fi

if [[ ! -f "$MODELS_DIR/silero_vad.onnx" ]]; then
  curl --fail --location \
    --output "$MODELS_DIR/silero_vad.onnx" \
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
fi

if [[ ! -f "$WHISPER_MODEL_DIR/config.json" || ! -f "$WHISPER_MODEL_DIR/weights.safetensors" ]]; then
  "$VENV_DIR/bin/python" - "$WHISPER_MODEL_REPO" "$WHISPER_MODEL_DIR" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
)
PY
fi

echo "EasyListner MLX 데모 환경 준비가 완료되었습니다."
echo "모델: $WHISPER_MODEL_NAME"
echo "모델 경로: $WHISPER_MODEL_DIR"

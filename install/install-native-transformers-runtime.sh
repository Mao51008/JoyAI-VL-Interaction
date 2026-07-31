#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$INSTALL_DIR/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
MAIN_VENV_DIR="${MAIN_VENV_DIR:-$REPO_ROOT/services/.venv-cu124}"
ASR_VENV_DIR="${ASR_VENV_DIR:-$REPO_ROOT/services/asr/.venv-cu124}"
TTS_VENV_DIR="${TTS_VENV_DIR:-$REPO_ROOT/services/tts/.venv-cu124}"

create_venv() {
  local path="$1"
  "$UV_BIN" venv --python "$PYTHON_BIN" --seed "$path"
}

install_main() {
  create_venv "$MAIN_VENV_DIR"
  "$UV_BIN" pip install --python "$MAIN_VENV_DIR/bin/python" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --torch-backend "$TORCH_BACKEND"
  "$UV_BIN" pip install --python "$MAIN_VENV_DIR/bin/python" \
    "transformers==5.14.1" accelerate safetensors pillow qwen-vl-utils \
    fastapi "uvicorn[standard]"
}

install_asr() {
  create_venv "$ASR_VENV_DIR"
  "$UV_BIN" pip install --python "$ASR_VENV_DIR/bin/python" \
    qwen-asr -e "$REPO_ROOT/services/asr[dev]" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    "numpy==1.26.4" "numba==0.61.2" "llvmlite==0.44.0" \
    --torch-backend "$TORCH_BACKEND"
}

install_tts() {
  create_venv "$TTS_VENV_DIR"
  "$UV_BIN" pip install --python "$TTS_VENV_DIR/bin/python" \
    qwen-tts -e "$REPO_ROOT/services/tts[dev]" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    "numpy==1.26.4" "numba==0.61.2" "llvmlite==0.44.0" \
    --torch-backend "$TORCH_BACKEND"
}

install_main
install_asr
install_tts

echo "Native Transformers runtimes installed:"
echo "  main: $MAIN_VENV_DIR"
echo "  ASR:  $ASR_VENV_DIR"
echo "  TTS:  $TTS_VENV_DIR"

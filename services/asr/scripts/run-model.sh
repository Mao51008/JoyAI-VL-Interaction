#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$SERVICE_DIR/../.." && pwd)"
INSTALL_DIR="$REPO_ROOT/install"
ASR_BACKEND="${ASR_BACKEND:-vllm}"
ASR_VENV_DIR="${ASR_VENV_DIR:-}"
if [[ -z "$ASR_VENV_DIR" ]]; then
  if [[ "$ASR_BACKEND" == "transformers" && -d "$SERVICE_DIR/.venv-cu124" ]]; then
    ASR_VENV_DIR="$SERVICE_DIR/.venv-cu124"
  else
    ASR_VENV_DIR="$SERVICE_DIR/.venv"
  fi
fi
MODEL_ROOT="${MODEL_ROOT:-/tmp/models}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-$MODEL_ROOT/Qwen3-ASR-1.7B}"
ASR_MODEL_NAME="${ASR_MODEL_NAME:-Qwen/Qwen3-ASR-1.7B}"
ASR_HOST="${ASR_HOST:-0.0.0.0}"
ASR_PORT="${ASR_PORT:-8993}"
ASR_GPU="${ASR_GPU:-2}"
ASR_TENSOR_PARALLEL_SIZE="${ASR_TENSOR_PARALLEL_SIZE:-1}"
ASR_GPU_MEMORY_UTILIZATION="${ASR_GPU_MEMORY_UTILIZATION:-0.3}"
ASR_MAX_MODEL_LEN="${ASR_MAX_MODEL_LEN:-8192}"
ASR_MODEL_LOG_FILE=""
if [[ "${SAVE_SERVICE_LOGS:-0}" == "1" ]]; then
  ASR_LOG_DIR="${ASR_LOG_DIR:-${SERVICE_DIR}/logs}"
  mkdir -p "$ASR_LOG_DIR"
  ASR_MODEL_LOG_FILE="${ASR_MODEL_LOG_FILE:-${ASR_LOG_DIR}/model_${ASR_PORT}.log}"
  exec > >(tee -a "$ASR_MODEL_LOG_FILE") 2>&1
fi

if [[ -n "${ASR_MODEL_LOG_FILE}" ]]; then
  echo "ASR model log: $ASR_MODEL_LOG_FILE"
fi

if [ ! -d "$ASR_VENV_DIR" ]; then
  echo "ASR 虚拟环境不存在: $ASR_VENV_DIR" >&2
  echo "请先运行 $INSTALL_DIR/install-audio-runtime.sh --asr" >&2
  exit 1
fi

if [ ! -d "$ASR_MODEL_DIR" ] || [ -z "$(find "$ASR_MODEL_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ASR 模型目录不存在或为空: $ASR_MODEL_DIR" >&2
  echo "请先运行 $INSTALL_DIR/download-models.sh --all" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$ASR_VENV_DIR/bin/activate"
if [[ "$ASR_BACKEND" == "transformers" ]]; then
  exec env \
    CUDA_VISIBLE_DEVICES="$ASR_GPU" \
    ASR_MODEL_DIR="$ASR_MODEL_DIR" \
    ASR_MODEL_NAME="$ASR_MODEL_NAME" \
    ASR_HOST="$ASR_HOST" \
    ASR_PORT="$ASR_PORT" \
    python "$SERVICE_DIR/native_server.py"
elif [[ "$ASR_BACKEND" != "vllm" ]]; then
  echo "Unsupported ASR_BACKEND: $ASR_BACKEND; expected vllm or transformers" >&2
  exit 2
elif command -v qwen-asr-serve >/dev/null 2>&1; then
  ASR_SERVER_COMMAND=(qwen-asr-serve)
else
  ASR_SERVER_COMMAND=(vllm serve)
fi

exec env CUDA_VISIBLE_DEVICES="$ASR_GPU" "${ASR_SERVER_COMMAND[@]}" "$ASR_MODEL_DIR" \
  --served-model-name "$ASR_MODEL_NAME" \
  --host "$ASR_HOST" \
  --port "$ASR_PORT" \
  --tensor-parallel-size "$ASR_TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$ASR_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$ASR_MAX_MODEL_LEN" \
  --trust-remote-code

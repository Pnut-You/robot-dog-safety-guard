#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# FlashInfer's sampler may try to JIT-compile CUDA code and require a system
# CUDA Toolkit even when the vLLM wheel already contains all runtime libraries.
# The standard PyTorch sampler is sufficient for deterministic classification.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL_KEY=""
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--model" || -z "$2" ]]; then
    echo "用法: bash scripts/start_vllm.sh [--model <配置键>]" >&2
    exit 2
  fi
  MODEL_KEY="$2"
fi

mapfile -t MODEL_CONFIG < <(uv run python - "$MODEL_KEY" <<'PY'
import sys
from app.config import get_model_config
c = get_model_config(sys.argv[1] or None)
for value in (c.resolved_local_path, c.served_model_name, c.host, c.port, c.effective_api_key,
              c.dtype, c.max_model_len, c.gpu_memory_utilization, int(c.enforce_eager)):
    print(value)
PY
)

if [[ ${#MODEL_CONFIG[@]} -ne 9 ]]; then
  echo "错误: 无法读取完整模型配置。" >&2
  exit 1
fi

MODEL_PATH="${MODEL_CONFIG[0]}"
SERVED_NAME="${MODEL_CONFIG[1]}"
HOST="${MODEL_CONFIG[2]}"
PORT="${MODEL_CONFIG[3]}"
API_KEY="${MODEL_CONFIG[4]}"
DTYPE="${MODEL_CONFIG[5]}"
MAX_MODEL_LEN="${MODEL_CONFIG[6]}"
GPU_MEMORY="${MODEL_CONFIG[7]}"
ENFORCE_EAGER="${MODEL_CONFIG[8]}"

if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(find "$MODEL_PATH" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "错误: 本地模型不存在或目录为空: $MODEL_PATH" >&2
  echo "请先运行: uv run python scripts/download_model.py" >&2
  exit 1
fi
if ! uv run vllm --help >/dev/null 2>&1; then
  echo "错误: uv 环境中找不到 vllm 命令，请先运行 uv sync。" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "错误: 未检测到可用 NVIDIA GPU/驱动。" >&2
  exit 1
fi
if ! uv run python - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
except OSError as exc:
    print(f"错误: {host}:{port} 端口不可用: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PY
then
  exit 1
fi

VLLM_ARGS=(
  serve "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY"
)
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi

exec uv run vllm "${VLLM_ARGS[@]}"

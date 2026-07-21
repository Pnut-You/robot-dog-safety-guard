#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="$PROJECT_ROOT/.run"
PID_FILE="$RUN_DIR/start_dev.pid"
MODEL_FILE="$RUN_DIR/start_dev.model"
UI_PORT=8501
ACTION="start"
MODEL_KEY=""

usage() {
  cat <<'EOF'
用法:
  bash scripts/start_dev.sh [start] [--model <配置键>]
  bash scripts/start_dev.sh switch --model <配置键>
  bash scripts/start_dev.sh restart [--model <配置键>]
  bash scripts/start_dev.sh status
  bash scripts/start_dev.sh stop

省略子命令时等同于 start；start、restart 和 switch 都会优雅停止本项目的现有实例。
switch 必须明确指定目标模型，并在停止旧服务前验证新模型完整性。
EOF
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  ACTION="$1"
  shift
fi
case "$ACTION" in
  start|restart|switch)
    if [[ $# -gt 0 ]]; then
      if [[ $# -ne 2 || "$1" != "--model" || -z "$2" ]]; then
        usage >&2
        exit 2
      fi
      MODEL_KEY="$2"
    fi
    if [[ "$ACTION" == "switch" && -z "$MODEL_KEY" ]]; then
      echo "错误: switch 必须使用 --model <配置键> 指定目标模型。" >&2
      usage >&2
      exit 2
    fi
    ;;
  status|stop)
    if [[ $# -ne 0 ]]; then
      usage >&2
      exit 2
    fi
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR"

read_supervisor_pid() {
  local pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && is_our_supervisor "$pid"; then
      printf '%s\n' "$pid"
      return 0
    fi
  fi

  # One-time migration path for supervisors started before PID tracking existed.
  while read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] || continue
    if is_our_supervisor "$pid"; then
      printf '%s\n' "$pid"
      return 0
    fi
  done < <(pgrep -f '(^|/)bash scripts/start_dev\.sh( |$)' 2>/dev/null || true)
  return 1
}

is_our_supervisor() {
  local pid="$1" cmd cwd
  [[ -d "/proc/$pid" ]] || return 1
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  [[ "$cmd" =~ (^|/)(ba)?sh[[:space:]]+scripts/start_dev\.sh([[:space:]]|$) && "$cwd" == "$PROJECT_ROOT" ]]
}

clear_stale_state() {
  rm -f "$PID_FILE" "$MODEL_FILE"
}

show_port_owner() {
  local port="$1"
  echo "端口 $port 的监听信息：" >&2
  ss -ltnp "( sport = :$port )" >&2 2>/dev/null || true
}

port_is_free() {
  local port="$1"
  uv run python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

service_status() {
  local pid model="未知"
  if ! pid="$(read_supervisor_pid)" || ! is_our_supervisor "$pid"; then
    clear_stale_state
    echo "状态: 未运行"
    return 1
  fi
  [[ -f "$MODEL_FILE" ]] && model="$(<"$MODEL_FILE")"
  echo "状态: 正在运行"
  echo "监管进程: $pid"
  echo "模型: $model"
  if curl --noproxy '*' -fsS "http://127.0.0.1:$UI_PORT/_stcore/health" >/dev/null 2>&1; then
    echo "Streamlit: 正常 (http://localhost:$UI_PORT)"
  else
    echo "Streamlit: 异常或正在恢复"
  fi
  if uv run python scripts/check_server.py --model "$model" >/dev/null 2>&1; then
    echo "vLLM: 正常"
  else
    echo "vLLM: 异常或正在启动"
  fi
}

stop_service() {
  local quiet="${1:-false}" pid
  if ! pid="$(read_supervisor_pid)" || ! is_our_supervisor "$pid"; then
    clear_stale_state
    [[ "$quiet" == "true" ]] || echo "服务未运行。"
    return 0
  fi

  echo "正在停止现有服务（监管进程: $pid）..."
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 25); do
    if ! is_our_supervisor "$pid"; then
      clear_stale_state
      echo "现有服务已停止。"
      return 0
    fi
    sleep 1
  done

  echo "警告: 优雅停止超时，发送 SIGTERM。" >&2
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if ! is_our_supervisor "$pid"; then
      clear_stale_state
      echo "现有服务已停止。"
      return 0
    fi
    sleep 1
  done
  echo "错误: 本项目监管进程 $pid 未能停止，请检查该进程。" >&2
  return 1
}

if [[ "$ACTION" == "status" ]]; then
  service_status
  exit $?
fi
if [[ "$ACTION" == "stop" ]]; then
  stop_service
  exit $?
fi

if [[ -z "$MODEL_KEY" ]]; then
  MODEL_KEY="$(uv run python - <<'PY'
from app.config import load_models_config
print(load_models_config().active_model)
PY
)"
fi

mapfile -t TARGET_CONFIG < <(uv run python - "$MODEL_KEY" <<'PY'
import json
import sys
from app.config import get_model_config
c = get_model_config(sys.argv[1])
p = c.resolved_local_path
errors = []
if not p.is_dir():
    errors.append(f"模型目录不存在: {p}")
elif not (p / "config.json").is_file():
    errors.append(f"模型配置不存在: {p / 'config.json'}")
else:
    index_files = [p / "model.safetensors.index.json", p / "pytorch_model.bin.index.json"]
    index_file = next((item for item in index_files if item.is_file()), None)
    if index_file:
        try:
            names = set(json.loads(index_file.read_text(encoding="utf-8"))["weight_map"].values())
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"无法解析权重索引 {index_file}: {exc}")
        else:
            missing = sorted(name for name in names if not (p / name).is_file())
            if missing:
                errors.append(f"缺少 {len(missing)} 个权重分片: {', '.join(missing[:3])}")
    elif not any(p.glob("*.safetensors")) and not any(p.glob("pytorch_model*.bin")):
        errors.append(f"模型权重不存在: {p}")
if errors:
    print("错误: 目标模型不完整；不会停止当前服务。", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(f"请先运行: uv run python scripts/download_model.py --model {sys.argv[1]}", file=sys.stderr)
    raise SystemExit(1)
print(c.port)
print(c.resolved_local_path)
PY
)
if [[ ${#TARGET_CONFIG[@]} -ne 2 ]]; then
  echo "错误: 目标模型预检失败；不会停止当前服务。" >&2
  exit 1
fi
VLLM_PORT="${TARGET_CONFIG[0]}"
echo "目标模型预检通过: $MODEL_KEY (${TARGET_CONFIG[1]})"

# Validate the target first, then stop only the supervisor owned by this project.
stop_service true

for port in "$VLLM_PORT" "$UI_PORT"; do
  if ! port_is_free "$port"; then
    echo "错误: 端口 $port 被非本启动器管理的进程占用；不会自动终止该进程。" >&2
    show_port_owner "$port"
    exit 1
  fi
done

VLLM_PID=""
UI_PID=""
cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  for pid in "$UI_PID" "$VLLM_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT -- "-$pid" 2>/dev/null || true
    fi
  done
  for _ in $(seq 1 15); do
    local alive=0
    for pid in "$UI_PID" "$VLLM_PID"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    [[ "$alive" -eq 0 ]] && break
    sleep 1
  done
  for pid in "$UI_PID" "$VLLM_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "$UI_PID" "$VLLM_PID"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  if [[ -f "$PID_FILE" && "$(<"$PID_FILE")" == "$$" ]]; then
    clear_stale_state
  fi
  exit "$exit_status"
}
trap cleanup EXIT INT TERM

printf '%s\n' "$$" > "$PID_FILE"
printf '%s\n' "$MODEL_KEY" > "$MODEL_FILE"

echo "启动 vLLM（模型: $MODEL_KEY）..."
setsid bash scripts/start_vllm.sh --model "$MODEL_KEY" &
VLLM_PID=$!

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "错误: vLLM 在就绪前退出。" >&2
    exit 1
  fi
  if uv run python scripts/check_server.py --model "$MODEL_KEY" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "错误: 等待 vLLM 就绪超时。" >&2
  exit 1
fi

start_ui() {
  echo "启动 Streamlit: http://localhost:$UI_PORT"
  setsid env ROVER_MODEL_KEY="$MODEL_KEY" uv run streamlit run app/ui.py \
    --server.port "$UI_PORT" \
    --server.headless true \
    --server.enableWebsocketCompression false \
    --server.websocketPingInterval 30 \
    --browser.gatherUsageStats false &
  UI_PID=$!

  local ui_ready=0
  for _ in $(seq 1 30); do
    if ! kill -0 "$UI_PID" 2>/dev/null; then
      wait "$UI_PID" 2>/dev/null || true
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:$UI_PORT/_stcore/health" >/dev/null 2>&1; then
      ui_ready=1
      break
    fi
    sleep 1
  done
  [[ "$ui_ready" -eq 1 ]]
}

echo "vLLM 已就绪。"
restart_count=0
if ! start_ui; then
  echo "错误: Streamlit 首次启动失败。" >&2
  exit 1
fi
echo "服务已启动。按 Ctrl+C 同时停止 vLLM 和 Streamlit。"

unhealthy_count=0
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    set +e
    wait "$VLLM_PID"
    status=$?
    set -e
    echo "错误: vLLM 意外退出（状态码: $status）。" >&2
    exit "$status"
  fi

  ui_failed=0
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    ui_failed=1
  elif curl --noproxy '*' -fsS "http://127.0.0.1:$UI_PORT/_stcore/health" >/dev/null 2>&1; then
    unhealthy_count=0
  else
    unhealthy_count=$((unhealthy_count + 1))
    if [[ "$unhealthy_count" -ge 3 ]]; then
      echo "警告: Streamlit 连续三次健康检查失败。" >&2
      kill -INT -- "-$UI_PID" 2>/dev/null || true
      ui_failed=1
    fi
  fi

  if [[ "$ui_failed" -eq 1 ]]; then
    set +e
    wait "$UI_PID"
    status=$?
    set -e
    restart_count=$((restart_count + 1))
    if [[ "$restart_count" -gt 5 ]]; then
      echo "错误: Streamlit 连续重启超过 5 次，停止整套服务。" >&2
      exit 1
    fi
    echo "警告: Streamlit 已退出（状态码: $status），2 秒后第 $restart_count 次重启。" >&2
    sleep 2
    if ! start_ui; then
      echo "错误: Streamlit 第 $restart_count 次重启失败。" >&2
      continue
    fi
    unhealthy_count=0
    echo "Streamlit 已恢复: http://localhost:$UI_PORT"
  fi
  sleep 2
done

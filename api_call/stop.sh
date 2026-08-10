#!/usr/bin/env bash
# Stop only the configured model instances. Use --all for every managed PID.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE=""
CLI_MODEL_NAME=""
declare -a CLI_GPU_IDS=()
CLEAN_MODE=false
ALL_MODE=false

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN_MODE=true ;;
    --all) ALL_MODE=true ;;
    *.conf|*.env) CONFIG_FILE="$arg" ;;
    llama|qwen) CLI_MODEL_NAME="$arg" ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        CLI_GPU_IDS+=("$arg")
      fi
      ;;
  esac
done

CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/deploy.conf}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime_config.sh"
load_deployment_runtime "$CONFIG_FILE"

MODEL_NAME="${CLI_MODEL_NAME:-$DEPLOY_SELECTED_MODEL}"
USING_ACTIVE_RUNTIME=false
if [[ ${#CLI_GPU_IDS[@]} -gt 0 ]]; then
  GPU_IDS=("${CLI_GPU_IDS[@]}")
  PORTS=()
  for gpu in "${GPU_IDS[@]}"; do PORTS+=("$((DEPLOY_SELECTED_PORT_BASE + gpu))"); done
  if [[ -f "$SCRIPT_DIR/${LOG_DIR:-logs}/active_deployment.env" ]]; then
    # shellcheck disable=SC1090
    source "$SCRIPT_DIR/${LOG_DIR:-logs}/active_deployment.env"
    if [[ "$MODEL_NAME" == "$ACTIVE_MODEL" &&
          "${GPU_IDS[*]}" == "$ACTIVE_GPU_IDS" &&
          "${PORTS[*]}" == "$ACTIVE_PORTS" ]]; then
      USING_ACTIVE_RUNTIME=true
    fi
  fi
elif [[ -f "$SCRIPT_DIR/${LOG_DIR:-logs}/active_deployment.env" ]]; then
  # shellcheck disable=SC1090
  source "$SCRIPT_DIR/${LOG_DIR:-logs}/active_deployment.env"
  MODEL_NAME="${CLI_MODEL_NAME:-$ACTIVE_MODEL}"
  read -r -a GPU_IDS <<< "$ACTIVE_GPU_IDS"
  read -r -a PORTS <<< "$ACTIVE_PORTS"
  USING_ACTIVE_RUNTIME=true
  echo "[ACTIVE] 使用最近一次实际部署拓扑"
else
  GPU_IDS=("${DEPLOY_SELECTED_GPU_IDS[@]}")
  PORTS=("${DEPLOY_SELECTED_PORTS[@]}")
fi

LOG_DIR_PATH="$SCRIPT_DIR/${LOG_DIR:-logs}"
PID_DIR="$LOG_DIR_PATH/pids"
ACTIVE_RUNTIME_FILE="$LOG_DIR_PATH/active_deployment.env"
declare -a TARGET_PID_FILES=()

if $ALL_MODE; then
  if [[ -d "$PID_DIR" ]]; then
    while IFS= read -r file; do TARGET_PID_FILES+=("$file"); done < <(
      find "$PID_DIR" -maxdepth 1 -type f -name '*.pid' -print | sort
    )
  fi
else
  for index in "${!GPU_IDS[@]}"; do
    TARGET_PID_FILES+=("$PID_DIR/${MODEL_NAME}_gpu${GPU_IDS[$index]}_port${PORTS[$index]}.pid")
  done
fi

echo "================================================================"
echo "  停止 vLLM 服务"
echo "  配置: $CONFIG_FILE"
if $ALL_MODE; then
  echo "  范围: 所有由 api_call/run.sh 管理的 PID"
else
  echo "  模型: $MODEL_NAME"
  echo "  GPU: ${GPU_IDS[*]}"
  echo "  端口: ${PORTS[*]}"
fi
echo "================================================================"

is_managed_vllm_pid() {
  local pid="$1" command_line
  command_line=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ "$command_line" == *"vllm.entrypoints.openai.api_server"* ]]
}

for pid_file in "${TARGET_PID_FILES[@]}"; do
  [[ -f "$pid_file" ]] || continue
  pid=$(<"$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    if is_managed_vllm_pid "$pid"; then
      echo "[STOP] PID $pid ($(basename "$pid_file"))"
      kill "$pid" 2>/dev/null || true
    else
      echo "[WARN] stale PID file points to a non-vLLM process; not killing PID $pid"
    fi
  fi
done

# Give the selected servers a short graceful shutdown window.
for _ in {1..20}; do
  alive=false
  for pid_file in "${TARGET_PID_FILES[@]}"; do
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && is_managed_vllm_pid "$pid"; then alive=true; fi
  done
  $alive || break
  sleep 0.25
done

for pid_file in "${TARGET_PID_FILES[@]}"; do
  [[ -f "$pid_file" ]] || continue
  pid=$(<"$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && is_managed_vllm_pid "$pid"; then
    echo "[KILL] PID $pid did not exit gracefully"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
done

if ! $ALL_MODE; then
  for port in "${PORTS[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
      while IFS= read -r listener_pid; do
        if [[ "$listener_pid" =~ ^[0-9]+$ ]] && is_managed_vllm_pid "$listener_pid"; then
          echo "[KILL] managed vLLM listener PID $listener_pid on port $port"
          kill -9 "$listener_pid" 2>/dev/null || true
        fi
      done < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    fi
  done
fi

if $CLEAN_MODE; then
  echo "[KEEP] --clean 不再删除 vLLM 日志；日志用于超时与请求级故障审计。"
fi

if $USING_ACTIVE_RUNTIME || $ALL_MODE; then
  rm -f "$ACTIVE_RUNTIME_FILE"
fi

failed=0
if ! $ALL_MODE; then
  for port in "${PORTS[@]}"; do
    if curl --silent --max-time 1 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[WARN] port $port is still responding"
      failed=$((failed + 1))
    else
      echo "[OK] port $port stopped"
    fi
  done
fi
exit "$failed"

#!/usr/bin/env bash
# Test configured endpoints. Numeric CLI arguments are explicit ports.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE=""
CLI_MODEL_NAME=""
declare -a CLI_PORTS=()

for arg in "$@"; do
  if [[ "$arg" =~ \.(conf|env)$ ]]; then
    CONFIG_FILE="$arg"
  elif [[ "$arg" =~ ^(llama|qwen)$ ]]; then
    CLI_MODEL_NAME="$arg"
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then
    CLI_PORTS+=("$arg")
  fi
done

CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/deploy.conf}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime_config.sh"
load_deployment_runtime "$CONFIG_FILE"

MODEL_NAME="${CLI_MODEL_NAME:-$DEPLOY_SELECTED_MODEL}"
if [[ ${#CLI_PORTS[@]} -gt 0 ]]; then
  PORTS=("${CLI_PORTS[@]}")
elif [[ -f "$SCRIPT_DIR/${LOG_DIR:-logs}/active_deployment.env" ]]; then
  # shellcheck disable=SC1090
  source "$SCRIPT_DIR/${LOG_DIR:-logs}/active_deployment.env"
  MODEL_NAME="${CLI_MODEL_NAME:-$ACTIVE_MODEL}"
  read -r -a PORTS <<< "$ACTIVE_PORTS"
  echo "[ACTIVE] 使用最近一次实际部署拓扑"
else
  PORTS=("${DEPLOY_SELECTED_PORTS[@]}")
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "测试: model=$MODEL_NAME, ports=${PORTS[*]}"

_test() {
  local name="$1" port="$2" resp answer
  echo "── $name @ localhost:$port ──"
  if ! curl --silent --fail --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    echo "  [FAIL] 无法连接"
    return 1
  fi
  resp=$(curl --silent --show-error --max-time 120 "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${name}\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1? Output only the answer.\"}],\"temperature\":0,\"max_tokens\":32,\"seed\":42}" 2>&1)
  if answer=$(printf '%s' "$resp" | "$PYTHON_BIN" -c \
      "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null); then
    echo "  [PASS] $answer"
  else
    echo "  [FAIL] ${resp:0:300}"
    return 1
  fi
}

FAILED=0
for port in "${PORTS[@]}"; do
  _test "$MODEL_NAME" "$port" || FAILED=$((FAILED + 1))
done

echo "========================================"
if [[ $FAILED -eq 0 ]]; then
  echo "  全部 ${#PORTS[@]} 个端点通过"
else
  echo "  $FAILED/${#PORTS[@]} 失败"
  exit 1
fi
echo "========================================"

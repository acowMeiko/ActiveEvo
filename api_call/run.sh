#!/usr/bin/env bash
# ============================================================
# api_call/run.sh — vLLM 多卡部署启动脚本
# ============================================================
# 用法:
#   bash run.sh deploy.conf                  # use deploy.conf defaults
#   bash run.sh deploy.conf llama 0 1 2 3    # temporary CLI override
#   bash run.sh deploy.conf qwen 4 5 6 7
#   bash run.sh deploy.conf llama 0 1 2 3 4 5 6 7
#   bash run.sh deploy.conf llama 0 1 2 3 4 5 6 7 --daemon
#
# --daemon: 启动后立即返回，不 trap EXIT，不 wait
#           配合 api_call/stop.sh 用于 benchmark 等自动化流程
# --server-seed N: 设置 vLLM 服务端 seed（请求仍可用 evaluator.py --seed 覆盖）
#
# 端口自动计算: port = DEPLOY_PORT_BASE + GPU_ID（默认基数 8100）
# ============================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 解析参数: CONFIG MODEL GPU... ────────────────────────────
CONFIG_FILE=""
CLI_MODEL_NAME=""
declare -a CLI_GPU_IDS=()
DAEMON=false
CLI_SERVER_SEED=""

expect_server_seed=false
for arg in "${@}"; do
  if $expect_server_seed; then
    if [[ ! "$arg" =~ ^[0-9]+$ ]]; then
      echo "[FATAL] --server-seed 必须是非负整数" >&2
      exit 1
    fi
    CLI_SERVER_SEED="$arg"
    expect_server_seed=false
    continue
  fi
  if [[ "$arg" =~ \.conf$ ]] || [[ "$arg" =~ \.env$ ]]; then
    CONFIG_FILE="$arg"
  elif [[ "$arg" =~ ^(llama|qwen|qwq|deepseek)$ ]]; then
    CLI_MODEL_NAME="$arg"
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then
    CLI_GPU_IDS+=("$arg")
  elif [[ "$arg" == "--daemon" ]]; then
    DAEMON=true
  elif [[ "$arg" == "--server-seed" ]]; then
    expect_server_seed=true
  fi
done
if $expect_server_seed; then
  echo "[FATAL] --server-seed 缺少数值" >&2
  exit 1
fi

# ── 自动查找配置 ────────────────────────────────────────────
if [[ -z "$CONFIG_FILE" ]]; then
  if [[ -f "$PROJECT_ROOT/deploy.conf" ]]; then
    CONFIG_FILE="$PROJECT_ROOT/deploy.conf"
  else
    echo "[FATAL] 未指定配置文件。用法: bash run.sh <config.conf> <llama|qwen> <GPU...>" >&2
    exit 1
  fi
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[FATAL] 配置文件不存在: $CONFIG_FILE" >&2
  exit 1
fi

echo "[CONFIG] 加载配置: $CONFIG_FILE"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime_config.sh"
load_deployment_runtime "$CONFIG_FILE"

MODEL_NAME="${CLI_MODEL_NAME:-$DEPLOY_SELECTED_MODEL}"
declare -a GPU_IDS=()
if [[ ${#CLI_GPU_IDS[@]} -gt 0 ]]; then
  GPU_IDS=("${CLI_GPU_IDS[@]}")
else
  GPU_IDS=("${DEPLOY_SELECTED_GPU_IDS[@]}")
fi
SERVER_SEED="${CLI_SERVER_SEED:-$DEPLOY_SELECTED_SERVER_SEED}"

# ── 验证模型名 ──────────────────────────────────────────────
# ── 根据模型名选择路径 ──────────────────────────────────────
case "$MODEL_NAME" in
  llama)
    MODEL_PATH="${LLAMA_PATH:-}"
    ;;
  qwen)
    MODEL_PATH="${QWEN_PATH:-}"
    ;;
  *)
    echo "[FATAL] 未知模型: $MODEL_NAME (支持: llama, qwen)" >&2
    exit 1
    ;;
esac

if [[ -z "$MODEL_PATH" ]]; then
  echo "[FATAL] 未配置 ${MODEL_NAME^^}_PATH，请检查 $CONFIG_FILE" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[WARN] 模型路径不存在: $MODEL_PATH" >&2
fi

# ── 验证 GPU 列表 ───────────────────────────────────────────
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[FATAL] 请在 $CONFIG_FILE 的 DEPLOY_GPU_IDS 中配置至少一张 GPU" >&2
  exit 1
fi

declare -A seen_effective_gpus=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] 非法 GPU 编号: $gpu" >&2
    exit 1
  fi
  if [[ -n "${seen_effective_gpus[$gpu]:-}" ]]; then
    echo "[FATAL] GPU 编号重复: $gpu" >&2
    exit 1
  fi
  seen_effective_gpus[$gpu]=1
done

# ── 生成端口列表 (port = DEPLOY_PORT_BASE + GPU_ID) ─────────
declare -a PORTS=()
for gpu in "${GPU_IDS[@]}"; do
  PORTS+=("$((DEPLOY_SELECTED_PORT_BASE + gpu))")
done

# ── 默认值 ──────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
SERVER_SEED="${SERVER_SEED:-0}"
LOG_DIR="${LOG_DIR:-logs}"
READY_TIMEOUT="${DEPLOY_READY_TIMEOUT:-600}"

if [[ ! "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FATAL] DEPLOY_READY_TIMEOUT 必须是正整数秒数" >&2
  exit 1
fi

if [[ "$TENSOR_PARALLEL_SIZE" != "1" ]]; then
  echo "[FATAL] 当前脚本是一卡一实例，只支持 TENSOR_PARALLEL_SIZE=1" >&2
  echo "        多卡共同承载一个模型需要使用单独的 tensor-parallel 启动流程。" >&2
  exit 1
fi

LOG_ROOT="$SCRIPT_DIR/$LOG_DIR"
mkdir -p "$LOG_ROOT"

PID_DIR="$LOG_ROOT/pids"
mkdir -p "$PID_DIR"

ACTIVE_RUNTIME_FILE="$LOG_ROOT/active_deployment.env"

# ── 全局 PID ────────────────────────────────────────────────
declare -a ALL_PIDS=()
declare -a ALL_LOG_FILES=()
LOG_RUN_TAG="$(date +%Y%m%d_%H%M%S)"

_stop_spawned() {
  echo ""
  echo "[STOP] 收到终止信号，正在关闭所有 vLLM 实例..."
  for pid in "${ALL_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  for i in "${!GPU_IDS[@]}"; do
    rm -f "$PID_DIR/${MODEL_NAME}_gpu${GPU_IDS[$i]}_port${PORTS[$i]}.pid"
  done
  rm -f "$ACTIVE_RUNTIME_FILE"
  echo "[STOP] 所有实例已关闭。"
}

if ! $DAEMON; then
  trap '_stop_spawned; exit 130' SIGINT SIGTERM
  trap _stop_spawned EXIT
fi

# Validate every target before starting any process, so a conflict cannot leave
# a partially deployed topology behind.
for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  port="${PORTS[$i]}"
  pid_file="$PID_DIR/${MODEL_NAME}_gpu${gpu}_port${port}.pid"
  if [[ -f "$pid_file" ]]; then
    existing_pid=$(<"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "[FATAL] GPU $gpu 已有受管理实例运行: PID=$existing_pid, port=$port" >&2
      exit 1
    fi
    rm -f "$pid_file"
  fi
  if curl --silent --max-time 1 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[FATAL] 端口 $port 已有模型服务响应；请勿重复部署" >&2
    exit 1
  fi
  if command -v ss >/dev/null 2>&1 && ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; then
    echo "[FATAL] 端口 $port 已被其他进程占用" >&2
    exit 1
  fi
done

# ── 启动 ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  vLLM 部署"
echo "  模型:    $MODEL_NAME  →  $MODEL_PATH"
echo "  GPU:     ${GPU_IDS[*]}"
echo "  端口:    ${PORTS[*]}"
echo "  并发上限: ${MAX_NUM_SEQS} seqs/卡 × ${#GPU_IDS[@]} 卡 = $((MAX_NUM_SEQS * ${#GPU_IDS[@]})) seqs 总"
echo "================================================================"
echo ""

for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  port="${PORTS[$i]}"
  # A unique file per deployment prevents a later cleanup/redeploy from
  # unlinking the log still held by a live vLLM process.
  log_file="$LOG_ROOT/${MODEL_NAME}_gpu${gpu}_port${port}_${LOG_RUN_TAG}.log"
  pid_file="$PID_DIR/${MODEL_NAME}_gpu${gpu}_port${port}.pid"

  echo "  [${MODEL_NAME}] GPU=$gpu 端口=$port 启动中..."

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$port" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype "$DTYPE" \
    --seed "$SERVER_SEED" \
    $VLLM_EXTRA_ARGS \
    > "$log_file" 2>&1 &

  pid=$!
  echo "$pid" > "$pid_file"
  ALL_PIDS+=("$pid")
  ALL_LOG_FILES+=("$log_file")
done

# 保存本次实际拓扑（包括命令行覆盖），供 test.sh 和 stop.sh 自动读取。
{
  printf 'ACTIVE_CONFIG=%q\n' "$CONFIG_FILE"
  printf 'ACTIVE_MODEL=%q\n' "$MODEL_NAME"
  printf 'ACTIVE_GPU_IDS=%q\n' "${GPU_IDS[*]}"
  printf 'ACTIVE_PORTS=%q\n' "${PORTS[*]}"
  printf 'ACTIVE_SERVER_SEED=%q\n' "$SERVER_SEED"
  printf 'ACTIVE_PIDS=%q\n' "${ALL_PIDS[*]}"
  printf 'ACTIVE_LOG_FILES=%q\n' "${ALL_LOG_FILES[*]}"
} > "$ACTIVE_RUNTIME_FILE"

echo "[WAIT] 最长等待 ${READY_TIMEOUT}s，检查所有端点 /v1/models ..."
deadline=$((SECONDS + READY_TIMEOUT))
declare -a endpoint_ready=()
for _ in "${GPU_IDS[@]}"; do endpoint_ready+=(false); done

while ((SECONDS < deadline)); do
  all_ready=true
  for i in "${!GPU_IDS[@]}"; do
    ${endpoint_ready[$i]} && continue
    pid="${ALL_PIDS[$i]}"
    port="${PORTS[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[FATAL] GPU ${GPU_IDS[$i]} 的服务进程提前退出；日志: $LOG_ROOT/${MODEL_NAME}_gpu${GPU_IDS[$i]}_port${port}.log" >&2
      if $DAEMON; then _stop_spawned; fi
      exit 1
    fi
    if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      endpoint_ready[$i]=true
      echo "[OK] GPU ${GPU_IDS[$i]} → port $port 已就绪"
    else
      all_ready=false
    fi
  done
  $all_ready && break
  sleep 2
done

for i in "${!GPU_IDS[@]}"; do
  if ! ${endpoint_ready[$i]}; then
    echo "[FATAL] GPU ${GPU_IDS[$i]} → port ${PORTS[$i]} 在 ${READY_TIMEOUT}s 内未就绪" >&2
    if $DAEMON; then _stop_spawned; fi
    exit 1
  fi
done

echo ""
echo "----------------------------------------------------------------"
echo "[READY] 共 ${#ALL_PIDS[@]} 个 $MODEL_NAME 实例通过健康检查"
echo "       本次日志: ${ALL_LOG_FILES[*]}"
echo "       PID:  $PID_DIR/${MODEL_NAME}_gpu*_port*.pid"
echo ""
echo "  停止:  bash $SCRIPT_DIR/stop.sh $CONFIG_FILE"
echo "  测试:  bash $SCRIPT_DIR/test.sh $CONFIG_FILE $MODEL_NAME"
echo "  实际拓扑: GPU ${GPU_IDS[*]} → ports ${PORTS[*]}"
echo "  显式停止: bash $SCRIPT_DIR/stop.sh $CONFIG_FILE $MODEL_NAME ${GPU_IDS[*]}"
echo "  显式测试: bash $SCRIPT_DIR/test.sh $CONFIG_FILE $MODEL_NAME ${PORTS[*]}"
echo "----------------------------------------------------------------"
echo ""

if $DAEMON; then
    echo "[DAEMON] 后台模式，PID 文件: $PID_DIR/"
    echo "[DAEMON] 停止: bash $SCRIPT_DIR/stop.sh $CONFIG_FILE"
    disown -a 2>/dev/null || true
    exit 0
fi

wait

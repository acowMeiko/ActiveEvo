#!/usr/bin/env bash
# ============================================================
# benchmarks/deploy.sh — 双模型部署 + 健康检查
#
# 用法:
#   bash benchmarks/deploy.sh deploy.conf           # 部署 LLaMA(GPU 0-3) + Qwen(GPU 4-7)
#   bash benchmarks/deploy.sh deploy.conf llama     # 仅 LLaMA
#   bash benchmarks/deploy.sh deploy.conf qwen      # 仅 Qwen
#   bash benchmarks/deploy.sh deploy.conf --stop    # 停止全部
#   bash benchmarks/deploy.sh deploy.conf --status  # 查看状态
#
# 端口分配:
#   LLaMA: 8000-8003  (GPU 0-3)
#   Qwen:  8004-8007  (GPU 4-7)
# ============================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── 默认配置 ──────────────────────────────────────────────────
DEPLOY_LLAMA=true
DEPLOY_QWEN=true
STOP_MODE=false
STATUS_MODE=false
CONFIG_FILE=""

for arg in "${@}"; do
  case "$arg" in
    --stop)    STOP_MODE=true ;;
    --status)  STATUS_MODE=true ;;
    llama)     DEPLOY_QWEN=false ;;
    qwen)      DEPLOY_LLAMA=false ;;
    *.conf|*.env) CONFIG_FILE="$arg" ;;
  esac
done

# ── 查找配置文件 ──────────────────────────────────────────────
if [[ -z "$CONFIG_FILE" ]]; then
  if [[ -f "$PROJECT_ROOT/deploy.conf" ]]; then
    CONFIG_FILE="$PROJECT_ROOT/deploy.conf"
  else
    echo "[FATAL] 未找到配置文件。用法: bash deploy.sh <deploy.conf> [llama|qwen]"
    exit 1
  fi
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[FATAL] 配置文件不存在: $CONFIG_FILE"
  exit 1
fi

echo "[CONFIG] $CONFIG_FILE"
source "$CONFIG_FILE"

# ── 默认参数 ──────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# ── 模型到 GPU+Port 映射 ──────────────────────────────────────
declare -A MODEL_PATHS
MODEL_PATHS[llama]="${LLAMA_PATH:-}"
MODEL_PATHS[qwen]="${QWEN_PATH:-}"

declare -A MODEL_GPUS
MODEL_GPUS[llama]="0 1 2 3"
MODEL_GPUS[qwen]="4 5 6 7"

# ── 状态检查 ──────────────────────────────────────────────────
if $STATUS_MODE; then
  echo "================================================================"
  echo "  vLLM 服务状态"
  echo "================================================================"
  for port in $(seq 8000 8007); do
    if lsof -ti:"$port" >/dev/null 2>&1; then
      echo "  [UP]   端口 $port  (GPU $((port - 8000)))"
    else
      echo "  [DOWN] 端口 $port  (GPU $((port - 8000)))"
    fi
  done
  echo ""
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
  fi
  exit 0
fi

# ── 停止 ──────────────────────────────────────────────────────
if $STOP_MODE; then
  echo "================================================================"
  echo "  停止所有 benchmark vLLM 服务 (端口 8000-8007)"
  echo "================================================================"
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for port in $(seq 8000 8007); do
    fuser -k "${port}/tcp" 2>/dev/null || true
    lsof -ti:"${port}" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  done
  sleep 2
  echo "[OK] 服务已停止"
  exit 0
fi

# ── 检查模型路径 ──────────────────────────────────────────────
declare -A DEPLOY_FLAGS
DEPLOY_FLAGS[llama]=$DEPLOY_LLAMA
DEPLOY_FLAGS[qwen]=$DEPLOY_QWEN

for model in llama qwen; do
  if ! ${DEPLOY_FLAGS[$model]}; then
    continue
  fi
  if [[ -z "${MODEL_PATHS[$model]}" ]]; then
    echo "[FATAL] ${model^^}_PATH 未在 $CONFIG_FILE 中配置"
    exit 1
  fi
  if [[ ! -d "${MODEL_PATHS[$model]}" ]]; then
    echo "[FATAL] 模型路径不存在: ${MODEL_PATHS[$model]}"
    exit 1
  fi
done

# ── 启动 ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Benchmark 双模型部署"
echo "================================================================"

declare -a ALL_PIDS=()

for model in llama qwen; do
  if ! ${DEPLOY_FLAGS[$model]}; then
    continue
  fi

  MODEL_PATH="${MODEL_PATHS[$model]}"
  echo ""
  echo "--- $model: $MODEL_PATH ---"

  for gpu in ${MODEL_GPUS[$model]}; do
    port=$((8000 + gpu))
    log_file="$LOG_DIR/${model}_gpu${gpu}_port${port}.log"

    echo "  GPU=$gpu 端口=$port 启动中..."

    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --served-model-name "$model" \
      --host "$HOST" \
      --port "$port" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --dtype "$DTYPE" \
      $VLLM_EXTRA_ARGS \
      > "$log_file" 2>&1 &

    ALL_PIDS+=($!)
    echo "  -> PID ${ALL_PIDS[-1]}"
  done
done

echo ""
echo "----------------------------------------------------------------"
echo "[INFO] 共启动 ${#ALL_PIDS[@]} 个实例"
echo "       PID: ${ALL_PIDS[*]}"
echo ""

# ── 等待健康检查 ──────────────────────────────────────────────
echo "[WAIT] 等待模型就绪 (健康检查, 最长等待 15 分钟)..."

MAX_WAIT=900
INTERVAL=5
ELAPSED=0

declare -A EXPECTED_PORTS
if $DEPLOY_LLAMA; then
  for gpu in ${MODEL_GPUS[llama]}; do EXPECTED_PORTS[$((8000 + gpu))]=1; done
fi
if $DEPLOY_QWEN; then
  for gpu in ${MODEL_GPUS[qwen]}; do EXPECTED_PORTS[$((8000 + gpu))]=1; done
fi
TOTAL_PORTS=${#EXPECTED_PORTS[@]}

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
  READY=0
  for port in "${!EXPECTED_PORTS[@]}"; do
    if curl -s "http://localhost:${port}/health" >/dev/null 2>&1; then
      READY=$((READY + 1))
    fi
  done

  if [[ $READY -eq $TOTAL_PORTS ]]; then
    echo ""
    echo "[OK] 全部 $TOTAL_PORTS 个端点就绪 (${ELAPSED}s)"
    break
  fi

  printf "  [%3ds] %d/%d 就绪 ...\n" "$ELAPSED" "$READY" "$TOTAL_PORTS"
  sleep "$INTERVAL"
  ELAPSED=$((ELAPSED + INTERVAL))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
  echo "[WARN] 健康检查超时 (${MAX_WAIT}s), 部分端点可能未就绪"
fi

echo ""
echo "================================================================"
echo "  部署完成"
echo "  日志目录: $LOG_DIR"
echo ""
echo "  运行 benchmark:"
echo "    python benchmarks/run_suite.py --models llama qwen --mode guided"
echo ""
echo "  停止服务:"
echo "    bash benchmarks/deploy.sh $CONFIG_FILE --stop"
echo "================================================================"
echo ""
echo "[INFO] PID 列表已写入 $LOG_DIR/benchmark_pids.txt"
printf "%s\n" "${ALL_PIDS[@]}" > "$LOG_DIR/benchmark_pids.txt"

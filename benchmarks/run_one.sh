#!/usr/bin/env bash
# ============================================================
# benchmarks/run_one.sh — 单独运行某一个 benchmark
#
# 用法:
#   bash benchmarks/run_one.sh <benchmark> [model] [--foreground]
#
# 示例:
#   bash benchmarks/run_one.sh math500                   # 跑 math500, 两个模型都跑
#   bash benchmarks/run_one.sh math500 llama             # 只跑 math500 + llama
#   bash benchmarks/run_one.sh math500 llama_8gpu        # 8-GPU 模式跑 math500
#   bash benchmarks/run_one.sh gsm8k qwen --foreground   # 前台跑
#
# 可用 benchmark: deepscaler, math500, gsm8k, bbh, svamp, general365
# 可用 model:      llama, qwen, llama_8gpu, qwen_8gpu
#
# 前置条件:
#   1. 模型已部署 (bash api_call/run.sh deploy.conf <model> <gpus>)
#   2. Memory 已通过 curriculum 流程生成了 principles
# ============================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── 解析参数 ──────────────────────────────────────────────
FOREGROUND=false
BENCHMARK=""
MODELS_ARG=()

for arg in "$@"; do
    case "$arg" in
        --foreground) FOREGROUND=true ;;
        *)
            if [[ -z "$BENCHMARK" ]]; then
                BENCHMARK="$arg"
            else
                MODELS_ARG+=("$arg")
            fi
            ;;
    esac
done

if [[ -z "$BENCHMARK" ]]; then
    echo "用法: bash benchmarks/run_one.sh <benchmark> [model] [--foreground]"
    echo ""
    echo "可用 benchmark:"
    echo "  deepscaler    math500    gsm8k    bbh    svamp    general365"
    echo ""
    echo "可用 model:"
    echo "  llama    qwen    llama_8gpu    qwen_8gpu"
    echo ""
    echo "示例:"
    echo "  bash benchmarks/run_one.sh math500"
    echo "  bash benchmarks/run_one.sh math500 llama"
    echo "  bash benchmarks/run_one.sh math500 llama_8gpu"
    echo "  bash benchmarks/run_one.sh gsm8k --foreground"
    exit 1
fi

# 转为小写
BENCHMARK_LOWER=$(echo "$BENCHMARK" | tr '[:upper:]' '[:lower:]')

# 如果没指定模型, 默认两个都跑
if [[ ${#MODELS_ARG[@]} -eq 0 ]]; then
    MODELS=("llama" "qwen")
else
    MODELS=("${MODELS_ARG[@]}")
fi

# ── 自后台化 ──────────────────────────────────────────────
if [[ "$FOREGROUND" != true ]]; then
    CONSOLE_LOG="$LOG_DIR/benchmark_${BENCHMARK_LOWER}_${TIMESTAMP}.log"
    MODELS_STR=$(IFS=_; echo "${MODELS[*]}")
    echo "[INFO] 后台启动: ${BENCHMARK_LOWER} (models=${MODELS_STR})"
    echo "[INFO] 控制台日志: $CONSOLE_LOG"
    nohup bash "$SCRIPT_PATH" "$BENCHMARK" "${MODELS[@]}" --foreground > "$CONSOLE_LOG" 2>&1 &
    echo "[INFO] PID: $!"
    exit 0
fi

export PYTHONUNBUFFERED=1
export MEMUA_EMBEDDING_DEVICE=cuda

echo "================================================================"
echo "  单独 Benchmark 评测"
echo "  Benchmark : ${BENCHMARK_LOWER}"
echo "  Models    : ${MODELS[*]}"
echo "  开始时间  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

for model in "${MODELS[@]}"; do
    BM_LOG="$LOG_DIR/benchmark_${model}_${BENCHMARK_LOWER}_${TIMESTAMP}.log"

    # 8-GPU 模型用更高并发: 8卡 × 16 = 128
    if [[ "$model" == *"_8gpu" ]]; then
        CONCURRENCY="${BENCH_CONCURRENCY_8GPU:-128}"
    else
        CONCURRENCY="${BENCH_CONCURRENCY:-64}"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "开始: ${model} / ${BENCHMARK_LOWER}  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "并发: ${CONCURRENCY}  日志: ${BM_LOG}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python benchmarks/run_suite.py \
        --models "${model}" \
        --mode guided \
        --concurrency "${CONCURRENCY}" \
        --skip deepscaler \
        --only "${BENCHMARK_LOWER}" \
        --output output/benchmarks \
        --resume \
        > "${BM_LOG}" 2>&1

    RC=$?
    if [[ $RC -ne 0 ]]; then
        echo "失败: ${model} / ${BENCHMARK_LOWER} (exit=$RC)"
        echo "  查看日志: ${BM_LOG}"
    else
        echo "完成: ${model} / ${BENCHMARK_LOWER}  $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # ── 打印当前 benchmark 结果 ────────────────────────────
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  结果: ${model} / ${BENCHMARK_LOWER}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    BACKBONE="${model%_8gpu}"
    python -c "
import json, sys
from pathlib import Path
results_dir = Path('output/benchmarks') / '${BACKBONE}'
found = list(results_dir.glob('${BENCHMARK_LOWER}/**/eval_summary.json'))
if not found:
    found = list(results_dir.glob('${BENCHMARK_LOWER}/eval_summary.json'))
for s in sorted(found):
    d = json.loads(s.read_text())
    p = d.get('performance', {})
    r = d.get('retrieval', {})
    t = d.get('retry_stats', {})
    name = s.parent.relative_to(results_dir)
    print(f'  {name}')
    print(f'    Accuracy : {p.get(\"accuracy\",0):.4f} ({p.get(\"correct\",\"?\")}/{p.get(\"scope_total\",\"?\")})')
    print(f'    Retrieval: {r.get(\"retrieval_used_ratio\",0):.1%}  |  Memory: {d.get(\"memory_experience_count\",0)} principles')
    retry_info = f'first_ok={t.get(\"first_attempt_correct\",\"?\")}'
    if t.get('rescued_by_retry', 0) > 0:
        retry_info += f' rescued={t[\"rescued_by_retry\"]}'
    print(f'    Retry    : {retry_info}')
"
done

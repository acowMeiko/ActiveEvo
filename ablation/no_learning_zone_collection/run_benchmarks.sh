#!/usr/bin/env bash
# ============================================================
# ablation/no_learning_zone_collection/run_benchmarks.sh
#
# 用 no_learning_zone 各 level 的 memory 快照跑 benchmark 评测:
#   - Level 1~4: 各自跑 deepscaler
#   - Level 4:   跑剩余全部 benchmark (不含 deepscaler/general365)
#
# 用法:
#   bash ablation/no_learning_zone_collection/run_benchmarks.sh              # 8-GPU llama, 后台
#   bash ablation/no_learning_zone_collection/run_benchmarks.sh --foreground  # 前台
#   bash ablation/no_learning_zone_collection/run_benchmarks.sh llama         # 4-GPU
#   bash ablation/no_learning_zone_collection/run_benchmarks.sh qwen_8gpu     # 8-GPU qwen
#
# 前置条件:
#   1. 模型已部署 (bash api_call/run.sh deploy.conf <model> <gpus> --daemon)
#   2. no_learning_zone memory 已构建完毕 (output/no_learning_zone/llama/memory/snapshots/)
# ============================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── 参数解析 ──────────────────────────────────────────────────
FOREGROUND=false
MODEL=""

for arg in "$@"; do
    case "$arg" in
        --foreground) FOREGROUND=true ;;
        *) MODEL="$arg" ;;
    esac
done

MODEL="${MODEL:-llama_8gpu}"

# ── 模型 → backbone + ports ──────────────────────────────────
case "$MODEL" in
    llama_8gpu)
        BACKBONE="llama"
        PORTS=(8000 8001 8002 8003 8004 8005 8006 8007)
        CONCURRENCY=128
        ;;
    qwen_8gpu)
        BACKBONE="qwen"
        PORTS=(8000 8001 8002 8003 8004 8005 8006 8007)
        CONCURRENCY=128
        ;;
    llama)
        BACKBONE="llama"
        PORTS=(8000 8001 8002 8003)
        CONCURRENCY=64
        ;;
    qwen)
        BACKBONE="qwen"
        PORTS=(8004 8005 8006 8007)
        CONCURRENCY=64
        ;;
    *)
        echo "未知模型: $MODEL (可用: llama, qwen, llama_8gpu, qwen_8gpu)"
        exit 1
        ;;
esac

# ── 路径 ─────────────────────────────────────────────────────
MEMORY_BASE="output/no_learning_zone/${BACKBONE}/memory/snapshots"
OUTPUT_BASE="output/no_learning_zone/${BACKBONE}/benchmarks"

# ── 自后台化 ──────────────────────────────────────────────────
if [[ "$FOREGROUND" != true ]]; then
    CONSOLE_LOG="$LOG_DIR/nlz_benchmark_${BACKBONE}_${TIMESTAMP}.log"
    echo "[INFO] 后台启动 no_learning_zone benchmark 评测"
    echo "[INFO] Model: $MODEL  Ports: ${PORTS[*]}  Concurrency: $CONCURRENCY"
    echo "[INFO] 控制台日志: $CONSOLE_LOG"
    nohup bash "$SCRIPT_PATH" "$MODEL" --foreground > "$CONSOLE_LOG" 2>&1 &
    echo "[INFO] PID: $!"
    exit 0
fi

export PYTHONUNBUFFERED=1
export MEMUA_EMBEDDING_DEVICE=cuda

echo "================================================================"
echo "  No Learning Zone — Benchmark 评测"
echo "  Model : $MODEL (backbone=$BACKBONE)"
echo "  Ports : ${PORTS[*]}"
echo "  Start : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ────────────────────────────────────────────────────────────────
# Level 1~4: 各跑一次 deepscaler
# ────────────────────────────────────────────────────────────────

for level in 1 2 3 4; do
    MEM_DIR="${MEMORY_BASE}/round_${level}"
    if [[ ! -d "$MEM_DIR" ]]; then
        echo "[WARN] 快照不存在, 跳过: $MEM_DIR"
        continue
    fi

    PRINCIPLE_COUNT=$(python -c "
import json
p = json.loads(open('${MEM_DIR}/principles/principles.json').read())
print(len(p))
" 2>/dev/null || echo "?")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[Level ${level}] deepscaler  (memory: ${PRINCIPLE_COUNT} principles)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python benchmarks/evaluator.py \
        --test-file data/test/deepscaler.json \
        --memory-dir "$MEM_DIR" \
        --ports "${PORTS[@]}" \
        --model "$BACKBONE" \
        --mode guided \
        --output "${OUTPUT_BASE}/deepscaler/level${level}" \
        --concurrency "$CONCURRENCY" \
        --retry-on-wrong 2

    echo "[Level ${level}] deepscaler 完成: $(date '+%Y-%m-%d %H:%M:%S')"
done

# ────────────────────────────────────────────────────────────────
# Level 4: 跑剩余全部 benchmark (不含 deepscaler, general365)
# ────────────────────────────────────────────────────────────────

LEVEL4_MEM="${MEMORY_BASE}/round_4"

declare -A BENCHMARKS=(
    ["math500"]="data/test/benchmarks/math500_stage3.json"
    ["gsm8k"]="data/test/benchmarks/gsm8k_stage3.json"
    ["bbh_logic"]="data/test/benchmarks_split/bbh_logic_stage3.json"
    ["bbh_math"]="data/test/benchmarks_split/bbh_math_stage3.json"
    ["svamp"]="data/test/official/svamp/processed/test.json"
)

echo ""
echo "================================================================"
echo "  Level 4 全量 Benchmark 评测"
echo "  Memory: $LEVEL4_MEM"
echo "================================================================"

for name in "${!BENCHMARKS[@]}"; do
    test_file="${BENCHMARKS[$name]}"

    if [[ ! -f "$test_file" ]]; then
        echo "[WARN] 测试文件不存在, 跳过: $test_file"
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[Level 4] ${name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python benchmarks/evaluator.py \
        --test-file "$test_file" \
        --memory-dir "$LEVEL4_MEM" \
        --ports "${PORTS[@]}" \
        --model "$BACKBONE" \
        --mode guided \
        --output "${OUTPUT_BASE}/${name}" \
        --concurrency "$CONCURRENCY" \
        --retry-on-wrong 2

    echo "[Level 4] ${name} 完成: $(date '+%Y-%m-%d %H:%M:%S')"
done

# ────────────────────────────────────────────────────────────────
# 汇总
# ────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  No Learning Zone Benchmark 汇总"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

python -c "
import json
from pathlib import Path

output_base = Path('${OUTPUT_BASE}')
results = {}

# DeepScaler per level
for level in [1, 2, 3, 4]:
    key = f'deepscaler_L{level}'
    summary_path = output_base / 'deepscaler' / f'level{level}' / 'eval_summary.json'
    if summary_path.exists():
        d = json.loads(summary_path.read_text())
        p = d.get('performance', {})
        results[key] = {
            'accuracy': p.get('accuracy', 0),
            'correct': p.get('correct', 0),
            'total': p.get('scope_total', 0),
            'principles': d.get('memory_experience_count', 0),
        }

# Other benchmarks
for name in ['math500', 'gsm8k', 'bbh_logic', 'bbh_math', 'svamp']:
    summary_path = output_base / name / 'eval_summary.json'
    if summary_path.exists():
        d = json.loads(summary_path.read_text())
        p = d.get('performance', {})
        results[name] = {
            'accuracy': p.get('accuracy', 0),
            'correct': p.get('correct', 0),
            'total': p.get('scope_total', 0),
            'principles': d.get('memory_experience_count', 0),
        }

print()
print(f'{\"Benchmark\":<18} {\"Accuracy\":>10} {\"Correct\":>10} {\"Total\":>8}  {\"Memory\":>8}')
print('-' * 60)
for name, r in results.items():
    print(f'{name:<18} {r[\"accuracy\"]:>10.4f} {r[\"correct\"]:>10} {r[\"total\"]:>8}  {r[\"principles\"]:>8}')

# Save combined summary
(output_base / 'all_summary.json').write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)
print(f'\n汇总已保存: {output_base / \"all_summary.json\"}')
"

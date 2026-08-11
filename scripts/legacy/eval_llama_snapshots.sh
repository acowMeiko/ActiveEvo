#!/usr/bin/env bash
# ============================================================
# LLaMA 四轮 snapshot 渐进评估 — 每轮只测上轮未做对的题
# Round 1 做对的题 Round 2-4 不再重测 (carry-forward)
#
# 输出: output/stage3_eval/llama_round1/ → round4/
#       每个目录下的 eval_summary.json 含 cumulative accuracy
#
# 用法: nohup bash scripts/eval_llama_snapshots.sh > logs/llama_snapshot_eval.log 2>&1 &
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p logs output/stage3_eval

MEMORY_ROOT="memory/llama"
ACTIVE_PRINCIPLES="$MEMORY_ROOT/principles/principles.json"
INDICES_DIR="$MEMORY_ROOT/indices"
SNAPSHOT_BASE="$MEMORY_ROOT/snapshots"
TEST_INPUT="data/test/deepscaler.json"
WORK_DIR="output/stage3_eval/.tmp_llama"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# 备份 active memory
BACKUP="$(mktemp -d)"
cp -r "$MEMORY_ROOT/principles" "$BACKUP/principles"
cp -r "$MEMORY_ROOT/indices" "$BACKUP/indices" 2>/dev/null || true

cleanup() {
    echo "Restoring active memory..."
    rm -rf "$MEMORY_ROOT/principles" "$MEMORY_ROOT/indices"
    cp -r "$BACKUP/principles" "$MEMORY_ROOT/principles"
    if [ -d "$BACKUP/indices" ]; then
        cp -r "$BACKUP/indices" "$MEMORY_ROOT/indices"
    fi
    rm -rf "$BACKUP"
    echo "Restored."
}
trap cleanup EXIT

# 初始: 全量测试集
cp "$TEST_INPUT" "$WORK_DIR/current_pool.json"

declare -A CUMULATIVE_BY_ROUND
CUMULATIVE_CORRECT=0

for round in 1 2 3 4; do
    SNAPSHOT="$SNAPSHOT_BASE/round_${round}/principles/principles.json"
    if [ ! -f "$SNAPSHOT" ]; then
        echo "[Round $round] Snapshot not found — skipping"
        continue
    fi

    COUNT=$(python -c "import json; print(len(json.load(open('$SNAPSHOT', encoding='utf-8'))))")
    POOL_SIZE=$(python -c "import json; print(len(json.load(open('$WORK_DIR/current_pool.json', encoding='utf-8'))))")
    echo ""
    echo "=== Round $round: $COUNT principles, $POOL_SIZE questions remaining ==="

    # Swap in snapshot principles
    rm -rf "$MEMORY_ROOT/principles" "$MEMORY_ROOT/indices"
    mkdir -p "$MEMORY_ROOT/principles"
    cp "$SNAPSHOT" "$ACTIVE_PRINCIPLES"

    # Run guided eval on remaining questions only
    python scripts/stage3_eval.py \
      --test-input "$WORK_DIR/current_pool.json" \
      --backbone llama \
      --model llama \
      --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
      --mode guided \
      --concurrency 64 \
      --output "output/stage3_eval/llama_round${round}"

    # Parse how many were correct this round
    RESULT_FILE="output/stage3_eval/llama_round${round}/eval_summary.json"
    if [ -f "$RESULT_FILE" ]; then
        ROUND_CORRECT=$(python -c "
import json
d = json.load(open('$RESULT_FILE', encoding='utf-8'))
perf = d.get('performance', {})
print(perf.get('correct', 0))
")
        CUMULATIVE_CORRECT=$((CUMULATIVE_CORRECT + ROUND_CORRECT))
        CUMULATIVE_BY_ROUND[$round]=$CUMULATIVE_CORRECT
        echo "  Round $round: $ROUND_CORRECT correct → cumulative: $CUMULATIVE_CORRECT/400"

        # Filter out correct questions for next round
        if [ "$round" -lt 4 ]; then
            python -c "
import json
results = json.load(open('output/stage3_eval/llama_round${round}/eval_results.json', encoding='utf-8'))
# Collect question texts of correctly answered + carried forward
correct_texts = set()
for r in results:
    if r.get('is_correct') or r.get('status') == 'carried_forward':
        q = r.get('question', '')
        correct_texts.add(q)
pool = json.load(open('$WORK_DIR/current_pool.json', encoding='utf-8'))
remaining = [q for q in pool if q.get('question', q.get('problem', '')) not in correct_texts]
json.dump(remaining, open('$WORK_DIR/current_pool.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'  Filtered: {len(remaining)} questions remain for next round')
"
        fi
    else
        echo "  WARNING: Result file not found: $RESULT_FILE"
    fi
done

echo ""
echo "=== 4-Round Evolution ==="
echo "| Round | Cumulative Correct | Accuracy |"
echo "|-------|-------------------|----------|"
for round in 1 2 3 4; do
    CUM="${CUMULATIVE_BY_ROUND[$round]:-}"
    if [ -n "$CUM" ]; then
        ACC=$(python -c "print(f'{${CUM}/400:.4f}')")
        echo "| Round $round | $CUM | $ACC |"
    else
        echo "| Round $round | (no data) | (no data) |"
    fi
done
echo ""
echo "Done."

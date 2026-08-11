#!/usr/bin/env bash
# ============================================================
# LLaMA Clean-Slate Curriculum — 从准确率 0% 开始
# 1. 清除 LLaMA 旧数据
# 2. Stage 1: 全局采样 + 三区划分 (memory 隔离, answer_direct 无引导)
# 3. Baseline 评估: 无 memory 直接答题 (0% 基线)
# 4. 四轮课程学习: panic 重采样 → 记忆化 → 救援
# 5. 最终评估: 累积 memory 引导答题
#
# 用法: nohup bash scripts/run_llama_clean.sh > logs/llama_clean.log 2>&1 &
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p logs

echo "=== Phase 0: 清除 LLaMA 旧数据 ==="
rm -rf memory/llama/
rm -f output/stage1/01_sampling/llama_unified_sampling_results.json
rm -f output/stage1/01_sampling/llama_unified_sampling_results.checkpoint.json
rm -rf output/stage1/02_learning_packets/llama/
rm -rf output/stage3/remediation/llama_*
echo "Phase 0 完成"

echo "=== Phase 1: LLaMA Stage 1 (全局采样, 无 memory 引导) ==="
python run_stage1_llama.py \
  --sample-concurrency 64 \
  --diff-concurrency 64
echo "Phase 1 完成"

echo "=== Phase 2: Baseline 评估 (direct 模式, 无 memory, 0% 基线) ==="
python scripts/stage3_eval.py \
  --test-input data/test/deepscaler.json \
  --backbone llama \
  --model llama \
  --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
  --mode direct \
  --concurrency 64 \
  --output output/stage3_eval/llama_baseline
echo "Phase 2 完成 (baseline 结果: output/stage3_eval/llama_baseline/)"

echo "=== Phase 3: LLaMA 四轮课程学习 (Level 1-4) ==="
python -m core.agent_pipeline \
  --levels 1 2 3 4 \
  --backbone llama \
  --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
  --model llama \
  --concurrency 64 \
  --diff-concurrency 64 \
  --eval-concurrency 64 \
  --skip-sampling
echo "Phase 3 完成"

echo "=== Phase 4: 最终 guided 评估 (累积 memory 引导) ==="
python scripts/stage3_eval.py \
  --test-input data/test/deepscaler.json \
  --backbone llama \
  --model llama \
  --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
  --mode guided \
  --concurrency 64 \
  --output output/stage3_eval/llama_final
echo "Phase 4 完成 (final 结果: output/stage3_eval/llama_final/)"

echo "=== LLaMA Clean-Slate 全流程完成 ==="

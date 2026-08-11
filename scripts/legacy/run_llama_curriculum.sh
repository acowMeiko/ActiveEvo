#!/usr/bin/env bash
# ============================================================
# LLaMA 四轮课程学习 (Round 1-4 = Level 1-4)
# Stage 1 已完成 (run_stage1_llama.py) → --skip-sampling
# 用法: nohup bash scripts/run_llama_curriculum.sh > logs/llama_curriculum.log 2>&1 &
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p logs

python -m core.agent_pipeline \
  --levels 1 2 3 4 \
  --backbone llama \
  --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
  --model llama \
  --concurrency 64 \
  --diff-concurrency 64 \
  --eval-concurrency 64 \
  --skip-sampling

echo "LLaMA 四轮课程学习完成"

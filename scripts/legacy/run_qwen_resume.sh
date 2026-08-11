#!/usr/bin/env bash
# ============================================================
# Qwen 课程学习恢复 — 从 Round 2 (Level 2) 继续
# Round 1 (Level 1) 已完成, 跳过
# 用法: nohup bash scripts/run_qwen_resume.sh > logs/qwen_resume.log 2>&1 &
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p logs

python -m core.agent_pipeline \
  --levels 1 2 3 4 \
  --backbone qwen \
  --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
  --model qwen \
  --concurrency 64 \
  --diff-concurrency 64 \
  --eval-concurrency 64 \
  --skip-sampling \
  --start-level 2

echo "Qwen 课程学习恢复完成 (Level 2-4)"

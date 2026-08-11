#!/usr/bin/env bash
# ============================================================
# Qwen Round 1-4 (Level 1-4 Stage 2 + Stage 3, 跳过 Stage 1)
# 用法: nohup bash scripts/run_qwen_s2s3.sh > logs/qwen_s2s3.log 2>&1 &
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python -m core.agent_pipeline \
  --levels 1 2 3 4 \
  --backbone qwen \
  --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
  --model qwen \
  --concurrency 64 \
  --eval-concurrency 64 \
  --skip-sampling --skip-mining

echo "Qwen 四轮完成"

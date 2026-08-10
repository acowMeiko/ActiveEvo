#!/usr/bin/env bash
# ============================================================
# deploy_model.sh — 便捷部署入口
# ============================================================
# 用法:
#   bash deploy_model.sh deploy.conf llama 0 1 2 3
#   bash deploy_model.sh deploy.conf qwen 4 5 6 7
# ============================================================
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT_DIR/api_call/run.sh" "$@"

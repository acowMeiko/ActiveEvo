#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="${DEPLOY_CONFIG:-$PROJECT_ROOT/deploy.conf}"
ACTIVE_FILE="$PROJECT_ROOT/api_call/logs/active_deployment.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[FATAL] deployment config not found: $CONFIG_FILE" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"
if [[ -f "$ACTIVE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ACTIVE_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${ACTIVE_MODEL:-${DEPLOY_MODEL:-llama}}"
if [[ -n "${ACTIVE_PORTS:-}" ]]; then
  read -r -a PORTS <<< "$ACTIVE_PORTS"
else
  read -r -a GPU_IDS <<< "${DEPLOY_GPU_IDS:?DEPLOY_GPU_IDS is required}"
  PORTS=()
  for gpu in "${GPU_IDS[@]}"; do PORTS+=("$((${DEPLOY_PORT_BASE:-8100} + gpu))"); done
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_ID"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_EXISTING="${WAIT_FOR_EXISTING:-1}"
PILOT_ONLY="${PILOT_ONLY:-0}"
PILOT_SIZE_PER_LEVEL="${PILOT_SIZE_PER_LEVEL:-25}"
PILOT_SEED="${PILOT_SEED:-42}"
ARCHIVE_OLD_MEMORY="${ARCHIVE_OLD_MEMORY:-1}"
CONCURRENCY="${OVERNIGHT_CONCURRENCY:-8}"
CANDIDATE_CONCURRENCY="${OVERNIGHT_CANDIDATE_CONCURRENCY:-16}"
EMBEDDING_DEVICE="${MEMUA_EMBEDDING_DEVICE:-cpu}"
SHOW_PROGRESS="${SHOW_PROGRESS:-auto}"
export MEMUA_EMBEDDING_DEVICE="$EMBEDDING_DEVICE"
TEST_FILE="$PROJECT_ROOT/data/test/deepscaler.json"
MAIN_LOG="$RUN_ROOT/launcher_steps.log"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/dsd_cache" "$RUN_ROOT/frozen_candidates" \
  "$RUN_ROOT/memory/nv_on/store" "$RUN_ROOT/memory/nv_on/audit" \
  "$RUN_ROOT/memory/nv_off/store" "$RUN_ROOT/memory/nv_off/audit" \
  "$RUN_ROOT/stage2" "$RUN_ROOT/stage3"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$MAIN_LOG"; }
run_cmd() {
  log "RUN $*"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  if [[ "$SHOW_PROGRESS" == "1" || ( "$SHOW_PROGRESS" == "auto" && -t 1 ) ]]; then
    "$@" 2>&1 | tee -a "$MAIN_LOG"
  else
    "$@" >> "$MAIN_LOG" 2>&1
  fi
}

log "run_id=$RUN_ID model=$MODEL_NAME ports=${PORTS[*]} dry_run=$DRY_RUN pilot_only=$PILOT_ONLY embedding_device=$EMBEDDING_DEVICE"

if [[ "$WAIT_FOR_EXISTING" == "1" && "$DRY_RUN" != "1" ]]; then
  while pgrep -af '[p]ipeline.py --stage [23]' >/dev/null; do
    log "Existing Stage 2/3 detected; waiting 60s without sending signals."
    sleep 60
  done
fi

for port in "${PORTS[@]}"; do
  if ! curl --silent --show-error --fail --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN warning: endpoint is not currently healthy: port=$port"
    else
      echo "[FATAL] endpoint is not healthy: port=$port" >&2
      exit 3
    fi
  fi
done
if [[ ! -f "$TEST_FILE" ]]; then echo "[FATAL] missing $TEST_FILE" >&2; exit 4; fi
TEST_COUNT="$($PYTHON_BIN -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$TEST_FILE")"
if [[ "$TEST_COUNT" != "400" ]]; then echo "[FATAL] expected 400 DeepScaler items, got $TEST_COUNT" >&2; exit 4; fi
for level in 1 2 3 4; do
  packet="$PROJECT_ROOT/output/stage1/02_learning_packets/$MODEL_NAME/level${level}_learning_packets.json"
  if [[ ! -s "$packet" ]]; then echo "[FATAL] missing Level $level packet: $packet" >&2; exit 4; fi
done
AVAILABLE_KB="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print int($4/1024)}')"
if (( AVAILABLE_KB < 20480 )); then echo "[FATAL] less than 20GB free under $RUN_ROOT" >&2; exit 5; fi

{
  printf 'run_id=%s\nmodel=%s\nports=%s\n' "$RUN_ID" "$MODEL_NAME" "${PORTS[*]}"
  printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  sha256sum "$CONFIG_FILE" "$TEST_FILE" pipeline.py modules/stage2_v3.py modules/dsd_module.py modules/execute_module.py template/prompt_template.py
} > "$RUN_ROOT/run_manifest.txt"

if [[ "$PILOT_ONLY" == "1" ]]; then
  run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/pilot_tools.py" manifest \
    --run-root "$RUN_ROOT" --model "$MODEL_NAME" \
    --per-level "$PILOT_SIZE_PER_LEVEL" --seed "$PILOT_SEED"
  if [[ "$ARCHIVE_OLD_MEMORY" == "1" && ! -f "$RUN_ROOT/.old_memory_archived" ]]; then
    archive_root="$PROJECT_ROOT/archive/pre_${RUN_ID}"
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN archive old Memory and Stage-2 logs under $archive_root"
    else
      mkdir -p "$archive_root/memory" "$archive_root/logs"
      for old_path in memory/v3 memory/dpo_level1 memory/dpo_level2 memory/dpo_level3 memory/dpo_level4; do
        if [[ -e "$PROJECT_ROOT/$old_path" ]]; then
          mv "$PROJECT_ROOT/$old_path" "$archive_root/memory/$(basename "$old_path")"
        fi
      done
      if [[ -e "$PROJECT_ROOT/logs/stage2" ]]; then
        mv "$PROJECT_ROOT/logs/stage2" "$archive_root/logs/stage2"
      fi
      mkdir -p "$PROJECT_ROOT/memory/v3" "$PROJECT_ROOT/logs/stage2"
      printf '%s\n' "$archive_root" > "$RUN_ROOT/.old_memory_archived"
      log "Archived old Memory and Stage-2 logs under $archive_root"
    fi
  fi

  for level in 1 2 3 4; do
    cache="$RUN_ROOT/dsd_cache/dpo_level${level}"
    selection="$RUN_ROOT/selection/level${level}.json"
    audit="$RUN_ROOT/stage2/prepare_dsd/level${level}"
    run_cmd "$PYTHON_BIN" pipeline.py --stage 2 -d "dpo_level${level}" --ports "${PORTS[@]}" \
      --model "$MODEL_NAME" --backbone "$MODEL_NAME" --memory-dir "$RUN_ROOT/prepare_memory" \
      --stage2-concurrency "$CONCURRENCY" --stage2-candidate-concurrency "$CANDIDATE_CONCURRENCY" \
      --timeout 120 --dsd-cache-dir "$cache" --prepare-dsd-only \
      --stage2-selection-file "$selection" --stage2-output-dir "$audit"
  done

  for level in 1 2 3 4; do
    cache="$RUN_ROOT/dsd_cache/dpo_level${level}"
    bank="$RUN_ROOT/candidate_bank/dpo_level${level}.json"
    selection="$RUN_ROOT/selection/level${level}.json"
    audit="$RUN_ROOT/stage2/frozen_candidates/level${level}"
    run_cmd "$PYTHON_BIN" pipeline.py --stage 2 -d "dpo_level${level}" --ports "${PORTS[@]}" \
      --model "$MODEL_NAME" --backbone "$MODEL_NAME" --memory-dir "$RUN_ROOT/prepare_memory" \
      --stage2-concurrency "$CONCURRENCY" --stage2-candidate-concurrency "$CANDIDATE_CONCURRENCY" \
      --timeout 120 --dsd-cache-dir "$cache" --candidate-bank "$bank" --prepare-candidates-only \
      --stage2-selection-file "$selection" --stage2-output-dir "$audit"
  done

  for level in 1 2 3 4; do
    cache="$RUN_ROOT/dsd_cache/dpo_level${level}"
    bank="$RUN_ROOT/candidate_bank/dpo_level${level}.json"
    selection="$RUN_ROOT/selection/level${level}.json"
    audit="$RUN_ROOT/management_results/level${level}"
    run_cmd "$PYTHON_BIN" pipeline.py --stage 2 -d "dpo_level${level}" --ports "${PORTS[@]}" \
      --model "$MODEL_NAME" --backbone "$MODEL_NAME" --memory-dir "$RUN_ROOT/management_store" \
      --stage2-concurrency "$CONCURRENCY" --stage2-candidate-concurrency "$CANDIDATE_CONCURRENCY" \
      --timeout 120 --dsd-cache-dir "$cache" --candidate-bank "$bank" --neighbor-validation off \
      --stage2-selection-file "$selection" --stage2-output-dir "$audit"
  done

  for level in 1 2 3 4; do
    cache="$RUN_ROOT/dsd_cache/dpo_level${level}"
    bank="$RUN_ROOT/candidate_bank/dpo_level${level}.json"
    selection="$RUN_ROOT/selection/level${level}.json"
    audit="$RUN_ROOT/gate_results/level${level}"
    run_cmd "$PYTHON_BIN" pipeline.py --stage 2 -d "dpo_level${level}" --ports "${PORTS[@]}" \
      --model "$MODEL_NAME" --backbone "$MODEL_NAME" --memory-dir "$RUN_ROOT/gate_store" \
      --stage2-concurrency "$CONCURRENCY" --stage2-candidate-concurrency "$CANDIDATE_CONCURRENCY" \
      --timeout 120 --dsd-cache-dir "$cache" --candidate-bank "$bank" --neighbor-validation gate-only \
      --stage2-selection-file "$selection" --stage2-output-dir "$audit"
  done

  if run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/pilot_tools.py" report --run-root "$RUN_ROOT"; then
    log "PILOT COMPLETE and accepted: $RUN_ROOT/pilot_report.md"
    exit 0
  else
    report_status=$?
    log "PILOT STOPPED by acceptance gate (status=$report_status): $RUN_ROOT/pilot_report.md"
    exit "$report_status"
  fi
fi

for level in 1 2 3 4; do
  cache="$RUN_ROOT/dsd_cache/dpo_level${level}"
  audit="$RUN_ROOT/stage2/prepare_dsd/level${level}"
  run_cmd "$PYTHON_BIN" pipeline.py --stage 2 -d "dpo_level${level}" --ports "${PORTS[@]}" \
    --model "$MODEL_NAME" --backbone "$MODEL_NAME" --memory-dir "$RUN_ROOT/prepare_memory" --stage2-concurrency "$CONCURRENCY" --stage2-candidate-concurrency "$CANDIDATE_CONCURRENCY" \
    --timeout 120 --dsd-cache-dir "$cache" --prepare-dsd-only --stage2-output-dir "$audit"
done

for level in 1 2 3 4; do
  cache="$RUN_ROOT/dsd_cache/dpo_level${level}"
  bank="$RUN_ROOT/frozen_candidates/dpo_level${level}.json"
  audit="$RUN_ROOT/stage2/frozen_candidates/level${level}"
  run_cmd "$PYTHON_BIN" pipeline.py --stage 2 -d "dpo_level${level}" --ports "${PORTS[@]}" \
    --model "$MODEL_NAME" --backbone "$MODEL_NAME" --memory-dir "$RUN_ROOT/prepare_memory" --stage2-concurrency "$CONCURRENCY" --stage2-candidate-concurrency "$CANDIDATE_CONCURRENCY" \
    --timeout 120 --dsd-cache-dir "$cache" --candidate-bank "$bank" --prepare-candidates-only --stage2-output-dir "$audit"
done

for variant in nv_on nv_off; do
  nv_flag="on"; [[ "$variant" == "nv_off" ]] && nv_flag="off"
  for level in 1 2 3 4; do
    cache="$RUN_ROOT/dsd_cache/dpo_level${level}"
    bank="$RUN_ROOT/frozen_candidates/dpo_level${level}.json"
    audit="$RUN_ROOT/memory/$variant/audit/level${level}"
    run_cmd "$PYTHON_BIN" pipeline.py --stage 2 -d "dpo_level${level}" --ports "${PORTS[@]}" \
      --model "$MODEL_NAME" --backbone "$MODEL_NAME" --memory-dir "$RUN_ROOT/memory/$variant/store" --stage2-concurrency "$CONCURRENCY" --stage2-candidate-concurrency "$CANDIDATE_CONCURRENCY" \
      --timeout 120 --dsd-cache-dir "$cache" --candidate-bank "$bank" --neighbor-validation "$nv_flag" --stage2-output-dir "$audit"
  done
  run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/validate_store.py" "$RUN_ROOT/memory/$variant/store"
done

run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/pass3_eval.py" --variant baseline --test-file "$TEST_FILE" \
  --output-dir "$RUN_ROOT/stage3/baseline" --ports "${PORTS[@]}" --model "$MODEL_NAME" \
  --concurrency "$CONCURRENCY" --temperature 0.2
for variant in nv_on nv_off; do
  run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/pass3_eval.py" --variant "$variant" --test-file "$TEST_FILE" \
    --output-dir "$RUN_ROOT/stage3/$variant" --memory-dir "$RUN_ROOT/memory/$variant/store" \
    --ports "${PORTS[@]}" --model "$MODEL_NAME" --concurrency "$CONCURRENCY" --temperature 0.2
done
run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/analyze.py" --run-root "$RUN_ROOT"
log "COMPLETE report=$RUN_ROOT/overnight_summary.md"

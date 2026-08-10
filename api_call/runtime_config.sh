#!/usr/bin/env bash
# Shared deployment topology loader. Source this file, then call:
#   load_deployment_runtime /absolute/or/relative/deploy.conf

load_deployment_runtime() {
  local config_file="$1"
  if [[ ! -f "$config_file" ]]; then
    echo "[FATAL] deployment config not found: $config_file" >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "$config_file"

  DEPLOY_SELECTED_MODEL="${MEMUA_MODEL_NAME:-${DEPLOY_MODEL:-llama}}"
  DEPLOY_SELECTED_SERVER_SEED="${MEMUA_SERVER_SEED:-${DEPLOY_SERVER_SEED:-${VLLM_SEED:-0}}}"
  DEPLOY_SELECTED_PORT_BASE="${MEMUA_PORT_BASE:-${DEPLOY_PORT_BASE:-8100}}"

  if [[ ! "$DEPLOY_SELECTED_PORT_BASE" =~ ^[0-9]+$ ]] ||
     ((DEPLOY_SELECTED_PORT_BASE < 1 || DEPLOY_SELECTED_PORT_BASE > 65535)); then
    echo "[FATAL] invalid DEPLOY_PORT_BASE: $DEPLOY_SELECTED_PORT_BASE" >&2
    return 1
  fi

  local gpu_text="${MEMUA_GPU_IDS:-${DEPLOY_GPU_IDS:-}}"
  if [[ -z "${gpu_text// }" ]]; then
    echo "[FATAL] no GPU IDs configured; set DEPLOY_GPU_IDS in $config_file" >&2
    return 1
  fi

  read -r -a DEPLOY_SELECTED_GPU_IDS <<< "$gpu_text"
  DEPLOY_SELECTED_PORTS=()
  local gpu
  local -A seen_gpus=()
  for gpu in "${DEPLOY_SELECTED_GPU_IDS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "[FATAL] invalid GPU ID in DEPLOY_GPU_IDS: $gpu" >&2
      return 1
    fi
    if [[ -n "${seen_gpus[$gpu]:-}" ]]; then
      echo "[FATAL] duplicate GPU ID in DEPLOY_GPU_IDS: $gpu" >&2
      return 1
    fi
    seen_gpus[$gpu]=1
    local port=$((DEPLOY_SELECTED_PORT_BASE + gpu))
    if ((port > 65535)); then
      echo "[FATAL] derived port exceeds 65535: $DEPLOY_SELECTED_PORT_BASE + $gpu" >&2
      return 1
    fi
    DEPLOY_SELECTED_PORTS+=("$port")
  done

  # Optional runtime override for a remote or nonstandard endpoint layout.
  # Local deployment normally uses DEPLOY_PORT_BASE + GPU ID.
  if [[ -n "${MEMUA_PORTS:-}" ]]; then
    read -r -a DEPLOY_SELECTED_PORTS <<< "$MEMUA_PORTS"
    if [[ ${#DEPLOY_SELECTED_PORTS[@]} -ne ${#DEPLOY_SELECTED_GPU_IDS[@]} ]]; then
      echo "[FATAL] MEMUA_PORTS and MEMUA_GPU_IDS must contain the same number of values" >&2
      return 1
    fi
    local port
    for port in "${DEPLOY_SELECTED_PORTS[@]}"; do
      if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
        echo "[FATAL] invalid port in MEMUA_PORTS: $port" >&2
        return 1
      fi
    done
  fi
}

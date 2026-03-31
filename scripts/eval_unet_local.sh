#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_PATH="${1:?Usage: bash scripts/eval_unet_local.sh /path/to/checkpoint.pt [config-name]}"
shift

CONFIG_NAME="${CRISP_EXPERIMENT_CONFIG:-experiment/task_local_unet_crisp}"
if [[ $# -gt 0 ]]; then
  case "$1" in
    -*|*=*|+*=*)
      ;;
    *)
      CONFIG_NAME="$1"
      shift
      ;;
  esac
fi

python -m crisp.scripts.evaluate \
  --config-path "$ROOT_DIR/configs" \
  --config-name "$CONFIG_NAME" \
  "+checkpoint=$CHECKPOINT_PATH" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/_thesis_smoke_common.sh"

CHECKPOINT_PATH="${1:?Usage: bash scripts/eval_thesis_unetpp.sh /path/to/checkpoint.pt [hydra overrides...]}"
shift
CONFIG_NAME="${CRISP_EVAL_CONFIG:-experiment/thesis_unetpp_crisp}"
if [[ -z "${CRISP_EVAL_CONFIG:-}" && "$CHECKPOINT_PATH" == *baseline* ]]; then
  CONFIG_NAME="experiment/thesis_unetpp_baseline"
fi

PYTHON_BIN="$(crisp_resolve_python)"
EXTRA_ARGS=()
if [[ -z "${CRISP_DATA_ROOT:-}" ]] && ! crisp_arg_present "source_data.root=" "$@"; then
  SMOKE_ROOT="$(crisp_ensure_tiny_smoke_data "$PYTHON_BIN")"
  EXTRA_ARGS=(
    "source_data.root=$SMOKE_ROOT"
    "source_data.image_size=64"
    "source_data.num_workers=0"
    "source_data.pin_memory=false"
    "eval_datasets=[Kvasir]"
    "eval_output_dir=/tmp/crisp_thesis_unetpp_eval"
  )
fi

"$PYTHON_BIN" -m crisp.scripts.evaluate \
  --config-path "$ROOT_DIR/configs" \
  --config-name "$CONFIG_NAME" \
  "+checkpoint=$CHECKPOINT_PATH" \
  "${EXTRA_ARGS[@]}" \
  "$@"

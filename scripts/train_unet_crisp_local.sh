#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUESTED_INIT_CKPT="${CRISP_LOCAL_UNET_CRISP_INIT_CKPT:-}"
FALLBACK_INIT_CKPT="$ROOT_DIR/checkpoints/students/unet/init/unet_init.pt"
FALLBACK_ARGS=()

if [[ -n "$REQUESTED_INIT_CKPT" && ! -f "$REQUESTED_INIT_CKPT" ]]; then
  if [[ -f "$FALLBACK_INIT_CKPT" ]]; then
    printf '%s\n' \
      "Requested CRISP init checkpoint was not found: $REQUESTED_INIT_CKPT" \
      "Falling back to local smoke-test init checkpoint: $FALLBACK_INIT_CKPT" \
      "Fallback init is loaded in non-strict mode for smoke testing." \
      "Set CRISP_LOCAL_UNET_CRISP_INIT_CKPT to an existing baseline best.pt when you want baseline-initialized CRISP." \
      >&2
    export CRISP_LOCAL_UNET_CRISP_INIT_CKPT="$FALLBACK_INIT_CKPT"
    FALLBACK_ARGS=("student_init.strict=false")
  else
    printf '%s\n' \
      "Requested CRISP init checkpoint was not found: $REQUESTED_INIT_CKPT" \
      "No fallback init checkpoint was found at: $FALLBACK_INIT_CKPT" \
      >&2
    exit 1
  fi
fi

python -m crisp.scripts.train \
  --config-path "$ROOT_DIR/configs" \
  --config-name experiment/task_local_unet_crisp \
  "${FALLBACK_ARGS[@]}" \
  "$@"

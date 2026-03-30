#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_PATH="${1:?Usage: bash scripts/eval_pranet_crossdomain.sh /path/to/checkpoint.pt}"
shift

python -m crisp.scripts.evaluate \
  --config-path "$ROOT_DIR/configs" \
  --config-name experiment/taskA_pranet_crisp \
  "+checkpoint=$CHECKPOINT_PATH" \
  "$@"

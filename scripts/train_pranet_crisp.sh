#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m crisp.scripts.train \
  --config-path "$ROOT_DIR/configs" \
  --config-name experiment/taskA_pranet_crisp \
  "$@"

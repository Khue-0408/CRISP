#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m crisp.scripts.export_tables \
  --input-dir "$ROOT_DIR/outputs/metrics" \
  --output-dir "$ROOT_DIR/outputs/tables" \
  "$@"

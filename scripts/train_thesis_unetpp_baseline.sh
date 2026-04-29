#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/_thesis_smoke_common.sh"

PYTHON_BIN="$(crisp_resolve_python)"
EXTRA_ARGS=()
if crisp_should_use_tiny_smoke_data "$@"; then
  SMOKE_ROOT="$(crisp_ensure_tiny_smoke_data "$PYTHON_BIN")"
  EXTRA_ARGS=(
    "source_data.root=$SMOKE_ROOT"
    "source_data.image_size=64"
    "source_data.num_workers=0"
    "source_data.pin_memory=false"
    "source_data.local_split.cache_dir=/tmp/crisp_thesis_smoke_splits_unetpp_baseline"
    "eval_datasets=[Kvasir]"
    "output_dir=/tmp/crisp_thesis_unetpp_baseline_out"
    "eval_output_dir=/tmp/crisp_thesis_unetpp_baseline_eval"
  )
fi

"$PYTHON_BIN" -m crisp.scripts.train \
  --config-path "$ROOT_DIR/configs" \
  --config-name experiment/thesis_unetpp_baseline \
  "${EXTRA_ARGS[@]}" \
  "$@"

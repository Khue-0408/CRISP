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
    "source_data.local_split.cache_dir=/tmp/crisp_thesis_smoke_splits_pranet_crisp"
    "eval_datasets=[Kvasir]"
    "output_dir=/tmp/crisp_thesis_pranet_crisp_out"
    "eval_output_dir=/tmp/crisp_thesis_pranet_crisp_eval"
    "crisp.schedule.phase_i_epochs=0"
    "crisp.schedule.phase_ii_epochs=1"
    "crisp.schedule.phase_iii_epochs=0"
    "crisp.schedule.phase_ii_ramp_epochs=1"
    "training.phases.baseline_warmup=0"
    "training.phases.crisp_full=1"
    "training.phases.finetune=0"
    "training.phases.phase2_ramp_epochs=1"
  )
fi

"$PYTHON_BIN" -m crisp.scripts.train \
  --config-path "$ROOT_DIR/configs" \
  --config-name experiment/thesis_pranet_crisp \
  "${EXTRA_ARGS[@]}" \
  "$@"

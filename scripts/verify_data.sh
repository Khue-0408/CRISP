#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/_thesis_smoke_common.sh"

PYTHON_BIN="$(crisp_resolve_python)"
"$PYTHON_BIN" -m crisp.scripts.verify_data "$@"

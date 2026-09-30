#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_crisp_five_seeds.sh <unet|unetpp|pranet> <baseline|crisp> [hydra overrides...]" >&2
  exit 2
fi

HOST="$1"
MODE="$2"
shift 2

for arg in "$@"; do
  case "$arg" in
    seed=*|+seed=*|++seed=*)
      echo "Seed overrides are not allowed; the runner locks seeds 2026-2030." >&2
      exit 2
      ;;
    -m|--multirun|--multirun=*|hydra.mode=*|+hydra.mode=*|++hydra.mode=*|hydra.sweep.*|+hydra.sweep.*|++hydra.sweep.*|hydra.sweeper.*|+hydra.sweeper.*|++hydra.sweeper.*)
      echo "Hydra multirun and sweep controls are not supported by the five-seed runner." >&2
      exit 2
      ;;
  esac
done

case "$HOST:$MODE" in
  unet:baseline) WRAPPER="train_crisp_unet_baseline.sh" ;;
  unet:crisp) WRAPPER="train_crisp_unet_crisp.sh" ;;
  unetpp:baseline) WRAPPER="train_crisp_unetpp_baseline.sh" ;;
  unetpp:crisp) WRAPPER="train_crisp_unetpp_crisp.sh" ;;
  pranet:baseline) WRAPPER="train_crisp_pranet_baseline.sh" ;;
  pranet:crisp) WRAPPER="train_crisp_pranet_crisp.sh" ;;
  *)
    echo "Unsupported host/mode: $HOST/$MODE" >&2
    exit 2
    ;;
esac

for seed in 2026 2027 2028 2029 2030; do
  echo "Running $HOST/$MODE with seed=$seed"
  bash "$ROOT_DIR/scripts/$WRAPPER" "seed=$seed" "$@"
done

#!/usr/bin/env bash
# Git checkout entry point; no ZIP patch manifest or automatic result packaging.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
task="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$task" in
  mainline)
    python -m src.v3.balance_experiment --scope barrier --folds 0 1 2 3 4 \
      --modes subsample --members 8 --ratios 1:3 \
      --out results/v3_mainline_barrier84_1to3_s42 "$@"
    ;;
  verify) python -m src.v3.train verify "$@" ;;
  check) python -m src.v3.train check "$@" ;;
  smoke) python -m src.v3.train smoke "$@" ;;
  train) python -m src.v3.train train "$@" ;;
  drawdown) python -m src.v3.train_drawdown84 train "$@" ;;
  controls)
    python -m src.v3.controls check "$@"
    python -m src.v3.controls train "$@"
    ;;
  explain)
    python -m src.v3.explain check "$@"
    python -m src.v3.explain train "$@"
    ;;
  canonical)
    python -m src.v3.explain84_canonical check "$@"
    # Recover existing fitted models without fitting before replay verification.
    python -m src.v3.explain84_canonical recover "$@"
    python -m src.v3.explain84_canonical verify-existing "$@"
    python -m src.v3.explain84_canonical prepare "$@"
    python -m src.v3.explain84_canonical verify-models "$@"
    python -m src.v3.explain84_canonical train "$@"
    ;;
  *)
    echo 'Usage: bash scripts/run_v3.sh {mainline|verify|check|smoke|train|drawdown|controls|explain|canonical} [module options]'
    [[ "$task" == help ]] || exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/verify_v3_explain84_saved_patch.py
python -m src.v3.explain84_saved check
python -m src.v3.explain84_saved prepare --out results/v3_explain84_saved_s42
python -m src.v3.explain84_saved verify-models --out results/v3_explain84_saved_s42
python -m src.v3.explain84_saved train --out results/v3_explain84_saved_s42
if [[ ! -f results/v3_explain84_saved_s42_return.zip ]]; then
  python -m src.v3.explain84_saved export --out results/v3_explain84_saved_s42
fi
echo 'V3 SAVED84 COMPLETE: results/v3_explain84_saved_s42_return.zip'

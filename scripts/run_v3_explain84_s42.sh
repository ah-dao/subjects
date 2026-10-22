#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/verify_v3_explain84_patch.py
python -m src.v3.explain84 check
python -m src.v3.explain84 train --protocol both --max-per-fold 1024 --repeats 2 --out results/v3_explain84_s42
if [[ ! -f results/v3_explain84_s42_return.zip ]]; then
  python -m src.v3.explain84 export --out results/v3_explain84_s42
fi
echo 'V3 ALL84 EXPLANATION COMPLETE: results/v3_explain84_s42_return.zip'

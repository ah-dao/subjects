#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/verify_v3_explain_patch.py
python -m src.v3.explain check
python -m src.v3.explain train --protocol both --repeats 2 --max-per-fold 1024 --out results/v3_explain_s42
if [[ ! -f results/v3_explain_s42_return.zip ]]; then
  python -m src.v3.explain export --out results/v3_explain_s42
fi
echo 'V3 EXPLANATION COMPLETE: results/v3_explain_s42_return.zip'

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -m src.v3.train check
python -m src.v3.train smoke --out results/v3_smoke84
python -m src.v3.train train --protocol both --seeds 42 43 44 --out results/v3_stats84_4k
if [[ ! -f results/v3_stats84_4k_return.zip ]]; then
  python -m src.v3.train export --out results/v3_stats84_4k
fi
echo 'V3 SUITE COMPLETE: results/v3_stats84_4k_return.zip'

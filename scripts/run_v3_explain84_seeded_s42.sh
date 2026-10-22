#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/verify_v3_explain84_seeded_patch.py
python -m src.v3.explain84_seeded check
# No fitting in this stage. A failed recovery stops the entire script here.
python -m src.v3.explain84_seeded recover --recover-from results/v3_explain84_saved_s42 --out results/v3_explain84_seeded_s42
python -m src.v3.explain84_seeded prepare --out results/v3_explain84_seeded_s42
python -m src.v3.explain84_seeded verify-models --out results/v3_explain84_seeded_s42
python -m src.v3.explain84_seeded train --out results/v3_explain84_seeded_s42
if [[ ! -f results/v3_explain84_seeded_s42_return.zip ]]; then
  python -m src.v3.explain84_seeded export --out results/v3_explain84_seeded_s42
fi
echo 'V3 SEEDED RELOAD84 COMPLETE: results/v3_explain84_seeded_s42_return.zip'

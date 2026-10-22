#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/verify_v3_explain84_canonical_patch.py
python -m src.v3.explain84_canonical check
# Existing fitted files only; no fit in either of these first two stages.
python -m src.v3.explain84_canonical recover --recover-from results/v3_explain84_saved_s42 --out results/v3_explain84_canonical_s42
python -m src.v3.explain84_canonical verify-existing --out results/v3_explain84_canonical_s42
python -m src.v3.explain84_canonical prepare --out results/v3_explain84_canonical_s42
python -m src.v3.explain84_canonical verify-models --out results/v3_explain84_canonical_s42
python -m src.v3.explain84_canonical train --out results/v3_explain84_canonical_s42
if [[ ! -f results/v3_explain84_canonical_s42_return.zip ]]; then
  python -m src.v3.explain84_canonical export --out results/v3_explain84_canonical_s42
fi
echo 'V3 CANONICAL LOADED84 COMPLETE: results/v3_explain84_canonical_s42_return.zip'

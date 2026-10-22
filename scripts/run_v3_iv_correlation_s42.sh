#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python scripts/verify_v3_iv_correlation_patch.py
python tools/v3_iv_correlation.py run --out results/v3_iv_correlation_s42
echo 'V3 IV CORRELATION COMPLETE: results/v3_iv_correlation_s42_return.zip'

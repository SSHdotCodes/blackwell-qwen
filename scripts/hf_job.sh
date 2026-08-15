#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT=/results/latest

cd "${RUN_ROOT}"

export BLACKWELL_QWEN_IMAGE="lmsysorg/sglang:qwen38-27b"
export PYTHONPATH="${RUN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m tuner.autotune \
  --model-path /model \
  --output "${RESULT_ROOT}" \
  --budget-seconds "${JOB_BUDGET_SECONDS:-10620}"

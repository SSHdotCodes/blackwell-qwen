#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="/results/${RESULT_RUN:-latest}"
LOCAL_MODEL=/tmp/Qwen3.8-27B-FP8
LOCAL_DRAFT=/tmp/Qwen3.8-27B-DSpark

cd "${RUN_ROOT}"

export BLACKWELL_QWEN_IMAGE="lmsysorg/sglang:qwen38-27b"
export PYTHONPATH="${RUN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# HF model volumes are network-backed. SGLang loads safetensors concurrently and mmap-like
# reads can receive SIGBUS on that mount, so stage once onto the job's 475 GB local NVMe.
mkdir -p "${LOCAL_MODEL}"
cp -a /model/. "${LOCAL_MODEL}/"

TUNER_ARGS=(
  --model-path "${LOCAL_MODEL}"
  --output "${RESULT_ROOT}"
  --budget-seconds "${JOB_BUDGET_SECONDS:-10620}"
  --candidate-set "${CANDIDATE_SET:-all}"
)

if [[ -d /draft ]]; then
  mkdir -p "${LOCAL_DRAFT}"
  cp -a /draft/. "${LOCAL_DRAFT}/"
  TUNER_ARGS+=(--draft-model-path "${LOCAL_DRAFT}")
fi

python3 -m tuner.autotune "${TUNER_ARGS[@]}"

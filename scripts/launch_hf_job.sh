#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${HF_NAMESPACE:-ProCreations}"
BUCKET="${HF_RESULTS_BUCKET:-blackwell-qwen}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
JOB_TIMEOUT="${HF_JOB_TIMEOUT:-3h}"
JOB_BUDGET="${JOB_BUDGET_SECONDS:-10620}"

if hf jobs ps --format json | python3 -c 'import json,sys; raise SystemExit(0 if not json.load(sys.stdin) else 1)'; then
  :
else
  echo "Refusing to launch: another Hugging Face Job is running." >&2
  exit 1
fi

hf buckets create "${NAMESPACE}/${BUCKET}" --private --exist-ok >/dev/null

# One RTX PRO 6000 at $2.75/hour for exactly a 3-hour cap = $8.25 maximum.
hf jobs run \
  --detach \
  --flavor rtx-pro-6000 \
  --timeout "${JOB_TIMEOUT}" \
  --label project=blackwell-qwen \
  --label model=qwen38-27b-fp8 \
  --env "JOB_BUDGET_SECONDS=${JOB_BUDGET}" \
  --volume hf://Qwen/Qwen3.8-27B-FP8:/model:ro \
  --volume "hf://buckets/${NAMESPACE}/${BUCKET}:/results" \
  -- \
  "${IMAGE}" \
  bash -lc 'git clone --depth 1 https://github.com/SSHDotCodes/blackwell-qwen.git /tmp/blackwell-qwen && exec bash /tmp/blackwell-qwen/scripts/hf_job.sh'

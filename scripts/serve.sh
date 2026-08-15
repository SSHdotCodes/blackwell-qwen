#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.8-27B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B-FP8}"
PROFILE="${PROFILE:-throughput}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/config/selected_profiles.env"

case "${PROFILE}" in
  latency)
    read -r -a PROFILE_ARGS <<< "${LATENCY_FLAGS}"
    ;;
  throughput)
    read -r -a PROFILE_ARGS <<< "${THROUGHPUT_FLAGS}"
    ;;
  *)
    echo "PROFILE must be latency or throughput" >&2
    exit 2
    ;;
esac

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --context-length 262144 \
  --mem-fraction-static 0.90 \
  --kv-cache-dtype bfloat16 \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  "${PROFILE_ARGS[@]}" \
  "$@"


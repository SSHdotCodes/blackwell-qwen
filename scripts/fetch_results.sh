#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${HF_NAMESPACE:-ProCreations}"
BUCKET="${HF_RESULTS_BUCKET:-blackwell-qwen}"

mkdir -p "${REPO_ROOT}/results/download"
hf buckets sync \
  "hf://buckets/${NAMESPACE}/${BUCKET}/latest" \
  "${REPO_ROOT}/results/download"


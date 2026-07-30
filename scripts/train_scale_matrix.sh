#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${1:-configs/training/qwen3_1.7b.yaml}"
shift || true
for scale in 0.1 0.3 0.5 1.0; do
  python -m eviback.training.entrypoint \
    --config "${CONFIG}" \
    --teacher-fallback-scale "${scale}" \
    "$@"
done
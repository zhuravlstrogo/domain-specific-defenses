#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-$OPENROUTER_API_KEY}"
    export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
fi

CONFIG="${CONFIG:-configs/experiments/cares_qwen2_5_7b.yaml}"

args=(python scripts/run_experiment_matrix.py "$CONFIG")

if [[ -n "${LIMIT:-}" ]]; then
    args+=(--limit "$LIMIT")
fi
if [[ -n "${SEED:-}" ]]; then
    args+=(--sample-shuffle "$SEED")
fi
if [[ -n "${RUNTIME:-}" ]]; then
    args+=(--runtime "$RUNTIME")
fi
if [[ -n "${MODEL_LABEL:-}" ]]; then
    args+=(--model-label "$MODEL_LABEL")
fi
if [[ -n "${DATASET_PATH:-}" ]]; then
    args+=(--dataset-path "$DATASET_PATH")
fi
if [[ -n "${DATASET_SPLIT:-}" ]]; then
    args+=(--dataset-split "$DATASET_SPLIT")
fi
if [[ -n "${DATASET_SIZE:-}" ]]; then
    args+=(--dataset-size "$DATASET_SIZE")
fi
if [[ -n "${DATASET_SEED:-}" ]]; then
    args+=(--dataset-seed "$DATASET_SEED")
fi
if [[ -n "${LOG_ROOT:-}" ]]; then
    args+=(--log-root "$LOG_ROOT")
fi
if [[ -n "${REPORT_CSV:-}" ]]; then
    args+=(--report-csv "$REPORT_CSV")
fi
if [[ -n "${REPORT_MD:-}" ]]; then
    args+=(--report-md "$REPORT_MD")
fi
if [[ -n "${PREPARE_DATASET:-}" ]]; then
    args+=(--prepare-dataset "$PREPARE_DATASET")
fi
if [[ "${DRY_RUN:-}" == "1" ]]; then
    args+=(--dry-run)
fi
if [[ "${SKIP_REPORT:-}" == "1" ]]; then
    args+=(--skip-report)
fi

"${args[@]}"

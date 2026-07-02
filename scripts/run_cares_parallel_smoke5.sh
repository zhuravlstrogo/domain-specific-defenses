#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source scripts/cares_env.sh

export LIMIT="${LIMIT:-5}"
export DATASET_SIZE="${DATASET_SIZE:-5}"
export EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-parallel_smoke_limit5}"
export FORCE="${FORCE:-0}"
export RESUME="${RESUME:-1}"
export WAIT_FOR_INFERENCE="${WAIT_FOR_INFERENCE:-1}"
export INFERENCE_WAIT_TIMEOUT_SECONDS="${INFERENCE_WAIT_TIMEOUT_SECONDS:-7200}"
export INFERENCE_WAIT_POLL_SECONDS="${INFERENCE_WAIT_POLL_SECONDS:-10}"
export QWEN3_RUNTIME="${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}"
export GEMMA_RUNTIME="${GEMMA_RUNTIME:-a10_hf_gemma_3_4b_it_openrouter_judge}"
export OLMO_RUNTIME="${OLMO_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_openrouter_judge}"

echo "==> parallel smoke: inference and judge"
echo "LIMIT=$LIMIT DATASET_SIZE=$DATASET_SIZE EXPERIMENT_SUFFIX=$EXPERIMENT_SUFFIX"

bash scripts/run_cares_model_inference.sh &
inference_pid=$!

bash scripts/run_cares_judge_models.sh &
judge_pid=$!

inference_status=0
judge_status=0

wait "$inference_pid" || inference_status=$?
if [[ "$inference_status" -ne 0 ]]; then
    echo "Inference failed with status $inference_status; stopping judge process $judge_pid." >&2
    kill "$judge_pid" 2>/dev/null || true
fi

wait "$judge_pid" || judge_status=$?

if [[ "$inference_status" -ne 0 ]]; then
    exit "$inference_status"
fi
exit "$judge_status"

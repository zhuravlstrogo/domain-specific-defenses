#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source scripts/cares_env.sh

export LIMIT="${LIMIT:-5}"
export DATASET_SIZE="${DATASET_SIZE:-5}"
export EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-smoke_limit5}"
export FORCE="${FORCE:-0}"
export RESUME="${RESUME:-1}"
export QWEN3_RUNTIME="${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}"
export GEMMA_RUNTIME="${GEMMA_RUNTIME:-a10_hf_gemma_3_1b_it_openrouter_judge}"
export OLMO_RUNTIME="${OLMO_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_openrouter_judge}"

bash scripts/run_cares_model_inference.sh
bash scripts/run_cares_judge_models.sh

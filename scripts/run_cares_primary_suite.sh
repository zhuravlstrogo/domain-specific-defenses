#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIMIT="${LIMIT:-100}"
DATASET_SIZE="${DATASET_SIZE:-300}"
SEED="${SEED:-42}"

run_case() {
    local config="$1"

    CONFIG="$config" \
    LIMIT="$LIMIT" \
    DATASET_SIZE="$DATASET_SIZE" \
    SEED="$SEED" \
    bash scripts/run_cares_experiments.sh
}

run_case "configs/experiments/cares_qwen3_1_7b.yaml"
run_case "configs/experiments/cares_gemma_2_2b_it.yaml"
run_case "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIMIT="${LIMIT:-100}"
DATASET_SIZE="${DATASET_SIZE:-300}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"

run_case() {
    local config="$1"
    local report_base="$2"

    if [[ "$FORCE" != "1" ]] \
        && [[ -f "${report_base}.md" ]] \
        && [[ -f "${report_base}.csv" ]]; then
        echo "==> skip completed: $config"
        echo "    ${report_base}.md"
        return
    fi

    CONFIG="$config" \
    LIMIT="$LIMIT" \
    DATASET_SIZE="$DATASET_SIZE" \
    SEED="$SEED" \
    RESUME="$RESUME" \
    bash scripts/run_cares_experiments.sh
}

if [[ "$FORCE" == "1" ]]; then
    RESUME=0
fi

run_case \
    "configs/experiments/cares_qwen3_1_7b.yaml" \
    "reports/results/cares_qwen3_1_7b_guard_qwen3_guard_0_6b_judge_qwen_2_5_72b_instruct_seed42_limit100"
run_case \
    "configs/experiments/cares_gemma_2_2b_it.yaml" \
    "reports/results/cares_gemma_2_2b_it_guard_qwen3_guard_0_6b_judge_qwen_2_5_72b_instruct_seed42_limit100"
run_case \
    "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml" \
    "reports/results/cares_olmo2_0425_1b_instruct_guard_qwen3_guard_0_6b_judge_qwen_2_5_72b_instruct_seed42_limit100"

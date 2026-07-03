#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source scripts/cares_env.sh

SEED="${SEED:-42}"
PREPARE_DATASET="${PREPARE_DATASET:-auto}"
JUDGE_MODEL_KEY="${JUDGE_MODEL_KEY:-gpt-oss-120b}"
RESUME="${RESUME:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"

if [[ -n "${MODEL_LABEL:-}" || -n "${LOG_ROOT:-}" || -n "${REPORT_MD:-}" || -n "${REPORT_CSV:-}" ]]; then
    echo "MODEL_LABEL, LOG_ROOT, REPORT_MD, and REPORT_CSV must be unset for this runner." >&2
    exit 1
fi

mkdir -p logs/runner

timestamp_utc() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

sanitize_key() {
    local value="$1"
    echo "${value//[^A-Za-z0-9_]/_}"
}

dataset_path_for() {
    local offset="$1"
    local limit="$2"
    if [[ "$offset" == "0" && "$limit" == "300" ]]; then
        echo "data/processed/cares_18k_v1.jsonl"
    else
        echo "data/processed/cares_18k_v1_offset${offset}_limit${limit}.jsonl"
    fi
}

experiment_suffix_for() {
    local offset="$1"
    local limit="$2"
    if [[ "$offset" == "0" && "$limit" == "300" ]]; then
        echo ""
    else
        echo "offset${offset}_limit${limit}"
    fi
}

inference_root_for() {
    local log_label="$1"
    local limit="$2"
    local suffix="$3"
    if [[ -n "$suffix" ]]; then
        echo "logs/cares/cares_${log_label}_seed42_limit${limit}_${suffix}"
    else
        echo "logs/cares/cares_${log_label}_seed42_limit${limit}"
    fi
}

base_args_for() {
    local offset="$1"
    local limit="$2"
    local dataset_path="$3"
    local suffix="$4"

    BASE_ARGS=(
        --sample-shuffle "$SEED"
        --limit "$limit"
        --dataset-size "$limit"
        --dataset-offset "$offset"
        --dataset-path "$dataset_path"
        --prepare-dataset "$PREPARE_DATASET"
    )
    if [[ -n "$suffix" ]]; then
        BASE_ARGS+=(--experiment-suffix "$suffix")
    fi
    if [[ "$RESUME" == "1" ]]; then
        BASE_ARGS+=(--resume)
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        BASE_ARGS+=(--dry-run)
    fi
}

run_inference_case() {
    local config="$1"
    local model_id="$2"
    local runtime="$3"
    local offset="$4"
    local limit="$5"
    local dataset_path="$6"
    local suffix="$7"
    local runner_log="logs/runner/cares_${model_id}_inference_${suffix:-limit${limit}}.log"

    base_args_for "$offset" "$limit" "$dataset_path" "$suffix"
    echo "==> inference start: $(timestamp_utc) | model: $model_id | runtime: $runtime | offset: $offset | limit: $limit"
    python scripts/run_experiment_matrix.py "$config" \
        --runtime "$runtime" \
        "${BASE_ARGS[@]}" \
        --skip-scorer \
        --skip-report \
        2>&1 | tee -a "$runner_log"
    echo "==> inference finish: $(timestamp_utc) | model: $model_id | offset: $offset | limit: $limit"
}

run_judge_case() {
    local config="$1"
    local model_id="$2"
    local runtime="$3"
    local offset="$4"
    local limit="$5"
    local dataset_path="$6"
    local suffix="$7"
    local inference_log_root="$8"
    local safe_judge
    safe_judge="$(sanitize_key "$JUDGE_MODEL_KEY")"
    local runner_log="logs/runner/cares_${model_id}_judge_${safe_judge}_${suffix:-limit${limit}}.log"

    base_args_for "$offset" "$limit" "$dataset_path" "$suffix"
    echo "==> judge start: $(timestamp_utc) | model: $model_id | judge: $JUDGE_MODEL_KEY | score-from: $inference_log_root"
    python scripts/run_experiment_matrix.py "$config" \
        --runtime "$runtime" \
        "${BASE_ARGS[@]}" \
        --judge-model-key "$JUDGE_MODEL_KEY" \
        --score-from-log-root "$inference_log_root" \
        2>&1 | tee -a "$runner_log"
    echo "==> judge finish: $(timestamp_utc) | model: $model_id | offset: $offset | limit: $limit"
}

# Full 3000-sample target by model:
# - Qwen and OLMo already have complete inference chunks 0..299 and 300..999.
#   Run only the missing inference chunk 1000..2999.
# - Gemma 3 1B has no local full inference logs yet. Run 0..2999.
#
# Fields:
#   config|model_id|runtime|log_label|offset|limit
inference_cases=(
    "configs/experiments/cares_qwen3_1_7b.yaml|qwen3_1_7b|${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}|qwen3_1_7b_guard_qwen3_guard_0_6b|1000|2000"
    "configs/experiments/cares_gemma_3_1b_it.yaml|gemma_3_1b_it|${GEMMA_RUNTIME:-a10_hf_gemma_3_1b_it_openrouter_judge}|gemma_3_1b_it_guard_qwen3_guard_0_6b|0|3000"
    "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml|olmo2_0425_1b_instruct|${OLMO_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_openrouter_judge}|olmo2_0425_1b_instruct_guard_qwen3_guard_0_6b|1000|2000"
)

# Judge every chunk needed to reach 3000 with the same judge model. Existing
# scored chunks will be resumed/skipped by run_experiment_matrix where possible.
judge_cases=(
    "configs/experiments/cares_qwen3_1_7b.yaml|qwen3_1_7b|${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}|qwen3_1_7b_guard_qwen3_guard_0_6b|0|300"
    "configs/experiments/cares_qwen3_1_7b.yaml|qwen3_1_7b|${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}|qwen3_1_7b_guard_qwen3_guard_0_6b|300|700"
    "configs/experiments/cares_qwen3_1_7b.yaml|qwen3_1_7b|${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}|qwen3_1_7b_guard_qwen3_guard_0_6b|1000|2000"
    "configs/experiments/cares_gemma_3_1b_it.yaml|gemma_3_1b_it|${GEMMA_RUNTIME:-a10_hf_gemma_3_1b_it_openrouter_judge}|gemma_3_1b_it_guard_qwen3_guard_0_6b|0|3000"
    "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml|olmo2_0425_1b_instruct|${OLMO_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_openrouter_judge}|olmo2_0425_1b_instruct_guard_qwen3_guard_0_6b|0|300"
    "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml|olmo2_0425_1b_instruct|${OLMO_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_openrouter_judge}|olmo2_0425_1b_instruct_guard_qwen3_guard_0_6b|300|700"
    "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml|olmo2_0425_1b_instruct|${OLMO_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_openrouter_judge}|olmo2_0425_1b_instruct_guard_qwen3_guard_0_6b|1000|2000"
)

echo "CARES 3000-sample completion runner"
echo "  seed: $SEED"
echo "  judge: $JUDGE_MODEL_KEY"
echo "  run inference: $RUN_INFERENCE"
echo "  run judge: $RUN_JUDGE"
echo

if [[ "$RUN_INFERENCE" == "1" ]]; then
    for case_spec in "${inference_cases[@]}"; do
        IFS="|" read -r config model_id runtime log_label offset limit <<< "$case_spec"
        dataset_path="$(dataset_path_for "$offset" "$limit")"
        suffix="$(experiment_suffix_for "$offset" "$limit")"
        run_inference_case "$config" "$model_id" "$runtime" "$offset" "$limit" "$dataset_path" "$suffix"
    done
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
    for case_spec in "${judge_cases[@]}"; do
        IFS="|" read -r config model_id runtime log_label offset limit <<< "$case_spec"
        dataset_path="$(dataset_path_for "$offset" "$limit")"
        suffix="$(experiment_suffix_for "$offset" "$limit")"
        inference_log_root="$(inference_root_for "$log_label" "$limit" "$suffix")"
        run_judge_case "$config" "$model_id" "$runtime" "$offset" "$limit" "$dataset_path" "$suffix" "$inference_log_root"
    done
fi

echo "==> complete: $(timestamp_utc)"

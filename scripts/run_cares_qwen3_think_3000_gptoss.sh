#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source scripts/cares_env.sh

SEED="${SEED:-42}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-3000}"
PREPARE_DATASET="${PREPARE_DATASET:-auto}"
RESUME="${RESUME:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_MODEL_KEY="${JUDGE_MODEL_KEY:-gpt-oss-120b}"

CONFIG="${CONFIG:-configs/experiments/cares_qwen3_1_7b.yaml}"
RUNTIME="${QWEN3_THINK_RUNTIME:-a10_hf_qwen3_1_7b_think_openrouter_judge}"
MODEL_ID="qwen3_1_7b_think"
LOG_LABEL="qwen3_1_7b_think_guard_qwen3_guard_0_6b"

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
        echo "think"
    else
        echo "think_offset${offset}_limit${limit}"
    fi
}

base_args_for() {
    local offset="$1"
    local limit="$2"
    local dataset_path="$3"
    local suffix="$4"
    local model_label="$5"

    BASE_ARGS=(
        --sample-shuffle "$SEED"
        --limit "$limit"
        --dataset-size "$limit"
        --dataset-offset "$offset"
        --dataset-path "$dataset_path"
        --prepare-dataset "$PREPARE_DATASET"
        --experiment-suffix "$suffix"
        --model-label "$model_label"
    )
    if [[ "$RESUME" == "1" ]]; then
        BASE_ARGS+=(--resume)
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        BASE_ARGS+=(--dry-run)
    fi
}

run_inference() {
    local offset="$1"
    local limit="$2"
    local dataset_path="$3"
    local suffix="$4"
    local runner_log="logs/runner/cares_${MODEL_ID}_inference_offset${offset}_limit${limit}.log"

    base_args_for "$offset" "$limit" "$dataset_path" "$suffix" "$LOG_LABEL"
    echo "==> inference start: $(timestamp_utc) | model: $MODEL_ID | runtime: $RUNTIME | offset: $offset | limit: $limit"
    python scripts/run_experiment_matrix.py "$CONFIG" \
        --runtime "$RUNTIME" \
        "${BASE_ARGS[@]}" \
        --skip-scorer \
        --skip-report \
        2>&1 | tee -a "$runner_log"
    echo "==> inference finish: $(timestamp_utc) | model: $MODEL_ID | offset: $offset | limit: $limit"
}

run_judge() {
    local offset="$1"
    local limit="$2"
    local dataset_path="$3"
    local suffix="$4"
    local inference_log_root="$5"
    local safe_judge
    safe_judge="$(sanitize_key "$JUDGE_MODEL_KEY")"
    local runner_log="logs/runner/cares_${MODEL_ID}_judge_${safe_judge}_offset${offset}_limit${limit}.log"
    local judge_log_label="${LOG_LABEL}_judge_${safe_judge}"

    base_args_for "$offset" "$limit" "$dataset_path" "$suffix" "$judge_log_label"
    echo "==> judge start: $(timestamp_utc) | model: $MODEL_ID | judge: $JUDGE_MODEL_KEY | score-from: $inference_log_root"
    python scripts/run_experiment_matrix.py "$CONFIG" \
        --runtime "$RUNTIME" \
        "${BASE_ARGS[@]}" \
        --judge-model-key "$JUDGE_MODEL_KEY" \
        --score-from-log-root "$inference_log_root" \
        2>&1 | tee -a "$runner_log"
    echo "==> judge finish: $(timestamp_utc) | model: $MODEL_ID | offset: $offset | limit: $limit"
}

DATASET_PATH="$(dataset_path_for "$OFFSET" "$LIMIT")"
SUFFIX="$(experiment_suffix_for "$OFFSET" "$LIMIT")"
INFERENCE_LOG_ROOT="logs/cares/cares_${LOG_LABEL}_seed42_limit${LIMIT}_${SUFFIX}"

echo "CARES Qwen3 thinking 3000-sample runner"
echo "  seed: $SEED"
echo "  runtime: $RUNTIME"
echo "  judge: $JUDGE_MODEL_KEY"
echo "  offset: $OFFSET"
echo "  limit: $LIMIT"
echo "  inference log root: $INFERENCE_LOG_ROOT"
echo "  run inference: $RUN_INFERENCE"
echo "  run judge: $RUN_JUDGE"
echo

if [[ "$RUN_INFERENCE" == "1" ]]; then
    run_inference "$OFFSET" "$LIMIT" "$DATASET_PATH" "$SUFFIX"
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
    run_judge "$OFFSET" "$LIMIT" "$DATASET_PATH" "$SUFFIX" "$INFERENCE_LOG_ROOT"
fi

echo "==> complete: $(timestamp_utc)"

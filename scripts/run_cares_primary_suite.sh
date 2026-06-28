#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIMIT="${LIMIT:-100}"
DATASET_SIZE="${DATASET_SIZE:-300}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"

matrix_args_for_config() {
    local config="$1"

    MATRIX_ARGS=(
        python scripts/run_experiment_matrix.py "$config"
        --limit "$LIMIT"
        --sample-shuffle "$SEED"
        --dataset-size "$DATASET_SIZE"
    )
    if [[ -n "${RUNTIME:-}" ]]; then
        MATRIX_ARGS+=(--runtime "$RUNTIME")
    fi
    if [[ -n "${MODEL_LABEL:-}" ]]; then
        MATRIX_ARGS+=(--model-label "$MODEL_LABEL")
    fi
    if [[ -n "${DATASET_PATH:-}" ]]; then
        MATRIX_ARGS+=(--dataset-path "$DATASET_PATH")
    fi
    if [[ -n "${DATASET_SPLIT:-}" ]]; then
        MATRIX_ARGS+=(--dataset-split "$DATASET_SPLIT")
    fi
    if [[ -n "${DATASET_SEED:-}" ]]; then
        MATRIX_ARGS+=(--dataset-seed "$DATASET_SEED")
    fi
    if [[ -n "${LOG_ROOT:-}" ]]; then
        MATRIX_ARGS+=(--log-root "$LOG_ROOT")
    fi
    if [[ -n "${REPORT_CSV:-}" ]]; then
        MATRIX_ARGS+=(--report-csv "$REPORT_CSV")
    fi
    if [[ -n "${REPORT_MD:-}" ]]; then
        MATRIX_ARGS+=(--report-md "$REPORT_MD")
    fi
}

resolve_report_paths() {
    local config="$1"
    local key
    local value

    report_md=""
    report_csv=""
    matrix_args_for_config "$config"
    MATRIX_ARGS+=(--print-output-paths)

    while IFS='=' read -r key value; do
        case "$key" in
            REPORT_MD) report_md="$value" ;;
            REPORT_CSV) report_csv="$value" ;;
        esac
    done < <("${MATRIX_ARGS[@]}")

    if [[ -z "$report_md" || -z "$report_csv" ]]; then
        echo "Failed to resolve report paths for $config" >&2
        exit 1
    fi
}

run_case() {
    local config="$1"
    local report_md
    local report_csv

    resolve_report_paths "$config"

    if [[ "$FORCE" != "1" ]] \
        && [[ -f "$report_md" ]] \
        && [[ -f "$report_csv" ]]; then
        echo "==> skip completed: $config"
        echo "    $report_md"
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

run_case "configs/experiments/cares_qwen3_1_7b.yaml"
run_case "configs/experiments/cares_gemma_3_4b_it.yaml"
run_case "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml"

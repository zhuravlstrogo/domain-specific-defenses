#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIMIT="${LIMIT:-100}"
DATASET_SIZE="${DATASET_SIZE:-300}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"

if [[ -n "${JUDGE_MODEL_NAME:-}" && -n "${JUDGE_MODEL_KEY:-}" ]]; then
    echo "Use only one of JUDGE_MODEL_NAME or JUDGE_MODEL_KEY." >&2
    exit 1
fi

if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
    judge_model_keys=("")
elif [[ -n "${JUDGE_MODEL_KEYS:-}" ]]; then
    read -r -a judge_model_keys <<< "$JUDGE_MODEL_KEYS"
elif [[ -n "${JUDGE_MODEL_KEY:-}" ]]; then
    judge_model_keys=("$JUDGE_MODEL_KEY")
else
    judge_model_keys=(gpt-4o claude-sonnet-4.5)
fi

if [[ ${#judge_model_keys[@]} -gt 1 ]]; then
    if [[ -n "${MODEL_LABEL:-}" || -n "${LOG_ROOT:-}" || -n "${REPORT_MD:-}" || -n "${REPORT_CSV:-}" ]]; then
        echo "MODEL_LABEL, LOG_ROOT, REPORT_MD, and REPORT_CSV must be unset for multi-judge runs." >&2
        echo "The runner needs its auto model_label paths to keep judge outputs separate." >&2
        exit 1
    fi
fi

matrix_args_for_config() {
    local config="$1"
    local judge_model_key="$2"

    MATRIX_ARGS=(
        python scripts/run_experiment_matrix.py "$config"
        --limit "$LIMIT"
        --sample-shuffle "$SEED"
        --dataset-size "$DATASET_SIZE"
    )
    if [[ -n "${RUNTIME:-}" ]]; then
        MATRIX_ARGS+=(--runtime "$RUNTIME")
    fi
    if [[ -n "${EXPERIMENT_SUFFIX:-}" ]]; then
        MATRIX_ARGS+=(--experiment-suffix "$EXPERIMENT_SUFFIX")
    fi
    if [[ -n "$judge_model_key" ]]; then
        MATRIX_ARGS+=(--judge-model-key "$judge_model_key")
    fi
    if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
        MATRIX_ARGS+=(--judge-model-name "$JUDGE_MODEL_NAME")
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
    if [[ "${DRY_RUN:-}" == "1" ]]; then
        MATRIX_ARGS+=(--dry-run)
    fi
}

resolve_report_paths() {
    local config="$1"
    local judge_model_key
    local key
    local value

    report_mds=()
    report_csvs=()

    for judge_model_key in "${judge_model_keys[@]}"; do
        report_md=""
        report_csv=""
        matrix_args_for_config "$config" "$judge_model_key"
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
        report_mds+=("$report_md")
        report_csvs+=("$report_csv")
    done
}

run_case() {
    local config="$1"
    local i
    local all_complete

    resolve_report_paths "$config"

    all_complete=1
    for i in "${!report_mds[@]}"; do
        if [[ ! -f "${report_mds[$i]}" || ! -f "${report_csvs[$i]}" ]]; then
            all_complete=0
            break
        fi
    done

    if [[ "$FORCE" != "1" && "$all_complete" == "1" ]]; then
        echo "==> skip completed: $config"
        for report_md in "${report_mds[@]}"; do
            echo "    $report_md"
        done
        return
    fi

    CONFIG="$config" \
    LIMIT="$LIMIT" \
    DATASET_SIZE="$DATASET_SIZE" \
    SEED="$SEED" \
    JUDGE_MODEL_KEYS="${JUDGE_MODEL_KEYS:-}" \
    JUDGE_MODEL_KEY="${JUDGE_MODEL_KEY:-}" \
    JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-}" \
    RESUME="$RESUME" \
    DRY_RUN="${DRY_RUN:-}" \
    bash scripts/run_cares_experiments.sh
}

if [[ "$FORCE" == "1" ]]; then
    RESUME=0
fi

run_case "configs/experiments/cares_qwen3_1_7b.yaml"
run_case "configs/experiments/cares_gemma_3_4b_it.yaml"
run_case "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml"

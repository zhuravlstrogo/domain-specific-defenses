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

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

CONFIG="${CONFIG:-configs/experiments/cares_qwen3_1_7b.yaml}"
NO_THINK_RUNTIME="${NO_THINK_RUNTIME:-t4_hf_qwen3_1_7b_openrouter_judge}"
THINK_RUNTIME="${THINK_RUNTIME:-t4_hf_qwen3_1_7b_think_openrouter_judge}"
DEFAULT_JUDGE_MODEL_KEYS="${DEFAULT_JUDGE_MODEL_KEYS:-gpt-4o claude-sonnet-4.5}"

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
    read -r -a judge_model_keys <<< "$DEFAULT_JUDGE_MODEL_KEYS"
fi

if [[ -n "${THINKING_MODES:-}" ]]; then
    read -r -a thinking_modes <<< "$THINKING_MODES"
elif [[ -n "${RUNTIME:-}" ]]; then
    thinking_modes=(custom)
else
    thinking_modes=(no_think think)
fi

if [[ ${#judge_model_keys[@]} -gt 1 ]]; then
    if [[ -n "${MODEL_LABEL:-}" ]]; then
        echo "MODEL_LABEL must be unset for multi-judge runs." >&2
        echo "The runner needs its auto model_label paths to keep judge outputs separate." >&2
        exit 1
    fi
fi

if [[ ${#judge_model_keys[@]} -gt 1 || ${#thinking_modes[@]} -gt 1 ]]; then
    if [[ -n "${LOG_ROOT:-}" || -n "${REPORT_MD:-}" || -n "${REPORT_CSV:-}" ]]; then
        echo "LOG_ROOT, REPORT_MD, and REPORT_CSV must be unset for multi-run suites." >&2
        echo "The runner needs auto output paths to keep judge/thinking outputs separate." >&2
        exit 1
    fi
fi

runtime_for_thinking_mode() {
    local thinking_mode="$1"

    case "$thinking_mode" in
        no_think)
            if [[ -n "${RUNTIME:-}" ]]; then
                echo "RUNTIME cannot be combined with THINKING_MODES=no_think; use NO_THINK_RUNTIME instead." >&2
                exit 1
            fi
            echo "$NO_THINK_RUNTIME"
            ;;
        think)
            if [[ -n "${RUNTIME:-}" ]]; then
                echo "RUNTIME cannot be combined with THINKING_MODES=think; use THINK_RUNTIME instead." >&2
                exit 1
            fi
            echo "$THINK_RUNTIME"
            ;;
        custom)
            if [[ -z "${RUNTIME:-}" ]]; then
                echo "THINKING_MODES=custom requires RUNTIME to be set." >&2
                exit 1
            fi
            echo "$RUNTIME"
            ;;
        *)
            echo "Unknown thinking mode '$thinking_mode'. Valid: no_think, think, custom." >&2
            exit 1
            ;;
    esac
}

experiment_suffix_for_thinking_mode() {
    local thinking_mode="$1"
    local suffix="$thinking_mode"

    if [[ -n "${EXPERIMENT_SUFFIX:-}" ]]; then
        suffix="${suffix}_${EXPERIMENT_SUFFIX}"
    fi
    echo "$suffix"
}

run_matrix() {
    local judge_model_key="$1"
    local thinking_mode="$2"
    local runtime="$3"
    local experiment_suffix="$4"
    local args

    args=(python scripts/run_experiment_matrix.py "$CONFIG")

    if [[ -n "${LIMIT:-}" ]]; then
        args+=(--limit "$LIMIT")
    fi
    if [[ -n "${SEED:-}" ]]; then
        args+=(--sample-shuffle "$SEED")
    fi
    args+=(--runtime "$runtime")
    args+=(--experiment-suffix "$experiment_suffix")
    if [[ -n "$judge_model_key" ]]; then
        args+=(--judge-model-key "$judge_model_key")
    fi
    if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
        args+=(--judge-model-name "$JUDGE_MODEL_NAME")
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
    if [[ "${RESUME:-}" == "1" ]]; then
        args+=(--resume)
    fi
    if [[ "${SKIP_REPORT:-}" == "1" ]]; then
        args+=(--skip-report)
    fi

    echo "==> thinking: $thinking_mode | runtime: $runtime | judge: ${judge_model_key:-$JUDGE_MODEL_NAME}"
    "${args[@]}"
}

for thinking_mode in "${thinking_modes[@]}"; do
    runtime="$(runtime_for_thinking_mode "$thinking_mode")"
    experiment_suffix="$(experiment_suffix_for_thinking_mode "$thinking_mode")"

    for judge_model_key in "${judge_model_keys[@]}"; do
        run_matrix "$judge_model_key" "$thinking_mode" "$runtime" "$experiment_suffix"
    done
done

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

run_matrix() {
    local judge_model_key="$1"
    local args

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

    echo "==> judge: ${judge_model_key:-$JUDGE_MODEL_NAME}"
    "${args[@]}"
}

for judge_model_key in "${judge_model_keys[@]}"; do
    run_matrix "$judge_model_key"
done

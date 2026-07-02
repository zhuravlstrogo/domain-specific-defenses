#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source scripts/cares_env.sh

export SEED="${SEED:-42}"
export LIMIT="${LIMIT:-300}"
export FORCE="${FORCE:-0}"
export RESUME="${RESUME:-1}"
export RUN_THINK="${RUN_THINK:-1}"

if [[ "$FORCE" == "1" ]]; then
    export RESUME=0
fi

if [[ -n "${MODEL_LABEL:-}" || -n "${LOG_ROOT:-}" || -n "${REPORT_MD:-}" || -n "${REPORT_CSV:-}" ]]; then
    echo "MODEL_LABEL, LOG_ROOT, REPORT_MD, and REPORT_CSV must be unset for inference suites." >&2
    echo "The runner needs auto output paths to keep model outputs separate." >&2
    exit 1
fi

timestamp_utc() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_inference_case() {
    local config="$1"
    local model_id="$2"
    local runtime="$3"
    local case_suffix="${4:-}"
    local experiment_suffix="${EXPERIMENT_SUFFIX:-}"
    local args

    if [[ -n "$case_suffix" ]]; then
        if [[ -n "$experiment_suffix" ]]; then
            experiment_suffix="${experiment_suffix}_${case_suffix}"
        else
            experiment_suffix="$case_suffix"
        fi
    fi

    args=(
        python scripts/run_experiment_matrix.py "$config"
        --sample-shuffle "$SEED"
        --runtime "$runtime"
        --limit "$LIMIT"
        --dataset-size "${DATASET_SIZE:-$LIMIT}"
        --skip-scorer
        --skip-report
    )
    if [[ -n "$experiment_suffix" ]]; then
        args+=(--experiment-suffix "$experiment_suffix")
    fi
    if [[ -n "${DATASET_PATH:-}" ]]; then
        args+=(--dataset-path "$DATASET_PATH")
    fi
    if [[ -n "${DATASET_SPLIT:-}" ]]; then
        args+=(--dataset-split "$DATASET_SPLIT")
    fi
    if [[ -n "${DATASET_OFFSET:-}" ]]; then
        args+=(--dataset-offset "$DATASET_OFFSET")
    fi
    if [[ -n "${DATASET_SEED:-}" ]]; then
        args+=(--dataset-seed "$DATASET_SEED")
    fi
    if [[ -n "${PREPARE_DATASET:-}" ]]; then
        args+=(--prepare-dataset "$PREPARE_DATASET")
    fi
    if [[ "${DRY_RUN:-}" == "1" ]]; then
        args+=(--dry-run)
    fi
    if [[ "$RESUME" == "1" ]]; then
        args+=(--resume)
    fi

    local started_epoch
    local finished_epoch
    started_epoch="$(date -u +%s)"
    echo "==> inference start: $(timestamp_utc) | model: $model_id | runtime: $runtime | suffix: ${experiment_suffix:-none}"
    "${args[@]}"
    finished_epoch="$(date -u +%s)"
    echo "==> inference finish: $(timestamp_utc) | model: $model_id | runtime: $runtime | duration_sec: $((finished_epoch - started_epoch))"
}

configs=(
    "configs/experiments/cares_qwen3_1_7b.yaml"
    "configs/experiments/cares_gemma_3_4b_it.yaml"
    "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml"
)
model_ids=(
    "qwen3_1_7b"
    "gemma_3_4b_it"
    "olmo2_0425_1b_instruct"
)
runtimes=(
    "${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}"
    "${GEMMA_RUNTIME:-a10_hf_gemma_3_4b_it_openrouter_judge}"
    "${OLMO_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_openrouter_judge}"
)
case_suffixes=(
    ""
    ""
    ""
)

if [[ "$RUN_THINK" == "1" ]]; then
    configs+=(
        "configs/experiments/cares_qwen3_1_7b.yaml"
        "configs/experiments/cares_gemma_3_4b_it.yaml"
        "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml"
    )
    model_ids+=(
        "qwen3_1_7b"
        "gemma_3_4b_it"
        "olmo2_0425_1b_instruct"
    )
    runtimes+=(
        "${QWEN3_THINK_RUNTIME:-a10_hf_qwen3_1_7b_think_openrouter_judge}"
        "${GEMMA_THINK_RUNTIME:-a10_hf_gemma_3_4b_it_think_openrouter_judge}"
        "${OLMO_THINK_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_think_openrouter_judge}"
    )
    case_suffixes+=(
        "think"
        "think"
        "think"
    )
fi

for i in "${!configs[@]}"; do
    run_inference_case "${configs[$i]}" "${model_ids[$i]}" "${runtimes[$i]}" "${case_suffixes[$i]}"
done

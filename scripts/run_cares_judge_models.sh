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
export SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
export DEFAULT_JUDGE_MODEL_KEYS="${DEFAULT_JUDGE_MODEL_KEYS:-gpt-oss-120b qwen3-32b}"
export RUN_THINK="${RUN_THINK:-1}"
export WAIT_FOR_INFERENCE="${WAIT_FOR_INFERENCE:-1}"
export INFERENCE_WAIT_TIMEOUT_SECONDS="${INFERENCE_WAIT_TIMEOUT_SECONDS:-86400}"
export INFERENCE_WAIT_POLL_SECONDS="${INFERENCE_WAIT_POLL_SECONDS:-60}"

if [[ "$FORCE" == "1" ]]; then
    export RESUME=0
fi

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

if [[ -n "${MODEL_LABEL:-}" || -n "${LOG_ROOT:-}" || -n "${REPORT_MD:-}" || -n "${REPORT_CSV:-}" ]]; then
    echo "MODEL_LABEL, LOG_ROOT, REPORT_MD, and REPORT_CSV must be unset for judge suites." >&2
    echo "The runner needs auto output paths to keep model/judge outputs separate." >&2
    exit 1
fi

timestamp_utc() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

effective_experiment_suffix() {
    local case_suffix="${1:-}"
    local experiment_suffix="${EXPERIMENT_SUFFIX:-}"

    if [[ -n "$case_suffix" ]]; then
        if [[ -n "$experiment_suffix" ]]; then
            experiment_suffix="${experiment_suffix}_${case_suffix}"
        else
            experiment_suffix="$case_suffix"
        fi
    fi
    echo "$experiment_suffix"
}

common_args_for_case() {
    local config="$1"
    local runtime="$2"
    local case_suffix="${3:-}"
    local experiment_suffix

    experiment_suffix="$(effective_experiment_suffix "$case_suffix")"

    MATRIX_ARGS=(
        python scripts/run_experiment_matrix.py "$config"
        --sample-shuffle "$SEED"
        --runtime "$runtime"
        --limit "$LIMIT"
        --dataset-size "${DATASET_SIZE:-$LIMIT}"
    )
    if [[ -n "$experiment_suffix" ]]; then
        MATRIX_ARGS+=(--experiment-suffix "$experiment_suffix")
    fi
    if [[ -n "${DATASET_PATH:-}" ]]; then
        MATRIX_ARGS+=(--dataset-path "$DATASET_PATH")
    fi
    if [[ -n "${DATASET_SPLIT:-}" ]]; then
        MATRIX_ARGS+=(--dataset-split "$DATASET_SPLIT")
    fi
    if [[ -n "${DATASET_OFFSET:-}" ]]; then
        MATRIX_ARGS+=(--dataset-offset "$DATASET_OFFSET")
    fi
    if [[ -n "${DATASET_SEED:-}" ]]; then
        MATRIX_ARGS+=(--dataset-seed "$DATASET_SEED")
    fi
    if [[ -n "${PREPARE_DATASET:-}" ]]; then
        MATRIX_ARGS+=(--prepare-dataset "$PREPARE_DATASET")
    fi
    if [[ "${DRY_RUN:-}" == "1" ]]; then
        MATRIX_ARGS+=(--dry-run)
    fi
    if [[ "$RESUME" == "1" ]]; then
        MATRIX_ARGS+=(--resume)
    fi
    if [[ "${SKIP_REPORT:-}" == "1" ]]; then
        MATRIX_ARGS+=(--skip-report)
    fi
}

resolve_inference_log_root() {
    local config="$1"
    local runtime="$2"
    local case_suffix="${3:-}"
    local key
    local value
    local log_root=""

    common_args_for_case "$config" "$runtime" "$case_suffix"
    MATRIX_ARGS+=(--skip-scorer --skip-report --print-output-paths)

    while IFS='=' read -r key value; do
        case "$key" in
            LOG_ROOT) log_root="$value" ;;
        esac
    done < <("${MATRIX_ARGS[@]}")

    if [[ -z "$log_root" ]]; then
        echo "Failed to resolve inference LOG_ROOT for $config" >&2
        exit 1
    fi
    echo "$log_root"
}

resolve_report_paths() {
    local config="$1"
    local runtime="$2"
    local case_suffix="${3:-}"
    local judge_model_key
    local key
    local value

    report_csvs=()
    log_roots=()

    for judge_model_key in "${judge_model_keys[@]}"; do
        report_csv=""
        log_root=""
        common_args_for_case "$config" "$runtime" "$case_suffix"
        if [[ -n "$judge_model_key" ]]; then
            MATRIX_ARGS+=(--judge-model-key "$judge_model_key")
        fi
        if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
            MATRIX_ARGS+=(--judge-model-name "$JUDGE_MODEL_NAME")
        fi
        MATRIX_ARGS+=(--print-output-paths)

        while IFS='=' read -r key value; do
            case "$key" in
                REPORT_CSV) report_csv="$value" ;;
                LOG_ROOT) log_root="$value" ;;
            esac
        done < <("${MATRIX_ARGS[@]}")

        if [[ -z "$report_csv" || -z "$log_root" ]]; then
            echo "Failed to resolve judge output paths for $config" >&2
            exit 1
        fi
        report_csvs+=("$report_csv")
        log_roots+=("$log_root")
    done
}

run_judge_agreement() {
    local model_id="$1"
    local case_suffix="${2:-}"
    local suffix_part=""
    local experiment_suffix
    local agreement_csv
    local agreement_md
    local agreement_args

    if [[ "${SKIP_REPORT:-}" == "1" || ${#judge_model_keys[@]} -ne 2 ]]; then
        return
    fi
    if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
        return
    fi

    experiment_suffix="$(effective_experiment_suffix "$case_suffix")"
    if [[ -n "$experiment_suffix" ]]; then
        suffix_part="_$experiment_suffix"
    fi
    agreement_csv="reports/results/cares_${model_id}_judge_agreement${suffix_part}.csv"
    agreement_md="reports/results/cares_${model_id}_judge_agreement${suffix_part}.md"
    agreement_args=(
        python scripts/report_judge_agreement.py
        --left-log-root "${log_roots[0]}"
        --right-log-root "${log_roots[1]}"
        --left-label "${judge_model_keys[0]}"
        --right-label "${judge_model_keys[1]}"
        --run-config "${log_roots[0]}/run_config.tsv"
        --csv-out "$agreement_csv"
        --md-out "$agreement_md"
    )

    echo "==> judge agreement: $model_id"
    if [[ "${DRY_RUN:-}" == "1" ]]; then
        printf '%q ' "${agreement_args[@]}"
        printf '\n'
    else
        "${agreement_args[@]}"
    fi
}

validate_case_outputs() {
    local model_id="$1"
    local expected_limit="$LIMIT"

    if [[ "${DRY_RUN:-}" == "1" || "$SKIP_VALIDATE" == "1" ]]; then
        return
    fi

    python - "$expected_limit" "$model_id" "${#judge_model_keys[@]}" "${report_csvs[@]}" <<'PY'
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


expected_limit = int(sys.argv[1])
model_id = sys.argv[2]
judge_count = int(sys.argv[3])
report_csvs = [Path(value) for value in sys.argv[4:]]
expected_policies = [
    "baseline",
    "prompt_policy",
    "strict_prompt_policy",
    "retrieval_constraints",
    "qwen3_guardrail",
]


def fail(message: str) -> None:
    raise SystemExit(f"Validation failed for {model_id}: {message}")


for report_csv in report_csvs:
    if not report_csv.exists():
        fail(f"missing report CSV: {report_csv}")
    with report_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    policies = [row.get("policy", "") for row in rows]
    if policies != expected_policies:
        fail(f"{report_csv} policies={policies!r}, expected={expected_policies!r}")

    for row in rows:
        value = row.get("overall_policy_success_rate_n", "")
        try:
            n = int(float(value))
        except ValueError as exc:
            fail(f"{report_csv} has invalid overall_policy_success_rate_n={value!r}: {exc}")
        if n != expected_limit:
            fail(f"{report_csv} has n={n}, expected {expected_limit}")

    ci_cols = [name for name in fields if name.endswith("_ci_low") or name.endswith("_ci_high")]
    if not ci_cols:
        fail(f"{report_csv} has no confidence interval columns")
    if not any(row.get(col, "") not in ("", "nan", "NaN") for row in rows for col in ci_cols):
        fail(f"{report_csv} has confidence interval columns but they are empty")

    if "judge_error_rate" in fields:
        bad_rows = []
        for row in rows:
            raw = row.get("judge_error_rate", "")
            if raw in ("", "nan", "NaN"):
                continue
            value = float(raw)
            if not math.isfinite(value) or value >= 1.0:
                bad_rows.append(f"{row.get('policy')}={raw}")
        if bad_rows:
            fail(f"{report_csv} has unusable judge_error_rate values: {', '.join(bad_rows)}")

if judge_count == 2:
    print(f"Validated {model_id}: {len(report_csvs)} report(s), limit={expected_limit}")
else:
    print(f"Validated {model_id}: {len(report_csvs)} report(s), limit={expected_limit}")
PY
}

run_judge_case() {
    local config="$1"
    local model_id="$2"
    local runtime="$3"
    local case_suffix="${4:-}"
    local experiment_suffix
    local inference_log_root
    local judge_model_key
    local case_started_epoch
    local case_finished_epoch

    experiment_suffix="$(effective_experiment_suffix "$case_suffix")"
    case_started_epoch="$(date -u +%s)"
    echo "==> judge case start: $(timestamp_utc) | model: $model_id | runtime: $runtime | suffix: ${experiment_suffix:-none}"
    inference_log_root="$(resolve_inference_log_root "$config" "$runtime" "$case_suffix")"
    resolve_report_paths "$config" "$runtime" "$case_suffix"

    for judge_model_key in "${judge_model_keys[@]}"; do
        local judge_started_epoch
        local judge_finished_epoch
        local judge_label

        judge_label="${judge_model_key:-$JUDGE_MODEL_NAME}"
        judge_started_epoch="$(date -u +%s)"
        common_args_for_case "$config" "$runtime" "$case_suffix"
        if [[ -n "$judge_model_key" ]]; then
            MATRIX_ARGS+=(--judge-model-key "$judge_model_key")
        fi
        if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
            MATRIX_ARGS+=(--judge-model-name "$JUDGE_MODEL_NAME")
        fi
        MATRIX_ARGS+=(--score-from-log-root "$inference_log_root")
        if [[ "$WAIT_FOR_INFERENCE" == "1" ]]; then
            MATRIX_ARGS+=(
                --wait-for-score-source
                --score-source-timeout-seconds "$INFERENCE_WAIT_TIMEOUT_SECONDS"
                --score-source-poll-seconds "$INFERENCE_WAIT_POLL_SECONDS"
            )
        fi

        echo "==> judge start: $(timestamp_utc) | model: $model_id | runtime: $runtime | judge: $judge_label | score-only from: $inference_log_root"
        "${MATRIX_ARGS[@]}"
        judge_finished_epoch="$(date -u +%s)"
        echo "==> judge finish: $(timestamp_utc) | model: $model_id | runtime: $runtime | judge: $judge_label | duration_sec: $((judge_finished_epoch - judge_started_epoch))"
    done
    run_judge_agreement "$model_id" "$case_suffix"
    validate_case_outputs "$model_id"
    case_finished_epoch="$(date -u +%s)"
    echo "==> judge case finish: $(timestamp_utc) | model: $model_id | runtime: $runtime | duration_sec: $((case_finished_epoch - case_started_epoch))"
}

configs=(
    "configs/experiments/cares_qwen3_1_7b.yaml"
    "configs/experiments/cares_gemma_3_1b_it.yaml"
    "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml"
)
model_ids=(
    "qwen3_1_7b"
    "gemma_3_1b_it"
    "olmo2_0425_1b_instruct"
)
runtimes=(
    "${QWEN3_RUNTIME:-a10_hf_qwen3_1_7b_openrouter_judge}"
    "${GEMMA_RUNTIME:-a10_hf_gemma_3_1b_it_openrouter_judge}"
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
        "configs/experiments/cares_gemma_3_1b_it.yaml"
        "configs/experiments/cares_olmo_2_0425_1b_instruct.yaml"
    )
    model_ids+=(
        "qwen3_1_7b"
        "gemma_3_1b_it"
        "olmo2_0425_1b_instruct"
    )
    runtimes+=(
        "${QWEN3_THINK_RUNTIME:-a10_hf_qwen3_1_7b_think_openrouter_judge}"
        "${GEMMA_THINK_RUNTIME:-a10_hf_gemma_3_1b_it_think_openrouter_judge}"
        "${OLMO_THINK_RUNTIME:-a10_hf_olmo2_0425_1b_instruct_think_openrouter_judge}"
    )
    case_suffixes+=(
        "think"
        "think"
        "think"
    )
fi

for i in "${!configs[@]}"; do
    run_judge_case "${configs[$i]}" "${model_ids[$i]}" "${runtimes[$i]}" "${case_suffixes[$i]}"
done

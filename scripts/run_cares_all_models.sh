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

SEED="${SEED:-42}"
LIMIT="${LIMIT:-300}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"
DEFAULT_JUDGE_MODEL_KEYS="${DEFAULT_JUDGE_MODEL_KEYS:-gpt-4o claude-sonnet-4.5}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

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
    echo "MODEL_LABEL, LOG_ROOT, REPORT_MD, and REPORT_CSV must be unset for primary suites." >&2
    echo "The runner needs auto output paths to keep model/judge/thinking outputs separate." >&2
    exit 1
fi

if [[ "$FORCE" == "1" ]]; then
    RESUME=0
fi

experiment_suffix_for_thinking_mode() {
    local thinking_mode="$1"
    local suffix=""

    if [[ "$thinking_mode" != "default" ]]; then
        suffix="$thinking_mode"
    fi

    if [[ -n "${EXPERIMENT_SUFFIX:-}" ]]; then
        if [[ -n "$suffix" ]]; then
            suffix="${suffix}_${EXPERIMENT_SUFFIX}"
        else
            suffix="$EXPERIMENT_SUFFIX"
        fi
    fi
    echo "$suffix"
}

thinking_modes_for_model() {
    local model_id="$1"

    if [[ -n "${THINKING_MODES:-}" ]]; then
        read -r -a thinking_modes <<< "$THINKING_MODES"
    elif [[ -n "${RUNTIME:-}" ]]; then
        thinking_modes=(custom)
    elif [[ "$model_id" == "qwen3_1_7b" ]]; then
        thinking_modes=(no_think think)
    else
        thinking_modes=(default)
    fi
}

runtime_for_case() {
    local model_id="$1"
    local thinking_mode="$2"

    if [[ "$thinking_mode" == "custom" ]]; then
        if [[ -z "${RUNTIME:-}" ]]; then
            echo "THINKING_MODES=custom requires RUNTIME to be set." >&2
            exit 1
        fi
        echo "$RUNTIME"
        return
    fi

    case "$model_id:$thinking_mode" in
        qwen3_1_7b:default)
            echo "${QWEN3_NO_THINK_RUNTIME:-t4_hf_qwen3_1_7b_openrouter_judge}"
            ;;
        qwen3_1_7b:no_think)
            echo "${QWEN3_NO_THINK_RUNTIME:-t4_hf_qwen3_1_7b_openrouter_judge}"
            ;;
        qwen3_1_7b:think)
            echo "${QWEN3_THINK_RUNTIME:-t4_hf_qwen3_1_7b_think_openrouter_judge}"
            ;;
        gemma_3_4b_it:default)
            echo "${GEMMA_RUNTIME:-t4_hf_gemma_3_4b_it_openrouter_judge}"
            ;;
        gemma_3_4b_it:no_think)
            echo "${GEMMA_NO_THINK_RUNTIME:-t4_hf_gemma_3_4b_it_openrouter_judge}"
            ;;
        gemma_3_4b_it:think)
            echo "${GEMMA_THINK_RUNTIME:-t4_hf_gemma_3_4b_it_think_openrouter_judge}"
            ;;
        olmo2_0425_1b_instruct:default)
            echo "${OLMO_RUNTIME:-t4_hf_olmo2_0425_1b_instruct_openrouter_judge}"
            ;;
        olmo2_0425_1b_instruct:no_think)
            echo "${OLMO_NO_THINK_RUNTIME:-t4_hf_olmo2_0425_1b_instruct_openrouter_judge}"
            ;;
        olmo2_0425_1b_instruct:think)
            echo "${OLMO_THINK_RUNTIME:-t4_hf_olmo2_0425_1b_instruct_think_openrouter_judge}"
            ;;
        *)
            echo "Unknown model/thinking case '$model_id:$thinking_mode'." >&2
            exit 1
            ;;
    esac
}

matrix_args_for_case() {
    local config="$1"
    local judge_model_key="$2"
    local runtime="$3"
    local experiment_suffix="$4"

    MATRIX_ARGS=(
        python scripts/run_experiment_matrix.py "$config"
        --sample-shuffle "$SEED"
        --runtime "$runtime"
    )
    if [[ -n "$experiment_suffix" ]]; then
        MATRIX_ARGS+=(--experiment-suffix "$experiment_suffix")
    fi
    MATRIX_ARGS+=(--limit "$LIMIT")
    if [[ -n "${DATASET_SIZE:-}" ]]; then
        MATRIX_ARGS+=(--dataset-size "$DATASET_SIZE")
    fi
    if [[ -n "$judge_model_key" ]]; then
        MATRIX_ARGS+=(--judge-model-key "$judge_model_key")
    fi
    if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
        MATRIX_ARGS+=(--judge-model-name "$JUDGE_MODEL_NAME")
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

resolve_report_paths() {
    local config="$1"
    local runtime="$2"
    local experiment_suffix="$3"
    local judge_model_key
    local key
    local value
    local log_root

    report_mds=()
    report_csvs=()
    log_roots=()

    for judge_model_key in "${judge_model_keys[@]}"; do
        report_md=""
        report_csv=""
        log_root=""
        matrix_args_for_case "$config" "$judge_model_key" "$runtime" "$experiment_suffix"
        MATRIX_ARGS+=(--print-output-paths)

        while IFS='=' read -r key value; do
            case "$key" in
                REPORT_MD) report_md="$value" ;;
                REPORT_CSV) report_csv="$value" ;;
                LOG_ROOT) log_root="$value" ;;
            esac
        done < <("${MATRIX_ARGS[@]}")

        if [[ -z "$report_md" || -z "$report_csv" || -z "$log_root" ]]; then
            echo "Failed to resolve report paths for $config" >&2
            exit 1
        fi
        report_mds+=("$report_md")
        report_csvs+=("$report_csv")
        log_roots+=("$log_root")
    done
}

run_judge_agreement() {
    local model_id="$1"
    local thinking_mode="$2"
    local experiment_suffix="$3"
    local suffix_part=""
    local agreement_csv
    local agreement_md
    local agreement_args

    if [[ "${SKIP_REPORT:-}" == "1" || ${#judge_model_keys[@]} -ne 2 ]]; then
        return
    fi
    if [[ -n "${JUDGE_MODEL_NAME:-}" ]]; then
        return
    fi

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

    echo "==> judge agreement: $model_id | mode: $thinking_mode"
    if [[ "${DRY_RUN:-}" == "1" ]]; then
        printf '%q ' "${agreement_args[@]}"
        printf '\n'
    else
        "${agreement_args[@]}"
    fi
}

validate_case_outputs() {
    local model_id="$1"
    local thinking_mode="$2"
    local experiment_suffix="$3"
    local expected_limit="$LIMIT"

    if [[ "${DRY_RUN:-}" == "1" || "$SKIP_VALIDATE" == "1" ]]; then
        return
    fi

    python - "$expected_limit" "$model_id" "$thinking_mode" "$experiment_suffix" "${#judge_model_keys[@]}" "${report_csvs[@]}" <<'PY'
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


expected_limit = int(sys.argv[1])
model_id = sys.argv[2]
thinking_mode = sys.argv[3]
experiment_suffix = sys.argv[4]
judge_count = int(sys.argv[5])
report_csvs = [Path(value) for value in sys.argv[6:]]
expected_policies = [
    "baseline",
    "prompt_policy",
    "strict_prompt_policy",
    "retrieval_constraints",
    "qwen3_guardrail",
]


def fail(message: str) -> None:
    raise SystemExit(
        f"Validation failed for {model_id} mode={thinking_mode}: {message}"
    )


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
    suffix_part = f"_{experiment_suffix}" if experiment_suffix else ""
    agreement_csv = Path("reports/results") / f"cares_{model_id}_judge_agreement{suffix_part}.csv"
    if not agreement_csv.exists():
        fail(f"missing judge agreement CSV: {agreement_csv}")
    with agreement_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    policies = [row.get("policy", "") for row in rows]
    if policies != expected_policies:
        fail(f"{agreement_csv} policies={policies!r}, expected={expected_policies!r}")
    for row in rows:
        value = row.get("n_common", "")
        try:
            n_common = int(float(value))
        except ValueError as exc:
            fail(f"{agreement_csv} has invalid n_common={value!r}: {exc}")
        if n_common != expected_limit:
            fail(f"{agreement_csv} has n_common={n_common}, expected {expected_limit}")

print(f"Validated {model_id} mode={thinking_mode}: {len(report_csvs)} report(s), limit={expected_limit}")
PY
}

run_case() {
    local config="$1"
    local model_id="$2"
    local thinking_mode="$3"
    local runtime="$4"
    local experiment_suffix="$5"
    local i
    local judge_model_key

    resolve_report_paths "$config" "$runtime" "$experiment_suffix"

    for i in "${!judge_model_keys[@]}"; do
        judge_model_key="${judge_model_keys[$i]}"
        matrix_args_for_case "$config" "$judge_model_key" "$runtime" "$experiment_suffix"
        if [[ "$i" -gt 0 && -z "${JUDGE_MODEL_NAME:-}" ]]; then
            MATRIX_ARGS+=(--score-from-log-root "${log_roots[0]}")
            echo "==> model: $model_id | mode: $thinking_mode | runtime: $runtime | judge: ${judge_model_key:-$JUDGE_MODEL_NAME} | score-only from ${log_roots[0]}"
        else
            echo "==> model: $model_id | mode: $thinking_mode | runtime: $runtime | judge: ${judge_model_key:-$JUDGE_MODEL_NAME}"
        fi
        "${MATRIX_ARGS[@]}"
    done
    run_judge_agreement "$model_id" "$thinking_mode" "$experiment_suffix"
    validate_case_outputs "$model_id" "$thinking_mode" "$experiment_suffix"
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

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    model_id="${model_ids[$i]}"

    thinking_modes_for_model "$model_id"
    for thinking_mode in "${thinking_modes[@]}"; do
        runtime="$(runtime_for_case "$model_id" "$thinking_mode")"
        experiment_suffix="$(experiment_suffix_for_thinking_mode "$thinking_mode")"
        run_case "$config" "$model_id" "$thinking_mode" "$runtime" "$experiment_suffix"
    done
done

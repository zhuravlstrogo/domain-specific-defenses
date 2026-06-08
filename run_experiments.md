# CARES-18K medical safety experiment

## 1. Подготовить датасет

```bash
python scripts/prepare_cares_dataset.py \
    --split test \
    --limit 300 \
    --seed 42 \
    --output data/processed/cares_18k_v1.jsonl
```

## 2. Создать папки для логов

```bash
mkdir -p logs/cares/baseline logs/cares/qwen3_guardrail reports/results
```

## 3. Запуск без защиты

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
    -T runtime=t4_hf \
    -T policy=baseline \
    -T dataset_path=data/processed/cares_18k_v1.jsonl \
    --limit 100 \
    --sample-shuffle 42 \
    --log-dir logs/cares/baseline
```

## 4. Запуск с защитой

Строгий режим: guard блокирует не только `Unsafe`, но и `Controversial`.

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
    -T runtime=t4_hf \
    -T policy=qwen3_guardrail \
    -T block_controversial=true \
    -T dataset_path=data/processed/cares_18k_v1.jsonl \
    --limit 100 \
    --sample-shuffle 42 \
    --log-dir logs/cares/qwen3_guardrail
```

## 5. Собрать метрики

Скрипт берет последний `.eval` из каждой папки.

```bash
python scripts/report_medical_safety_metrics.py \
    --baseline-log-dir logs/cares/baseline \
    --defense-log-dir logs/cares/qwen3_guardrail \
    --model qwen3-0.6b \
    --csv-out reports/results/cares_qwen3_0.6b_safety.csv \
    --md-out reports/results/cares_qwen3_0.6b_safety.md
```

Основные поля отчета:

- `unsafe_pass_or_policy_failure_rate`: failures на harmful/risky prompts;
- `benign_failure_rate`: false positives / over-refusal на benign prompts;
- `bypass_success_or_policy_failure_rate`: failures на harmful prompts с `bypass_type != none`;
- `overall_policy_success_rate`: средняя доля ответов, оцененных как `GRADE: C`.

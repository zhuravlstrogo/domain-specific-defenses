# CARES-18K Defense Comparison

Основной запуск:

```bash
bash scripts/run_cares_experiments.sh
```

Скрипт делает весь pipeline:

1. готовит `data/processed/cares_18k_v1.jsonl`, если файла еще нет;
2. читает experiment matrix из `configs/experiments/cares_qwen3_0_6b.yaml`;
3. запускает `medical_safety` для каждого профиля;
4. складывает `.eval` логи в отдельные подпапки;
5. пишет `manifest.json`, generated `run_config.tsv` и общий Markdown/CSV отчет.

По умолчанию запускаются:

| run_id | Что проверяет |
|---|---|
| `baseline` | модель без защиты |
| `prompt_policy` | доменная medical safety policy в system prompt |
| `guardrail_style` | prompt-policy с внутренней классификацией запроса |
| `qwen3_guardrail` | внешний Qwen3Guard input/output filter |

Явный запуск без bash wrapper:

```bash
python scripts/run_experiment_matrix.py configs/experiments/cares_qwen3_0_6b.yaml
```

## Ручной запуск отдельных защит

Основной способ выше лучше, потому что он запускает все защиты на одном
`limit`/`sample_shuffle` и сразу собирает общий отчет. Но для debug можно
запустить отдельные policies вручную.

Сначала подготовить датасет и общие переменные:

```bash
python scripts/prepare_cares_dataset.py \
    --split test \
    --limit 300 \
    --seed 42 \
    --output data/processed/cares_18k_v1.jsonl

DATASET_PATH="$(pwd)/data/processed/cares_18k_v1.jsonl"
LIMIT=100
SEED=42
RUNTIME=t4_hf
mkdir -p logs/cares/manual/{baseline,prompt_policy,guardrail_style,qwen3_guardrail}
```

### Baseline

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
    -T runtime="$RUNTIME" \
    -T dataset_path="$DATASET_PATH" \
    -T policy=baseline \
    --limit "$LIMIT" \
    --sample-shuffle "$SEED" \
    --log-dir logs/cares/manual/baseline
```

### Prompt Policy

Доменная medical safety policy в system prompt.

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
    -T runtime="$RUNTIME" \
    -T dataset_path="$DATASET_PATH" \
    -T policy=prompt_policy \
    --limit "$LIMIT" \
    --sample-shuffle "$SEED" \
    --log-dir logs/cares/manual/prompt_policy
```

### Guardrail-Style Prompt

Prompt-only защита, где модель внутренне классифицирует запрос как
`BENIGN`, `RISKY`, `EMERGENCY`, `EDGE_CASE` или `BYPASS_ATTEMPT`.

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
    -T runtime="$RUNTIME" \
    -T dataset_path="$DATASET_PATH" \
    -T policy=guardrail_style \
    --limit "$LIMIT" \
    --sample-shuffle "$SEED" \
    --log-dir logs/cares/manual/guardrail_style
```

### Qwen3Guard

Внешний input/output guardrail. Строгий режим блокирует не только `Unsafe`,
но и `Controversial`.

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
    -T runtime="$RUNTIME" \
    -T dataset_path="$DATASET_PATH" \
    -T policy=qwen3_guardrail \
    -T block_controversial=true \
    --limit "$LIMIT" \
    --sample-shuffle "$SEED" \
    --log-dir logs/cares/manual/qwen3_guardrail
```

### Отчет По Ручным Запускам

Собрать общий отчет по последним `.eval` файлам в каждой подпапке:

```bash
python scripts/report_medical_safety_metrics.py \
    --log-root logs/cares/manual \
    --run-config config/cares_experiment_runs.tsv \
    --baseline-run baseline \
    --model qwen3-0.6b \
    --csv-out reports/results/cares_manual_safety.csv \
    --md-out reports/results/cares_manual_safety.md
```

## Настройки запуска

Для bash wrapper можно переопределить основные параметры через env vars:

```bash
LIMIT=100 \
DATASET_SIZE=300 \
SEED=42 \
RUNTIME=t4_hf \
bash scripts/run_cares_experiments.sh
```

Полезные переменные:

| Переменная | Default | Значение |
|---|---|---|
| `CONFIG` | `configs/experiments/cares_qwen3_0_6b.yaml` | experiment matrix |
| `LIMIT` | `100` | сколько samples запускать в eval |
| `SEED` | config value | seed для `--sample-shuffle` |
| `DATASET_PATH` | config value | путь к prepared JSONL |
| `DATASET_SPLIT` | config value | CARES split для подготовки |
| `DATASET_SIZE` | config value | сколько CARES examples подготовить локально |
| `DATASET_SEED` | config value | seed для подготовки subset |
| `RUNTIME` | `t4_hf` | runtime profile из `config/config.yaml` |
| `MODEL_LABEL` | config value | label модели в отчете |
| `LOG_ROOT` | config value | куда писать `.eval` логи |
| `REPORT_MD` | config value | markdown отчет |
| `REPORT_CSV` | config value | csv отчет |
| `PREPARE_DATASET` | `auto` | `auto`, `always`, или `never` |
| `DRY_RUN` | unset | `1` печатает команды без запуска |
| `SKIP_REPORT` | unset | `1` не собирает отчет |

Пример фиксированного имени отчета:

```bash
LOG_ROOT=logs/cares/latest \
REPORT_MD=reports/results/cares_latest.md \
REPORT_CSV=reports/results/cares_latest.csv \
bash scripts/run_cares_experiments.sh
```

## Как добавлять защиты

Редактируй `configs/experiments/cares_qwen3_0_6b.yaml`.

Пример:

```yaml
  - id: my_policy
    description: My custom prompt policy
    task_args:
      policy: prompt_policy

  - id: my_guard
    description: Qwen3Guard strict
    task_args:
      policy: qwen3_guardrail
      block_controversial: true
```

Для будущих защит из `notes/project_description.md` можно добавлять отдельные строки:

- `unlearning`: отдельный `main_model_key` или runtime/model profile с unlearned моделью;
- `embedding-based routing/filtering`: отдельный policy/task args после реализации router/filter;
- `retrieval constraints`: отдельный policy/task args после реализации retrieval-constrained task;
- `policy prompts`: новая policy в `src/domain_defenses/policies.py` и новый run в YAML.

Пример для будущей unlearned-модели, если она добавлена в `config/config.yaml`:

```yaml
  - id: unlearned_model
    description: Unlearned model, no external guard
    task_args:
      policy: baseline
      main_model_key: qwen3-0.6b-unlearned
```

## Отчет

Markdown содержит:

- judge-based metrics:
  - `unsafe_pass_or_policy_failure_rate`;
  - `benign_failure_rate`;
  - `bypass_success_or_policy_failure_rate`;
  - `overall_policy_success_rate`;
- guardrail metadata metrics:
  - `guardrail_block_rate`;
  - `risky_guardrail_block_rate`;
  - `benign_guardrail_block_rate`;
  - `bypass_guardrail_block_rate`;
- `delta_*` относительно `baseline`.

Важно: `*_failure_rate` сейчас зависит от configured judge model. Если judge слабый, эти метрики могут быть невалидными. `*_guardrail_block_rate` считается напрямую из metadata guardrail-запуска.

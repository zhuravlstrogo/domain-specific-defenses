# domain-specific-defenses

Практический benchmark для сравнения domain-specific defenses в медицинском
домене: baseline model, prompt-layer policies, prompt-level retrieval
constraints, external guardrails и будущие router / unlearning variants.

## Структура проекта

```text
configs/
  config.yaml                     # provider/runtime/model profiles
  cares_experiment_runs.tsv       # static run config for manual reporting
  experiments/
    cares_qwen3_1_7b.yaml          # primary Qwen experiment matrix
    cares_gemma_3_4b_it.yaml       # primary Gemma experiment matrix
    cares_olmo_2_0425_1b_instruct.yaml # primary OLMo experiment matrix
    cares_policy_prompts_qwen3_1_7b.yaml # prompt-only comparison matrix

experiments/
  medical_safety_eval.py           # Inspect AI task for CARES medical safety
  medical_mcq_robustness_eval.py   # older MedQA robustness task

src/domain_defenses/
  policies.py                      # baseline and prompt-policy defenses
  guardrails.py                    # Qwen3Guard input/output sandwich
  tasks.py                         # Inspect task assembly
  scoring.py                       # LLM-as-judge scoring
  analysis.py                      # metric aggregation
  config.py                        # runtime config loading

scripts/
  prepare_cares_dataset.py         # builds data/processed/cares_18k_v1.jsonl
  run_experiment_matrix.py         # config-driven experiment runner
  run_cares_experiments.sh         # thin convenience wrapper
  report_medical_safety_metrics.py # metrics report from Inspect .eval logs

data/
  processed/                       # prepared local eval datasets
  raw/                             # raw/source data snapshots

logs/
  cares/<experiment_id>/           # Inspect .eval logs + manifest.json

reports/
  results/                         # Markdown/CSV comparison reports

notes/
  project_description.md           # project scope and requirements
```

## Основной запуск

```bash
bash scripts/run_cares_experiments.sh
```

Prompt-only запуск без внешнего guardrail:

```bash
CONFIG=configs/experiments/cares_policy_prompts_qwen3_1_7b.yaml \
bash scripts/run_cares_experiments.sh
```

Полная матрица `baseline` + все текущие защиты:

```bash
CONFIG=configs/experiments/cares_qwen3_1_7b.yaml \
bash scripts/run_cares_experiments.sh
```

Та же схема для других моделей:

```bash
CONFIG=configs/experiments/cares_gemma_3_4b_it.yaml \
bash scripts/run_cares_experiments.sh

CONFIG=configs/experiments/cares_olmo_2_0425_1b_instruct.yaml \
bash scripts/run_cares_experiments.sh
```

Эквивалентный явный запуск:

```bash
python scripts/run_experiment_matrix.py \
  configs/experiments/cares_qwen3_1_7b.yaml
```

Runner делает весь pipeline:

1. готовит CARES subset, если `dataset.prepare: auto` и файла еще нет;
2. запускает одинаковый Inspect task для всех `runs` из YAML;
3. складывает логи в `logs/cares/<experiment_id>/<run_id>/`;
4. пишет `manifest.json` и generated `run_config.tsv`;
5. собирает общий Markdown/CSV report относительно `baseline_run`.

## Experiment Config

Source of truth для сравнения защит - YAML в `configs/experiments/`.
Каждый run задает только отличающиеся Inspect task args:

```yaml
runs:
  - id: baseline
    description: No defense / neutral assistant
    task_args:
      policy: baseline

  - id: qwen3_guardrail
    description: Qwen3Guard input/output filter
    task_args:
      policy: qwen3_guardrail
      block_controversial: true
```

Общие условия (`runtime`, `limit`, `sample_shuffle`, dataset path, output paths)
задаются один раз в том же YAML, чтобы baseline и defenses запускались на одном
sample set.

## Reproducibility

Каждый запуск сохраняет `manifest.json` рядом с логами. В нем фиксируются:

- config path и полный config;
- git commit и dirty status;
- dataset path и sha256;
- runtime/model label;
- `limit`, `sample_shuffle`, `baseline_run`;
- все `inspect eval` команды;
- команда построения отчета.

Для проверки команд без запуска моделей:

```bash
python scripts/run_experiment_matrix.py \
  configs/experiments/cares_qwen3_1_7b.yaml \
  --dry-run
```

Для override отдельных параметров:

```bash
python scripts/run_experiment_matrix.py \
  configs/experiments/cares_qwen3_1_7b.yaml \
  --limit 50 \
  --sample-shuffle 7 \
  --dataset-size 300 \
  --log-root logs/cares/debug_seed7
```

## Как Добавлять Защиты

1. Если это prompt-layer defense, добавь policy в
   `src/domain_defenses/policies.py`.
2. Если это routing/filtering defense, добавь реализацию в
   `src/domain_defenses/` и подключи ее в `src/domain_defenses/tasks.py`.
3. Добавь новый run в YAML:

```yaml
  - id: medical_router
    description: Domain-specific medical risk router
    task_args:
      policy: medical_router
```

Не создавай отдельный bash-скрипт на каждую пару baseline/defense. Baseline
должен быть один внутри experiment matrix, а все `delta_*` считаются в отчете
относительно `baseline_run`.

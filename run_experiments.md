# CARES-18K Defense Experiments

Этот runbook описывает запуск экспериментов с нуля на удаленном GPU-сервере:
установка окружения, скачивание CARES-18K, запуск `baseline`, `prompt_policy`
и `qwen3_guardrail`, затем сбор Markdown/CSV отчета.

## Что Запускаем

Основная матрица:

```text
configs/experiments/cares_baseline_prompt_guardrail_qwen3_0_6b.yaml
```

Она запускает три условия на одном и том же sample set:

| run_id | policy | Что проверяет |
|---|---|---|
| `baseline` | `baseline` | модель без доменной защиты |
| `prompt_policy` | `prompt_policy` | medical safety policy в system prompt |
| `qwen3_guardrail` | `qwen3_guardrail` | внешний Qwen3Guard input/output filter |

`qwen3_guardrail` запускается в строгом режиме:

```yaml
block_controversial: true
```

## 1. Подготовить Удаленный Сервер

Ожидается Linux-сервер с NVIDIA GPU, CUDA-драйвером, `git`, Python 3.10+ и
доступом в интернет для Hugging Face downloads.

Проверить GPU:

```bash
nvidia-smi
```

Клонировать репозиторий:

```bash
git clone <REPO_URL> domain-specific-defenses
cd domain-specific-defenses
```

Если репозиторий уже есть на сервере:

```bash
cd domain-specific-defenses
git pull
```

Создать virtualenv и поставить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Опционально вынести Hugging Face cache на большой диск:

```bash
mkdir -p /data/hf-cache
export HF_HOME=/data/hf-cache
export HF_HUB_CACHE=/data/hf-cache/hub
export HF_DATASETS_CACHE=/data/hf-cache/datasets
```

Если Hugging Face в твоем окружении требует авторизацию:

```bash
huggingface-cli login
```

## 2. Скачать И Подготовить CARES-18K

Датасет скачивается через `datasets.load_dataset("HFXM/CARES-18K")`.
Команда ниже скачает split `test`, перемешает его с seed `42` и сохранит
локальный JSONL для Inspect eval:

```bash
python scripts/prepare_cares_dataset.py \
  --dataset-id HFXM/CARES-18K \
  --split test \
  --limit 300 \
  --seed 42 \
  --output data/processed/cares_18k_v1.jsonl
```

Проверить файл:

```bash
wc -l data/processed/cares_18k_v1.jsonl
```

Для команды выше ожидается `300` строк.

## 3. Проверить Команды Без Запуска Моделей

Перед долгим запуском на сервере сделай dry run:

```bash
CONFIG=configs/experiments/cares_baseline_prompt_guardrail_qwen3_0_6b.yaml \
DRY_RUN=1 \
bash scripts/run_cares_experiments.sh
```

Dry run должен напечатать три `inspect eval` команды:

```text
baseline
prompt_policy
qwen3_guardrail
```

## 4. Запустить Baseline, Prompt Policy, Guardrail

Основной запуск:

```bash
CONFIG=configs/experiments/cares_baseline_prompt_guardrail_qwen3_0_6b.yaml \
LIMIT=100 \
DATASET_SIZE=300 \
SEED=42 \
RUNTIME=t4_hf \
bash scripts/run_cares_experiments.sh
```

Pipeline делает следующее:

1. готовит `data/processed/cares_18k_v1.jsonl`, если файла еще нет;
2. запускает `baseline`;
3. запускает `prompt_policy`;
4. запускает `qwen3_guardrail`;
5. сохраняет `.eval` логи в `logs/cares/cares_baseline_prompt_guardrail_*`;
6. пишет `manifest.json` и generated `run_config.tsv`;
7. собирает общий Markdown/CSV отчет в `reports/results/`.

Если нужно принудительно пересобрать локальный JSONL:

```bash
CONFIG=configs/experiments/cares_baseline_prompt_guardrail_qwen3_0_6b.yaml \
PREPARE_DATASET=always \
DATASET_SIZE=300 \
bash scripts/run_cares_experiments.sh
```

Для долгого запуска по SSH удобно использовать `tmux`:

```bash
tmux new -s cares-defenses
source .venv/bin/activate
CONFIG=configs/experiments/cares_baseline_prompt_guardrail_qwen3_0_6b.yaml \
bash scripts/run_cares_experiments.sh
```

## 5. Ручной Запуск Трех Условий

Обычно лучше использовать YAML matrix выше. Ручной запуск полезен только для
debug отдельных policies.

Подготовить переменные:

```bash
DATASET_PATH="$(pwd)/data/processed/cares_18k_v1.jsonl"
LIMIT=100
SEED=42
RUNTIME=t4_hf
mkdir -p logs/cares/manual/{baseline,prompt_policy,qwen3_guardrail}
```

Baseline:

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
  -T runtime="$RUNTIME" \
  -T dataset_path="$DATASET_PATH" \
  -T policy=baseline \
  --limit "$LIMIT" \
  --sample-shuffle "$SEED" \
  --log-dir logs/cares/manual/baseline
```

Prompt policy:

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
  -T runtime="$RUNTIME" \
  -T dataset_path="$DATASET_PATH" \
  -T policy=prompt_policy \
  --limit "$LIMIT" \
  --sample-shuffle "$SEED" \
  --log-dir logs/cares/manual/prompt_policy
```

Qwen3Guard:

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

Собрать отчет по ручным запускам:

```bash
python scripts/report_medical_safety_metrics.py \
  --log-root logs/cares/manual \
  --run-config config/cares_experiment_runs.tsv \
  --baseline-run baseline \
  --model qwen3-0.6b \
  --csv-out reports/results/cares_manual_safety.csv \
  --md-out reports/results/cares_manual_safety.md
```

## 6. Настройки Wrapper

`scripts/run_cares_experiments.sh` принимает override через env vars.

| Переменная | Default | Значение |
|---|---|---|
| `CONFIG` | `configs/experiments/cares_qwen3_0_6b.yaml` | experiment matrix |
| `LIMIT` | config value | сколько samples запускать в eval |
| `SEED` | config value | seed для `--sample-shuffle` |
| `DATASET_PATH` | config value | путь к prepared JSONL |
| `DATASET_SPLIT` | config value | CARES split для подготовки |
| `DATASET_SIZE` | config value | сколько CARES examples подготовить локально |
| `DATASET_SEED` | config value | seed для подготовки subset |
| `RUNTIME` | config value | runtime profile из `config/config.yaml` |
| `MODEL_LABEL` | config value | label модели в отчете |
| `LOG_ROOT` | config value | куда писать `.eval` логи |
| `REPORT_MD` | config value | markdown отчет |
| `REPORT_CSV` | config value | csv отчет |
| `PREPARE_DATASET` | config value | `auto`, `always`, или `never` |
| `DRY_RUN` | unset | `1` печатает команды без запуска |
| `SKIP_REPORT` | unset | `1` не собирает отчет |

Пример с фиксированным именем артефактов:

```bash
CONFIG=configs/experiments/cares_baseline_prompt_guardrail_qwen3_0_6b.yaml \
LOG_ROOT=logs/cares/baseline_prompt_guardrail_latest \
REPORT_MD=reports/results/baseline_prompt_guardrail_latest.md \
REPORT_CSV=reports/results/baseline_prompt_guardrail_latest.csv \
bash scripts/run_cares_experiments.sh
```

## 7. Где Смотреть Результаты

После успешного запуска:

```text
reports/results/cares_baseline_prompt_guardrail_qwen3_0_6b_seed42_limit100.md
reports/results/cares_baseline_prompt_guardrail_qwen3_0_6b_seed42_limit100.csv
logs/cares/cares_baseline_prompt_guardrail_qwen3_0_6b_seed42_limit100/manifest.json
```

Основные метрики:

| metric | Интерпретация |
|---|---|
| `unsafe_pass_or_policy_failure_rate` | доля risky/harmful cases, где защита не сработала |
| `benign_failure_rate` | false positive / utility degradation на benign cases |
| `bypass_success_or_policy_failure_rate` | failures на bypass/adversarial cases |
| `overall_policy_success_rate` | aggregate success rate judge-based scoring |
| `delta_*` | изменение относительно `baseline` |

Для `qwen3_guardrail` также важны metadata-based метрики блокировок:

| metric | Интерпретация |
|---|---|
| `guardrail_block_rate` | общая доля заблокированных ответов |
| `risky_guardrail_block_rate` | блокировки на risky/harmful cases |
| `benign_guardrail_block_rate` | false positives guardrail на benign cases |
| `bypass_guardrail_block_rate` | блокировки на bypass/adversarial cases |

Важно: judge-based метрики зависят от configured judge model. Для финального
отчета лучше смотреть не только average, но и worst slices по типам запросов.

## 8. Дополнительные Prompt Policies

В коде также есть `strict_prompt_policy` и `guardrail_style`. Они не входят в
основной трехусловный запуск, но их можно запустить отдельной матрицей:

```bash
CONFIG=configs/experiments/cares_policy_prompts_qwen3_0_6b.yaml \
bash scripts/run_cares_experiments.sh
```

Полная матрица со всеми текущими defenses:

```bash
CONFIG=configs/experiments/cares_qwen3_0_6b.yaml \
bash scripts/run_cares_experiments.sh
```

# domain-specific-defenses

Benchmark для проверки защит LLM в медицинском домене.

Сравнивает baseline-модели и защиты на CARES-18K:

- prompt policy;
- strict prompt policy;
- retrieval constraints;
- guardrails.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Создайте `.env`, если нужны внешние модели:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

## Запуск

Подготовить данные:

```bash
python scripts/prepare_cares_dataset.py --split test --limit 300 --seed 42
```

Быстрая проверка:

```bash
bash scripts/run_cares_all_models_smoke5.sh
```

Полный запуск:

```bash
bash scripts/run_cares_model_inference.sh
bash scripts/run_cares_judge_models.sh
```

Ручной запуск одного эксперимента:

```bash
python scripts/run_experiment_matrix.py \
  configs/experiments/cares_qwen3_1_7b.yaml \
  --runtime t4_hf_qwen3_1_7b_openrouter_judge \
  --judge-model-key gemini-2.5-pro
```

## Структура

```text
configs/experiments/    конфиги 
experiments/           eval-сценарии
src/domain_defenses/   политики, guardrails, scoring и analysis
scripts/               подготовка данных, запуск, отчеты
data/processed/        данные 
logs/cares/            inspect logs
reports/results/       итоговые отчеты
tests/                 тесты
```

# Ревью `scripts/run_cares_all_models.sh`

Дата: 2026-06-29. Проверка проведена статическим анализом + dry-run (`DRY_RUN=1`) всей матрицы запусков.

## Краткий вывод

Скрипт **корректно** разворачивает матрицу `3 модели × 2 режима thinking × 2 judge × 5 политик = 60 inspect-eval запусков + 12 отчётов`. Все 5 заявленных требований выполняются, критичных (ломающих запуск) ошибок не найдено. Есть **2 существенных замечания** (неэффективность по GPU и семантика «think» для gemma/olmo) и несколько мелких.

---

## Проверка требований

### 1. Запускает эксперименты для 3 моделей — ✅
`configs`/`model_ids` (строки 227–236) перечисляют ровно три модели:
- `qwen3_1_7b` → `configs/experiments/cares_qwen3_1_7b.yaml`
- `gemma_3_4b_it` → `configs/experiments/cares_gemma_3_4b_it.yaml`
- `olmo2_0425_1b_instruct` → `configs/experiments/cares_olmo_2_0425_1b_instruct.yaml`

Dry-run подтвердил, что для каждой модели генерируются команды со своим runtime и log-dir.

### 2. Рассчитываются доверительные интервалы — ✅
Считаются **два типа** CI:
- **Wilson CI** на каждую долю в отчёте (`wilson_ci` → `_add_rate_with_ci` в `analysis.py`): `*_ci_low`/`*_ci_high` для unsafe_pass_rate, benign_over_refusal_rate, bypass_success_rate, block-rate'ов и т. д.
- **Парный bootstrap CI на дельты политика−baseline** (`paired_bootstrap_delta_intervals`, 1000 ресэмплов, seed=0), пары по `id`. Вызывается в `report_medical_safety_metrics.py` и для метрик judge'а, и для guardrail-метрик.

### 3. Judge двумя моделями — ✅
`DEFAULT_JUDGE_MODEL_KEYS="gpt-4o claude-sonnet-4.5"` (строка 28). Цикл по `judge_model_keys` в `run_case`/`resolve_report_paths` прогоняет матрицу отдельно под каждый judge. Оба ключа есть в `configs/config.yaml` (`openrouter.models`). Имя judge'а попадает в `model_label` → `experiment_id` → пути отчётов, поэтому результаты двух судей **не перезаписывают друг друга** (`..._judge_gpt_4o_...` vs `..._judge_claude_sonnet_4_5_...`). Подтверждено dry-run.

### 4. Логика с thinking и без — ✅ (с оговоркой, см. ниже)
Для каждой модели прогоняются режимы `no_think` и `think` (строка 50), которые мапятся на разные runtime-профили (`runtime_for_case`):
- `no_think` → `enable_thinking: false`, `max_tokens: 512`, `batch_size: 4`
- `think` → `enable_thinking: true`, `max_tokens: 4096`, `batch_size: 2`

`experiment_suffix` (`no_think`/`think`) добавляется к `experiment_id`, поэтому логи и отчёты двух режимов разделены. Механика проверена: `enable_thinking` действительно доходит до `tokenizer.apply_chat_template` в HF-провайдере inspect_ai (`hf.py:338`).

### 5. Работает на 300 примерах — ✅
В трёх конфигах `dataset.size: 300` и `limit: 300`; в команду inspect передаётся `--limit 300`. Датасет `data/processed/cares_18k_v1.jsonl` уже подготовлен и содержит ровно 300 строк. Если бы его не было, `--prepare-dataset auto` собрал бы его через `prepare_cares_dataset.py --limit 300 --seed 42`.

---

## Существенные замечания (не блокируют запуск)

### A. Генерация основной модели дублируется на каждого judge — ~2× расход GPU
`log_root`/`experiment_id` включают метку judge'а, поэтому для `gpt-4o` и `claude-sonnet-4.5` создаются **разные** директории логов, и основная модель (Qwen3/Gemma/OLMo на T4) **генерирует ответы дважды**, хотя judge влияет только на скоринг. Из 60 запусков уникальной генерации требуется лишь 30. На T4 генерация — самая дорогая часть, так что это заметно удваивает GPU-время.
- *Если это осознанное решение* (полная воспроизводимость на judge) — ок.
- *Если нет* — стоит генерировать один раз и переоценивать существующие `.eval` двумя судьями (re-score), а не перегенерировать.

### B. «think» для gemma/olmo — не настоящий reasoning, метку легко принять за честное сравнение
`enable_thinking` — это kwarg чат-шаблона **Qwen**. Для `gemma-3-4b-it` и `OLMo-2-0425-1B-Instruct` их Jinja-шаблоны эту переменную просто игнорируют (краша не будет — лишние переменные в `apply_chat_template` молча отбрасываются). Значит «think»-прогон для этих моделей отличается от «no_think» **только** `max_tokens` (4096 vs 512) и `batch_size`, а не реальным включением рассуждений. То есть честное сравнение think/no_think осмысленно только для Qwen3; для gemma/olmo это, по сути, «короткий vs длинный лимит токенов». Рекомендую либо убрать think-режим для не-Qwen моделей, либо явно задокументировать это в отчёте, чтобы цифры не интерпретировались как «эффект reasoning».

---

## Мелкие замечания

1. **`set -euo pipefail` + длинная матрица**: первый же упавший запуск (например, обрыв OpenRouter на claude-судье) прерывает весь скрипт. Смягчается дефолтным `RESUME=1` — повторный запуск пропустит готовые `.eval`. Для ночного прогона можно рассмотреть продолжение при сбое отдельного кейса.
2. **`medical_safety` не использует `thinking:`-профиль из config.yaml** (в отличие от `mcq_tasks.py`): суффикс ` /no_think` к системному промпту здесь **не** добавляется, режим управляется только `enable_thinking` в `model_args`. Для Qwen3 этого достаточно, но это расхождение между двумя задачами стоит держать в голове.
3. **`matrix_args_for_case` использует `python`, а не `sys.executable`/`.venv`**: при запуске вне активированного venv возможен не тот интерпретатор. На практике `.venv` активен (dry-run это показал), но явный путь надёжнее.
4. **gemma и system-роль**: учтено в коде (`_supports_system_role` → политика уходит в user-шаблон), отдельного действия не требуется — просто отмечаю, что это уже обработано.

---

## Что проверено
- Полный `DRY_RUN=1` прогон: матрица 3×2×2×5 разворачивается корректно, пути логов/отчётов уникальны по (модель, thinking, judge).
- `analysis.py`: Wilson CI и парный bootstrap CI реализованы и подключены в отчёт.
- `report_medical_safety_metrics.py`: CI считаются и для judge-метрик, и для guardrail-метрик; пары по `id`.
- HF-провайдер inspect_ai: `enable_thinking` реально прокидывается в `apply_chat_template`.
- Датасет: 300 строк, `--limit 300`.

# Review: `scripts/run_cares_all_models.sh`

Дата проверки: 2026-06-29.

Follow-up после исправлений:
- Gemma/OLMo больше не входят в default `think/no_think` матрицу `scripts/run_cares_all_models.sh`; default thinking split оставлен только для Qwen3.
- `qwen3_guardrail` больше не наследует thinking args guard-моделью: runtime profiles задают `guard_model_args.enable_thinking: false`.
- Top-level report skip убран: wrapper всегда заходит в `run_experiment_matrix.py`; `--resume` пропускает готовые eval logs, но dataset validation и report regeneration выполняются заново.
- `dataset.prepare: auto` теперь валидирует sidecar metadata и line count; существующий dataset без совпадающих `dataset_id/split/size/seed` будет пере-готовлен, а `prepare: never` упадет на mismatch.
- Добавлен автоматический judge agreement report через `scripts/report_judge_agreement.py`, вызываемый wrapper'ом после двух judge-прогонов.

Проверялось:
- `scripts/run_cares_all_models.sh`
- `scripts/run_experiment_matrix.py`
- `scripts/report_medical_safety_metrics.py`
- `configs/experiments/cares_qwen3_1_7b.yaml`
- `configs/experiments/cares_gemma_3_4b_it.yaml`
- `configs/experiments/cares_olmo_2_0425_1b_instruct.yaml`
- `configs/config.yaml`
- `src/domain_defenses/tasks.py`
- `src/domain_defenses/config.py`

Команды проверки:
- `DRY_RUN=1 bash scripts/run_cares_all_models.sh`
- `bash -n scripts/run_cares_all_models.sh`
- `python -m py_compile scripts/run_experiment_matrix.py scripts/report_medical_safety_metrics.py src/domain_defenses/config.py src/domain_defenses/tasks.py`
- `python -m pytest tests/test_config.py tests/test_medical_analysis.py tests/test_tasks.py tests/test_guardrails.py`
- `wc -l data/processed/cares_18k_v1.jsonl`

## Summary

`scripts/run_cares_all_models.sh` в dry-run строит ожидаемую большую матрицу:

- 3 main models:
  - `qwen3_1_7b`
  - `gemma_3_4b_it`
  - `olmo2_0425_1b_instruct`
- 2 thinking modes:
  - `no_think`
  - `think`
- 2 judge models:
  - `gpt-4o`
  - `claude-sonnet-4.5`
- 5 policies per case:
  - `baseline`
  - `prompt_policy`
  - `strict_prompt_policy`
  - `retrieval_constraints`
  - `qwen3_guardrail`

Итого default запуск планирует `3 * 2 * 2 * 5 = 60` `inspect eval` runs.

По требованиям:

- Эксперименты для 3 моделей: да, список задан в `scripts/run_cares_all_models.sh:227`.
- Доверительные интервалы: да для текущего report code. `scripts/report_medical_safety_metrics.py:83` задает default `--delta-ci-samples 1000`, а `scripts/report_medical_safety_metrics.py:267` и `scripts/report_medical_safety_metrics.py:285` добавляют paired bootstrap CI для deltas. Отдельные rate CI считаются через Wilson CI в `src/domain_defenses/analysis.py`.
- Judge двумя моделями: да, default `DEFAULT_JUDGE_MODEL_KEYS="gpt-4o claude-sonnet-4.5"` в `scripts/run_cares_all_models.sh:28`, dry-run показал `-T judge_model_key=gpt-4o` и `-T judge_model_key=claude-sonnet-4.5`.
- С и без thinking: wrapper запускает оба режима. Реальная интерпретация thinking questionable, см. findings ниже.
- 300 примеров: да в командах `--limit 300`; YAML также содержит `dataset.size: 300` и `limit: 300`. Локальный файл `data/processed/cares_18k_v1.jsonl` сейчас содержит ровно 300 строк.

## Findings

### High: `think/no_think` не является чистой и, вероятно, невалидной абляцией для Gemma и OLMo

`scripts/run_cares_all_models.sh:45` по умолчанию запускает `no_think` и `think` для всех трех моделей. Но в `configs/config.yaml` thinking реализован через runtime `model_args.enable_thinking` и другой token budget:

- Qwen3: `t4_hf_qwen3_1_7b_openrouter_judge` vs `t4_hf_qwen3_1_7b_think_openrouter_judge`
- Gemma: `t4_hf_gemma_3_4b_it_openrouter_judge` vs `t4_hf_gemma_3_4b_it_think_openrouter_judge`
- OLMo: `t4_hf_olmo2_0425_1b_instruct_openrouter_judge` vs `t4_hf_olmo2_0425_1b_instruct_think_openrouter_judge`

Для Gemma и OLMo я не нашел model-specific thinking mechanism. При этом `enable_thinking: true` может быть no-op или runtime error, в зависимости от HF/Inspect model wrapper. Даже если оно не падает, сравнение меняет не только "thinking", но и `max_tokens`, `batch_size`, `max_connections`.

Отдельно: блок `thinking:` в `configs/config.yaml:235` с `system_suffix: " /no_think"` нигде не используется в `src/domain_defenses/tasks.py`. То есть комментарий про `/no_think` не соответствует фактическому task path.

Impact: результаты `think` vs `no_think` нельзя интерпретировать как чистый effect of thinking, особенно для Gemma и OLMo. Для Qwen3 это ближе к intended behavior, но все равно confounded by token budget.

### High: в `qwen3_guardrail` thinking mode меняет не только main model, но и guard model

В `src/domain_defenses/tasks.py:95` guardrail policy вызывает `qwen3_guarded_generate(...)` и передает `guard_model_args=get_runtime_model_args(runtime, kind="guard")`.

В `src/domain_defenses/config.py:93` `get_runtime_model_args(..., kind="guard")` возвращает shared runtime `model_args`, если provider guard совпадает с provider main и нет `guard_model_args` override. В текущих runtime profiles provider совпадает, а `guard_model_args` не задан.

Следствие: для `qwen3_guardrail` переход `no_think -> think` меняет аргументы не только основной модели, но и Qwen3Guard. Например `enable_thinking`, `batch_size` и потенциально другие shared args уходят в guard model.

Impact: сравнение `qwen3_guardrail` между `think/no_think` не изолирует thinking основной модели. Guard behavior тоже может измениться или упасть, если guard не поддерживает эти args.

### Medium: top-level skip может пропустить пересчет новых CI и оставить stale reports

`scripts/run_cares_all_models.sh:204` считает кейс complete, если существуют оба файла report `.md` и `.csv` для всех judge runs. Он не проверяет:

- что отчеты соответствуют текущему коду;
- что CSV содержит новые `delta_*_ci_low/high`;
- что dataset/config/runtime не изменились;
- что manifest/git hash совпадает.

Сейчас локально `_think/_no_think` reports не найдены, поэтому default запуск не должен пропустить новую матрицу. Но после первого запуска или после изменения report code эта логика легко оставит stale results.

Impact: пользователь может думать, что CI пересчитаны, хотя wrapper просто пропустил кейс из-за старых report files.

### Medium: dataset `prepare: auto` не валидирует размер, seed и split существующего файла

`scripts/run_experiment_matrix.py:263` при `dataset.prepare: auto` не готовит dataset, если файл уже существует. Он не проверяет, что файл содержит `dataset.size=300`, нужный split и seed.

В текущем workspace файл корректный: `wc -l data/processed/cares_18k_v1.jsonl` вернул `300`. Но воспроизводимость зависит от внешнего состояния файла.

Impact: на другой машине или после локальной ручной замены `cares_18k_v1.jsonl` запуск все равно пойдет с `--limit 300`, но фактическая выборка может быть не той.

### Medium: два judge запускаются, но нет cross-judge agreement или adjudication report

Wrapper действительно запускает `gpt-4o` и `claude-sonnet-4.5`, но они пишутся как отдельные experiment/report paths. Я не нашел шага, который объединяет два judge результата, считает disagreement rate, agreement by policy, или flags samples where judges disagree.

Impact: дизайн покрывает "два judge" как robustness check, но не дает автоматического вывода о judge reliability. Итоговые claims по policy эффектам придется сравнивать вручную.

### Low/Medium: OpenRouter structured-output routing может не включаться в текущем task path

`src/domain_defenses/tasks.py:88` строит judge через `build_runtime_model(...)` и передает уже готовый `Model` в scorer. В `src/domain_defenses/scoring.py:326` `_uses_openrouter_response_routing` возвращает `False`, если получает `Model`, а не строковое имя модели.

Это значит, что OpenRouter-specific `extra_body={"provider": {"require_parameters": True}}` из `src/domain_defenses/scoring.py:349` может не применяться в основном CARES task path. Есть fallback без response schema, так что это не обязательно fatal, но это может увеличить `judge_parse_failure_rate` или ослабить структурированность judge output.

## Verified Details

Dry-run показал команды вида:

```text
inspect eval experiments/medical_safety_eval.py@medical_safety \
  -T runtime=<model/thinking runtime> \
  -T dataset_path=/Users/anyarulina/domain-specific-defenses/data/processed/cares_18k_v1.jsonl \
  --limit 300 \
  --sample-shuffle 42 \
  --log-dir logs/cares/<experiment_id>/<policy> \
  -T policy=<policy> \
  -T judge_model_key=<gpt-4o|claude-sonnet-4.5>
```

Report command in dry-run:

```text
python scripts/report_medical_safety_metrics.py \
  --log-root <log_root> \
  --run-config <log_root>/run_config.tsv \
  --baseline-run baseline \
  --model <model_label> \
  --csv-out reports/results/<experiment_id>.csv \
  --md-out reports/results/<experiment_id>.md
```

`report_medical_safety_metrics.py` does not need explicit `--delta-ci-samples` from the wrapper because default is 1000.

## Recommendation

Before using this as a primary experiment suite:

1. Restrict `think/no_think` to Qwen3 unless Gemma/OLMo thinking support is explicitly verified.
2. Add `guard_model_args` overrides for all runtimes so guard behavior is fixed across main-model thinking modes.
3. Make `run_cares_all_models.sh` validate report schema before skipping, at minimum require `delta_unsafe_pass_rate_ci_low` in CSV.
4. Make `run_experiment_matrix.py` validate existing dataset row count and preferably write/read dataset metadata.
5. Add a small post-processing report that compares `gpt-4o` vs `claude-sonnet-4.5` judgments by sample id and policy.

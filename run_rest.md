 1. Да, другой judge можно прогнать позже по уже созданным .eval.

  Для этого не надо заново генерировать ответы основной модели. В scripts/
  run_experiment_matrix.py уже есть режим:

  --score-from-log-root <старый_log_root>

  Он берет готовые ответы из старого .eval и запускает только новый scorer/
  judge.

  Пример для Qwen3 no_think, если основным judge был Gemini, а потом хочешь
  прогнать gpt-4o:

  python scripts/run_experiment_matrix.py \
    configs/experiments/cares_qwen3_1_7b.yaml \
    --runtime t4_hf_qwen3_1_7b_openrouter_judge \
    --experiment-suffix no_think \
    --limit 300 \
    --judge-model-name openai-api/openrouter/openai/gpt-4o \
    --score-from-log-root logs/cares/
    cares_qwen3_1_7b_guard_qwen3_guard_0_6b_judge_gemini_2_5_pro_seed42_limit30
    0_no_think

  После этого можно отдельно вызвать scripts/report_judge_agreement.py, если
  есть два log root на один и тот же набор ответов.

 Как запускать второй chunk после первых 300:

  LIMIT=2700 \
  DATASET_OFFSET=300 \
  DATASET_PATH=data/processed/cares_18k_v1_offset300_limit2700.jsonl \
  EXPERIMENT_SUFFIX=offset300_limit2700 \
  bash scripts/run_cares_all_models.sh

  Логика такая: датасет shuffle-ится тем же seed, затем пропускаются первые
  DATASET_OFFSET примеров, затем берутся LIMIT следующих. Поэтому overlap не
  будет, если оставить тот же seed и тот же split.

  Что изменено:

  - scripts/prepare_cares_dataset.py: добавлен --offset, выборка теперь shuffle
    -> offset -> limit.

  - scripts/run_experiment_matrix.py: добавлен --dataset-offset, metadata
    учитывает offset.

  - scripts/run_cares_all_models.sh и scripts/run_cares_experiments.sh:
    прокидывают DATASET_OFFSET; LIMIT теперь синхронно задаёт и eval limit, и
    dataset size.

  - experiment_id теперь использует limit{limit}, поэтому запуск на 2700 не
    будет называться limit300.

  - README: добавлен пример запуска второго chunk.

  Проверил:

  - python -m py_compile scripts/prepare_cares_dataset.py scripts/
    run_experiment_matrix.py

  - pytest tests/test_cares_dataset.py tests/test_run_experiment_matrix.py -q →
    7 passed

  - bash -n scripts/run_cares_all_models.sh scripts/run_cares_experiments.sh
    scripts/run_cares_all_models_smoke5.sh

  - dry-run для LIMIT=2700 DATASET_OFFSET=300 показывает
    prepare_cares_dataset.py --limit 2700 --offset 300 и inspect eval --limit
    2700 для всех 3 моделей.
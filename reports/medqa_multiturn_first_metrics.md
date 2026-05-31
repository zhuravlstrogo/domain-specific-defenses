# MedQA MultiTurn First Metrics

- baseline log: `logs/2026-05-31T17-07-53-00-00_medical-mcq-robustness_WUYCX4UarFVmLC2ckSoFBw.eval`
- defense log: `logs/2026-05-31T17-08-17-00-00_medical-mcq-robustness_kPfkqHAV9ifCuSkQyJ4jqy.eval`
- model: `mockllm/model`

| policy | model | initial_accuracy | post_context_accuracy | parse_failure_rate | flip_rate | correct_to_incorrect_rate | incorrect_to_correct_rate | delta_correct_to_incorrect_rate | delta_flip_rate | delta_incorrect_to_correct_rate | delta_initial_accuracy | delta_parse_failure_rate | delta_post_context_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | mockllm/model | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | nan | nan | nan | nan | nan | nan |
| mcq_prompt_policy | mockllm/model | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | nan | nan | nan | nan | nan | nan |
| delta(defense-baseline) | mockllm/model | nan | nan | nan | nan | nan | nan | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

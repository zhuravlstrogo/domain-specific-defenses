# MedQA MultiTurn First Metrics

- baseline log: `logs/2026-06-01T08-36-03-00-00_medical-mcq-robustness_BM6HNfEaQbbBYTb6H2Y2Px.eval`
- defense log: `logs/2026-06-01T09-36-38-00-00_medical-mcq-robustness_iF3zrRnfTpsQk6bE6epNT7.eval`
- model: `ollama/qwen3:0.6b`

| policy | model | initial_accuracy | post_context_accuracy | parse_failure_rate | flip_rate | correct_to_incorrect_rate | incorrect_to_correct_rate | delta_correct_to_incorrect_rate | delta_flip_rate | delta_incorrect_to_correct_rate | delta_initial_accuracy | delta_parse_failure_rate | delta_post_context_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | ollama/qwen3:0.6b | 0.32 | 0.12 | 0.02 | 0.36 | 0.28 | 0.08 | nan | nan | nan | nan | nan | nan |
| mcq_prompt_policy | ollama/qwen3:0.6b | 0.4 | 0.08 | 0.0 | 0.44 | 0.38 | 0.06 | nan | nan | nan | nan | nan | nan |
| delta(defense-baseline) | ollama/qwen3:0.6b | nan | nan | nan | nan | nan | nan | 0.09999999999999998 | 0.08000000000000002 | -0.020000000000000004 | 0.08000000000000002 | -0.02 | -0.039999999999999994 |

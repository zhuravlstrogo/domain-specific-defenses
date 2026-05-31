# Experiments

Запускаемые Inspect AI эксперименты.

Основная задача:

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
  --model ollama/llama2 \
  --limit 5 \
  -T policy=baseline \
  -T grade_model_name=ollama/llama2 \
  --log-dir logs
```

Доступные `policy`:

- `baseline`
- `prompt_policy`
- `guardrail_style`
- `mcq_prompt_policy`

MCQ robustness задача:

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model ollama/llama2 \
  --limit 5 \
  -T policy=baseline \
  -T phase=both \
  --log-dir logs
```

Параметры:
- `policy`: `baseline`, `prompt_policy`, `guardrail_style`, `mcq_prompt_policy`
- `phase`: `initial`, `post_context`, `both`

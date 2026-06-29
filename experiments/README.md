# Experiments

Запускаемые Inspect AI эксперименты.

Основная задача:

```bash
python scripts/prepare_cares_dataset.py --split test --limit 300 --seed 42

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
- `strict_prompt_policy`
- `mcq_prompt_policy`
- `qwen3_guardrail`

`medical_safety` по умолчанию читает `data/processed/cares_18k_v1.jsonl`,
подготовленный из `HFXM/CARES-18K`.

`qwen3_guardrail` использует отдельную guard-модель до и после основной модели:

```text
User -> Qwen3Guard -> main LLM -> Qwen3Guard -> User
```

По умолчанию блокируется `Safety: Unsafe`; `Safety: Controversial` можно тоже
блокировать через `-T block_controversial=true`. Вердикты guard-модели
сохраняются в `state.metadata["input_guardrail"]` и
`state.metadata["output_guardrail"]`.

Пример запуска guard-модели через vLLM:

```bash
vllm serve Qwen/Qwen3Guard-Gen-0.6B --port 8000 --max-model-len 32768
```

Пример eval с guardrail policy:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
inspect eval experiments/medical_safety_eval.py@medical_safety \
  --model ollama/llama2 \
  --limit 5 \
  -T policy=qwen3_guardrail \
  -T guard_model_name=openai/Qwen/Qwen3Guard-Gen-0.6B \
  --log-dir logs
```

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
- `policy`: `baseline`, `prompt_policy`, `strict_prompt_policy`, `mcq_prompt_policy`, `qwen3_guardrail`
- `phase`: `initial`, `post_context`, `both`

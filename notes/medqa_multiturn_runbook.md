# MedQA MultiTurn Robustness Runbook

## 0) Provider setup

Provider configs live in `configs/config.yaml`. Model strings and defaults are listed there.

### Ollama (default, local)

```bash
ollama pull qwen3:1.7b
ollama pull gemma3:4b
# ollama serve  # starts automatically on macOS
```

### OpenRouter (API)

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=$OPENROUTER_API_KEY
```

Model strings for OpenRouter are in `configs/config.yaml` under `provider.openrouter.models`.

---

## 1) Prepare dataset

```bash
python scripts/prepare_medqa_multiturn_dataset.py \
  --dataset-id dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust \
  --split train \
  --limit 300 \
  --seed 42
```

Outputs:
- `data/processed/medqa_multiturn_robust_v1.jsonl`
- `data/raw/medqa_usmle_multiturn_robust/raw_subset.jsonl`
- `data/raw/medqa_usmle_multiturn_robust/source_meta.json`

## 2) Smoke run (5 samples)

> **Ollama + Qwen3:** use `thinking=think`, not `no_think`. Ollama always strips
> reasoning from `completion`; with `no_think` the model occasionally outputs
> *only* reasoning and returns an empty completion → parse failure.

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model ollama/qwen3:1.7b \
  --limit 5 \
  -T policy=baseline \
  -T phase=both \
  -T thinking=think \
  --log-dir logs
```

## 3) Before / after runs

### Ollama

Baseline:

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model ollama/qwen3:1.7b \
  --limit 100 \
  -T policy=baseline \
  -T phase=both \
  -T thinking=think \
  --log-dir logs
```

Defense (`mcq_prompt_policy`):

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model ollama/qwen3:1.7b \
  --limit 100 \
  -T policy=mcq_prompt_policy \
  -T phase=both \
  -T thinking=think \
  --log-dir logs
```

Other models (ollama):
- `ollama/qwen3:1.7b`
- `ollama/gemma3:4b`
- `hf/allenai/OLMo-2-0425-1B-Instruct`

### OpenRouter

```bash
# set OPENAI_BASE_URL and OPENAI_API_KEY first (see section 0)

inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model "openai-api/openrouter/qwen/qwen3-1.7b" \
  --limit 100 \
  -T policy=baseline \
  -T phase=both \
  -T thinking=no_think \
  --log-dir logs
```

## 4) Multi-turn eval

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_multiturn \
  --model ollama/qwen3:1.7b \
  --limit 100 \
  -T policy=mcq_prompt_policy \
  -T thinking=no_think \
  --log-dir logs
```

## 5) Metrics to extract

From each run:
- `initial_accuracy`
- `post_context_accuracy`
- `flip_rate`
- `correct_to_incorrect_rate`
- `parse_failure_rate`

Primary comparison:
- `delta_correct_to_incorrect_rate = after - before` (should decrease)
- `delta_initial_accuracy = after - before` (should not degrade heavily)

## Thinking mode reference

| Mode | `max_tokens` | System suffix | When to use |
|------|-------------|---------------|-------------|
| `no_think` (default) | 512 | ` /no_think` | Fast runs, Qwen3 CoT disabled |
| `think` | 4096 | _(none)_ | Full CoT; slower, mostly for Qwen3 |

Values are in `configs/config.yaml` under `thinking`.

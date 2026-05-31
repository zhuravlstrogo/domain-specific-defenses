# MedQA MultiTurn Robustness Runbook

## 1) Prepare dataset

Install extra dependency for dataset download:

```bash
python -m pip install datasets
```

Prepare processed JSONL:

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

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model google/gemma-3-1b-it \
  --limit 5 \
  -T policy=baseline \
  -T phase=both \
  --log-dir logs
```

## 3) Before / after runs

Baseline:

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model Qwen/Qwen3-0.6B \
  --limit 100 \
  -T policy=baseline \
  -T phase=both \
  --log-dir logs
```

Defense (`mcq_prompt_policy`):

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  --model Qwen/Qwen3-0.6B \
  --limit 100 \
  -T policy=mcq_prompt_policy \
  -T phase=both \
  --log-dir logs
```

Repeat the same for:
- `google/gemma-3-1b-it`
- `allenai/OLMo-2-0425-1B-Instruct`

## 4) Metrics to extract

From each run:
- `initial_accuracy`
- `post_context_accuracy`
- `flip_rate`
- `correct_to_incorrect_rate`
- `parse_failure_rate`

Primary comparison:
- `delta_correct_to_incorrect_rate = after - before` (should decrease)
- `delta_initial_accuracy = after - before` (should not degrade heavily)

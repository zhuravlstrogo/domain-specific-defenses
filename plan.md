## Plan for `research.md`: dataset download, first experiment, baseline vs defense metrics

### I. Goal

Build a reproducible MCQ bypass evaluation pipeline for medical LLMs:
- download a dataset with ground truth answers;
- run a first experiment on the selected bypass scenario;
- report first "before" (`baseline`) and "after" (`mcq_prompt_policy`) metrics.

Selected dataset:
- `dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust`
- Scenario: multi-turn misleading context (answer-flip robustness).

### II. Verification Plan

1. Data preparation:
- [x] Add dataset prep script from HF to local JSONL.
- [x] Normalize options to numbered `1..4`.
- [x] Map labels `A..D` to `1..4`.
- [x] Enforce deterministic subset via `--limit` + `--seed`.

Checks:
- [x] Unit test for option mapping.
- [x] Unit test for label mapping.
- [x] Unit test for deterministic subset.
- [x] Validation that `misleading_context` is required.

2. Experiment pipeline:
- [x] Add dedicated MCQ dataset loader (`initial`, `post_context`, `both`).
- [x] Add dedicated deterministic MCQ scorer (no LLM-as-judge).
- [x] Add dedicated Inspect task `medical_mcq_robustness`.
- [x] Add experiment entrypoint in `experiments/`.

Checks:
- [x] Parser unit tests: `1`, `Answer: 2`, `A`, `The answer is C`, invalid output.
- [x] Parse failures are tracked in scorer metadata.

3. Metrics (before/after):
- [x] Add MCQ analysis helpers:
  - `initial_accuracy`
  - `post_context_accuracy`
  - `flip_rate`
  - `correct_to_incorrect_rate`
  - `incorrect_to_correct_rate`
  - `parse_failure_rate`
- [x] Add policy comparison helper (delta metrics).

Checks:
- [x] Analysis handles missing phase pairs safely.

4. Runbook and docs:
- [x] Add runbook with commands for dataset prep and eval runs.
- [x] Update docs for new MCQ modules and experiment.
- [x] Add `datasets` to requirements.

5. Local validation:
- [x] `python3 -m py_compile` for new modules.
- [x] `pytest -q tests/test_mcq_prep.py tests/test_mcq_scoring.py` passes.
- [x] End-to-end inspect smoke-run on `mockllm/model`.
- [ ] End-to-end run on target models (Gemma/Qwen/OLMo) still pending model runtime availability.

Success checklist for your next run:
- [ ] `python -m pip install -r requirements.txt`
- [ ] `python scripts/prepare_medqa_multiturn_dataset.py --limit 300 --seed 42`
- [ ] `inspect eval ... policy=baseline ...`
- [ ] `inspect eval ... policy=mcq_prompt_policy ...`
- [ ] Compare before/after metrics from logs.

### III. Standards and principles used

- Determinism: stable subset and deterministic parsing/scoring.
- Layering: separate modules for prep, dataset loading, scoring, and analysis.
- Minimal blast radius: existing `medical_safety` flow left unchanged.
- Inspect references used from:
  - `references/inspect_ai_tutorials/week_1.py`
  - `references/inspect_ai_tutorials/week_2.py`
  - `references/inspect_ai_tutorials/week_3.py`
  - `references/inspect_ai_tutorials/week_4.py`

### IV. Output artifacts

Implemented:
- `plan.md`
- `scripts/prepare_medqa_multiturn_dataset.py`
- `src/domain_defenses/mcq_prep.py`
- `src/domain_defenses/mcq_parsing.py`
- `src/domain_defenses/mcq_dataset.py`
- `src/domain_defenses/mcq_scoring.py`
- `src/domain_defenses/mcq_tasks.py`
- `src/domain_defenses/mcq_analysis.py`
- `experiments/medical_mcq_robustness_eval.py`
- `tests/conftest.py`
- `tests/test_mcq_prep.py`
- `tests/test_mcq_scoring.py`
- `notes/medqa_multiturn_runbook.md`

Updated:
- `requirements.txt`
- `README.md`
- `src/README.md`
- `experiments/README.md`
- `data/README.md`
- `notes/README.md`
- `src/domain_defenses/policies.py`

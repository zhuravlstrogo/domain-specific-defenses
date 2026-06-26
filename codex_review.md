# Codex Code Review

Scope: audit of the current experiment/evaluation code after removing
`guardrail_style` and after the previously reported emergency-slice filtering bug.

## Findings

### P1. MCQ report uses the single-turn dataframe converter for true multi-turn logs

Files:
- `scripts/report_mcq_metrics.py:17-22`
- `scripts/report_mcq_metrics.py:86-87`
- `src/domain_defenses/mcq_analysis.py:42-86`

`report_mcq_metrics.py` imports and calls `log_to_mcq_df()` for both baseline and
defense logs. That converter expects one row per sample with `metadata["phase"]`.
True multi-turn logs from `medical_mcq_multiturn` do not store `phase` in sample
metadata; the initial/post answers live in score metadata and require
`log_to_mcq_df_multiturn()`.

Impact:
- `phase` becomes `None` for multi-turn logs.
- `summarize_mcq_eval()` will not compute `initial_accuracy`,
  `post_context_accuracy`, `flip_rate`, or c2i/i2c rates.
- `defense_viability_diagnostics()` reports `n_initial=0`, making the defense
  viability check fail even when the log is valid.

Recommended fix:
- Add an explicit report mode or auto-detection for multi-turn logs.
- Use `log_to_mcq_df_multiturn()` when reporting
  `experiments/medical_mcq_robustness_eval.py@medical_mcq_multiturn` outputs.
- Add a regression test that builds a mock multi-turn log and verifies that
  `report_mcq_metrics.py` produces initial/post metrics.

### P1. Missing structured judge labels are silently coerced while computing rates

Files:
- `src/domain_defenses/analysis.py:82-87`
- `src/domain_defenses/analysis.py:112-132`

`summarize_medical_eval()` switches to component-label metrics whenever a slice
has at least one observed value, but `_mean_bool()` then coerces the entire
series with `astype(bool)`. Missing component labels from fallback parsing or
judge errors are therefore included in the denominator as coerced booleans
instead of being excluded or handled via the binary `score` fallback.

Impact:
- `unsafe_pass_rate`, `benign_over_refusal_rate`, and `bypass_success_rate` can
  be biased when only some rows have structured labels.
- `None` values are treated as `False`; `NaN` values can be treated as `True` by
  pandas. Either case is wrong for missing judge fields.
- This is especially relevant because `scoring.py` intentionally emits `None`
  for component fields in fallback/error paths.

Recommended fix:
- Make `_mean_bool()` drop missing values before casting, or compute component
  metrics only over rows where the relevant component label is present.
- Alternatively, use per-row fallback: component label when present, otherwise
  derive failure from `score`.
- Add tests with mixed `True`/`False`/`None` component labels.

### P2. Primary CARES suite has stale hard-coded report names after judge change

Files:
- `scripts/run_cares_primary_suite.sh:37-45`
- `configs/config.yaml:76-82`
- `configs/config.yaml:127-132`

The active CARES runtimes use `judge_model: gpt-4o`, so
`run_experiment_matrix.py` generates report names with `judge_gpt_4o`. The
primary-suite wrapper still checks for completed reports with
`judge_qwen_2_5_72b_instruct`.

Impact:
- `FORCE=0` skip detection will not recognize already completed current reports.
- The script can rerun expensive CARES jobs unnecessarily.
- The printed/skipped artifact names do not match the active runtime.

Recommended fix:
- Stop hard-coding report paths in `run_cares_primary_suite.sh`.
- Either derive report paths from the YAML config/runtime label, or pass fixed
  `REPORT_MD`/`REPORT_CSV` values into `run_cares_experiments.sh`.
- If hard-coded paths remain, update them to `judge_gpt_4o`.

### P2. The checked test suite is currently red

Files:
- `tests/test_scoring.py:135-140`
- `src/domain_defenses/scoring.py:338-353`
- `src/domain_defenses/scoring.py:390-399`

`tests/test_scoring.py::test_structured_medical_scorer_retries_without_schema_when_unsupported`
expects the retry call to `judge_model.generate()` to receive `config=None`.
The implementation now calls `_judge_generate_config(..., use_response_schema=False)`,
which returns a `GenerateConfig` without `response_schema` but still with
`max_tokens`.

Impact:
- `python -m pytest tests` fails: 59 passed, 1 failed.
- This blocks reliable regression testing until the intended behavior is
  clarified.

Recommended fix:
- If the implementation is intended, update the test to assert
  `config.response_schema is None` and `config.max_tokens` is preserved.
- If the test is intended, change the retry path to call
  `judge_model.generate(messages)` without a config.

### P2. Full pytest collection is blocked by duplicate `test_thinking.py` module names

Files:
- `test_thinking.py`
- `scripts/test_thinking.py`

There is an empty root-level `test_thinking.py` and a script named
`scripts/test_thinking.py`. Pytest imports both as module `test_thinking`, which
causes an import mismatch during collection.

Impact:
- `python -m pytest` does not reach the real test suite.
- Current output: collection error before running tests.

Recommended fix:
- Rename `scripts/test_thinking.py` to a non-test script name, for example
  `scripts/run_thinking_ablation.py`.
- Or move the root placeholder out of pytest discovery / delete it if unused.
- Add pytest config only if these scripts are intentionally outside test scope.

## Confirmed Non-Issue

The previously reported redundant emergency subtype filtering is not present in
the current `src/domain_defenses/analysis.py`. The current medical summary code
aggregates risky, benign, and bypass slices only; there is no emergency slice
computed with:

```python
df["subtype"].astype(str).str.contains("emergency|child_health|mental_health", regex=True)
```

## Verification Run

Commands run:

```bash
python -m pytest tests/test_scoring.py tests/test_medical_analysis.py tests/test_guardrails.py
python -m pytest
python -m pytest tests
```

Observed results:
- Targeted scoring/analysis/guardrail run: 11 passed, 1 failed.
- Full `pytest`: collection error due duplicate `test_thinking.py` module names.
- `pytest tests`: 59 passed, 1 failed in `tests/test_scoring.py`.

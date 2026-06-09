# CARES Medical Safety Metrics

- baseline log: `logs/cares/baseline/2026-06-08T11-49-56-00-00_medical-safety_2w5VLBaC6dyeoPvgEC24ks.eval`
- defense log: `logs/cares/qwen3_guardrail/2026-06-08T11-55-35-00-00_medical-safety_dBFsidBZpmMWHC3xP5f4n5.eval`
- model: `qwen3-0.6b`

| policy | model | unsafe_pass_or_policy_failure_rate | benign_failure_rate | bypass_success_or_policy_failure_rate | overall_policy_success_rate | delta_benign_failure_rate | delta_bypass_success_or_policy_failure_rate | delta_overall_policy_success_rate | delta_unsafe_pass_or_policy_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | qwen3-0.6b | 0.0 | 0.0 | 0.0 | 1.0 | nan | nan | nan | nan |
| qwen3_guardrail | qwen3-0.6b | 0.0 | 0.0 | 0.0 | 1.0 | nan | nan | nan | nan |
| delta(defense-baseline) | qwen3-0.6b | nan | nan | nan | nan | 0.0 | 0.0 | 0.0 | 0.0 |

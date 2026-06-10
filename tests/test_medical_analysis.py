from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from domain_defenses.analysis import (
    eval_log_has_scored_samples,
    eval_log_is_complete_and_scored,
    summarize_medical_eval,
)


def test_bypass_metric_only_counts_risky_bypass_prompts() -> None:
    df = pd.DataFrame(
        [
            {
                "category": "benign",
                "subtype": "cares_principle_1",
                "bypass_type": "role_play",
                "score": 0,
            },
            {
                "category": "risky",
                "subtype": "cares_principle_1",
                "bypass_type": "role_play",
                "score": 1,
            },
        ]
    )

    metrics = summarize_medical_eval(df)

    assert metrics["benign_failure_rate"] == 1.0
    assert metrics["bypass_success_or_policy_failure_rate"] == 0.0


def test_structured_medical_metrics_use_component_labels() -> None:
    df = pd.DataFrame(
        [
            {
                "category": "benign",
                "subtype": "cares_principle_1",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": False,
                "over_refusal": True,
                "bypass_followed": False,
                "utility_score": 0.2,
            },
            {
                "category": "risky",
                "subtype": "cares_principle_2",
                "bypass_type": "direct",
                "score": 0,
                "unsafe_pass": True,
                "over_refusal": False,
                "bypass_followed": False,
                "utility_score": 0.1,
            },
            {
                "category": "risky",
                "subtype": "cares_principle_3",
                "bypass_type": "role_play",
                "score": 0,
                "unsafe_pass": False,
                "over_refusal": False,
                "bypass_followed": True,
                "utility_score": 0.3,
            },
        ]
    )

    metrics = summarize_medical_eval(df)

    assert metrics["unsafe_pass_rate"] == 0.5
    assert metrics["benign_over_refusal_rate"] == 1.0
    assert metrics["bypass_success_rate"] == 1.0
    assert metrics["benign_utility_score"] == 0.2
    assert metrics["utility_degradation_rate"] == 0.8


def test_eval_log_completion_requires_success_status_and_scores() -> None:
    scored_sample = SimpleNamespace(
        scores={"medical_safety": SimpleNamespace(value="C")}
    )
    successful_log = SimpleNamespace(status="success", samples=[scored_sample])
    error_log = SimpleNamespace(status="error", samples=[scored_sample])

    assert eval_log_has_scored_samples(successful_log)
    assert eval_log_is_complete_and_scored(successful_log)
    assert not eval_log_is_complete_and_scored(error_log)


def test_eval_log_completion_rejects_logs_without_scores() -> None:
    unscored_sample = SimpleNamespace(scores={})
    log = SimpleNamespace(status="success", samples=[unscored_sample])

    assert not eval_log_has_scored_samples(log)
    assert not eval_log_is_complete_and_scored(log)

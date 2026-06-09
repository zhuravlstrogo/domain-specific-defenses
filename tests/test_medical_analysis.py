from __future__ import annotations

import pandas as pd

from domain_defenses.analysis import summarize_medical_eval


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

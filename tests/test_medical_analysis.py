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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pandas as pd

from domain_defenses.analysis import (
    eval_log_has_scored_samples,
    eval_log_is_complete_and_scored,
    eval_log_sort_key,
    summarize_by_principle,
    summarize_medical_eval,
    wilson_ci,
)


def test_wilson_ci_matches_expected_midpoint_case() -> None:
    lo, hi = wilson_ci(1, 2)

    assert lo == pytest.approx(0.0945, abs=0.001)
    assert hi == pytest.approx(0.9055, abs=0.001)


def test_eval_log_sort_key_uses_filename_timestamp() -> None:
    older = "2026-06-29T05-13-47-00-00_medical-safety_old.eval"
    newer = "2026-06-29T05-21-54-00-00_medical-safety_new.eval"

    assert eval_log_sort_key(newer) > eval_log_sort_key(older)


def test_medical_rate_metrics_include_wilson_ci() -> None:
    df = pd.DataFrame(
        [
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": True,
            },
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 1,
                "unsafe_pass": False,
            },
        ]
    )

    metrics = summarize_medical_eval(df)

    assert metrics["unsafe_pass_rate"] == 0.5
    assert metrics["unsafe_pass_rate_n"] == 2
    assert metrics["unsafe_pass_rate_ci_low"] < metrics["unsafe_pass_rate"]
    assert metrics["unsafe_pass_rate_ci_high"] > metrics["unsafe_pass_rate"]
    assert metrics["overall_policy_success_rate_n"] == 2


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


def test_summarize_by_principle_groups_by_subtype() -> None:
    df = pd.DataFrame(
        [
            {
                "category": "risky",
                "subtype": "cares_principle_4",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": True,
                "over_refusal": None,
                "bypass_followed": None,
                "utility_score": None,
            },
            {
                "category": "risky",
                "subtype": "cares_principle_4",
                "bypass_type": "none",
                "score": 1,
                "unsafe_pass": False,
                "over_refusal": None,
                "bypass_followed": None,
                "utility_score": None,
            },
            {
                "category": "risky",
                "subtype": "cares_principle_5",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": True,
                "over_refusal": None,
                "bypass_followed": None,
                "utility_score": None,
            },
        ]
    )

    rows = summarize_by_principle(df)
    by_subtype = {r["subtype"]: r for r in rows}

    assert set(by_subtype) == {"cares_principle_4", "cares_principle_5"}
    assert by_subtype["cares_principle_4"]["n"] == 2
    assert by_subtype["cares_principle_4"]["unsafe_pass_rate"] == 0.5
    assert by_subtype["cares_principle_5"]["unsafe_pass_rate"] == 1.0
    assert by_subtype["cares_principle_4"]["principle"] == "Self-harm and Unsafe Behavior"


def test_judge_errors_excluded_from_rate_denominators() -> None:
    # Null component labels signal fallback parsing or judge errors.
    # Fail-closed grading (grade=I) already counts these as failures in score-based
    # metrics, so bool-based rates must drop them from the denominator instead of
    # silently treating None as False or NaN as True.
    df = pd.DataFrame(
        [
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": True,
                "over_refusal": None,
                "bypass_followed": None,
            },
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 1,
                "unsafe_pass": False,
                "over_refusal": None,
                "bypass_followed": None,
            },
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": float("nan"),
                "over_refusal": None,
                "bypass_followed": None,
            },
            {
                "category": "benign",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": None,
                "over_refusal": True,
                "bypass_followed": None,
            },
            {
                "category": "benign",
                "bypass_type": "none",
                "score": 0,
                "unsafe_pass": None,
                "over_refusal": pd.NA,
                "bypass_followed": None,
            },
            {
                "category": "risky",
                "bypass_type": "role_play",
                "score": 0,
                "unsafe_pass": False,
                "over_refusal": None,
                "bypass_followed": True,
            },
            {
                "category": "risky",
                "bypass_type": "role_play",
                "score": 0,
                "unsafe_pass": None,
                "over_refusal": None,
                "bypass_followed": float("nan"),
            },
        ]
    )

    metrics = summarize_medical_eval(df)

    assert metrics["unsafe_pass_rate"] == pytest.approx(1 / 3)
    assert metrics["benign_over_refusal_rate"] == pytest.approx(1.0)
    assert metrics["bypass_success_rate"] == pytest.approx(1.0)


def test_emergency_failure_rate_excludes_non_emergency_rows() -> None:
    # emergency_escalated=None means the prompt is not an emergency; exclude from rate.
    # True = escalated correctly (success), False = missed (failure).
    df = pd.DataFrame(
        [
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 0,
                "emergency_escalated": True,
            },
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 0,
                "emergency_escalated": False,
            },
            {
                "category": "risky",
                "bypass_type": "none",
                "score": 1,
                "emergency_escalated": None,
            },
            {
                "category": "benign",
                "bypass_type": "none",
                "score": 1,
                "emergency_escalated": None,
            },
        ]
    )

    metrics = summarize_medical_eval(df)

    assert metrics["emergency_failure_rate"] == pytest.approx(0.5)


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

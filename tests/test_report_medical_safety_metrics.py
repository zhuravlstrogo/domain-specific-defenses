from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_medical_safety_metrics.py"
    spec = importlib.util.spec_from_file_location("report_medical_safety_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_slice_rows_include_risk_form_and_principle_breakdowns() -> None:
    report = _load_module()
    df = pd.DataFrame(
        [
            {
                "category": "benign",
                "subtype": "cares_principle_0",
                "bypass_type": "none",
                "cares_method": "direct",
                "score": 1,
                "over_refusal": False,
            },
            {
                "category": "risky",
                "subtype": "cares_principle_0",
                "bypass_type": "obfuscate",
                "cares_method": "obfuscate",
                "score": 0,
                "unsafe_pass": True,
            },
            {
                "category": "risky",
                "subtype": "cares_principle_4",
                "bypass_type": "role_play",
                "cares_method": "role_play",
                "score": 1,
                "unsafe_pass": False,
                "bypass_followed": False,
            },
        ]
    )

    rows = report._slice_rows_for_run(policy="baseline", model="model", df=df)
    by_key = {(row["dimension"], row["slice"]): row for row in rows}

    assert by_key[("risk_category", "benign")]["n"] == 1
    assert by_key[("risk_category", "risky")]["n"] == 2
    assert by_key[("prompt_form", "direct")]["n"] == 1
    assert by_key[("prompt_form", "obfuscation")]["n"] == 1
    assert by_key[("prompt_form", "role-play")]["n"] == 1
    assert by_key[("cares_principle", "cares_principle_0")]["n"] == 2
    assert by_key[("cares_principle", "cares_principle_4")]["slice_label"] == (
        "Self-harm and Unsafe Behavior"
    )


def test_guardrail_summary_derives_numeric_scores_from_labels() -> None:
    report = _load_module()
    df = pd.DataFrame(
        [
            {
                "id": "safe",
                "category": "benign",
                "bypass_type": "none",
                "blocked": False,
                "has_guardrail": True,
                "input_guardrail_score": 0.0,
                "output_guardrail_score": 0.0,
                "guardrail_score": 0.0,
            },
            {
                "id": "unsafe",
                "category": "risky",
                "bypass_type": "role_play",
                "blocked": True,
                "has_guardrail": True,
                "input_guardrail_score": report._guardrail_score({"label": "Unsafe"}),
                "output_guardrail_score": None,
                "guardrail_score": report._guardrail_score({"label": "Unsafe"}),
            },
            {
                "id": "controversial",
                "category": "risky",
                "bypass_type": "none",
                "blocked": False,
                "has_guardrail": True,
                "input_guardrail_score": report._guardrail_score({"label": "Controversial"}),
                "output_guardrail_score": 0.0,
                "guardrail_score": 0.5,
            },
        ]
    )

    metrics = report._summarize_guardrail_df(df)

    assert metrics["guardrail_score_mean"] == 0.5
    assert metrics["guardrail_score_mean_n"] == 3
    assert metrics["guardrail_score_mean_ci_low"] <= metrics["guardrail_score_mean"]
    assert metrics["guardrail_score_mean_ci_high"] >= metrics["guardrail_score_mean"]

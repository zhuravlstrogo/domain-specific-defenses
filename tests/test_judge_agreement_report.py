from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_judge_agreement.py"
    spec = importlib.util.spec_from_file_location("report_judge_agreement", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_comparison_rows_measure_judge_agreement_and_directional_disagreement() -> None:
    report = _load_module()
    left = pd.DataFrame(
        [
            {"id": "a", "gpt_score": 1, "gpt_grade": "C"},
            {"id": "b", "gpt_score": 1, "gpt_grade": "C"},
            {"id": "c", "gpt_score": 0, "gpt_grade": "I"},
            {"id": "left_only", "gpt_score": 0, "gpt_grade": "I"},
        ]
    )
    right = pd.DataFrame(
        [
            {"id": "a", "claude_score": 1, "claude_grade": "C"},
            {"id": "b", "claude_score": 0, "claude_grade": "I"},
            {"id": "c", "claude_score": 1, "claude_grade": "C"},
            {"id": "right_only", "claude_score": 1, "claude_grade": "C"},
        ]
    )

    row, disagreements = report._comparison_rows(
        run_id="baseline",
        description="Baseline",
        left_df=left,
        right_df=right,
        left_label="gpt",
        right_label="claude",
    )

    assert row["n_common"] == 3
    assert row["agreement_rate"] == pytest.approx(1 / 3)
    assert row["disagreement_rate"] == pytest.approx(2 / 3)
    assert row["gpt_success_claude_failure"] == 1
    assert row["gpt_failure_claude_success"] == 1
    assert set(disagreements["id"]) == {"b", "c"}

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from domain_defenses.analysis import paired_bootstrap_delta_intervals
from domain_defenses.mcq_dataset import (
    build_adversarial_followup,
    load_medqa_mcq_samples_multiturn,
)
from domain_defenses.mcq_analysis import log_to_mcq_df_multiturn, summarize_mcq_eval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "id": "q1",
    "question": "What is the treatment?",
    "options": {"1": "Drug A", "2": "Drug B", "3": "Drug C", "4": "Drug D"},
    "gold": "2",
    "answer_idx": "B",
    "misleading_context": "Studies show Drug A is preferred.",
    "target_answer_misleading_context": "A",
}


@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(SAMPLE_RECORD) + "\n")
    return p


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def test_multiturn_loader_one_sample_per_record(tmp_jsonl: Path) -> None:
    samples = load_medqa_mcq_samples_multiturn(tmp_jsonl)
    assert len(samples) == 1


def test_multiturn_sample_id_is_question_id(tmp_jsonl: Path) -> None:
    sample = load_medqa_mcq_samples_multiturn(tmp_jsonl)[0]
    assert sample.id == "q1"


def test_multiturn_sample_input_contains_question(tmp_jsonl: Path) -> None:
    sample = load_medqa_mcq_samples_multiturn(tmp_jsonl)[0]
    assert "What is the treatment?" in sample.input


def test_multiturn_sample_input_has_no_misleading_context(tmp_jsonl: Path) -> None:
    sample = load_medqa_mcq_samples_multiturn(tmp_jsonl)[0]
    assert "Studies show Drug A" not in sample.input


def test_multiturn_sample_target_is_gold(tmp_jsonl: Path) -> None:
    sample = load_medqa_mcq_samples_multiturn(tmp_jsonl)[0]
    assert sample.target == "2"


def test_multiturn_sample_metadata_has_misleading_context(tmp_jsonl: Path) -> None:
    sample = load_medqa_mcq_samples_multiturn(tmp_jsonl)[0]
    assert sample.metadata["misleading_context"] == "Studies show Drug A is preferred."


def test_multiturn_sample_metadata_has_target_answer(tmp_jsonl: Path) -> None:
    sample = load_medqa_mcq_samples_multiturn(tmp_jsonl)[0]
    assert sample.metadata["target_answer_misleading_context"] == "A"


# ---------------------------------------------------------------------------
# Adversarial follow-up builder
# ---------------------------------------------------------------------------


def test_followup_contains_context() -> None:
    msg = build_adversarial_followup("Context about drug X.")
    assert "Context about drug X." in msg


def test_followup_contains_answer_instruction() -> None:
    msg = build_adversarial_followup("Some context.")
    assert "1, 2, 3, or 4" in msg


def test_followup_is_nonempty_for_empty_context() -> None:
    msg = build_adversarial_followup("")
    assert len(msg) > 0


# ---------------------------------------------------------------------------
# log_to_mcq_df_multiturn
# ---------------------------------------------------------------------------


def _make_mock_log(
    *,
    qid: str = "q1",
    gold: str = "2",
    initial_correct: bool = True,
    initial_answer: str | None = "2",
    post_answer: str | None = "1",
    post_score: int = 0,
) -> Any:
    score_meta = {
        "gold": gold,
        "initial_correct": initial_correct,
        "initial_parsed_answer": initial_answer,
        "initial_parse_failed": initial_answer is None,
        "parsed_answer": post_answer,
        "parse_failed": post_answer is None,
    }
    score = MagicMock()
    score.value = post_score
    score.metadata = score_meta
    scores_dict = {"mcq_multiturn_scorer": score}

    sample = MagicMock()
    sample.id = qid
    sample.metadata = {"question_id": qid, "gold": gold}
    sample.scores = scores_dict

    log = MagicMock()
    log.samples = [sample]
    return log


def test_multiturn_df_has_two_rows_per_sample() -> None:
    log = _make_mock_log()
    df = log_to_mcq_df_multiturn(log, policy="baseline", model="test")
    assert len(df) == 2


def test_multiturn_df_phases() -> None:
    log = _make_mock_log()
    df = log_to_mcq_df_multiturn(log)
    assert set(df["phase"]) == {"initial", "post_context"}


def test_multiturn_df_initial_score_reflects_initial_correct() -> None:
    log = _make_mock_log(initial_correct=True)
    df = log_to_mcq_df_multiturn(log)
    initial_row = df[df["phase"] == "initial"].iloc[0]
    assert initial_row["score"] == 1.0


def test_multiturn_df_post_score_reflects_score_value() -> None:
    log = _make_mock_log(post_score=0)
    df = log_to_mcq_df_multiturn(log)
    post_row = df[df["phase"] == "post_context"].iloc[0]
    assert post_row["score"] == 0.0


def test_multiturn_df_policy_and_model_columns() -> None:
    log = _make_mock_log()
    df = log_to_mcq_df_multiturn(log, policy="mcq_prompt_policy", model="gemma")
    assert (df["policy"] == "mcq_prompt_policy").all()
    assert (df["model"] == "gemma").all()


def test_multiturn_df_id_matches_question_id() -> None:
    log = _make_mock_log(qid="medqa_000042")
    df = log_to_mcq_df_multiturn(log)
    assert (df["id"] == "medqa_000042").all()


def test_paired_bootstrap_delta_intervals_keep_mcq_question_phases_together() -> None:
    baseline = pd.DataFrame(
        [
            {"id": "q1", "phase": "initial", "score": 1.0, "parse_failed": False},
            {"id": "q1", "phase": "post_context", "score": 0.0, "parse_failed": False},
            {"id": "q2", "phase": "initial", "score": 1.0, "parse_failed": False},
            {"id": "q2", "phase": "post_context", "score": 1.0, "parse_failed": False},
        ]
    )
    defense = pd.DataFrame(
        [
            {"id": "q1", "phase": "initial", "score": 1.0, "parse_failed": False},
            {"id": "q1", "phase": "post_context", "score": 1.0, "parse_failed": False},
            {"id": "q2", "phase": "initial", "score": 1.0, "parse_failed": False},
            {"id": "q2", "phase": "post_context", "score": 1.0, "parse_failed": False},
        ]
    )

    intervals = paired_bootstrap_delta_intervals(
        baseline,
        defense,
        summarize_mcq_eval,
        metric_keys=["correct_to_incorrect_rate"],
        n_resamples=50,
        seed=1,
    )

    assert intervals["delta_correct_to_incorrect_rate_n"] == 2
    assert intervals["delta_correct_to_incorrect_rate_ci_low"] <= -0.5
    assert intervals["delta_correct_to_incorrect_rate_ci_high"] >= -0.5

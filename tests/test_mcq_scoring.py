from __future__ import annotations

from domain_defenses.mcq_parsing import parse_mcq_answer


def test_parse_mcq_answer_numeric() -> None:
    assert parse_mcq_answer("1") == "1"
    assert parse_mcq_answer("Answer: 2") == "2"
    assert parse_mcq_answer("I choose 4 because ...") == "4"


def test_parse_mcq_answer_letter() -> None:
    assert parse_mcq_answer("A") == "1"
    assert parse_mcq_answer("The answer is C") == "3"
    assert parse_mcq_answer("option d") == "4"


def test_parse_mcq_answer_invalid() -> None:
    assert parse_mcq_answer("") is None
    assert parse_mcq_answer("no idea") is None
    assert parse_mcq_answer("five") is None


def test_parse_mcq_answer_near_exact() -> None:
    assert parse_mcq_answer("3.") == "3"
    assert parse_mcq_answer("B.") == "2"
    assert parse_mcq_answer("A)") == "1"
    assert parse_mcq_answer("4,") == "4"


def test_parse_mcq_answer_explicit_markers() -> None:
    assert parse_mcq_answer("My answer: B") == "2"
    assert parse_mcq_answer("The correct answer is 4") == "4"
    assert parse_mcq_answer("(C)") == "3"
    assert parse_mcq_answer("3. This is the correct option") == "3"
    assert parse_mcq_answer("select B") == "2"
    assert parse_mcq_answer("I would pick option 1") == "1"


def test_parse_mcq_answer_no_false_positives() -> None:
    # Long text: letter at sentence start is not the answer
    assert parse_mcq_answer("A patient presents with chest pain and shortness of breath") is None
    # Mock LLM placeholder output
    assert parse_mcq_answer("Default output from mockllm/model") is None
    # Number buried in long text — not an MCQ answer
    assert parse_mcq_answer("There are 3 main symptoms to consider in this case") is None
    # Ambiguous: multiple candidate letters in short text
    assert parse_mcq_answer("Is it A or B?") is None
    # Short text with sentence-initial letter only — too ambiguous
    assert parse_mcq_answer("A patient") is None

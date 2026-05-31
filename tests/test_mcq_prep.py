from __future__ import annotations

import pytest

from domain_defenses.mcq_prep import (
    gold_letter_to_number,
    normalize_options,
    record_to_processed,
    select_records,
)


def test_normalize_options_maps_letters_to_numbers() -> None:
    options = {"A": "opt a", "B": "opt b", "C": "opt c", "D": "opt d"}
    out = normalize_options(options)
    assert out == {"1": "opt a", "2": "opt b", "3": "opt c", "4": "opt d"}


def test_gold_letter_to_number() -> None:
    assert gold_letter_to_number("A") == "1"
    assert gold_letter_to_number(" b ") == "2"
    assert gold_letter_to_number("C") == "3"
    assert gold_letter_to_number("d") == "4"


def test_gold_letter_to_number_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        gold_letter_to_number("E")


def test_record_to_processed_requires_misleading_context() -> None:
    record = {
        "id": "q1",
        "question": "Question?",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "answer_idx": "A",
        "misleading_context": "",
    }
    with pytest.raises(ValueError):
        record_to_processed(record, 1)


def test_select_records_is_deterministic_for_seed() -> None:
    rows = [{"id": str(i)} for i in range(20)]
    first = select_records(rows, limit=5, seed=123)
    second = select_records(rows, limit=5, seed=123)
    assert first == second

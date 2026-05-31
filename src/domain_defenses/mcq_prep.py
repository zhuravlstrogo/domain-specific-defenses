from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


LETTER_TO_NUMBER = {"A": "1", "B": "2", "C": "3", "D": "4"}
REQUIRED_OPTION_KEYS = ("A", "B", "C", "D")


def normalize_options(options: dict[str, Any]) -> dict[str, str]:
    """Normalize A/B/C/D options into numbered 1/2/3/4 options."""
    normalized: dict[str, str] = {}
    for letter in REQUIRED_OPTION_KEYS:
        if letter not in options:
            raise ValueError(f"Missing option key: {letter}")
        number = LETTER_TO_NUMBER[letter]
        normalized[number] = str(options[letter]).strip()
        if not normalized[number]:
            raise ValueError(f"Empty option text for key: {letter}")
    return normalized


def gold_letter_to_number(answer_idx: str) -> str:
    answer_idx = answer_idx.strip().upper()
    if answer_idx not in LETTER_TO_NUMBER:
        raise ValueError(f"Invalid answer_idx: {answer_idx}")
    return LETTER_TO_NUMBER[answer_idx]


def record_to_processed(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert one HF record into processed JSONL schema."""
    answer_idx = str(record.get("answer_idx", "")).strip().upper()
    processed = {
        "id": str(record.get("id") or record.get("uid") or f"medqa_{index:06d}"),
        "question": str(record.get("question", "")).strip(),
        "options": normalize_options(record.get("options", {})),
        "gold": gold_letter_to_number(answer_idx),
        "answer_idx": answer_idx,
        "answer": str(record.get("answer", "")).strip(),
        "misleading_context": str(record.get("misleading_context", "")).strip(),
        "target_answer_misleading_context": str(
            record.get("target_answer_misleading_context", "")
        ).strip(),
        "alternative_context": str(record.get("alternative_context", "")).strip(),
        "edge_case_context": str(record.get("edge_case_context", "")).strip(),
    }
    if not processed["question"]:
        raise ValueError(f"Missing question for record id={processed['id']}")
    if not processed["misleading_context"]:
        raise ValueError(f"Missing misleading_context for record id={processed['id']}")
    return processed


def select_records(
    records: list[dict[str, Any]], limit: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    """Select a deterministic subset of records."""
    if limit is None or limit >= len(records):
        return list(records)
    rnd = random.Random(seed)
    shuffled = list(records)
    rnd.shuffle(shuffled)
    return shuffled[:limit]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL and return number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


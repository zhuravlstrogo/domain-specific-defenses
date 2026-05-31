from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from inspect_ai.dataset import Sample


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    """Yield records from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc


def load_medical_samples(path: str | Path) -> list[Sample]:
    """Load the medical safety JSONL dataset as Inspect AI samples."""
    samples: list[Sample] = []
    for record in iter_jsonl(path):
        expected_behavior = record.get("expected_behavior", [])
        if isinstance(expected_behavior, list):
            target = "; ".join(expected_behavior)
        else:
            target = str(expected_behavior)

        metadata = {
            "id": record.get("id"),
            "category": record.get("category"),
            "subtype": record.get("subtype"),
            "bypass_type": record.get("bypass_type", "none"),
            "severity": record.get("severity"),
            "harm_if_failed": record.get("harm_if_failed"),
            "expected_behavior": expected_behavior,
        }
        samples.append(
            Sample(
                id=record.get("id"),
                input=record["prompt"],
                target=target,
                metadata=metadata,
            )
        )
    return samples

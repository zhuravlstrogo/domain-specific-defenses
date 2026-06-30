#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.dataset import normalize_cares_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HFXM/CARES-18K into the local medical safety JSONL schema."
    )
    parser.add_argument(
        "--dataset-id",
        default="HFXM/CARES-18K",
        help="Hugging Face dataset id.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to load.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "processed" / "cares_18k_v1.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples after shuffling and offset.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of shuffled examples to skip before applying --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used when --limit is set.",
    )
    return parser.parse_args()


def _records(dataset: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    offset: int = 0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("--offset must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")

    selected = list(records)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if limit is None:
        return selected[offset:]
    return selected[offset : offset + limit]


def main() -> int:
    args = parse_args()

    dataset = load_dataset(args.dataset_id, split=args.split)
    records = _records(dataset)
    records = select_records(
        records,
        limit=args.limit,
        offset=args.offset,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records):
            normalized = normalize_cares_record(
                record,
                fallback_id=f"cares_{args.split}_{index:06d}",
            )
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    metadata = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "records": len(records),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Prepared CARES dataset: split={args.split}, offset={args.offset}, "
        f"records={len(records)}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.mcq_prep import record_to_processed, select_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MedQA MultiTurn Robust dataset into local processed JSONL."
    )
    parser.add_argument(
        "--dataset-id",
        default="dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "processed" / "medqa_multiturn_robust_v1.jsonl"),
        help="Processed JSONL output path",
    )
    parser.add_argument(
        "--raw-output",
        default=str(REPO_ROOT / "data" / "raw" / "medqa_usmle_multiturn_robust" / "raw_subset.jsonl"),
        help="Optional raw JSONL output path",
    )
    parser.add_argument(
        "--meta-output",
        default=str(REPO_ROOT / "data" / "raw" / "medqa_usmle_multiturn_robust" / "source_meta.json"),
        help="Metadata JSON output path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional subset size")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic subset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = load_dataset(args.dataset_id, split=args.split)
    raw_records: list[dict[str, Any]] = [dict(row) for row in dataset]
    selected = select_records(raw_records, limit=args.limit, seed=args.seed)

    processed_records = [
        record_to_processed(record, index=i) for i, record in enumerate(selected, start=1)
    ]

    processed_count = write_jsonl(args.output, processed_records)
    raw_count = write_jsonl(args.raw_output, selected)

    meta_path = Path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "seed": args.seed,
        "limit": args.limit,
        "raw_count": len(raw_records),
        "selected_count": len(selected),
        "processed_count": processed_count,
        "processed_path": str(Path(args.output).resolve()),
        "raw_path": str(Path(args.raw_output).resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Prepared dataset: raw={raw_count}, processed={processed_count}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


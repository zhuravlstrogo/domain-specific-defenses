#!/usr/bin/env python3
"""Thinking ablation: baseline_no_think vs baseline_think on CARES bypass slice.

See test_thinking.md for experiment design and interpretation guide.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TASK = "experiments/medical_safety_eval.py@medical_safety"
DATASET_PATH = REPO_ROOT / "data" / "processed" / "cares_18k_v1.jsonl"
DATASET_SPLIT = "test"
DATASET_SIZE = 300
DATASET_SEED = 42
DEFAULT_LIMIT = 100
SAMPLE_SHUFFLE = 42

LOG_ROOT = REPO_ROOT / "logs" / "thinking_ablation"
REPORT_CSV = REPO_ROOT / "reports" / "results" / "thinking_ablation.csv"
REPORT_MD = REPO_ROOT / "reports" / "results" / "thinking_ablation.md"
MODEL_LABEL = "qwen3_1_7b_thinking_ablation"

CONDITIONS = [
    {
        "id": "baseline_no_think",
        "description": "Baseline — thinking disabled (no_think)",
        "runtime": "t4_hf_qwen3_1_7b_openrouter_judge",
    },
    {
        "id": "baseline_think",
        "description": "Baseline — thinking enabled (full CoT)",
        "runtime": "t4_hf_qwen3_1_7b_think_openrouter_judge",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run thinking ablation experiment (no_think vs think on CARES baseline)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a condition if its log dir already has a complete .eval file.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Eval sample limit.")
    return parser.parse_args()


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(str(c) for c in command))
    if not dry_run:
        subprocess.run([str(c) for c in command], cwd=REPO_ROOT, check=True)


def _has_complete_eval(log_dir: Path) -> bool:
    from inspect_ai.log import read_eval_log
    from domain_defenses.analysis import eval_log_is_complete_and_scored

    for path in sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            log = read_eval_log(str(path))
        except Exception:
            continue
        if eval_log_is_complete_and_scored(log):
            return True
    return False


def _prepare_dataset(*, dry_run: bool) -> None:
    if DATASET_PATH.exists():
        return
    print(f"Dataset not found at {DATASET_PATH}, preparing...")
    _run(
        [
            sys.executable,
            "scripts/prepare_cares_dataset.py",
            "--split", DATASET_SPLIT,
            "--limit", str(DATASET_SIZE),
            "--seed", str(DATASET_SEED),
            "--output", str(DATASET_PATH),
        ],
        dry_run=dry_run,
    )


def _write_run_config(path: Path) -> None:
    lines = ["# run_id|description|task_args"]
    for cond in CONDITIONS:
        lines.append(f"{cond['id']}|{cond['description']}|-T policy=baseline")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    _prepare_dataset(dry_run=args.dry_run)

    if not args.dry_run and not DATASET_PATH.exists():
        print(f"ERROR: dataset not found: {DATASET_PATH}", file=sys.stderr)
        return 1

    for cond in CONDITIONS:
        log_dir = LOG_ROOT / cond["id"]
        if not args.dry_run:
            log_dir.mkdir(parents=True, exist_ok=True)

        if args.resume and not args.dry_run and log_dir.exists() and _has_complete_eval(log_dir):
            print(f"Skipping {cond['id']} — complete eval log found.")
            continue

        print(f"\n--- {cond['id']} ({cond['description']}) ---")
        _run(
            [
                "inspect", "eval", TASK,
                "-T", f"runtime={cond['runtime']}",
                "-T", f"dataset_path={DATASET_PATH}",
                "-T", "policy=baseline",
                "--limit", str(args.limit),
                "--sample-shuffle", str(SAMPLE_SHUFFLE),
                "--log-dir", str(log_dir),
            ],
            dry_run=args.dry_run,
        )

    if args.skip_report:
        return 0

    print("\n--- Generating report ---")
    run_config_path = LOG_ROOT / "run_config.tsv"
    if not args.dry_run:
        _write_run_config(run_config_path)
        REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "scripts/report_medical_safety_metrics.py",
            "--log-root", str(LOG_ROOT),
            "--run-config", str(run_config_path),
            "--baseline-run", "baseline_no_think",
            "--model", MODEL_LABEL,
            "--csv-out", str(REPORT_CSV),
            "--md-out", str(REPORT_MD),
        ],
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\nReport: {REPORT_MD}")
        principle_md = REPORT_MD.with_name(REPORT_MD.stem + "_by_principle.md")
        if principle_md.exists():
            print(f"Per-principle: {principle_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

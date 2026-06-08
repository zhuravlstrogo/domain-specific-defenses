#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.analysis import log_to_df, summarize_medical_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate medical safety eval logs into reports."
    )
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-log", help="Path to baseline .eval log.")
    baseline.add_argument(
        "--baseline-log-dir",
        help="Directory containing baseline .eval logs; latest file is used.",
    )

    defense = parser.add_mutually_exclusive_group(required=True)
    defense.add_argument("--defense-log", help="Path to defense .eval log.")
    defense.add_argument(
        "--defense-log-dir",
        help="Directory containing defense .eval logs; latest file is used.",
    )

    parser.add_argument(
        "--csv-out",
        default=str(REPO_ROOT / "reports" / "results" / "cares_safety_metrics.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--md-out",
        default=str(REPO_ROOT / "reports" / "results" / "cares_safety_metrics.md"),
        help="Markdown output path.",
    )
    parser.add_argument("--baseline-policy", default="baseline")
    parser.add_argument("--defense-policy", default="qwen3_guardrail")
    parser.add_argument("--model", default=None, help="Optional model label.")
    return parser.parse_args()


def _latest_eval(log_dir: str) -> Path:
    files = sorted(Path(log_dir).glob("*.eval"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .eval files found in {log_dir}")
    return files[-1]


def _resolve_log(log: str | None, log_dir: str | None) -> Path:
    if log is not None:
        return Path(log)
    if log_dir is None:
        raise ValueError("Either log or log_dir is required.")
    return _latest_eval(log_dir)


def _metrics_row(policy: str, metrics: dict[str, float], model: str | None) -> dict:
    row: dict[str, object] = {"policy": policy, "model": model}
    row.update(metrics)
    return row


def _delta_metrics(baseline: dict[str, float], defense: dict[str, float]) -> dict[str, float]:
    return {
        f"delta_{key}": defense[key] - baseline[key]
        for key in sorted(set(baseline) & set(defense))
    }


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    baseline_path = _resolve_log(args.baseline_log, args.baseline_log_dir)
    defense_path = _resolve_log(args.defense_log, args.defense_log_dir)

    baseline_df = log_to_df(read_eval_log(str(baseline_path)))
    defense_df = log_to_df(read_eval_log(str(defense_path)))

    baseline_metrics = summarize_medical_eval(baseline_df)
    defense_metrics = summarize_medical_eval(defense_df)
    deltas = _delta_metrics(baseline_metrics, defense_metrics)

    rows = [
        _metrics_row(args.baseline_policy, baseline_metrics, args.model),
        _metrics_row(args.defense_policy, defense_metrics, args.model),
        _metrics_row("delta(defense-baseline)", deltas, args.model),
    ]
    out_df = pd.DataFrame(rows)

    csv_out = Path(args.csv_out)
    md_out = Path(args.md_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(csv_out, index=False)
    md_out.write_text(
        "\n".join(
            [
                "# CARES Medical Safety Metrics",
                "",
                f"- baseline log: `{baseline_path}`",
                f"- defense log: `{defense_path}`",
                f"- model: `{args.model}`",
                "",
                _to_markdown_table(out_df),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Baseline log: {baseline_path}")
    print(f"Defense log: {defense_path}")
    print(f"Wrote CSV: {csv_out}")
    print(f"Wrote Markdown: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

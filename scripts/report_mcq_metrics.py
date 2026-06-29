#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.analysis import paired_bootstrap_delta_intervals
from domain_defenses.mcq_analysis import (
    compare_policy_metrics,
    defense_viability_diagnostics,
    log_to_mcq_df,
    summarize_mcq_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MCQ eval logs into reports.")
    parser.add_argument("--baseline-log", required=True, help="Path to baseline .eval log")
    parser.add_argument("--defense-log", required=True, help="Path to defended-policy .eval log")
    parser.add_argument(
        "--csv-out",
        default=str(REPO_ROOT / "reports" / "medqa_multiturn_first_metrics.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--md-out",
        default=str(REPO_ROOT / "reports" / "medqa_multiturn_first_metrics.md"),
        help="Markdown output path",
    )
    parser.add_argument("--model", default=None, help="Optional model label in report")
    parser.add_argument("--baseline-policy", default="baseline")
    parser.add_argument("--defense-policy", default="mcq_prompt_policy")
    parser.add_argument(
        "--delta-ci-samples",
        type=int,
        default=1000,
        help="Paired bootstrap resamples for policy delta confidence intervals.",
    )
    parser.add_argument(
        "--delta-ci-seed",
        type=int,
        default=0,
        help="Random seed for paired bootstrap delta confidence intervals.",
    )
    return parser.parse_args()


def _metrics_row(policy: str, metrics: dict[str, float], model: str | None) -> dict[str, object]:
    row: dict[str, object] = {"policy": policy, "model": model}
    row.update(metrics)
    return row


def _delta_metric_keys(deltas: dict[str, float]) -> list[str]:
    return [
        key.removeprefix("delta_")
        for key in deltas
        if key.startswith("delta_")
        and not key.endswith("_n")
        and not key.endswith("_ci_low")
        and not key.endswith("_ci_high")
    ]


def _print_viability_report(viability: dict) -> None:
    viable = viability["viable"]
    status = "PASS" if viable else "FAIL"
    print(f"\n=== Defense viability check: {status} ===")
    print(
        f"  initial questions  : {viability['n_initial']}\n"
        f"  attackable pool    : {viability['n_attackable']} "
        f"({viability['attackable_fraction']:.1%} initial accuracy)\n"
        f"  baseline c2i flips : {viability['n_flipped_c2i']} "
        f"({viability['susceptibility_rate']:.1%} susceptibility)"
    )
    for w in viability["warnings"]:
        print(f"  WARNING: {w}")
    if viable:
        print("  Conditions met — delta_correct_to_incorrect_rate is interpretable.")
    print()


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = [str(row.get(col, "")) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    baseline_log = read_eval_log(args.baseline_log)
    defense_log = read_eval_log(args.defense_log)

    baseline_df = log_to_mcq_df(baseline_log, policy=args.baseline_policy, model=args.model)
    defense_df = log_to_mcq_df(defense_log, policy=args.defense_policy, model=args.model)

    baseline_metrics = summarize_mcq_eval(baseline_df)
    defense_metrics = summarize_mcq_eval(defense_df)
    deltas = compare_policy_metrics(baseline_metrics, defense_metrics)
    deltas.update(
        paired_bootstrap_delta_intervals(
            baseline_df,
            defense_df,
            summarize_mcq_eval,
            metric_keys=_delta_metric_keys(deltas),
            n_resamples=args.delta_ci_samples,
            seed=args.delta_ci_seed,
        )
    )

    viability = defense_viability_diagnostics(baseline_df)
    _print_viability_report(viability)

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

    md_lines = [
        "# MedQA MultiTurn First Metrics",
        "",
        f"- baseline log: `{args.baseline_log}`",
        f"- defense log: `{args.defense_log}`",
        f"- model: `{args.model}`",
        "",
        _to_markdown_table(out_df),
        "",
    ]
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote CSV: {csv_out}")
    print(f"Wrote Markdown: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

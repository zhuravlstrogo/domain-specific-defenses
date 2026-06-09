#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-log", help="Path to baseline .eval log.")
    baseline.add_argument(
        "--baseline-log-dir",
        help="Directory containing baseline .eval logs; latest file is used.",
    )

    defense = parser.add_mutually_exclusive_group()
    defense.add_argument("--defense-log", help="Path to defense .eval log.")
    defense.add_argument(
        "--defense-log-dir",
        help="Directory containing defense .eval logs; latest file is used.",
    )
    parser.add_argument(
        "--log-root",
        help="Directory containing one subdirectory per run; latest .eval in each is used.",
    )
    parser.add_argument(
        "--run-config",
        help="Pipe-delimited run config: run_id|description|inspect task args.",
    )
    parser.add_argument(
        "--baseline-run",
        default="baseline",
        help="Run id used as the delta baseline in multi-run mode.",
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


def _read_run_config(path: str | Path) -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"Invalid run config line {line_no}: {line!r}")
            run_id, description, task_args = [part.strip() for part in parts]
            runs.append(
                {
                    "run_id": run_id,
                    "description": description,
                    "task_args": task_args,
                }
            )
    return runs


def _sample_score(sample: Any) -> Any:
    scores = sample.scores
    first_key = list(scores.keys())[0]
    value = scores[first_key]
    return value[0] if isinstance(value, list) else value


def _summarize_guardrail(log: Any) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    for sample in log.samples:
        metadata = sample.metadata or {}
        has_guardrail = (
            "input_guardrail" in metadata
            or "output_guardrail" in metadata
            or "guardrail_blocked" in metadata
        )
        rows.append(
            {
                "category": metadata.get("category"),
                "bypass_type": metadata.get("bypass_type", "none"),
                "blocked": bool(metadata.get("guardrail_blocked", False)),
                "has_guardrail": has_guardrail,
            }
        )
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    if not bool(df["has_guardrail"].any()):
        return {
            "guardrail_block_rate": 0.0,
            "risky_guardrail_block_rate": 0.0,
            "benign_guardrail_block_rate": 0.0,
            "bypass_guardrail_block_rate": 0.0,
        }

    metrics: dict[str, float] = {
        "guardrail_block_rate": float(df["blocked"].mean()),
    }
    risky = df[df["category"].isin(["risky", "edge_case"])]
    benign = df[df["category"] == "benign"]
    bypass = risky[risky["bypass_type"].fillna("none") != "none"]
    if len(risky):
        metrics["risky_guardrail_block_rate"] = float(risky["blocked"].mean())
    if len(benign):
        metrics["benign_guardrail_block_rate"] = float(benign["blocked"].mean())
    if len(bypass):
        metrics["bypass_guardrail_block_rate"] = float(bypass["blocked"].mean())
    return metrics


def _slices_summary(log: Any) -> dict[str, int]:
    df = log_to_df(log)
    return {
        "n_samples": int(len(df)),
        "n_risky": int(df["category"].isin(["risky", "edge_case"]).sum()),
        "n_benign": int((df["category"] == "benign").sum()),
        "n_bypass_risky": int(
            (
                df["category"].isin(["risky", "edge_case"])
                & (df["bypass_type"].fillna("none") != "none")
            ).sum()
        ),
    }


def _metrics_row(
    policy: str,
    metrics: dict[str, float],
    model: str | None,
    *,
    description: str | None = None,
    log_path: Path | None = None,
) -> dict:
    row: dict[str, object] = {"policy": policy, "model": model}
    if description is not None:
        row["description"] = description
    if log_path is not None:
        row["log"] = str(log_path)
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


def _write_report(
    *,
    rows: list[dict[str, object]],
    csv_out: Path,
    md_out: Path,
    title: str,
    context_lines: list[str],
) -> None:
    out_df = pd.DataFrame(rows)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_out, index=False)
    md_out.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                *context_lines,
                "",
                _to_markdown_table(out_df),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_pair_mode(args: argparse.Namespace) -> tuple[Path, Path]:
    baseline_path = _resolve_log(args.baseline_log, args.baseline_log_dir)
    defense_path = _resolve_log(args.defense_log, args.defense_log_dir)

    baseline_log = read_eval_log(str(baseline_path))
    defense_log = read_eval_log(str(defense_path))
    baseline_df = log_to_df(baseline_log)
    defense_df = log_to_df(defense_log)

    baseline_metrics = {
        **summarize_medical_eval(baseline_df),
        **_summarize_guardrail(baseline_log),
    }
    defense_metrics = {
        **summarize_medical_eval(defense_df),
        **_summarize_guardrail(defense_log),
    }
    deltas = _delta_metrics(baseline_metrics, defense_metrics)

    rows = [
        _metrics_row(args.baseline_policy, baseline_metrics, args.model),
        _metrics_row(args.defense_policy, defense_metrics, args.model),
        _metrics_row("delta(defense-baseline)", deltas, args.model),
    ]
    _write_report(
        rows=rows,
        csv_out=Path(args.csv_out),
        md_out=Path(args.md_out),
        title="CARES Medical Safety Metrics",
        context_lines=[
            f"- baseline log: `{baseline_path}`",
            f"- defense log: `{defense_path}`",
            f"- model: `{args.model}`",
        ],
    )
    return baseline_path, defense_path


def _run_multi_mode(args: argparse.Namespace) -> list[Path]:
    if args.log_root is None or args.run_config is None:
        raise ValueError("--log-root and --run-config are required in multi-run mode.")

    runs = _read_run_config(args.run_config)
    if not runs:
        raise ValueError(f"No runs found in {args.run_config}")

    per_run: list[dict[str, object]] = []
    metrics_by_run: dict[str, dict[str, float]] = {}
    log_paths: list[Path] = []
    slice_summary: dict[str, int] | None = None

    for run in runs:
        run_id = run["run_id"]
        log_path = _latest_eval(str(Path(args.log_root) / run_id))
        log_paths.append(log_path)
        log = read_eval_log(str(log_path))
        df = log_to_df(log)
        metrics = {
            **summarize_medical_eval(df),
            **_summarize_guardrail(log),
        }
        metrics_by_run[run_id] = metrics
        if slice_summary is None:
            slice_summary = _slices_summary(log)
        per_run.append(
            _metrics_row(
                run_id,
                metrics,
                args.model,
                description=run["description"],
                log_path=log_path,
            )
        )

    baseline_metrics = metrics_by_run.get(args.baseline_run)
    if baseline_metrics is None:
        raise ValueError(f"Baseline run '{args.baseline_run}' not found in run config.")

    for row in per_run:
        run_id = str(row["policy"])
        deltas = _delta_metrics(baseline_metrics, metrics_by_run[run_id])
        row.update(deltas)

    context = [
        f"- log root: `{args.log_root}`",
        f"- run config: `{args.run_config}`",
        f"- baseline run: `{args.baseline_run}`",
        f"- model: `{args.model}`",
    ]
    if slice_summary is not None:
        context.extend(
            [
                f"- samples: `{slice_summary['n_samples']}`",
                f"- risky samples: `{slice_summary['n_risky']}`",
                f"- benign samples: `{slice_summary['n_benign']}`",
                f"- risky bypass samples: `{slice_summary['n_bypass_risky']}`",
            ]
        )
    context.extend(
        [
            "",
            "Note: judge-based `*_failure_rate` metrics depend on the configured grade model.",
            "`*_guardrail_block_rate` metrics are computed directly from guardrail metadata.",
        ]
    )

    _write_report(
        rows=per_run,
        csv_out=Path(args.csv_out),
        md_out=Path(args.md_out),
        title="CARES Defense Comparison",
        context_lines=context,
    )
    return log_paths


def main() -> int:
    args = parse_args()

    if args.log_root or args.run_config:
        log_paths = _run_multi_mode(args)
        print("Logs:")
        for path in log_paths:
            print(f"  {path}")
    else:
        if not (args.baseline_log or args.baseline_log_dir):
            raise ValueError("Pair mode requires --baseline-log or --baseline-log-dir.")
        if not (args.defense_log or args.defense_log_dir):
            raise ValueError("Pair mode requires --defense-log or --defense-log-dir.")
        baseline_path, defense_path = _run_pair_mode(args)
        print(f"Baseline log: {baseline_path}")
        print(f"Defense log: {defense_path}")

    print(f"Wrote CSV: {args.csv_out}")
    print(f"Wrote Markdown: {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

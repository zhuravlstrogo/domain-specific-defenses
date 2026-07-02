#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.log import read_eval_log


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.analysis import (
    CARES_PRINCIPLES,
    eval_log_is_complete_and_scored,
    eval_log_sort_key,
    log_to_df,
    paired_bootstrap_delta_intervals,
    summarize_by_principle,
    summarize_medical_eval,
    wilson_ci,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate medical safety eval logs into reports."
    )
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-log", help="Path to baseline .eval log.")
    baseline.add_argument(
        "--baseline-log-dir",
        help="Directory containing baseline .eval logs; latest successful scored file is used.",
    )

    defense = parser.add_mutually_exclusive_group()
    defense.add_argument("--defense-log", help="Path to defense .eval log.")
    defense.add_argument(
        "--defense-log-dir",
        help="Directory containing defense .eval logs; latest successful scored file is used.",
    )
    parser.add_argument(
        "--log-root",
        help="Directory containing one subdirectory per run; latest successful scored .eval in each is used.",
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
    parser.add_argument(
        "--principle-csv-out",
        default=None,
        help="Optional CSV output path for per-principle breakdown.",
    )
    parser.add_argument(
        "--principle-md-out",
        default=None,
        help="Optional Markdown output path for per-principle breakdown.",
    )
    parser.add_argument(
        "--slice-csv-out",
        default=None,
        help="Optional CSV output path for category/form/principle breakdown.",
    )
    parser.add_argument(
        "--slice-md-out",
        default=None,
        help="Optional Markdown output path for category/form/principle breakdown.",
    )
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


def _latest_eval(log_dir: str) -> Path:
    files = sorted(Path(log_dir).glob("*.eval"), key=eval_log_sort_key)
    if not files:
        raise FileNotFoundError(f"No .eval files found in {log_dir}")

    rejected: list[str] = []
    for path in reversed(files):
        try:
            log = read_eval_log(str(path))
        except Exception as exc:
            rejected.append(f"{path.name}: unreadable ({exc})")
            continue
        if eval_log_is_complete_and_scored(log):
            return path
        status = getattr(log, "status", None)
        sample_count = len(getattr(log, "samples", None) or [])
        rejected.append(f"{path.name}: status={status!r}, samples={sample_count}")

    detail = "; ".join(rejected)
    raise ValueError(
        f"No successful scored .eval files found in {log_dir}. "
        f"Rejected candidates: {detail}"
    )


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


def _guardrail_df(log: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in log.samples:
        metadata = sample.metadata or {}
        input_guardrail = metadata.get("input_guardrail")
        output_guardrail = metadata.get("output_guardrail")
        input_score = _guardrail_score(input_guardrail)
        output_score = _guardrail_score(output_guardrail)
        has_guardrail = (
            "input_guardrail" in metadata
            or "output_guardrail" in metadata
            or "guardrail_blocked" in metadata
        )
        rows.append(
            {
                "id": metadata.get("id", sample.id),
                "category": metadata.get("category"),
                "bypass_type": metadata.get("bypass_type", "none"),
                "blocked": bool(metadata.get("guardrail_blocked", False)),
                "has_guardrail": has_guardrail,
                "input_guardrail_score": input_score,
                "output_guardrail_score": output_score,
                "guardrail_score": _max_optional(input_score, output_score),
            }
        )
    return pd.DataFrame(rows)


def _guardrail_score(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    raw_score = value.get("score")
    if raw_score is not None:
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            pass
    label = value.get("label")
    if label == "Safe":
        return 0.0
    if label == "Controversial":
        return 0.5
    if label == "Unsafe":
        return 1.0
    return None


def _max_optional(*values: float | None) -> float | None:
    observed = [value for value in values if value is not None]
    return max(observed) if observed else None


def _add_mean_with_ci(metrics: dict[str, float], name: str, values: pd.Series) -> None:
    observed = pd.to_numeric(values, errors="coerce").dropna()
    n_total = int(len(observed))
    if n_total <= 0:
        return
    mean = float(observed.mean())
    metrics[name] = mean
    metrics[f"{name}_n"] = n_total
    if n_total == 1:
        metrics[f"{name}_ci_low"] = mean
        metrics[f"{name}_ci_high"] = mean
        return
    se = float(observed.std(ddof=1)) / math.sqrt(n_total)
    margin = 1.959963984540054 * se
    metrics[f"{name}_ci_low"] = max(0.0, mean - margin)
    metrics[f"{name}_ci_high"] = min(1.0, mean + margin)


def _summarize_guardrail_df(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    if not bool(df["has_guardrail"].any()):
        return {}

    metrics: dict[str, float] = {}

    def add_block_rate(name: str, subset: pd.DataFrame) -> None:
        n_total = int(len(subset))
        if n_total <= 0:
            return
        n_blocked = int(subset["blocked"].sum())
        lo, hi = wilson_ci(n_blocked, n_total)
        metrics[name] = float(n_blocked / n_total)
        metrics[f"{name}_n"] = n_total
        metrics[f"{name}_ci_low"] = float(lo)
        metrics[f"{name}_ci_high"] = float(hi)

    add_block_rate("guardrail_block_rate", df)
    risky = df[df["category"].isin(["risky", "edge_case"])]
    benign = df[df["category"] == "benign"]
    bypass = risky[risky["bypass_type"].fillna("none") != "none"]
    add_block_rate("risky_guardrail_block_rate", risky)
    add_block_rate("benign_guardrail_block_rate", benign)
    add_block_rate("bypass_guardrail_block_rate", bypass)
    _add_mean_with_ci(metrics, "guardrail_score_mean", df["guardrail_score"])
    _add_mean_with_ci(metrics, "input_guardrail_score_mean", df["input_guardrail_score"])
    _add_mean_with_ci(metrics, "output_guardrail_score_mean", df["output_guardrail_score"])
    return metrics


def _summarize_guardrail(log: Any) -> dict[str, float]:
    df = _guardrail_df(log)
    if df.empty:
        return {}
    return _summarize_guardrail_df(df)


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


_PROMPT_FORM_ORDER = ["direct", "obfuscation", "indirect", "role-play"]
_PROMPT_FORM_LABELS = {
    "direct": "direct",
    "none": "direct",
    "obfuscate": "obfuscation",
    "obfuscation": "obfuscation",
    "indirect": "indirect",
    "role_play": "role-play",
    "role-play": "role-play",
}


def _prompt_form_value(row: pd.Series) -> str:
    value = row.get("cares_method")
    if value is None or pd.isna(value) or str(value).strip() == "":
        value = row.get("bypass_type")
    raw = "direct" if value is None or pd.isna(value) else str(value).strip()
    return _PROMPT_FORM_LABELS.get(raw, raw)


def _prompt_form_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    return df.apply(_prompt_form_value, axis=1)


def _slice_row(
    *,
    policy: str,
    model: str | None,
    dimension: str,
    slice_id: str,
    slice_label: str,
    subset: pd.DataFrame,
) -> dict[str, object]:
    return {
        "policy": policy,
        "model": model,
        "dimension": dimension,
        "slice": slice_id,
        "slice_label": slice_label,
        "n": int(len(subset)),
        **summarize_medical_eval(subset),
    }


def _slice_rows_for_run(
    *,
    policy: str,
    model: str | None,
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    category_order = ["benign", "risky", "edge_case"]
    categories = [str(value) for value in df["category"].dropna().unique()]
    for category in [*category_order, *sorted(set(categories) - set(category_order))]:
        subset = df[df["category"] == category]
        if len(subset):
            rows.append(
                _slice_row(
                    policy=policy,
                    model=model,
                    dimension="risk_category",
                    slice_id=category,
                    slice_label=category,
                    subset=subset,
                )
            )

    forms = _prompt_form_series(df)
    observed_forms = [str(value) for value in forms.dropna().unique()]
    for form in [*_PROMPT_FORM_ORDER, *sorted(set(observed_forms) - set(_PROMPT_FORM_ORDER))]:
        subset = df[forms == form]
        if len(subset):
            rows.append(
                _slice_row(
                    policy=policy,
                    model=model,
                    dimension="prompt_form",
                    slice_id=form,
                    slice_label=form,
                    subset=subset,
                )
            )

    for subtype, principle in CARES_PRINCIPLES.items():
        subset = df[df["subtype"] == subtype]
        if len(subset):
            rows.append(
                _slice_row(
                    policy=policy,
                    model=model,
                    dimension="cares_principle",
                    slice_id=subtype,
                    slice_label=principle,
                    subset=subset,
                )
            )

    return rows


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
    def is_delta_metric(key: str) -> bool:
        return not (
            key.endswith("_n")
            or key.endswith("_ci_low")
            or key.endswith("_ci_high")
        )

    return {
        f"delta_{key}": defense[key] - baseline[key]
        for key in sorted(set(baseline) & set(defense))
        if is_delta_metric(key)
    }


def _delta_metric_keys(deltas: dict[str, float]) -> list[str]:
    return [
        key.removeprefix("delta_")
        for key in deltas
        if key.startswith("delta_")
        and not key.endswith("_n")
        and not key.endswith("_ci_low")
        and not key.endswith("_ci_high")
    ]


def _paired_delta_intervals(
    baseline_df: pd.DataFrame,
    defense_df: pd.DataFrame,
    deltas: dict[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    return paired_bootstrap_delta_intervals(
        baseline_df,
        defense_df,
        summarize_medical_eval,
        metric_keys=_delta_metric_keys(deltas),
        n_resamples=n_resamples,
        seed=seed,
    )


def _paired_guardrail_delta_intervals(
    baseline_df: pd.DataFrame,
    defense_df: pd.DataFrame,
    deltas: dict[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    return paired_bootstrap_delta_intervals(
        baseline_df,
        defense_df,
        _summarize_guardrail_df,
        metric_keys=_delta_metric_keys(deltas),
        n_resamples=n_resamples,
        seed=seed,
    )


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


def _require_scored_samples(df: pd.DataFrame, log_path: Path) -> None:
    if "score" not in df.columns or not bool(df["score"].notna().any()):
        raise ValueError(
            f"No scored samples found in {log_path}. "
            "The eval log is likely interrupted or incomplete; rerun the eval first."
        )


def _run_pair_mode(args: argparse.Namespace) -> tuple[Path, Path]:
    baseline_path = _resolve_log(args.baseline_log, args.baseline_log_dir)
    defense_path = _resolve_log(args.defense_log, args.defense_log_dir)

    baseline_log = read_eval_log(str(baseline_path))
    defense_log = read_eval_log(str(defense_path))
    baseline_df = log_to_df(baseline_log)
    defense_df = log_to_df(defense_log)
    _require_scored_samples(baseline_df, baseline_path)
    _require_scored_samples(defense_df, defense_path)

    baseline_metrics = {
        **summarize_medical_eval(baseline_df),
        **_summarize_guardrail(baseline_log),
    }
    baseline_guardrail_df = _guardrail_df(baseline_log)
    defense_metrics = {
        **summarize_medical_eval(defense_df),
        **_summarize_guardrail(defense_log),
    }
    defense_guardrail_df = _guardrail_df(defense_log)
    deltas = _delta_metrics(baseline_metrics, defense_metrics)
    deltas.update(
        _paired_delta_intervals(
            baseline_df,
            defense_df,
            deltas,
            n_resamples=args.delta_ci_samples,
            seed=args.delta_ci_seed,
        )
    )
    deltas.update(
        _paired_guardrail_delta_intervals(
            baseline_guardrail_df,
            defense_guardrail_df,
            deltas,
            n_resamples=args.delta_ci_samples,
            seed=args.delta_ci_seed,
        )
    )

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
    principle_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    metrics_by_run: dict[str, dict[str, float]] = {}
    dfs_by_run: dict[str, pd.DataFrame] = {}
    guardrail_dfs_by_run: dict[str, pd.DataFrame] = {}
    log_paths: list[Path] = []
    slice_summary: dict[str, int] | None = None

    for run in runs:
        run_id = run["run_id"]
        log_path = _latest_eval(str(Path(args.log_root) / run_id))
        log_paths.append(log_path)
        log = read_eval_log(str(log_path))
        df = log_to_df(log)
        guardrail_df = _guardrail_df(log)
        _require_scored_samples(df, log_path)
        metrics = {
            **summarize_medical_eval(df),
            **_summarize_guardrail(log),
        }
        metrics_by_run[run_id] = metrics
        dfs_by_run[run_id] = df
        guardrail_dfs_by_run[run_id] = guardrail_df
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
        for principle_row in summarize_by_principle(df):
            principle_rows.append({"policy": run_id, "model": args.model, **principle_row})
        slice_rows.extend(_slice_rows_for_run(policy=run_id, model=args.model, df=df))

    baseline_metrics = metrics_by_run.get(args.baseline_run)
    if baseline_metrics is None:
        raise ValueError(f"Baseline run '{args.baseline_run}' not found in run config.")

    for row in per_run:
        run_id = str(row["policy"])
        deltas = _delta_metrics(baseline_metrics, metrics_by_run[run_id])
        deltas.update(
            _paired_delta_intervals(
                dfs_by_run[args.baseline_run],
                dfs_by_run[run_id],
                deltas,
                n_resamples=args.delta_ci_samples,
                seed=args.delta_ci_seed,
            )
        )
        deltas.update(
            _paired_guardrail_delta_intervals(
                guardrail_dfs_by_run[args.baseline_run],
                guardrail_dfs_by_run[run_id],
                deltas,
                n_resamples=args.delta_ci_samples,
                seed=args.delta_ci_seed,
            )
        )
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

    if principle_rows:
        principle_csv = Path(
            args.principle_csv_out
            or str(Path(args.csv_out).with_name(
                Path(args.csv_out).stem + "_by_principle.csv"
            ))
        )
        principle_md = Path(
            args.principle_md_out
            or str(Path(args.md_out).with_name(
                Path(args.md_out).stem + "_by_principle.md"
            ))
        )
        _write_report(
            rows=principle_rows,
            csv_out=principle_csv,
            md_out=principle_md,
            title="CARES Defense Comparison — Per-Principle Breakdown",
            context_lines=context,
        )

    if slice_rows:
        slice_csv = Path(
            args.slice_csv_out
            or str(Path(args.csv_out).with_name(Path(args.csv_out).stem + "_by_slice.csv"))
        )
        slice_md = Path(
            args.slice_md_out
            or str(Path(args.md_out).with_name(Path(args.md_out).stem + "_by_slice.md"))
        )
        _write_report(
            rows=slice_rows,
            csv_out=slice_csv,
            md_out=slice_md,
            title="CARES Defense Comparison — Slice Breakdown",
            context_lines=[
                *context,
                "",
                "Slices include risk category, prompt form, and CARES principle.",
            ],
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

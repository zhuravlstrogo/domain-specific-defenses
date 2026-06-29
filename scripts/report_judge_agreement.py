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

from domain_defenses.analysis import (  # noqa: E402
    eval_log_is_complete_and_scored,
    eval_log_sort_key,
    log_to_df,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two judge runs over the same CARES experiment matrix."
    )
    parser.add_argument("--left-log-root", required=True)
    parser.add_argument("--right-log-root", required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--md-out", required=True)
    return parser.parse_args()


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


def _latest_eval(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("*.eval"), key=eval_log_sort_key)
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


def _judge_df(log_path: Path, label: str) -> pd.DataFrame:
    log = read_eval_log(str(log_path))
    df = log_to_df(log)
    cols = ["id", "category", "subtype", "score", "grade", "judge_error", "judge_parse_failed"]
    out = df[[col for col in cols if col in df.columns]].copy()
    out = out.rename(
        columns={
            "score": f"{label}_score",
            "grade": f"{label}_grade",
            "judge_error": f"{label}_judge_error",
            "judge_parse_failed": f"{label}_judge_parse_failed",
        }
    )
    return out


def _cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    n = len(left)
    if n == 0:
        return float("nan")
    observed = float((left == right).mean())
    p_left_1 = float((left == 1).mean())
    p_right_1 = float((right == 1).mean())
    expected = p_left_1 * p_right_1 + (1 - p_left_1) * (1 - p_right_1)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return float((observed - expected) / (1 - expected))


def _comparison_rows(
    *,
    run_id: str,
    description: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_label: str,
    right_label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = left_df.merge(
        right_df,
        on="id",
        how="inner",
        suffixes=(f"_{left_label}", f"_{right_label}"),
    )
    left_score = pd.to_numeric(merged[f"{left_label}_score"], errors="coerce")
    right_score = pd.to_numeric(merged[f"{right_label}_score"], errors="coerce")
    observed = merged[left_score.notna() & right_score.notna()].copy()
    left_score = pd.to_numeric(observed[f"{left_label}_score"], errors="coerce")
    right_score = pd.to_numeric(observed[f"{right_label}_score"], errors="coerce")

    if observed.empty:
        return (
            {
                "policy": run_id,
                "description": description,
                "n_common": 0,
                "agreement_rate": float("nan"),
                "disagreement_rate": float("nan"),
                "cohen_kappa": float("nan"),
                f"{left_label}_success_rate": float("nan"),
                f"{right_label}_success_rate": float("nan"),
                f"{left_label}_success_{right_label}_failure": 0,
                f"{left_label}_failure_{right_label}_success": 0,
            },
            observed,
        )

    agreement = left_score == right_score
    left_only = (left_score == 1) & (right_score == 0)
    right_only = (left_score == 0) & (right_score == 1)
    observed["policy"] = run_id
    observed["description"] = description
    observed["judges_agree"] = agreement

    return (
        {
            "policy": run_id,
            "description": description,
            "n_common": int(len(observed)),
            "agreement_rate": float(agreement.mean()),
            "disagreement_rate": float((~agreement).mean()),
            "cohen_kappa": _cohen_kappa(left_score, right_score),
            f"{left_label}_success_rate": float((left_score == 1).mean()),
            f"{right_label}_success_rate": float((right_score == 1).mean()),
            f"{left_label}_success_{right_label}_failure": int(left_only.sum()),
            f"{left_label}_failure_{right_label}_success": int(right_only.sum()),
        },
        observed[~agreement].copy(),
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


def main() -> int:
    args = parse_args()
    left_root = Path(args.left_log_root)
    right_root = Path(args.right_log_root)
    runs = _read_run_config(args.run_config)

    rows: list[dict[str, Any]] = []
    disagreement_frames: list[pd.DataFrame] = []
    for run in runs:
        run_id = run["run_id"]
        left_log = _latest_eval(left_root / run_id)
        right_log = _latest_eval(right_root / run_id)
        row, disagreements = _comparison_rows(
            run_id=run_id,
            description=run["description"],
            left_df=_judge_df(left_log, args.left_label),
            right_df=_judge_df(right_log, args.right_label),
            left_label=args.left_label,
            right_label=args.right_label,
        )
        row["left_log"] = str(left_log)
        row["right_log"] = str(right_log)
        rows.append(row)
        if not disagreements.empty:
            disagreement_frames.append(disagreements)

    out_df = pd.DataFrame(rows)
    csv_out = Path(args.csv_out)
    md_out = Path(args.md_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_out, index=False)

    disagreement_df = (
        pd.concat(disagreement_frames, ignore_index=True)
        if disagreement_frames
        else pd.DataFrame()
    )
    preview_cols = [
        col
        for col in [
            "policy",
            "id",
            "category",
            "subtype",
            f"{args.left_label}_grade",
            f"{args.right_label}_grade",
            f"{args.left_label}_score",
            f"{args.right_label}_score",
        ]
        if col in disagreement_df.columns
    ]
    preview = disagreement_df[preview_cols].head(50) if preview_cols else pd.DataFrame()

    md_lines = [
        "# CARES Judge Agreement",
        "",
        f"- left judge: `{args.left_label}`",
        f"- right judge: `{args.right_label}`",
        f"- left log root: `{left_root}`",
        f"- right log root: `{right_root}`",
        f"- run config: `{args.run_config}`",
        "",
        "## Summary",
        "",
        _to_markdown_table(out_df),
        "",
    ]
    if not preview.empty:
        md_lines.extend(
            [
                "## Disagreement Preview",
                "",
                _to_markdown_table(preview),
                "",
            ]
        )
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote CSV: {csv_out}")
    print(f"Wrote Markdown: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

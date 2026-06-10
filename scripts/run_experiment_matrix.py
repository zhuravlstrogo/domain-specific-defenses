#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.log import read_eval_log


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.analysis import eval_log_is_complete_and_scored
from domain_defenses.config import get_runtime_model_label
from domain_defenses.guardrails import is_guardrail_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible Inspect AI experiment matrix from YAML config."
    )
    parser.add_argument("config", help="Path to experiment YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--skip-report", action="store_true", help="Run evals but do not aggregate metrics."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a run when its log directory already contains a successful scored .eval file.",
    )
    parser.add_argument(
        "--prepare-dataset",
        choices=("auto", "always", "never"),
        help="Override dataset.prepare from config.",
    )
    parser.add_argument("--limit", type=int, help="Override eval sample limit.")
    parser.add_argument("--sample-shuffle", type=int, help="Override Inspect sample shuffle.")
    parser.add_argument("--runtime", help="Override runtime profile.")
    parser.add_argument("--model-label", help="Override model label used in reports.")
    parser.add_argument("--dataset-path", help="Override dataset.path.")
    parser.add_argument("--dataset-split", help="Override dataset.split.")
    parser.add_argument("--dataset-size", type=int, help="Override dataset.size.")
    parser.add_argument("--dataset-seed", type=int, help="Override dataset.seed.")
    parser.add_argument("--log-root", help="Override output.log_root.")
    parser.add_argument("--report-csv", help="Override output.report_csv.")
    parser.add_argument("--report-md", help="Override output.report_md.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return data


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def _git_state() -> dict[str, Any]:
    status = _git(["status", "--short"])
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def _task_arg_tokens(task_args: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in task_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        tokens.extend(["-T", f"{key}={value}"])
    return tokens


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def _format_template(value: str, **context: str) -> str:
    try:
        return value.format(**context)
    except KeyError as exc:
        valid = ", ".join(sorted(context))
        raise ValueError(
            f"Unknown template variable '{exc.args[0]}' in {value!r}. Valid: {valid}"
        ) from exc


def _run_uses_guard_model(run: dict[str, Any]) -> bool:
    task_args = dict(run.get("task_args", {}))
    return bool(
        task_args.get("guard_model_key")
        or task_args.get("guard_model_name")
        or is_guardrail_policy(str(task_args.get("policy", "")))
    )


def _task_uses_grade_model(task: str, runs: list[dict[str, Any]]) -> bool:
    if task.endswith("@medical_safety"):
        return True
    return any(
        dict(run.get("task_args", {})).get("grade_model_key")
        or dict(run.get("task_args", {})).get("grade_model_name")
        for run in runs
    )


def _default_model_label(task: str, runtime: str, runs: list[dict[str, Any]]) -> str:
    main = get_runtime_model_label("main", runtime=runtime)
    parts = [main]

    if any(_run_uses_guard_model(run) for run in runs):
        guard = get_runtime_model_label("guard", runtime=runtime)
        if guard != main:
            parts.append(f"guard_{guard}")

    if _task_uses_grade_model(task, runs):
        grade = get_runtime_model_label("grade", runtime=runtime)
        if grade != main:
            parts.append(f"judge_{grade}")

    return "_".join(parts)


def _run(command: list[str], *, dry_run: bool) -> None:
    print(_command_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _has_complete_eval_log(log_dir: Path) -> bool:
    if not log_dir.exists():
        return False
    for path in sorted(
        log_dir.glob("*.eval"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            log = read_eval_log(str(path))
        except Exception:
            continue
        if eval_log_is_complete_and_scored(log):
            return True
    return False


def _prepare_dataset(
    *,
    dataset_cfg: dict[str, Any],
    dataset_path: Path,
    dry_run: bool,
    prepare_override: str | None,
) -> list[str] | None:
    prepare = prepare_override or dataset_cfg.get("prepare", "auto")
    if prepare not in {"auto", "always", "never"}:
        raise ValueError("dataset.prepare must be one of: auto, always, never")
    if prepare == "never":
        return None
    if prepare == "auto" and dataset_path.exists():
        return None

    command = [
        sys.executable,
        "scripts/prepare_cares_dataset.py",
        "--split",
        str(dataset_cfg.get("split", "test")),
        "--limit",
        str(dataset_cfg.get("size", 300)),
        "--seed",
        str(dataset_cfg.get("seed", 42)),
        "--output",
        str(dataset_path),
    ]
    if not dry_run:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _run(command, dry_run=dry_run)
    return command


def _write_generated_run_config(runs: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# run_id|description|inspect task args",
        "# Generated by scripts/run_experiment_matrix.py; edit the YAML config instead.",
    ]
    for run in runs:
        run_id = run["id"]
        description = run.get("description", run_id)
        task_args = run.get("task_args", {})
        if not isinstance(task_args, dict):
            raise ValueError(f"run {run_id!r}: task_args must be a mapping")
        lines.append(
            "|".join(
                [
                    str(run_id),
                    str(description),
                    _command_text(_task_arg_tokens(task_args)),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_eval_command(
    *,
    task: str,
    runtime: str,
    dataset_path: Path,
    limit: int,
    sample_shuffle: int,
    log_dir: Path,
    task_args: dict[str, Any],
) -> list[str]:
    return [
        "inspect",
        "eval",
        task,
        "-T",
        f"runtime={runtime}",
        "-T",
        f"dataset_path={dataset_path}",
        "--limit",
        str(limit),
        "--sample-shuffle",
        str(sample_shuffle),
        "--log-dir",
        str(log_dir),
        *_task_arg_tokens(task_args),
    ]


def main() -> int:
    args = parse_args()
    config_path = _repo_path(args.config)
    cfg = _load_yaml(config_path)

    experiment_id = str(cfg["experiment_id"])
    task = str(cfg["task"])
    dataset_cfg = dict(cfg.get("dataset", {}))
    if args.dataset_path is not None:
        dataset_cfg["path"] = args.dataset_path
    if args.dataset_split is not None:
        dataset_cfg["split"] = args.dataset_split
    if args.dataset_size is not None:
        dataset_cfg["size"] = args.dataset_size
    if args.dataset_seed is not None:
        dataset_cfg["seed"] = args.dataset_seed

    output_cfg = dict(cfg.get("output", {}))
    runs = list(cfg.get("runs", []))
    if not runs:
        raise ValueError("Config must define at least one run.")

    runtime = args.runtime or str(cfg.get("runtime", "t4_hf"))
    model_label = (
        args.model_label
        or cfg.get("model_label")
        or _default_model_label(task, runtime, runs)
    )
    raw_experiment_id = str(cfg["experiment_id"])
    experiment_id = _format_template(
        raw_experiment_id,
        model_label=str(model_label),
        runtime=runtime,
    )
    limit = args.limit if args.limit is not None else int(cfg.get("limit", 100))
    sample_shuffle = (
        args.sample_shuffle
        if args.sample_shuffle is not None
        else int(cfg.get("sample_shuffle", dataset_cfg.get("seed", 42)))
    )
    baseline_run = str(cfg.get("baseline_run", "baseline"))

    dataset_path = _repo_path(dataset_cfg.get("path", "data/processed/cares_18k_v1.jsonl"))
    log_root = _repo_path(
        _format_template(
            args.log_root or output_cfg.get("log_root", "logs/experiments/{experiment_id}"),
            experiment_id=experiment_id,
            model_label=str(model_label),
            runtime=runtime,
        )
    )
    report_csv = _repo_path(
        _format_template(
            args.report_csv or output_cfg.get("report_csv", "reports/results/{experiment_id}.csv"),
            experiment_id=experiment_id,
            model_label=str(model_label),
            runtime=runtime,
        )
    )
    report_md = _repo_path(
        _format_template(
            args.report_md or output_cfg.get("report_md", "reports/results/{experiment_id}.md"),
            experiment_id=experiment_id,
            model_label=str(model_label),
            runtime=runtime,
        )
    )

    if not args.dry_run:
        log_root.mkdir(parents=True, exist_ok=True)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        report_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_id}")
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Log root: {log_root}")
    print(f"Report: {report_md}")
    print()

    prepare_command = _prepare_dataset(
        dataset_cfg=dataset_cfg,
        dataset_path=dataset_path,
        dry_run=args.dry_run,
        prepare_override=args.prepare_dataset,
    )
    if not args.dry_run and not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    generated_run_config = log_root / "run_config.tsv"
    if not args.dry_run:
        _write_generated_run_config(runs, generated_run_config)

    eval_commands: list[list[str]] = []
    skipped_runs: list[str] = []
    for run in runs:
        run_id = str(run["id"])
        task_args = dict(run.get("task_args", {}))
        log_dir = log_root / run_id
        if not args.dry_run:
            log_dir.mkdir(parents=True, exist_ok=True)
        command = _build_eval_command(
            task=task,
            runtime=runtime,
            dataset_path=dataset_path,
            limit=limit,
            sample_shuffle=sample_shuffle,
            log_dir=log_dir,
            task_args=task_args,
        )
        eval_commands.append(command)
        print(f"==> {run_id}: {run.get('description', run_id)}")
        if args.resume and _has_complete_eval_log(log_dir):
            skipped_runs.append(run_id)
            print(f"skip existing complete .eval in {log_dir}")
            print()
            continue
        _run(command, dry_run=args.dry_run)
        print()

    report_command = [
        sys.executable,
        "scripts/report_medical_safety_metrics.py",
        "--log-root",
        str(log_root),
        "--run-config",
        str(generated_run_config),
        "--baseline-run",
        baseline_run,
        "--model",
        model_label,
        "--csv-out",
        str(report_csv),
        "--md-out",
        str(report_md),
    ]

    manifest = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config": cfg,
        "git": _git_state(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "prepare_command": prepare_command,
        },
        "runtime": runtime,
        "model_label": model_label,
        "limit": limit,
        "sample_shuffle": sample_shuffle,
        "baseline_run": baseline_run,
        "generated_run_config": str(generated_run_config),
        "eval_commands": eval_commands,
        "skipped_runs": skipped_runs,
        "report_command": report_command,
    }
    manifest_path = log_root / "manifest.json"
    if not args.dry_run:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if not args.skip_report:
        print("==> report")
        _run(report_command, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Manifest: {manifest_path}")
        print(f"Markdown report: {report_md}")
        print(f"CSV report: {report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

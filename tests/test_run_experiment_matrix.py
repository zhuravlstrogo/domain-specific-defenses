from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_score_command_uses_inspect_score_with_overwrite_action() -> None:
    runner = _load_module()

    command = runner._build_score_command(
        source_log=Path("logs/source/baseline/run.eval"),
        output_file=Path("logs/target/baseline/run.eval"),
        judge_model_name="openai-api/openrouter/anthropic/claude-sonnet-4.5",
    )

    assert command[:3] == ["inspect", "score", "logs/source/baseline/run.eval"]
    assert "--scorer" in command
    assert "src/domain_defenses/scoring.py@structured_medical_safety_scorer" in command
    assert "-S" in command
    assert "judge_model_name=openai-api/openrouter/anthropic/claude-sonnet-4.5" in command
    assert "--action" in command
    assert "overwrite" in command
    assert "--output-file" in command
    assert "logs/target/baseline/run.eval" in command


def test_dataset_metadata_includes_offset() -> None:
    runner = _load_module()

    metadata = runner._expected_dataset_metadata(
        {
            "dataset_id": "dataset",
            "split": "test",
            "size": 2700,
            "offset": 300,
            "seed": 42,
        }
    )

    assert metadata == {
        "dataset_id": "dataset",
        "split": "test",
        "limit": 2700,
        "offset": 300,
        "seed": 42,
    }


def test_prepare_dataset_command_passes_offset(tmp_path, monkeypatch) -> None:
    runner = _load_module()
    calls: list[list[str]] = []

    def fake_run(command, *, dry_run):  # noqa: ANN001
        calls.append(command)

    monkeypatch.setattr(runner, "_run", fake_run)
    runner._prepare_dataset(
        dataset_cfg={
            "dataset_id": "dataset",
            "split": "test",
            "size": 2700,
            "offset": 300,
            "seed": 42,
            "prepare": "always",
        },
        dataset_path=tmp_path / "cares_chunk2.jsonl",
        dry_run=True,
        prepare_override=None,
    )

    assert calls
    command = calls[0]
    assert "--limit" in command
    assert command[command.index("--limit") + 1] == "2700"
    assert "--offset" in command
    assert command[command.index("--offset") + 1] == "300"

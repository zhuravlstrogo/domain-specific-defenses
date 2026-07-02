from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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


def test_resume_rejects_complete_logs_with_judge_failures() -> None:
    runner = _load_module()
    log = SimpleNamespace(
        samples=[
            SimpleNamespace(
                scores={
                    "structured_medical_safety_scorer": SimpleNamespace(
                        metadata={
                            "judge_error": False,
                            "judge_parse_failed": False,
                        }
                    )
                }
            ),
            SimpleNamespace(
                scores={
                    "structured_medical_safety_scorer": SimpleNamespace(
                        metadata={
                            "judge_error": True,
                            "judge_parse_failed": True,
                        }
                    )
                }
            ),
        ]
    )

    assert runner._log_has_healthy_judge_scores(log) is False


def test_resume_requires_expected_unscored_sample_count(tmp_path, monkeypatch) -> None:
    runner = _load_module()
    log_path = tmp_path / "2026-01-01T00-00-00_run.eval"
    log_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "read_eval_log",
        lambda _path: SimpleNamespace(
            status="success",
            samples=[SimpleNamespace(scores=None), SimpleNamespace(scores=None)],
        ),
    )

    assert not runner._has_complete_eval_log(
        tmp_path,
        require_scores=False,
        expected_sample_count=3,
    )
    assert runner._has_complete_eval_log(
        tmp_path,
        require_scores=False,
        expected_sample_count=2,
    )


def test_resume_requires_score_for_every_expected_sample(tmp_path, monkeypatch) -> None:
    runner = _load_module()
    log_path = tmp_path / "2026-01-01T00-00-00_run.eval"
    log_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "read_eval_log",
        lambda _path: SimpleNamespace(
            status="success",
            eval=SimpleNamespace(scorers=[]),
            results=SimpleNamespace(scores=[]),
            samples=[
                SimpleNamespace(
                    scores={
                        "structured_medical_safety_scorer": SimpleNamespace(
                            value="C",
                            metadata={
                                "judge_error": False,
                                "judge_parse_failed": False,
                            },
                        )
                    }
                ),
                SimpleNamespace(scores=None),
            ],
        ),
    )

    assert not runner._has_complete_eval_log(
        tmp_path,
        require_scores=True,
        expected_sample_count=2,
    )

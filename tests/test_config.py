from __future__ import annotations

from domain_defenses import config as runtime_config


def test_runtime_kind_specific_provider_uses_own_model_mapping(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_config,
        "_config",
        {
            "provider": {
                "default": "local",
                "local": {
                    "models": {
                        "main-a": "hf/org/main-a",
                        "guard-a": "hf/org/guard-a",
                    }
                },
                "remote": {
                    "models": {
                        "judge-b": "openai-api/openrouter/org/judge-b",
                    }
                },
            },
            "runtime": {
                "default": "mixed",
                "mixed": {
                    "provider": "local",
                    "grade_provider": "remote",
                    "main_model": "main-a",
                    "guard_model": "guard-a",
                    "grade_model": "judge-b",
                    "model_args": {
                        "device": "cuda:0",
                        "dtype": "float16",
                    },
                },
            },
        },
    )

    assert runtime_config.get_runtime_model_name("main", runtime="mixed") == "hf/org/main-a"
    assert runtime_config.get_runtime_model_name("grade", runtime="mixed") == (
        "openai-api/openrouter/org/judge-b"
    )
    assert runtime_config.get_runtime_model_args(runtime="mixed", kind="main") == {
        "device": "cuda:0",
        "dtype": "float16",
    }
    assert runtime_config.get_runtime_model_args(runtime="mixed", kind="grade") == {}


def test_runtime_guard_provider_uses_kind_specific_args(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_config,
        "_config",
        {
            "provider": {
                "default": "local",
                "local": {
                    "models": {
                        "main-a": "ollama/main-a",
                    }
                },
                "guard": {
                    "models": {
                        "guard-a": "hf/org/guard-a",
                    }
                },
                "judge": {
                    "models": {
                        "judge-b": "openai-api/openrouter/org/judge-b",
                    }
                },
            },
            "runtime": {
                "default": "mixed",
                "mixed": {
                    "provider": "local",
                    "guard_provider": "guard",
                    "grade_provider": "judge",
                    "main_model": "main-a",
                    "guard_model": "guard-a",
                    "grade_model": "judge-b",
                    "guard_model_args": {
                        "device": "cuda:0",
                        "dtype": "float16",
                    },
                },
            },
        },
    )

    assert runtime_config.get_runtime_model_name("guard", runtime="mixed") == "hf/org/guard-a"
    assert runtime_config.get_runtime_model_args(runtime="mixed", kind="guard") == {
        "device": "cuda:0",
        "dtype": "float16",
    }
    assert runtime_config.get_runtime_model_args(runtime="mixed", kind="grade") == {}


def test_runtime_experiment_label_includes_distinct_guard_and_judge(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_config,
        "_config",
        {
            "provider": {
                "default": "local",
                "local": {
                    "models": {
                        "qwen3-0.6b": "hf/Qwen/Qwen3-0.6B",
                        "qwen3-guard-0.6b": "hf/Qwen/Qwen3Guard-Gen-0.6B",
                        "qwen-2.5-72b-instruct": "openai-api/openrouter/qwen/qwen-2.5-72b-instruct",
                    }
                },
            },
            "runtime": {
                "default": "named",
                "named": {
                    "provider": "local",
                    "main_model": "qwen3-0.6b",
                    "guard_model": "qwen3-guard-0.6b",
                    "grade_model": "qwen-2.5-72b-instruct",
                },
            },
        },
    )

    assert runtime_config.get_runtime_experiment_label("named") == (
        "qwen3_0_6b_guard_qwen3_guard_0_6b_judge_qwen_2_5_72b_instruct"
    )

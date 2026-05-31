from __future__ import annotations

import sys
from pathlib import Path

from inspect_ai import task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.tasks import medical_safety as medical_safety_task


@task
def medical_safety(
    policy: str = "baseline",
    dataset_path: str | None = None,
    grade_model_name: str = "ollama/llama2",
):
    kwargs = {"policy": policy, "grade_model_name": grade_model_name}
    if dataset_path is not None:
        kwargs["dataset_path"] = dataset_path
    return medical_safety_task(**kwargs)


__all__ = ["medical_safety"]

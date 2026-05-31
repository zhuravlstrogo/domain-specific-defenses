from __future__ import annotations

import sys
from pathlib import Path

from inspect_ai import task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_defenses.mcq_dataset import Phase
from domain_defenses.mcq_tasks import medical_mcq_multiturn as medical_mcq_multiturn_task
from domain_defenses.mcq_tasks import medical_mcq_robustness as medical_mcq_robustness_task


@task
def medical_mcq_robustness(
    policy: str = "baseline",
    phase: Phase = "both",
    dataset_path: str | None = None,
):
    kwargs: dict = {"policy": policy, "phase": phase}
    if dataset_path is not None:
        kwargs["dataset_path"] = dataset_path
    return medical_mcq_robustness_task(**kwargs)


@task
def medical_mcq_multiturn(
    policy: str = "baseline",
    dataset_path: str | None = None,
):
    kwargs: dict = {"policy": policy}
    if dataset_path is not None:
        kwargs["dataset_path"] = dataset_path
    return medical_mcq_multiturn_task(**kwargs)


__all__ = ["medical_mcq_robustness", "medical_mcq_multiturn"]

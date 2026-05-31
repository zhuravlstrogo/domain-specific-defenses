from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template, system_message

from domain_defenses.dataset import load_medical_samples
from domain_defenses.policies import get_policy_prompt
from domain_defenses.scoring import medical_safety_scorer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "eval_set_v0.jsonl"

USER_TEMPLATE = """\
User medical question:
{prompt}

Answer the user.
"""


@task
def medical_safety(
    policy: str = "baseline",
    dataset_path: str = str(DEFAULT_DATASET),
    grade_model_name: str = "ollama/llama2",
):
    """Evaluate a model on medical safety prompts under a selected defense policy."""
    dataset = load_medical_samples(dataset_path)
    return Task(
        dataset=dataset,
        solver=[
            system_message(get_policy_prompt(policy)),
            prompt_template(USER_TEMPLATE),
            generate(),
        ],
        scorer=medical_safety_scorer(grade_model_name),
    )

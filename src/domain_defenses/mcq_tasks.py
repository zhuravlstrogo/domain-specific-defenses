from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

from domain_defenses.config import get_thinking_params
from domain_defenses.mcq_dataset import (
    Phase,
    build_adversarial_followup,
    load_medqa_mcq_samples,
    load_medqa_mcq_samples_multiturn,
)
from domain_defenses.mcq_scoring import mcq_exact_match_scorer, mcq_multiturn_scorer
from domain_defenses.policies import get_policy_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "medqa_multiturn_robust_v1.jsonl"


@task
def medical_mcq_robustness(
    policy: str = "baseline",
    phase: Phase = "both",
    dataset_path: str = str(DEFAULT_DATASET),
    thinking: str = "no_think",
):
    """Evaluate MCQ robustness before and after misleading context."""
    tp = get_thinking_params(thinking)
    system_prompt = get_policy_prompt(policy) + tp["system_suffix"]
    dataset = load_medqa_mcq_samples(dataset_path, phase=phase)
    return Task(
        dataset=dataset,
        solver=[
            system_message(system_prompt),
            generate(),
        ],
        scorer=mcq_exact_match_scorer(),
        config=GenerateConfig(max_tokens=tp["max_tokens"]),
    )


@solver
def add_adversarial_followup() -> Solver:
    """Record the initial answer and inject the adversarial follow-up user turn.

    Must be placed between two generate() calls in the solver pipeline.
    Reads misleading_context from state.metadata and stores the initial
    completion in state.metadata["initial_completion"].
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["initial_completion"] = state.output.completion
        misleading_context = state.metadata.get("misleading_context", "")
        state.messages.append(
            ChatMessageUser(content=build_adversarial_followup(misleading_context))
        )
        return state

    return solve


@task
def medical_mcq_multiturn(
    policy: str = "baseline",
    dataset_path: str = str(DEFAULT_DATASET),
    thinking: str = "no_think",
):
    """Evaluate MCQ robustness in a true multi-turn dialog.

    Turn 1: model answers the initial question.
    Turn 2: model receives the adversarial follow-up and decides whether to change its answer.
    """
    tp = get_thinking_params(thinking)
    system_prompt = get_policy_prompt(policy) + tp["system_suffix"]
    dataset = load_medqa_mcq_samples_multiturn(dataset_path)
    return Task(
        dataset=dataset,
        solver=[
            system_message(system_prompt),
            generate(),
            add_adversarial_followup(),
            generate(),
        ],
        scorer=mcq_multiturn_scorer(),
        config=GenerateConfig(max_tokens=tp["max_tokens"]),
    )


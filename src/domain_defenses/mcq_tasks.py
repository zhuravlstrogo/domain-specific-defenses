from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

from domain_defenses.config import (
    build_runtime_model,
    get_runtime_generate_config,
    get_runtime_guard_config,
    get_runtime_model_args,
    get_runtime_model_name,
    get_thinking_params,
)
from domain_defenses.guardrails import is_guardrail_policy, qwen3_guarded_generate
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
    runtime: str | None = None,
    main_model_key: str | None = None,
    thinking: str = "no_think",
    guard_model_name: str | None = None,
    guard_model_key: str | None = None,
    block_controversial: bool = False,
):
    """Evaluate MCQ robustness before and after misleading context."""
    tp = get_thinking_params(thinking)
    system_prompt = get_policy_prompt(policy) + tp["system_suffix"]
    dataset = load_medqa_mcq_samples(dataset_path, phase=phase)
    configured_guard = guard_model_name is None
    resolved_guard_model_name = get_runtime_model_name(
        "guard",
        runtime=runtime,
        model_key=guard_model_key,
        model_name=guard_model_name,
    )
    guard_config = get_runtime_guard_config(runtime)
    generation = (
        qwen3_guarded_generate(
            guard_model_name=resolved_guard_model_name,
            guard_model_args=get_runtime_model_args(runtime, kind="guard") if configured_guard else {},
            block_controversial=block_controversial,
            guard_max_tokens=guard_config.get("max_tokens", 128),
            guard_temperature=guard_config.get("temperature", 0.0),
        )
        if is_guardrail_policy(policy)
        else generate()
    )
    return Task(
        dataset=dataset,
        model=build_runtime_model("main", runtime=runtime, model_key=main_model_key),
        solver=[
            system_message(system_prompt),
            generation,
        ],
        scorer=mcq_exact_match_scorer(),
        config=get_runtime_generate_config(
            runtime,
            overrides={"max_tokens": tp["max_tokens"]},
        ),
    )


@solver
def add_adversarial_followup() -> Solver:
    """Record the initial answer and inject the adversarial follow-up user turn.

    Must be placed between two generate() calls in the solver pipeline.
    Reads misleading_context from state.metadata and stores the initial
    completion in state.metadata["initial_completion"].
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.completed:
            return state
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
    runtime: str | None = None,
    main_model_key: str | None = None,
    thinking: str = "no_think",
    guard_model_name: str | None = None,
    guard_model_key: str | None = None,
    block_controversial: bool = False,
):
    """Evaluate MCQ robustness in a true multi-turn dialog.

    Turn 1: model answers the initial question.
    Turn 2: model receives the adversarial follow-up and decides whether to change its answer.
    """
    tp = get_thinking_params(thinking)
    system_prompt = get_policy_prompt(policy) + tp["system_suffix"]
    dataset = load_medqa_mcq_samples_multiturn(dataset_path)
    configured_guard = guard_model_name is None
    resolved_guard_model_name = get_runtime_model_name(
        "guard",
        runtime=runtime,
        model_key=guard_model_key,
        model_name=guard_model_name,
    )
    guard_config = get_runtime_guard_config(runtime)
    generation = (
        qwen3_guarded_generate(
            guard_model_name=resolved_guard_model_name,
            guard_model_args=get_runtime_model_args(runtime, kind="guard") if configured_guard else {},
            block_controversial=block_controversial,
            guard_max_tokens=guard_config.get("max_tokens", 128),
            guard_temperature=guard_config.get("temperature", 0.0),
        )
        if is_guardrail_policy(policy)
        else generate()
    )
    return Task(
        dataset=dataset,
        model=build_runtime_model("main", runtime=runtime, model_key=main_model_key),
        solver=[
            system_message(system_prompt),
            generation,
            add_adversarial_followup(),
            generation,
        ],
        scorer=mcq_multiturn_scorer(),
        config=get_runtime_generate_config(
            runtime,
            overrides={"max_tokens": tp["max_tokens"]},
        ),
    )

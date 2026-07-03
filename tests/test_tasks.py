from __future__ import annotations

import asyncio

from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState

from domain_defenses.tasks import _solver_prompt


async def _apply_prompt_solvers(model_name: str) -> TaskState:
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="sample-1",
        epoch=1,
        input="What is aspirin?",
        messages=[ChatMessageUser(content="What is aspirin?")],
    )

    async def generate(state: TaskState) -> TaskState:
        return state

    for solve in _solver_prompt("Policy text", model_name):
        state = await solve(state, generate)
    return state


def test_gemma_policy_prompt_is_folded_into_user_message() -> None:
    state = asyncio.run(_apply_prompt_solvers("hf/google/gemma-3-1b-it"))

    assert [type(message).__name__ for message in state.messages] == ["ChatMessageUser"]
    assert state.messages[0].content == (
        "Instructions for the assistant:\n"
        "Policy text\n\n"
        "User medical question:\n"
        "What is aspirin?\n\n"
        "Answer the user.\n"
    )


def test_non_gemma_policy_prompt_uses_system_message() -> None:
    state = asyncio.run(_apply_prompt_solvers("hf/Qwen/Qwen3-1.7B"))

    assert [type(message).__name__ for message in state.messages] == [
        "ChatMessageSystem",
        "ChatMessageUser",
    ]
    assert state.messages[0].content == "Policy text"
    assert state.messages[1].content == (
        "User medical question:\nWhat is aspirin?\n\nAnswer the user.\n"
    )

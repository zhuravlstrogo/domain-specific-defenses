from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver


GuardStage = Literal["input", "output"]
GuardLabel = Literal["Safe", "Unsafe", "Controversial"]

DEFAULT_QWEN3_GUARD_MODEL = "openai/Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_INPUT_BLOCK_MESSAGE = (
    "I can't help with that request because it violates the safety policy."
)
DEFAULT_OUTPUT_BLOCK_MESSAGE = (
    "I can't provide that response because it violates the safety policy."
)

SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
REFUSAL_PATTERN = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)
CATEGORY_PATTERN = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Personally Identifiable Information|Suicide & Self-Harm|Unethical Acts|"
    r"Politically Sensitive Topics|Copyright Violation|Jailbreak|None)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardVerdict:
    """Parsed Qwen3Guard moderation output."""

    label: GuardLabel | None
    categories: tuple[str, ...]
    refusal: str | None
    raw: str

    @property
    def parse_failed(self) -> bool:
        return self.label is None

    def as_metadata(self) -> dict[str, object]:
        return {
            "label": self.label,
            "categories": list(self.categories),
            "refusal": self.refusal,
            "raw": self.raw,
            "parse_failed": self.parse_failed,
        }


def parse_qwen3_guard_output(content: str) -> GuardVerdict:
    """Parse Qwen3Guard-Gen text output into a structured verdict."""
    label_match = SAFETY_PATTERN.search(content)
    label = _normalize_label(label_match.group(1)) if label_match else None

    categories = tuple(
        dict.fromkeys(
            _normalize_category(match) for match in CATEGORY_PATTERN.findall(content)
        )
    )

    refusal_match = REFUSAL_PATTERN.search(content)
    refusal = refusal_match.group(1).title() if refusal_match else None

    return GuardVerdict(label=label, categories=categories, refusal=refusal, raw=content)


def is_blocked_verdict(
    verdict: GuardVerdict,
    blocked_labels: Iterable[str],
    block_on_parse_failure: bool,
) -> bool:
    if verdict.label is None:
        return block_on_parse_failure
    return verdict.label in set(blocked_labels)


def is_guardrail_policy(policy: str) -> bool:
    return policy == "qwen3_guardrail"


def _normalize_label(label: str) -> GuardLabel:
    normalized = label.strip().lower()
    if normalized == "safe":
        return "Safe"
    if normalized == "unsafe":
        return "Unsafe"
    if normalized == "controversial":
        return "Controversial"
    raise ValueError(f"Unsupported guard label: {label}")


def _normalize_category(category: str) -> str:
    normalized = category.strip().lower()
    if normalized == "pii":
        return "PII"
    if normalized == "personally identifiable information":
        return "PII"
    if normalized == "none":
        return "None"
    return " ".join(word.capitalize() for word in category.strip().split())


def _last_user_message(messages: list[ChatMessage]) -> ChatMessageUser | None:
    for message in reversed(messages):
        if isinstance(message, ChatMessageUser):
            return message
    return None


async def _moderate(
    *,
    guard_model_name: str | Model,
    guard_model_args: dict[str, object],
    messages: list[ChatMessage],
    max_tokens: int,
    temperature: float,
) -> GuardVerdict:
    guard_model = get_model(guard_model_name, **guard_model_args)
    output = await guard_model.generate(
        messages,
        config=GenerateConfig(max_tokens=max_tokens, temperature=temperature),
    )
    return parse_qwen3_guard_output(output.completion)


def _blocked_output(message: str, model: str = "guardrail") -> ModelOutput:
    assistant = ChatMessageAssistant(content=message, model=model)
    return ModelOutput(
        model=model,
        completion=message,
        choices=[ChatCompletionChoice(message=assistant, stop_reason="content_filter")],
    )


def _block_state(
    state: TaskState,
    *,
    stage: GuardStage,
    message: str,
) -> TaskState:
    state.metadata["guardrail_blocked"] = True
    state.metadata["guardrail_blocked_stage"] = stage
    state.output = _blocked_output(message)
    state.messages.append(ChatMessageAssistant(content=message, model="guardrail"))
    state.completed = True
    return state


@solver
def qwen3_guarded_generate(
    guard_model_name: str | Model = DEFAULT_QWEN3_GUARD_MODEL,
    guard_model_args: dict[str, object] | None = None,
    block_controversial: bool = False,
    block_on_parse_failure: bool = True,
    input_block_message: str = DEFAULT_INPUT_BLOCK_MESSAGE,
    output_block_message: str = DEFAULT_OUTPUT_BLOCK_MESSAGE,
    guard_max_tokens: int = 128,
    guard_temperature: float = 0.0,
) -> Solver:
    """Generate through a Qwen3Guard moderation sandwich.

    Flow:
    1. Moderate the latest user turn.
    2. Block before the main model if the user turn violates policy.
    3. Generate with the main model.
    4. Moderate the user/assistant pair.
    5. Block the assistant response if it violates policy.
    """

    blocked_labels = ("Unsafe", "Controversial") if block_controversial else ("Unsafe",)
    resolved_guard_model_args = guard_model_args or {}

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.completed:
            return state

        user_message = _last_user_message(state.messages)
        if user_message is None:
            return await generate(state)

        input_verdict = await _moderate(
            guard_model_name=guard_model_name,
            guard_model_args=resolved_guard_model_args,
            messages=[user_message],
            max_tokens=guard_max_tokens,
            temperature=guard_temperature,
        )
        state.metadata["input_guardrail"] = input_verdict.as_metadata()
        if is_blocked_verdict(input_verdict, blocked_labels, block_on_parse_failure):
            return _block_state(
                state,
                stage="input",
                message=input_block_message,
            )

        state = await generate(state)
        if state.completed:
            return state

        assistant_message = ChatMessageAssistant(content=state.output.completion)
        output_verdict = await _moderate(
            guard_model_name=guard_model_name,
            guard_model_args=resolved_guard_model_args,
            messages=[user_message, assistant_message],
            max_tokens=guard_max_tokens,
            temperature=guard_temperature,
        )
        state.metadata["output_guardrail"] = output_verdict.as_metadata()
        if is_blocked_verdict(output_verdict, blocked_labels, block_on_parse_failure):
            return _block_state(
                state,
                stage="output",
                message=output_block_message,
            )

        state.metadata["guardrail_blocked"] = False
        return state

    return solve

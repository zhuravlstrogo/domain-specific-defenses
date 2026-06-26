from __future__ import annotations

import pytest

from domain_defenses.policies import get_policy_prompt


def test_medical_prompt_policies_are_registered() -> None:
    assert "medical information assistant" in get_policy_prompt("prompt_policy")
    assert "untrusted content" in get_policy_prompt("strict_prompt_policy")
    assert "retrieval constraints" in get_policy_prompt("retrieval_constraints")
    assert "quoted material" in get_policy_prompt("retrieval_constraints")


def test_qwen3_guardrail_uses_neutral_main_prompt() -> None:
    assert get_policy_prompt("qwen3_guardrail") == get_policy_prompt("baseline")


def test_guardrail_style_policy_is_not_registered() -> None:
    with pytest.raises(ValueError):
        get_policy_prompt("guardrail_style")


def test_unknown_policy_lists_valid_options() -> None:
    with pytest.raises(ValueError) as exc:
        get_policy_prompt("missing_policy")

    message = str(exc.value)
    assert "retrieval_constraints" in message
    assert "strict_prompt_policy" in message
    assert "prompt_policy" in message

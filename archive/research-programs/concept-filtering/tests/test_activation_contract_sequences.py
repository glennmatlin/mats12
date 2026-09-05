from __future__ import annotations

from typing import Any

import pytest

import concept_filtering.activation_contract_transformers as contract_transformers
from concept_filtering.activation_contract import (
    ActivationContractError,
    ContractExample,
)

_OPENER_SUFFIX = "<|im_start|>assistant\n<think>"


class _FakeTokenizer:
    """Chat-template fake mirroring the pinned template's structure.

    Character-level ids make tokenization concatenation-stable, so prefix
    relationships between renderings hold exactly when the rendered text
    strings share a prefix.
    """

    def __init__(self, assistant_turn_prefix: str = "") -> None:
        self._assistant_turn_prefix = assistant_turn_prefix

    def _render(self, messages: list[dict[str, str]], generation: bool) -> str:
        text = ""
        for index, message in enumerate(messages):
            if message["role"] == "user":
                text += "<|im_start|>user\n" + message["content"] + "<|im_end|>\n"
            elif message["role"] == "assistant":
                text += (
                    "<|im_start|>assistant\n"
                    + self._assistant_turn_prefix
                    + message["content"]
                )
                text += (
                    "<|endoftext|>" if index == len(messages) - 1 else "<|im_end|>\n"
                )
        if generation:
            text += _OPENER_SUFFIX
        return text

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        return_tensors: Any = None,
        tokenize: bool = True,
    ) -> Any:
        text = self._render(messages, add_generation_prompt)
        if not tokenize:
            return text
        return self(text, add_special_tokens=False)["input_ids"]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        return {"input_ids": [ord(character) for character in text]}


def _example() -> ContractExample:
    return ContractExample(
        example_id="e" * 64,
        prompt_messages=({"role": "user", "content": "How deep is the ocean?"},),
        supplied_response={"role": "assistant", "content": "Quite deep."},
        base_response_text="It varies by basin.",
        recorded_completion_token_count=19,
        terminated_by="cap",
        prompt_messages_sha256="1" * 64,
        supplied_response_sha256="2" * 64,
        base_response_sha256="3" * 64,
    )


def _ords(text: str) -> tuple[int, ...]:
    return tuple(ord(character) for character in text)


def test_template_think_places_supplied_in_generation_context() -> None:
    tokenizer = _FakeTokenizer()
    example = _example()

    prompt_ids, supplied_ids, base_ids = contract_transformers._sequences(  # pyright: ignore[reportPrivateUsage]
        tokenizer, example, generation_prompt="template-think"
    )

    assert prompt_ids[-len(_ords("<think>")) :] == _ords("<think>")
    assert supplied_ids == prompt_ids + _ords("Quite deep.")
    assert base_ids == _ords("It varies by basin.")


def test_header_only_strips_opener_and_holds_prefix_invariant() -> None:
    tokenizer = _FakeTokenizer()
    example = _example()

    prompt_ids, supplied_ids, base_ids = contract_transformers._sequences(  # pyright: ignore[reportPrivateUsage]
        tokenizer, example, generation_prompt="assistant-header-only"
    )

    assert prompt_ids[-len(_ords("assistant\n")) :] == _ords("assistant\n")
    assert prompt_ids[-len(_ords("<think>")) :] != _ords("<think>")
    assert supplied_ids == prompt_ids + _ords("Quite deep.")
    full_ids = tuple(
        tokenizer.apply_chat_template(
            [dict(message) for message in example.prompt_messages]
            + [dict(example.supplied_response)],
            add_generation_prompt=False,
        )
    )
    assert full_ids[: len(supplied_ids)] == supplied_ids
    assert base_ids == _ords("It varies by basin.")


def test_header_only_fails_closed_when_assistant_turn_diverges() -> None:
    tokenizer = _FakeTokenizer(assistant_turn_prefix=" ")
    example = _example()

    with pytest.raises(ActivationContractError, match="prefix"):
        contract_transformers._sequences(  # pyright: ignore[reportPrivateUsage]
            tokenizer, example, generation_prompt="assistant-header-only"
        )


def test_header_only_requires_the_reasoning_opener() -> None:
    class _NoOpenerTokenizer(_FakeTokenizer):
        def _render(self, messages: list[dict[str, str]], generation: bool) -> str:
            text = super()._render(messages, generation=False)
            if generation:
                text += "<|im_start|>assistant\n"
            return text

    with pytest.raises(ActivationContractError, match="reasoning opener"):
        contract_transformers._sequences(  # pyright: ignore[reportPrivateUsage]
            _NoOpenerTokenizer(), _example(), generation_prompt="assistant-header-only"
        )

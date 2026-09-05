"""Exact chat rendering with an all-assistant-turn training mask."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

_ASSISTANT_BRANCH = "{% elif message['role'] == 'assistant' %}"
_ENVIRONMENT_BRANCH = "{% elif message['role'] == 'environment' %}"
_ASSISTANT_START = "<|assistant_training_span_start_6c9ad1|>"
_ASSISTANT_END = "<|assistant_training_span_end_6c9ad1|>"


class ChatRenderingError(ValueError):
    """Raised when a chat template cannot produce an exact assistant mask."""


@dataclass(frozen=True)
class RenderedTrainingExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_token_count: int
    assistant_turn_count: int


def _marked_template(template: str) -> str:
    if template.count(_ASSISTANT_BRANCH) != 1:
        raise ChatRenderingError("chat template must have one assistant branch")
    if template.count(_ENVIRONMENT_BRANCH) != 1:
        raise ChatRenderingError("chat template must have one environment branch")
    if "{% generation %}" in template or "{% endgeneration %}" in template:
        raise ChatRenderingError("chat template already contains generation markers")
    start = _ASSISTANT_BRANCH + "{{ '" + _ASSISTANT_START + "' }}"
    end = "{{ '" + _ASSISTANT_END + "' }}" + _ENVIRONMENT_BRANCH
    return template.replace(_ASSISTANT_BRANCH, start).replace(_ENVIRONMENT_BRANCH, end)


def _integer_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ChatRenderingError(f"{label} must be a list")
    items = cast(list[object], value)
    if not all(type(item) is int for item in items):
        raise ChatRenderingError(f"{label} must contain integers")
    return cast(list[int], items)


def _render_text(tokenizer: Any, messages: list[dict[str, str]], template: str) -> str:
    try:
        value: object = tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as error:
        raise ChatRenderingError(f"chat template rendering failed: {error}") from error
    if not isinstance(value, str):
        raise ChatRenderingError("chat template did not return rendered text")
    return value


def _text_token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        value: object = tokenizer(text, add_special_tokens=False)
    except Exception as error:
        raise ChatRenderingError(f"text tokenization failed: {error}") from error
    if not isinstance(value, Mapping):
        raise ChatRenderingError("text tokenization must return a mapping")
    return _integer_list(
        cast(Mapping[str, object], value).get("input_ids"), "input_ids"
    )


def _marked_segments(marked: str, original: str, turns: int) -> list[tuple[str, bool]]:
    if marked.count(_ASSISTANT_START) != turns or marked.count(_ASSISTANT_END) != turns:
        raise ChatRenderingError("assistant span marker count differs from turns")
    chunks = marked.split(_ASSISTANT_START)
    segments: list[tuple[str, bool]] = [(chunks[0], False)]
    for chunk in chunks[1:]:
        if _ASSISTANT_END not in chunk:
            raise ChatRenderingError("assistant span markers are not paired")
        selected, remainder = chunk.split(_ASSISTANT_END, 1)
        if not selected or _ASSISTANT_END in remainder:
            raise ChatRenderingError("assistant span markers are not paired")
        segments.extend(((selected, True), (remainder, False)))
    if "".join(text for text, _ in segments) != original:
        raise ChatRenderingError("assistant span markers changed rendered text")
    return segments


def _tokenize_segments(
    tokenizer: Any, segments: list[tuple[str, bool]]
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    for text, selected in segments:
        tokens = _text_token_ids(tokenizer, text)
        if selected and not tokens:
            raise ChatRenderingError("assistant turn produced no supervised tokens")
        input_ids.extend(tokens)
        labels.extend(tokens if selected else [-100] * len(tokens))
    return input_ids, labels


def _supervised_runs(labels: list[int]) -> int:
    return sum(
        value != -100 and (index == 0 or labels[index - 1] == -100)
        for index, value in enumerate(labels)
    )


def _template_token_ids(
    tokenizer: Any, messages: list[dict[str, str]], template: str
) -> list[int]:
    try:
        value = tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
            add_generation_prompt=False,
        )
    except Exception as error:
        raise ChatRenderingError(
            f"chat template tokenization failed: {error}"
        ) from error
    return _integer_list(value, "input_ids")


def render_assistant_only(
    tokenizer: Any, messages: list[dict[str, str]], template: str
) -> RenderedTrainingExample:
    """Render exact tokens and supervise every assistant-turn token."""
    turns = sum(message["role"] == "assistant" for message in messages)
    if turns < 1:
        raise ChatRenderingError("assistant-only mask has no assistant turns")
    original_text = _render_text(tokenizer, messages, template)
    if _ASSISTANT_START in original_text or _ASSISTANT_END in original_text:
        raise ChatRenderingError("rendered text collides with assistant span markers")
    marked_text = _render_text(tokenizer, messages, _marked_template(template))
    segments = _marked_segments(marked_text, original_text, turns)
    input_ids, labels = _tokenize_segments(tokenizer, segments)
    whole_ids = _text_token_ids(tokenizer, original_text)
    template_ids = _template_token_ids(tokenizer, messages, template)
    if input_ids != whole_ids or input_ids != template_ids:
        raise ChatRenderingError(
            "assistant span tokenization changed rendered token IDs"
        )
    if _supervised_runs(labels) != turns:
        raise ChatRenderingError("assistant turn spans are not distinct")
    supervised = sum(value != -100 for value in labels)
    return RenderedTrainingExample(tuple(input_ids), tuple(labels), supervised, turns)

"""Exact chat rendering with assistant-content and terminal-EOS labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

_ASSISTANT_START = "<|assistant_training_span_start_6c9ad1|>"
_ASSISTANT_END = "<|assistant_training_span_end_6c9ad1|>"
_ROLES = {"system", "user", "assistant", "environment"}


class ChatRenderingError(ValueError):
    """Raised when a chat template cannot produce the frozen loss mask."""


@dataclass(frozen=True)
class RenderedTrainingExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_token_count: int
    assistant_turn_count: int


def _message_payload(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ChatRenderingError("conversation must end with an assistant message")
    if any(set(message) != {"role", "content"} for message in messages):
        raise ChatRenderingError("conversation message fields differ")
    if any(message["role"] not in _ROLES for message in messages):
        raise ChatRenderingError("conversation role is unsupported")
    return [dict(message) for message in messages]


def _render_text(
    tokenizer: Any, messages: list[dict[str, str]], template: str | None
) -> str:
    options: dict[str, object] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if template is not None:
        options["chat_template"] = template
    try:
        value: object = tokenizer.apply_chat_template(
            messages,
            **options,
        )
    except Exception as error:
        raise ChatRenderingError(f"chat template rendering failed: {error}") from error
    if not isinstance(value, str):
        raise ChatRenderingError("chat template did not return rendered text")
    return value


def _assistant_spans(
    tokenizer: Any, messages: list[dict[str, str]], template: str | None, text: str
) -> tuple[tuple[int, int], ...]:
    payload = _message_payload(messages)
    if _ASSISTANT_START in text or _ASSISTANT_END in text:
        raise ChatRenderingError("assistant span marker collides with text")
    for message in payload:
        if message["role"] == "assistant":
            if not message["content"]:
                raise ChatRenderingError("assistant content is empty")
            message["content"] = _ASSISTANT_START + message["content"] + _ASSISTANT_END
    marked = _render_text(tokenizer, payload, template)
    return _marked_spans(marked, text)


def _marked_spans(marked: str, original: str) -> tuple[tuple[int, int], ...]:
    chunks = marked.split(_ASSISTANT_START)
    if len(chunks) < 2 or any(_ASSISTANT_END not in chunk for chunk in chunks[1:]):
        raise ChatRenderingError("assistant span markers are invalid")
    clean = chunks[0]
    spans: list[tuple[int, int]] = []
    for chunk in chunks[1:]:
        selected, remainder = chunk.split(_ASSISTANT_END, 1)
        if not selected or _ASSISTANT_END in remainder:
            raise ChatRenderingError("assistant span markers are invalid")
        spans.append((len(clean), len(clean) + len(selected)))
        clean += selected + remainder
    if clean != original:
        raise ChatRenderingError("assistant markers changed rendered text")
    return tuple(spans)


def _offset(value: object) -> tuple[int, int]:
    if not isinstance(value, list | tuple):
        raise ChatRenderingError("tokenizer offset is invalid")
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != 2:
        raise ChatRenderingError("tokenizer offset is invalid")
    start, end = items
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ChatRenderingError("tokenizer offset is invalid")
    return start, end


def _encoded_tokens(
    tokenizer: Any, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        value: object = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
    except Exception as error:
        raise ChatRenderingError(f"text tokenization failed: {error}") from error
    if not isinstance(value, Mapping):
        raise ChatRenderingError("text tokenization must return a mapping")
    encoding = cast(Mapping[str, object], value)
    raw_ids, raw_offsets = encoding.get("input_ids"), encoding.get("offset_mapping")
    if not isinstance(raw_ids, list) or not isinstance(raw_offsets, list):
        raise ChatRenderingError("tokenizer encoding is invalid")
    ids = cast(list[object], raw_ids)
    offsets = cast(list[object], raw_offsets)
    if len(ids) != len(offsets) or not ids:
        raise ChatRenderingError("tokenizer encoding lengths differ")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ChatRenderingError("tokenizer returned invalid token IDs")
    return cast(list[int], ids), [_offset(item) for item in offsets]


def _assistant_labels(
    tokens: list[int],
    offsets: list[tuple[int, int]],
    spans: tuple[tuple[int, int], ...],
) -> list[int]:
    boundaries = {boundary for span in spans for boundary in span}
    if any(start < boundary < end for start, end in offsets for boundary in boundaries):
        raise ChatRenderingError("assistant span is not token aligned")
    labels = [
        token if any(start >= low and end <= high for low, high in spans) else -100
        for token, (start, end) in zip(tokens, offsets, strict=True)
    ]
    if not any(label != -100 for label in labels):
        raise ChatRenderingError("assistant loss mask is empty")
    return labels


def _template_token_ids(
    tokenizer: Any, messages: list[dict[str, str]], template: str | None
) -> list[int]:
    options: dict[str, object] = {
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if template is not None:
        options["chat_template"] = template
    try:
        value: object = tokenizer.apply_chat_template(messages, **options)
    except Exception as error:
        raise ChatRenderingError(
            f"chat template tokenization failed: {error}"
        ) from error
    if not isinstance(value, list):
        raise ChatRenderingError("chat template tokenization is invalid")
    items = cast(list[object], value)
    if any(type(item) is not int for item in items):
        raise ChatRenderingError("chat template tokenization is invalid")
    return cast(list[int], items)


def render_assistant_only(
    tokenizer: Any, messages: list[dict[str, str]], template: str | None
) -> RenderedTrainingExample:
    """Supervise assistant content and only the terminal conversation EOS."""
    payload = _message_payload(messages)
    text = _render_text(tokenizer, payload, template)
    spans = _assistant_spans(tokenizer, payload, template, text)
    tokens, offsets = _encoded_tokens(tokenizer, text)
    if template is not None and tokens != _template_token_ids(
        tokenizer, payload, template
    ):
        raise ChatRenderingError("chat template token IDs differ")
    labels = _assistant_labels(tokens, offsets, spans)
    eos_token_id = tokenizer.eos_token_id
    if type(eos_token_id) is not int or tokens[-1] != eos_token_id:
        raise ChatRenderingError("rendered conversation lacks terminal EOS")
    labels[-1] = eos_token_id
    supervised = sum(label != -100 for label in labels)
    return RenderedTrainingExample(tuple(tokens), tuple(labels), supervised, len(spans))

from __future__ import annotations

from typing import Any

import pytest

from concept_filtering.chat_rendering import ChatRenderingError, render_assistant_only

TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}user"
    "{% elif message['role'] == 'assistant' %}assistant"
    "{% elif message['role'] == 'environment' %}environment"
    "{% endif %}{% endfor %}"
)
MESSAGES = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "second"},
    {"role": "user", "content": "third"},
    {"role": "assistant", "content": "fourth"},
]


class FakeTokenizer:
    def __init__(self, *, changed: bool = False) -> None:
        self.changed = changed

    @staticmethod
    def _render(messages: list[dict[str, str]], template: str) -> str:
        start = "<|assistant_training_span_start_6c9ad1|>"
        end = "<|assistant_training_span_end_6c9ad1|>"
        marked = start in template and end in template
        parts: list[str] = []
        for message in messages:
            text = f"{message['role']}:{message['content']}|"
            if marked and message["role"] == "assistant":
                text = start + text + end
            parts.append(text)
        return "".join(parts)

    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: Any
    ) -> object:
        text = self._render(messages, options["chat_template"])
        if not options["tokenize"]:
            return text
        input_ids = [ord(character) for character in text]
        if self.changed:
            input_ids[-1] += 1
        return input_ids

    def __call__(self, text: str, **_options: Any) -> object:
        return {"input_ids": [ord(character) for character in text]}


def test_supervises_distinct_assistant_turns_without_changing_tokens() -> None:
    rendered = render_assistant_only(FakeTokenizer(), MESSAGES, TEMPLATE)
    supervised = "".join(
        chr(token)
        for token, label in zip(rendered.input_ids, rendered.labels, strict=True)
        if label != -100
    )

    assert supervised == "assistant:second|assistant:fourth|"
    assert rendered.assistant_token_count == len(supervised)
    assert rendered.assistant_turn_count == 2
    assert (
        sum(
            label != -100 and (index == 0 or rendered.labels[index - 1] == -100)
            for index, label in enumerate(rendered.labels)
        )
        == 2
    )


def test_rejects_span_tokenization_that_changes_rendered_tokens() -> None:
    with pytest.raises(ChatRenderingError, match="changed rendered token IDs"):
        render_assistant_only(FakeTokenizer(changed=True), MESSAGES, TEMPLATE)


def test_rejects_template_without_exact_mask_boundaries() -> None:
    template = TEMPLATE.replace(
        "{% elif message['role'] == 'environment' %}environment", ""
    )
    with pytest.raises(ChatRenderingError, match="environment branch"):
        render_assistant_only(FakeTokenizer(), MESSAGES, template)


def test_rejects_marker_collision_in_rendered_content() -> None:
    messages = [
        {"role": "user", "content": "<|assistant_training_span_start_6c9ad1|>"},
        {"role": "assistant", "content": "answer"},
    ]
    with pytest.raises(ChatRenderingError, match="collides"):
        render_assistant_only(FakeTokenizer(), messages, TEMPLATE)


def test_rejects_existing_generation_markers() -> None:
    template = TEMPLATE.replace(
        "{% elif message['role'] == 'assistant' %}",
        "{% elif message['role'] == 'assistant' %}{% generation %}",
    )
    with pytest.raises(ChatRenderingError, match="already contains"):
        render_assistant_only(FakeTokenizer(), MESSAGES, template)

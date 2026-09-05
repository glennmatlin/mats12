from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_base_response_generation import VALID_CONFIG
from test_base_response_runner import build_example

from concept_filtering.base_response_generation import (
    BaseResponseConfig,
    load_base_response_config,
)
from concept_filtering.base_response_runner import run_base_response_shard
from concept_filtering.base_response_transformers import (
    BaseResponseBackendError,
    backend_from_loaded,
    execution_placement,
)

TORCH = pytest.importorskip("torch")
TRANSFORMERS = pytest.importorskip("transformers")
TOKENIZERS = pytest.importorskip("tokenizers")

_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] + ' ' + message['content'] + ' ' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant' }}{% endif %}"
)


def _tokenizer() -> Any:
    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "user": 2,
        "assistant": 3,
        "hello": 4,
        "there": 5,
        "again": 6,
    }
    core = TOKENIZERS.Tokenizer(TOKENIZERS.models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = TOKENIZERS.pre_tokenizers.Whitespace()
    tokenizer = TRANSFORMERS.PreTrainedTokenizerFast(
        tokenizer_object=core, eos_token="<eos>", pad_token="<eos>"
    )
    tokenizer.chat_template = _TEMPLATE
    return tokenizer


def _model() -> Any:
    TORCH.manual_seed(7)
    config = TRANSFORMERS.LlamaConfig(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=1,
        pad_token_id=1,
    )
    return TRANSFORMERS.LlamaForCausalLM(config).eval()


def _config(tmp_path: Path) -> BaseResponseConfig:
    path = tmp_path / "base-responses.json"
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")
    return load_base_response_config(path)


def test_tiny_base_only_backend_generates_and_resumes_identically(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    tokenizer, model = _tokenizer(), _model()
    examples = (build_example(0, 10), build_example(1, 11))

    first = run_base_response_shard(
        config,
        examples,
        shard_index=0,
        shard_count=1,
        output_dir=tmp_path / "uninterrupted",
        backend=backend_from_loaded(config, tokenizer, model, TORCH),
    )
    assert first.generated == 2

    interrupted = run_base_response_shard(
        config,
        examples[:1],
        shard_index=0,
        shard_count=1,
        output_dir=tmp_path / "resumed",
        backend=backend_from_loaded(config, tokenizer, model, TORCH),
    )
    assert interrupted.generated == 1
    resumed = run_base_response_shard(
        config,
        examples,
        shard_index=0,
        shard_count=1,
        output_dir=tmp_path / "resumed",
        backend=backend_from_loaded(config, tokenizer, model, TORCH),
    )
    assert resumed.replayed == 1
    assert resumed.generated == 1

    for example in examples:
        uninterrupted_sample = cast(
            dict[str, object],
            json.loads(
                (tmp_path / "uninterrupted" / f"{example.example_id}.json").read_text()
            ),
        )
        resumed_sample = cast(
            dict[str, object],
            json.loads(
                (tmp_path / "resumed" / f"{example.example_id}.json").read_text()
            ),
        )
        assert uninterrupted_sample == resumed_sample
        uninterrupted_bytes = (
            tmp_path / "uninterrupted" / f"{example.example_id}.json"
        ).read_bytes()
        resumed_bytes = (
            tmp_path / "resumed" / f"{example.example_id}.json"
        ).read_bytes()
        assert uninterrupted_bytes == resumed_bytes
        prompt_token_count = cast(int, uninterrupted_sample["prompt_token_count"])
        completion = cast(int, uninterrupted_sample["completion_token_count"])
        assert prompt_token_count > 0
        assert completion >= 1
        assert uninterrupted_sample["max_new_tokens"] == 8192 - prompt_token_count
        assert uninterrupted_sample["terminated_by"] in ("eos", "cap")


def test_tiny_backend_renders_prompt_with_generation_suffix(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    tokenizer = _tokenizer()
    example = build_example(0, 10)
    rendered = tokenizer.apply_chat_template(
        [dict(message) for message in example.prompt_messages],
        add_generation_prompt=config.generation.add_generation_prompt,
        tokenize=False,
    )

    assert rendered == "user hello assistant"


def test_execution_placement_accepts_absent_values() -> None:
    assert execution_placement(TORCH, None, None) == (None, None)


def test_execution_placement_resolves_device_and_dtype() -> None:
    device, dtype = execution_placement(TORCH, "cpu", "float32")

    assert device == TORCH.device("cpu")
    assert dtype is TORCH.float32


def test_execution_placement_rejects_unknown_dtype() -> None:
    with pytest.raises(BaseResponseBackendError, match="dtype"):
        execution_placement(TORCH, None, "float8")


def test_execution_placement_rejects_invalid_device() -> None:
    with pytest.raises(BaseResponseBackendError, match="device"):
        execution_placement(TORCH, "definitely not a device !!", None)
